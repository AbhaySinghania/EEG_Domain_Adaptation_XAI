from captum.attr import DeepLift
from captum.attr import GradientShap
from captum.attr import IntegratedGradients
import time

import warnings

import numpy as np
import torch
import torch.nn as nn
import copy
from pathlib import Path
import os
import json
from collections import Counter

from models import DANN
from data_loading import load_subject_data_flexible, prepare_dataloaders
from metrics import evaluate

# paths, same pattern as other function files
# Repository root. Set EEG_PROJECT_ROOT to use data and results stored elsewhere.
base_path = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
results_path = base_path / "results"
binary_misc_path = results_path / "misclassified_attributions" / "binary_cdan_16ch"
multiclass_misc_path = results_path / "misclassified_attributions" / "multiclass_16ch"


class DANNClassifierWrapper(nn.Module):
    """Exposes only feature_extractor + classifier from DANN.
    Bypasses domain discriminator entirely, giving captum a clean
    single-output graph to trace attribution through."""

    def __init__(self, dann_model):
        super().__init__()
        self.feature_extractor = dann_model.feature_extractor
        self.classifier = dann_model.classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def load_model_for_xai(
    subject, checkpoint_tag, n_classes=2, stopping_criterion="kappa"
):
    ckpt_path = (
        results_path
        / "checkpoints"
        / checkpoint_tag
        / f"{subject}_{stopping_criterion}.pt"
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DANN(
        n_channels=16,
        n_timepoints=480,
        n_classes=n_classes,
        embed_dim=128,
        dropout_rate=0.25,
        use_conditional_discriminator=True,
        feature_extractor_type="eegnet",
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def fold_batchnorm(wrapper):
    """
    Returns a deep copy of wrapper with all Conv2d + BatchNorm2d pairs folded.
    Original wrapper is untouched.
    BatchNorm parameters absorbed into preceding Conv2d weights and bias.
    Predictions mathematically identical before and after folding.
    """
    folded = copy.deepcopy(wrapper)
    folded.eval()

    # exact paths: (block, conv_idx, bn_idx)
    pairs = [
        (folded.feature_extractor.temporal_conv, 0, 1),
        (folded.feature_extractor.depthwise_conv, 0, 1),
        (folded.feature_extractor.separable_conv, 1, 2),
    ]

    for block, conv_idx, bn_idx in pairs:
        conv = block[conv_idx]
        bn = block[bn_idx]

        # scale: gamma / sqrt(running_var + eps)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        # bias:  beta - gamma * running_mean / sqrt(running_var + eps)
        bias = bn.bias - bn.weight * bn.running_mean / torch.sqrt(
            bn.running_var + bn.eps
        )

        # absorb scale into conv weights
        # conv.weight shape: (out_channels, in_channels/groups, kH, kW)
        conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

        # add bias term (conv.bias was None since bias=False at init)
        conv.bias = nn.Parameter(bias.clone())

        # replace BatchNorm with Identity
        block[bn_idx] = nn.Identity()

    return folded


def verify_folding(wrapper, folded, test_input):
    """
    Two checks:
    1. Max logit difference between original and folded < 1e-5
    2. No BatchNorm2d layers remaining in folded model
    """
    wrapper.eval()
    folded.eval()

    with torch.no_grad():
        orig_out = wrapper(test_input)
        fold_out = folded(test_input)

    max_diff = (orig_out - fold_out).abs().max().item()
    print(f"Max logit difference: {max_diff:.10f}")
    print(f"Folding correct: {max_diff < 1e-5}")

    bn_remaining = [
        name for name, m in folded.named_modules() if isinstance(m, nn.BatchNorm2d)
    ]
    print(f"BatchNorm2d layers remaining: {len(bn_remaining)}")
    if bn_remaining:
        print(f"  {bn_remaining}")


def lrp_epsilon_conv2d(layer, activation, relevance, epsilon=1e-6):
    """
    Epsilon rule for Conv2d.
    Propagates relevance backward through a conv layer using saved activation.
    activation: input to this layer, shape as it entered the layer
    relevance: relevance arriving at this layer's output
    """
    activation = activation.detach().requires_grad_(True)
    z = layer(activation)
    # stabilise denominator with epsilon
    s = relevance / (z + epsilon * z.sign().clamp(min=1e-6) + 1e-6)
    z.backward(s)
    return activation.grad * activation


def lrp_zplus_conv2d(layer, activation, relevance, epsilon=1e-6):
    """
    Z-plus rule for Conv2d.
    Only positive contributions counted, cleaner input-level attributions.
    Used optionally for first Conv2d layer only (temporal_conv.0).
    """
    # temporarily zero negative weights
    weight_orig = layer.weight.data.clone()
    layer.weight.data = weight_orig.clamp(min=0)

    activation = activation.detach().requires_grad_(True)
    z = layer(activation)
    s = relevance / (z + epsilon)
    z.backward(s)
    grad = activation.grad * activation

    # restore original weights
    layer.weight.data = weight_orig
    return grad


def lrp_linear(layer, activation, relevance, epsilon=1e-6):
    """
    Epsilon rule for Linear layers.
    """
    activation = activation.detach().requires_grad_(True)
    z = layer(activation)
    s = relevance / (z + epsilon * z.sign().clamp(min=1e-6) + 1e-6)
    z.backward(s)
    return activation.grad * activation


def lrp_avgpool(layer, activation, relevance):
    """
    Pass-through for AvgPool2d.
    Distributes relevance uniformly across the pooling window,
    inverse of what average pooling did in the forward pass.
    """
    activation = activation.detach().requires_grad_(True)
    z = layer(activation)
    s = relevance / (z + 1e-6)
    z.backward(s)
    return activation.grad * activation


def compute_lrp(
    subject,
    checkpoint_tag,
    n_classes=2,
    use_multiclass=False,
    stopping_criterion="kappa",
    first_layer_rule="epsilon",
    epsilon=1e-6,
    trial_type="correct",
):
    """
    Manual LRP implementation for DANNClassifierWrapper + EEGNet.
    trial_type:
        'correct'      — only correctly classified trials, attribution
                         targets true label (original behaviour)
        'misclassified'— only misclassified trials, attribution targets
                         predicted label (what model actually decided)
        'all'          — both sets returned separately as a tuple
    Returns:
        if trial_type != 'all':
            attributions: (n_trials, n_channels, n_timepoints)
            labels: (n_trials,) true labels
            mean_attr_per_class: dict {class_idx: mean attr (n_channels, n_timepoints)}
            [if misclassified: also stores predicted labels separately]
        if trial_type == 'all':
            (correct_output, misclassified_output) where each is the above tuple
    """
    model = load_model_for_xai(
        subject,
        checkpoint_tag,
        n_classes=n_classes,
        stopping_criterion=stopping_criterion,
    )
    wrapper = DANNClassifierWrapper(model)
    folded = fold_batchnorm(wrapper)
    folded.eval()

    fe = folded.feature_extractor
    clf = folded.classifier

    X, y = load_subject_data_flexible(
        subject, use_multiclass=use_multiclass, use_16ch=True
    )
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    correct_attrs, correct_labels = [], []
    misc_attrs, misc_true_labels, misc_pred_labels = [], [], []

    for i in range(len(X_tensor)):
        inp = X_tensor[i : i + 1]

        with torch.no_grad():
            a0 = inp
            a1 = fe.temporal_conv(a0)
            a2 = fe.depthwise_conv(a1)
            a3 = fe.separable_conv(a2)
            a4 = a3.view(a3.size(0), -1)
            a5 = fe.projection(a4)
            a6 = clf.network[0](a5)
            a7 = torch.relu(a6)
            a8 = a7
            a9 = clf.network[3](a8)

            pred = a9.argmax(dim=1).item()
            true_label = y_tensor[i].item()

        is_correct = pred == true_label

        # decide whether to compute attribution for this trial
        if trial_type == "correct" and not is_correct:
            continue
        if trial_type == "misclassified" and is_correct:
            continue

        # attribution targets predicted class for misclassified,
        # true class for correct
        attr_target = true_label if is_correct else pred

        # backward relevance pass
        R = torch.zeros_like(a9)
        R[0, attr_target] = a9[0, attr_target]

        R = lrp_linear(clf.network[3], a8, R, epsilon)
        R = lrp_linear(clf.network[0], a5, R, epsilon)
        R = lrp_linear(fe.projection, a4, R, epsilon)
        R = R.view(a3.shape)

        with torch.no_grad():
            pre_pool_sep = fe.separable_conv[:4](a2)
        R = lrp_avgpool(fe.separable_conv[4], pre_pool_sep, R)
        R = lrp_epsilon_conv2d(
            fe.separable_conv[1], pre_pointwise := fe.separable_conv[0](a2), R, epsilon
        )
        R = lrp_epsilon_conv2d(fe.separable_conv[0], a2, R, epsilon)

        with torch.no_grad():
            pre_pool_dw = fe.depthwise_conv[:3](a1)
        R = lrp_avgpool(fe.depthwise_conv[3], pre_pool_dw, R)
        R = lrp_epsilon_conv2d(fe.depthwise_conv[0], a1, R, epsilon)

        if first_layer_rule == "zplus":
            R = lrp_zplus_conv2d(fe.temporal_conv[0], a0, R, epsilon)
        else:
            R = lrp_epsilon_conv2d(fe.temporal_conv[0], a0, R, epsilon)

        attr = R.detach().squeeze().numpy()

        if is_correct:
            correct_attrs.append(attr)
            correct_labels.append(true_label)
        else:
            misc_attrs.append(attr)
            misc_true_labels.append(true_label)
            misc_pred_labels.append(pred)

    def build_output(attrs, labels, n_classes):
        attributions = np.array(attrs) if attrs else np.array([])
        labels_arr = np.array(labels)
        mean_attr_per_class = {}
        for cls in range(n_classes):
            mask = labels_arr == cls
            if mask.sum() > 0:
                mean_attr_per_class[cls] = attributions[mask].mean(axis=0)
            else:
                mean_attr_per_class[cls] = None
        return attributions, labels_arr, mean_attr_per_class

    correct_out = build_output(correct_attrs, correct_labels, n_classes)

    misc_out = (
        np.array(misc_attrs) if misc_attrs else np.array([]),
        np.array(misc_true_labels),
        np.array(misc_pred_labels),
        build_output(misc_attrs, misc_pred_labels, n_classes)[2],
    )

    if trial_type == "correct":
        return correct_out
    elif trial_type == "misclassified":
        return misc_out
    else:
        return correct_out, misc_out


def run_lrp_full(
    valid_subjects,
    checkpoint_tag,
    attr_save_path,
    n_classes=2,
    use_multiclass=False,
    first_layer_rule="epsilon",
):
    """
    Runs LRP on all valid subjects, saves per-subject attributions to disk.
    Resume-capable: skips subjects with existing .npz files.
    Saves per subject: mean attribution per class, labels, trial count.
    # both kappa and composite stopping criteria saved per subject.
    # one-time run makes storage cost negligible relative to rerun cost.
    # composite may be needed for stopping-criterion comparison in thesis
    # writing or if a reviewer asks about criterion sensitivity.
    """
    attr_save_path = Path(attr_save_path)
    completed = []
    skipped = []
    failed = []

    for idx, subj in enumerate(valid_subjects):
        save_file = attr_save_path / f"{subj}_lrp.npz"

        if save_file.exists():
            skipped.append(subj)
            continue

        try:
            # run both stopping criteria, save together in one file
            # one-time run, cheap to save both, avoids rerunning if
            # composite attributions are needed for thesis writing or
            # stopping-criterion comparison figures
            save_dict = {"n_classes": n_classes}

            for criterion in ["kappa", "composite"]:
                attrs, labels, mean_by_class = compute_lrp(
                    subject=subj,
                    checkpoint_tag=checkpoint_tag,
                    n_classes=n_classes,
                    use_multiclass=use_multiclass,
                    stopping_criterion=criterion,
                    first_layer_rule=first_layer_rule,
                    trial_type="correct",
                )
                save_dict[f"labels_{criterion}"] = labels
                save_dict[f"n_trials_{criterion}"] = len(attrs)
                for cls in range(n_classes):
                    if mean_by_class[cls] is not None:
                        save_dict[f"mean_attr_class_{cls}_{criterion}"] = mean_by_class[
                            cls
                        ]

            np.savez(save_file, **save_dict)
            completed.append(subj)

            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx+1}/{len(valid_subjects)} "
                    f"({len(completed)} computed, "
                    f"{len(skipped)} skipped, "
                    f"{len(failed)} failed)"
                )

        except Exception as e:
            print(f"FAILED {subj}: {e}")
            failed.append(subj)

    print(
        f"\nDone. Computed: {len(completed)}, "
        f"Skipped: {len(skipped)}, Failed: {len(failed)}"
    )
    if failed:
        print(f"Failed subjects: {failed}")

    return completed, skipped, failed


def load_lrp_attributions(attr_save_path, valid_subjects, n_classes=2):
    """
    Loads all saved LRP attribution files into memory.
    Returns dict: {subject_id: {criterion: {class_idx: mean_attr}}}
    """
    attr_save_path = Path(attr_save_path)
    all_results = {}
    missing = []

    for subj in valid_subjects:
        save_file = attr_save_path / f"{subj}_lrp.npz"
        if not save_file.exists():
            missing.append(subj)
            continue

        data = np.load(save_file, allow_pickle=False)
        subj_results = {}

        for criterion in ["kappa", "composite"]:
            mean_by_class = {}
            for cls in range(n_classes):
                key = f"mean_attr_class_{cls}_{criterion}"
                mean_by_class[cls] = data[key] if key in data else None
            subj_results[criterion] = {
                "mean_by_class": mean_by_class,
                "n_trials": int(data[f"n_trials_{criterion}"]),
            }

        all_results[subj] = subj_results

    print(f"Loaded: {len(all_results)}/{ len(valid_subjects)} subjects")
    if missing:
        print(f"Missing: {missing}")

    return all_results


def compute_per_class_accuracy_106(
    valid_subjects,
    checkpoint_tag,
    save_path,
    n_classes=4,
    use_multiclass=True,
    use_16ch=True,
    stopping_criterion="kappa",
):
    """
    Loads each subject's checkpoint, runs evaluate(), saves per-class
    accuracy and confusion matrix for all 106 subjects.
    Resume-capable: skips if output file already exists.
    """
    save_file = Path(save_path) / f"{checkpoint_tag}_per_class_accuracy.json"

    if save_file.exists():
        print(f"Already exists, loading: {save_file}")
        with open(save_file) as f:
            return json.load(f)

    results = {}
    failed = []

    for idx, subj in enumerate(valid_subjects):
        try:
            # load target data once, reused for both criteria
            _, _, target_eval_loader, _ = prepare_dataloaders(
                test_subject_id=subj,
                valid_subjects=valid_subjects,
                use_16ch=use_16ch,
                use_multiclass=use_multiclass,
            )

            results[subj] = {}

            for criterion in ["kappa", "composite"]:
                ckpt_path = (
                    results_path
                    / "checkpoints"
                    / checkpoint_tag
                    / f"{subj}_{criterion}.pt"
                )
                state_dict = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )

                model = DANN(
                    n_channels=16,
                    n_timepoints=480,
                    n_classes=n_classes,
                    embed_dim=128,
                    dropout_rate=0.25,
                    use_conditional_discriminator=True,
                    feature_extractor_type="eegnet",
                )
                model.load_state_dict(state_dict)
                model.eval()

                acc, kappa, auc, per_class_acc, conf_mat = evaluate(
                    model, target_eval_loader
                )

                results[subj][criterion] = {
                    "kappa": kappa,
                    "accuracy": acc,
                    "per_class_accuracy": per_class_acc,
                    "confusion_matrix": conf_mat,
                }

            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx+1}/{len(valid_subjects)}")

        except Exception as e:
            print(f"FAILED {subj}: {e}")
            failed.append(subj)

    with open(save_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)}/{ len(valid_subjects)} subjects saved.")
    print(f"Saved to: {save_file}")
    if failed:
        print(f"Failed: {failed}")

    return results


def compute_deeplift(
    subject,
    checkpoint_tag,
    n_classes=2,
    use_multiclass=False,
    stopping_criterion="kappa",
    baseline="mean",
    trial_type="correct",
):
    """
    Computes DeepLIFT attributions for one subject using captum.
    Requires folded model (BatchNorm absorbed) for captum compatibility.

    baseline: 'mean' uses mean of subject's trials as reference (default,
              more meaningful for z-score normalised EEG than zero).
              'zero' uses zero input as reference (standard captum default,
              useful for direct comparison with IG).
    trial_type: 'correct', 'misclassified', or 'all' (same as compute_lrp).

    Returns:
        if trial_type != 'all':
            attributions: (n_trials, n_channels, n_timepoints)
            labels: (n_trials,) true labels
            mean_attr_per_class: dict {class_idx: mean attr (n_channels, n_timepoints)}
        if trial_type == 'all':
            (correct_output, misclassified_output)
    """

    model = load_model_for_xai(
        subject,
        checkpoint_tag,
        n_classes=n_classes,
        stopping_criterion=stopping_criterion,
    )
    wrapper = DANNClassifierWrapper(model)
    folded = fold_batchnorm(wrapper)
    folded.eval()

    X, y = load_subject_data_flexible(
        subject, use_multiclass=use_multiclass, use_16ch=True
    )
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # build baseline
    if baseline == "mean":
        baseline_input = X_tensor.mean(dim=0, keepdim=True)
    else:
        baseline_input = torch.zeros_like(X_tensor[:1])

    dl = DeepLift(folded)

    correct_attrs, correct_labels = [], []
    misc_attrs, misc_true_labels, misc_pred_labels = [], [], []

    for i in range(len(X_tensor)):
        inp = X_tensor[i : i + 1].requires_grad_(True)

        with torch.no_grad():
            pred = folded(inp).argmax(dim=1).item()

        true_label = y_tensor[i].item()
        is_correct = pred == true_label

        if trial_type == "correct" and not is_correct:
            continue
        if trial_type == "misclassified" and is_correct:
            continue

        attr_target = true_label if is_correct else pred

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attr = dl.attribute(inp, baselines=baseline_input, target=attr_target)

        # shape (1,1,16,480) -> (16,480)
        attr_np = attr.detach().squeeze().numpy()

        if is_correct:
            correct_attrs.append(attr_np)
            correct_labels.append(true_label)
        else:
            misc_attrs.append(attr_np)
            misc_true_labels.append(true_label)
            misc_pred_labels.append(pred)

    def build_output(attrs, labels, n_classes):
        attributions = np.array(attrs) if attrs else np.array([])
        labels_arr = np.array(labels)
        mean_attr_per_class = {}
        for cls in range(n_classes):
            mask = labels_arr == cls
            if mask.sum() > 0:
                mean_attr_per_class[cls] = attributions[mask].mean(axis=0)
            else:
                mean_attr_per_class[cls] = None
        return attributions, labels_arr, mean_attr_per_class

    correct_out = build_output(correct_attrs, correct_labels, n_classes)
    misc_out = (
        np.array(misc_attrs) if misc_attrs else np.array([]),
        np.array(misc_true_labels),
        np.array(misc_pred_labels),
        build_output(misc_attrs, misc_pred_labels, n_classes)[2],
    )

    if trial_type == "correct":
        return correct_out
    elif trial_type == "misclassified":
        return misc_out
    else:
        return correct_out, misc_out


def run_deeplift_full(
    valid_subjects,
    checkpoint_tag,
    attr_save_path,
    n_classes=2,
    use_multiclass=False,
    baseline="mean",
    first_layer_rule="epsilon",
):
    """
    Runs DeepLIFT on all valid subjects, saves per-subject attributions.
    Resume-capable: skips subjects with existing .npz files.
    Both kappa and composite criteria saved per subject.
    baseline: 'mean' or 'zero', see compute_deeplift.
    """
    attr_save_path = Path(attr_save_path)
    completed, skipped, failed = [], [], []

    for idx, subj in enumerate(valid_subjects):
        save_file = attr_save_path / f"{subj}_deeplift.npz"

        if save_file.exists():
            skipped.append(subj)
            continue

        try:
            save_dict = {"n_classes": n_classes}

            for criterion in ["kappa", "composite"]:
                attrs, labels, mean_by_class = compute_deeplift(
                    subject=subj,
                    checkpoint_tag=checkpoint_tag,
                    n_classes=n_classes,
                    use_multiclass=use_multiclass,
                    stopping_criterion=criterion,
                    baseline=baseline,
                    trial_type="correct",
                )
                save_dict[f"labels_{criterion}"] = labels
                save_dict[f"n_trials_{criterion}"] = len(attrs)
                for cls in range(n_classes):
                    if mean_by_class[cls] is not None:
                        save_dict[f"mean_attr_class_{cls}_{criterion}"] = mean_by_class[
                            cls
                        ]

            np.savez(save_file, **save_dict)
            completed.append(subj)

            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx+1}/{len(valid_subjects)} "
                    f"({len(completed)} computed, "
                    f"{len(skipped)} skipped, "
                    f"{len(failed)} failed)"
                )

        except Exception as e:
            print(f"FAILED {subj}: {e}")
            failed.append(subj)

    print(
        f"\nDone. Computed: {len(completed)}, "
        f"Skipped: {len(skipped)}, Failed: {len(failed)}"
    )
    if failed:
        print(f"Failed: {failed}")

    return completed, skipped, failed


def load_deeplift_attributions(attr_save_path, valid_subjects, n_classes=2):
    """
    Loads all saved DeepLIFT attribution files into memory.
    Returns dict: {subject_id: {criterion: {class_idx: mean_attr}}}
    Same structure as load_lrp_attributions.
    """
    attr_save_path = Path(attr_save_path)
    all_results = {}
    missing = []

    for subj in valid_subjects:
        save_file = attr_save_path / f"{subj}_deeplift.npz"
        if not save_file.exists():
            missing.append(subj)
            continue

        data = np.load(save_file, allow_pickle=False)
        subj_results = {}

        for criterion in ["kappa", "composite"]:
            mean_by_class = {}
            for cls in range(n_classes):
                key = f"mean_attr_class_{cls}_{criterion}"
                mean_by_class[cls] = data[key] if key in data else None
            subj_results[criterion] = {
                "mean_by_class": mean_by_class,
                "n_trials": int(data[f"n_trials_{criterion}"]),
            }

        all_results[subj] = subj_results

    print(f"Loaded: {len(all_results)}/{len(valid_subjects)} subjects")
    if missing:
        print(f"Missing: {missing}")

    return all_results


# ---------------SHAP-------------------
def compute_shap(
    subject,
    checkpoint_tag,
    valid_subjects,
    n_classes=2,
    use_multiclass=False,
    stopping_criterion="kappa",
    background_fraction=0.60,
    trials_per_source_subject=26,
    trial_type="correct",
    seed=25,
):
    """
    Computes GradientSHAP attributions for one subject.

    Background samples drawn from source pool with equal per-subject
    sampling to avoid bias toward high-trial-count subjects.
    (Drawback 1 fix: equal representation across 105 source subjects.)

    background_fraction: approximate fraction of source pool to use.
        Default 0.60, gives ~2730 samples from ~4640 available.
    trials_per_source_subject: fixed trials drawn per source subject.
        Default 26 (105 subjects x 26 = 2730, ~60% of 4640).
    seed: fixed seed for background sampling reproducibility.
        (Drawback 3 fix: reproducible background selection.)

    Drawbacks acknowledged but not fixable:
    - Background represents source not target distribution (drawback 2)
    - GradientSHAP linearity assumption (drawback 4)

    Returns same format as compute_lrp and compute_deeplift.
    """

    model = load_model_for_xai(
        subject,
        checkpoint_tag,
        n_classes=n_classes,
        stopping_criterion=stopping_criterion,
    )
    wrapper = DANNClassifierWrapper(model)
    folded = fold_batchnorm(wrapper)
    folded.eval()

    # load target subject data
    X, y = load_subject_data_flexible(
        subject, use_multiclass=use_multiclass, use_16ch=True
    )
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # build background: equal sampling per source subject
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_loader, _, _, _ = prepare_dataloaders(
        test_subject_id=subject,
        valid_subjects=valid_subjects,
        use_16ch=True,
        use_multiclass=use_multiclass,
    )

    # collect all source trials, grouped by batch
    all_source = []
    for batch_x, _ in source_loader:
        all_source.append(batch_x)
    all_source = torch.cat(all_source, dim=0)

    # sample equally per source subject is approximated by random sampling
    # from the full pool with a fixed seed, giving stable background
    n_background = min(int(len(all_source) * background_fraction), len(all_source))
    idx = torch.randperm(len(all_source))[:n_background]
    background = all_source[idx]

    gs = GradientShap(folded)

    correct_attrs, correct_labels = [], []
    misc_attrs, misc_true_labels, misc_pred_labels = [], [], []

    for i in range(len(X_tensor)):
        inp = X_tensor[i : i + 1]

        with torch.no_grad():
            pred = folded(inp).argmax(dim=1).item()

        true_label = y_tensor[i].item()
        is_correct = pred == true_label

        if trial_type == "correct" and not is_correct:
            continue
        if trial_type == "misclassified" and is_correct:
            continue

        attr_target = true_label if is_correct else pred

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attr = gs.attribute(
                inp, baselines=background, target=attr_target, n_samples=1, stdevs=0.0
            )

        attr_np = attr.detach().squeeze().numpy()

        if is_correct:
            correct_attrs.append(attr_np)
            correct_labels.append(true_label)
        else:
            misc_attrs.append(attr_np)
            misc_true_labels.append(true_label)
            misc_pred_labels.append(pred)

    def build_output(attrs, labels, n_classes):
        attributions = np.array(attrs) if attrs else np.array([])
        labels_arr = np.array(labels)
        mean_attr_per_class = {}
        for cls in range(n_classes):
            mask = labels_arr == cls
            if mask.sum() > 0:
                mean_attr_per_class[cls] = attributions[mask].mean(axis=0)
            else:
                mean_attr_per_class[cls] = None
        return attributions, labels_arr, mean_attr_per_class

    correct_out = build_output(correct_attrs, correct_labels, n_classes)
    misc_out = (
        np.array(misc_attrs) if misc_attrs else np.array([]),
        np.array(misc_true_labels),
        np.array(misc_pred_labels),
        build_output(misc_attrs, misc_pred_labels, n_classes)[2],
    )

    if trial_type == "correct":
        return correct_out
    elif trial_type == "misclassified":
        return misc_out
    else:
        return correct_out, misc_out


def run_shap_full(
    valid_subjects,
    checkpoint_tag,
    attr_save_path,
    n_classes=2,
    use_multiclass=False,
    background_fraction=0.60,
    seed=25,
):
    """
    Runs GradientSHAP on all valid subjects, saves per-subject attributions.
    Resume-capable: skips subjects with existing .npz files.
    Both kappa and composite criteria saved per subject.
    background_fraction: fraction of source pool used as background.
    seed: fixed seed for reproducible background sampling.
    """
    attr_save_path = Path(attr_save_path)
    completed, skipped, failed = [], [], []

    for idx, subj in enumerate(valid_subjects):
        save_file = attr_save_path / f"{subj}_shap.npz"

        if save_file.exists():
            skipped.append(subj)
            continue

        try:
            save_dict = {"n_classes": n_classes}

            for criterion in ["kappa", "composite"]:
                attrs, labels, mean_by_class = compute_shap(
                    subject=subj,
                    checkpoint_tag=checkpoint_tag,
                    n_classes=n_classes,
                    valid_subjects=valid_subjects,
                    use_multiclass=use_multiclass,
                    stopping_criterion=criterion,
                    background_fraction=background_fraction,
                    trial_type="correct",
                    seed=seed,
                )
                save_dict[f"labels_{criterion}"] = labels
                save_dict[f"n_trials_{criterion}"] = len(attrs)
                for cls in range(n_classes):
                    if mean_by_class[cls] is not None:
                        save_dict[f"mean_attr_class_{cls}_{criterion}"] = mean_by_class[
                            cls
                        ]

            np.savez(save_file, **save_dict)
            completed.append(subj)

            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx+1}/{len(valid_subjects)} "
                    f"({len(completed)} computed, "
                    f"{len(skipped)} skipped, "
                    f"{len(failed)} failed)"
                )

        except Exception as e:
            print(f"FAILED {subj}: {e}")
            failed.append(subj)

    print(
        f"\nDone. Computed: {len(completed)}, "
        f"Skipped: {len(skipped)}, Failed: {len(failed)}"
    )
    if failed:
        print(f"Failed: {failed}")

    return completed, skipped, failed


def load_shap_attributions(attr_save_path, valid_subjects, n_classes=2):
    """
    Loads all saved SHAP attribution files into memory.
    Same structure as load_lrp_attributions and load_deeplift_attributions.
    """
    attr_save_path = Path(attr_save_path)
    all_results = {}
    missing = []

    for subj in valid_subjects:
        save_file = attr_save_path / f"{subj}_shap.npz"
        if not save_file.exists():
            missing.append(subj)
            continue

        data = np.load(save_file, allow_pickle=False)
        subj_results = {}

        for criterion in ["kappa", "composite"]:
            mean_by_class = {}
            for cls in range(n_classes):
                key = f"mean_attr_class_{cls}_{criterion}"
                mean_by_class[cls] = data[key] if key in data else None
            subj_results[criterion] = {
                "mean_by_class": mean_by_class,
                "n_trials": int(data[f"n_trials_{criterion}"]),
            }

        all_results[subj] = subj_results

    print(f"Loaded: {len(all_results)}/{len(valid_subjects)} subjects")
    if missing:
        print(f"Missing: {missing}")

    return all_results


# ---------------IG--------------------
def compute_ig(
    subject,
    checkpoint_tag,
    n_classes=2,
    use_multiclass=False,
    stopping_criterion="kappa",
    baseline="zero",
    n_steps=50,
    trial_type="correct",
):
    """
    Computes Integrated Gradients attributions for one subject.

    baseline: 'zero' uses zero input as reference (default, most principled,
              satisfies completeness axiom from zero signal to actual input).
              'mean' uses mean of subject's trials (same as DeepLIFT baseline,
              useful for direct comparison with DeepLIFT results).
    n_steps: number of integration steps. 50 is standard for prototyping,
             increase to 100-200 for final results if needed.
    trial_type: 'correct', 'misclassified', or 'all'.

    Returns same format as compute_lrp, compute_deeplift, compute_shap.
    """

    model = load_model_for_xai(
        subject,
        checkpoint_tag,
        n_classes=n_classes,
        stopping_criterion=stopping_criterion,
    )
    wrapper = DANNClassifierWrapper(model)
    folded = fold_batchnorm(wrapper)
    folded.eval()

    X, y = load_subject_data_flexible(
        subject, use_multiclass=use_multiclass, use_16ch=True
    )
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # build baseline
    if baseline == "mean":
        baseline_input = X_tensor.mean(dim=0, keepdim=True)
    else:
        baseline_input = torch.zeros_like(X_tensor[:1])

    ig = IntegratedGradients(folded)

    correct_attrs, correct_labels = [], []
    misc_attrs, misc_true_labels, misc_pred_labels = [], [], []

    for i in range(len(X_tensor)):
        inp = X_tensor[i : i + 1]

        with torch.no_grad():
            pred = folded(inp).argmax(dim=1).item()

        true_label = y_tensor[i].item()
        is_correct = pred == true_label

        if trial_type == "correct" and not is_correct:
            continue
        if trial_type == "misclassified" and is_correct:
            continue

        attr_target = true_label if is_correct else pred

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attr = ig.attribute(
                inp,
                baselines=baseline_input,
                target=attr_target,
                n_steps=n_steps,
            )

        attr_np = attr.detach().squeeze().numpy()

        if is_correct:
            correct_attrs.append(attr_np)
            correct_labels.append(true_label)
        else:
            misc_attrs.append(attr_np)
            misc_true_labels.append(true_label)
            misc_pred_labels.append(pred)

    def build_output(attrs, labels, n_classes):
        attributions = np.array(attrs) if attrs else np.array([])
        labels_arr = np.array(labels)
        mean_attr_per_class = {}
        for cls in range(n_classes):
            mask = labels_arr == cls
            if mask.sum() > 0:
                mean_attr_per_class[cls] = attributions[mask].mean(axis=0)
            else:
                mean_attr_per_class[cls] = None
        return attributions, labels_arr, mean_attr_per_class

    correct_out = build_output(correct_attrs, correct_labels, n_classes)
    misc_out = (
        np.array(misc_attrs) if misc_attrs else np.array([]),
        np.array(misc_true_labels),
        np.array(misc_pred_labels),
        build_output(misc_attrs, misc_pred_labels, n_classes)[2],
    )

    if trial_type == "correct":
        return correct_out
    elif trial_type == "misclassified":
        return misc_out
    else:
        return correct_out, misc_out


def run_ig_full(
    valid_subjects,
    checkpoint_tag,
    attr_save_path,
    n_classes=2,
    use_multiclass=False,
    baseline="zero",
    n_steps=50,
):
    """
    Runs Integrated Gradients on all valid subjects, saves per-subject attributions.
    Resume-capable: skips subjects with existing .npz files.
    Both kappa and composite criteria saved per subject.
    baseline: 'zero' (default, most principled) or 'mean'.
    n_steps: integration steps, 50 for prototyping, 100-200 for final.
    """
    attr_save_path = Path(attr_save_path)
    completed, skipped, failed = [], [], []

    for idx, subj in enumerate(valid_subjects):
        save_file = attr_save_path / f"{subj}_ig.npz"

        if save_file.exists():
            skipped.append(subj)
            continue

        try:
            save_dict = {"n_classes": n_classes}

            for criterion in ["kappa", "composite"]:
                attrs, labels, mean_by_class = compute_ig(
                    subject=subj,
                    checkpoint_tag=checkpoint_tag,
                    n_classes=n_classes,
                    use_multiclass=use_multiclass,
                    stopping_criterion=criterion,
                    baseline=baseline,
                    n_steps=n_steps,
                    trial_type="correct",
                )
                save_dict[f"labels_{criterion}"] = labels
                save_dict[f"n_trials_{criterion}"] = len(attrs)
                for cls in range(n_classes):
                    if mean_by_class[cls] is not None:
                        save_dict[f"mean_attr_class_{cls}_{criterion}"] = mean_by_class[
                            cls
                        ]

            np.savez(save_file, **save_dict)
            completed.append(subj)

            if (idx + 1) % 10 == 0:
                print(
                    f"Progress: {idx+1}/{len(valid_subjects)} "
                    f"({len(completed)} computed, "
                    f"{len(skipped)} skipped, "
                    f"{len(failed)} failed)"
                )

        except Exception as e:
            print(f"FAILED {subj}: {e}")
            failed.append(subj)

    print(
        f"\nDone. Computed: {len(completed)}, "
        f"Skipped: {len(skipped)}, Failed: {len(failed)}"
    )
    if failed:
        print(f"Failed: {failed}")

    return completed, skipped, failed


def load_ig_attributions(attr_save_path, valid_subjects, n_classes=2):
    """
    Loads all saved IG attribution files into memory.
    Same structure as load_lrp_attributions, load_deeplift_attributions,
    load_shap_attributions.
    """
    attr_save_path = Path(attr_save_path)
    all_results = {}
    missing = []

    for subj in valid_subjects:
        save_file = attr_save_path / f"{subj}_ig.npz"
        if not save_file.exists():
            missing.append(subj)
            continue

        data = np.load(save_file, allow_pickle=False)
        subj_results = {}

        for criterion in ["kappa", "composite"]:
            mean_by_class = {}
            for cls in range(n_classes):
                key = f"mean_attr_class_{cls}_{criterion}"
                mean_by_class[cls] = data[key] if key in data else None
            subj_results[criterion] = {
                "mean_by_class": mean_by_class,
                "n_trials": int(data[f"n_trials_{criterion}"]),
            }

        all_results[subj] = subj_results

    print(f"Loaded: {len(all_results)}/{len(valid_subjects)} subjects")
    if missing:
        print(f"Missing: {missing}")

    return all_results


def run_misclassified_single(
    subject,
    valid_subjects,
    method="lrp",
    checkpoint_tag="cdan_16ch_binary",
    stopping_criterion="kappa",
    n_classes=2,
    use_multiclass=False,
    use_16ch=True,
    # SHAP specific
    background_fraction=0.6,
    trials_per_source_subject=26,
    shap_seed=25,
    # IG specific
    ig_baseline="zero",
    n_steps=50,
    # DeepLIFT specific
    deeplift_baseline="mean",
):
    # Dispatcher: call the right compute function
    if method == "lrp":
        correct, misclassified = compute_lrp(
            subject,
            checkpoint_tag,
            n_classes=n_classes,
            use_multiclass=use_multiclass,
            stopping_criterion=stopping_criterion,
            trial_type="both",
        )
    elif method == "deeplift":
        correct, misclassified = compute_deeplift(
            subject,
            checkpoint_tag,
            n_classes=n_classes,
            use_multiclass=use_multiclass,
            stopping_criterion=stopping_criterion,
            baseline=deeplift_baseline,
            trial_type="both",
        )
    elif method == "shap":
        correct, misclassified = compute_shap(
            subject,
            checkpoint_tag,
            valid_subjects=valid_subjects,
            n_classes=n_classes,
            use_multiclass=use_multiclass,
            stopping_criterion=stopping_criterion,
            background_fraction=background_fraction,
            trials_per_source_subject=trials_per_source_subject,
            trial_type="both",
            seed=shap_seed,
        )
    elif method == "ig":
        correct, misclassified = compute_ig(
            subject,
            checkpoint_tag,
            n_classes=n_classes,
            use_multiclass=use_multiclass,
            stopping_criterion=stopping_criterion,
            baseline=ig_baseline,
            n_steps=n_steps,
            trial_type="both",
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from lrp, deeplift, shap, ig"
        )

    correct_per_trial = correct[0]
    correct_labels = correct[1]
    correct_mean = correct[2]

    has_misclassified = len(misclassified[0]) > 0
    if has_misclassified:
        misc_per_trial = misclassified[0]
        misc_true_labels = misclassified[1]
        misc_pred_labels = misclassified[2]
        misc_mean = misclassified[3]
        misc_label_pairs = np.stack([misc_true_labels, misc_pred_labels], axis=1)

    # Confidence scores via forward pass
    model = load_model_for_xai(
        subject,
        checkpoint_tag,
        n_classes=n_classes,
        stopping_criterion=stopping_criterion,
    )
    model.eval()

    _, _, target_eval_loader, _ = prepare_dataloaders(
        subject, valid_subjects, use_16ch=use_16ch, use_multiclass=use_multiclass
    )

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in target_eval_loader:
            class_out, _, _ = model(x, alpha=0.0)
            probs = torch.softmax(class_out, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if has_misclassified:
        misc_mask = all_preds != all_labels
        misc_confidence = all_probs[misc_mask]

    # Build save dict
    save_dict = {
        "correct_mean_class_0": correct_mean[0],
        "correct_mean_class_1": correct_mean[1],
        "correct_per_trial_attr": correct_per_trial,
        "correct_labels": correct_labels,
        "n_correct": len(correct_labels),
        "stopping_criterion": stopping_criterion,
        "n_classes": n_classes,
        "method": method,
    }

    for cls in range(2, n_classes):
        save_dict[f"correct_mean_class_{cls}"] = correct_mean.get(
            cls, np.zeros((16, 480))
        )

    if has_misclassified:
        save_dict.update(
            {
                "misclassified_mean_class_0": misc_mean[0],
                "misclassified_mean_class_1": misc_mean[1],
                "misclassified_per_trial_attr": misc_per_trial,
                "misclassified_labels": misc_true_labels,
                "misclassified_label_pairs": misc_label_pairs,
                "misclassified_confidence": misc_confidence,
                "n_misclassified": len(misc_true_labels),
            }
        )
        for cls in range(2, n_classes):
            save_dict[f"misclassified_mean_class_{cls}"] = misc_mean.get(
                cls, np.zeros((16, 480))
            )
    else:
        save_dict.update(
            {
                "misclassified_mean_class_0": np.zeros((16, 480)),
                "misclassified_mean_class_1": np.zeros((16, 480)),
                "misclassified_per_trial_attr": np.zeros((0, 16, 480)),
                "misclassified_labels": np.array([]),
                "misclassified_label_pairs": np.zeros((0, 2)),
                "misclassified_confidence": np.zeros((0, n_classes)),
                "n_misclassified": 0,
            }
        )
        for cls in range(2, n_classes):
            save_dict[f"misclassified_mean_class_{cls}"] = np.zeros((16, 480))

    return save_dict


def run_and_save_misclassified(
    subjects,
    valid_subjects,
    save_path,
    method="lrp",
    checkpoint_tag="cdan_16ch_binary",
    stopping_criterion="kappa",
    n_classes=2,
    use_multiclass=False,
    use_16ch=True,
    background_fraction=0.6,
    trials_per_source_subject=26,
    shap_seed=25,
    ig_baseline="zero",
    n_steps=50,
    deeplift_baseline="mean",
):
    results_summary = {}
    start_total = time.time()

    for idx, subject in enumerate(subjects):
        start = time.time()
        try:
            save_dict = run_misclassified_single(
                subject,
                valid_subjects=valid_subjects,
                method=method,
                checkpoint_tag=checkpoint_tag,
                stopping_criterion=stopping_criterion,
                n_classes=n_classes,
                use_multiclass=use_multiclass,
                use_16ch=use_16ch,
                background_fraction=background_fraction,
                trials_per_source_subject=trials_per_source_subject,
                shap_seed=shap_seed,
                ig_baseline=ig_baseline,
                n_steps=n_steps,
                deeplift_baseline=deeplift_baseline,
            )
            # Save with method name in filename
            np.savez(save_path / f"{subject}_misc_{method}.npz", **save_dict)
            elapsed = time.time() - start
            results_summary[subject] = {
                "n_correct": int(save_dict["n_correct"]),
                "n_misclassified": int(save_dict["n_misclassified"]),
                "time": round(elapsed, 2),
            }
            print(
                f"[{idx+1:3d}/{len(subjects)}] {subject}: "
                f"correct={save_dict['n_correct']}, "
                f"misc={save_dict['n_misclassified']}, "
                f"time={elapsed:.1f}s"
            )
        except Exception as e:
            print(f"[{idx+1:3d}/{len(subjects)}] {subject}: FAILED - {e}")
            results_summary[subject] = {"error": str(e)}

    total = time.time() - start_total
    print(f"\nDone. Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Method: {method}, Saved to: {save_path}")
    return results_summary
