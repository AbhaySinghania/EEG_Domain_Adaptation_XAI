import json
import csv
from datetime import datetime
import numpy as np
import torch.optim as optim
from pathlib import Path
import os
from models import DANN
import torch
from data_loading import prepare_dataloaders
from metrics import compute_pre_adaptation_mmd, compute_post_adaptation_mmd, evaluate
from training_utils import (
    get_optimizer,
    EarlyStopping,
    BestModelCheckpoint,
    initialise_log,
    apply_adabn,
    train_epoch,
    get_alpha,
)

# Repository root. Set EEG_PROJECT_ROOT to use data and results stored elsewhere.
base_path = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
results_path = base_path / "results"


def train_dann_single_fold(
    target_subject,
    valid_subjects,
    stopping_criterion="kappa",
    n_epochs_max=100,
    batch_size=32,
    optimizer_choice="adamw",
    embed_dim=128,
    dropout_rate=0.25,
    lambda_domain=0.6,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    composite_threshold=1.0,
    min_epochs=20,
    warmup_epochs=10,
    use_gradient_penalty=True,
    use_conditional_discriminator=False,
    use_entropy_weighting=True,
    use_riemannian_alignment=False,
    use_adabn=False,
    feature_extractor_type="eegnet",
    use_augmentation=False,
    use_multiclass=False,
    use_windowed_data=False,
    n_classes=2,
    use_16ch=False,
    window_start_sec=0.5,
    window_end_sec=2.5,
    sfreq=160,
    save_checkpoint=False,
    checkpoint_tag=None,
):
    """
    trains DANN for one LOSO fold
    target_subject: subject ID used as test subject
    valid_subjects: full list of valid subject IDs
    stopping_criterion: 'kappa' or 'composite'
    returns: trained model with best weights, complete log dictionary
    """
    if save_checkpoint and checkpoint_tag is None:
        raise ValueError(
            "checkpoint_tag is required when save_checkpoint=True, to prevent "
            "different configs (e.g. binary vs multiclass) from overwriting "
            "each other's checkpoint files."
        )
    n_channels = 16 if use_16ch else 8
    n_timepoints = (
        int((window_end_sec - window_start_sec) * sfreq) if use_windowed_data else 480
    )

    # step 1: prepare data loaders
    source_loader, target_loader, target_eval_loader, y_target = prepare_dataloaders(
        target_subject,
        valid_subjects,
        batch_size=batch_size,
        use_riemannian_alignment=use_riemannian_alignment,
        use_windowed_data=use_windowed_data,
        use_multiclass=use_multiclass,
        use_16ch=use_16ch,
    )

    # step 2: pre-adaptation MMD
    log = initialise_log()
    pre_mmd_ms, pre_mmd_opt, pre_bw, pre_riem = compute_pre_adaptation_mmd(
        source_loader, target_loader
    )
    log["pre_mmd_bandpower_multiscale"] = pre_mmd_ms
    log["pre_mmd_bandpower_optimal"] = pre_mmd_opt
    log["pre_mmd_bandpower_optimal_bandwidth"] = pre_bw
    log["pre_riemannian"] = pre_riem

    # step 3: initialise model, optimizer, early stopping, checkpoint
    model = DANN(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        embed_dim=embed_dim,
        dropout_rate=dropout_rate,
        use_conditional_discriminator=use_conditional_discriminator,
        feature_extractor_type=feature_extractor_type,
    )

    optimizer = get_optimizer(model, choice=optimizer_choice)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    early_stopper = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_min_delta, mode="max"
    )

    domain_stopper = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_min_delta, mode="min"
    )

    checkpoint = BestModelCheckpoint(mode="max")

    # step 4: calculate total steps for alpha scheduling
    total_steps = len(source_loader) * n_epochs_max
    current_step = 0
    initial_domain_loss = None

    # step 5: training loop
    for epoch in range(n_epochs_max):

        # train one epoch
        # progressive warmup: classification only for first warmup_epochs
        current_lambda = 0.0 if epoch < warmup_epochs else lambda_domain

        (
            class_loss,
            domain_loss,
            domain_accuracy,
            source_domain_accuracy,
            target_domain_accuracy,
            current_step,
        ) = train_epoch(
            model,
            source_loader,
            target_loader,
            optimizer,
            current_step,
            total_steps,
            lambda_domain=current_lambda,
            use_gradient_penalty=use_gradient_penalty,
            use_entropy_weighting=use_entropy_weighting,
            use_augmentation=use_augmentation,
        )

        # store initial domain loss for adaptation quality
        if epoch == 0:
            initial_domain_loss = domain_loss

        # evaluate on target subject
        accuracy, kappa, auc, per_class_accuracy, confusion_mat = evaluate(
            model, target_eval_loader
        )

        # get current alpha and learning rate
        current_alpha = get_alpha(current_step, total_steps)
        current_lr = optimizer.param_groups[0]["lr"]

        # log all metrics
        log["epoch"].append(epoch + 1)
        log["class_loss"].append(class_loss)
        log["domain_loss"].append(domain_loss)
        log["accuracy"].append(accuracy)
        log["kappa"].append(kappa)
        log["auc"].append(auc)
        log["alpha"].append(current_alpha)
        log["learning_rate"].append(current_lr)
        log["domain_accuracy"].append(domain_accuracy)
        log["source_domain_accuracy"].append(source_domain_accuracy)
        log["target_domain_accuracy"].append(target_domain_accuracy)
        log["per_class_accuracy"].append(per_class_accuracy)
        log["confusion_matrix"].append(confusion_mat)

        # determine stopping metric based on criterion
        if stopping_criterion == "kappa":
            stopping_metric = kappa

        elif stopping_criterion == "composite":
            # dual early stopping: monitor kappa and domain loss independently training continues as long as either is still improving
            stopping_metric = kappa

        # save best weights
        saved = checkpoint.step(model, stopping_metric, epoch)
        log["best_weights_saved"].append(saved)

        # update learning rate scheduler
        scheduler.step(stopping_metric)

        # check early stopping based on criterion
        if stopping_criterion == "kappa":
            if epoch >= min_epochs:
                early_stopper.step(kappa)
                if early_stopper.should_stop:
                    log["early_stopping_triggered"] = True
                    break

        elif stopping_criterion == "composite":
            if epoch >= min_epochs:
                early_stopper.step(kappa)
                domain_stopper.step(domain_loss)
                if early_stopper.should_stop and domain_stopper.should_stop:
                    log["early_stopping_triggered"] = True
                    break

    # step 6: load best weights
    checkpoint.load_best_weights(model)
    # updates batch norm statistics using target data only
    if save_checkpoint:
        checkpoint_dir = results_path / "checkpoints" / checkpoint_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_filename = f"{target_subject}_{stopping_criterion}.pt"
        torch.save(model.state_dict(), checkpoint_dir / checkpoint_filename)
    if use_adabn:
        model = apply_adabn(model, target_loader)

    # step 7: post-adaptation MMD
    post_mmd_results = compute_post_adaptation_mmd(model, source_loader, target_loader)
    log.update(post_mmd_results)

    # step 8: adaptation quality score
    final_domain_loss = log["domain_loss"][-1]
    log["initial_domain_loss"] = initial_domain_loss
    log["final_domain_loss"] = final_domain_loss

    if initial_domain_loss and initial_domain_loss > 0:
        log["adaptation_quality"] = final_domain_loss / initial_domain_loss
    else:
        log["adaptation_quality"] = None

    # step 9: fill summary fields
    log["best_epoch"] = checkpoint.best_epoch + 1
    log["best_kappa"] = checkpoint.best_value
    log["total_epochs_run"] = len(log["epoch"])

    # at the end of train_dann_single_fold, after loading best weights
    (
        final_accuracy,
        final_kappa,
        final_auc,
        final_per_class_accuracy,
        final_confusion_mat,
    ) = evaluate(model, target_eval_loader)

    log["final_accuracy"] = final_accuracy
    log["final_kappa"] = final_kappa
    log["final_auc"] = final_auc
    log["final_per_class_accuracy"] = final_per_class_accuracy
    log["final_confusion_matrix"] = final_confusion_mat

    return model, log


def save_results(results, results_path, results_filename="loso_results.json"):
    """
    saves full results dictionary to json
    called after each subject for incremental saving
    """
    results_file = results_path / results_filename

    # convert numpy types to python native for json serialisation
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_convert(i) for i in d]
        else:
            return convert(d)

    with open(results_file, "w") as f:
        json.dump(recursive_convert(results), f, indent=2)


def save_summary_csv(results, results_path, summary_filename="loso_summary.csv"):
    """
    saves lightweight summary csv with final metrics per subject
    one row per subject, one column per metric
    """
    summary_file = results_path / summary_filename

    fieldnames = [
        "subject_id",
        "kappa_accuracy",
        "kappa_kappa",
        "kappa_auc",
        "kappa_best_epoch",
        "kappa_total_epochs",
        "kappa_early_stopping",
        "kappa_pre_mmd_multiscale",
        "kappa_post_mmd_multiscale",
        "kappa_pre_riemannian",
        "kappa_post_riemannian",
        "kappa_adaptation_quality",
        "kappa_source_domain_accuracy",
        "kappa_target_domain_accuracy",
        "composite_accuracy",
        "composite_kappa",
        "composite_auc",
        "composite_best_epoch",
        "composite_total_epochs",
        "composite_early_stopping",
        "composite_pre_mmd_multiscale",
        "composite_post_mmd_multiscale",
        "composite_pre_riemannian",
        "composite_post_riemannian",
        "composite_adaptation_quality",
        "composite_source_domain_accuracy",
        "composite_target_domain_accuracy",
    ]

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for subject_id, subject_results in results.items():
            k = subject_results["kappa_stopping"]
            c = subject_results["composite_stopping"]

            row = {
                "subject_id": subject_id,
                "kappa_accuracy": k["final_accuracy"],
                "kappa_kappa": k["final_kappa"],
                "kappa_auc": k["final_auc"],
                "kappa_best_epoch": k["log"]["best_epoch"],
                "kappa_total_epochs": k["log"]["total_epochs_run"],
                "kappa_early_stopping": k["log"]["early_stopping_triggered"],
                "kappa_pre_mmd_multiscale": k["log"]["pre_mmd_bandpower_multiscale"],
                "kappa_post_mmd_multiscale": k["log"]["post_mmd_bandpower_multiscale"],
                "kappa_pre_riemannian": k["log"]["pre_riemannian"],
                "kappa_post_riemannian": k["log"]["post_riemannian"],
                "kappa_adaptation_quality": k["log"]["adaptation_quality"],
                "kappa_source_domain_accuracy": (
                    np.mean(k["log"]["source_domain_accuracy"])
                    if k["log"]["source_domain_accuracy"]
                    else None
                ),
                "kappa_target_domain_accuracy": (
                    np.mean(k["log"]["target_domain_accuracy"])
                    if k["log"]["target_domain_accuracy"]
                    else None
                ),
                "composite_accuracy": c["final_accuracy"],
                "composite_kappa": c["final_kappa"],
                "composite_auc": c["final_auc"],
                "composite_best_epoch": c["log"]["best_epoch"],
                "composite_total_epochs": c["log"]["total_epochs_run"],
                "composite_early_stopping": c["log"]["early_stopping_triggered"],
                "composite_pre_mmd_multiscale": c["log"][
                    "pre_mmd_bandpower_multiscale"
                ],
                "composite_post_mmd_multiscale": c["log"][
                    "post_mmd_bandpower_multiscale"
                ],
                "composite_pre_riemannian": c["log"]["pre_riemannian"],
                "composite_post_riemannian": c["log"]["post_riemannian"],
                "composite_adaptation_quality": c["log"]["adaptation_quality"],
                "composite_source_domain_accuracy": (
                    np.mean(c["log"]["source_domain_accuracy"])
                    if c["log"]["source_domain_accuracy"]
                    else None
                ),
                "composite_target_domain_accuracy": (
                    np.mean(c["log"]["target_domain_accuracy"])
                    if c["log"]["target_domain_accuracy"]
                    else None
                ),
            }
            writer.writerow(row)


def run_loso(
    valid_subjects,
    stopping_criteria=["kappa", "composite"],
    n_epochs_max=100,
    batch_size=32,
    optimizer_choice="adamw",
    embed_dim=128,
    dropout_rate=0.25,
    lambda_domain=0.6,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    resume=True,
    results_filename="loso_results.json",
    min_epochs=20,
    warmup_epochs=10,
    use_gradient_penalty=True,
    use_conditional_discriminator=False,
    use_entropy_weighting=True,
    use_riemannian_alignment=False,
    use_adabn=False,
    feature_extractor_type="eegnet",
    use_augmentation=False,
    n_classes=2,
    use_multiclass=False,
    use_windowed_data=False,
    use_16ch=False,
    save_checkpoint=False,
    checkpoint_tag=None,
):
    """
    runs full leave one subject out cross validation, trains DANN for each subject as test subject, runs both stopping criteria per subject and saves results incrementally after each subject and resume=True skips already processed subjects
    """
    # load existing results if resuming
    if save_checkpoint and checkpoint_tag is None:
        raise ValueError(
            "checkpoint_tag is required when save_checkpoint=True, to prevent "
            "different configs (e.g. binary vs multiclass) from overwriting "
            "each other's checkpoint files."
        )

    # load existing results if resuming
    results_file = results_path / results_filename

    if resume and results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        print(
            f"Resuming from existing results: {len(results)} subjects already processed"
        )
    else:
        results = {}
        print(f"Starting fresh LOSO run")

    total_subjects = len(valid_subjects)
    start_time = datetime.now()

    print(f"Total subjects: {total_subjects}")
    print(f"Stopping criteria: {stopping_criteria}")
    print(f"Max epochs: {n_epochs_max}")
    print(f"Optimizer: {optimizer_choice}")
    print()

    for idx, subject_id in enumerate(valid_subjects):

        # skip if already processed
        if resume and subject_id in results:
            print(f"Skipping {subject_id} (already processed)")
            continue

        subject_start = datetime.now()
        print(f"Processing {subject_id} ({idx+1}/{total_subjects})")

        subject_results = {}

        for criterion in stopping_criteria:

            print(f"  Training with {criterion} stopping criterion")

            try:
                model, log = train_dann_single_fold(
                    target_subject=subject_id,
                    valid_subjects=valid_subjects,
                    stopping_criterion=criterion,
                    n_epochs_max=n_epochs_max,
                    batch_size=batch_size,
                    optimizer_choice=optimizer_choice,
                    embed_dim=embed_dim,
                    dropout_rate=dropout_rate,
                    lambda_domain=lambda_domain,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    min_epochs=min_epochs,
                    warmup_epochs=warmup_epochs,
                    use_gradient_penalty=use_gradient_penalty,
                    use_conditional_discriminator=use_conditional_discriminator,
                    use_entropy_weighting=use_entropy_weighting,
                    use_riemannian_alignment=use_riemannian_alignment,
                    use_adabn=use_adabn,
                    feature_extractor_type=feature_extractor_type,
                    use_augmentation=use_augmentation,
                    n_classes=n_classes,
                    use_multiclass=use_multiclass,
                    use_windowed_data=use_windowed_data,
                    use_16ch=use_16ch,
                    save_checkpoint=save_checkpoint,
                    checkpoint_tag=checkpoint_tag,
                )

                # final metrics already computed inside train_dann_single_fold
                final_accuracy = log["final_accuracy"]
                final_kappa = log["final_kappa"]
                final_auc = log["final_auc"]

                subject_results[f"{criterion}_stopping"] = {
                    "final_accuracy": final_accuracy,
                    "final_kappa": final_kappa,
                    "final_auc": final_auc,
                    "log": log,
                }

                print(
                    f"  {criterion}: accuracy {final_accuracy:.4f} | "
                    f"kappa {final_kappa:.4f} | "
                    f"auc {final_auc:.4f} | "
                    f"epochs {log['total_epochs_run']} | "
                    f"best epoch {log['best_epoch']}"
                )

            except Exception as e:
                print(f"  Error with {criterion} stopping for {subject_id}: {e}")
                subject_results[f"{criterion}_stopping"] = None

        # save subject results
        results[subject_id] = subject_results

        # incremental save after each subject
        save_results(results, results_path, results_filename)
        # save_summary_csv(results, results_path,
        #                 results_filename.replace('.json', '.csv'))

        if "kappa" in stopping_criteria and "composite" in stopping_criteria:
            save_summary_csv(
                results, results_path, results_filename.replace(".json", ".csv")
            )

        subject_time = datetime.now() - subject_start
        elapsed = datetime.now() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total_subjects - idx - 1)

        print(
            f"  Time for subject: {subject_time.seconds}s | "
            f"Elapsed: {elapsed.seconds//60}m | "
            f"Estimated remaining: {remaining.seconds//60}m"
        )
        print()

    print(f"LOSO complete. Results saved to {results_path}")
    return results
