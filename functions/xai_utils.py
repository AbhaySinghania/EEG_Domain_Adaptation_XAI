import numpy as np
from collections import Counter
from pathlib import Path
import os
import pandas as pd

# Repository root. Set EEG_PROJECT_ROOT to use data and results stored elsewhere.
base_path = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

results_path = base_path / "results"

# channel list, matches preprocessing output exactly
# verified against MNE 1.12.1 pick_channels() on 1 July
ch16_list = [
    "C3",
    "Cz",
    "C4",
    "FC3",
    "FCz",
    "CP3",
    "CPz",
    "CP4",
    "CP5",
    "CP6",
    "P3",
    "P4",
    "C5",
    "C6",
    "P5",
    "P6",
]


def assign_score_groups(kappas_dict, std_multiplier=0.7, chance_threshold=0.1):
    """
    Assigns score-based groups to subjects using std deviation bands.
    Groups derived from full population distribution, not hardcoded.

    kappas_dict: {subject_id: kappa_value} for the full population
    std_multiplier: controls band width, 0.7 chosen by minimising
                    within-group variance while maintaining full
                    5-group coverage on XAI subset
    chance_threshold: kappa below this = negative transfer (default 0.1,
                      matches existing above-chance threshold in thesis)

    Returns:
        group_assignments: {subject_id: group_name}
        boundaries: dict with mean, std, low and high boundaries
    """
    kappa_values = np.array(list(kappas_dict.values()))
    mean_k = kappa_values.mean()
    std_k = kappa_values.std()

    low_boundary = mean_k - (std_multiplier * std_k)
    high_boundary = mean_k + (std_multiplier * std_k)

    group_assignments = {}
    for subj, k in kappas_dict.items():
        if k < chance_threshold:
            group_assignments[subj] = "negative_transfer"
        elif k < low_boundary:
            group_assignments[subj] = "low"
        elif k < mean_k:
            group_assignments[subj] = "mid_low"
        elif k < high_boundary:
            group_assignments[subj] = "mid_high"
        else:
            group_assignments[subj] = "high"

    boundaries = {
        "mean": round(mean_k, 4),
        "std": round(std_k, 4),
        "std_multiplier": std_multiplier,
        "low_boundary": round(low_boundary, 4),
        "high_boundary": round(high_boundary, 4),
        "chance_threshold": chance_threshold,
    }

    return group_assignments, boundaries


def get_grouped_order(xai_subjects, lrp_results):
    """Returns subjects in group order, with group label per subject."""
    ordered = []
    subject_to_group = {}
    for group_name, subjects in xai_subjects.items():
        for s in subjects:
            if s in lrp_results:
                ordered.append(s)
                subject_to_group[s] = group_name
    return ordered, subject_to_group


def get_score_grouped_order(score_groups, lrp_results, xai_subset):
    """
    Returns subjects ordered by score group, top to bottom:
    high -> mid_high -> mid_low -> low -> negative_transfer.
    Only includes subjects present in both lrp_results and xai_subset.
    """
    group_order = ["high", "mid_high", "mid_low", "low", "negative_transfer"]
    ordered = []
    subject_to_group = {}

    for grp in group_order:
        for subj in xai_subset:
            if subj in lrp_results and score_groups.get(subj) == grp:
                ordered.append(subj)
                subject_to_group[subj] = grp

    return ordered, subject_to_group


def classify_failure_groups(
    all_per_class,
    all_kappas,
    spread_threshold=0.4,
    min_acc_threshold=0.2,
    flat_spread_threshold=0.2,
    flat_kappa_threshold=0.3,
):
    """
    Classifies subjects into failure groups based on per-class accuracy.

    Group A (selective collapse): spread > spread_threshold AND
        min class accuracy < min_acc_threshold.
        One class is substantially worse than others.

    Group B (flat near-chance): spread < flat_spread_threshold AND
        overall kappa < flat_kappa_threshold.
        All classes performing similarly poorly.

    Neither: everything else.

    Thresholds derived from full 106-subject distribution:
        spread mean=0.343, std=0.145
        min_acc mean=0.332, std=0.165
    """
    group_a = []
    group_b = []
    neither = []

    for subj in all_per_class:
        pca = all_per_class[subj]
        spread = max(pca.values()) - min(pca.values())
        min_acc = min(pca.values())
        kappa = all_kappas[subj]
        collapsed_class = min(pca, key=pca.get)

        if kappa < 0.1:
            neither.append(subj)

        elif spread > spread_threshold and min_acc < min_acc_threshold:
            group_a.append(
                {
                    "subject": subj,
                    "spread": round(spread, 3),
                    "min_acc": round(min_acc, 3),
                    "collapsed_class": collapsed_class,
                    "kappa": round(kappa, 4),
                }
            )
        elif spread < flat_spread_threshold and kappa < flat_kappa_threshold:
            group_b.append(
                {
                    "subject": subj,
                    "spread": round(spread, 3),
                    "min_acc": round(min_acc, 3),
                    "kappa": round(kappa, 4),
                }
            )
        else:
            neither.append(subj)

    return group_a, group_b, neither


def compute_lrp_deeplift_agreement(
    lrp_attrs, dl_attrs, valid_subjects, criterion="kappa", n_classes=2
):
    """
    Computes per-subject correlation between LRP and DeepLIFT attribution maps.
    Returns correlation per subject per class.
    """
    results = {}

    for subj in valid_subjects:
        if subj not in lrp_attrs or subj not in dl_attrs:
            continue

        results[subj] = {}
        for cls in range(n_classes):
            lrp_map = lrp_attrs[subj][criterion]["mean_by_class"].get(cls)
            dl_map = dl_attrs[subj][criterion]["mean_by_class"].get(cls)

            if lrp_map is None or dl_map is None:
                results[subj][cls] = None
                continue

            r = np.corrcoef(lrp_map.flatten(), dl_map.flatten())[0, 1]
            results[subj][cls] = round(float(r), 4)

    return results


def compute_method_agreement(
    attrs_a,
    attrs_b,
    valid_subjects,
    criterion="kappa",
    n_classes=2,
    method_a="LRP",
    method_b="SHAP",
):
    """
    Computes per-subject correlation between two attribution methods.
    Same structure as compute_lrp_deeplift_agreement.
    """
    results = {}
    for subj in valid_subjects:
        if subj not in attrs_a or subj not in attrs_b:
            continue
        results[subj] = {}
        for cls in range(n_classes):
            map_a = attrs_a[subj][criterion]["mean_by_class"].get(cls)
            map_b = attrs_b[subj][criterion]["mean_by_class"].get(cls)
            if map_a is None or map_b is None:
                results[subj][cls] = None
                continue
            r = np.corrcoef(map_a.flatten(), map_b.flatten())[0, 1]
            results[subj][cls] = round(float(r), 4)
    return results


def compute_within_group_consistency(
    attrs_106,
    group_assignments,
    valid_subjects,
    grouping="category",
    config="binary",
    criterion="kappa",
    n_classes=2,
):
    """
    Computes within-group attribution consistency using pairwise correlation.

    Parameters:
    -----------
    attrs_106: dict, binary_attrs_106 or mc_attrs_106
    group_assignments: dict, category_groups_106 or score_groups
    valid_subjects: list of all valid subject IDs
    grouping: 'category' or 'score', which grouping to use
    config: 'binary' (n_classes=2) or 'multiclass' (n_classes=4)
    criterion: 'kappa' or 'composite'
    n_classes: 2 for binary, 4 for multiclass

    Returns:
    --------
    consistency_results: dict {group_name: {class_idx: mean_correlation}}
    """
    # get unique groups
    groups = sorted(set(group_assignments.values()))
    consistency_results = {}

    for grp in groups:
        grp_subjects = [
            s for s in valid_subjects if group_assignments[s] == grp and s in attrs_106
        ]

        if len(grp_subjects) < 2:
            consistency_results[grp] = None
            continue

        consistency_results[grp] = {}

        for cls in range(n_classes):
            # collect attribution maps for this class across group members
            maps = []
            for subj in grp_subjects:
                mean_attr = attrs_106[subj][criterion]["mean_by_class"].get(cls)
                if mean_attr is not None:
                    maps.append(mean_attr.flatten())

            if len(maps) < 2:
                consistency_results[grp][cls] = None
                continue

            # pairwise correlation across all subject pairs
            maps = np.array(maps)
            n = len(maps)
            correlations = []
            for i in range(n):
                for j in range(i + 1, n):
                    r = np.corrcoef(maps[i], maps[j])[0, 1]
                    correlations.append(r)

            consistency_results[grp][cls] = {
                "mean_r": round(float(np.mean(correlations)), 4),
                "std_r": round(float(np.std(correlations)), 4),
                "n_subjects": n,
                "n_pairs": len(correlations),
            }

    return consistency_results


def print_consistency_results(consistency_results, n_classes=2, class_names=None):
    """Prints within-group consistency results in readable format."""
    if class_names is None:
        class_names = (
            ["left", "right"]
            if n_classes == 2
            else ["left", "right", "both-fists", "both-feet"]
        )

    for grp, grp_results in consistency_results.items():
        if grp_results is None:
            print(f"\n{grp}: insufficient subjects (<2), skipped")
            continue
        print(f"\n{grp}:")
        for cls in range(n_classes):
            r = grp_results.get(cls)
            if r is None:
                print(f"  class {class_names[cls]}: no data")
            else:
                print(
                    f"  class {class_names[cls]}: "
                    f"mean_r={r['mean_r']:.4f} "
                    f"(std={r['std_r']:.4f}, "
                    f"n={r['n_subjects']} subjects, "
                    f"{r['n_pairs']} pairs)"
                )


def build_binary_relevance_table(attrs_106, score_groups, ch16_names, method_name=""):
    """
    Build mean relevance score table for binary task, score grouping.
    Returns a long-format DataFrame with columns:
    [group, class, channel, mean_relevance]
    """
    channel_avg = {}
    for subj in attrs_106:
        channel_avg[subj] = {
            "class_0": np.mean(attrs_106[subj]["kappa"]["mean_by_class"][0], axis=1),
            "class_1": np.mean(attrs_106[subj]["kappa"]["mean_by_class"][1], axis=1),
        }

    rows = []
    for group in ["high", "mid_high", "mid_low", "low", "negative_transfer"]:
        group_subjects = [s for s, g in score_groups.items() if g == group]
        for cls, cls_name in [(0, "left"), (1, "right")]:
            group_attrs = np.array(
                [channel_avg[s][f"class_{cls}"] for s in group_subjects]
            )
            group_mean = np.mean(group_attrs, axis=0)
            for ch_idx, ch_name in enumerate(ch16_names):
                rows.append(
                    {
                        "group": group,
                        "class": cls_name,
                        "channel": ch_name,
                        "mean_relevance": round(group_mean[ch_idx], 6),
                    }
                )

    return pd.DataFrame(rows)


def build_multiclass_relevance_table(
    attrs_106, category_groups, ch16_names, method_name=""
):
    """
    Build mean relevance score table for multiclass task, category grouping.
    Handles empty arrays (e.g. S018 class 2) by replacing with zeros.
    Returns a long-format DataFrame with columns:
    [group, class, channel, mean_relevance]
    """
    class_names = {0: "left", 1: "right", 2: "both_fists", 3: "both_feet"}
    n_classes = 4

    channel_avg = {}
    for subj in attrs_106:
        channel_avg[subj] = {}
        for cls in range(n_classes):
            arr = attrs_106[subj]["kappa"]["mean_by_class"][cls]
            if np.array(arr).ndim == 2:
                channel_avg[subj][cls] = np.mean(arr, axis=1)
            else:
                channel_avg[subj][cls] = np.zeros(len(ch16_names))

    rows = []
    for group in ["neither", "group_a_collapse", "group_b_flat", "negative_transfer"]:
        group_subjects = [s for s, g in category_groups.items() if g == group]
        for cls in range(n_classes):
            group_attrs = np.array([channel_avg[s][cls] for s in group_subjects])
            group_mean = np.mean(group_attrs, axis=0)
            for ch_idx, ch_name in enumerate(ch16_names):
                rows.append(
                    {
                        "group": group,
                        "class": class_names[cls],
                        "channel": ch_name,
                        "mean_relevance": round(group_mean[ch_idx], 6),
                    }
                )

    return pd.DataFrame(rows)


def get_top_bottom_channels(relevance_df, class_names, n=3, group_order=None):
    """
    For each group and class, print top n and bottom n channels
    by mean relevance score.
    """
    if group_order is None:
        group_order = relevance_df["group"].unique().tolist()

    for cls_name in class_names:
        print(f"\nClass: {cls_name}")
        print("-" * 60)
        for group in group_order:
            subset = relevance_df[
                (relevance_df["class"] == cls_name) & (relevance_df["group"] == group)
            ].sort_values("mean_relevance", ascending=False)

            if subset.empty:
                continue

            top_n = subset.head(n)[["channel", "mean_relevance"]].values
            bot_n = subset.tail(n)[["channel", "mean_relevance"]].values

            print(f"\n  {group}:")
            print(f"    Top {n}:    ", end="")
            for ch, val in top_n:
                print(f"{ch}={val:.6f}  ", end="")
            print(f"\n    Bottom {n}: ", end="")
            for ch, val in bot_n:
                print(f"{ch}={val:.6f}  ", end="")
        print()
