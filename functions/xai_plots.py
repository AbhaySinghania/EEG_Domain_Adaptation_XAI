import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
from xai_utils import ch16_list

# Repository root. Set EEG_PROJECT_ROOT to use data and results stored elsewhere.
base_path = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

results_path = base_path / "results"


def plot_group_mean_attribution(
    binary_attrs_106,
    category_groups_106,
    score_groups,
    valid_subjects,
    all_kappas,
    sfreq=160,
    save=True,
    title_tag="LRP binary CDAN-16ch",
):
    """
    Group-level mean attribution for all 106 subjects.
    Two figures: category groups and score groups.
    Rows = groups, columns = classes (left, right for binary).
    Each cell = mean attribution averaged across all subjects in that group.
    """
    # derive n_classes and class_names from actual data
    n_classes = len(next(iter(binary_attrs_106.values()))["kappa"]["mean_by_class"])
    class_names = (
        ["left", "right"]
        if n_classes == 2
        else ["left", "right", "both-fists", "both-feet"]
    )

    # category group order and labels
    cat_group_order = [
        "negative_transfer",
        "group_a_collapse",
        "group_b_flat",
        "neither",
    ]
    cat_group_labels = {
        "negative_transfer": f'negative transfer (n={sum(1 for s in valid_subjects if category_groups_106[s]=="negative_transfer")})',
        "group_a_collapse": f'group A collapse (n={sum(1 for s in valid_subjects if category_groups_106[s]=="group_a_collapse")})',
        "group_b_flat": f'group B flat (n={sum(1 for s in valid_subjects if category_groups_106[s]=="group_b_flat")})',
        "neither": f'neither (n={sum(1 for s in valid_subjects if category_groups_106[s]=="neither")})',
    }

    # score group order
    score_group_order = ["high", "mid_high", "mid_low", "low", "negative_transfer"]
    score_group_labels = {
        g: f"{g} (n={sum(1 for s in valid_subjects if score_groups[s]==g)})"
        for g in score_group_order
    }

    timepoints = 480
    tick_positions = list(range(0, timepoints, 80))
    tick_labels = [f"{t/sfreq:.1f}" for t in tick_positions]

    for grouping_name, group_order, group_labels, group_dict in [
        ("category", cat_group_order, cat_group_labels, category_groups_106),
        ("score", score_group_order, score_group_labels, score_groups),
    ]:
        n_groups = len(group_order)
        fig, axes = plt.subplots(
            n_groups, n_classes, figsize=(7 * n_classes, 3.5 * n_groups)
        )

        # compute global vmax across all groups and classes
        all_means = []
        group_mean_maps = {}
        for grp in group_order:
            group_mean_maps[grp] = {}
            for cls in range(n_classes):
                members = [
                    s
                    for s in valid_subjects
                    if group_dict[s] == grp
                    and binary_attrs_106[s]["kappa"]["mean_by_class"][cls] is not None
                ]
                if members:
                    stacked = np.array(
                        [
                            binary_attrs_106[s]["kappa"]["mean_by_class"][cls]
                            for s in members
                        ]
                    )
                    group_mean_maps[grp][cls] = stacked.mean(axis=0)
                    all_means.append(np.abs(group_mean_maps[grp][cls]).max())
                else:
                    group_mean_maps[grp][cls] = None

        global_vmax = max(all_means) if all_means else 0.04

        for row_idx, grp in enumerate(group_order):
            for col_idx, cls in enumerate(range(n_classes)):
                ax = axes[row_idx, col_idx]
                mean_attr = group_mean_maps[grp][cls]

                if mean_attr is None:
                    ax.set_facecolor("#f5f5f5")
                    ax.text(
                        0.5,
                        0.5,
                        "no subjects",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                sns.heatmap(
                    mean_attr,
                    ax=ax,
                    cmap="RdBu_r",
                    vmin=-global_vmax,
                    vmax=global_vmax,
                    xticklabels=False,
                    yticklabels=ch16_list if col_idx == 0 else False,
                    cbar=col_idx == n_classes - 1,
                    cbar_kws={"label": "relevance", "shrink": 0.8},
                )
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
                ax.set_xlabel("time (s)", fontsize=8)

                if col_idx == 0:
                    ax.set_ylabel(group_labels[grp], fontsize=8)
                    ax.tick_params(axis="y", labelsize=7)

                if row_idx == 0:
                    ax.set_title(f"class: {class_names[cls]}", fontsize=9)

        plt.suptitle(
            f"{title_tag} — group-level mean attribution ({grouping_name} grouping)",
            fontsize=12,
            y=1.001,
        )
        plt.tight_layout()

        tag_clean = title_tag.replace(" ", "_").lower()
        if save:
            save_path = (
                results_path / "figures" / f"{tag_clean}_group_mean_{grouping_name}.png"
            )
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


def plot_stopping_criterion_consistency(
    kappa_attrs, composite_attrs, subjects, n_classes=2, title_tag="LRP", save=True
):
    """
    Compares kappa vs composite stopping attribution for same subjects.
    One row per subject, 4 columns:
    left-kappa | left-composite | right-kappa | right-composite
    Shared colormap per subject (not global) so pattern is visible
    regardless of magnitude differences between criteria.
    # Input format note:
    # LRP notebook format: kappa_attrs[subj] is a tuple (attrs, labels, mean_by_class)
    #   pass lrp_results as kappa_attrs and lrp_results_composite as composite_attrs
    #   example: plot_stopping_criterion_consistency(lrp_results, lrp_results_composite, ...)
    #
    # Saved file format (LRP/DeepLIFT/SHAP from .npz):
    #   kappa_attrs[subj] is a dict with keys 'kappa' and 'composite'
    #   both criteria live in the same dict, so pass the same dict twice
    #   example: plot_stopping_criterion_consistency(binary_shap_106, binary_shap_106, ...)
    #
    # The isinstance check inside the loop handles both formats automatically.
    """
    n_subjects = len(subjects)

    timepoints = 480
    tick_positions = list(range(0, timepoints, 80))
    tick_labels = [f"{t/160:.1f}" for t in tick_positions]

    n_cols = n_classes * 2  # kappa + composite per class
    fig, axes = plt.subplots(
        n_subjects, n_cols, figsize=(5.5 * n_cols, 3.5 * n_subjects)
    )
    if n_subjects == 1:
        axes = axes.reshape(1, -1)

    # col_titles built dynamically from panels in the loop

    for row_idx, subj in enumerate(subjects):
        # handle both notebook tuple format for LRP (attrs, labels, mean_by_class)
        # and local attribution file format {criterion: {mean_by_class: ...}}
        if isinstance(kappa_attrs[subj], tuple):
            _, _, mean_kappa = kappa_attrs[subj]
            _, _, mean_comp = composite_attrs[subj]
        else:
            mean_kappa = kappa_attrs[subj]["kappa"]["mean_by_class"]
            mean_comp = kappa_attrs[subj]["composite"]["mean_by_class"]

        # per-subject vmax across both criteria and both classes
        # so kappa and composite are on same scale for fair comparison
        subj_vmax = max(
            np.abs(mean_kappa[cls]).max()
            for cls in range(n_classes)
            if mean_kappa[cls] is not None
        )
        comp_vmax = max(
            np.abs(mean_comp[cls]).max()
            for cls in range(n_classes)
            if mean_comp[cls] is not None
        )
        vmax = max(subj_vmax, comp_vmax)

        # build panels: kappa then composite for each class
        class_names = (
            ["left", "right"]
            if n_classes == 2
            else ["left", "right", "both-fists", "both-feet"]
        )
        panels = []
        for cls in range(n_classes):
            panels.append((f"{class_names[cls]} | kappa", mean_kappa[cls]))
            panels.append((f"{class_names[cls]} | composite", mean_comp[cls]))

        for col_idx, (panel_title, mean_attr) in enumerate(panels):
            ax = axes[row_idx, col_idx]

            if mean_attr is None:
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5,
                    0.5,
                    "no trials",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="grey",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            show_yticks = col_idx == 0
            sns.heatmap(
                mean_attr,
                ax=ax,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                xticklabels=False,
                yticklabels=ch16_list if show_yticks else False,
                cbar=col_idx == 3,
                cbar_kws={"label": "relevance", "shrink": 0.8},
            )
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
            ax.set_xlabel("time (s)", fontsize=8)

            if show_yticks:
                ax.set_ylabel(subj, fontsize=9)
                ax.tick_params(axis="y", labelsize=7)

            if row_idx == 0:
                ax.set_title(panel_title, fontsize=9)

    plt.suptitle(
        f"{title_tag} stopping-criterion consistency — kappa vs composite",
        fontsize=12,
        y=1.001,
    )
    plt.tight_layout()

    if save:
        tag_clean = title_tag.replace(" ", "_").lower()
        save_path = results_path / "figures" / f"{tag_clean}_stopping_consistency.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
