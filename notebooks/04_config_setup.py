# ### Config and setup


# #### Function refresh


# %load_ext autoreload
# %autoreload 2
# # %reload_ext autoreload

import sys

sys.path.insert(0, "../functions")

# #### Library imports


import torch
import torch.nn as nn
import numpy as np
import json
import random
import os
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

import time

import seaborn as sns
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

torch.manual_seed(25)
np.random.seed(25)
random.seed(25)

# #### Functions import


from models import DANN
from data_loading import prepare_dataloaders, load_subject_data_flexible
from metrics import evaluate

from explainability import (
    DANNClassifierWrapper,
    load_model_for_xai,
    fold_batchnorm,
    verify_folding,
    lrp_epsilon_conv2d,
    lrp_zplus_conv2d,
    lrp_linear,
    lrp_avgpool,
    compute_lrp,
    run_lrp_full,
    load_lrp_attributions,
    compute_per_class_accuracy_106,
    compute_deeplift,
    run_deeplift_full,
    load_deeplift_attributions,
    compute_shap,
    run_shap_full,
    load_shap_attributions,
    compute_ig,
    run_ig_full,
    load_ig_attributions,
)

from xai_utils import (
    assign_score_groups,
    get_grouped_order,
    get_score_grouped_order,
    classify_failure_groups,
    compute_lrp_deeplift_agreement,
    compute_method_agreement,
    compute_within_group_consistency,
    print_consistency_results,
    ch16_list,
)

from xai_plots import plot_group_mean_attribution, plot_stopping_criterion_consistency

# #### Path

base_path = Path("..")
processed_path = base_path / "Data/processed"
results_path = base_path / "results"
ch16_path = base_path / "Data/processed_16ch"

# existing attribution folders for loading LRP and DeepLIFT
lrp_attr_path = results_path / "lrp_attributions"
binary_attr_path = lrp_attr_path / "binary_cdan_16ch"
multiclass_attr_path = lrp_attr_path / "multiclass_16ch"

deeplift_attr_path = results_path / "deeplift_attributions"
binary_deeplift_path = deeplift_attr_path / "binary_cdan_16ch"
multiclass_deeplift_path = deeplift_attr_path / "multiclass_16ch"

per_class_path = results_path / "per_class_accuracy"

shap_attr_path = results_path / "shap_attributions"
binary_shap_path = shap_attr_path / "binary_cdan_16ch"
multiclass_shap_path = shap_attr_path / "multiclass_16ch"

# new IG folders
ig_attr_path = results_path / "ig_attributions"
binary_ig_zero_path = ig_attr_path / "binary_cdan_16ch_zero"
binary_ig_mean_path = ig_attr_path / "binary_cdan_16ch_mean"
multiclass_ig_zero_path = ig_attr_path / "multiclass_16ch_zero"
multiclass_ig_mean_path = ig_attr_path / "multiclass_16ch_mean"

# only creating zero baseline folders since mean is dropped
binary_ig_zero_path.mkdir(parents=True, exist_ok=True)
multiclass_ig_zero_path.mkdir(parents=True, exist_ok=True)

# misclassified data path
binary_misc_path = results_path / "misclassified_attributions" / "binary_cdan_16ch"
multiclass_misc_path = results_path / "misclassified_attributions" / "multiclass_16ch"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# #### 16 channel importing from xai_utils


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

# #### Valid Subjects


# 106 subjects after removing S088, S092, S100 (incompatible sampling rate)
excluded = ["S088", "S092", "S100"]

valid_subjects = []
for i in range(1, 110):
    subject_id = f"S{str(i).zfill(3)}"
    subject_file = processed_path / f"{subject_id}_processed.npz"
    if subject_file.exists() and subject_id not in excluded:
        valid_subjects.append(subject_id)

print(f"Valid subjects: {len(valid_subjects)}")
print(f"First five: {valid_subjects[:5]}")
print(f"Last five: {valid_subjects[-5:]}")

# #### Setup cell for groupings rebuild


consistency_subjects = [
    "S007",
    "S085",
    "S108",
    "S107",
    "S080",
    "S097",
    "S037",
    "S006",
    "S039",
    "S018",
]

# load kappas
with open(results_path / "loso_results_cdan_16ch_RETRAIN.json") as f:
    cdan_results = json.load(f)

all_kappas = {
    subj: cdan_results[subj]["kappa_stopping"]["final_kappa"] for subj in valid_subjects
}

# load per-class accuracy
with open(per_class_path / "multiclass_16ch_per_class_accuracy.json") as f:
    per_class_data = json.load(f)

all_per_class = {}
for subj in valid_subjects:
    pca = per_class_data[subj]["kappa"]["per_class_accuracy"]
    all_per_class[subj] = {int(cls): float(acc) for cls, acc in pca.items()}

# rebuild groupings
score_groups, boundaries = assign_score_groups(all_kappas, std_multiplier=0.7)

all_kappas_binary = {
    subj: cdan_results[subj]["kappa_stopping"]["final_kappa"] for subj in valid_subjects
}
group_a, group_b, neither = classify_failure_groups(all_per_class, all_kappas_binary)
group_a_subjects = [s["subject"] for s in group_a]
group_b_subjects = [s["subject"] for s in group_b]

category_groups_106 = {}
for subj in valid_subjects:
    if all_kappas[subj] < 0.1:
        category_groups_106[subj] = "negative_transfer"
    elif subj in group_a_subjects:
        category_groups_106[subj] = "group_a_collapse"
    elif subj in group_b_subjects:
        category_groups_106[subj] = "group_b_flat"
    else:
        category_groups_106[subj] = "neither"

xai_subjects = {
    "top": ["S007", "S085"],
    "bottom_kappa": ["S018", "S097", "S107"],
    "group_a_collapse": ["S037", "S006"],
    "group_b_flat": ["S039", "S020"],
    "typical": ["S080", "S065", "S030", "S108"],
}
xai_subset = [s for group in xai_subjects.values() for s in group]

# load existing attributions for comparison
binary_attrs_106 = load_lrp_attributions(binary_attr_path, valid_subjects, n_classes=2)

mc_attrs_106 = load_lrp_attributions(multiclass_attr_path, valid_subjects, n_classes=4)

binary_dl_106 = load_deeplift_attributions(
    binary_deeplift_path, valid_subjects, n_classes=2
)

mc_dl_106 = load_deeplift_attributions(
    multiclass_deeplift_path, valid_subjects, n_classes=4
)

binary_shap_106 = load_shap_attributions(binary_shap_path, valid_subjects, n_classes=2)

mc_shap_106 = load_shap_attributions(multiclass_shap_path, valid_subjects, n_classes=4)

binary_ig_106 = load_ig_attributions(binary_ig_zero_path, valid_subjects, n_classes=2)

mc_ig_106 = load_ig_attributions(multiclass_ig_zero_path, valid_subjects, n_classes=4)

print("Setup complete")
print(f"Score groups: {dict(Counter(score_groups.values()))}")
print(f"Category groups: {dict(Counter(category_groups_106.values()))}")
print(f"XAI subset: {len(xai_subset)} subjects")
print(
    f"LRP binary: {len(binary_attrs_106)}, DeepLIFT: {len(binary_dl_106)}, "
    f"SHAP: {len(binary_shap_106)}, IG: {len(binary_ig_106)}"
)

# ### End of config and setup
