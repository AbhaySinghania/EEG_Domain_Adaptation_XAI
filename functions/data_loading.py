import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
from metrics import compute_subject_mmd, apply_riemannian_alignment

# base path, single source of truth for all data folders
# Repository root. Set EEG_PROJECT_ROOT to use data and results stored elsewhere.
base_path = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
processed_path = base_path / "Data/processed"
multiclass_path = base_path / "Data/processed_multiclass"
ch16_path = base_path / "Data/processed_16ch"
multiclass_16ch_path = base_path / "Data/processed_multiclass_16ch"


def load_subject_data(subject_id):
    """
    loads preprocessed data for one subjrct from npz file anf returns data as tensor with correct shape for EEGNet shape: (trains,1,channels,timepoints)
    """
    file_path = processed_path / f"{subject_id}_processed.npz"
    data = np.load(file_path)

    x = data["data"]
    y = data["labels"]

    # converting the labels from 2,3 to 0,1 for cross entropy loss, where T1 = 2 become 0 (left fist), T2 = 3 become 1 (right fist)
    y = (y == 3).astype(np.int64)

    # current channel dim for EEGNet is ( trails, channels, timepoints), becomes (trails, 1, channels, timepoints)
    x = x[:, np.newaxis, :, :]

    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.LongTensor(y)
    y_tensor = y_tensor.long()

    return x_tensor, y_tensor


def load_subject_data_windowed(
    subject_id, window_start_sec=0.5, window_end_sec=2.5, sfreq=160
):
    """
    loads existing npz data and slices to a narrower time window
    e.g. 0.5-2.5s instead of full 0-3s
    """
    x, y = load_subject_data(subject_id)
    start_idx = int(window_start_sec * sfreq)
    end_idx = int(window_end_sec * sfreq)
    x_windowed = x[:, :, :, start_idx:end_idx]
    return x_windowed, y


def load_subject_data_multiclass(subject_id):
    """
    loads pre-saved 4-class npz from Data/processed_multiclass/
    labels: 0=left fist, 1=right fist, 2=both fists, 3=both feet
    """

    npz_path = multiclass_path / f"{subject_id}_multiclass.npz"
    data = np.load(npz_path)
    x = torch.FloatTensor(data["data"]).unsqueeze(1)
    y = torch.LongTensor(data["labels"])
    return x, y


def load_subject_data_16ch(subject_id):
    """
    loads pre-saved 16-channel npz from Data/processed_16ch/
    same 2-class left/right task, just 16 channels instead of 8
    """
    npz_path = ch16_path / f"{subject_id}_16ch.npz"
    data = np.load(npz_path)
    x = torch.FloatTensor(data["data"]).unsqueeze(1)
    y = torch.LongTensor(data["labels"])
    return x, y


def load_subject_data_multiclass_16ch(subject_id):
    """
    loads pre-saved 4-class, 16-channel npz from
    Data/processed_multiclass_16ch/
    labels already correctly 0/1/2/3 from preprocess_subject_multiclass
    """
    # mc16ch_path = Path("Data/processed_multiclass_16ch")
    npz_path = multiclass_16ch_path / f"{subject_id}_multiclass_16ch.npz"
    data = np.load(npz_path)
    x = torch.FloatTensor(data["data"]).unsqueeze(1)
    y = torch.LongTensor(data["labels"])
    return x, y


def load_subject_data_flexible(
    subject_id, use_multiclass=False, use_16ch=False, use_windowed_data=False
):
    """
    single flexible loader, builds correct data path/file from
    independent flags rather than separate hardcoded branches
    """
    if use_multiclass and use_16ch:
        folder = "Data/processed_multiclass_16ch"
        suffix = "_multiclass_16ch"
    elif use_multiclass:
        folder = "Data/processed_multiclass"
        suffix = "_multiclass"
    elif use_16ch:
        folder = "Data/processed_16ch"
        suffix = "_16ch"
    else:
        folder = "Data/processed"
        suffix = "_processed"

    npz_path = base_path / folder / f"{subject_id}{suffix}.npz"
    data = np.load(npz_path)
    x = torch.FloatTensor(data["data"]).unsqueeze(1)
    y = torch.LongTensor(data["labels"])

    if not use_multiclass and not use_16ch:
        y = torch.where(y == 2, torch.tensor(0), torch.tensor(1))

    if use_windowed_data:
        x = x[:, :, :, int(0.5 * 160) : int(2.5 * 160)]

    return x, y


def prepare_dataloaders(
    test_subject_id,
    valid_subjects,
    batch_size=32,
    use_source_weighting=True,
    use_riemannian_alignment=False,
    use_multiclass=False,
    use_windowed_data=False,
    use_16ch=False,
):
    """
    prepares source and target dataloaders for one LOSO fold
    use_source_weighting: if True, weights source subjects by
    similarity to target using MMD, closer subjects sampled more
    """
    x_target, y_target = load_subject_data_flexible(
        test_subject_id,
        use_multiclass=use_multiclass,
        use_16ch=use_16ch,
        use_windowed_data=use_windowed_data,
    )

    # load each source subject separately for weighting
    x_source_list = []
    y_source_list = []
    subject_weights = []

    target_np = x_target.squeeze(1).numpy()

    for subject_id in valid_subjects:
        if subject_id == test_subject_id:
            continue
        x_s, y_s = load_subject_data_flexible(
            subject_id,
            use_multiclass=use_multiclass,
            use_16ch=use_16ch,
            use_windowed_data=use_windowed_data,
        )

        x_source_list.append(x_s)
        y_source_list.append(y_s)

        if use_source_weighting:
            source_np = x_s.squeeze(1).numpy()
            mmd = compute_subject_mmd(source_np, target_np)
            # inverse MMD as weight: closer subjects get higher weight
            # add small epsilon to avoid division by zero
            weight = 1.0 / (mmd + 1e-6)
            subject_weights.append(weight)
        else:
            subject_weights.append(1.0)

    x_source = torch.cat(x_source_list, dim=0)
    y_source = torch.cat(y_source_list, dim=0)

    # apply riemannian alignment if requested
    if use_riemannian_alignment:
        # convert to numpy for alignment
        source_np = x_source.squeeze(1).numpy()
        target_np = x_target.squeeze(1).numpy()

        # apply alignment
        source_aligned, target_aligned = apply_riemannian_alignment(
            source_np, target_np
        )

        # convert back to tensor with channel dimension
        x_source = torch.FloatTensor(source_aligned).unsqueeze(1)
        x_target = torch.FloatTensor(target_aligned).unsqueeze(1)

    # create per-trial weights by repeating subject weight
    # for all trials belonging to that subject
    if use_source_weighting:
        trial_weights = []
        for i, x_s in enumerate(x_source_list):
            n_trials = x_s.shape[0]
            trial_weights.extend([subject_weights[i]] * n_trials)

        trial_weights = torch.FloatTensor(trial_weights)
        trial_weights = trial_weights / trial_weights.sum()

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=trial_weights, num_samples=len(trial_weights), replacement=True
        )

        source_dataset = TensorDataset(x_source, y_source)
        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
        )
    else:
        source_dataset = TensorDataset(x_source, y_source)
        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    target_dataset = TensorDataset(x_target)
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    target_eval_dataset = TensorDataset(x_target, y_target)
    target_eval_loader = DataLoader(
        target_eval_dataset, batch_size=batch_size, shuffle=False
    )

    return source_loader, target_loader, target_eval_loader, y_target
