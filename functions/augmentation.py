import torch


def patch_perturbation(x):
    """zeros or adds noise to a random time-window patch across some channels, for ALL samples in batch"""
    x_aug = x.clone()
    batch_size, _, n_channels, n_timepoints = x.shape
    for i in range(batch_size):
        win_len = torch.randint(96, n_timepoints + 1, (1,)).item()
        win_start = torch.randint(0, max(1, n_timepoints - win_len + 1), (1,)).item()
        n_perturb = torch.randint(1, n_channels, (1,)).item()
        channels_to_perturb = torch.randperm(n_channels)[:n_perturb]
        if torch.rand(1).item() > 0.5:
            x_aug[i, 0, channels_to_perturb, win_start : win_start + win_len] = 0.0
        else:
            std = x[i, 0, channels_to_perturb, win_start : win_start + win_len].std()
            noise_std = std * (0.01 + torch.rand(1).item() * 1.99)
            noise = (
                torch.randn_like(
                    x[i, 0, channels_to_perturb, win_start : win_start + win_len]
                )
                * noise_std
            )
            x_aug[i, 0, channels_to_perturb, win_start : win_start + win_len] += noise
    return x_aug


def hemisphere_perturbation(x, left_idx=[0, 3, 5], right_idx=[2, 7]):
    """shuffles or noise-replaces one hemisphere's channels, for ALL samples in batch"""
    x_aug = x.clone()
    batch_size = x.shape[0]
    for i in range(batch_size):
        hemi_idx = left_idx if torch.rand(1).item() > 0.5 else right_idx
        if torch.rand(1).item() > 0.5:
            shuffled = torch.tensor(hemi_idx)[torch.randperm(len(hemi_idx))]
            x_aug[i, 0, hemi_idx, :] = x[i, 0, shuffled, :]
        else:
            x_aug[i, 0, hemi_idx, :] = torch.randn_like(x[i, 0, hemi_idx, :])
    return x_aug


def random_shift(x, max_shift=80):
    """shifts trial onset forward by up to max_shift samples, for ALL samples in batch"""
    x_aug = x.clone()
    batch_size, _, n_channels, n_timepoints = x.shape
    for i in range(batch_size):
        shift = torch.randint(0, max_shift + 1, (1,)).item()
        if shift > 0:
            x_aug[i, 0, :, :-shift] = x[i, 0, :, shift:]
            x_aug[i, 0, :, -shift:] = 0.0
    return x_aug


def apply_augmentation(x):
    """per-batch, applies ONE of: patch perturbation, hemisphere perturbation, random shift, or no augmentation"""
    choice = torch.randint(0, 4, (1,)).item()
    if choice == 0:
        return patch_perturbation(x)
    elif choice == 1:
        return hemisphere_perturbation(x)
    elif choice == 2:
        return random_shift(x)
    else:
        return x
