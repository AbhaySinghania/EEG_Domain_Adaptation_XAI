import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.signal import welch
from sklearn.metrics import (
    cohen_kappa_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
import pyriemann
from pyriemann.utils.covariance import covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann, mean_covariance
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm, sqrtm

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_band_power_features(raw_data, fs=160):
    """
    extracts mu and beta band power per channel per trial
    mu band: 8 to 13 Hz
    beta band: 13 to 30 Hz
    """
    n_trials, n_channels, n_timepoints = raw_data.shape

    # output: n_trials x (n_channels * 2 bands)
    features = np.zeros((n_trials, n_channels * 2))

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            signal = raw_data[trial_idx, ch_idx, :]

            # compute power spectral density using Welch method
            freqs, psd = welch(signal, fs=fs, nperseg=min(256, n_timepoints))

            # mu band power: 8 to 13 Hz
            mu_mask = (freqs >= 8) & (freqs <= 13)
            mu_power = np.mean(psd[mu_mask])

            # beta band power: 13 to 30 Hz
            beta_mask = (freqs > 13) & (freqs <= 30)
            beta_power = np.mean(psd[beta_mask])

            # store: first n_channels are mu, next n_channels are beta
            features[trial_idx, ch_idx] = mu_power
            features[trial_idx, n_channels + ch_idx] = beta_power

    return features


def compute_mmd_multiscale(source_features, target_features):
    if torch.is_tensor(source_features):
        source_features = source_features.detach().cpu().numpy()
    if torch.is_tensor(target_features):
        target_features = target_features.detach().cpu().numpy()

    all_features = np.vstack([source_features, target_features])
    pairwise_distances = cdist(all_features, all_features, metric="euclidean")
    median_distance = np.median(pairwise_distances[pairwise_distances > 0])

    bandwidths = [0.5 * median_distance, 1.0 * median_distance, 2.0 * median_distance]

    mmd_total = 0.0

    for sigma in bandwidths:

        def rbf_kernel(x, y, sigma):
            sq_dist = cdist(x, y, metric="sqeuclidean")
            return np.exp(-sq_dist / (2 * sigma**2))

        k_ss = rbf_kernel(source_features, source_features, sigma)
        k_tt = rbf_kernel(target_features, target_features, sigma)
        k_st = rbf_kernel(source_features, target_features, sigma)

        # biased estimator: keep diagonal, always non-negative
        mmd_k = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
        mmd_total += mmd_k

    mmd_total /= len(bandwidths)

    # biased estimator is always non-negative, safe to sqrt
    return float(np.sqrt(mmd_total))


def compute_mmd_optimal_bandwidth(source_features, target_features, n_bandwidths=10):
    if torch.is_tensor(source_features):
        source_features = source_features.detach().cpu().numpy()
    if torch.is_tensor(target_features):
        target_features = target_features.detach().cpu().numpy()

    all_features = np.vstack([source_features, target_features])
    pairwise_distances = cdist(all_features, all_features, metric="euclidean")
    median_distance = np.median(pairwise_distances[pairwise_distances > 0])

    bandwidth_range = np.linspace(0.1, 5.0, n_bandwidths) * median_distance

    best_mmd = -np.inf
    best_bandwidth = None

    for sigma in bandwidth_range:
        sq_dist_ss = cdist(source_features, source_features, metric="sqeuclidean")
        sq_dist_tt = cdist(target_features, target_features, metric="sqeuclidean")
        sq_dist_st = cdist(source_features, target_features, metric="sqeuclidean")

        k_ss = np.exp(-sq_dist_ss / (2 * sigma**2))
        k_tt = np.exp(-sq_dist_tt / (2 * sigma**2))
        k_st = np.exp(-sq_dist_st / (2 * sigma**2))

        # biased estimator: no diagonal zeroing
        mmd_k = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

        if mmd_k > best_mmd:
            best_mmd = mmd_k
            best_bandwidth = sigma

    return float(np.sqrt(max(0, best_mmd))), float(best_bandwidth)


def compute_riemannian_distance(source_raw, target_raw):
    """
    computes Riemannian distance between source and target EEG distributions
    uses covariance matrices on the positive definite matrix manifold
    """
    # compute covariance matrices per trial
    # pyriemann expects (n_trials, n_channels, n_timepoints)
    source_cov = covariances(source_raw, estimator="oas")
    target_cov = covariances(target_raw, estimator="oas")

    # compute Riemannian mean of each set
    source_mean = mean_riemann(source_cov)
    target_mean = mean_riemann(target_cov)

    # compute geodesic distance between the two means
    riemannian_dist = distance_riemann(source_mean, target_mean)

    return float(riemannian_dist)


def apply_riemannian_alignment(source_x, target_x):
    """
    applies Riemannian alignment to reduce distribution shift
    between source and target subjects before training
    aligns covariance matrices to geometric mean of source subjects
    """

    # compute covariance matrices for source trials
    cov_estimator = Covariances(estimator="oas")
    source_covs = cov_estimator.fit_transform(source_x)
    target_covs = cov_estimator.fit_transform(target_x)

    # compute geometric mean of source covariance matrices
    ref_cov = mean_covariance(source_covs, metric="riemann")

    # compute whitening matrix from reference
    ref_invsqrt = invsqrtm(ref_cov)

    # align source trials
    source_aligned = np.array(
        [ref_invsqrt @ cov @ ref_invsqrt.T for cov in source_covs]
    )

    # align target trials using same reference
    target_aligned = np.array(
        [ref_invsqrt @ cov @ ref_invsqrt.T for cov in target_covs]
    )

    # reconstruct signals from aligned covariances
    # use whitening transform on original signals
    ref_invsqrt_signal = invsqrtm(ref_cov)

    source_x_aligned = np.array(
        [(ref_invsqrt_signal @ source_x[i]) for i in range(len(source_x))]
    )

    target_x_aligned = np.array(
        [(ref_invsqrt_signal @ target_x[i]) for i in range(len(target_x))]
    )

    return source_x_aligned, target_x_aligned


def compute_subject_mmd(subject_x, target_x):
    """
    computes MMD between one source subject and target subject
    used for source subject weighting before training
    subject_x: numpy array shape (trials, channels, timepoints)
    target_x: numpy array shape (trials, channels, timepoints)
    returns scalar MMD value
    """
    subject_band = extract_band_power_features(subject_x)
    target_band = extract_band_power_features(target_x)
    mmd = compute_mmd_multiscale(subject_band, target_band)
    return mmd


def compute_pre_adaptation_mmd(source_loader, target_loader):
    """
    computes MMD between source and target before any training
    uses band power features (mu and beta) as low dimensional
    model independent representation of each trial
    16 features per trial: 8 channels x 2 frequency bands
    riemannian distance computed on covariance matrices
    """
    source_raw = []
    target_raw = []

    for source_x, _ in source_loader:
        source_raw.append(source_x.squeeze(1).numpy())

    for target_batch in target_loader:
        target_x = target_batch[0]
        target_raw.append(target_x.squeeze(1).numpy())

    source_raw = np.vstack(source_raw)
    target_raw = np.vstack(target_raw)

    # extract band power features for MMD
    # 16 dimensional: 8 channels x 2 bands (mu and beta)
    source_band = extract_band_power_features(source_raw)
    target_band = extract_band_power_features(target_raw)

    # compute MMD on band power features
    mmd_multiscale = compute_mmd_multiscale(source_band, target_band)
    mmd_optimal, best_bw = compute_mmd_optimal_bandwidth(source_band, target_band)

    # riemannian distance on covariance matrices
    riemannian = compute_riemannian_distance(source_raw, target_raw)

    return mmd_multiscale, mmd_optimal, best_bw, riemannian


def compute_post_adaptation_mmd(model, source_loader, target_loader):
    """
    computes MMD on trained model embeddings after adaptation and measures how well model aligned source and target representations. it uses 128 dimensional EEGNet feature vectors, model dependent, reflects what the model learned, also computes band power MMD for direct comparison with pre-adaptation
    """
    model.eval()
    model.to(device)

    source_embeddings = []
    target_embeddings = []
    source_raw = []
    target_raw = []

    with torch.no_grad():
        for source_x, _ in source_loader:
            source_x = source_x.to(device)
            features = model.feature_extractor(source_x)
            source_embeddings.append(features.cpu().numpy())
            source_raw.append(source_x.squeeze(1).cpu().numpy())

        for target_batch in target_loader:
            target_x = target_batch[0].to(device)
            features = model.feature_extractor(target_x)
            target_embeddings.append(features.cpu().numpy())
            target_raw.append(target_x.squeeze(1).cpu().numpy())

    source_embeddings = np.vstack(source_embeddings)
    target_embeddings = np.vstack(target_embeddings)
    source_raw = np.vstack(source_raw)
    target_raw = np.vstack(target_raw)

    # band power MMD for direct comparison with pre-adaptation
    source_band = extract_band_power_features(source_raw)
    target_band = extract_band_power_features(target_raw)

    post_mmd_bp_ms = compute_mmd_multiscale(source_band, target_band)
    post_mmd_bp_opt, post_bw_bp = compute_mmd_optimal_bandwidth(
        source_band, target_band
    )
    post_riemannian = compute_riemannian_distance(source_raw, target_raw)

    # embedding MMD for understanding model alignment
    post_mmd_emb_ms = compute_mmd_multiscale(source_embeddings, target_embeddings)
    post_mmd_emb_opt, post_bw_emb = compute_mmd_optimal_bandwidth(
        source_embeddings, target_embeddings
    )

    return {
        "post_mmd_bandpower_multiscale": post_mmd_bp_ms,
        "post_mmd_bandpower_optimal": post_mmd_bp_opt,
        "post_mmd_bandpower_optimal_bandwidth": post_bw_bp,
        "post_riemannian": post_riemannian,
        "post_mmd_embedding_multiscale": post_mmd_emb_ms,
        "post_mmd_embedding_optimal": post_mmd_emb_opt,
        "post_mmd_embedding_optimal_bandwidth": post_bw_emb,
    }


def evaluate(model, target_eval_loader):
    model.eval()
    model.to(device)
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y in target_eval_loader:
            x = x.to(device)
            class_out, _, _ = model(x, alpha=0.0)
            probs = torch.softmax(class_out, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    n_classes_detected = all_probs.shape[1]
    if n_classes_detected == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        per_class_accuracy = None
        confusion_mat = None
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        conf_mat = confusion_matrix(
            all_labels, all_preds, labels=sorted(set(all_labels))
        )
        confusion_mat = conf_mat.tolist()
        per_class_accuracy = {
            int(cls): float((all_preds[all_labels == cls] == cls).mean())
            for cls in sorted(set(all_labels))
        }

    return accuracy, kappa, auc, per_class_accuracy, confusion_mat
