import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn  # (may not be used; kept for compatibility)
import torch.nn as nn
import torch.nn.functional as F


# --- Safe loader resolver (supports dict-like and bare DataLoader) ---
def _resolve_loader(loaders, key_hint="train"):
    import collections
    from torch.utils.data import DataLoader
    if isinstance(loaders, DataLoader):
        return loaders
    if isinstance(loaders, collections.abc.Mapping):
        for k in (key_hint, "train_loader", "id_train"):
            if k in loaders:
                return loaders[k]
        return next(iter(loaders.values()))
    raise TypeError("loaders must be dict-like or a DataLoader")


def check(x):
    tx = torch.as_tensor(x)
    nan = torch.sum(torch.isnan(tx))
    inf = torch.sum(torch.isinf(tx))
    if (inf + nan) != 0:
        tx = torch.nan_to_num(tx, nan=0.0, posinf=1e6, neginf=-1e6)
    if isinstance(x, torch.Tensor):
        return tx.to(x.dtype).to(x.device)
    return tx.detach().cpu().numpy()


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]


def centered_cov_torch(x):
    n = x.shape[0]
    if n <= 1:
        return torch.eye(x.shape[1], device=x.device) * 1e-4
    res = x.t().mm(x) / (n - 1)
    res = 0.5 * (res + res.t())
    return res


def get_embeddings(net, loader, num_dim, dtype, device, storage_device):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.long, device=storage_device)

    with torch.no_grad():
        start = 0
        progress_bar = tqdm(loader, desc="Feature Extraction")
        for data, label in progress_bar:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if isinstance(net, nn.DataParallel):
                _ = net.module(data)
                out = getattr(net.module, "feature", None)
            else:
                _ = net(data)
                out = getattr(net, "feature", None)

            if out is None:
                out = _
                if isinstance(out, torch.Tensor):
                    out = out.view(out.size(0), -1)
                else:
                    out = torch.zeros(data.size(0), num_dim, device=device)

            out = F.normalize(out, p=2, dim=1)

            end = start + len(data)
            embeddings[start:end].copy_(out.to(storage_device), non_blocking=True)
            labels[start:end].copy_(label.to(storage_device), non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):
    if isinstance(net, nn.DataParallel):
        _ = net.module(data_B_X)
        features_B_Z = getattr(net.module, "feature", None)
    else:
        _ = net(data_B_X)
        features_B_Z = getattr(net, "feature", None)

    if features_B_Z is None:
        features_B_Z = _
        if isinstance(features_B_Z, torch.Tensor):
            features_B_Z = features_B_Z.view(features_B_Z.size(0), -1)
        else:
            raise RuntimeError("Cannot obtain features from network for GMM forward.")

    features_B_Z = F.normalize(features_B_Z, p=2, dim=1)
    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])
    log_probs_B_Y = torch.clamp(log_probs_B_Y, min=-1e9, max=1e9)
    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.long, device=storage_device)

    with torch.no_grad():
        start = 0
        progress_bar = tqdm(loader, desc="GMM Evaluation")
        for data, label in progress_bar:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logit_B_C = gmm_forward(net, gaussians_model, data)
            logit_B_C = torch.clamp(logit_B_C, min=-1e9, max=1e9)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C.to(storage_device), non_blocking=True)
            labels_N[start:end].copy_(label.to(storage_device), non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    log_probs_B_Y = torch.clamp(log_probs_B_Y, min=-1e9, max=1e9)
    return log_probs_B_Y


def _safe_symmetrize(mat):
    mat = 0.5 * (mat + mat.t())
    eps = 1e-6
    mat = mat + eps * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    return mat


def fit_gda(model, trainloader, num_classes, embedding_dim, device):
    """
    Class-Contrastive Dynamic Density Estimation (version with better stability)
    - Extract features
    - Compute class means/covariances and fit GMM
    """

    # Disable Langevin Dynamics for better stability
    use_langevin = False
    print("Using stable density estimation (Langevin Dynamics disabled)")

    # 1) Extract features
    embeddings, labels = get_embeddings(model.base_model, trainloader, embedding_dim, torch.float, device, device)

    # 2) Normalize features
    try:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e3, neginf=-1e3)
        print("Features normalized successfully")
    except Exception as e:
        print(f"Warning: Feature normalization failed: {e}")

    refined_embeddings = embeddings.clone()

    # 3) Class-wise mean and covariance
    classwise_mean_features = []
    classwise_cov_features = []
    for c in range(num_classes):
        class_mask = labels == c
        class_features = refined_embeddings[class_mask]
        class_count = class_features.shape[0]

        if class_count < 2:
            print(f"Warning: Class {c} has insufficient samples ({class_count}). Using global features.")
            class_features = refined_embeddings
            class_count = class_features.shape[0]

        mean = torch.mean(class_features, dim=0)
        centered = class_features - mean
        cov = centered_cov_torch(centered)
        cov = _safe_symmetrize(cov)
        cov = cov + 1e-4 * torch.eye(cov.shape[0], device=device, dtype=cov.dtype)

        classwise_mean_features.append(mean)
        classwise_cov_features.append(cov)

    classwise_mean_features = torch.stack(classwise_mean_features)
    classwise_cov_features = torch.stack(classwise_cov_features)

    # 4) Fit GMM with incremental jitters
    max_attempts = len(JITTERS)
    gmm = None
    for attempt, jitter_eps in enumerate(JITTERS):
        try:
            jitter_mat = (jitter_eps * torch.eye(classwise_cov_features.shape[1], device=device,
                                                 dtype=classwise_cov_features.dtype)).unsqueeze(0)
            covs = classwise_cov_features + jitter_mat
            covs = 0.5 * (covs + covs.transpose(-1, -2))
            gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features, covariance_matrix=covs)
            test_sample = refined_embeddings[:1]
            _ = gmm.log_prob(test_sample[:, None, :])
            print(f"Class-Contrastive GMM fit succeeded with jitter={jitter_eps}")
            break
        except Exception as e:
            print(f"GMM attempt {attempt + 1}/{max_attempts} failed (jitter={jitter_eps}): {e}")
            if attempt == max_attempts - 1:
                eye = torch.eye(classwise_cov_features.shape[1], device=device, dtype=classwise_cov_features.dtype)
                covs = eye.unsqueeze(0) * 0.1
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features, covariance_matrix=covs)
                print("Fallback diagonal covariance GMM used.")

    # 5) Compute p_z_train and energy range (with robust reporting)
    train_log_probs, _ = gmm_evaluate(model.base_model, gmm, trainloader, device, num_classes, device)
    p_z_train = torch.logsumexp(train_log_probs, dim=-1)

    energy_stats = {}
    with torch.no_grad():
        # 🔥 FIX: Use ENERGY NETWORK as primary (better range ~7 vs GMM ~1)
        energies_network = model.energy_network(embeddings).flatten()
        energies_network = torch.nan_to_num(energies_network, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Also compute GMM-based energy for reference
        gmm_energies = -torch.logsumexp(train_log_probs, dim=1)
        gmm_energies = torch.nan_to_num(gmm_energies, nan=0.0, posinf=1e9, neginf=-1e9)

        # ---- Energy Network stats ----
        E_min_net = float(torch.quantile(energies_network, 0.01))
        E_max_net = float(torch.quantile(energies_network, 0.99))
        E_range_net = E_max_net - E_min_net
        
        # ---- GMM stats ----
        E_min_gmm = float(torch.quantile(gmm_energies, 0.01))
        E_max_gmm = float(torch.quantile(gmm_energies, 0.99))
        E_range_gmm = E_max_gmm - E_min_gmm
        
        print(f"GEM Energy Network Range: E_min={E_min_net:.4f}, E_max={E_max_net:.4f}, Range={E_range_net:.4f}")
        print(f"GEM GMM Energy Range: E_min={E_min_gmm:.4f}, E_max={E_max_gmm:.4f}, Range={E_range_gmm:.4f}")

        # 🔥 CHOOSE: Use whichever has BETTER range (more discriminative)
        if E_range_net >= E_range_gmm:
            e_min_q = E_min_net
            e_max_q = E_max_net
            e_range_q = E_range_net
            primary_energies = energies_network
            note = "energy_network (primary - better range)"
        else:
            e_min_q = E_min_gmm
            e_max_q = E_max_gmm
            e_range_q = E_range_gmm
            primary_energies = gmm_energies
            note = "GMM (primary - better range)"
        
        print(f"GEM Energy Range: E_min={e_min_q:.4f}, E_max={e_max_q:.4f}, Range={e_range_q:.4f} ({note})")

        # Rich stats for use in OOD phase
        en_np = primary_energies.detach().cpu().numpy().astype("float64")
        en_np = en_np[np.isfinite(en_np)]
        if en_np.size > 0:
            p1 = float(np.percentile(en_np, 1.0))
            p99 = float(np.percentile(en_np, 99.0))
            mu = float(np.mean(en_np))
            sd = float(np.std(en_np) + 1e-12)
        else:
            p1, p99, mu, sd = e_min_q, e_max_q, (e_min_q + e_max_q) / 2.0, max(1e-3, e_range_q / 6.0)

        energy_stats = {
            'E_min_raw': float(E_min_net),
            'E_max_raw': float(E_max_net),
            'E_min': float(e_min_q),
            'E_max': float(e_max_q),
            'p1': float(p1),
            'p99': float(p99),
            'mean': float(mu),
            'std': float(sd),
            'source': note
        }

    return gmm, p_z_train, energy_stats


fit_gem_gda = fit_gda


def density_from_logpz(log_pz):
    log_pz = torch.clamp(log_pz, min=-1e9, max=1e9)
    return torch.sigmoid(1.5 * log_pz)
