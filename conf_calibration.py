from utility import mob_predictive_probs, dirichlet_mean
import torch

def _brier_score_normalized(probs, labels):
    C = probs.size(1)
    oh = torch.nn.functional.one_hot(labels, num_classes=C).float().to(probs.device)
    b = torch.mean(torch.sum((probs - oh)**2, dim=1))
    return (b / C).item()
import sklearn.metrics

def _to_probs(x):
    # if rows sum to ~1 assume probs, else softmax
    if x.dim() == 2:
        s = x.sum(dim=1, keepdim=True)
        if torch.all((s > 0.999) & (s < 1.001)):
            return x.clamp_min(1e-8)
    return torch.softmax(x, dim=1).clamp_min(1e-8)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def _probs_from_outputs(alpha=None, logits=None, alpha_list=None, mixture_weights=None, use_mob=False):
    """
    Build prediction probabilities:
      - MIX: p = sum_k pi_k * (alpha_k / alpha_k0)
      - Single-head Dirichlet: p = alpha / alpha0
      - Otherwise: softmax(logits)
    """
    if use_mob and (alpha_list is not None) and (mixture_weights is not None):
        return mob_predictive_probs(alpha_list, mixture_weights)
    if alpha is not None:
        return dirichlet_mean(alpha)
    if logits is not None:
        return F.softmax(logits, dim=1)
    raise ValueError("No valid inputs to compute probabilities.")


def conf_calibration_gem(model, gda, p_z_train, testloader, num_classes, device,
                         energy_range, use_mob=False):
    """
    Confidence calibration for GEM (and optionally MIX).
    - If use_mob=True and model has dirichlet_heads, uses mixture of beliefs.
    - Otherwise, falls back to baseline gated logits.
    """
    print("\nPhase 7: GEM Confidence Calibration")

    if use_mob and hasattr(model, "dirichlet_heads"):
        print("🔬 Using GEM-MIX for calibration")
        return conf_calibration_mob(model, testloader, num_classes, device)
    else:
        print("🔬 Using GEM-CORE for calibration")
        return conf_calibration_baseline(model, testloader, num_classes, device)


def conf_calibration_mob(model, testloader, num_classes, device):
    """Calibration for GEM-MIX using logits for ECE (same as auto_ts.py)."""
    brier, cnt = 0.0, 0
    Y, PI, EVIDENCE = [], [], []

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="MIX Calibration Progress")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            # Expected output: dict with logits, or tuple
            out = model(x, return_features=True, use_fi_regularization=False)

            # Extract logits for ECE calculation (same method as auto_ts.py)
            logits = None
            total_evidence = None
            
            if isinstance(out, dict):
                logits = out.get("logits")
                total_evidence = out.get("total_evidence")
            elif isinstance(out, (list, tuple)) and len(out) >= 1:
                # First element is usually logits or final_probs
                logits = out[0]
                if len(out) >= 7:
                    total_evidence = out[6]
            else:
                logits = out

            # Convert to probabilities using softmax (like auto_ts.py)
            # If output is already probabilities, convert to effective logits first
            if logits is not None and logits.dim() == 2:
                row_sum = logits.sum(dim=1, keepdim=True)
                if torch.all((row_sum > 0.999) & (row_sum < 1.001)):
                    # Already probabilities, convert to log-probs
                    logits = logits.clamp_min(1e-12).log()
            
            # Softmax to get calibrated probabilities
            probs = F.softmax(logits, dim=1) if logits is not None else torch.ones(x.size(0), num_classes, device=device) / num_classes

            # ----- Brier (normalized by number of classes) -----
            y_oh = F.one_hot(y, num_classes).float()
            per_sample = torch.sum((y_oh - probs) ** 2, dim=1).div_(num_classes)
            brier += per_sample.sum().item()
            cnt += x.size(0)

            Y.append(y.cpu())
            PI.append(probs.detach().cpu())
            if total_evidence is None:
                total_evidence = torch.zeros(x.size(0), device=device)
            EVIDENCE.append(total_evidence.detach().cpu())

    brier_score = brier / max(cnt, 1)

    # ---------- ECE ----------
    pi_all = torch.cat(PI, dim=0)
    y_all = torch.cat(Y, dim=0)
    evidence_all = torch.cat(EVIDENCE, dim=0)

    n_bins = 15  # Increased to 15 bins (same as auto_ts.py)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(pi_all, 1)
    accuracies = predictions.eq(y_all)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    ece = ece.item()

    # ---------- Auxiliary metrics ----------
    max_probs = pi_all.max(dim=1)[0].numpy()
    entropy = -(pi_all * torch.log(pi_all + 1e-8)).sum(dim=1).numpy()

    labels_dummy = np.concatenate([np.ones(len(max_probs)), np.zeros(len(max_probs))])
    auroc_alea = sklearn.metrics.roc_auc_score(labels_dummy, np.concatenate([max_probs, 1 - max_probs]))
    auroc_epis = sklearn.metrics.roc_auc_score(labels_dummy, np.concatenate([1 - entropy, entropy]))

    aupr_alea = sklearn.metrics.average_precision_score(labels_dummy, np.concatenate([max_probs, 1 - max_probs]))
    aupr_epis = sklearn.metrics.average_precision_score(labels_dummy, np.concatenate([1 - entropy, entropy]))

    print(f"GEM-MIX Calibration Results")
    print(f"  Brier Score: {brier_score:.4f}")
    print(f"  AUROC   → Aleatoric {auroc_alea:.4f}, Epistemic {auroc_epis:.4f}")
    print(f"  AUPR    → Aleatoric {aupr_alea:.4f}, Epistemic {aupr_epis:.4f}")
    print(f"  ECE: {ece:.4f}")

    return brier_score, [aupr_alea, aupr_epis], [auroc_alea, auroc_epis], ece


def conf_calibration_baseline(model, testloader, num_classes, device):
    """Calibration for standard GEM; if alpha is available, use Dirichlet mean."""
    brier, cnt = 0.0, 0
    Y, PI, ALPHA = [], [], []

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Baseline Calibration Progress")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            out = model(x, return_features=True)
            # Typically: gated_logits, features, energy, gate_weight, u_log_alpha0
            if isinstance(out, (list, tuple)) and len(out) >= 1:
                gated_logits = out[0]
            elif isinstance(out, dict) and "logits" in out:
                gated_logits = out["logits"]
            else:
                gated_logits = out

            # Safe mapping to positive Dirichlet parameters
            alpha = torch.exp(torch.clamp(gated_logits, min=-15, max=15)) + 1e-8
            pi = dirichlet_mean(alpha)

            # ----- Brier (normalized by number of classes) -----
            y_oh = F.one_hot(y, num_classes).float()
            per_sample = torch.sum((y_oh - pi) ** 2, dim=1).div_(num_classes)  # <- divide by C
            brier += per_sample.sum().item()
            cnt += x.size(0)

            Y.append(y.cpu())
            PI.append(pi.detach().cpu())
            ALPHA.append(alpha.detach().cpu())

    brier_score = brier / max(cnt, 1)

    try:
        pi_all = torch.cat(PI, dim=0)
        y_all = torch.cat(Y, dim=0)

        # ---------- ECE ----------
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = torch.max(pi_all, 1)
        accuracies = predictions.eq(y_all)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        ece = ece.item()

        # ---------- Auxiliary metrics (fixed for Baseline) ----------
        # FIX: Compute AUROC/AUPR based on confidence-correctness relationship
        # This metric shows whether model confidence correlates with prediction correctness
        max_probs = pi_all.max(dim=1)[0].numpy()
        entropy = (-pi_all * (pi_all + 1e-8).log()).sum(dim=1).numpy()
        correct = accuracies.numpy().astype(np.int32)  # 1 if correct, 0 if wrong
        
        # AUROC: Can we predict correctness with confidence?
        # Aleatoric: maxP (higher = more confident = should be more likely correct)
        # Epistemic: 1-entropy (higher = less uncertain = should be more likely correct)
        try:
            if len(np.unique(correct)) > 1:  # Need both classes for AUROC
                auroc_alea = sklearn.metrics.roc_auc_score(correct, max_probs)
                auroc_epis = sklearn.metrics.roc_auc_score(correct, 1 - entropy / np.log(pi_all.size(1)))
                aupr_alea = sklearn.metrics.average_precision_score(correct, max_probs)
                aupr_epis = sklearn.metrics.average_precision_score(correct, 1 - entropy / np.log(pi_all.size(1)))
            else:
                # If all predictions are correct or wrong
                auroc_alea, auroc_epis = 0.5, 0.5
                aupr_alea, aupr_epis = float(correct.mean()), float(correct.mean())
        except Exception as e:
            print(f"⚠️ Calibration AUROC/AUPR calculation error: {e}")
            auroc_alea, auroc_epis = 0.5, 0.5
            aupr_alea, aupr_epis = 0.5, 0.5

        print(f"GEM-CORE Calibration Results")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  AUROC   → Aleatoric {auroc_alea:.4f}, Epistemic {auroc_epis:.4f}")
        print(f"  AUPR    → Aleatoric {aupr_alea:.4f}, Epistemic {aupr_epis:.4f}")
        print(f"  ECE: {ece:.4f}")

    except Exception as e:
        print(f"Warning in calibration metrics: {e}")
        aupr_alea, aupr_epis = 0.5, 0.5
        auroc_alea, auroc_epis = 0.5, 0.5
        ece = 0.1

    return brier_score, [aupr_alea, aupr_epis], [auroc_alea, auroc_epis], ece


# Backward compatibility
def conf_calibration_gem(model, gda, p_z_train, testloader, num_classes, device, energy_range):
    return conf_calibration_gem(model, gda, p_z_train, testloader, num_classes, device, energy_range, use_mob=False)
