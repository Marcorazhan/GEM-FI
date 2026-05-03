# ====================================================
# train.py - VERSION ENHANCED FOR 97-98% CIFAR-10 PERFORMANCE
# ====================================================

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div
from tqdm import tqdm
import json
import time


# ====================================================
# Reproducibility helper
# ====================================================
def set_global_seed(seed: int = 42):
    print(f"[Enhanced GEM] Setting global seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for better performance


# ====================================================
# Enhanced Metrics
# ====================================================
def calculate_expected_brier_score(logits, labels, num_classes):
    try:
        probabilities = F.softmax(logits, dim=1)
        probabilities = torch.clamp(probabilities, min=1e-8, max=1.0 - 1e-8)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        true_labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        brier_score = torch.mean(torch.sum((probabilities - true_labels_one_hot) ** 2, dim=1))
        return brier_score.item()
    except Exception as e:
        print(f"Error in Expected Brier score calculation: {e}")
        return 1.0


def calculate_model_brier_score(model, testloader, num_classes, device):
    """Compute Brier score using Dirichlet mean (matches GEM original)"""
    model.eval()
    all_probabilities, all_labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Handle different output types
            if isinstance(out, dict):
                logits = out.get("logits", out.get("alpha"))
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            # Convert logits to Dirichlet alpha, then to probabilities (matches GEM)
            alpha = torch.exp(logits.clamp(-15, 15)) + 1e-8
            a0 = alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            probabilities = alpha / a0
            all_probabilities.append(probabilities.cpu())
            all_labels.append(y.cpu())
    probabilities = torch.cat(all_probabilities, dim=0)
    labels = torch.cat(all_labels, dim=0)
    true_labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    # Normalized by num_classes for consistency
    brier_score = torch.mean(torch.sum((probabilities - true_labels_one_hot) ** 2, dim=1) / num_classes)
    return brier_score.item()


# ====================================================
# Enhanced Utils
# ====================================================
def detect_dataset_type(trainloader):
    if hasattr(trainloader.dataset, 'dataset'):
        dataset_str = str(trainloader.dataset.dataset)
    else:
        dataset_str = str(trainloader.dataset)
    if 'MNIST' in dataset_str:
        return "MNIST"
    elif 'CIFAR' in dataset_str:
        return "CIFAR-10"  # CIFAR-100 is only used as OOD, not ID
    else:
        return "UNKNOWN"


def validate_enhanced_gem_model(model, validloader, num_classes, device):
    """
    Fixed validation function - without temperature manipulation
    """
    model.eval()
    total, correct = 0, 0
    val_loss, val_batches = 0.0, 0

    with torch.no_grad():
        for x, y in validloader:
            x, y = x.to(device), y.to(device)
            try:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_loss += loss.item()
                val_batches += 1
            except Exception:
                continue

    model.train()
    if total > 0:
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        return accuracy, avg_val_loss
    else:
        return 0.0, 0.0


def eval_enhanced_gem(model, testloader, device):
    """Improved evaluation function - fixing temperature scaling issue"""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Enhanced GEM Evaluation")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            try:
                logits = model(x)

                # Important fix: Only use temperature scaling if actually trained
                if hasattr(model, 'temperature') and hasattr(model.temperature, 'requires_grad'):
                    if model.temperature.requires_grad:  # Only if trained
                        T = model.temperature.detach().clamp_min(1e-6)
                        if abs(T.item() - 1.0) > 1e-3:  # Meaningful difference from 1.0
                            logits = logits / T

                _, predicted = torch.max(logits.data, 1)
                batch_total = y.size(0)
                batch_correct = (predicted == y).sum().item()
                total += batch_total
                correct += batch_correct
                progress_bar.set_postfix({
                    'Batch_Acc': f'{100 * batch_correct / batch_total:.1f}%',
                    'Overall_Acc': f'{100 * correct / total:.1f}%'
                })
            except Exception:
                continue
    final_accuracy = 100 * correct / total if total > 0 else 0.0
    return final_accuracy


# ====================================================
# Enhanced SAFE helpers for energy tensors
# ====================================================
def _ensure_tensor_energy(energy, device):
    """
    Ensure energy is a Tensor on the correct device.
    Accepts float / int / scalar-tensor / 1D/2D tensors / tuple(list).
    Returns a 1D tensor (batch) or scalar tensor when appropriate.
    """
    if isinstance(energy, (tuple, list)):
        energy = energy[0]
    if not torch.is_tensor(energy):
        energy = torch.tensor(energy, dtype=torch.float32, device=device)
    else:
        energy = energy.to(device)
    return energy


# ====================================================
# Enhanced MIX Loss Functions
# ====================================================
def compute_enhanced_mob_loss(final_probs, mixture_weights, component_alphas, labels, fi_traces, fi_lambda, reg_param, no_fi_loss=False):
    """Compute improved loss functions for MIX+FI
    
    Args:
        no_fi_loss: If True, disables L_FI loss term (for ablation). FI traces are still computed for modulation.
    """
    num_components = mixture_weights.size(1)

    # 1) Predictive Loss
    log_final = torch.log(final_probs.clamp_min(1e-8))
    L_pred = F.nll_loss(log_final, labels)

    # 2) KL Regularization
    L_KL = 0.0
    for k in range(num_components):
        alpha_k = torch.clamp(component_alphas[k], min=1e-3)
        dirichlet_k = Dirichlet(alpha_k)
        prior = Dirichlet(torch.ones_like(alpha_k))
        kl_diverg = torch.distributions.kl.kl_divergence(dirichlet_k, prior)
        L_KL += (mixture_weights[:, k] * kl_diverg).mean()

    # 3) FI Regularization (can be disabled for ablation)
    L_FI_reg = torch.tensor(0.0, device=final_probs.device)
    if fi_traces is not None and not no_fi_loss:
        L_FI_reg = (mixture_weights * fi_traces).sum(dim=1).mean()

    # 4) Combine
    total_loss = L_pred + reg_param * L_KL + fi_lambda * L_FI_reg

    # beta * E[tr(FI)] - also conditionally disabled
    beta = 0.02  # Increased weight
    L_FI_expected = 0.0
    if fi_traces is not None and not no_fi_loss:
        L_FI_expected = fi_traces.mean()
    total_loss = total_loss + beta * L_FI_expected

    return total_loss, L_pred, L_KL, L_FI_reg



def compute_fi_traces_baseline(logits, labels):
    """Approximate Fisher Information trace for baseline GEM model.

    Args:
        logits: Tensor of shape [B, C], pre-softmax class logits.
        labels: Tensor of shape [B], ground-truth class indices.

    Returns:
        Tensor of shape [B] with per-sample FI trace values.
    """
    batch_size = logits.size(0)
    device = logits.device

    log_probs = F.log_softmax(logits, dim=1)
    fi_traces = torch.zeros(batch_size, device=device)
    with torch.enable_grad():
        for i in range(batch_size):
            log_prob_i = log_probs[i, labels[i]]
            grad_first = torch.autograd.grad(log_prob_i, logits, retain_graph=True, create_graph=False)[0]
            if grad_first is not None:
                fi_traces[i] = (grad_first[i] ** 2).sum()
    return fi_traces


def compute_enhanced_baseline_loss(gated_logits, labels, reg_param):
    """Compute improved loss for regular GEM"""
    alpha = torch.exp(torch.clamp(gated_logits, min=-10.0, max=10.0)) + 1e-8
    alpha0 = alpha.sum(dim=1, keepdim=True)
    y_oh = F.one_hot(labels, alpha.shape[1]).float().to(gated_logits.device)

    term1 = torch.sum((y_oh - alpha / alpha0) ** 2, dim=1).mean()
    term2 = torch.sum(alpha * (alpha0 - alpha) / (alpha0 ** 2 * (alpha0 + 1)), dim=1).mean()
    loss_edl = term1 + term2

    alpha_tilde = y_oh + (1 - y_oh) * alpha
    dirichlet_posterior = Dirichlet(torch.clamp(alpha_tilde, min=1e-6))
    dirichlet_prior = Dirichlet(torch.ones_like(alpha_tilde))
    kl_regularizer = kl_div(dirichlet_posterior, dirichlet_prior).mean()

    total_loss = loss_edl + reg_param * kl_regularizer
    return total_loss, loss_edl, kl_regularizer


# ====================================================
# ENHANCED Training loop for 97-98% CIFAR-10 PERFORMANCE
# ====================================================
def train_enhanced_gem(model, learning_rate, reg_param, num_epochs, trainloader, validloader,
                        num_classes, device, ood_loader1=None, ood_loader2=None,
                        use_scheduler=True, seed=None, resume=False,
                        use_mob=False, num_components=3, use_fi_regularization=True, fi_lambda=0.5,
                        output_dir=None, gmm_model=None, use_amp=False,
                        # ---- Checkpoint Selection ----
                        ckpt_metric: str = "ood_aupr",
                        ckpt_eval_freq: int = 1,
                        # ---- VOS (Virtual Outlier Synthesis) as EBM negatives ----
                        # Defaults aligned with +run_cifar10_mob_fi_k3.py
                        use_vos: bool = False,
                        vos_ratio: float = 0.3,
                        vos_start_epoch: int = 30,
                        vos_ramp_epochs: int = 30,
                        vos_lambda_neg: float = 0.4,
                        vos_margin_start: float = 0.5,
                        vos_margin: float = 3.0,
                        vos_mix_beta: float = 0.3,
                        vos_pgd_frac: float = 0.5,
                        vos_pgd_eps: float = 12/255,
                        vos_pgd_step: float = 3/255,
                        vos_pgd_steps: int = 5,
                        vos_pgd_random_init: bool = True,
                        vos_mem_size: int = 2048,
                        vos_mem_use_frac: float = 0.15,
                        vos_mem_add_topk: int = 32,
                        # ---- Ablation: FI Loss ----
                        no_fi_loss: bool = False):
    if seed is not None:
        set_global_seed(seed)

    # Initialize AMP scaler
    # AMP Scaler
    scaler = None
    if use_amp and str(device) == 'cuda':
        try:
            # Modern PyTorch (suppress FutureWarning)
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            # Legacy PyTorch
            scaler = torch.cuda.amp.GradScaler()
    if use_amp:
        print("⚡ Automatic Mixed Precision (AMP) Enabled")


    # ====================================================
    # VOS (Virtual Outlier Synthesis) as EBM negatives
    # FIX Issue 10: Class Queues + GMM (matching original VOS)
    # ====================================================
    # Design:
    #   - Maintain per-class feature queues
    #   - Fit Gaussian per class and sample from low-density regions
    #   - Feed VOS samples as negative samples to the EBM term
    
    vos_mem = None  # stored on CPU as float32 tensor [N,C,H,W]
    vos_mem_ptr = 0

    # FIX Issue 10: Class Queues for feature space sampling
    vos_class_queue_size = 500  # Number of features stored per class
    vos_class_queues = {}  # dict: class_id -> tensor [queue_size, feature_dim]
    vos_class_queue_ptr = {}  # dict: class_id -> int (pointer)
    vos_queue_initialized = False  # Are queues filled?
    vos_sample_from = 1000  # Number of samples from distribution
    vos_select_topk = 50  # Number of low-density samples selected

    def _update_class_queues(features, labels, num_classes_local):
        """Update class queues with new features"""
        nonlocal vos_class_queues, vos_class_queue_ptr, vos_queue_initialized
        
        feature_dim = features.size(1)
        
        with torch.no_grad():
            for c in range(num_classes_local):
                mask = (labels == c)
                if not mask.any():
                    continue
                    
                class_features = features[mask].detach().cpu()
                
                # Initialize queue if needed
                if c not in vos_class_queues:
                    vos_class_queues[c] = torch.zeros(vos_class_queue_size, feature_dim)
                    vos_class_queue_ptr[c] = 0
                
                # Add features to queue
                for feat in class_features:
                    ptr = vos_class_queue_ptr[c] % vos_class_queue_size
                    vos_class_queues[c][ptr] = feat
                    vos_class_queue_ptr[c] += 1
            
            # Check if queues are full
            if len(vos_class_queues) >= num_classes_local:
                min_ptr = min(vos_class_queue_ptr.values())
                if min_ptr >= vos_class_queue_size:
                    vos_queue_initialized = True
    
    def _sample_vos_from_gmm(num_samples, device, num_classes_local):
        """FIX Issue 10: Sampling from low-density regions (matching original VOS)

        Optimization: All calculations done on GPU
        """
        if not vos_queue_initialized or len(vos_class_queues) < num_classes_local:
            return None
            
        all_virtual_features = []
        
        for c in range(num_classes_local):
            if c not in vos_class_queues:
                continue

            # FIX: Transfer queue to GPU for faster calculations
            queue = vos_class_queues[c].to(device)

            # Calculate mean and covariance on GPU
            mean = queue.mean(dim=0)
            centered = queue - mean
            covariance = torch.mm(centered.t(), centered) / (vos_class_queue_size - 1)
            # Add small diagonal for numerical stability
            covariance = covariance + 0.0001 * torch.eye(covariance.size(0), device=device)
            
            try:
                # Create Gaussian distribution on GPU
                dist = torch.distributions.MultivariateNormal(mean, covariance)

                # Sampling on GPU
                samples = dist.rsample((vos_sample_from,))  # [sample_from, feature_dim]

                # Calculate log probability (density) on GPU
                log_probs = dist.log_prob(samples)  # [sample_from]

                # Select low-density samples (lowest log_prob = lowest density)
                _, low_density_idx = torch.topk(-log_probs, k=min(vos_select_topk, vos_sample_from))
                virtual_features = samples[low_density_idx]  # [select_topk, feature_dim]
                
                all_virtual_features.append(virtual_features)

            except Exception as e:
                # If covariance is singular, skip
                continue
        
        if len(all_virtual_features) == 0:
            return None

        # Concatenate all virtual features (all on GPU)
        virtual_features = torch.cat(all_virtual_features, dim=0)  # [num_classes * select_topk, feature_dim]
        
        # Randomly select num_samples
        indices = torch.randperm(virtual_features.size(0), device=device)[:num_samples]
        selected = virtual_features[indices]  # Already on GPU
        
        return selected

    def _vos_ramp(epoch_idx: int):
        if (not use_vos) or (epoch_idx < int(vos_start_epoch)):
            return 0.0, 0.0
        t = (epoch_idx - int(vos_start_epoch)) / float(max(1, int(vos_ramp_epochs)))
        t = 1.0 if t > 1.0 else (0.0 if t < 0.0 else t)
        lam = float(vos_lambda_neg) * t
        m = float(vos_margin_start) + (float(vos_margin) - float(vos_margin_start)) * t
        return lam, m

    def _boundary_mix(xb, yb):
        """FIX Issue 3: Generating samples more OOD-like instead of ID-like

        Improvements:
        1. Adding Gaussian noise
        2. Using lambda closer to 0.5 for more boundary samples
        3. Cutout/CutMix style augmentation
        """
        B = xb.size(0)
        perm = torch.randperm(B, device=xb.device)

        # Try to mismatch labels
        try:
            for _ in range(3):  # More attempts for mismatch
                same = (yb == yb[perm])
                if same.any():
                    perm[same] = perm[same].roll(1)
        except Exception:
            pass
        x2 = xb[perm]

        # FIX: Using lambda closer to 0.5 for more Out-of-Manifold samples
        a = max(0.1, float(vos_mix_beta))  # Increased min from 1e-3 to 0.1
        lam = torch.distributions.Beta(a, a).sample((B,)).to(xb.device)
        # Force lambda to be closer to 0.5 (more OOD-like)
        lam = 0.3 + 0.4 * lam  # lambda in [0.3, 0.7] instead of [0, 1]

        lam = lam.view(B, 1, 1, 1)
        xm = lam * xb + (1.0 - lam) * x2

        # FIX: Adding Gaussian noise for more OOD-like
        noise_std = 0.05  # Mild noise
        noise = torch.randn_like(xm) * noise_std
        xm = xm + noise

        # FIX: Random Erasing (like Cutout) for more unnatural samples
        if torch.rand(()) < 0.3:  # 30% chance
            h, w = xm.shape[2], xm.shape[3]
            eh, ew = h // 4, w // 4  # 1/4 size window
            eh_start = torch.randint(0, h - eh, (1,)).item()
            ew_start = torch.randint(0, w - ew, (1,)).item()
            xm[:, :, eh_start:eh_start+eh, ew_start:ew_start+ew] = torch.rand(1).item()
        
        return xm.clamp(0.0, 1.0)

    def _extract_energy(model_, x_in, y_in=None):
        """FIX Issue 4: Consistent energy extraction

        Previously 3 different methods that could return different values.
        Now only get_energy() is used for consistency.
        """
        # FIX: Only use get_energy() (consistent and reliable)
        if hasattr(model_, "get_energy") and callable(getattr(model_, "get_energy")):
            e = model_.get_energy(x_in)
            if torch.is_tensor(e):
                return e.squeeze() if e.dim() > 1 else e

        # Fallback: If no get_energy, use energy_network directly
        if hasattr(model_, 'get_features_consistent') and hasattr(model_, 'energy_network'):
            feats = model_.get_features_consistent(x_in)
            e = model_.energy_network(feats)
            return e.squeeze()

        # Removed inconsistent fallbacks
        raise RuntimeError("Model must have get_energy() method for VOS-EBM.")

    def _pgd_max_energy(model_, x0, eps, step, steps, random_init=True):
        """Energy-guided PGD to maximize energy: produce hard negatives near the decision boundary.

        FIX: Use model.eval() during PGD to prevent BN/Dropout noise
        """
        # FIX: Switch to eval mode to stabilize BN/Dropout during PGD
        was_training = model_.training
        model_.eval()
        
        x_adv = x0.detach().clone()
        if random_init:
            x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)
        
        for _ in range(int(steps)):
            x_adv.requires_grad_(True)
            
            # 🔧 FIX: Energy extraction for gradient only (no graph retention needed)
            E = _extract_energy(model_, x_adv)
            loss = (-E).mean()  # maximize E (want high energy for OOD-like samples)
            
            grad_tuple = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False, allow_unused=True)
            grad = grad_tuple[0] if grad_tuple[0] is not None else torch.zeros_like(x_adv)
            
            # Update with sign gradient
            x_adv = x_adv.detach() + step * grad.sign()
            # Project to eps-ball around x0
            x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
            x_adv = x_adv.clamp(0.0, 1.0)
        
        # 🔧 FIX: Restore training mode
        if was_training:
            model_.train()
        
        return x_adv.detach()

    def _vos_negative_energy_loss(model_, xb, yb, epoch_idx, features_id=None):
        """Return (L_neg * lam_neg, E_neg_mean) and update mem-bank.

        FIX Issue 10: Using Class Queues + GMM
          - First update class queues
          - If queues are full, use GMM sampling
          - Otherwise use Mixup + PGD
        """
        nonlocal vos_mem, vos_mem_ptr, vos_queue_initialized
        lam_neg, margin_base = _vos_ramp(epoch_idx)
        if lam_neg <= 0.0:
            return torch.tensor(0.0, device=xb.device), 0.0

        B = xb.size(0)
        n_vos = max(1, int(float(vos_ratio) * B))

        # FIX Issue 10: Update class queues with new features
        if features_id is not None:
            _update_class_queues(features_id, yb, num_classes)

        # FIX Issue 10: Try GMM sampling first
        virtual_features = _sample_vos_from_gmm(n_vos, xb.device, num_classes)
        
        if virtual_features is not None:
            # GMM-based: Direct energy from virtual features
            E_neg = model_.energy_network(virtual_features).squeeze()
        else:
            # Fallback to Mixup (for early epochs before queues are full)
            x_vos = _boundary_mix(xb[:n_vos].detach(), yb[:n_vos])

            # PGD for harder negatives
            hard_n = int(float(vos_pgd_frac) * n_vos)
            if hard_n > 0 and int(vos_pgd_steps) > 0:
                effective_eps = max(float(vos_pgd_eps), 12/255)
                effective_steps = max(int(vos_pgd_steps), 5)
                x_h = _pgd_max_energy(
                    model_, x_vos[:hard_n],
                    eps=effective_eps, 
                    step=effective_eps / 4,
                    steps=effective_steps, 
                    random_init=bool(vos_pgd_random_init)
                )
                x_vos = torch.cat([x_h, x_vos[hard_n:]], dim=0)

            # Mix in mem-bank negatives
            if (vos_mem is not None) and (vos_mem.size(0) > 0):
                k = int(float(vos_mem_use_frac) * n_vos)
                if k > 0:
                    idx = torch.randint(0, vos_mem.size(0), (k,))
                    x_mem = vos_mem[idx].to(xb.device, non_blocking=True)
                    x_vos[:k] = x_mem
            
            E_neg = _extract_energy(model_, x_vos)

        with torch.no_grad():
            E_id = _extract_energy(model_, xb[:min(32, B)])
            E_id_mean = E_id.mean().item()
            E_neg_mean_val = E_neg.mean().item()
            # Margin = midpoint between ID and VOS energy + buffer
            adaptive_margin = E_id_mean + abs(E_neg_mean_val - E_id_mean) * 0.3
            margin = max(margin_base, adaptive_margin)
        
        # Clamp energy for numerical stability
        E_neg = torch.clamp(E_neg, min=-50.0, max=50.0)

        # Loss: penalize if VOS energy is below margin (want high energy for OOD)
        L_neg_raw = F.softplus(margin - E_neg).mean()

        # Note: Memory bank update only happens for Mixup path (x_vos exists)
        # GMM path doesn't update mem bank since it operates in feature space

        return (lam_neg * L_neg_raw), float(E_neg.mean().detach().item())

    dataset_name = detect_dataset_type(trainloader)

    # FIX Issue 8: Using input parameters instead of hardcode
    # Input parameters (learning_rate, reg_param) are preserved
    # Only internal parameters that don't come from outside are set
    
    if dataset_name == "CIFAR-10":
        # Only internal parameters that aren't arguments
        ebm_weight = 0.01
        clip_value = 1.0
        # FIX Issue 9: Reduce L_UNC interference with VOS
        # When VOS is active, betas should be lower to avoid interference
        if use_vos:
            beta_id = 0.002   # Reduced from 0.005 (VOS active)
            beta_ood = 0.001  # Reduced from 0.002 (VOS active)
        else:
            beta_id = 0.005
            beta_ood = 0.002
    else:  # MNIST
        ebm_weight = 0.01
        clip_value = 1.0
        if use_vos:
            beta_id = 0.0005  # Reduced (VOS active)
            beta_ood = 0.02
        else:
            beta_id = 0.001
            beta_ood = 0.05

    # Using AdamW with optimized settings for CIFAR-10
    if dataset_name == "CIFAR-10":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,  # Reduced weight decay
            betas=(0.9, 0.999)
        )

        # Using Cosine Annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=1e-5  # Increased T_0 and reduced eta_min
        ) if use_scheduler else None
    else:
        # For MNIST use AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(trainloader),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        ) if use_scheduler else None

    model.to(device)

    # --- EMA init ---
    ema_params = [p.clone().detach() for p in model.parameters()]

    # Set GMM model for density scaling
    if use_mob and hasattr(model, 'set_gmm_model') and gmm_model is not None:
        model.set_gmm_model(gmm_model)
        print("✅ GMM model set for density scaling")

    # Save path
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
    else:
        model_type = "Enhanced_GEM_MoB" if use_mob else "Enhanced_GEM"
        checkpoint_dir = os.path.join("./saved_results",
                                      f"{model_type}_{dataset_name}_seed{seed}",
                                      "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
    temp_ckpt = os.path.join(checkpoint_dir, "temp_checkpoint.pt")
    training_info_file = os.path.join(checkpoint_dir, "training_info.json")

    start_epoch = 0
    best_val_acc = 0.0
    best_ood_aupr = 0.0  # For ckpt_metric=ood_aupr (average)
    best_far_ood_aupr = 0.0   # Best Far-OOD AUPR (SVHN) - display only
    best_near_ood_aupr = 0.0  # Best Near-OOD AUPR (CIFAR-100) - display only

    # Quick OOD evaluation function for checkpoint selection
    def _quick_ood_eval(model, id_loader, ood_loader, device, n_samples=500):
        """Quick OOD evaluation with MaxP for checkpoint selection"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        model.eval()
        id_scores, ood_scores = [], []
        with torch.no_grad():
            # ID samples
            count = 0
            for x, _ in id_loader:
                if count >= n_samples:
                    break
                x = x.to(device)
                out = model(x)
                probs = out[0] if isinstance(out, tuple) else out
                if hasattr(probs, 'sum') and probs.dim() == 2:
                    row_sum = probs.sum(dim=1, keepdim=True)
                    if torch.all((row_sum > 0.99) & (row_sum < 1.01)):
                        maxp = probs.max(dim=1)[0]
                    else:
                        maxp = torch.softmax(probs, dim=1).max(dim=1)[0]
                else:
                    maxp = torch.softmax(probs, dim=1).max(dim=1)[0]
                id_scores.extend(maxp.cpu().numpy().tolist())
                count += len(x)
            # OOD samples
            count = 0
            for x, _ in ood_loader:
                if count >= n_samples:
                    break
                x = x.to(device)
                out = model(x)
                probs = out[0] if isinstance(out, tuple) else out
                if hasattr(probs, 'sum') and probs.dim() == 2:
                    row_sum = probs.sum(dim=1, keepdim=True)
                    if torch.all((row_sum > 0.99) & (row_sum < 1.01)):
                        maxp = probs.max(dim=1)[0]
                    else:
                        maxp = torch.softmax(probs, dim=1).max(dim=1)[0]
                else:
                    maxp = torch.softmax(probs, dim=1).max(dim=1)[0]
                ood_scores.extend(maxp.cpu().numpy().tolist())
                count += len(x)
        # Compute metrics (ID=1, OOD=0)
        import numpy as np
        labels = [1]*len(id_scores) + [0]*len(ood_scores)
        scores = id_scores + ood_scores
        if len(set(labels)) < 2:
            return 0.5, 0.5
        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        model.train()
        return auroc, aupr

    # Save training info
    def save_training_info(current_epoch, best_accuracy, is_complete=False):
        # Determine backbone based on dataset
        backbone = "ResNet18" if dataset_name in ("CIFAR-10", "CIFAR-100") else "ConvNet3C3F"
        training_info = {
            'total_epochs': num_epochs,
            'completed_epochs': current_epoch + 1,
            'current_epoch': current_epoch,
            'best_val_acc': best_accuracy,
            'training_complete': is_complete or (current_epoch >= num_epochs - 1),
            'dataset': dataset_name,
            'backbone': backbone,  # Added for auto_ts compatibility
            'learning_rate': learning_rate,
            'reg_param': reg_param,
            'use_mob': use_mob,
            'num_components': num_components if use_mob else 1,
            'use_fi_regularization': use_fi_regularization,
            'fi_lambda': fi_lambda,
            'completion_time': time.time(),
            'seed': seed
        }
        try:
            with open(training_info_file, 'w') as f:
                json.dump(training_info, f, indent=4)
        except Exception as e:
            print(f"❌ Error saving training info: {e}")

    # Load training info
    def load_training_info():
        if os.path.exists(training_info_file):
            try:
                with open(training_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading training info: {e}")
        return None

    # Improved checkpoint management and EMA parameters
    if resume and os.path.exists(last_ckpt):
        try:
            print(f"🔁 Resuming from checkpoint: {last_ckpt}")
            checkpoint = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load EMA parameters
            if 'ema_params' in checkpoint:
                ema_params = [p.to(device) for p in checkpoint['ema_params']]
            else:
                # If not present, create from current model
                ema_params = [p.clone().detach() for p in model.parameters()]

            if scheduler and checkpoint.get('scheduler_state'):
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            training_info = load_training_info()
            if training_info:
                print(f"📊 Training progress: {training_info.get('completed_epochs', 0)}/{num_epochs} epochs")
                print(f"🏆 Best validation accuracy: {training_info.get('best_val_acc', 0):.2f}%")
            print(f"Resumed successfully at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint, restarting from scratch. ({e})")
            if os.path.exists(last_ckpt):
                try:
                    os.remove(last_ckpt)
                    print("🗑️ Removed corrupted checkpoint file.")
                except Exception:
                    pass
    elif resume:
        print("⚠️ Resume requested but no checkpoint found. Starting from scratch.")

    def save_checkpoint(state, filename):
        import shutil
        import time
        max_retries = 5

        # Save EMA parameters in checkpoint
        try:
            state['ema_params'] = [p.cpu() for p in ema_params]
        except Exception:
            pass

        # Try atomic save with retry
        max_retries = 10
        for attempt in range(max_retries):
            try:
                torch.save(state, temp_ckpt)

                # If previous file exists, try to remove first (with retry)
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except OSError:
                        pass

                # Move temp file to final
                shutil.move(temp_ckpt, filename)
                return True

            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(3.0)  # Wait for file to be released
                    continue
                else:
                    print(f"❌ Error saving checkpoint (WinError 32 fix failed): {e}")
                    
            except Exception as e:
                print(f"❌ Unexpected error saving checkpoint: {e}")
                break
                 
        # Cleanup temp if failed
        if os.path.exists(temp_ckpt):
            try:
                os.remove(temp_ckpt)
            except:
                pass
        return False

    def load_checkpoint_safe(filename):
        try:
            if os.path.exists(filename):
                checkpoint = torch.load(filename, map_location=device, weights_only=False)
                return checkpoint
            return None
        except Exception as e:
            print(f"❌ Error loading checkpoint {filename}: {e}")
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"🗑️ Removed corrupted file: {filename}")
                except Exception:
                    pass
            return None

    save_training_info(start_epoch - 1 if start_epoch > 0 else 0, best_val_acc)

    if use_mob:
        if use_fi_regularization:
            print(f"🎯 Training Enhanced GEM-FI with {num_components} components (λ={fi_lambda})")
        else:
            print(f"🎯 Training Enhanced GEM-MIX with {num_components} components (no FI)")
    else:
        print("🎯 Training Enhanced GEM Baseline")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_total_loss = running_pred_loss = running_kl_loss = running_fi_loss = 0.0
            running_ebm_loss = running_unc_loss = 0.0
            running_accuracy = 0.0
            num_batches = 0

            ood_iterators = []
            if ood_loader1 is not None:
                ood_iterators.append(iter(ood_loader1))
            if ood_loader2 is not None:
                ood_iterators.append(iter(ood_loader2))

            progress_bar = tqdm(
                trainloader,
                desc=f'Enhanced GEM{"-MIX" if use_mob else ""}{"+FI" if use_fi_regularization else ""} {dataset_name} Epoch {epoch + 1}/{num_epochs}'
            )
            for batch_idx, (x, y) in enumerate(progress_bar):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                _Eneg_mean = 0.0  # Initialize for logging

                try:
                    if use_mob and hasattr(model, 'dirichlet_heads'):
                        # ---------------- Enhanced MIX forward ----------------
                        (final_probs, features, energy, gate_weights, mixture_weights, component_alphas,
                         fi_traces, alpha0_effective) = model(
                            x, return_features=True,
                            use_fi_regularization=use_fi_regularization,
                            full_output=True
                        )

                        # core MIX loss
                        total_loss, L_pred, L_KL, L_FI_reg = compute_enhanced_mob_loss(
                            final_probs, mixture_weights, component_alphas, y,
                            fi_traces, fi_lambda, reg_param, no_fi_loss=no_fi_loss
                        )

                        # accuracy
                        _, predicted = torch.max(final_probs, 1)
                        batch_accuracy = (predicted == y).float().mean().item()

                        # ---- Enhanced EBM loss (ID + VOS negatives) ----
                        energy = _ensure_tensor_energy(energy, device)
                        energy = torch.clamp(energy, min=-50.0, max=50.0)
                        L_EBM_id = F.softplus(energy).mean()

                        # VOS negatives feed into EBM (passing features for class queues)
                        L_EBM_neg, _Eneg_mean = _vos_negative_energy_loss(model, x, y, epoch, features_id=features)
                        L_EBM = L_EBM_id + L_EBM_neg

                        # ---- ENHANCED Uncertainty loss ----
                        entropy_id = -(final_probs * torch.log(final_probs.clamp_min(1e-8))).sum(dim=1).mean()
                        L_UNC = beta_id * entropy_id

                        for it in ood_iterators:
                            try:
                                x_ood, _ = next(it)
                                x_ood = x_ood.to(device)
                                probs_ood = model(x_ood)
                                entropy_ood = -(probs_ood * torch.log(probs_ood.clamp_min(1e-8))).sum(dim=1).mean()
                                L_UNC = L_UNC - beta_ood * entropy_ood
                            except StopIteration:
                                continue

                        total_loss = total_loss + ebm_weight * L_EBM + L_UNC

                    else:
                        # ---------------- Enhanced Baseline forward ----------------
                        gated_logits, features, energy, gate_weights, u_log_alpha0, alpha0 = model(
                            x, return_features=True
                        )

                        # baseline loss
                        total_loss, L_pred, L_KL = compute_enhanced_baseline_loss(
                            gated_logits, y, reg_param
                        )

                        # FI regularization for baseline (GEM+FI)
                        L_FI_reg = torch.tensor(0.0, device=device)
                        if use_fi_regularization:
                            fi_traces = compute_fi_traces_baseline(gated_logits, y)
                            L_FI_reg = fi_traces.mean()
                            total_loss = total_loss + fi_lambda * L_FI_reg

                        # accuracy
                        _, predicted = torch.max(gated_logits, 1)
                        batch_accuracy = (predicted == y).float().mean().item()

                        # ---- Enhanced EBM loss (ID + VOS negatives) ----
                        energy = _ensure_tensor_energy(energy, device)
                        energy = torch.clamp(energy, min=-50.0, max=50.0)
                        L_EBM_id = F.softplus(energy).mean()

                        # VOS negatives feed into EBM (passing features for class queues)
                        L_EBM_neg, _Eneg_mean = _vos_negative_energy_loss(model, x, y, epoch, features_id=features)
                        L_EBM = L_EBM_id + L_EBM_neg

                        # ---- ENHANCED Uncertainty loss ----
                        p_id = F.softmax(gated_logits, dim=1)
                        entropy_id = -(p_id * torch.log(p_id.clamp_min(1e-8))).sum(dim=1).mean()
                        L_UNC = beta_id * entropy_id

                        for it in ood_iterators:
                            try:
                                x_ood, _ = next(it)
                                x_ood = x_ood.to(device)
                                logits_ood = model(x_ood)
                                p_ood = F.softmax(logits_ood, dim=1)
                                entropy_ood = -(p_ood * torch.log(p_ood.clamp_min(1e-8))).sum(dim=1).mean()
                                L_UNC = L_UNC - beta_ood * entropy_ood
                            except StopIteration:
                                continue

                        total_loss = total_loss + ebm_weight * L_EBM + L_UNC

                    # --- backward with AMP support
                    if use_amp and scaler is not None:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        total_loss.backward()

                    # Improved gradient clipping - use only one method
                    if dataset_name == "CIFAR-10":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=1.0,
                            norm_type=2.0,
                            error_if_nonfinite=True
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    if scheduler:
                        try:
                            # If Scheduler like OneCycleLR has fixed max, prevent going over
                            max_steps = getattr(scheduler, "total_steps", None)
                            step_count = getattr(scheduler, "_step_count", None)
                            if max_steps is not None and step_count is not None and step_count >= max_steps:
                                pass  # Don't step anymore
                            else:
                                scheduler.step()
                        except Exception as e:
                            print(f"[scheduler] skip step -> {e}")  # Optional

                    # --- EMA update
                    with torch.no_grad():
                        for p, ema_p in zip(model.parameters(), ema_params):
                            ema_p.mul_(0.995).add_(p, alpha=0.005)

                    # running means
                    running_total_loss += total_loss.item()
                    running_pred_loss += L_pred.item()
                    running_kl_loss += L_KL.item()
                    
                    # Robust FI update
                    val_fi = L_FI_reg.item() if torch.is_tensor(L_FI_reg) else L_FI_reg
                    running_fi_loss += (val_fi if use_fi_regularization else 0.0)
                    
                    running_ebm_loss += L_EBM.item()
                    running_unc_loss += L_UNC.item()
                    running_accuracy += batch_accuracy
                    num_batches += 1

                    # postfix
                    lr_val = scheduler.get_last_lr()[0] if scheduler else learning_rate
                    postfix_dict = {
                        'Loss': f'{total_loss.item():.4f}',
                        'Pred': f'{L_pred.item():.4f}',
                        'KL': f'{L_KL.item():.4f}',
                        'EBM': f'{(running_ebm_loss / max(num_batches, 1)):.4f}',
                        'UNC': f'{(running_unc_loss / max(num_batches, 1)):.4f}',
                        'Acc': f'{batch_accuracy:.3f}',
                        'LR': f'{lr_val:.6f}',
                        'Evos': f'{_Eneg_mean:.2f}'
                    }
                    if use_fi_regularization:
                        val_fi = L_FI_reg.item() if torch.is_tensor(L_FI_reg) else L_FI_reg
                        postfix_dict['FI'] = f'{val_fi:.4f}'
                    progress_bar.set_postfix(postfix_dict)

                except Exception as e:
                    print(f"Warning in batch {batch_idx}: {e}")
                    continue

            # --- validation (with EMA swap) ---
            backup_params = [p.clone() for p in model.parameters()]
            for p, ema_p in zip(model.parameters(), ema_params):
                p.data.copy_(ema_p.data)
            val_acc, val_loss = validate_enhanced_gem_model(model, validloader, num_classes, device)
            for p, backup_p in zip(model.parameters(), backup_params):
                p.data.copy_(backup_p.data)

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler else None,
                'best_val_acc': best_val_acc,
                'ema_params': [p.cpu() for p in ema_params]  # Save EMA parameters
            }

            is_new_best = False

            # Checkpoint selection based on ckpt_metric
            if ckpt_metric == "ood_aupr" and ood_loader1 is not None:
                # Always track best_val_acc (for display)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                # Evaluate OOD-AUPR every ckpt_eval_freq epoch
                if (epoch + 1) % ckpt_eval_freq == 0 or epoch == 0:
                    print(f"  🔍 Evaluating OOD-AUPR for checkpoint selection...")
                    
                    # Evaluate Far-OOD (SVHN)
                    _, current_far_aupr = _quick_ood_eval(model, validloader, ood_loader1, device, n_samples=500)
                    print(f"  📊 Far-OOD AUPR:  {current_far_aupr:.4f} (Best: {best_far_ood_aupr:.4f})")

                    # Evaluate Near-OOD (CIFAR-100) if available
                    current_near_aupr = 0.0
                    if ood_loader2 is not None:
                        _, current_near_aupr = _quick_ood_eval(model, validloader, ood_loader2, device, n_samples=500)
                        print(f"  📊 Near-OOD AUPR: {current_near_aupr:.4f} (Best: {best_near_ood_aupr:.4f})")

                    # Track best values (without saving separate file)
                    if current_far_aupr > best_far_ood_aupr:
                        best_far_ood_aupr = current_far_aupr
                        print(f"  🎯 New best Far-OOD AUPR: {best_far_ood_aupr:.4f}")
                    
                    if ood_loader2 is not None and current_near_aupr > best_near_ood_aupr:
                        best_near_ood_aupr = current_near_aupr
                        print(f"  🎯 New best Near-OOD AUPR: {best_near_ood_aupr:.4f}")

                    # Save best_model.pt based on average
                    if ood_loader2 is not None:
                        current_avg_aupr = (current_far_aupr + current_near_aupr) / 2
                    else:
                        current_avg_aupr = current_far_aupr
                    
                    if current_avg_aupr > best_ood_aupr:
                        best_ood_aupr = current_avg_aupr
                        is_new_best = True
                        state['best_ood_aupr'] = best_ood_aupr
                        if save_checkpoint(state, best_ckpt):
                            print(f"💾 Saved best model at epoch {epoch + 1}")
                            print(f"✅ New best model saved (Epoch {epoch + 1}, Avg OOD-AUPR={best_ood_aupr:.4f})")
                            print("-" * 50)
                        else:
                            print(f"❌ Failed to save best model at epoch {epoch + 1}")
            else:
                # Main logic: save based on val_acc
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    is_new_best = True
                    state['best_val_acc'] = best_val_acc
                    if save_checkpoint(state, best_ckpt):
                        print(f"💾 Saved best model at epoch {epoch + 1}")
                        print(f"✅ New best model saved (Epoch {epoch + 1}, Val Acc={best_val_acc:.2f}%)")
                        print("-" * 50)
                    else:
                        print(f"❌ Failed to save best model at epoch {epoch + 1}")

            if save_checkpoint(state, last_ckpt):
                print(f"💾 Saved last checkpoint at epoch {epoch + 1}")

            save_training_info(epoch, best_val_acc)

            print("\n" + "=" * 70)
            model_type_str = f"Enhanced GEM{' - MIX' if use_mob else ''}{' + FI' if use_fi_regularization else ''}"
            print(f"{model_type_str} {dataset_name} Epoch {epoch + 1}/{num_epochs} Summary")
            print(f"  Total Loss:  {running_total_loss / max(num_batches, 1):.4f}")
            print(f"  Pred Loss:   {running_pred_loss / max(num_batches, 1):.4f}")
            print(f"  KL Loss:     {running_kl_loss / max(num_batches, 1):.4f}")
            if use_fi_regularization:
                print(f"  FI Loss:     {running_fi_loss / max(num_batches, 1):.4f}")
            print(f"  EBM Loss:    {running_ebm_loss / max(num_batches, 1):.4f}")
            print(f"  Unc Loss:    {running_unc_loss / max(num_batches, 1):.4f}")
            print(f"  Train Acc:   {running_accuracy / max(num_batches, 1):.2%}")
            print("  Validation:")
            print(f"    Acc:       {val_acc:.2f}%")
            print(f"    Loss:      {val_loss:.4f}")
            print(f"    Best Acc:  {best_val_acc:.2f}%")
            # Display Best OOD-AUPR for both Far and Near
            if ckpt_metric == "ood_aupr":
                print(f"    Best AUPR Far-OOD:  {best_far_ood_aupr:.4f}")
                print(f"    Best AUPR Near-OOD: {best_near_ood_aupr:.4f}")
            if is_new_best:
                print(f"    Status:    🎯 NEW BEST!")
            if scheduler:
                print(f"    LR:        {scheduler.get_last_lr()[0]:.6f}")
            print("=" * 70)

        save_training_info(num_epochs - 1, best_val_acc, is_complete=True)
        print(f"🎉 Training completed! Final best validation accuracy: {best_val_acc:.2f}%")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving last checkpoint...")
        interrupted_epoch = epoch
        state = {
            'epoch': interrupted_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'best_val_acc': best_val_acc,
            'ema_params': [p.cpu() for p in ema_params]  # Save EMA parameters
        }
        save_training_info(interrupted_epoch, best_val_acc)
        if save_checkpoint(state, last_ckpt):
            print(f"✅ Last checkpoint saved safely at epoch {interrupted_epoch}. You can resume training later.")
        else:
            print("❌ Failed to save checkpoint during interruption.")

    model_type_str = f"Enhanced GEM{' - MIX' if use_mob else ''}{' + FI' if use_fi_regularization else ''}"
    print(f"\n[{model_type_str}] Training completed for {dataset_name}!")
    print(f"Final Validation Accuracy: {best_val_acc:.2f}%")
    return model


# ====================================================
# Aliases for backward-compatibility
# ====================================================
def train_gem(*args, **kwargs):
    return train_enhanced_gem(*args, **kwargs)


def eval_gem(model, testloader, device):
    return eval_enhanced_gem(model, testloader, device)