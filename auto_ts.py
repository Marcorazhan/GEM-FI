import os
import subprocess
import time
import json
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =======================
# Utilities
# =======================

def _to_probs(x):
    if x.dim() == 2:
        s = x.sum(dim=1, keepdim=True)
        if torch.all((s > 0.999) & (s < 1.001)):
            return x.clamp_min(1e-8)
    return torch.softmax(x, dim=1).clamp_min(1e-8)


def _softmax_with_temp(logits: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(logits / max(float(T), 1e-6), dim=1).clamp_min(1e-12)


@torch.no_grad()
def _metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, T: float = 1.0, num_bins: int = 15):
    probs = _softmax_with_temp(logits, T)
    pred = probs.argmax(dim=1)
    acc = (pred == targets).float().mean().item()
    n = probs.size(0)
    nll = (-probs[torch.arange(n), targets].log()).mean().item()
    num_classes = probs.size(1)
    onehot = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1.0)
    brier = ((probs - onehot).pow(2).sum(dim=1) / num_classes).mean().item()
    conf, _ = probs.max(dim=1)
    correct = (pred == targets).float()
    bins = torch.linspace(0, 1, steps=num_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(num_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < num_bins - 1 else conf <= bins[i + 1])
        if m.any():
            gap = correct[m].mean() - conf[m].mean()
            ece += m.float().mean() * gap.abs()
    return {"accuracy": acc, "nll": nll, "brier": brier, "ece": float(ece)}


class _TemperatureWrapper(nn.Module):
    def __init__(self, model: nn.Module, init_T: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, x, **kwargs):
        out = self.model(x, **kwargs)
        if isinstance(out, tuple):
            logits = out[0]
        elif isinstance(out, dict) and 'logits' in out:
            logits = out['logits']
        else:
            logits = out
        T = self.temperature.clamp_min(1e-4)
        return logits / T


def _nll_criterion(logits, targets):
    return nn.CrossEntropyLoss(reduction="mean")(logits, targets)


def _gather_logits_targets(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, return_features=False) if hasattr(model, 'forward') else model(x)
            # Unpack common containers
            if isinstance(out, tuple):
                logits = out[0]
            elif isinstance(out, dict) and 'logits' in out:
                logits = out['logits']
            else:
                logits = out
            # If output looks like probabilities, convert to effective logits via log
            if logits.dim() == 2:
                row_sum = logits.sum(dim=1, keepdim=True)
                # also check value range to be safe
                if torch.all((row_sum > 0.999) & (row_sum < 1.001)):
                    logits = logits.clamp_min(1e-12).log()
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(y.detach().long().cpu())
    import torch as _torch
    return _torch.cat(all_logits, dim=0), _torch.cat(all_targets, dim=0)


def enhanced_temperature_scaling(model, val_loader, device):
    """Temperature Scaling with advanced settings"""
    wrapper = _TemperatureWrapper(model).to(device)
    logits, targets = _gather_logits_targets(model, val_loader, device)
    logits = logits.to(device)
    targets = targets.to(device)

    # Use more optimal optimizer
    optimizer = torch.optim.LBFGS([wrapper.temperature], lr=0.1, max_iter=100, line_search_fn='strong_wolfe')

    # Add early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scaled = logits / wrapper.temperature.clamp_min(1e-6)
        loss = _nll_criterion(scaled, targets)
        loss.backward()
        return loss

    for i in range(100):
        loss = optimizer.step(closure)
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    T = float(wrapper.temperature.detach().cpu().item())
    return T


def _optimize_temperature(model, val_loader, device):
    """Main function that uses enhanced version"""
    return enhanced_temperature_scaling(model, val_loader, device)


def _infer_flags_from_dir(output_dir: str):
    name = os.path.basename(output_dir)
    name_up = name.upper()

    # Try to load configuration from training_info.json first
    info_path = os.path.join(output_dir, "checkpoints", "training_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            # Read from JSON
            dataset = info.get("dataset", "CIFAR-10")
            # Normalize dataset name (handle cases like "CIFAR" -> "CIFAR-10")
            if dataset.upper() == "CIFAR":
                dataset = "CIFAR-10"
            elif dataset.upper() == "CIFAR100":
                dataset = "CIFAR-100"
            
            num_components = info.get("num_components", 1)
            use_mob = info.get("use_mob", False)
            use_vos = info.get("use_vos", False)  # 🔥 NEW: Read use_vos
            
            # Backbone might not be in older JSONs, so we infer it if missing
            backbone = info.get("backbone", None)
            embedding_dim = 576 if dataset == "MNIST" else 512
            
            if backbone is None:
                 # Fallback backbone inference - prefer dataset-specific defaults
                 if dataset in ("CIFAR-10", "CIFAR-100"):
                     backbone = "ResNet18"
                 elif dataset == "MNIST":
                     backbone = "ConvNet3C3F"
                 elif "RESNET" in name_up or "RESNET18" in name_up:
                     backbone = "ResNet18"
                 elif "VGG" in name_up or "VGG16" in name_up:
                     backbone = "VGG16"
                 elif "CONVNET" in name_up:
                     backbone = "ConvNet3C3F"
                 else:
                     backbone = "ResNet18" if dataset in ("CIFAR-10", "CIFAR-100") else "ConvNet3C3F"
            
            print(f"[auto_ts] ✅ Loaded config from training_info.json: {dataset}, K={num_components}, MIX={use_mob}, VOS={use_vos}")
            return dataset, use_mob, embedding_dim, num_components, backbone, use_vos
        except Exception as e:
            print(f"[auto_ts] ⚠️ Failed to read training_info.json ({e}), falling back to directory parsing.")

    # FALLBACK: Directory name parsing
    if "CIFAR100" in name_up or "CIFAR-100" in name_up:
        dataset = "CIFAR-100"
    elif "CIFAR10" in name_up or "CIFAR-10" in name_up or "CIFAR" in name_up:
        dataset = "CIFAR-10"
    elif "MNIST" in name_up:
        dataset = "MNIST"
    else:
        dataset = "MNIST"

    use_mob = "MOB" in name_up
    # 🔥 Infer use_vos: VOS is active for GEM-FI models
    use_vos = use_mob and ("FI" in name_up) and ("NOFI" not in name_up)
    embedding_dim = 576 if dataset == "MNIST" else 512

    # Backbone inference from directory name
    if "RESNET" in name_up or "RESNET18" in name_up:
        backbone = "ResNet18"
    elif "VGG" in name_up or "VGG16" in name_up:
        backbone = "VGG16"
    elif "CONVNET" in name_up or "CONVNET3C3F" in name_up:
        backbone = "ConvNet3C3F"
    else:
        # Default: CIFAR uses ResNet18, MNIST uses ConvNet3C3F
        backbone = "ResNet18" if dataset in ("CIFAR-10", "CIFAR-100") else "ConvNet3C3F"

    # Try to parse num_components from name (e.g., "_K5_", "_mob5_", etc.)
    num_components = 1
    import re
    match = re.search(r"[_](?:K|mob)(\d+)[_]", name_up + "_")
    if match:
        num_components = int(match.group(1))
    elif ("NOFI" in name_up or "FI" in name_up or "MOB" in name_up):
        num_components = 3
    else:
        num_components = 1
    return dataset, use_mob, embedding_dim, num_components, backbone, use_vos


def _load_val_loader(dataset: str, batch_size: int, data_dir: str = "./data"):
    try:
        from utility import load_datasets
        trainloader, validloader, testloader, ood1, ood2 = load_datasets(
            dataset, batch_size=batch_size, val_size=0.1, data_dir=data_dir
        )
        return validloader
    except Exception as e:
        print(f"[auto_ts] Could not construct validation loader: {e}")
        return None


def _build_model(dataset: str, device, use_mob: bool, num_components: int, embedding_dim: int, backbone: str, use_vos: bool = False):
    try:
        from utility import load_model
        model = load_model(
            ID_dataset=dataset,
            pretrained=False,
            index=0,
            dropout_rate=0.0,
            device=device,
            embedding_dim=embedding_dim,
            use_mob=use_mob,
            num_components=num_components,
            backbone=backbone,
            use_vos=use_vos  # 🔥 NEW: Pass use_vos to ensure correct Tanh setting
        )
        return model
    except Exception as e:
        print(f"[auto_ts] Could not build model: {e}")
        return None


def _load_weights_into(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            tensor_like = {k: v for k, v in ckpt.items() if hasattr(v, "shape")}
            if tensor_like:
                state = tensor_like
    if state is None:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[auto_ts] (info) missing keys: {len(missing)}")
    if unexpected:
        print(f"[auto_ts] (info) unexpected keys: {len(unexpected)}")
    return model


def _save_ts_checkpoint(output_dir: str, model, temperature: float, src_ckpt: str):
    out_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "best_model_ts.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "temperature": float(temperature),
        "source_checkpoint": os.path.basename(src_ckpt),
    }
    torch.save(payload, target)
    print(f"[auto_ts] Saved temperature-scaled checkpoint → {target} (T={temperature:.4f})")
    return target


def run_temperature_scaling_if_phase_at_least(output_dir: str, phase: int, min_phase: int = 3, batch_size: int = 128,
                                              device_str: str = "cuda"):
    try:
        if phase < min_phase:
            print(f"[auto_ts] Skipping TS (phase={phase} < {min_phase})")
            return

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        src_ckpt = os.path.join(ckpt_dir, "best_model.pt")
        if not os.path.exists(src_ckpt):
            print(f"[auto_ts] No best_model.pt found at: {src_ckpt} → skip TS.")
            return

        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        dataset, use_mob, embedding_dim, num_components, backbone, use_vos = _infer_flags_from_dir(output_dir)

        print(f"[auto_ts] Detected config: dataset={dataset}, backbone={backbone}, "
              f"use_mob={use_mob}, num_components={num_components}, use_vos={use_vos}")

        model = _build_model(dataset, device, use_mob, num_components, embedding_dim, backbone, use_vos=use_vos)
        if model is None:
            print("[auto_ts] Could not build model → skip TS.")
            return
        model = model.to(device)
        model = _load_weights_into(model, src_ckpt, device)
        model.eval()

        val_loader = _load_val_loader(dataset, batch_size=batch_size)
        if val_loader is None:
            print("[auto_ts] No validation loader available → skip TS.")
            return

        # BEFORE metrics
        print("[auto_ts] Computing pre-TS metrics...")
        logits_val, y_val = _gather_logits_targets(model, val_loader, device)
        before = _metrics_from_logits(logits_val, y_val, T=1.0)

        # Optimize temperature with enhanced version
        print("[auto_ts] Optimizing temperature...")
        T = _optimize_temperature(model, val_loader, device)

        # AFTER metrics
        print("[auto_ts] Computing post-TS metrics...")
        after = _metrics_from_logits(logits_val, y_val, T=float(T))

        # Save TS checkpoint
        dst_ckpt = _save_ts_checkpoint(output_dir, model, T, src_ckpt)

        # Save JSON report
        rep_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(rep_dir, exist_ok=True)
        rep_path = os.path.join(rep_dir, "ts_report.json")
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "backbone": backbone,
            "use_mob": use_mob,
            "num_components": num_components,
            "temperature": float(T),
            "before": before,
            "after": after,
            "improvement": {
                "accuracy": after["accuracy"] - before["accuracy"],
                "nll": after["nll"] - before["nll"],
                "brier": after["brier"] - before["brier"],
                "ece": after["ece"] - before["ece"]
            },
            "checkpoint_in": src_ckpt,
            "checkpoint_out": dst_ckpt
        }
        with open(rep_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[auto_ts] Wrote before/after JSON report → {rep_path}")

        # Display results
        print("\n" + "=" * 50)
        print("📊 TEMPERATURE SCALING RESULTS")
        print("=" * 50)
        print(f"Optimal Temperature: {T:.4f}")
        print(
            f"Accuracy:  {before['accuracy']:.4f} → {after['accuracy']:.4f} ({report['improvement']['accuracy']:+.4f})")
        print(f"NLL:       {before['nll']:.4f} → {after['nll']:.4f} ({report['improvement']['nll']:+.4f})")
        print(f"Brier:     {before['brier']:.4f} → {after['brier']:.4f} ({report['improvement']['brier']:+.4f})")
        print(f"ECE:       {before['ece']:.4f} → {after['ece']:.4f} ({report['improvement']['ece']:+.4f})")
        print("=" * 50)

        # Update results.json if exists
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)

                # Add temperature scaling information
                results["Temperature_Scaling"] = {
                    "optimal_temperature": float(T),
                    "accuracy_before": before["accuracy"],
                    "accuracy_after": after["accuracy"],
                    "nll_before": before["nll"],
                    "nll_after": after["nll"],
                    "brier_before": before["brier"],
                    "brier_after": after["brier"],
                    "ece_before": before["ece"],
                    "ece_after": after["ece"]
                }

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"[auto_ts] Updated results.json with TS metrics")
            except Exception as e:
                print(f"[auto_ts] Warning: Could not update results.json: {e}")

    except Exception as e:
        print(f"[auto_ts] TS failed safely: {e}")
        import traceback
        print(f"[auto_ts] Error details: {traceback.format_exc()}")


# =======================
# Legacy compatibility functions
# =======================

def temperature_scaling_calibrate(model, val_loader, device, max_iter=100, lr=0.01):
    """
    Legacy function for backward compatibility with old code
    """
    print("[auto_ts] Using legacy temperature scaling interface")
    return enhanced_temperature_scaling(model, val_loader, device)


def find_optimal_temperature(model, val_loader, device):
    """
    Another legacy function for compatibility
    """
    return temperature_scaling_calibrate(model, val_loader, device)


if __name__ == "__main__":
    # Standalone test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory containing checkpoints")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for validation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--phase", type=int, default=3, help="Current phase (must be >= 3 to run TS)")

    args = parser.parse_args()

    print("🔧 Running standalone Temperature Scaling...")
    run_temperature_scaling_if_phase_at_least(
        output_dir=args.output_dir,
        phase=args.phase,
        batch_size=args.batch_size,
        device_str=args.device
    )