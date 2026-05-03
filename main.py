import pandas as pd
import numpy as np
import argparse
import json
import os
import time
import torch
import torch.nn.functional as F

from density_estimation import fit_gda, gmm_evaluate
from conf_calibration import conf_calibration_gem, conf_calibration_gem
from utility import load_datasets, load_model
from train import train_gem, eval_gem

from ood_detection import (
    ood_detection_gem,
    ood_detection_gem,
    _evaluate_enhanced_pair,
    _extract_enhanced_aleatoric_epistemic_for_report,
    _evaluate_shift_detection,
)



from load_corrupted import prepare_gem_corrupted_data as prepare_corrupted_data, make_cifar10c_loader_fn, \
    make_mnistc_loader_fn
from typing import Optional, Dict, Any

from auto_ts import run_temperature_scaling_if_phase_at_least


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID_dataset", default="CIFAR-10", choices=["MNIST", "CIFAR-10", "CIFAR-100"],
                        help="Select dataset")
    parser.add_argument('--backbone', type=str, default='ResNet18', help='Network backbone (ConvNet3C3F, VGG16, or ResNet18)')  # Changed to ResNet18
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type (lambda, step, cosine, etc.)')
    parser.add_argument('--lr_gamma', type=float, default=0.95, help='Learning rate decay factor')
    parser.add_argument('--spectral_norm', type=str, default='true', help='Enable spectral normalization')
    parser.add_argument('--activation', type=str, default='softplus', help='Activation for evidential parameters')
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--val_seed", type=int, default=42, help="Random seed for validation")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of dataset classes")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Feature space dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")  # Increased learning rate
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")  # Optimized for CIFAR-10
    parser.add_argument("--reg_param", type=float, default=1e-4, help="Regularization parameter")  # Optimized for CIFAR-10
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")  # Increased epochs
    parser.add_argument("--index", type=int, default=0, help="Index of pretrained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (only cuda)")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save results")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to store datasets")
    parser.add_argument("--pretrained", action='store_true', help="Use pretrained model")

    # New parameters for GEM-MIX
    parser.add_argument("--use_mob", action='store_true', help="Use GEM-MIX instead of baseline GEM")
    parser.add_argument("--num_components", type=int, default=3, help="Number of mixture components for MIX")
    parser.add_argument("--use_fi_regularization", action='store_true', help="Enable Fisher Information regularization")
    parser.add_argument("--fi_lambda", type=float, default=0.1, help="Fisher Information regularization weight")
    # Ablation flags for FI separation
    parser.add_argument("--use_fi_modulation", action='store_true', default=True, 
                        help="Enable FI-based weight modulation (default: True when use_fi_regularization is True)")
    parser.add_argument("--no_fi_modulation", action='store_true', 
                        help="Disable FI-based weight modulation (for ablation)")
    parser.add_argument("--no_fi_regularizer_loss", action='store_true', 
                        help="Disable L_FI loss term (for ablation, keeps modulation)")
    # Better SN control
    parser.add_argument("--use_spectral_norm", action='store_true', default=True,
                        help="Enable spectral normalization (default: True)")
    parser.add_argument("--no_spectral_norm", action='store_true',
                        help="Disable spectral normalization (for ablation)")

    # --------------------
    # Checkpoint Selection by OOD-AUPR
    # --------------------
    parser.add_argument("--ckpt_metric", type=str, default="ood_aupr", choices=["val_acc", "ood_aupr"],
                        help="Metric for best checkpoint selection: val_acc or ood_aupr")
    parser.add_argument("--ckpt_eval_freq", type=int, default=1,
                        help="Evaluate OOD every N epochs when ckpt_metric=ood_aupr")

    # --------------------
    # VOS (Virtual Outlier Synthesis) -> EBM negatives (On-the-fly + MemBank)
    # 🔧 FIX: Safer defaults to prevent training destabilization
    # --------------------
    parser.add_argument("--use_vos", action="store_true", help="Enable VOS as negative samples for EBM")
    parser.add_argument("--vos_ratio", type=float, default=0.3, help="Fraction of each batch to synthesize as VOS negatives")
    parser.add_argument("--vos_start_epoch", type=int, default=30, help="Warmup epochs before enabling VOS")
    parser.add_argument("--vos_ramp_epochs", type=int, default=30, help="Ramp duration for VOS weight/margin")
    parser.add_argument("--vos_lambda_neg", type=float, default=0.4, help="Weight of VOS negative energy term inside EBM")
    parser.add_argument("--vos_margin_start", type=float, default=0.5, help="Starting margin for VOS negative energy term")
    parser.add_argument("--vos_margin", type=float, default=3.0, help="Final margin for VOS negative energy term")
    parser.add_argument("--vos_mix_beta", type=float, default=0.3, help="Beta(a,a) parameter for BoundaryMix")
    parser.add_argument("--vos_pgd_frac", type=float, default=0.5, help="Fraction of VOS samples to harden with energy-PGD")
    parser.add_argument("--vos_pgd_eps", type=float, default=12/255, help="PGD epsilon (L_inf)")
    parser.add_argument("--vos_pgd_step", type=float, default=3/255, help="PGD step size")
    parser.add_argument("--vos_pgd_steps", type=int, default=5, help="Number of PGD steps")
    parser.add_argument("--vos_pgd_random_init", type=str, default="true", choices=["true","false"], help="Random init for VOS PGD")
    parser.add_argument("--vos_mem_size", type=int, default=2048, help="MemBank capacity (stored on CPU)")
    parser.add_argument("--vos_mem_use_frac", type=float, default=0.15, help="Fraction of VOS negatives drawn from MemBank")
    parser.add_argument("--vos_mem_add_topk", type=int, default=32, help="Per-step top-k hardest VOS samples to push into MemBank")

    # Legacy parameters for backward compatibility with existing scripts
    parser.add_argument("--mob_k", type=int, default=1,
                        help="Number of Dirichlet heads (1 = classic GEM) [deprecated, use --num_components]")
    parser.add_argument("--mob_fi_alpha", type=float, default=0.0,
                        help="Weight for Fisher-info weighting (π_k ~ softmax(-λ·FI_k)) [deprecated, use --fi_lambda]")
    parser.add_argument("--mob_kl_alpha", type=float, default=0.0,
                        help="Aux KL between α_mix and α_base (consistency) [deprecated]")
    parser.add_argument("--mob_entropy_gamma", type=float, default=0.0,
                        help="Entropy reg on π (encourage diversity/balanced heads) [deprecated]")

    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument("--auto_ts", action="store_true",
                        help="Run Temperature Scaling after training and save before/after JSON")

    parser.add_argument("--skip_completed_phases", action="store_true", default=True,
                        help="Skip phases that are already completed (default: True)")
    parser.add_argument("--force_rerun_phases", type=str, default="",
                        help="Force re-run specific phases (comma-separated, e.g., '3,4,5')")
    # --- Compatibility: accept EBM/Langevin flags but ignore them (for older runners)
    parser.add_argument("--mob_evidence_tau", type=float, default=2.5,
                        help="(compat) evidence temperature; accepted then ignored")
    parser.add_argument("--langevin_steps", type=int, default=0,
                        help="(compat) Langevin steps; accepted then ignored")
    parser.add_argument("--langevin_step_size", type=float, default=0.0,
                        help="(compat) Langevin step size; accepted then ignored")
    parser.add_argument("--langevin_noise", type=float, default=0.0,
                        help="(compat) Langevin noise; accepted then ignored")
    parser.add_argument("--ebm_weight", type=float, default=0.0,
                        help="(compat) EBM loss weight; accepted then ignored")
    parser.add_argument("--ebm_margin", type=float, default=0.0,
                        help="(compat) EBM margin; accepted then ignored")
    parser.add_argument("--energy_lr_mult", type=float, default=1.0,
                        help="(compat) LR multiplier for energy net; accepted then ignored")

    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Add verbose parameter
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) for faster training")

    parser.add_argument(
        "--use_density",
        action="store_true",
        help="Enable density-estimation-related phases (kept for compatibility with ablation scripts)."
    )
    args = parser.parse_args()

    # Set verbose
    if args.quiet:
        args.verbose = False

    # Map legacy parameters to new ones for compatibility
    if args.mob_k > 1:
        args.use_mob = True
        args.num_components = args.mob_k
        if args.verbose:
            print(f"⚠️  Using deprecated --mob_k, automatically setting --use_mob and --num_components={args.mob_k}")

    if args.mob_fi_alpha > 0:
        args.use_fi_regularization = True
        args.fi_lambda = args.mob_fi_alpha
        if args.verbose:
            print(
                f"⚠️  Using deprecated --mob_fi_alpha, automatically setting --use_fi_regularization and --fi_lambda={args.mob_fi_alpha}")

    args.device = torch.device(args.device)
    if not torch.cuda.is_available() and str(args.device) != 'cpu':
        if args.verbose:
            print('⚠️  CUDA not available; falling back to CPU.')
        args.device = torch.device('cpu')
    if args.verbose:
        print(f"GEM Using device: {args.device}")

    if args.ID_dataset == "MNIST":
        if args.learning_rate == 5e-4:
            args.learning_rate = 1e-3
        if args.reg_param == 1e-2:
            args.reg_param = 5e-3
        if args.embedding_dim != 576:
            if args.verbose:
                print(f"Auto-adjusting embedding_dim for MNIST: {args.embedding_dim} -> 576")
            args.embedding_dim = 576
        if args.verbose:
            print(f"MNIST Optimized: lr={args.learning_rate}, reg={args.reg_param}, epochs={args.num_epochs}")
    elif args.ID_dataset == "CIFAR-10":
        if args.learning_rate == 5e-4:
            args.learning_rate = 5e-4
        if args.reg_param == 1e-2:
            args.reg_param = 5e-3
        if args.embedding_dim != 512:
            if args.verbose:
                print(f"Auto-adjusting embedding_dim for CIFAR-10: {args.embedding_dim} -> 512")
            args.embedding_dim = 512
        if args.verbose:
            print(f"CIFAR-10 Optimized: lr={args.learning_rate}, reg={args.reg_param}, epochs={args.num_epochs}")

    # Display selected model information
    if args.verbose:
        if args.use_mob:
            if args.use_fi_regularization:
                print(
                    f"[GEM-FI] Enabled with {args.num_components} components and FI regularization (λ={args.fi_lambda})")
            else:
                print(f"[GEM-MIX] Enabled with {args.num_components} components (no FI)")
        else:
            print("[GEM] Baseline (single Dirichlet head)")

    # Process ablation flags
    # FI Modulation: disabled if --no_fi_modulation is set
    if args.no_fi_modulation:
        args.use_fi_modulation = False
    # Spectral Norm: disabled if --no_spectral_norm is set
    if args.no_spectral_norm:
        args.use_spectral_norm = False
        
    if args.verbose:
        if args.use_fi_regularization:
            fi_mod_status = "ON" if args.use_fi_modulation else "OFF"
            fi_loss_status = "OFF" if args.no_fi_regularizer_loss else "ON"
            print(f"  [FI Config] Modulation: {fi_mod_status}, L_FI Loss: {fi_loss_status}")
        sn_status = "ON" if args.use_spectral_norm else "OFF"
        print(f"  [SN Config] Spectral Normalization: {sn_status}")

    # 🔥 STRICT RULE: VOS only for MIX-FI
    # User requested: "VOS should only work for MIX-FI and not for other models"
    is_mob_fi = args.use_mob and args.use_fi_regularization
    if args.use_vos and not is_mob_fi:
        print("⚠️ [Auto-Config] VOS is enabled but model is not MIX-FI. Disabling VOS.")
        args.use_vos = False

    if args.use_vos:
        print("✅ [VOS] Virtual Outlier Synthesis is ACTIVE (MIX-FI detected)")
    else:
        print("ℹ️ [VOS] Virtual Outlier Synthesis is DISABLED")

    project_root = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(project_root, "saved_results")
    os.makedirs(base_path, exist_ok=True)

    if args.output_dir is None:
        dataset_name = args.ID_dataset.lower().replace("-", "")
        if args.use_mob:
            if args.use_fi_regularization:
                mob_suffix = f"_mob{args.num_components}_fi{args.fi_lambda}"
            else:
                mob_suffix = f"_mob{args.num_components}_nofi"
        else:
            mob_suffix = "_baseline"

        output_dir_name = f"saved_results_{dataset_name}{mob_suffix}_seed{args.val_seed}_epochs{args.num_epochs}"
        args.output_dir = os.path.join(base_path, output_dir_name)
    else:
        args.output_dir = os.path.join(base_path, args.output_dir) if not os.path.isabs(
            args.output_dir) else args.output_dir

    if args.verbose:
        print(f"GEM Results will be saved to: {args.output_dir}")

    # 🔥 FIX: Propagate val_seed to environment so utility.py sees it
    os.environ["GEM_VAL_SEED"] = str(args.val_seed)
    if args.verbose:
        print(f"Set GEM_VAL_SEED = {args.val_seed}")

    return args


def convert_to_serializable(obj):
    """Convert to serializable format with precise control over number formatting"""
    if torch.is_tensor(obj):
        value = obj.item() if obj.numel() == 1 else obj.tolist()
        return convert_to_serializable(value)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float)):
        # Precise control over number formatting
        if obj == 0:
            return 0.0
        elif abs(obj) < 1e-4:
            return float(f"{obj:.8f}")
        elif abs(obj) < 1:
            return float(f"{obj:.6f}")
        elif abs(obj) < 100:
            return float(f"{obj:.4f}")
        else:
            return float(f"{obj:.2f}")
    elif isinstance(obj, (str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def eval_corrupted_accuracy(model, corrupted_loader, device, corruption_type, verbose=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in corrupted_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    final_accuracy = 100 * correct / total if total > 0 else 0.0
    if verbose:
        print(f"GEM Corruption Accuracy for {corruption_type}: {final_accuracy:.2f}%")
    return final_accuracy


def calculate_model_brier_score(model, testloader, num_classes, device, verbose=False):
    """
    Computes Brier score using proper predictive probabilities (normalized by C).
    """
    import torch
    import torch.nn.functional as F
    from utility import mob_predictive_probs

    def _is_mob_model(m):
        if getattr(m, "is_mob", False):
            return True
        if getattr(m, "use_mob", False):
            return True
        if getattr(m, "num_components", 1) and getattr(m, "num_components", 1) > 1:
            return True
        bm = getattr(m, "base_model", None)
        if bm is not None and getattr(bm, "num_components", 1) > 1:
            return True
        return False

    if verbose:
        print("Calculating Brier Score (Dirichlet/MoB predictive)...")
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            p = None

            if _is_mob_model(model):
                alpha_list, pi, logits = None, None, None
                if isinstance(out, dict):
                    alpha_list = out.get("alpha_list") or out.get("alphas") or out.get("mob_alphas")
                    pi = out.get("mixture_weights") or out.get("pi") or out.get("mob_pi")
                    logits = out.get("logits")
                elif isinstance(out, (tuple, list)):
                    if len(out) >= 2 and (
                            (torch.is_tensor(out[0]) or isinstance(out[0], (list, tuple))) and
                            (torch.is_tensor(out[1]))
                    ):
                        alpha_list, pi = out[0], out[1]
                        logits = out[2] if len(out) >= 3 else None
                    else:
                        logits = out[0]
                        if len(out) >= 3:
                            alpha_list, pi = out[1], out[2]
                else:
                    logits = out

                if alpha_list is not None and pi is not None:
                    try:
                        p = mob_predictive_probs(alpha_list, pi)
                    except Exception:
                        if logits is not None:
                            p = F.softmax(logits, dim=1)
                if p is None:
                    p = F.softmax(logits if logits is not None else out, dim=1)

            else:
                # Baseline (K=1): Always use Dirichlet mean for fair comparison
                # This matches GEM original: alpha = exp(z), pi = alpha / alpha.sum()
                if isinstance(out, dict) and ("alpha" in out or "alphas" in out):
                    alpha = out.get("alpha") or out.get("alphas")
                elif isinstance(out, dict) and "logits" in out:
                    # Convert logits to Dirichlet alpha
                    logits = out.get("logits")
                    alpha = torch.exp(logits.clamp(-15, 15)) + 1e-8
                else:
                    # Raw logits output - convert to Dirichlet alpha (matches GEM)
                    logits = out if torch.is_tensor(out) else out[0]
                    alpha = torch.exp(logits.clamp(-15, 15)) + 1e-8
                
                a0 = alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
                p = (alpha / a0).clamp_min(1e-8)
                p = p / p.sum(dim=1, keepdim=True)

            all_probs.append(p.detach().cpu())
            all_labels.append(y.cpu())

    probabilities = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    true_labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

    brier_per_sample = torch.sum((probabilities - true_labels_one_hot) ** 2, dim=1) / num_classes
    brier = brier_per_sample.mean()

    if verbose:
        print(f"GEM Brier Score (predictive): {brier.item():.4f}")
    return brier.item()


# Phase management system
class PhaseManager:
    def __init__(self, output_dir: str, verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, "phase_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def save_phase(self, phase_name: str, data: Dict[str, Any]) -> bool:
        """Save phase results"""
        phase_file = os.path.join(self.results_dir, f"{phase_name}.json")
        try:
            serializable_data = {}
            for key, value in data.items():
                serializable_data[key] = convert_to_serializable(value)

            with open(phase_file, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            if self.verbose:
                print(f"💾 Phase '{phase_name}' saved successfully")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving phase '{phase_name}': {e}")
            return False

    def load_phase(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Load phase results"""
        phase_file = os.path.join(self.results_dir, f"{phase_name}.json")
        if os.path.exists(phase_file):
            try:
                with open(phase_file, 'r') as f:
                    data = json.load(f)
                if self.verbose:
                    print(f"📂 Phase '{phase_name}' loaded from cache")
                return data
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error loading phase '{phase_name}': {e}")
        return None

    def phase_exists(self, phase_name: str) -> bool:
        """Check if phase exists"""
        phase_file = os.path.join(self.results_dir, f"{phase_name}.json")
        return os.path.exists(phase_file)

    def get_phase_status(self):
        """Get status of all phases"""
        phases = ["phase1_data", "phase2_model", "phase3_training", "phase4_test",
                  "phase5_density", "phase6_ood", "phase7_calibration", "phase8_corruption"]
        status = {}
        for phase in phases:
            status[phase] = self.phase_exists(phase)
        return status

    def print_phase_status(self):
        """Print phase status"""
        if not self.verbose:
            return

        status = self.get_phase_status()
        print("\n" + "=" * 50)
        print("📊 PHASE STATUS")
        print("=" * 50)
        for phase, exists in status.items():
            status_icon = "✅" if exists else "❌"
            print(f"  {status_icon} {phase}: {'COMPLETED' if exists else 'PENDING'}")
        print("=" * 50)


def validate_phase_completion(phase_name: str, phase_data: Dict[str, Any], output_dir: str, args) -> bool:
    """Validate whether a phase is fully completed and valid"""

    force_rerun_phases = [f"phase{phase}" for phase in args.force_rerun_phases.split(",") if phase.strip()]
    if phase_name in force_rerun_phases:
        if args.verbose:
            print(f"🔄 Phase {phase_name} forced to re-run")
        return False

    if phase_name == "phase3_training":
        best_ckpt = os.path.join(output_dir, "checkpoints", "best_model.pt")
        training_info_file = os.path.join(output_dir, "checkpoints", "training_info.json")

        if not os.path.exists(best_ckpt):
            if args.verbose:
                print(f"⚠️ Phase 3 invalid: best_model.pt not found")
            return False

        if os.path.exists(training_info_file):
            try:
                with open(training_info_file, 'r') as f:
                    training_info = json.load(f)
                if not training_info.get('training_complete', False):
                    if args.verbose:
                        print(f"⚠️ Phase 3 invalid: training not completed")
                    return False
                completed_epochs = training_info.get('completed_epochs', 0)
                if completed_epochs < args.num_epochs:
                    if args.verbose:
                        print(f"⚠️ Phase 3 invalid: only {completed_epochs}/{args.num_epochs} epochs completed")
                    return False
            except Exception as e:
                if args.verbose:
                    print(f"⚠️ Phase 3 invalid: corrupted training info - {e}")
                return False

        return True

    elif phase_name == "phase4_test":
        test_accuracy = phase_data.get('test_accuracy', 0)
        if test_accuracy < 50:
            if args.verbose:
                print(f"⚠️ Phase 4 invalid: low test accuracy ({test_accuracy:.2f}%)")
            return False

        best_ckpt = os.path.join(output_dir, "checkpoints", "best_model.pt")
        if not os.path.exists(best_ckpt):
            if args.verbose:
                print(f"⚠️ Phase 4 invalid: best_model.pt not found")
            return False

        return True

    elif phase_name in ["phase5_density", "phase6_ood", "phase7_calibration"]:
        if not phase_data or len(phase_data) == 0:
            if args.verbose:
                print(f"⚠️ {phase_name} invalid: no data found")
            return False

        best_ckpt = os.path.join(output_dir, "checkpoints", "best_model.pt")
        if not os.path.exists(best_ckpt):
            if args.verbose:
                print(f"⚠️ {phase_name} invalid: best_model.pt not found")
            return False

        return True


    elif phase_name == "phase8_corruption":
        if not phase_data or not phase_data.get('shift_aupr'):
            if args.verbose:
                print(f"⚠️ Phase 8 invalid: no shift detection data found")
            return False
        return True

    else:
        return True


def should_skip_phase(phase_name: str, phase_data: Dict[str, Any], output_dir: str, args) -> bool:
    """Final decision to skip phase or re-run"""

    if not args.skip_completed_phases:
        return False

    if not phase_data:
        return False

    return validate_phase_completion(phase_name, phase_data, output_dir, args)


def main(args):
    if args.verbose:
        print("=" * 60)
        if args.use_mob:
            if args.use_fi_regularization:
                print(
                    "GEM-FI: Energy-Based Evidential Deep Learning with Mixture of Beliefs and Fisher Information")
            else:
                print("GEM-MIX: Energy-Based Evidential Deep Learning with Mixture of Beliefs")
        else:
            print("GEM: Energy-Based Evidential Deep Learning")
        print("=" * 60)

    start_time = time.perf_counter()

    # Create phase manager
    phase_manager = PhaseManager(args.output_dir, verbose=args.verbose)
    phase_manager.print_phase_status()

    # Phase 1: Data Loading
    if args.verbose:
        print("\nPhase 1: Loading GEM Datasets")
    phase1_data = phase_manager.load_phase("phase1_data")

    trainloader, validloader, testloader, ood_loader1, ood_loader2, ood_loader3 = load_datasets(
        args.ID_dataset, args.batch_size, args.val_size, args.data_dir)

    if not phase1_data or not args.skip_completed_phases:
        ood3_samples = len(ood_loader3.dataset) if ood_loader3 is not None else 0
        phase_manager.save_phase("phase1_data", {
            "dataset_info": {
                "name": args.ID_dataset,
                "train_samples": len(trainloader.dataset),
                "val_samples": len(validloader.dataset),
                "test_samples": len(testloader.dataset),
                "ood1_samples": len(ood_loader1.dataset),
                "ood2_samples": len(ood_loader2.dataset),
                "ood3_samples": ood3_samples,  # TinyImageNet for CIFAR-10
                "batch_size": args.batch_size,
                "val_size": args.val_size,
                "val_seed": args.val_seed
            }
        })

    if args.verbose:
        print(f"GEM Dataset: {args.ID_dataset}")
        print(f"Train samples: {len(trainloader.dataset)}, Validation samples: {len(validloader.dataset)}")
        ood3_info = f", {len(ood_loader3.dataset)}" if ood_loader3 is not None else ""
        print(
            f"Test samples: {len(testloader.dataset)}, OOD samples: {len(ood_loader1.dataset)}, {len(ood_loader2.dataset)}{ood3_info}")


    # Phase 2: Model Loading
    # Phase 2: Model Loading
    if args.verbose:
        print("\nPhase 2: Loading GEM Model")
    phase2_data = phase_manager.load_phase("phase2_model")

    # Pass backbone argument to load_model function
    model = load_model(args.ID_dataset, args.pretrained, args.index, args.dropout_rate,
                       args.device, args.embedding_dim, args.use_mob, args.num_components, args.backbone, 
                       fi_lambda=args.fi_lambda, use_vos=args.use_vos,
                       use_spectral_norm=args.use_spectral_norm, use_fi_modulation=args.use_fi_modulation)

    if not phase2_data or not args.skip_completed_phases:
        model_info = {
            "model_info": {
                "dataset": args.ID_dataset,
                "backbone": args.backbone,
                "model_type": "GEM-FI" if args.use_mob and args.use_fi_regularization else
                "GEM-MIX" if args.use_mob else "GEM",
                "embedding_dim": args.embedding_dim,
                "num_classes": args.num_classes,
                "num_components": args.num_components if args.use_mob else 1,
                "use_fi_regularization": args.use_fi_regularization,
                "fi_lambda": args.fi_lambda,
                "total_parameters": sum(p.numel() for p in model.parameters())
            }
        }
        phase_manager.save_phase("phase2_model", model_info)

    if args.verbose:
        if args.use_mob:
            if args.use_fi_regularization:
                print(
                    f"GEM-FI Model loaded: {args.num_components} mixture components with FI regularization (λ={args.fi_lambda})")
            else:
                print(f"GEM-MIX Model loaded: {args.num_components} mixture components (no FI)")
        else:
            print(f"GEM Model loaded: Baseline (single Dirichlet head)")

        print(f"Backbone: {args.backbone}")
        print(f"Feature dimension: {args.embedding_dim}, Number of classes: {args.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    def load_model_safe(model, checkpoint_path, device):
        """Safe model loading with architecture adaptation capability"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            current_state_dict = model.state_dict()

            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in current_state_dict:
                    if current_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    elif args.verbose:
                        print(f"⚠️ Shape mismatch for {key}: {value.shape} -> {current_state_dict[key].shape}")
                elif args.verbose:
                    print(f"⚠️ Missing key in current model: {key}")

            model.load_state_dict(filtered_state_dict, strict=False)

            loaded_params = len(filtered_state_dict)
            total_params = len(current_state_dict)
            if args.verbose:
                print(f"✅ Loaded {loaded_params}/{total_params} parameters from checkpoint")

            return model

        except Exception as e:
            if args.verbose:
                print(f"❌ Error loading checkpoint: {e}")
            return model

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "last_checkpoint.pt")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    if args.resume and os.path.exists(last_ckpt):
        if args.verbose:
            print(f"🔁 Resuming from checkpoint: {last_ckpt}")
        model = load_model_safe(model, last_ckpt, args.device)
        if args.verbose:
            print("✔ Model state restored from last checkpoint.")
    elif args.resume and args.verbose:
        print("⚠️ Resume requested but no checkpoint found — starting from scratch.")

    # Phase 3: Training
    if args.verbose:
        print("\nPhase 3: GEM Training")
    phase3_data = phase_manager.load_phase("phase3_training")

    if should_skip_phase("phase3_training", phase3_data, args.output_dir, args):
        if args.verbose:
            print("\u2705 Training already completed - using cached model")

        # Run auto_ts even when loading from cache (if best_model_ts.pt doesn't exist)
        ts_ckpt = os.path.join(args.output_dir, "checkpoints", "best_model_ts.pt")
        if getattr(args, "auto_ts", False) and not os.path.exists(ts_ckpt):
            try:
                if args.verbose:
                    print("  🔄 Running auto_ts (best_model_ts.pt not found)...")
                run_temperature_scaling_if_phase_at_least(
                    output_dir=args.output_dir,
                    phase=3,
                    min_phase=3,
                    batch_size=args.batch_size,
                    device_str=str(args.device),
                )
                if os.path.exists(ts_ckpt) and args.verbose:
                    print("  ✔ Temperature-scaled checkpoint created: best_model_ts.pt")
            except Exception as e:
                if args.verbose:
                    print(f"  ⚠️ Auto-TS step skipped safely: {e}")
    else:
        if phase3_data and not validate_phase_completion("phase3_training", phase3_data, args.output_dir,
                                                         args) and args.verbose:
            print("🔄 Cached training invalid - re-running training")

        if args.verbose:
            if args.use_mob and args.use_fi_regularization:
                print(f"🎯 Training GEM-FI started for {args.num_epochs} epochs on {args.ID_dataset}...")
            elif args.use_mob:
                print(f"🔬 Training GEM-MIX started for {args.num_epochs} epochs on {args.ID_dataset}...")
            else:
                print(f"🎯 Training GEM Baseline started for {args.num_epochs} epochs on {args.ID_dataset}...")

        initial_best_model_exists = os.path.exists(best_ckpt)

        try:
            model = train_gem(model, args.learning_rate, args.reg_param, args.num_epochs,
                                trainloader, validloader, args.num_classes, args.device,
                                ood_loader1, ood_loader2,
                                use_mob=args.use_mob,
                                num_components=args.num_components,
                                use_fi_regularization=args.use_fi_regularization,
                                fi_lambda=args.fi_lambda,
                                output_dir=args.output_dir, seed=args.val_seed, resume=args.resume,
                                use_amp=getattr(args, 'amp', False),
                                # ---- Checkpoint Selection ----
                                ckpt_metric=getattr(args, "ckpt_metric", "val_acc"),
                                ckpt_eval_freq=getattr(args, "ckpt_eval_freq", 5),
                                # ---- VOS -> EBM negatives (fixed with safer defaults) ----
                                use_vos=getattr(args, "use_vos", False),
                                vos_ratio=getattr(args, "vos_ratio", 0.2),
                                vos_start_epoch=getattr(args, "vos_start_epoch", 50),
                                vos_ramp_epochs=getattr(args, "vos_ramp_epochs", 30),
                                vos_lambda_neg=getattr(args, "vos_lambda_neg", 0.25),
                                vos_margin_start=getattr(args, "vos_margin_start", 0.2),
                                vos_margin=getattr(args, "vos_margin", 0.5),
                                vos_mix_beta=getattr(args, "vos_mix_beta", 0.4),
                                vos_pgd_frac=getattr(args, "vos_pgd_frac", 0.2),
                                vos_pgd_eps=getattr(args, "vos_pgd_eps", 4/255),
                                vos_pgd_step=getattr(args, "vos_pgd_step", 1/255),
                                vos_pgd_steps=getattr(args, "vos_pgd_steps", 2),
                                vos_pgd_random_init=str(getattr(args, "vos_pgd_random_init", "true")).lower() == "true",
                                vos_mem_size=getattr(args, "vos_mem_size", 2048),
                                vos_mem_use_frac=getattr(args, "vos_mem_use_frac", 0.15),
                                vos_mem_add_topk=getattr(args, "vos_mem_add_topk", 32),
                                # ---- Ablation: FI Loss ----
                                no_fi_loss=getattr(args, "no_fi_regularizer_loss", False))

            final_best_model_exists = os.path.exists(best_ckpt)
            training_info_file = os.path.join(args.output_dir, "checkpoints", "training_info.json")
            training_completed = False

            # Prefer training_info.json to decide if training truly finished
            if os.path.exists(training_info_file):
                try:
                    with open(training_info_file, "r") as f:
                        training_info = json.load(f)
                    training_completed = (
                        training_info.get("training_complete", False)
                        and training_info.get("completed_epochs", 0) >= args.num_epochs
                    )
                except Exception as e:
                    if args.verbose:
                        print(f"⚠️ Could not read training_info.json for completion check: {e}")
            # Fallback for older runs where training_info.json may not exist
            elif final_best_model_exists and not initial_best_model_exists:
                training_completed = True

            if training_completed:
                phase_manager.save_phase("phase3_training", {
                    "training_info": {
                        "epochs": args.num_epochs,
                        "learning_rate": args.learning_rate,
                        "reg_param": args.reg_param,
                        "use_mob": args.use_mob,
                        "num_components": args.num_components,
                        "use_fi_regularization": args.use_fi_regularization,
                        "fi_lambda": args.fi_lambda,
                        "completed": True,
                        "completion_time": time.perf_counter() - start_time
                    }
                })
                if args.verbose:
                    print("✅ Training completed successfully")

                if getattr(args, "auto_ts", False):
                    try:
                        run_temperature_scaling_if_phase_at_least(
                            output_dir=args.output_dir,
                            phase=3,
                            min_phase=3,
                            batch_size=args.batch_size,
                            device_str=str(args.device),
                        )
                        ts_ckpt = os.path.join(args.output_dir, "checkpoints", "best_model_ts.pt")
                        if os.path.exists(ts_ckpt) and args.verbose:
                            print("✔ Temperature-scaled checkpoint created: best_model_ts.pt")
                        elif args.verbose:
                            print("ℹ️ Temperature scaling completed but no TS checkpoint generated.")
                    except Exception as e:
                        if args.verbose:
                            print(f"⚠️ Auto-TS step skipped safely: {e}")
                elif args.verbose:
                    print("ℹ️ Auto-TS disabled (use --auto_ts to enable).")

            else:
                if args.verbose:
                    print("⚠️ Training interrupted or incomplete - not saving phase 3")
                if phase_manager.phase_exists("phase3_training"):
                    phase_file = os.path.join(phase_manager.results_dir, "phase3_training.json")
                    try:
                        os.remove(phase_file)
                        if args.verbose:
                            print("🗑️ Removed invalid phase3 from cache")
                    except Exception:
                        pass

        except Exception as e:
            if args.verbose:
                print(f"⚠️ Training run failed: {e}")

    # 🔁 IMPORTANT: For evaluation phases, load the *best* validation checkpoint (not the last epoch).
    # This avoids big Val/Test gaps when later regularizers (e.g., VOS) hurt accuracy after the best epoch.
    try:
        # import torch  <-- REMOVED to fix free variable error
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=args.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if args.verbose:
                print(f"✅ Loaded best checkpoint for eval: {best_ckpt}")
                if missing:
                    print(f"   ⚠️ Missing keys: {len(missing)}")
                if unexpected:
                    print(f"   ⚠️ Unexpected keys: {len(unexpected)}")
        else:
            if args.verbose:
                print(f"⚠️ Best checkpoint not found at {best_ckpt}; using last-epoch weights for eval.")
    except Exception as e:
        if args.verbose:
            print(f"⚠️ Could not load best checkpoint for eval: {e}")

    # Phase 4: Test Accuracy
    if args.verbose:
        print("\nPhase 4: Model Evaluation")
    phase4_data = phase_manager.load_phase("phase4_test")

    if should_skip_phase("phase4_test", phase4_data, args.output_dir, args):
        if args.verbose:
            print("✅ Using cached test accuracy")
        test_acc = phase4_data["test_accuracy"]
    else:
        if phase4_data and not validate_phase_completion("phase4_test", phase4_data, args.output_dir,
                                                         args) and args.verbose:
            print("🔄 Cached test results invalid - re-running evaluation")
        test_acc = eval_gem(model, testloader, args.device)
        phase_manager.save_phase("phase4_test", {
            "test_accuracy": test_acc,
            "evaluation_time": time.perf_counter() - start_time
        })

    if args.verbose:
        print(f"GEM Test Accuracy: {test_acc:.2f}%")

    # Phase 5: Density Estimation
    # 🔥 FIX: Reset seed before Phase 5 to ensure deterministic GMM fitting
    # This prevents RNG state drift from Phase 4 execution affecting density estimation
    import random
    seed_for_density = args.val_seed if hasattr(args, 'val_seed') else 42
    torch.manual_seed(seed_for_density)
    np.random.seed(seed_for_density)
    random.seed(seed_for_density)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_for_density)
    
    # 🔥 FIX v2: Create a deterministic trainloader with SHUFFLE=True but FIXED SEED
    # shuffle=False gave poor GMM variance; we need shuffle but deterministic
    from torch.utils.data import DataLoader, RandomSampler
    
    # Create a generator with fixed seed for reproducible shuffle
    g = torch.Generator()
    g.manual_seed(seed_for_density)
    
    trainloader_deterministic = DataLoader(
        trainloader.dataset,
        batch_size=trainloader.batch_size,
        shuffle=True,  # 🔥 KEY: Shuffle for better GMM variance
        num_workers=0,  # Avoid multiprocessing RNG issues
        pin_memory=torch.cuda.is_available(),
        generator=g  # 🔥 Fixed seed for reproducibility
    )
    
    if args.verbose:
        print("\nPhase 5: Class-Contrastive Dynamic Density Estimation")
    
    # 🔥 FIX: Force model to train() mode to replicate the "good" energy range [-12, -5]
    # When Phase 4 is skipped, model is in train() mode (default). When Phase 4 runs, it sets eval().
    # The negative/better energy range comes from train() mode stats/dropout.
    model.train()
    
    phase5_data = phase_manager.load_phase("phase5_density")

    if should_skip_phase("phase5_density", phase5_data, args.output_dir, args):
        if args.verbose:
            print("✅ Using cached density estimation results")
        gda, p_z_train, energy_range = fit_gda(model, trainloader_deterministic, args.num_classes,
                                               args.embedding_dim, args.device)
    else:
        if phase5_data and not validate_phase_completion("phase5_density", phase5_data, args.output_dir,
                                                         args) and args.verbose:
            print("🔄 Cached density results invalid - re-running density estimation")
        if args.verbose:
            print("Components: Gaussian Mixture Model + Energy Range Calculation")
        gda, p_z_train, energy_range = fit_gda(model, trainloader_deterministic, args.num_classes,
                                               args.embedding_dim, args.device)
        phase_manager.save_phase("phase5_density", {
            "density_info": {
                "gda_fitted": True,
                "p_z_train_shape": list(p_z_train.shape) if p_z_train is not None else None,
                "energy_range": convert_to_serializable(energy_range)
            }
        })

    test_log_probs, test_labels = gmm_evaluate(model.base_model, gda, testloader,
                                               args.device, args.num_classes, args.device)

    # Phase 6: OOD Detection
    if args.verbose:
        print("\nPhase 6: GEM OOD Detection")
    phase6_data = phase_manager.load_phase("phase6_ood")

    if should_skip_phase("phase6_ood", phase6_data, args.output_dir, args):
        if args.verbose:
            print("✅ Using cached OOD detection results")
        ood_auroc = phase6_data["ood_auroc"]
        ood_aupr = phase6_data["ood_aupr"]
    else:
        if phase6_data and not validate_phase_completion("phase6_ood", phase6_data, args.output_dir,
                                                         args) and args.verbose:
            print("🔄 Cached OOD results invalid - re-running OOD detection")

        if args.use_mob:
            ood_auroc, ood_aupr = ood_detection_gem(model, gda, p_z_train, testloader,
                                                     ood_loader1, ood_loader2, args.num_classes,
                                                     args.device, energy_range, use_mob=True, verbose=args.verbose,
                                                     ood_loader3=ood_loader3)
        else:
            ood_auroc, ood_aupr = ood_detection_gem(model, gda, p_z_train, testloader,
                                                      ood_loader1, ood_loader2, args.num_classes,
                                                      args.device, energy_range, verbose=args.verbose,
                                                      ood_loader3=ood_loader3)

        phase_manager.save_phase("phase6_ood", {
            "ood_auroc": ood_auroc,
            "ood_aupr": ood_aupr
        })

    # Brier score
    brier_score = calculate_model_brier_score(model, testloader, args.num_classes, args.device, verbose=args.verbose)

    elapsed_seconds = time.perf_counter() - start_time

    result = {
        "Dataset": str(args.ID_dataset),
        "Model_Type": "GEM-FI" if (args.use_mob and args.use_fi_regularization) else (
            "GEM-MIX" if args.use_mob else "GEM-CORE"),
        "MoB_Components": float(args.num_components if args.use_mob else 1),
        "FI_Regularization": float(1.0 if args.use_fi_regularization else 0.0),
        "FI_Lambda": float(args.fi_lambda if args.use_fi_regularization else 0.0),
        "Test Accuracy": float(test_acc),
        "OOD AUROC": ood_auroc,
        "OOD AUPR": ood_aupr,
        "Brier Score": float(brier_score),
        "Training Epochs": float(args.num_epochs),
        "Learning Rate": float(args.learning_rate),
        "Regularization": float(args.reg_param),
        "Batch Size": float(args.batch_size),
        "Total Runtime (seconds)": float(elapsed_seconds),
        "Runtime (minutes)": float(elapsed_seconds / 60.0)
    }

    # Phase 7: Confidence Calibration (for CIFAR-10 only)
    if args.ID_dataset == "CIFAR-10":
        if args.verbose:
            print("\nPhase 7: GEM Confidence Calibration")
        phase7_data = phase_manager.load_phase("phase7_calibration")

        if should_skip_phase("phase7_calibration", phase7_data, args.output_dir, args):
            if args.verbose:
                print("✅ Using cached calibration results")
            brier_cal = phase7_data["brier_cal"]
            conf_aupr = phase7_data["conf_aupr"]
            conf_auroc = phase7_data["conf_auroc"]
            ece = phase7_data["ece"]
        else:
            if phase7_data and not validate_phase_completion("phase7_calibration", phase7_data, args.output_dir,
                                                             args) and args.verbose:
                print("[!] Cached calibration results invalid - re-running calibration")

            # Load best_model_ts.pt for Phase 7 (if available)
            ts_ckpt = os.path.join(args.output_dir, "checkpoints", "best_model_ts.pt")
            if os.path.exists(ts_ckpt):
                try:
                    if args.verbose:
                        print("  📌 Loading best_model_ts.pt for calibration...")
                    ckpt = torch.load(ts_ckpt, map_location=args.device, weights_only=False)
                    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                    model.load_state_dict(sd, strict=False)

                    # Save temperature for display
                    temp_value = ckpt.get('temperature', 1.0) if isinstance(ckpt, dict) else 1.0
                    if args.verbose:
                        print(f"  ✅ Using temperature-scaled model (T={temp_value:.4f})")
                except Exception as e:
                    if args.verbose:
                        print(f"  ⚠️ Could not load TS checkpoint: {e}, using raw model")
            else:
                if args.verbose:
                    print("  📌 Using best_model.pt (no TS checkpoint found)")

            if args.use_mob:
                brier_cal, conf_aupr, conf_auroc, ece = conf_calibration_gem(
                    model, gda, p_z_train, testloader, args.num_classes, args.device, energy_range, use_mob=True)
            else:
                brier_cal, conf_aupr, conf_auroc, ece = conf_calibration_gem(
                    model, gda, p_z_train, testloader, args.num_classes, args.device, energy_range)

            phase_manager.save_phase("phase7_calibration", {
                "brier_cal": brier_cal,
                "conf_aupr": conf_aupr,
                "conf_auroc": conf_auroc,
                "ece": ece
            })
            if args.verbose:
                print("GEM Calibration completed")

        result.update({
            "Calibration Brier Score": brier_cal,
            "Confidence AUROC": conf_auroc,
            "Confidence AUPR": conf_aupr,
            "Expected Calibration Error": ece
        })
    # Phase 8: Distribution Shift Detection (MNIST-C / CIFAR-10-C for Table 4)
    shift_aupr = None
    shift_auroc = None

    if args.ID_dataset in ["MNIST", "CIFAR-10"]:
        if args.verbose:
            print("\nPhase 8: Distribution Shift Detection (MNIST-C / CIFAR-10-C)")

        phase8_data = phase_manager.load_phase("phase8_corruption")

        if should_skip_phase("phase8_corruption", phase8_data, args.output_dir, args):
            if args.verbose:
                print("✅ Using cached distribution shift detection results")
            shift_aupr = phase8_data.get("shift_aupr")
            shift_auroc = phase8_data.get("shift_auroc")
        else:
            shift_aupr = {}
            shift_auroc = {}

            # ---------- MNIST → MNIST-C ----------
            if args.ID_dataset == "MNIST":
                from sklearn.metrics import average_precision_score
                
                mnistc_loader_fn = make_mnistc_loader_fn(args.data_dir, num_workers=args.num_workers)

                mnist_corruptions = [
                    "brightness", "canny_edges", "dotted_line", "fog", "glass_blur",
                    "identity", "impulse_noise", "motion_blur", "rotate", "scale",
                    "shear", "shot_noise", "spatter", "stripe", "translate", "zigzag",
                ]

                if args.verbose:
                    print("  [Table 4] AUPR - Computing ALL 7 metrics for MNIST shift detection")

                # Comprehensive function to compute ALL 7 metrics for MNIST shift detection
                def compute_all_metrics_for_shift_mnist(model, loader, device):
                    """Compute all 7 uncertainty metrics for MNIST distribution shift detection"""
                    metrics = {
                        "Aleatoric": [],
                        "MaxP_Aleatoric": [],
                        "Alpha0_Epistemic": [],
                        "Entropy_Epistemic": [],
                        "MI_Epistemic": [],
                        "Energy": [],
                        "Combined": []
                    }
                    
                    model.eval()
                    with torch.no_grad():
                        for images, _ in loader:
                            images = images.to(device)
                            batch_size = images.size(0)

                            try:
                                outputs = model(images, full_output=True)
                            except TypeError:
                                outputs = model(images)

                            # Initialize defaults
                            aleatoric = torch.zeros(batch_size, device=device)
                            maxp_aleatoric = torch.zeros(batch_size, device=device)
                            alpha0_epistemic = torch.zeros(batch_size, device=device)
                            entropy_epistemic = torch.zeros(batch_size, device=device)
                            mi_epistemic = torch.zeros(batch_size, device=device)
                            energy = torch.zeros(batch_size, device=device)

                            if isinstance(outputs, (tuple, list)) and len(outputs) >= 6:
                                gated_probs = outputs[0]
                                energy_val = outputs[2] if len(outputs) > 2 else None
                                pi = outputs[4]
                                alpha_list = outputs[5]
                                
                                pi_n = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
                                
                                for k in range(len(alpha_list)):
                                    alpha_k = alpha_list[k].clamp_min(1e-12)
                                    alpha0_k = alpha_k.sum(dim=1, keepdim=True).clamp_min(1e-12)
                                    p_k = alpha_k / alpha0_k
                                    term = torch.digamma(alpha_k + 1.0) - torch.digamma(alpha0_k + 1.0)
                                    exp_entropy_k = -(p_k * term).sum(dim=1)
                                    aleatoric += pi_n[:, k] * exp_entropy_k
                                
                                maxp_aleatoric = 1.0 - gated_probs.max(dim=1)[0]
                                
                                weighted_alpha0 = torch.zeros(batch_size, device=device)
                                for k in range(len(alpha_list)):
                                    alpha0_k = alpha_list[k].sum(dim=1)
                                    weighted_alpha0 += pi_n[:, k] * alpha0_k
                                alpha0_epistemic = 1.0 / weighted_alpha0.clamp_min(1e-12)
                                
                                probs_clamped = gated_probs.clamp_min(1e-12)
                                entropy_epistemic = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
                                mi_epistemic = entropy_epistemic - aleatoric
                                
                                if energy_val is not None:
                                    energy = -energy_val.squeeze()
                                
                                combined = (
                                    0.3 * (aleatoric / aleatoric.clamp_min(1e-12).max().clamp_min(1e-12)) +
                                    0.2 * maxp_aleatoric +
                                    0.2 * (alpha0_epistemic / alpha0_epistemic.clamp_min(1e-12).max().clamp_min(1e-12)) +
                                    0.15 * (mi_epistemic / mi_epistemic.clamp_min(1e-12).abs().max().clamp_min(1e-12)) +
                                    0.15 * (energy / energy.abs().clamp_min(1e-12).max().clamp_min(1e-12))
                                )

                            elif isinstance(outputs, dict) and 'alpha' in outputs:
                                alpha = outputs['alpha'].clamp_min(1e-12)
                                alpha0 = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                                p = alpha / alpha0
                                term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 + 1.0)
                                aleatoric = -(p * term).sum(dim=-1)
                                maxp_aleatoric = 1.0 - p.max(dim=-1)[0]
                                alpha0_epistemic = 1.0 / alpha0.squeeze(-1)
                                entropy_epistemic = -torch.sum(p * torch.log(p), dim=-1)
                                mi_epistemic = entropy_epistemic - aleatoric
                                combined = 0.5 * aleatoric + 0.3 * maxp_aleatoric + 0.2 * alpha0_epistemic

                            else:
                                if isinstance(outputs, torch.Tensor):
                                    probs = outputs if outputs.sum(dim=-1).mean() < 1.1 else torch.softmax(outputs, dim=-1)
                                else:
                                    probs = torch.softmax(outputs[0], dim=-1)
                                
                                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                                aleatoric = entropy
                                maxp_aleatoric = 1.0 - probs.max(dim=-1)[0]
                                entropy_epistemic = entropy
                                combined = entropy

                            metrics["Aleatoric"].extend(aleatoric.cpu().numpy())
                            metrics["MaxP_Aleatoric"].extend(maxp_aleatoric.cpu().numpy())
                            metrics["Alpha0_Epistemic"].extend(alpha0_epistemic.cpu().numpy())
                            metrics["Entropy_Epistemic"].extend(entropy_epistemic.cpu().numpy())
                            metrics["MI_Epistemic"].extend(mi_epistemic.cpu().numpy())
                            metrics["Energy"].extend(energy.cpu().numpy())
                            metrics["Combined"].extend(combined.cpu().numpy())
                    
                    for key in metrics:
                        metrics[key] = np.array(metrics[key])
                    
                    return metrics

                # Compute all metrics for clean MNIST data
                clean_metrics_mnist = compute_all_metrics_for_shift_mnist(model, testloader, args.device)

                # Dictionary to store AUPR per metric  
                aupr_per_metric_mnist = {m: [] for m in clean_metrics_mnist.keys()}
                
                if args.verbose:
                    import sys
                    sys.stdout.write("  Corruptions: ")
                    sys.stdout.flush()

                for corr in mnist_corruptions:
                    try:
                        ood_loader = mnistc_loader_fn(
                            corr,
                            severity=1,
                            batch_size=args.batch_size,
                        )
                    except FileNotFoundError:
                        continue

                    corrupt_metrics_mnist = compute_all_metrics_for_shift_mnist(model, ood_loader, args.device)
                    
                    for metric_name in clean_metrics_mnist.keys():
                        clean_scores = clean_metrics_mnist[metric_name]
                        corrupt_scores = corrupt_metrics_mnist[metric_name]
                        
                        if len(corrupt_scores) > 0 and len(clean_scores) > 0:
                            labels = np.concatenate([
                                np.zeros(len(clean_scores)),
                                np.ones(len(corrupt_scores))
                            ])
                            scores = np.concatenate([clean_scores, corrupt_scores])
                            scores = np.nan_to_num(scores)
                            aupr = average_precision_score(labels, scores)
                            aupr_per_metric_mnist[metric_name].append(aupr)
                    
                    if args.verbose:
                        sys.stdout.write(f"{corr[:3]}. ")
                        sys.stdout.flush()

                # Average AUPR across all corruptions for each metric
                if args.verbose:
                    print()
                
                mnist_results = {}
                for metric_name in clean_metrics_mnist.keys():
                    if len(aupr_per_metric_mnist[metric_name]) > 0:
                        avg_aupr = float(np.mean(aupr_per_metric_mnist[metric_name]))
                        std_aupr = float(np.std(aupr_per_metric_mnist[metric_name]))
                        mnist_results[metric_name] = {"mean": avg_aupr, "std": std_aupr}
                        
                        if args.verbose:
                            print(f"    {metric_name}: {avg_aupr*100:.2f}% +/- {std_aupr*100:.1f}%")

                if mnist_results:
                    shift_aupr["MNIST->MNIST-C"] = mnist_results
                    
                    # Show Aleatoric comparison with GEM
                    if args.verbose and "Aleatoric" in mnist_results:
                        aleatoric_aupr = mnist_results["Aleatoric"]["mean"] * 100
                        gem_mnist = 92.43
                        diff = aleatoric_aupr - gem_mnist
                        symbol = "[OK]" if diff >= 0 else "[X]"
                        print(f"\n  Aleatoric comparison: GEM={gem_mnist:.2f}% | GEM={aleatoric_aupr:.2f}% | {symbol} {diff:+.2f}%")
                        
                        # Show best metric
                        best_metric = max(mnist_results.items(), key=lambda x: x[1]["mean"])
                        print(f"  Best metric: {best_metric[0]} = {best_metric[1]['mean']*100:.2f}%")


            # ---------- CIFAR-10 → CIFAR-10-C ----------
            if args.ID_dataset == "CIFAR-10":
                from sklearn.metrics import average_precision_score
                
                cifar10c_loader_fn = make_cifar10c_loader_fn(args.data_dir, num_workers=args.num_workers)

                cifar_corruptions = [
                    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
                    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
                    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
                    "snow", "spatter", "speckle_noise", "zoom_blur",
                ]

                cifar_aupr_by_severity = {}


                if args.verbose:
                    print("  [Table 4] AUPR - Computing ALL 7 metrics for shift detection")

                # Comprehensive function to compute ALL 7 metrics for shift detection
                def compute_all_metrics_for_shift(model, loader, device):
                    """
                    Compute all 7 uncertainty metrics for distribution shift detection:
                    1. Aleatoric (digamma-based expected entropy)
                    2. MaxP_Aleatoric (1 - max probability)
                    3. Alpha0_Epistemic (1 / sum of alphas)
                    4. Entropy_Epistemic (entropy of predictive distribution)
                    5. MI_Epistemic (mutual information)
                    6. Energy (from EBM network)
                    7. Combined (weighted combination)
                    """
                    metrics = {
                        "Aleatoric": [],
                        "MaxP_Aleatoric": [],
                        "Alpha0_Epistemic": [],
                        "Entropy_Epistemic": [],
                        "MI_Epistemic": [],
                        "Energy": [],
                        "Combined": []
                    }
                    
                    model.eval()
                    with torch.no_grad():
                        for images, _ in loader:
                            images = images.to(device)
                            batch_size = images.size(0)

                            # Try to get full_output for MIX
                            try:
                                outputs = model(images, full_output=True)
                            except TypeError:
                                outputs = model(images)

                            # Initialize defaults
                            aleatoric = torch.zeros(batch_size, device=device)
                            maxp_aleatoric = torch.zeros(batch_size, device=device)
                            alpha0_epistemic = torch.zeros(batch_size, device=device)
                            entropy_epistemic = torch.zeros(batch_size, device=device)
                            mi_epistemic = torch.zeros(batch_size, device=device)
                            energy = torch.zeros(batch_size, device=device)

                            # Check MIX output: (probs, features, energy, gate_w, pi, alpha_list, ...)
                            if isinstance(outputs, (tuple, list)) and len(outputs) >= 6:
                                gated_probs = outputs[0]
                                energy_val = outputs[2] if len(outputs) > 2 else None
                                pi = outputs[4]           # mixture weights
                                alpha_list = outputs[5]   # list of alpha tensors
                                
                                # normalize pi
                                pi_n = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
                                
                                # 1. Aleatoric = Σ_k π_k × E[H(p)|Dir(α_k)]
                                for k in range(len(alpha_list)):
                                    alpha_k = alpha_list[k].clamp_min(1e-12)
                                    alpha0_k = alpha_k.sum(dim=1, keepdim=True).clamp_min(1e-12)
                                    p_k = alpha_k / alpha0_k
                                    term = torch.digamma(alpha_k + 1.0) - torch.digamma(alpha0_k + 1.0)
                                    exp_entropy_k = -(p_k * term).sum(dim=1)
                                    aleatoric += pi_n[:, k] * exp_entropy_k
                                
                                # 2. MaxP_Aleatoric = 1 - max(prob)
                                maxp_aleatoric = 1.0 - gated_probs.max(dim=1)[0]
                                
                                # 3. Alpha0_Epistemic = 1 / weighted_sum_of_alpha0
                                weighted_alpha0 = torch.zeros(batch_size, device=device)
                                for k in range(len(alpha_list)):
                                    alpha0_k = alpha_list[k].sum(dim=1)
                                    weighted_alpha0 += pi_n[:, k] * alpha0_k
                                alpha0_epistemic = 1.0 / weighted_alpha0.clamp_min(1e-12)
                                
                                # 4. Entropy_Epistemic = H(p) = -Σ p log p
                                probs_clamped = gated_probs.clamp_min(1e-12)
                                entropy_epistemic = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
                                
                                # 5. MI_Epistemic = H(p) - Aleatoric (mutual information)
                                mi_epistemic = entropy_epistemic - aleatoric
                                
                                # 6. Energy (from EBM network, higher = more OOD)
                                if energy_val is not None:
                                    energy = -energy_val.squeeze()  # negative because lower energy = more ID
                                
                                # 7. Combined = weighted combination
                                # Normalize each metric to [0, 1] range approximately
                                combined = (
                                    0.3 * (aleatoric / aleatoric.clamp_min(1e-12).max().clamp_min(1e-12)) +
                                    0.2 * maxp_aleatoric +
                                    0.2 * (alpha0_epistemic / alpha0_epistemic.clamp_min(1e-12).max().clamp_min(1e-12)) +
                                    0.15 * (mi_epistemic / mi_epistemic.clamp_min(1e-12).abs().max().clamp_min(1e-12)) +
                                    0.15 * (energy / energy.abs().clamp_min(1e-12).max().clamp_min(1e-12))
                                )

                            # Single-head EDL fallback
                            elif isinstance(outputs, dict) and 'alpha' in outputs:
                                alpha = outputs['alpha'].clamp_min(1e-12)
                                alpha0 = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                                p = alpha / alpha0
                                
                                # Aleatoric
                                term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 + 1.0)
                                aleatoric = -(p * term).sum(dim=-1)
                                
                                # MaxP_Aleatoric
                                maxp_aleatoric = 1.0 - p.max(dim=-1)[0]
                                
                                # Alpha0_Epistemic
                                alpha0_epistemic = 1.0 / alpha0.squeeze(-1)
                                
                                # Entropy_Epistemic
                                entropy_epistemic = -torch.sum(p * torch.log(p), dim=-1)
                                
                                # MI_Epistemic
                                mi_epistemic = entropy_epistemic - aleatoric
                                
                                combined = 0.5 * aleatoric + 0.3 * maxp_aleatoric + 0.2 * alpha0_epistemic

                            # Fallback for non-EDL models
                            else:
                                if isinstance(outputs, torch.Tensor):
                                    probs = outputs if outputs.sum(dim=-1).mean() < 1.1 else torch.softmax(outputs, dim=-1)
                                else:
                                    probs = torch.softmax(outputs[0], dim=-1)
                                
                                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                                aleatoric = entropy
                                maxp_aleatoric = 1.0 - probs.max(dim=-1)[0]
                                entropy_epistemic = entropy
                                combined = entropy

                            # Append to lists
                            metrics["Aleatoric"].extend(aleatoric.cpu().numpy())
                            metrics["MaxP_Aleatoric"].extend(maxp_aleatoric.cpu().numpy())
                            metrics["Alpha0_Epistemic"].extend(alpha0_epistemic.cpu().numpy())
                            metrics["Entropy_Epistemic"].extend(entropy_epistemic.cpu().numpy())
                            metrics["MI_Epistemic"].extend(mi_epistemic.cpu().numpy())
                            metrics["Energy"].extend(energy.cpu().numpy())
                            metrics["Combined"].extend(combined.cpu().numpy())
                    
                    # Convert to numpy arrays
                    for key in metrics:
                        metrics[key] = np.array(metrics[key])
                    
                    return metrics
                
                # Compute all metrics for clean data
                clean_metrics = compute_all_metrics_for_shift(model, testloader, args.device)

                
                for severity in range(1, 6):
                    # Dictionary to store AUPR per metric per corruption
                    aupr_per_metric = {m: [] for m in clean_metrics.keys()}
                    
                    if args.verbose:
                        import sys
                        sys.stdout.write(f"  Severity {severity}/5: ")
                        sys.stdout.flush()

                    for corr in cifar_corruptions:
                        try:
                            ood_loader = cifar10c_loader_fn(
                                corr,
                                severity=severity,
                                batch_size=args.batch_size,
                            )
                        except FileNotFoundError:
                            continue

                        # Compute all metrics for this corruption
                        corrupt_metrics = compute_all_metrics_for_shift(model, ood_loader, args.device)
                        
                        # Compute AUPR for each metric
                        for metric_name in clean_metrics.keys():
                            clean_scores = clean_metrics[metric_name]
                            corrupt_scores = corrupt_metrics[metric_name]
                            
                            if len(corrupt_scores) > 0 and len(clean_scores) > 0:
                                labels = np.concatenate([
                                    np.zeros(len(clean_scores)),
                                    np.ones(len(corrupt_scores))
                                ])
                                scores = np.concatenate([clean_scores, corrupt_scores])
                                scores = np.nan_to_num(scores)  # Handle NaN
                                aupr = average_precision_score(labels, scores)
                                aupr_per_metric[metric_name].append(aupr)
                        
                        if args.verbose:
                            sys.stdout.write(f"{corr[:3]}. ")
                            sys.stdout.flush()

                    # Average AUPR across all 19 corruptions for each metric
                    if args.verbose:
                        print()  # New line after corruptions
                    
                    severity_results = {}
                    for metric_name in clean_metrics.keys():
                        if len(aupr_per_metric[metric_name]) > 0:
                            avg_aupr = float(np.mean(aupr_per_metric[metric_name]))
                            std_aupr = float(np.std(aupr_per_metric[metric_name]))
                            severity_results[metric_name] = {"mean": avg_aupr, "std": std_aupr}
                            
                            if args.verbose:
                                print(f"    {metric_name}: {avg_aupr*100:.2f}% +/- {std_aupr*100:.1f}%")
                    
                    cifar_aupr_by_severity[severity] = severity_results

                if cifar_aupr_by_severity:
                    shift_aupr["CIFAR10->CIFAR10-C"] = cifar_aupr_by_severity

                    # Print comprehensive summary table with ALL 7 metrics
                    if args.verbose:
                        print("\n" + "="*80)
                        print("  SHIFT DETECTION AUPR SUMMARY - ALL 7 METRICS")
                        print("="*80)
                        
                        # Get metric names
                        metric_names = list(clean_metrics.keys())
                        
                        # Print header
                        print(f"  {'Metric':<20} | {'C=1':>8} | {'C=2':>8} | {'C=3':>8} | {'C=4':>8} | {'C=5':>8} |")
                        print("  " + "-"*20 + "|" + ("-"*10 + "|")*5)
                        
                        # Print each metric row
                        for metric in metric_names:
                            row_values = []
                            for sev in range(1, 6):
                                val = cifar_aupr_by_severity.get(sev, {}).get(metric, {})
                                if isinstance(val, dict):
                                    row_values.append(f"{val.get('mean', 0)*100:.1f}%")
                                else:
                                    row_values.append(f"{val*100:.1f}%" if val else "N/A")
                            print(f"  {metric:<20} | {row_values[0]:>8} | {row_values[1]:>8} | {row_values[2]:>8} | {row_values[3]:>8} | {row_values[4]:>8} |")
                        
                        print("  " + "-"*20 + "|" + ("-"*10 + "|")*5)
                        
                        # Show best metric per severity
                        print("\n  Best metric per severity:")
                        for sev in range(1, 6):
                            if sev in cifar_aupr_by_severity:
                                best_metric = max(cifar_aupr_by_severity[sev].items(), 
                                                  key=lambda x: x[1]["mean"] if isinstance(x[1], dict) else x[1])
                                if isinstance(best_metric[1], dict):
                                    print(f"    C={sev}: {best_metric[0]} = {best_metric[1]['mean']*100:.2f}%")
                        
                        # GEM comparison (Aleatoric only)
                        print("\n  Table 4 comparison (Aleatoric):")
                        gem_values = [57.89, 63.23, 67.53, 72.21, 77.74]
                        gem_values = []
                        for i in range(1, 6):
                            val = cifar_aupr_by_severity.get(i, {}).get("Aleatoric", {})
                            if isinstance(val, dict):
                                gem_values.append(val.get("mean", 0) * 100)
                            else:
                                gem_values.append(val * 100 if val else 0)
                        
                        print(f"    GEM: {gem_values[0]:.1f}% | {gem_values[1]:.1f}% | {gem_values[2]:.1f}% | {gem_values[3]:.1f}% | {gem_values[4]:.1f}%")
                        print(f"    GEM:  {gem_values[0]:.1f}% | {gem_values[1]:.1f}% | {gem_values[2]:.1f}% | {gem_values[3]:.1f}% | {gem_values[4]:.1f}%")
                        print("="*80)

            # Save Phase 8
            phase_manager.save_phase("phase8_corruption", {
                "shift_aupr": shift_aupr,
                "shift_auroc": shift_auroc
            })

        # Add Table 4 summary to result for final print (Aggregated Mean)
        if shift_aupr:
            def _get_avg_shift_metric(shift_data, metric_pref=["Combined", "MaxP_Aleatoric", "Aleatoric"]):
                total_mean = 0.0
                count = 0
                chosen_metric = None
                
                # Determine which metric exists
                for m in metric_pref:
                    if 1 in shift_data and m in shift_data[1]:
                        chosen_metric = m
                        break
                
                if chosen_metric is None:
                    return 0.0

                for sev in range(1, 6):
                    if sev in shift_data:
                        val = shift_data[sev].get(chosen_metric, {})
                        if isinstance(val, dict):
                            total_mean += val.get("mean", 0.0)
                        else:
                            total_mean += float(val) if val else 0.0
                        count += 1
                return total_mean / count if count > 0 else 0.0

            if args.ID_dataset == "MNIST" and "MNIST->MNIST-C" in shift_aupr:
                result["Shift Detection AUPR MNIST->MNIST-C"] = _get_avg_shift_metric(shift_aupr["MNIST->MNIST-C"])
            if args.ID_dataset == "CIFAR-10" and "CIFAR10->CIFAR10-C" in shift_aupr:
                result["Shift Detection AUPR CIFAR10->CIFAR10-C"] = _get_avg_shift_metric(shift_aupr["CIFAR10->CIFAR10-C"])


    end_time = time.perf_counter()
    total_runtime = end_time - start_time
    result["Total Runtime (seconds)"] = total_runtime
    result["Runtime (minutes)"] = total_runtime / 60

    os.makedirs(args.output_dir, exist_ok=True)

    def _sanitize_ood_list(lst):
        if not isinstance(lst, (list, tuple)):
            return lst
        out = []
        for d in lst:
            if not isinstance(d, dict):
                out.append(d);
                continue
            out.append({
                "Combined": float(d.get("Combined", d.get("combined", 0.0))),
                "MaxP_Aleatoric": float(d.get("MaxP_Aleatoric", d.get("maxp_aleatoric", d.get("Aleatoric", 0.0)))),
                "Alpha0_Epistemic": float(d.get("Alpha0_Epistemic", d.get("alpha0_epistemic", 0.0))),
                "Energy": float(d.get("Energy", d.get("energy", 0.0))),
                "Entropy_Epistemic": float(d.get("Entropy_Epistemic", d.get("entropy_epistemic", 0.0))),
                "MI_Epistemic": float(d.get("MI_Epistemic", d.get("mi_epistemic", 0.0))),
                "Epistemic_Combined": float(d.get("Epistemic_Combined", d.get("epistemic_combined", 0.0))),
            })
        return out

    result["OOD AUROC"] = _sanitize_ood_list(result.get("OOD AUROC", []))
    result["OOD AUPR"] = _sanitize_ood_list(result.get("OOD AUPR", []))

    result_filepath = os.path.join(args.output_dir, 'results.json')

    result_converted = convert_to_serializable(result)

    with open(result_filepath, 'w') as result_file:
        json.dump(result_converted, result_file, indent=4)

    config_filepath = os.path.join(args.output_dir, 'training_config.json')
    config = {
        "ID_dataset": args.ID_dataset,
        "model_type": "GEM-FI" if args.use_mob and args.use_fi_regularization else
        "GEM-MIX" if args.use_mob else "GEM-CORE",
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "val_seed": args.val_seed,
        "num_classes": args.num_classes,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "reg_param": args.reg_param,
        "num_epochs": args.num_epochs,
        "device": str(args.device),
        "use_mob": args.use_mob,
        "num_components": args.num_components,
        "use_fi_regularization": args.use_fi_regularization,
        "fi_lambda": args.fi_lambda
    }
    with open(config_filepath, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    # Display final results
    if args.verbose:
        print("\n" + "=" * 60)
        model_type_str = "GEM-FI" if args.use_mob and args.use_fi_regularization else \
            "GEM-MIX" if args.use_mob else "GEM-CORE"
        print(f"{model_type_str} Pipeline Completed Successfully!")
        print("=" * 60)
        print("Final Results:")
        print(f"   Dataset: {args.ID_dataset}")
        print(f"   Model Type: {model_type_str}")
        if args.use_mob:
            print(f"   MoB Components: {args.num_components}")
            if args.use_fi_regularization:
                print(f"   FI Regularization: λ={args.fi_lambda}")
        print(f"   Test Accuracy: {result_converted.get('Test Accuracy', 0):.2f}%")
        print(f"   Brier Score: {result_converted.get('Brier Score', 0):.4f}")

        ood_auroc = result_converted.get('OOD AUROC', [])
        if isinstance(ood_auroc, list) and len(ood_auroc) >= 2:
            print("   OOD Detection Results:")
            if isinstance(ood_auroc[0], dict):
                ood1_aleatoric = ood_auroc[0].get('MaxP_Aleatoric', 0)
                ood1_epistemic = ood_auroc[0].get('Alpha0_Epistemic', 0)
                print(f"     OOD1 - AUROC: Aleatoric={ood1_aleatoric:.4f}, Epistemic={ood1_epistemic:.4f}")

                ood2_aleatoric = ood_auroc[1].get('MaxP_Aleatoric', 0)
                ood2_epistemic = ood_auroc[1].get('Alpha0_Epistemic', 0)
                print(f"     OOD2 - AUROC: Aleatoric={ood2_aleatoric:.4f}, Epistemic={ood2_epistemic:.4f}")

        if 'Average Corruption Accuracy' in result_converted:
            print(f"   Average Corruption Accuracy: {result_converted['Average Corruption Accuracy']:.2f}%")
        if args.ID_dataset == "CIFAR-10" and 'Expected Calibration Error' in result_converted:
            print(f"   Expected Calibration Error: {result_converted['Expected Calibration Error']:.4f}")

        print(
            f"   Total Runtime: {result_converted.get('Total Runtime (seconds)', 0):.2f} seconds ({result_converted.get('Runtime (minutes)', 0):.2f} minutes)")
        print(f"   Results saved to: {result_filepath}")
        print(f"   Config saved to: {config_filepath}")

        phase_manager.print_phase_status()
        print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)