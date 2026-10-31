"""
Module: train_sota_few_shot.py (Final Corrected Edition)
Description:
    Trains the student Transformer under 8 ablation configurations:
        - Baseline / Gradual_Unfreezing / SIPAR / CAFT
        - with or without MAML prior
    Implements LLRD, SIPAR regularization, phase-wise training, and representation audits.

Fixes / Design:
    1. Correct prior loading with soft interpolation (alpha configurable).
    2. LLRD coefficients aligned with has_prior flag.
    3. Unified SIPAR regularizer (align + var + cov) across phases.
    4. Repr-Analyzer erank / avg_cos audit at each phase.
    5. Phase 1 / Phase 2 optimizer grouping strictly aligned.
    6. Early stopping, best-checkpoint save, convergence tracking.
    7. Full signature + config dict for 8 ablation settings.
    8. Type hints and boundary checks.
"""

import os
import copy
import math
import random
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_pretrain_full_strong import NeuralTransformerStudent
from representation_analyzer import RepresentationAnalyzer


# =============================================================================
# 1. PRIOR LOADING (SOFT INTERPOLATION)
# =============================================================================

def load_prior_weights_soft(
    model: nn.Module,
    prior_path: str,
    device: torch.device,
    alpha: float = 0.4,
    noise_std: float = 0.005,
    skip_keys: Optional[List[str]] = None,
) -> int:
    """
    Softly interpolate prior weights with initialized weights:
        W = alpha * W_prior + (1 - alpha) * W_init

    This avoids collapsing into a hard low-rank subspace and preserves
    manifold elasticity for downstream adaptation.

    Args:
        model: Student model.
        prior_path: Path to Reptile-pretrained weights.
        device: Torch device.
        alpha: Interpolation coefficient in [0, 1].
        noise_std: Std of Gaussian noise added to weight matrices.
        skip_keys: Substrings to skip (e.g., embedding, fc).

    Returns:
        Number of loaded parameters.
    """
    if not os.path.exists(prior_path):
        print(f"[Prior Loader] Checkpoint not found: {prior_path}. Initializing randomly.")
        return 0

    if skip_keys is None:
        skip_keys = ['embedding', 'wte', 'shared', 'fc', 'classifier']

    checkpoint = torch.load(prior_path, map_location=device)
    state_dict = model.state_dict()
    loaded_count = 0

    for k, v in checkpoint.items():
        if any(s in k.lower() for s in skip_keys):
            continue
        if k not in state_dict:
            continue

        prior_weight = v.clone().to(device)
        init_weight = state_dict[k].clone().to(device)

        # Soft interpolation
        mixed_weight = alpha * prior_weight + (1.0 - alpha) * init_weight

        # Add noise to weight matrices (not norms/biases)
        if 'weight' in k.lower() and not any(x in k.lower() for x in ['norm', 'ln']):
            noise = torch.randn_like(mixed_weight) * noise_std
            mixed_weight = mixed_weight + noise

        state_dict[k].copy_(mixed_weight)
        loaded_count += 1

    model.load_state_dict(state_dict)
    print(f"[Prior Loader] Soft-loaded {loaded_count} params (alpha={alpha}, noise={noise_std}).")
    return loaded_count


# =============================================================================
# 2. SIPAR REGULARIZER
# =============================================================================

def compute_sipar_loss(
    f_norm: torch.Tensor,
    f_scram: torch.Tensor,
    align_weight: float = 0.3,
    var_weight: float = 0.05,
    cov_weight: float = 0.02,
    decay_ratio: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SIPAR: Structural Invariance via Perturbation-Aligned Regularization.

    Components:
        1. Alignment: cosine similarity between normalized and scrambled representations.
        2. Variance: variance hinge loss encouraging spread.
        3. Covariance: off-diagonal covariance penalty.

    Args:
        f_norm: [B, D] representations under normal input.
        f_scram: [B, D] representations under scrambled input.
        align_weight, var_weight, cov_weight: loss weights.
        decay_ratio: annealing factor in [0, 1].

    Returns:
        (total_loss, components_dict)
    """
    components = {}

    # Center representations
    f_norm_c = f_norm - f_norm.mean(dim=0, keepdim=True)
    f_scram_c = f_scram - f_scram.mean(dim=0, keepdim=True)

    # 1. Alignment loss
    if f_norm_c.size(0) > 1 and f_scram_c.size(0) > 1:
        align_loss = 1.0 - F.cosine_similarity(f_norm_c, f_scram_c, dim=-1).mean()
    else:
        align_loss = torch.tensor(0.0, device=f_norm.device)
    components['align'] = float(align_loss.item())

    # 2. Variance hinge loss
    var_loss = torch.tensor(0.0, device=f_norm.device)
    if f_norm_c.size(0) > 1:
        std_f = torch.sqrt(f_norm_c.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std_f))
    components['var'] = float(var_loss.item())

    # 3. Covariance off-diagonal penalty
    cov_loss = torch.tensor(0.0, device=f_norm.device)
    if f_norm_c.size(0) > 1:
        cov = (f_norm_c.T @ f_norm_c) / (f_norm_c.size(0) - 1)
        diag = torch.diagonal(cov)
        cov_loss = (cov - torch.diag(diag)).pow(2).mean()
    components['cov'] = float(cov_loss.item())

    total = (
        align_weight * align_loss
        + (var_weight * var_loss + cov_weight * cov_loss) * decay_ratio
    )
    components['total'] = float(total.item())
    return total, components


def extract_mask_representations(
    hidden_states: Any,
    batch_tasks: List[Dict[str, Any]],
) -> torch.Tensor:
    """
    Extract [MASK] position representations from hidden states.
    """
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    if hidden_states.dim() == 3:
        reps = []
        for idx, task in enumerate(batch_tasks):
            m_idx = task.get('mask_idx', 0)
            if m_idx < 0 or m_idx >= hidden_states.shape[1]:
                m_idx = 0
            reps.append(hidden_states[idx, m_idx])
        return torch.stack(reps)
    return hidden_states


# =============================================================================
# 3. TRAINING CONFIG
# =============================================================================

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Baseline": {
        "use_gradual_unfreezing": False,
        "use_sipar": False,
        "use_llrd": False,
    },
    "Gradual_Unfreezing_Only": {
        "use_gradual_unfreezing": True,
        "use_sipar": False,
        "use_llrd": False,
    },
    "SIPAR_Only": {
        "use_gradual_unfreezing": False,
        "use_sipar": True,
        "use_llrd": False,
    },
    "CAFT": {
        "use_gradual_unfreezing": True,
        "use_sipar": True,
        "use_llrd": True,
    },
}


def get_config(config_type: str, has_prior: bool) -> Dict[str, Any]:
    """
    Get training config for a given type and prior flag.
    """
    base = ABLATION_CONFIGS.get(config_type, ABLATION_CONFIGS["Baseline"]).copy()
    base['has_prior'] = has_prior

    # LLRD coefficients
    if base['use_llrd']:
        if has_prior:
            base['llrd_layer0'] = 0.65
            base['llrd_layer1'] = 0.85
        else:
            base['llrd_layer0'] = 0.55
            base['llrd_layer1'] = 0.80

    return base


# =============================================================================
# 4. TRAIN FUNCTION
# =============================================================================

def train_sota_few_shot(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    processor: Any,
    train_t: List[Dict[str, Any]],
    is_ours: bool = False,
    has_prior: bool = False,
    device: torch.device = torch.device('cpu'),
    config_type: str = "Baseline",
    phase1_epochs: int = 20,
    phase2_epochs: int = 60,
    batch_size: int = 8,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-3,
    lr_encoder: float = 2e-4,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    grad_clip: float = 0.5,
    patience: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the student Transformer under a given ablation config.

    Returns:
        {
            'model': trained model,
            'metrics': { 'step0': {...}, 'end_phase1': {...}, 'end_phase2': {...} },
            'loss_curve': [...],
        }
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # === PATCH D: fail fast on empty training set ===
    assert len(X) > 0, (
        f"Empty training set (len(X)=0). "
        f"Check split_three_way: train_t may be empty."
    )

    config = get_config(config_type, has_prior)
    use_grad_unfreeze = config.get('use_gradual_unfreezing', False)
    use_sipar = config.get('use_sipar', False)
    use_llrd = config.get('use_llrd', False)

    # Locate encoder
    encoder_name = ""
    encoder_module = None
    for name, child in model.named_children():
        if 'encoder' in name.lower() or 'transformer' in name.lower():
            encoder_name = name
            encoder_module = child
            break
    if encoder_module is None:
        encoder_module = model

    if verbose:
        print(f"\n>>> Config: [{config_type}] | has_prior={has_prior} | "
              f"unfreeze={use_grad_unfreeze} | sipar={use_sipar} | llrd={use_llrd}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    dataset_size = len(X)
    metrics = {}
    loss_curve: List[float] = []

    # ---- Step 0 metrics ----
    analyzer = RepresentationAnalyzer(model)
    try:
        init_features = analyzer.collect_representations(X, device=device)
        init_erank, init_cos = analyzer.calculate_metrics(init_features)
        metrics['step0'] = {'erank': init_erank, 'avg_cos': init_cos}
        if verbose:
            print(f"[REP-METRICS] Step 0 | erank={init_erank:.4f} | cos={init_cos:.4f}")
    except Exception as e:
        if verbose:
            print(f"[WARN] Step 0 repr metrics failed: {e}")

    # ==========================================================
    # PHASE 1
    # ==========================================================
    model.train()
    handle = None
    feats: Dict[str, torch.Tensor] = {}

    if use_grad_unfreeze and config_type in ["CAFT", "Gradual_Unfreezing_Only"]:
        # BitFit warmup: only biases, norms, embeddings
        if has_prior:
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ['bias', 'norm', 'embedding', 'wte']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in encoder_module.parameters():
                param.requires_grad = False
        for name, param in model.named_parameters():
            if encoder_name not in name:
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt_phase1 = torch.optim.AdamW(trainable, lr=lr_phase1)
    else:
        for param in model.parameters():
            param.requires_grad = True
        opt_phase1 = torch.optim.AdamW([
            {'params': encoder_module.parameters(), 'lr': lr_encoder, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if encoder_name not in n],
             'lr': lr_phase1},
        ])

    # Register hook if SIPAR
    if use_sipar:
        def hook_fn(m, inp, out):
            feats['out'] = out[0] if isinstance(out, tuple) else out
        handle = encoder_module.register_forward_hook(hook_fn)

    for epoch in range(phase1_epochs):
        idx = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = idx[i: i + batch_size]
            bX, by = X[indices], y[indices]

            opt_phase1.zero_grad()
            logits = model(bX)
            loss = criterion(logits, by)

            if use_sipar and len(indices) > 1 and feats.get('out') is not None:
                batch_tasks = [train_t[idx_item.item()] for idx_item in indices]
                f_norm = extract_mask_representations(feats['out'], batch_tasks)

                bX_scram, _, _ = processor.encode(batch_tasks, scramble_type='modifier')
                _ = model(bX_scram.to(device))
                f_scram = extract_mask_representations(feats['out'], batch_tasks)

                decay_ratio = max(0.1, 1.0 - (epoch / max(phase1_epochs, 1)))
                sipar_loss, comps = compute_sipar_loss(f_norm, f_scram, decay_ratio=decay_ratio)
                loss = loss + sipar_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt_phase1.step()

        # === PATCH D: only append when loss is defined ===
        if dataset_size > 0:
            loss_curve.append(loss.item())

    if handle is not None:
        handle.remove()
        feats.clear()

    # ---- End of Phase 1 metrics ----
    try:
        p1_features = analyzer.collect_representations(X, device=device)
        p1_erank, p1_cos = analyzer.calculate_metrics(p1_features)
        metrics['end_phase1'] = {'erank': p1_erank, 'avg_cos': p1_cos}
        if verbose:
            print(f"[REP-METRICS] End Phase 1 | erank={p1_erank:.4f} | cos={p1_cos:.4f}")
    except Exception as e:
        if verbose:
            print(f"[WARN] End Phase 1 repr metrics failed: {e}")

    # ==========================================================
    # PHASE 2
    # ==========================================================
    for param in model.parameters():
        param.requires_grad = True

    if use_llrd:
        params_group = []
        for name, param in model.named_parameters():
            if encoder_name in name:
                if any(x in name.lower() for x in ['layer.0', 'embedding', 'wte']):
                    decay = config.get('llrd_layer0', 0.55)
                    params_group.append({'params': [param], 'lr': lr_encoder * decay,
                                         'weight_decay': 0.1})
                elif 'layer.1' in name.lower():
                    decay = config.get('llrd_layer1', 0.80)
                    params_group.append({'params': [param], 'lr': lr_encoder * decay,
                                         'weight_decay': weight_decay})
                else:
                    params_group.append({'params': [param], 'lr': lr_encoder,
                                         'weight_decay': weight_decay})
            else:
                params_group.append({'params': [param], 'lr': lr_phase2, 'weight_decay': 0.01})
        opt_phase2 = torch.optim.AdamW(params_group)
    else:
        opt_phase2 = torch.optim.AdamW([
            {'params': encoder_module.parameters(), 'lr': lr_encoder, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if encoder_name not in n],
             'lr': lr_phase2},
        ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase2, T_max=phase2_epochs, eta_min=1e-6)

    feats = {}
    handle = None
    if use_sipar:
        def hook_fn2(m, inp, out):
            feats['out'] = out[0] if isinstance(out, tuple) else out
        handle = encoder_module.register_forward_hook(hook_fn2)

    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    converged = False

    for epoch in range(phase2_epochs):
        idx = torch.randperm(dataset_size)
        epoch_loss = 0.0

        for i in range(0, dataset_size, batch_size):
            indices = idx[i: i + batch_size]
            bX, by = X[indices], y[indices]

            opt_phase2.zero_grad()
            logits = model(bX)
            loss = criterion(logits, by)

            if use_sipar and len(indices) > 1 and feats.get('out') is not None:
                batch_tasks = [train_t[idx_item.item()] for idx_item in indices]
                f_norm = extract_mask_representations(feats['out'], batch_tasks)

                bX_scram, _, _ = processor.encode(batch_tasks, scramble_type='modifier')
                _ = model(bX_scram.to(device))
                f_scram = extract_mask_representations(feats['out'], batch_tasks)

                decay_ratio = max(0.1, 1.0 - (epoch / max(phase2_epochs, 1)))
                sipar_loss, _ = compute_sipar_loss(f_norm, f_scram, decay_ratio=decay_ratio)
                loss = loss + sipar_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt_phase2.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= max(dataset_size / batch_size, 1)
        loss_curve.append(epoch_loss)

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                converged = True
                if verbose:
                    print(f"[EARLY-STOP] Phase 2 stopped at epoch {epoch}")
                break

    if handle is not None:
        handle.remove()

    # Restore best
    model.load_state_dict(best_state)

    # ---- End of Phase 2 metrics ----
    try:
        p2_features = analyzer.collect_representations(X, device=device)
        p2_erank, p2_cos = analyzer.calculate_metrics(p2_features)
        metrics['end_phase2'] = {'erank': p2_erank, 'avg_cos': p2_cos}
        if verbose:
            print(f"[REP-METRICS] End Phase 2 | erank={p2_erank:.4f} | cos={p2_cos:.4f}")
    except Exception as e:
        if verbose:
            print(f"[WARN] End Phase 2 repr metrics failed: {e}")

    return {
        'model': model,
        'metrics': metrics,
        'loss_curve': loss_curve,
        'converged': converged,
    }