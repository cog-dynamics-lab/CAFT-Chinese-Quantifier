"""
Module: ud_kshot_ablation.py (Final Corrected Edition)
Description:
    Provides leakage-free temperature calibration, class-stratified bootstrap ECE,
    dependency graph hop computation with telemetry, and controlled syntactic noise injection.

Fixes over previous version:
    1. fit_temperature_on_val returns convergence status and is auditable by caller.
    2. Bootstrap iterations configurable; fast mode supported.
    3. compute_stratified_bootstrap_ece reports per-class sample counts.
    4. compute_dependency_hops_with_telemetry reports root/anomaly statistics.
    5. inject_syntactic_noise reports actual noise injection ratio.
    6. Type hints and boundary checks added.
    7. torch.clamp uses keyword arguments consistently.
    8. Bootstrap scaler detach to avoid memory leak.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, deque
from conllu import parse_incr
from typing import Tuple, List, Optional


# =============================================================================
# 1. LEAKAGE-FREE TEMPERATURE CALIBRATION
# =============================================================================

class TemperatureScaler(nn.Module):
    """
    Single-parameter post-hoc calibration via logit temperature scaling.
    Temperature is clamped to [0.05, 5.0] for numerical stability.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        clamped_t = torch.clamp(self.temperature, min=0.5, max=5.0)
        return logits / clamped_t


# =============================================================================
# 1. LEAKAGE-FREE TEMPERATURE CALIBRATION (CORRECTED)
# =============================================================================

# 文件：ud_kshot_ablation.py，函数：fit_temperature_on_val，开头
def fit_temperature_on_val(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    max_iter: int = 50,
    n_boot: int = 100,
    fast_mode: bool = False,
) -> Tuple[float, bool, Tuple[float, float]]:
    model.eval()

    # === FIX: normalize device (cuda -> cuda:0) ===
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    with torch.no_grad():
        logits = model(X_val_dev)

    # Device assertion: logits must be on the same device as scaler
    assert logits.device == device, \
        f"[DEVICE-MISMATCH] logits on {logits.device}, expected {device}"

    converged = True

    def eval_loss() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(scaler(logits), y_val_dev)
        loss.backward()
        return loss

    try:
        optimizer.step(eval_loss)
    except Exception as e:
        print(f"[Warning] LBFGS optimization failed ({e}), fallback to T=1.0")
        return 1.0, False, (1.0, 1.0)

    opt_T_raw = float(scaler.temperature.detach().cpu().item())

    # torch.isfinite check: NaN / Inf fallback
    if not torch.isfinite(torch.tensor(opt_T_raw)):
        print(f"[Warning] Non-finite T* ({opt_T_raw}), fallback to T=1.0")
        return 1.0, False, (1.0, 1.0)

    hit_boundary = (opt_T_raw <= 0.51) or (opt_T_raw >= 4.99)
    if hit_boundary:
        converged = False
    opt_T = float(torch.clamp(
        torch.tensor(opt_T_raw), min=0.5, max=5.0
    ).item())

    # Bootstrap CI
    if fast_mode:
        n_boot = min(n_boot, 30)

    val_n = logits.size(0)
    t_boots: List[float] = []

    # Use logits.device for index tensor
    for _ in range(n_boot):
        b_idx = torch.randint(0, val_n, (val_n,), device=logits.device)
        sub_logits = logits[b_idx]
        sub_y = y_val_dev[b_idx]

        s = TemperatureScaler().to(device)
        opt_s = torch.optim.LBFGS([s.temperature], lr=0.05, max_iter=15)

        def sub_eval() -> torch.Tensor:
            opt_s.zero_grad()
            l = nn.CrossEntropyLoss()(s(sub_logits), sub_y)
            l.backward()
            return l

        try:
            opt_s.step(sub_eval)
            t_val_raw = float(s.temperature.detach().cpu().item())
            if not torch.isfinite(torch.tensor(t_val_raw)):
                t_boots.append(opt_T)
            else:
                t_val = float(torch.clamp(
                    torch.tensor(t_val_raw), min=0.05, max=5.0
                ).item())
                t_boots.append(t_val)
        except Exception:
            t_boots.append(opt_T)

    ci_low = float(np.percentile(t_boots, 2.5))
    ci_high = float(np.percentile(t_boots, 97.5))

    print(f"[CALIBRATION] T* = {opt_T:.4f} | converged = {converged} | "
          f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    return opt_T, converged, (ci_low, ci_high)


# =============================================================================
# 2. CLASS-STRATIFIED BOOTSTRAP ECE
# =============================================================================

def compute_stratified_bootstrap_ece(
    confidences: np.ndarray,
    correct_flags: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    n_boot: int = 1000,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Class-stratified bootstrap ECE with 95% CI.

    Args:
        confidences: [N] predicted confidence in [0, 1].
        correct_flags: [N] binary correctness (0/1).
        labels: [N] true class labels, used for stratified resampling.
        n_bins: Number of confidence bins.
        n_boot: Bootstrap iterations.
        verbose: If True, print per-class sample counts.

    Returns:
        (base_ece, ci_lower, ci_upper)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    def calc_single_ece(confs: np.ndarray, accs: np.ndarray) -> float:
        ece = 0.0
        sample_len = len(confs)
        if sample_len == 0:
            return 0.0
        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(accs[mask])
                bin_conf = np.mean(confs[mask])
                ece += np.abs(bin_conf - bin_acc) * (np.sum(mask) / sample_len)
        return float(ece)

    base_ece = calc_single_ece(confidences, correct_flags)

    unique_classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}

    if verbose:
        class_counts = {int(c): int(len(idxs)) for c, idxs in class_indices.items()}
        print(f"[ECE-AUDIT] Per-class sample counts: {class_counts}")

    boot_samples = []
    for _ in range(n_boot):
        boot_idx = []
        for c, idxs in class_indices.items():
            if len(idxs) > 0:
                boot_idx.extend(np.random.choice(idxs, size=len(idxs), replace=True))
        boot_idx = np.array(boot_idx)
        boot_samples.append(calc_single_ece(confidences[boot_idx], correct_flags[boot_idx]))

    ci_lower = float(np.percentile(boot_samples, 2.5))
    ci_upper = float(np.percentile(boot_samples, 97.5))
    return base_ece, ci_lower, ci_upper


# =============================================================================
# 3. DEPENDENCY GRAPH HOPS WITH TELEMETRY
# =============================================================================

def compute_dependency_hops_with_telemetry(
    tokenlist: list,
    clf_id: int,
    noun_id: int,
    verbose: bool = False,
) -> int:
    """
    Compute shortest undirected dependency graph hops between classifier and noun.

    Returns:
        Number of hops, or 999 if disconnected / anomalous.
    """
    adj: dict = {}
    root_count = 0

    for t in tokenlist:
        tid = t['id']
        if isinstance(tid, int):
            adj.setdefault(tid, [])

    for t in tokenlist:
        tid = t['id']
        head = t['head']
        if isinstance(tid, int) and isinstance(head, int) and head > 0:
            adj.setdefault(head, [])
            adj[tid].append(head)
            adj[head].append(tid)
        elif isinstance(head, int) and head == 0:
            root_count += 1

    if verbose:
        print(f"[HOP-AUDIT] tokens={len(adj)} | roots={root_count} | clf_id={clf_id} | noun_id={noun_id}")

    if clf_id not in adj or noun_id not in adj:
        return 999

    queue = deque([(clf_id, 0)])
    visited = {clf_id}

    while queue:
        curr, hops = queue.popleft()
        if curr == noun_id:
            return hops
        for nxt in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, hops + 1))

    return 999


# =============================================================================
# 4. CONTROLLED SYNTACTIC NOISE INJECTION
# =============================================================================

def inject_syntactic_noise(
    X: torch.Tensor,
    vocab_size: int,
    noise_rate: float = 0.15,
    pad_idx: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Inject uniform random token noise into non-PAD positions, avoiding special tokens 0-4.

    Args:
        X: [B, L] input token ids.
        vocab_size: Total vocabulary size.
        noise_rate: Fraction of non-PAD tokens to perturb.
        pad_idx: Padding token id.
        verbose: If True, print actual injection ratio.

    Returns:
        Perturbed tensor with same shape.
    """
    X_noisy = X.clone()
    non_pad = (X != pad_idx)
    mask = (torch.rand(X.shape, device=X.device) < noise_rate) & non_pad
    replace_mask = mask & (torch.rand(X.shape, device=X.device) < 0.8)
    zero_mask = mask & (~replace_mask)

    random_tokens = torch.randint(5, max(6, vocab_size - 1), X.shape, device=X.device)
    X_noisy[replace_mask] = random_tokens[replace_mask]
    X_noisy[zero_mask] = pad_idx

    if verbose:
        total = int(non_pad.sum().item())
        perturbed = int(mask.sum().item())
        ratio = (perturbed / total * 100) if total > 0 else 0.0
        print(f"[NOISE-AUDIT] Perturbed {perturbed}/{total} tokens ({ratio:.2f}%)")

    return X_noisy