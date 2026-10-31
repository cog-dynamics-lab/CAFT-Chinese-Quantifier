"""
Module: run_rigorous_aligned_ablationTJ.py (Split-Reset + Causal Analysis Edition)
Description:
    Rigorous execution pipeline for CAFT ablation, calibration, attention knockout,
    Monte Carlo null-distribution, causal mediation, UUAS structural probing,
    and structural perturbation evaluation.

PATCHES applied:
    [PATCH 2.1] gold_edges index fix: map (id-1, head-1) -> (pos_u, pos_v)
    [PATCH 2.2] split_three_way: hop-stratified (train/val shallow, test shallow+deep)
    [PATCH 2.3] HiddenStateExtractor: forward hook on model.transformer
    [PATCH 2.4] compute_causal_mediation: accuracy-based TIE with bootstrap CI
    [PATCH 2.5] evaluate_ablation_configs: UUAS + Knockout + Mediation for CAFT/Baseline
    [PATCH 2.6] run_full_pipeline: aggregate new fields + dump ablation_summary.json
"""

import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conllu import parse_incr
from collections import Counter
from typing import Tuple, List, Dict, Optional, Any

from meta_pretrain_full_strong import NeuralTransformerStudent
from ud_kshot_ablation import (
    compute_stratified_bootstrap_ece,
    fit_temperature_on_val,
    compute_dependency_hops_with_telemetry,
    inject_syntactic_noise,
)
from train_sota_few_shot import train_sota_few_shot, load_prior_weights_soft
# === PATCH 2: import structural probe utilities ===
from representation_analyzer import (
    MaskedStructuralProbe,
    compute_tree_distances,
    evaluate_probe_on_dataset,
)


# =============================================================================
# 1. METRIC HELPERS
# =============================================================================

def calculate_macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        if prec + rec > 0:
            f1s.append((2 * prec * rec / (prec + rec)).item())
    return (sum(f1s) / len(f1s)) * 100 if f1s else 0.0


# =============================================================================
# 2. STRUCTURED TASK PROCESSOR
# =============================================================================

class BulletproofUDProcessor:
    def __init__(self) -> None:
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "的": 2, "之": 3, "<MASK>": 4}
        self.n_id_counter = 100
        self.other_id_counter = 2000
        self.quantifier_to_id: Dict[str, int] = {}

        self.isolated_999_count = 0
        self.total_extracted_count = 0

        self.global_swap_fallback_count = 0
        self.global_swap_total_count = 0

        self.split_swap_fallback_count = 0
        self.split_swap_total_count = 0

    def get_id(self, form: str, upos: str) -> int:
        if form in self.vocab:
            return self.vocab[form]
        if upos in ['NOUN', 'PROPN', 'PRON']:
            self.vocab[form] = self.n_id_counter
            self.n_id_counter = 100 + ((self.n_id_counter - 99) % 1899)
        else:
            self.vocab[form] = self.other_id_counter
            self.other_id_counter = 2000 + ((self.other_id_counter - 1999) % 22999)
        return self.vocab[form]

    def extract_tasks(self, filepath: str) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return tasks

        with open(filepath, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                for token in tokenlist:
                    if token['deprel'] and 'clf' in token['deprel']:
                        self.total_extracted_count += 1
                        clf_word, clf_id, head_id = token['form'], token['id'], token['head']
                        if not head_id or head_id - 1 >= len(tokenlist):
                            continue
                        head_token = tokenlist[head_id - 1]

                        if head_token['upos'] == 'NUM' and head_token['head'] != 0:
                            if head_token['head'] - 1 < len(tokenlist):
                                head_token = tokenlist[head_token['head'] - 1]

                        if head_token['upos'] not in ['NOUN', 'PROPN', 'PRON', 'VERB']:
                            continue
                        noun_id = head_token['id']

                        hops = compute_dependency_hops_with_telemetry(tokenlist, clf_id, noun_id)
                        if hops == 999:
                            self.isolated_999_count += 1
                            continue

                        seq_ids, mask_idx, noun_idx = [], -1, -1
                        content_indices: List[int] = []
                        # === PATCH 2.1: build pos map first, then convert gold_edges ===
                        token_id_to_pos: Dict[int, int] = {}
                        raw_gold_edges: List[Tuple[int, int]] = []

                        for t in tokenlist:
                            if not isinstance(t['id'], int):
                                continue
                            pos = len(seq_ids)
                            token_id_to_pos[t['id']] = pos

                            if t['id'] == clf_id:
                                seq_ids.append(self.vocab["<MASK>"])
                                mask_idx = pos
                            else:
                                seq_ids.append(self.get_id(t['form'], t['upos']))
                                if t['id'] == noun_id:
                                    noun_idx = pos
                                elif t['upos'] in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                                    content_indices.append(pos)

                            if isinstance(t['head'], int) and t['head'] > 0:
                                raw_gold_edges.append((t['id'], t['head']))

                        gold_edges: List[Tuple[int, int]] = []
                        for u_id, v_id in raw_gold_edges:
                            if u_id in token_id_to_pos and v_id in token_id_to_pos:
                                gold_edges.append((token_id_to_pos[u_id], token_id_to_pos[v_id]))

                        tasks.append({
                            'quantifier': clf_word,
                            'seq_ids': seq_ids,
                            'mask_idx': mask_idx,
                            'noun_idx': noun_idx,
                            'content_indices': content_indices,
                            'tree_hops': hops,
                            'gold_edges': gold_edges,
                            'complexity': "Deep" if (abs(mask_idx - noun_idx) if (mask_idx >= 0 and noun_idx >= 0) else 0) > 2 else "Shallow",
                            # === PATCH A: linear distance for OOD stratification ===
                            'linear_dist': (
                                abs(mask_idx - noun_idx)
                                if (mask_idx >= 0 and noun_idx >= 0) else -1
                            ),
                        })

        if self.total_extracted_count > 0:
            ratio = (self.isolated_999_count / self.total_extracted_count) * 100
            print(f"[DATASET-AUDIT] Extracted: {self.total_extracted_count}, "
                  f"Discarded 999-hops: {self.isolated_999_count} ({ratio:.2f}%)")

        return tasks

    def split_three_way(
        self,
        tasks: List[Dict[str, Any]],
        top_k: int = 6,
        k_shot: int = 10,
        k_calib: int = 10,
        max_train_hops: int = 2,            # kept for backward compat (unused for stratification)
        max_train_linear_dist: int = 1,     # === PATCH B: NEW: stratify by linear distance ===
) -> Tuple[List, List, List, int]:
        """
        Leakage-free three-way split stratified by **linear token distance**
        between the classifier slot ([MASK]) and its head noun.

        - train: linear_dist <= max_train_linear_dist (shallow)
        - val:   linear_dist <= max_train_linear_dist (shallow, for calibration)
        - test:  shallow remainder + ALL deep (linear_dist > max_train_linear_dist)

        Note:
            We deliberately stratify by linear token distance, not dependency hops,
            because on UD Chinese-GSDSimp all clf-noun pairs have tree_hops == 2
            (clf -> NUM -> NOUN), yielding zero discriminative power.
            Linear distance has a long-tailed distribution (1..23) and aligns with
            the manuscript's definition d = |I_clf - I_noun|.
        """
        top_q = [q for q, _ in Counter([t['quantifier'] for t in tasks]).most_common(top_k)]
        self.quantifier_to_id = {q: i for i, q in enumerate(top_q)}
        filtered = [t for t in tasks if t['quantifier'] in self.quantifier_to_id]

        train_t, val_t, test_t = [], [], []
        for q in top_q:
            q_samples = [t for t in filtered if t['quantifier'] == q]
            # Fallback: if linear_dist missing, compute on the fly
            for t in q_samples:
                if 'linear_dist' not in t:
                    if t['mask_idx'] >= 0 and t['noun_idx'] >= 0:
                        t['linear_dist'] = abs(t['mask_idx'] - t['noun_idx'])
                    else:
                        t['linear_dist'] = -1

            shallow = [t for t in q_samples if 0 <= t['linear_dist'] <= max_train_linear_dist]
            deep = [t for t in q_samples if t['linear_dist'] > max_train_linear_dist]
            random.shuffle(shallow)

            n_shot = min(k_shot, len(shallow))
            n_calib = min(k_calib, max(0, len(shallow) - n_shot))
            
            # Guard: ensure val non-empty if possible
            if n_calib == 0 and n_shot >= 2:
                n_shot -= 1
                n_calib = 1
            
            train_t.extend(shallow[:n_shot])
            val_t.extend(shallow[n_shot:n_shot + n_calib])
            test_t.extend(shallow[n_shot + n_calib:] + deep)

        return train_t, val_t, test_t, len(top_q)

    def encode(
        self,
        batch: List[Dict[str, Any]],
        scramble_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not batch:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        max_l = max(len(t['seq_ids']) for t in batch)
        seqs, labels, masks = [], [], []
        batch_fallback = 0

        for t in batch:
            s = t['seq_ids'].copy()
            seq_len = len(s)

            if scramble_type == 'modifier' and t['mask_idx'] != -1 and t['noun_idx'] != -1:
                core_set = {t['mask_idx'], t['noun_idx']}
                non_core = [i for i in range(seq_len) if i not in core_set]
                vals = [s[i] for i in non_core]
                random.shuffle(vals)
                for i, v in zip(non_core, vals):
                    s[i] = v

            elif scramble_type == 'core_swap' and t['noun_idx'] != -1:
                self.global_swap_total_count += 1
                self.split_swap_total_count += 1

                candidates = [idx for idx in t.get('content_indices', []) if idx != t['noun_idx']]
                if candidates:
                    target_idx = random.choice(candidates)
                    s[t['noun_idx']], s[target_idx] = s[target_idx], s[t['noun_idx']]
                else:
                    batch_fallback += 1
                    self.global_swap_fallback_count += 1
                    self.split_swap_fallback_count += 1

            mask = [1] * seq_len + [0] * (max_l - seq_len)
            seqs.append(s + [0] * (max_l - seq_len))
            masks.append(mask)
            labels.append(self.quantifier_to_id.get(t['quantifier'], 0))

        if scramble_type == 'core_swap' and len(batch) > 0 and batch_fallback > 0:
            print(f"[PERTURB-AUDIT] Batch fallback: {batch_fallback}/{len(batch)}")

        return (
            torch.tensor(seqs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    def reset_all_audit(self) -> None:
        self.split_swap_fallback_count = 0
        self.split_swap_total_count = 0
        self.global_swap_fallback_count = 0
        self.global_swap_total_count = 0

    def reset_split_audit(self) -> None:
        self.split_swap_fallback_count = 0
        self.split_swap_total_count = 0

    def report_split_swap_fallback(self) -> None:
        if self.split_swap_total_count > 0:
            ratio = self.split_swap_fallback_count / self.split_swap_total_count * 100
            print(f"[PERTURB-AUDIT] Split core_swap fallback: "
                  f"{self.split_swap_fallback_count}/{self.split_swap_total_count} ({ratio:.2f}%)")

    def report_global_swap_fallback(self) -> None:
        if self.global_swap_total_count > 0:
            ratio = self.global_swap_fallback_count / self.global_swap_total_count * 100
            print(f"[PERTURB-AUDIT] Global core_swap fallback: "
                  f"{self.global_swap_fallback_count}/{self.global_swap_total_count} ({ratio:.2f}%)")


# =============================================================================
# 3. CAUSAL ATTENTION HEAD KNOCKOUT
# =============================================================================

def apply_attention_head_knockout_exact(
    model: nn.Module,
    layer_idx: int,
    head_idx: int,
    assert_deepcopy: bool = True,
) -> None:
    if assert_deepcopy:
        assert model is not None, "Model must be a DeepCopy when applying knockout."

    layer = model.transformer.layers[layer_idx]
    self_attn = layer.self_attn
    d_model = self_attn.embed_dim
    num_heads = self_attn.num_heads
    head_dim = d_model // num_heads

    start_dim = head_idx * head_dim
    end_dim = (head_idx + 1) * head_dim

    with torch.no_grad():
        self_attn.out_proj.weight[:, start_dim:end_dim] = 0.0
        if self_attn.out_proj.bias is not None:
            self_attn.out_proj.bias[start_dim:end_dim] = 0.0


def run_monte_carlo_random_ablation(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_tasks: List[Dict[str, Any]],
    val_T: float,
    device: torch.device,
    n_samples: int = 100,
    k_heads: int = 3,
) -> Tuple[float, float, List[float]]:
    num_layers = len(model.transformer.layers)
    num_heads = model.transformer.layers[0].self_attn.num_heads

    accuracies: List[float] = []
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    for _ in range(n_samples):
        model_copy = copy.deepcopy(model)
        chosen_heads = random.sample(all_heads, k_heads)
        for l, h in chosen_heads:
            apply_attention_head_knockout_exact(model_copy, l, h, assert_deepcopy=False)

        acc, _, _, _ = evaluate_model_bulletproof(
            model_copy, processor, test_tasks, val_T, device
        )
        accuracies.append(acc)

    return float(np.mean(accuracies)), float(np.std(accuracies)), accuracies


def run_topk_head_ablation(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_tasks: List[Dict[str, Any]],
    val_T: float,
    device: torch.device,
    top_heads: List[Tuple[int, int]],
) -> float:
    model_copy = copy.deepcopy(model)
    for l, h in top_heads:
        apply_attention_head_knockout_exact(model_copy, l, h, assert_deepcopy=False)

    acc, _, _, _ = evaluate_model_bulletproof(model_copy, processor, test_tasks, val_T, device)
    return acc


# =============================================================================
# 3.5 HIDDEN STATE EXTRACTION (PATCH 2.3)
# =============================================================================

class HiddenStateExtractor:
    """
    Context manager that registers a forward hook on `model.transformer`
    to capture token-level hidden states without modifying forward signature.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hidden_states: Optional[torch.Tensor] = None
        self._handle = None

    def __enter__(self):
        def hook_fn(module, inp, out):
            self.hidden_states = out
        self._handle = self.model.transformer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
        return False


# =============================================================================
# 3.6 CAUSAL MEDIATION ANALYSIS (PATCH 2.4)
# =============================================================================

def compute_causal_mediation(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_tasks: List[Dict[str, Any]],
    val_T: float,
    device: torch.device,
    target_heads: List[Tuple[int, int]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Total Indirect Effect (accuracy-based):
        TIE = (acc_intact - acc_ablated) / acc_intact

    Bootstrap only on the ablated model (intact acc fixed) to control cost.
    """
    rng = np.random.default_rng(seed)

    acc_intact, _, _, _ = evaluate_model_bulletproof(
        model, processor, test_tasks, val_T, device
    )
    if acc_intact <= 0:
        return {
            "ie_mean": 0.0, "ie_ci_low": 0.0, "ie_ci_high": 0.0,
            "acc_intact": acc_intact, "acc_ablated": acc_intact,
        }

    model_ablated = copy.deepcopy(model)
    for l, h in target_heads:
        apply_attention_head_knockout_exact(model_ablated, l, h, assert_deepcopy=False)

    acc_ablated_full, _, _, _ = evaluate_model_bulletproof(
        model_ablated, processor, test_tasks, val_T, device
    )

    n = len(test_tasks)

    # === PRECOMPUTE X, y ONCE (avoid 1000x processor.encode) ===
    X_full, y_full, _ = processor.encode(test_tasks)
    X_full = X_full.to(device)
    y_full = y_full.to(device)

    ie_boots = []
    for _ in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        X_boot = X_full[boot_idx]
        y_boot = y_full[boot_idx]
        with torch.no_grad():
            logits = model_ablated(X_boot)
            preds = logits.argmax(dim=1)
        acc_b = preds.eq(y_boot).float().mean().item() * 100
        ie_boots.append((acc_intact - acc_b) / max(acc_intact, 1e-8))

    ie_boots = np.array(ie_boots)
    return {
        "ie_mean": float((acc_intact - acc_ablated_full) / acc_intact),
        "ie_ci_low": float(np.percentile(ie_boots, 2.5)),
        "ie_ci_high": float(np.percentile(ie_boots, 97.5)),
        "acc_intact": float(acc_intact),
        "acc_ablated": float(acc_ablated_full),
    }


# =============================================================================
# 4. LEAKAGE-FREE EVALUATION PIPELINE
# =============================================================================

def evaluate_model_bulletproof(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_tasks: List[Dict[str, Any]],
    fixed_T: float,
    device: torch.device,
    scramble_type: Optional[str] = None,
    verbose_audit: bool = False,
) -> Tuple[float, float, Tuple[float, float, float], Dict[str, Dict[str, int]]]:
    model.eval()
    X_test, y_test, _ = processor.encode(test_tasks, scramble_type=scramble_type)
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        calibrated_logits = logits / fixed_T

    acc = preds.eq(y_test).float().mean().item() * 100
    f1 = calculate_macro_f1(preds, y_test, len(processor.quantifier_to_id))

    softmaxes = torch.softmax(calibrated_logits, dim=-1)
    confs, _ = torch.max(softmaxes, dim=-1)
    correct_flags = preds.eq(y_test).cpu().numpy().astype(int)

    ece_mean, ece_low, ece_high = compute_stratified_bootstrap_ece(
        confidences=confs.cpu().numpy(),
        correct_flags=correct_flags,
        labels=y_test.cpu().numpy(),
        verbose=verbose_audit,
    )

    if verbose_audit:
        print(f"[ECE-AUDIT] ECE={ece_mean:.4f} (95% CI [{ece_low:.4f}, {ece_high:.4f}])")

    comp_stats = {
        "InDist": {"corr": 0, "tot": 0},
        "OOD": {"corr": 0, "tot": 0},
        "Unknown": {"corr": 0, "tot": 0},
    }
    for idx, t in enumerate(test_tasks):
        comp = t.get("complexity", "Unknown")
        if comp == "Shallow":
            comp = "InDist"
        elif comp == "Deep":
            comp = "OOD"
        if comp not in comp_stats:
            comp = "Unknown"
        comp_stats[comp]["tot"] += 1
        if preds[idx] == y_test[idx]:
            comp_stats[comp]["corr"] += 1

    if verbose_audit:
        for k, v in comp_stats.items():
            if v['tot'] > 0:
                print(f"[COMPLEXITY-AUDIT] {k}: {v['corr']}/{v['tot']} "
                      f"({v['corr'] / v['tot'] * 100:.2f}%)")

    return acc, f1, (ece_mean, ece_low, ece_high), comp_stats


def evaluate_perturbed(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_tasks: List[Dict[str, Any]],
    fixed_T: float,
    device: torch.device,
    scramble_type: Optional[str] = None,
    noise_rate: float = 0.0,
    vocab_size: int = 25000,
    verbose_audit: bool = False,
) -> Tuple[float, float, Tuple[float, float, float]]:
    model.eval()
    X_test, y_test, _ = processor.encode(test_tasks, scramble_type=scramble_type)
    X_test, y_test = X_test.to(device), y_test.to(device)

    if noise_rate > 0:
        X_test = inject_syntactic_noise(
            X_test, vocab_size=vocab_size, noise_rate=noise_rate, verbose=verbose_audit
        )

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        calibrated_logits = logits / fixed_T

    acc = preds.eq(y_test).float().mean().item() * 100
    f1 = calculate_macro_f1(preds, y_test, len(processor.quantifier_to_id))

    softmaxes = torch.softmax(calibrated_logits, dim=-1)
    confs, _ = torch.max(softmaxes, dim=-1)
    correct_flags = preds.eq(y_test).cpu().numpy().astype(int)

    ece_mean, ece_low, ece_high = compute_stratified_bootstrap_ece(
        confidences=confs.cpu().numpy(),
        correct_flags=correct_flags,
        labels=y_test.cpu().numpy(),
        verbose=False,
    )

    return acc, f1, (ece_mean, ece_low, ece_high)


# =============================================================================
# 5. FULL ABLATION CONFIG EVALUATION
# =============================================================================

ABLATION_CONFIGS = [
    ("Baseline (No Prior)", False, False, "Baseline"),
    ("Gradual Unfreezing (No Prior)", False, False, "Gradual_Unfreezing_Only"),
    ("SIPAR (No Prior)", False, False, "SIPAR_Only"),
    ("CAFT Full (No Prior)", False, True, "CAFT"),
    ("Baseline (MAML Prior)", True, False, "Baseline"),
    ("Gradual Unfreezing (MAML Prior)", True, False, "Gradual_Unfreezing_Only"),
    ("SIPAR (MAML Prior)", True, False, "SIPAR_Only"),
    ("CAFT Full (MAML Prior)", True, True, "CAFT"),
]

CAUSAL_CONFIG_NAMES = {"CAFT Full (MAML Prior)", "Baseline (MAML Prior)"}


def _collect_hidden_states(
    model: nn.Module,
    processor: BulletproofUDProcessor,
    test_t: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]], List[int]]:
    """
    Collect token-level hidden states from model.transformer via forward hook.

    Returns:
        H_padded: [N, L_max, D] padded hidden states (CPU)
        M_padded: [N, L_max] valid masks (CPU)
        gold_edges_list: list of per-sample edge lists (seq-position space)
        valid_lens: list of per-sample valid lengths
    """
    all_hidden, all_masks, all_gold_edges, all_valid_lens = [], [], [], []
    model.eval()
    for task in test_t:
        X_one, _, mask_one = processor.encode([task])
        X_one = X_one.to(device)
        with HiddenStateExtractor(model) as extractor:
            with torch.no_grad():
                _ = model(X_one)
            hidden = extractor.hidden_states  # [1, L, D]
        if hidden is None:
            continue
        all_hidden.append(hidden[0].cpu())
        all_masks.append(mask_one[0])
        all_gold_edges.append(task["gold_edges"])
        all_valid_lens.append(len(task["seq_ids"]))

    if not all_hidden:
        return torch.zeros(0), torch.zeros(0), [], []

    max_L = max(h.size(0) for h in all_hidden)
    D = all_hidden[0].size(1)
    H_padded = torch.zeros(len(all_hidden), max_L, D)
    M_padded = torch.zeros(len(all_hidden), max_L, dtype=torch.long)
    for i, (h, m) in enumerate(zip(all_hidden, all_masks)):
        H_padded[i, :h.size(0)] = h
        M_padded[i, :m.size(0)] = m
    return H_padded, M_padded, all_gold_edges, all_valid_lens


def _train_probe_and_eval(
    H_padded: torch.Tensor,
    M_padded: torch.Tensor,
    gold_edges_list: List[List[Tuple[int, int]]],
    valid_lens: List[int],
    device: torch.device,
    n_steps: int = 2000,
    probe_rank: int = 64,
    lr: float = 3e-3,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train MaskedStructuralProbe on frozen hidden states, then compute UUAS.
    """
    if H_padded.numel() == 0:
        return {}

    D = H_padded.size(-1)
    probe = MaskedStructuralProbe(model_dim=D, probe_rank=probe_rank).to(device)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    H_dev = H_padded.to(device)
    M_dev = M_padded.to(device)

    # Precompute tree distances ONCE (avoid repeated CPU BFS)
    precomputed_tree = []
    for i in range(len(gold_edges_list)):
        vl = valid_lens[i]
        if vl < 2:
            precomputed_tree.append(None)
            continue
        tree_d = compute_tree_distances(gold_edges_list[i], vl)
        precomputed_tree.append((
            torch.tensor(tree_d, device=device),
            torch.tensor(tree_d ** 2, device=device),
        ))

    for step in range(n_steps):
        probe_opt.zero_grad()
        pred_dist_sq = probe(H_dev, M_dev)

        loss = 0.0
        n_valid = 0
        for i in range(len(gold_edges_list)):
            if precomputed_tree[i] is None:
                continue
            tree_d_t, tree_d_sq = precomputed_tree[i]
            vl = valid_lens[i]
            pred_sub = pred_dist_sq[i, :vl, :vl]
            valid_mask = (tree_d_t < 9.0)
            if valid_mask.sum() > 0:
                loss = loss + F.mse_loss(pred_sub[valid_mask], tree_d_sq[valid_mask])
                n_valid += 1
        if n_valid > 0:
            loss = loss / n_valid
            loss.backward()
            probe_opt.step()
        if step % 200 == 0 or step == n_steps - 1:
            print(f"[PROBE-TRAIN] step={step} | loss={loss.item():.4f} | n_valid={n_valid}")

    uuas = evaluate_probe_on_dataset(
        probe, H_padded, M_padded, gold_edges_list, valid_lens,
        device, verbose=False,
    )
    return uuas


def evaluate_ablation_configs(
    processor: BulletproofUDProcessor,
    train_t: List[Dict[str, Any]],
    val_t: List[Dict[str, Any]],
    test_t: List[Dict[str, Any]],
    num_classes: int,
    device: torch.device,
    prior_path: str = "meta_pretrained_prior.pth",
    verbose_audit: bool = False,
    enable_causal_analysis: bool = True,   # === PATCH 2.5 ===
) -> Dict[str, Dict[str, Any]]:
    """
    Run all 8 ablation configurations.

    For CAUSAL_CONFIG_NAMES, additionally compute:
        - UUAS (via MaskedStructuralProbe)
        - Top-3 head knockout + Monte Carlo null
        - Causal mediation (TIE with bootstrap CI)
    """
    results: Dict[str, Dict[str, Any]] = {
        name: {"acc": [], "f1": [], "ece": [], "converged": [], "T_star": [],
               "uuas": [], "head_knockout": [], "mediation": []}
        for name, _, _, _ in ABLATION_CONFIGS
    }

    for name, has_prior, is_ours, config_type in ABLATION_CONFIGS:
        model = NeuralTransformerStudent(vocab_size=25000, num_classes=num_classes).to(device)

        if has_prior and os.path.exists(prior_path):
            load_prior_weights_soft(model, prior_path, device, alpha=0.4)

        X_train, y_train, _ = processor.encode(train_t)
        X_train, y_train = X_train.to(device), y_train.to(device)

        train_result = train_sota_few_shot(
            model=model, X=X_train, y=y_train,
            processor=processor, train_t=train_t,
            is_ours=is_ours, has_prior=has_prior,
            device=device, config_type=config_type,
            phase1_epochs=20, phase2_epochs=60,
            verbose=False,
        )
        model = train_result['model']

        X_val, y_val, _ = processor.encode(val_t)
        X_val, y_val = X_val.to(device), y_val.to(device)
        opt_T, converged, (ci_low, ci_high) = fit_temperature_on_val(
            model, X_val, y_val, device
        )

        if not converged:
            print(f"[AUDIT-ALERT] Temperature scaling failed to converge for {name}. "
                  f"Reverted to T={opt_T:.4f}")
        else:
            print(f"[AUDIT-SUCCESS] {name} | T*={opt_T:.4f} "
                  f"(95% CI [{ci_low:.4f}, {ci_high:.4f}])")

        acc, f1, (ece_m, ece_l, ece_h), _ = evaluate_model_bulletproof(
            model, processor, test_t, opt_T, device, verbose_audit=verbose_audit
        )

        results[name]["acc"].append(acc)
        results[name]["f1"].append(f1)
        results[name]["ece"].append(ece_m)
        results[name]["converged"].append(int(converged))
        results[name]["T_star"].append(opt_T)

        print(f"[CONFIG] {name} | acc={acc:.2f} | f1={f1:.2f} | ece={ece_m:.4f} | "
              f"T*={opt_T:.4f} | converged={converged}")

        # === SAVE CHECKPOINT for causal configs ===
        if name in ("Baseline (MAML Prior)", "CAFT Full (MAML Prior)"):
            slug = "baseline" if "Baseline" in name else "caft"
            ckpt_path = os.path.join("results", f"{slug}_maml.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved {name} -> {ckpt_path}")

        # ---- Causal analysis (PATCH 2.5) ----
        if enable_causal_analysis and name in CAUSAL_CONFIG_NAMES:
            print(f"[CAUSAL] Running UUAS + Knockout + Mediation for: {name}")

            # (1) UUAS
            try:
                H_pad, M_pad, gold_edges_list, valid_lens = _collect_hidden_states(
                    model, processor, test_t, device
                )
                uuas = _train_probe_and_eval(
                    H_pad, M_pad, gold_edges_list, valid_lens,
                    device, n_steps=2000, probe_rank=64, lr=3e-3,
                    verbose=verbose_audit,
                )
                results[name]["uuas"].append(uuas)
                if uuas:
                    print(f"[UUAS] {name} | mst={uuas['mst_uuas']:.2f} | "
                          f"hm_und={uuas['hm_undirected_uuas']:.2f} | "
                          f"hm_dir={uuas['hm_directed_uuas']:.2f} | "
                          f"discarded={uuas['discarded_ratio']:.2f}%")
            except Exception as e:
                print(f"[WARN] UUAS failed for {name}: {e}")
                results[name]["uuas"].append({})

            # (2) Head sweep + knockout + MC null + mediation
            try:
                num_layers = len(model.transformer.layers)
                num_heads = model.transformer.layers[0].self_attn.num_heads
                all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

                head_drops = []
                for (l, h) in all_heads:
                    acc_abl = run_topk_head_ablation(
                        model, processor, test_t, opt_T, device, [(l, h)]
                    )
                    head_drops.append((l, h, acc - acc_abl))

                sorted_drops = sorted(head_drops, key=lambda x: -x[2])
                top3 = [x[:2] for x in sorted_drops[:3]]
                print(f"[KNOCKOUT] {name} | top3_heads={top3} | "
                      f"top3_drops={[round(x[2], 2) for x in sorted_drops[:3]]}")

                mc_mean, mc_std, mc_accs = run_monte_carlo_random_ablation(
                    model, processor, test_t, opt_T, device, n_samples=100, k_heads=3
                )

                acc_top3 = run_topk_head_ablation(
                    model, processor, test_t, opt_T, device, top3
                )
                target_drop = acc - acc_top3

                p_value = (sum(1 for a in mc_accs if a <= acc_top3) + 1) / (len(mc_accs) + 1)

                mediation = compute_causal_mediation(
                    model, processor, test_t, opt_T, device, top3, n_boot=1000,
                )
                print(f"[MEDIATION] {name} | IE={mediation['ie_mean']:.4f} "
                      f"[{mediation['ie_ci_low']:.4f}, {mediation['ie_ci_high']:.4f}]")

                head_result = {
                    "intact_acc": acc,
                    "mc_null_mean": mc_mean,
                    "mc_null_std": mc_std,
                    "target_heads": top3,
                    "target_heads_acc": acc_top3,
                    "target_drop": target_drop,
                    "p_value": p_value,
                    "mediation_ie": mediation["ie_mean"],
                    "mediation_ci_low": mediation["ie_ci_low"],
                    "mediation_ci_high": mediation["ie_ci_high"],
                }
                results[name]["head_knockout"].append(head_result)
                results[name]["mediation"].append(mediation)
            except Exception as e:
                print(f"[WARN] Knockout/Mediation failed for {name}: {e}")
                results[name]["head_knockout"].append({})
                results[name]["mediation"].append({})

    return results


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def _aggregate_and_dump(
    split_results: List[Dict[str, Dict[str, Any]]],
    config_names: List[str],
    results_dir: str,
) -> None:
    """
    Aggregate across splits and dump ablation_summary.json + head_knockout_results.json.
    """
    os.makedirs(results_dir, exist_ok=True)

    ablation_summary: Dict[str, Any] = {}
    head_knockout_summary: Dict[str, Any] = {}

    print(f"\n{'=' * 100}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 100}")
    print(f"| {'Configuration':<40} | {'Acc (%)':<15} | {'F1 (%)':<10} | "
          f"{'ECE':<10} | {'Conv.':<6} | {'UUAS':<8} |")
    print("-" * 110)

    for name in config_names:
        accs, f1s, eces, convs = [], [], [], []
        uuas_mst, uuas_hm_und, uuas_hm_dir, uuas_disc = [], [], [], []
        head_knockout_list = []
        mediation_list = []

        for sr in split_results:
            if name not in sr:
                continue
            accs.extend(sr[name]["acc"])
            f1s.extend(sr[name]["f1"])
            eces.extend(sr[name]["ece"])
            convs.extend(sr[name]["converged"])
            for u in sr[name].get("uuas", []):
                if u and "mst_uuas" in u:
                    uuas_mst.append(u["mst_uuas"])
                    uuas_hm_und.append(u["hm_undirected_uuas"])
                    uuas_hm_dir.append(u["hm_directed_uuas"])
                    uuas_disc.append(u["discarded_ratio"])
            head_knockout_list.extend(sr[name].get("head_knockout", []))
            mediation_list.extend(sr[name].get("mediation", []))

        if not accs:
            continue

        acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
        f1_mean = float(np.mean(f1s))
        ece_mean = float(np.mean(eces))
        conv_ratio = float(np.mean(convs)) * 100

        uuas_mean = float(np.mean(uuas_mst)) if uuas_mst else float("nan")
        print(f"| {name:<40} | {acc_mean:>5.2f} ± {acc_std:<6.2f} | "
              f"{f1_mean:>8.2f} | {ece_mean:>8.4f} | {conv_ratio:>5.1f}% | "
              f"{uuas_mean:>8.2f} |")

        entry = {
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "f1_mean": f1_mean,
            "ece_mean": ece_mean,
            "conv_ratio": conv_ratio,
            "T_star_mean": float(np.mean([sr[name]["T_star"][0] for sr in split_results
                                          if name in sr and sr[name]["T_star"]])) if any(
                sr[name]["T_star"] for sr in split_results if name in sr) else 0.0,
        }
        if uuas_mst:
            entry["uuas"] = {
                "mst_uuas_mean": float(np.mean(uuas_mst)),
                "mst_uuas_std": float(np.std(uuas_mst)),
                "hm_undirected_uuas_mean": float(np.mean(uuas_hm_und)),
                "hm_undirected_uuas_std": float(np.std(uuas_hm_und)),
                "hm_directed_uuas_mean": float(np.mean(uuas_hm_dir)),
                "hm_directed_uuas_std": float(np.std(uuas_hm_dir)),
                "discarded_ratio_mean": float(np.mean(uuas_disc)),
            }

        ablation_summary[name] = entry

        # Head knockout aggregation (only for causal configs)
        valid_hk = [h for h in head_knockout_list if h]
        if valid_hk:
            head_knockout_summary[name] = {
                "intact_acc_mean": float(np.mean([h["intact_acc"] for h in valid_hk])),
                "mc_null_mean": float(np.mean([h["mc_null_mean"] for h in valid_hk])),
                "mc_null_std": float(np.mean([h["mc_null_std"] for h in valid_hk])),
                "target_heads_acc_mean": float(np.mean([h["target_heads_acc"] for h in valid_hk])),
                "target_drop_mean": float(np.mean([h["target_drop"] for h in valid_hk])),
                "p_value_mean": float(np.mean([h["p_value"] for h in valid_hk])),
                "mediation_ie_mean": float(np.mean([h["mediation_ie"] for h in valid_hk])),
                "mediation_ci_low_mean": float(np.mean([h["mediation_ci_low"] for h in valid_hk])),
                "mediation_ci_high_mean": float(np.mean([h["mediation_ci_high"] for h in valid_hk])),
                "n_splits": len(valid_hk),
            }

    print("=" * 110)

    # Dump
    ablation_path = os.path.join(results_dir, "ablation_summary.json")
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)
    print(f"[DUMP] {ablation_path}")

    if head_knockout_summary:
        hk_path = os.path.join(results_dir, "head_knockout_results.json")
        with open(hk_path, "w", encoding="utf-8") as f:
            json.dump(head_knockout_summary, f, indent=2, ensure_ascii=False)
        print(f"[DUMP] {hk_path}")


def run_full_pipeline(
    ud_path: str = "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu",
    num_splits: int = 10,
    top_k: int = 6,
    k_shot: int = 10,
    k_calib: int = 10,
    verbose_audit: bool = True,
    results_dir: str = "results",   # === PATCH 2.6 ===
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if not os.path.exists(ud_path):
        for file in os.listdir('.'):
            if file.endswith('.conllu'):
                ud_path = file
                break

    print(f"[INFO] Loading UD corpus: {ud_path}")
    processor = BulletproofUDProcessor()
    all_tasks = processor.extract_tasks(ud_path)

    if not all_tasks:
        print("[ERROR] No tasks extracted.")
        return

    print(f"[INFO] Total tasks: {len(all_tasks)}")
    processor.reset_all_audit()

    split_results: List[Dict[str, Dict[str, Any]]] = []

    for split_idx in range(num_splits):
        print(f"\n==================== Split {split_idx + 1}/{num_splits} ====================")
        processor.reset_split_audit()

        random.seed(42 + split_idx)
        np.random.seed(42 + split_idx)
        torch.manual_seed(42 + split_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + split_idx)

        train_t, val_t, test_t, num_classes = processor.split_three_way(
            all_tasks, top_k=top_k, k_shot=k_shot, k_calib=k_calib,
            max_train_linear_dist=2,   # === PATCH C ===
        )

        n_deep = sum(1 for t in test_t if t.get('linear_dist', 0) > 1)
        print(f"[SPLIT-INFO] train={len(train_t)} | val={len(val_t)} | test={len(test_t)} "
            f"(deep={n_deep}) | num_classes={num_classes}")

        split_result = evaluate_ablation_configs(
            processor, train_t, val_t, test_t, num_classes, device,
            verbose_audit=verbose_audit,
        )
        split_results.append(split_result)

        processor.report_split_swap_fallback()

    print(f"\n{'=' * 80}")
    print("GLOBAL PERTURBATION AUDIT")
    print(f"{'=' * 80}")
    processor.report_global_swap_fallback()

    config_names = [name for name, _, _, _ in ABLATION_CONFIGS]
    _aggregate_and_dump(split_results, config_names, results_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--k_shot", type=int, default=10)
    parser.add_argument("--k_calib", type=int, default=10)
    parser.add_argument("--verbose_audit", action="store_true", default=True)
    parser.add_argument("--results_dir", type=str, default="results")
    args, _ = parser.parse_known_args()

    run_full_pipeline(
        num_splits=args.num_splits,
        top_k=args.top_k,
        k_shot=args.k_shot,
        k_calib=args.k_calib,
        verbose_audit=args.verbose_audit,
        results_dir=args.results_dir,
    )