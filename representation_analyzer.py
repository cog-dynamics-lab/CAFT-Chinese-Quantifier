"""
Module: representation_analyzer.py (Final Corrected Edition)
Description:
    Provides masked structural probing, MST-based UUAS, Hewitt & Manning per-token UUAS,
    effective rank, and cosine anisotropy diagnostics for representation geometry analysis.

Return Value Contract:
    compute_mst_and_hm_uuas() strictly returns a 4-tuple:
        (mst_uuas, hm_undirected_uuas, hm_directed_uuas, discarded_ratio)

    Callers MUST unpack all 4 values. The discarded_ratio reflects the fraction of
    gold edges that fall outside valid_len and are silently dropped during UUAS
    computation, which should be reported in the appendix for transparency.

Fixes over previous version:
    1. compute_mst_and_hm_uuas return arity explicitly documented as 4-tuple.
    2. evaluate_probe_on_dataset correctly unpacks 4-tuple and reports discarded_ratio.
    3. MaskedStructuralProbe diagonal protection logic commented and preserved.
    4. calculate_metrics adds SVD failure audit.
    5. __all__ export declaration added.
    6. Type hints added.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Dict

# =============================================================================
# 0. UUAS RESULT DATACLASS
# =============================================================================

from dataclasses import dataclass
from typing import Iterator

@dataclass
class UUASResult:
    """
    Container for UUAS metrics. Iterable as a 4-tuple for backward compatibility:
        mst, hm_und, hm_dir, disc = result
    """
    mst_uuas: float
    hm_undirected_uuas: float
    hm_directed_uuas: float
    discarded_ratio: float

    def __iter__(self) -> Iterator[float]:
        yield self.mst_uuas
        yield self.hm_undirected_uuas
        yield self.hm_directed_uuas
        yield self.discarded_ratio

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> float:
        return (self.mst_uuas, self.hm_undirected_uuas,
                self.hm_directed_uuas, self.discarded_ratio)[idx]

__all__ = [
    "MaskedStructuralProbe",
    "UUASResult",
    "compute_mst_and_hm_uuas",
    "compute_tree_distances",       # === PATCH 1: new ===
    "RepresentationAnalyzer",
    "evaluate_probe_on_dataset",
]


# =============================================================================
# 1. MASKED STRUCTURAL PROBE (HEWITT & MANNING, 2019)
# =============================================================================

class MaskedStructuralProbe(nn.Module):
    """
    Low-rank bilinear structural probe (Hewitt & Manning, 2019).

    Learns a linear projection B such that squared Euclidean distances in the
    projected space approximate dependency tree distances.

    Masking rules:
        - Valid-token self-distances (diagonal) are preserved at 0.0.
        - Padding-related pairs (including padding self-distances) are set to +INF,
          preventing MST from erroneously connecting padding nodes.
    """

    def __init__(self, model_dim: int = 128, probe_rank: int = 32) -> None:
        super().__init__()
        self.B = nn.Parameter(torch.randn(probe_rank, model_dim) * 0.1)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, L, D] hidden representations.
            mask: [B, L] binary mask (1 = valid token, 0 = padding).

        Returns:
            dist_sq: [B, L, L] squared Euclidean distance matrix.
        """
        h_f32 = h.float()
        h_proj = torch.matmul(h_f32, self.B.t())          # [B, L, Rank]
        diff = h_proj.unsqueeze(2) - h_proj.unsqueeze(1)
        dist_sq = torch.sum(diff ** 2, dim=-1)            # [B, L, L]

        if mask is not None:
            valid_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()   # [B, L, L]
            diag_mask = torch.eye(
                dist_sq.size(1), device=dist_sq.device
            ).bool().unsqueeze(0)                                          # [1, L, L]

            # Only preserve diagonal for valid tokens;
            # padding self-diagonal and all padding-related pairs become +INF.
            valid_diag = valid_mask & diag_mask
            fill_mask = (~valid_mask) & (~valid_diag)
            dist_sq = dist_sq.masked_fill(fill_mask, float('inf'))

        return dist_sq


# =============================================================================
# 2. UUAS COMPUTATION: MST AND HEWITT & MANNING
# =============================================================================

def compute_mst_and_hm_uuas(
    dist_sq_matrix: np.ndarray,
    gold_edges: List[Tuple[int, int]],
    valid_len: int,
    exclude_root: bool = True,
    verbose: bool = False,
) -> UUASResult:
    """
    Compute MST-based UUAS and Hewitt & Manning per-token UUAS.

    Returns:
        UUASResult (iterable as 4-tuple for backward compatibility)
    """
    assert dist_sq_matrix.shape[0] >= valid_len, "Matrix dimension smaller than valid length"

    if valid_len <= 1 or not gold_edges:
        return UUASResult(100.0, 100.0, 100.0, 0.0)

    sub_dist_sq = dist_sq_matrix[:valid_len, :valid_len]
    safe_dist_sq = np.where(np.isinf(sub_dist_sq), 1e9, sub_dist_sq)
    dists = np.sqrt(np.maximum(safe_dist_sq, 0.0))

    gold_directed = set()
    gold_undirected = set()
    discarded = 0
    for u, v in gold_edges:
        if u < valid_len and v < valid_len:
            gold_directed.add((u, v))
            gold_directed.add((v, u))
            gold_undirected.add(tuple(sorted([u, v])))
        else:
            discarded += 1

    total_gold = len(gold_edges)
    discarded_ratio = (discarded / total_gold * 100) if total_gold > 0 else 0.0

    if verbose:
        print(f"[UUAS-AUDIT] valid_len={valid_len} | "
              f"gold_directed={len(gold_directed)} | "
              f"gold_undirected={len(gold_undirected)} | "
              f"discarded={discarded} ({discarded_ratio:.2f}%)")

    if not gold_undirected:
        return UUASResult(0.0, 0.0, 0.0, discarded_ratio)

    # MST-based UUAS
    G = nx.Graph()
    for i in range(valid_len):
        for j in range(i + 1, valid_len):
            G.add_edge(i, j, weight=float(dists[i, j]))
    mst = nx.minimum_spanning_tree(G)
    mst_edges = set(tuple(sorted([u, v])) for u, v in mst.edges())
    mst_uuas = (len(mst_edges.intersection(gold_undirected)) / len(gold_undirected)) * 100.0

    # H&M per-token
    dists_no_diag = dists.copy()
    np.fill_diagonal(dists_no_diag, np.inf)

    gold_heads = {u for u, v in gold_edges if u < valid_len and v < valid_len}
    root_idx = -1
    for i in range(valid_len):
        if i not in gold_heads:
            root_idx = i
            break

    hm_correct_undirected = 0
    hm_correct_directed = 0
    hm_total = 0

    for i in range(valid_len):
        if exclude_root and i == root_idx:
            continue
        pred_head = int(np.argmin(dists_no_diag[i]))
        hm_total += 1

        if tuple(sorted([i, pred_head])) in gold_undirected:
            hm_correct_undirected += 1
        if (i, pred_head) in gold_directed:
            hm_correct_directed += 1

    hm_undirected_uuas = (hm_correct_undirected / hm_total * 100.0) if hm_total > 0 else 0.0
    hm_directed_uuas = (hm_correct_directed / hm_total * 100.0) if hm_total > 0 else 0.0

    return UUASResult(
        mst_uuas=float(mst_uuas),
        hm_undirected_uuas=float(hm_undirected_uuas),
        hm_directed_uuas=float(hm_directed_uuas),
        discarded_ratio=float(discarded_ratio),
    )


# =============================================================================
# 3. REPRESENTATION ANALYZER
# =============================================================================

class RepresentationAnalyzer:
    """
    Collects representations and computes effective rank, cosine anisotropy,
    and structural probe diagnostics.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def collect_representations(
        self,
        X: torch.Tensor,
        device: Optional[torch.device] = None,
        return_mask: bool = False,
    ):
        """
        Collect hidden representations from the model.

        Args:
            X: [B, L] input token ids.
            device: torch device.
            return_mask: if True, also return valid-token mask.

        Returns:
            feats: [B, D] pooled representation.
            mask: [B, L] optional.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.to(device))
            feats = outputs[0] if isinstance(outputs, tuple) else outputs

        mask = (X != 0).long().to(device) if return_mask else None

        if feats.dim() == 3:
            feats = feats.mean(dim=1)

        if return_mask:
            return feats, mask
        return feats

    def calculate_metrics(
        self,
        H: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute effective rank and average cosine similarity on centered representations.

        Args:
            H: [N, D] representation matrix.
            verbose: if True, print SVD diagnostics.

        Returns:
            (erank, avg_cos)
        """
        if H.dim() > 2:
            H = H.view(H.size(0), -1)

        H_centered = H - H.mean(dim=0, keepdim=True)
        N = H.size(0)

        if N < 2:
            return 1.0, 1.0

        # 1. Average cosine similarity on centered manifold
        norm_H = F.normalize(H_centered, p=2, dim=1)
        cos_sim = torch.mm(norm_H, norm_H.t())
        triu_indices = torch.triu_indices(N, N, offset=1)
        avg_cos = cos_sim[triu_indices[0], triu_indices[1]].mean().item()

        # 2. Effective rank via SVD entropy
        erank = 1.0
        try:
            _, S, _ = torch.linalg.svd(H_centered.float(), full_matrices=False)
            sum_s = torch.sum(S)
            if sum_s > 1e-8:
                p = S / sum_s
                entropy = -torch.sum(p * torch.log(p + 1e-9))
                erank = torch.exp(entropy).item()
                erank = max(erank, 1.0)
        except Exception as e:
            if verbose:
                print(f"[SVD-WARNING] SVD failed: {e}. Falling back to erank=1.0.")
            erank = 1.0

        if verbose:
            print(f"[REP-METRICS] N={N} | erank={erank:.4f} | avg_cos={avg_cos:.4f}")

        return float(erank), float(avg_cos)


# =============================================================================
# 4. LAYER-WISE PROBE EVALUATION HELPER
# =============================================================================

def evaluate_probe_on_dataset(
    probe: MaskedStructuralProbe,
    representations: torch.Tensor,
    masks: torch.Tensor,
    gold_edges_list: List[List[Tuple[int, int]]],
    valid_lens: List[int],
    device: torch.device,
    exclude_root: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained structural probe on a dataset.

    IMPORTANT:
        compute_mst_and_hm_uuas() returns a 4-tuple. Callers MUST unpack all 4
        values, including discarded_ratio. This function reports the mean
        discarded_ratio in the returned dict for appendix transparency.

    Args:
        probe: trained MaskedStructuralProbe.
        representations: [N, L, D] hidden states.
        masks: [N, L] valid-token masks.
        gold_edges_list: list of gold edge lists per sample.
        valid_lens: list of valid lengths per sample.
        device: torch device.
        exclude_root: whether to exclude root in H&M UUAS.
        verbose: if True, print per-sample UUAS audit.

    Returns:
        {
            "mst_uuas": float,
            "hm_undirected_uuas": float,
            "hm_directed_uuas": float,
            "discarded_ratio": float,
            "num_samples": int,
        }
    """
    probe.eval()
    mst_scores, hm_und_scores, hm_dir_scores, discarded_ratios = [], [], [], []

    with torch.no_grad():
        dist_sq = probe(representations.to(device), masks.to(device))

    dist_sq_np = dist_sq.cpu().numpy()

    for i in range(len(valid_lens)):
        result = compute_mst_and_hm_uuas(
            dist_sq_matrix=dist_sq_np[i],
            gold_edges=gold_edges_list[i],
            valid_len=valid_lens[i],
            exclude_root=exclude_root,
            verbose=verbose,
        )
        # Dataclass field access; no unpacking
        mst_scores.append(result.mst_uuas)
        hm_und_scores.append(result.hm_undirected_uuas)
        hm_dir_scores.append(result.hm_directed_uuas)
        discarded_ratios.append(result.discarded_ratio)

    result = {
        "mst_uuas": float(np.mean(mst_scores)),
        "hm_undirected_uuas": float(np.mean(hm_und_scores)),
        "hm_directed_uuas": float(np.mean(hm_dir_scores)),
        "discarded_ratio": float(np.mean(discarded_ratios)),
        "num_samples": len(valid_lens),
    }

    if verbose:
        print(f"[PROBE-EVAL] mst_uuas={result['mst_uuas']:.2f} | "
              f"hm_undirected_uuas={result['hm_undirected_uuas']:.2f} | "
              f"hm_directed_uuas={result['hm_directed_uuas']:.2f} | "
              f"discarded_ratio={result['discarded_ratio']:.2f}% | "
              f"n={result['num_samples']}")

    return result

# =============================================================================
# 5. TREE DISTANCE COMPUTATION (FOR STRUCTURAL PROBE TRAINING)
# =============================================================================

def compute_tree_distances(
    gold_edges: List[Tuple[int, int]],
    valid_len: int,
    unreachable_dist: float = 10.0,
) -> np.ndarray:
    """
    Compute pairwise shortest-path tree distance matrix via BFS.

    This is used to train MaskedStructuralProbe: the probe learns B such that
    squared Euclidean distance in projected space approximates tree_dist^2.

    Args:
        gold_edges: list of (u, v) edges in seq-position space.
        valid_len: number of valid tokens.
        unreachable_dist: fill value for disconnected pairs.

    Returns:
        dist: [valid_len, valid_len] float32 array, dist[i, i] = 0.
    """
    G = nx.Graph()
    G.add_nodes_from(range(valid_len))
    for u, v in gold_edges:
        if u < valid_len and v < valid_len and u != v:
            G.add_edge(u, v)

    dist = np.full((valid_len, valid_len), fill_value=unreachable_dist, dtype=np.float32)
    for i in range(valid_len):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=valid_len)
        for j, d in lengths.items():
            dist[i, j] = float(d)
    return dist