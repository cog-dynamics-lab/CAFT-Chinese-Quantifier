#!/usr/bin/env python3
"""
Module: verify_encode.py
Description:
    Verify that NeuralPredictor._encode_sentence correctly performs Max-Length
    Forward Matching over noun tokens, avoiding systematic <UNK> degradation
    for multi-character nouns.

Checks performed:
    1. Unit regression on 12 hand-crafted sentences covering:
       - single-character nouns (书, 马)
       - 2/3/4-character nouns (木盒, 长方桌子, ...)
       - [MASK] special-case
       - whitespace skipping
       - unknown character fallback
    2. Benchmark-level <UNK> rate audit over double_blind_benchmark.json.
    3. Noun hit rate and candidate dropped ratio audit.

Exit code:
    0 - all checks pass
    1 - at least one check failed

Usage:
    python verify_encode.py \
        --train_corpus UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu \
        --benchmark double_blind_benchmark.json
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn


# =============================================================================
# 1. MOCK MODEL (no inference needed, we only test encoding)
# =============================================================================

class MockModel(nn.Module):
    """Minimal stub: forward returns zeros of shape [B, 6]."""
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.num_classes, device=x.device)


# =============================================================================
# 2. UNIT TEST CASES
# =============================================================================

# (sentence, expected_unk_max, description)
UNIT_TESTS: List[Tuple[str, int, str]] = [
    # Single-character noun (must hit vocabulary via fallback, not <UNK>)
    ("这[MASK]书。", 0, "single-char noun 书"),
    ("这[MASK]马。", 0, "single-char noun 马"),

    # 2-character nouns (must hit via longest match)
    ("这[MASK]木盒。", 0, "2-char noun 木盒"),
    ("这[MASK]长凳。", 0, "2-char noun 长凳"),
    ("这[MASK]宝剑。", 0, "2-char noun 宝剑"),

    # Benchmark-style sentences
    ("这[MASK]摆放在木盒内部的、修长柔韧的托格。", 5, "condition A w/ wug"),
    ("这[MASK]装着修长柔韧的托格的宝剑。", 5, "condition B w/ wug"),
    ("这[MASK]摆放在长凳旁边的、平整木制的桌子。", 5, "condition C sibling"),

    # Edge cases
    ("[MASK]", 0, "only [MASK]"),
    ("", -1, "empty string (should return None)"),
    ("这  [MASK]  木盒。", 0, "whitespace skipping"),
    ("这[MASK]某个未登录词xyz。", 10, "unknown character fallback"),
]


# =============================================================================
# 3. UTILITIES
# =============================================================================

def find_noun_words(vocab: Dict[str, int]) -> set:
    """Return the set of noun words (id in [100, 2000))."""
    return {w for w, i in vocab.items() if 100 <= i < 2000}


def find_unk_words(vocab: Dict[str, int]) -> set:
    """Return the set of non-noun, non-special tokens (id >= 2000)."""
    return {w for w, i in vocab.items() if i >= 2000}


def check_sentence(
    predictor: Any,
    sentence: str,
    unk_id: int,
    verbose: bool = True,
) -> Tuple[Optional[List[int]], float]:
    """
    Encode a sentence and return (ids, unk_ratio).

    Returns:
        (None, -1.0) if encoding failed.
        (ids, unk_ratio) otherwise.
    """
    ids_tensor = predictor._encode_sentence(sentence)
    if ids_tensor is None:
        if verbose:
            print(f"  [ENCODE-FAIL] {sentence!r} -> None")
        return None, -1.0

    ids = ids_tensor[0].tolist()
    unk_count = sum(1 for i in ids if i == unk_id)
    total = len(ids)
    ratio = (unk_count / total * 100.0) if total > 0 else 0.0

    if verbose:
        print(f"  Input : {sentence!r}")
        print(f"  IDs   : {ids}")
        print(f"  <UNK> : {unk_count}/{total} ({ratio:.1f}%)")

    return ids, ratio


# =============================================================================
# 4. UNIT TEST RUNNER
# =============================================================================

def run_unit_tests(predictor: Any, unk_id: int) -> int:
    """
    Run all hand-crafted unit tests.

    Returns:
        Number of failed tests.
    """
    print("=" * 80)
    print("UNIT TESTS")
    print("=" * 80)

    failures = 0
    for idx, (sentence, expected_unk_max, desc) in enumerate(UNIT_TESTS, 1):
        print(f"\n[Test {idx:02d}] {desc}")
        ids, ratio = check_sentence(predictor, sentence, unk_id, verbose=True)

        if ids is None:
            if expected_unk_max < 0:
                print(f"  [PASS] empty string correctly returned None")
            else:
                print(f"  [FAIL] unexpected None for non-empty string")
                failures += 1
            continue

        if expected_unk_max < 0:
            print(f"  [FAIL] expected None but got {len(ids)} tokens")
            failures += 1
            continue

        if ratio <= expected_unk_max:
            print(f"  [PASS] <UNK> ratio {ratio:.1f}% <= {expected_unk_max}%")
        else:
            print(f"  [FAIL] <UNK> ratio {ratio:.1f}% > {expected_unk_max}%")
            failures += 1

    return failures


# =============================================================================
# 5. BENCHMARK-LEVEL AUDIT
# =============================================================================

def audit_benchmark(
    predictor: Any,
    benchmark_path: str,
    unk_id: int,
) -> Tuple[int, Dict[str, float]]:
    """
    Audit the benchmark sentences for <UNK> rate and noun hit rate.

    Returns:
        (failures, stats) where stats = {
            'num_samples', 'avg_unk_ratio', 'max_unk_ratio',
            'noun_hit_rate', 'sentence_count'
        }
    """
    print("\n" + "=" * 80)
    print("BENCHMARK AUDIT")
    print("=" * 80)

    if not os.path.exists(benchmark_path):
        print(f"[SKIP] Benchmark not found: {benchmark_path}")
        return 0, {}

    with open(benchmark_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", [])
    if not samples:
        print("[SKIP] No samples found.")
        return 0, {}

    print(f"[INFO] Loaded {len(samples)} benchmark samples")

    unk_ratios = []
    noun_hits = 0
    total_nouns = 0
    failed_encode = 0

    for idx, item in enumerate(samples):
        sentence = item.get("sentence", "")
        if not sentence:
            continue

        ids_tensor = predictor._encode_sentence(sentence)
        if ids_tensor is None:
            failed_encode += 1
            continue

        ids = ids_tensor[0].tolist()
        unk_count = sum(1 for i in ids if i == unk_id)
        total = len(ids)
        ratio = (unk_count / total * 100.0) if total > 0 else 0.0
        unk_ratios.append(ratio)

        # Check gold noun presence
        gold_noun = item.get("core_noun", "")
        if gold_noun:
            total_nouns += 1
            noun_id = predictor.processor.vocab.get(gold_noun, None)
            if noun_id is not None and noun_id in ids:
                noun_hits += 1

    if not unk_ratios:
        print("[SKIP] No valid encoded sentences.")
        return 0, {}

    avg_unk = sum(unk_ratios) / len(unk_ratios)
    max_unk = max(unk_ratios)
    noun_hit_rate = (noun_hits / total_nouns * 100.0) if total_nouns > 0 else 0.0

    print(f"\n[STATS]")
    print(f"  Samples           : {len(unk_ratios)}")
    print(f"  Failed encode     : {failed_encode}")
    print(f"  Avg <UNK> ratio   : {avg_unk:.2f}%")
    print(f"  Max <UNK> ratio   : {max_unk:.2f}%")
    print(f"  Noun hit rate     : {noun_hit_rate:.2f}% ({noun_hits}/{total_nouns})")

    # Thresholds
    failures = 0
    if avg_unk > 30.0:
        print(f"  [FAIL] Avg <UNK> ratio {avg_unk:.1f}% > 30%")
        failures += 1
    else:
        print(f"  [PASS] Avg <UNK> ratio {avg_unk:.1f}% <= 30%")

    if noun_hit_rate < 80.0:
        print(f"  [FAIL] Noun hit rate {noun_hit_rate:.1f}% < 80%")
        failures += 1
    else:
        print(f"  [PASS] Noun hit rate {noun_hit_rate:.1f}% >= 80%")

    stats = {
        "num_samples": len(unk_ratios),
        "avg_unk_ratio": avg_unk,
        "max_unk_ratio": max_unk,
        "noun_hit_rate": noun_hit_rate,
        "failed_encode": failed_encode,
    }
    return failures, stats


# =============================================================================
# 6. CANDIDATE DROPPED RATIO AUDIT
# =============================================================================

def audit_candidate_dropped_ratio(
    predictor: Any,
    benchmark_path: str,
) -> Tuple[int, Dict[str, Any]]:
    """
    Simulate candidate filtering to estimate dropped ratio.

    Returns:
        (failures, stats)
    """
    print("\n" + "=" * 80)
    print("CANDIDATE DROPPED RATIO AUDIT")
    print("=" * 80)

    if not os.path.exists(benchmark_path):
        print(f"[SKIP] Benchmark not found: {benchmark_path}")
        return 0, {}

    with open(benchmark_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", [])
    if not samples:
        return 0, {}

    vocab = predictor.processor.vocab
    total_dropped = 0
    total_candidates = 0

    for item in samples:
        for cand_type in ["closed", "open"]:
            key = f"{cand_type}_candidates"
            candidates = item.get(key, [])
            for c in candidates:
                total_candidates += 1
                if c not in vocab:
                    total_dropped += 1

    dropped_ratio = (total_dropped / total_candidates * 100.0) if total_candidates > 0 else 0.0

    print(f"[STATS]")
    print(f"  Total candidates  : {total_candidates}")
    print(f"  Dropped           : {total_dropped}")
    print(f"  Dropped ratio     : {dropped_ratio:.2f}%")

    # Thresholds
    failures = 0
    if dropped_ratio > 30.0:
        print(f"  [FAIL] Dropped ratio {dropped_ratio:.1f}% > 30%")
        failures += 1
    else:
        print(f"  [PASS] Dropped ratio {dropped_ratio:.1f}% <= 30%")

    return failures, {
        "total_candidates": total_candidates,
        "total_dropped": total_dropped,
        "dropped_ratio": dropped_ratio,
    }


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify NeuralPredictor._encode_sentence on regression + benchmark."
    )
    parser.add_argument("--train_corpus", type=str,
                        default="UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu",
                        help="Path to CoNLL-U training corpus for vocab building.")
    parser.add_argument("--benchmark", type=str,
                        default="double_blind_benchmark.json",
                        help="Path to double-blind benchmark JSON.")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of quantifier classes (top_k).")
    parser.add_argument("--skip_unit", action="store_true",
                        help="Skip unit regression tests.")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="Skip benchmark-level audits.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup: build processor + predictor
    # ------------------------------------------------------------------
    print("=" * 80)
    print("SETUP")
    print("=" * 80)

    if not os.path.exists(args.train_corpus):
        print(f"[FATAL] Training corpus not found: {args.train_corpus}")
        print("Please download the UD Chinese GSD treebank first.")
        return

    from run_rigorous_aligned_ablationTJ import BulletproofUDProcessor
    from eval_double_blind_attractor import NeuralPredictor

    processor = BulletproofUDProcessor()
    print(f"[INFO] Building vocab from: {args.train_corpus}")
    all_tasks = processor.extract_tasks(args.train_corpus)
    _, _, _, _ = processor.split_three_way(
        all_tasks, top_k=args.num_classes, k_shot=10, k_calib=10
    )

    vocab_size = len(processor.vocab)
    noun_words = find_noun_words(processor.vocab)
    unk_words = find_unk_words(processor.vocab)
    print(f"[INFO] Vocab size       : {vocab_size}")
    print(f"[INFO] Noun words       : {len(noun_words)}")
    print(f"[INFO] Non-noun words   : {len(unk_words)}")
    print(f"[INFO] quantifier_to_id : {processor.quantifier_to_id}")

    device = torch.device("cpu")
    predictor = NeuralPredictor(
        model=MockModel(num_classes=args.num_classes),
        processor=processor,
        device=device,
        model_name="Verify",
        verbose=True,
    )

    unk_id = processor.vocab.get("<UNK>", 1)

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------
    total_failures = 0

    if not args.skip_unit:
        failures = run_unit_tests(predictor, unk_id)
        total_failures += failures
        print(f"\n[UNIT SUMMARY] Failures: {failures}/{len(UNIT_TESTS)}")

    if not args.skip_benchmark:
        failures_b, _ = audit_benchmark(predictor, args.benchmark, unk_id)
        total_failures += failures_b

        failures_c, _ = audit_candidate_dropped_ratio(predictor, args.benchmark)
        total_failures += failures_c

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if total_failures == 0:
        print("[PASS] All checks passed. Encoding is safe to use.")
    else:
        print(f"[FAIL] {total_failures} check(s) failed. Please review above.")

    # Exit code for CI integration
    exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()