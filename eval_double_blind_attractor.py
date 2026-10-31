"""
Module: eval_double_blind_attractor.py (Final Corrected Edition)
Description:
    Evaluate CAFT, baselines, statistical models, and optionally LLMs on the
    three-condition (A/B/C) double-blind attractor benchmark.

Usage:
    python eval_double_blind_attractor.py \
        --benchmark double_blind_benchmark.json \
        --train_corpus UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu \
        --caft_ckpt /path/to/caft_model.pth \
        --baseline_ckpt /path/to/baseline_model.pth \
        --prior_ckpt /path/to/meta_pretrained_prior.pth \
        --corpus /path/to/corpus.txt \
        --fasttext /path/to/fasttext.model \
        --output results/double_blind_results.json

Outputs:
    - Per-condition accuracy for each model
    - Per-condition macro F1
    - Open-set vs closed-set accuracy
    - A -> B accuracy drop (adversarial inversion)
    - JSON result file + console table

Fixes over previous version:
    1. Added --train_corpus parameter to build vocab correctly for neural models.
       Without it, NeuralPredictor encodes all Chinese characters as <UNK>,
       making CAFT / Baseline predictions meaningless.
    2. processor.quantifier_to_id is now aligned with the training-time mapping
       via processor.split_three_way(), ensuring consistent class id assignment.
    3. Retained fallback for statistical-only evaluation (no train corpus).
    4. _encode_sentence now uses Max-Length Forward Matching over noun tokens
       (id in [100, 2000)) to preserve whole-word semantics and avoid splitting
       multi-character nouns into <UNK>.
"""

import os
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_pretrain_full_strong import NeuralTransformerStudent
from run_rigorous_aligned_ablationTJ import (
    BulletproofUDProcessor,
    calculate_macro_f1,
)


# =============================================================================
# 1. BENCHMARK LOADING
# =============================================================================

def load_benchmark(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"[BENCHMARK] Loaded {len(payload['samples'])} samples from {path}")
    print(f"[BENCHMARK] Audit: {json.dumps(payload['audit'], ensure_ascii=False)[:200]}...")
    return payload


def extract_candidate_list(item: Dict[str, Any], candidate_type: str) -> List[str]:
    key = f"{candidate_type}_candidates"
    if key not in item:
        return []
    return list(dict.fromkeys(item[key]))


# =============================================================================
# 2. STATISTICAL BASELINES
# =============================================================================

class NgramBaseline:
    """
    Pure statistical N-gram baseline: choose classifier with highest PMI
    with the linearly nearest noun (lure).
    """

    def __init__(self, corpus_path: Optional[str] = None) -> None:
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        self.total_tokens = 0
        self.loaded = False
        if corpus_path and os.path.exists(corpus_path):
            self._load_corpus(corpus_path)

    def _load_corpus(self, path: str) -> None:
        print(f"[NGRAM] Loading corpus: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                self.unigram.update(tokens)
                self.total_tokens += len(tokens)
                for i in range(len(tokens) - 1):
                    self.bigram[(tokens[i], tokens[i + 1])] += 1
        self.loaded = True
        print(f"[NGRAM] Loaded {self.total_tokens} tokens, "
              f"{len(self.unigram)} unigrams, {len(self.bigram)} bigrams")

    def _pmi(self, w1: str, w2: str) -> float:
        if not self.loaded or self.total_tokens == 0:
            return 0.0
        p_w1 = self.unigram.get(w1, 0) / self.total_tokens
        p_w2 = self.unigram.get(w2, 0) / self.total_tokens
        p_pair = self.bigram.get((w1, w2), 0) / max(self.total_tokens - 1, 1)
        if p_w1 <= 0 or p_w2 <= 0 or p_pair <= 0:
            return float('-inf')
        return math.log(p_pair / (p_w1 * p_w2) + 1e-12)

    def predict(self, item: Dict[str, Any], candidates: List[str]) -> str:
        if not candidates:
            return ""
        lure_noun = item.get("wug", "")
        if not lure_noun:
            return candidates[0]
        best_clf, best_score = candidates[0], float('-inf')
        for clf in candidates:
            score = self._pmi(clf, lure_noun)
            if score > best_score:
                best_score = score
                best_clf = clf
        return best_clf


class FastTextBaseline:
    """
    Pure FastText similarity baseline: choose classifier with highest cosine
    similarity to the linearly nearest noun (lure).
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                from gensim.models import FastText
                self.model = FastText.load(model_path)
                print(f"[FASTTEXT] Loaded: {model_path}")
            except Exception as e:
                print(f"[FASTTEXT] Failed to load: {e}")

    def predict(self, item: Dict[str, Any], candidates: List[str]) -> str:
        if not candidates or self.model is None:
            return random.choice(candidates) if candidates else ""

        lure_noun = item.get("wug", "")
        if not lure_noun or lure_noun not in self.model.wv:
            return candidates[0]

        best_clf, best_score = candidates[0], float('-inf')
        for clf in candidates:
            if clf not in self.model.wv:
                continue
            score = self.model.wv.similarity(clf, lure_noun)
            if score > best_score:
                best_score = score
                best_clf = clf
        return best_clf


# =============================================================================
# 3. NEURAL MODEL PREDICTOR (REAL IMPLEMENTATION)
# =============================================================================

class NeuralPredictor:
    """
    Wrapper for CAFT / Baseline Transformer prediction on the double-blind benchmark.

    The benchmark sentences are raw text with a [MASK] placeholder. To feed them
    into a model trained on UD CoNLL-U vocab, we must re-encode them with the SAME
    vocab mapping used during training. This predictor therefore accepts a shared
    `BulletproofUDProcessor` and encodes each benchmark sentence token-by-token.

    Encoding strategy (Max-Length Forward Matching):
        - [MASK] is mapped to the dedicated <MASK> id.
        - Noun tokens (id in [100, 2000)) are matched greedily (4 -> 2 chars)
          to preserve whole-word semantics and avoid splitting multi-character
          nouns into single-character <UNK>.
        - Fallback: per-character lookup; unknown chars -> <UNK>.

    Candidate filtering:
        Only candidates whose character is present in the processor vocab are
        kept. Any dropped candidates are reported for transparency.
    """

    # Noun id interval (matches BulletproofUDProcessor.get_id assignment)
    NOUN_ID_MIN = 100
    NOUN_ID_MAX = 2000

    def __init__(
        self,
        model: nn.Module,
        processor: BulletproofUDProcessor,
        device: torch.device,
        model_name: str = "Neural",
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.model.eval()
        self.processor = processor
        self.device = device
        self.model_name = model_name
        self.verbose = verbose
        self.dropped_candidate_count = 0
        self.total_candidate_count = 0

        # Precompute noun lexicon once (id interval [100, 2000))
        self.noun_words = {
            w for w, i in self.processor.vocab.items()
            if self.NOUN_ID_MIN <= i < self.NOUN_ID_MAX
        }
        if verbose:
            print(f"[{self.model_name}] Noun lexicon size: {len(self.noun_words)}")

    def _encode_sentence(self, sentence: str) -> Optional[torch.Tensor]:
        """
        Encode a benchmark sentence (with [MASK]) into token ids.

        Strategy (Max-Length Forward Matching):
            1. [MASK] -> <MASK> id.
            2. For known NOUN words (id in [100, 2000)), perform longest-match
               forward matching (4 -> 2 characters) to preserve whole-word
               semantics.
            3. Fallback: per-character lookup; unknown chars -> <UNK>.

        Note:
            Using `len(w) > 1` to identify nouns would incorrectly exclude
            single-character nouns such as "书" / "马". Instead, we rely on
            the id interval defined by BulletproofUDProcessor.get_id().

        Returns:
            [1, L] tensor, or None if encoding failed.
        """
        vocab = self.processor.vocab
        unk_id = vocab.get("<UNK>", 1)
        mask_id = vocab.get("<MASK>", 4)

        ids: List[int] = []
        i = 0
        n = len(sentence)

        while i < n:
            # 1. [MASK] special-case
            if sentence.startswith("[MASK]", i):
                ids.append(mask_id)
                i += len("[MASK]")
                continue

            # 2. Longest forward match on nouns (4 -> 2 chars)
            matched = False
            for length in range(min(4, n - i), 1, -1):
                sub = sentence[i:i + length]
                if sub in self.noun_words:
                    ids.append(vocab[sub])
                    i += length
                    matched = True
                    break
            if matched:
                continue

            # 3. Fallback: per-character lookup
            ch = sentence[i]
            if ch.isspace():
                i += 1
                continue
            ids.append(vocab.get(ch, unk_id))
            i += 1

        # Truncate to model max context if defined
        if hasattr(self.processor, "max_len"):
            ids = ids[:self.processor.max_len]

        if not ids:
            return None
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _filter_candidates(self, candidates: List[str]) -> List[str]:
        """
        Keep only candidates whose character is present in processor vocab.
        """
        vocab = self.processor.vocab
        kept = []
        for c in candidates:
            self.total_candidate_count += 1
            if c in vocab:
                kept.append(c)
            else:
                self.dropped_candidate_count += 1
        return kept

    def predict(self, item: Dict[str, Any], candidates: List[str]) -> str:
        """
        Predict the classifier from a candidate list.

        Returns:
            Predicted classifier string, or "" if prediction failed.
        """
        if not candidates:
            return ""

        kept = self._filter_candidates(candidates)
        if not kept:
            return candidates[0]

        sentence = item.get("sentence", "")
        if not sentence:
            return kept[0]

        X = self._encode_sentence(sentence)
        if X is None:
            return kept[0]

        with torch.no_grad():
            logits = self.model(X)          # [1, num_classes]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        # Map candidate classifier -> class id via processor.quantifier_to_id
        q2id = self.processor.quantifier_to_id
        best_clf, best_score = kept[0], float('-inf')
        for clf in kept:
            cid = q2id.get(clf, None)
            if cid is None:
                continue
            score = float(probs[cid])
            if score > best_score:
                best_score = score
                best_clf = clf
        return best_clf

    def report_dropped_candidates(self) -> None:
        if self.total_candidate_count > 0:
            ratio = self.dropped_candidate_count / self.total_candidate_count * 100
            print(f"[{self.model_name}] Dropped {self.dropped_candidate_count}/"
                  f"{self.total_candidate_count} candidates not in vocab "
                  f"({ratio:.2f}%)")


# =============================================================================
# 4. NEURAL MODEL LOADING
# =============================================================================

def load_neural_model(
    ckpt_path: str,
    processor: BulletproofUDProcessor,
    num_classes: int,
    device: torch.device,
    use_prior: bool = False,
    prior_path: Optional[str] = None,
    prior_alpha: float = 0.4,
) -> nn.Module:
    """
    Load a neural model checkpoint for evaluation.

    If use_prior is True, first soft-load the prior weights with alpha blending,
    then overwrite with the fine-tuned checkpoint.
    """
    model = NeuralTransformerStudent(vocab_size=25000, num_classes=num_classes).to(device)

    if use_prior and prior_path and os.path.exists(prior_path):
        from train_sota_few_shot import load_prior_weights_soft
        load_prior_weights_soft(model, prior_path, device, alpha=prior_alpha)

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state)
            print(f"[MODEL] Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[WARNING] Partial load for {ckpt_path}: {e}")
            model.load_state_dict(state, strict=False)
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path}. Using initialized weights.")

    model.eval()
    return model


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(
    predictor: Any,
    dataset: List[Dict[str, Any]],
    candidate_type: str = "closed",
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a model on the full dataset with given candidate type.
    """
    per_condition = defaultdict(lambda: {"correct": 0, "total": 0})
    per_classifier = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds, all_golds = [], []

    for item in dataset:
        candidates = extract_candidate_list(item, candidate_type)
        if not candidates:
            continue

        pred = predictor.predict(item, candidates)
        gold = item["gold_clf"]
        condition = item["condition"]

        correct = 1 if pred == gold else 0
        per_condition[condition]["correct"] += correct
        per_condition[condition]["total"] += 1
        per_classifier[gold]["correct"] += correct
        per_classifier[gold]["total"] += 1

        all_preds.append(pred)
        all_golds.append(gold)

    overall_correct = sum(v["correct"] for v in per_condition.values())
    overall_total = sum(v["total"] for v in per_condition.values())
    overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0

    cond_a = per_condition.get("A_Forward_Binding", {"correct": 0, "total": 0})
    cond_b = per_condition.get("B_Adversarial_Inversion", {"correct": 0, "total": 0})
    acc_a = (cond_a["correct"] / cond_a["total"] * 100) if cond_a["total"] > 0 else 0.0
    acc_b = (cond_b["correct"] / cond_b["total"] * 100) if cond_b["total"] > 0 else 0.0
    a_to_b_drop = acc_a - acc_b

    result = {
        "model_name": model_name,
        "candidate_type": candidate_type,
        "per_condition": {
            k: {
                "acc": (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0.0,
                "n": v["total"],
                "correct": v["correct"],
            }
            for k, v in per_condition.items()
        },
        "overall": {
            "acc": overall_acc,
            "n": overall_total,
            "correct": overall_correct,
        },
        "per_classifier": {
            k: {
                "acc": (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0.0,
                "n": v["total"],
                "correct": v["correct"],
            }
            for k, v in per_classifier.items()
        },
        "a_to_b_drop": a_to_b_drop,
    }

    print(f"\n[{model_name} | {candidate_type}-set]")
    print(f"  Overall: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    for cond, stats in result["per_condition"].items():
        print(f"  {cond}: {stats['acc']:.2f}% ({stats['correct']}/{stats['n']})")
    print(f"  A -> B drop: {a_to_b_drop:.2f} pp")

    return result


# =============================================================================
# 6. REPORTING
# =============================================================================

def build_summary_table(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("=" * 120)
    lines.append(
        f"| {'Model':<30} | {'Cand':<8} | {'Overall':<10} | "
        f"{'A (Forward)':<12} | {'B (Adversarial)':<15} | "
        f"{'C (Neutral)':<12} | {'A->B drop':<10} |"
    )
    lines.append("-" * 120)

    for r in all_results:
        overall = r["overall"]["acc"]
        cond_a = r["per_condition"].get("A_Forward_Binding", {}).get("acc", 0.0)
        cond_b = r["per_condition"].get("B_Adversarial_Inversion", {}).get("acc", 0.0)
        cond_c = r["per_condition"].get("C_Neutral_Control", {}).get("acc", 0.0)
        drop = r.get("a_to_b_drop", cond_a - cond_b)

        lines.append(
            f"| {r['model_name']:<30} | {r['candidate_type']:<8} | "
            f"{overall:>6.2f}%    | {cond_a:>6.2f}%      | {cond_b:>6.2f}%         | "
            f"{cond_c:>6.2f}%      | {drop:>6.2f} pp |"
        )

    lines.append("=" * 120)
    return "\n".join(lines)


def export_results(all_results: List[Dict[str, Any]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[EXPORT] Results saved to {output_path}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on double-blind benchmark.")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to double-blind benchmark JSON.")
    parser.add_argument("--train_corpus", type=str, default=None,
                        help="Path to UD CoNLL-U training corpus for vocab building. "
                             "MUST be the same corpus used during training. "
                             "If omitted, neural predictors will encode all Chinese "
                             "characters as <UNK> and predictions will be meaningless.")
    parser.add_argument("--caft_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--prior_ckpt", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of quantifier classes. MUST match training-time top_k.")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to external text corpus for N-gram PMI baseline.")
    parser.add_argument("--fasttext", type=str, default=None,
                        help="Path to FastText model for FastText baseline.")
    parser.add_argument("--output", type=str, default="double_blind_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    payload = load_benchmark(args.benchmark)
    dataset = payload["samples"]

    all_results: List[Dict[str, Any]] = []

    # ------------------------
    # Random baseline
    # ------------------------
    class RandomBaseline:
        def predict(self, item, candidates):
            return random.choice(candidates) if candidates else ""

    all_results.append(evaluate_model(RandomBaseline(), dataset, "closed", "Random (closed)"))
    all_results.append(evaluate_model(RandomBaseline(), dataset, "open", "Random (open)"))

    # ------------------------
    # N-gram baseline
    # ------------------------
    ngram = NgramBaseline(args.corpus)
    if ngram.loaded:
        all_results.append(evaluate_model(ngram, dataset, "closed", "N-gram (closed)"))
        all_results.append(evaluate_model(ngram, dataset, "open", "N-gram (open)"))

    # ------------------------
    # FastText baseline
    # ------------------------
    ft = FastTextBaseline(args.fasttext)
    if ft.model is not None:
        all_results.append(evaluate_model(ft, dataset, "closed", "FastText (closed)"))
        all_results.append(evaluate_model(ft, dataset, "open", "FastText (open)"))

    # ------------------------
    # Shared processor for neural models
    # ------------------------
    processor = BulletproofUDProcessor()

    if args.train_corpus and os.path.exists(args.train_corpus):
        print(f"[INFO] Loading training corpus to build vocab: {args.train_corpus}")
        all_tasks = processor.extract_tasks(args.train_corpus)
        _, _, _, _ = processor.split_three_way(
            all_tasks,
            top_k=args.num_classes,
            k_shot=10,
            k_calib=10,
        )
        print(f"[INFO] Processor vocab size: {len(processor.vocab)}")
        print(f"[INFO] Processor quantifier_to_id: {processor.quantifier_to_id}")
    else:
        print("[WARNING] Training corpus not provided. "
              "Neural predictors will encode all Chinese characters as <UNK>. "
              "CAFT / Baseline predictions will be meaningless.")
        unique_gold = sorted(set(item["gold_clf"] for item in dataset))
        processor.quantifier_to_id = {q: i for i, q in enumerate(unique_gold[:args.num_classes])}
        print(f"[INFO] Fallback quantifier_to_id: {processor.quantifier_to_id}")

    # ------------------------
    # Baseline neural model
    # ------------------------
    if args.baseline_ckpt:
        if not args.train_corpus:
            print("[WARNING] --baseline_ckpt provided but --train_corpus missing. "
                  "Predictions will use <UNK>-only vocab.")

        baseline_model = load_neural_model(
            ckpt_path=args.baseline_ckpt,
            processor=processor,
            num_classes=args.num_classes,
            device=device,
            use_prior=False,
        )
        baseline_predictor = NeuralPredictor(
            model=baseline_model,
            processor=processor,
            device=device,
            model_name="Baseline",
            verbose=True,
        )
        all_results.append(evaluate_model(baseline_predictor, dataset, "closed", "Baseline (closed)"))
        all_results.append(evaluate_model(baseline_predictor, dataset, "open", "Baseline (open)"))
        baseline_predictor.report_dropped_candidates()

    # ------------------------
    # CAFT neural model
    # ------------------------
    if args.caft_ckpt:
        if not args.train_corpus:
            print("[WARNING] --caft_ckpt provided but --train_corpus missing. "
                  "Predictions will use <UNK>-only vocab.")

        caft_model = load_neural_model(
            ckpt_path=args.caft_ckpt,
            processor=processor,
            num_classes=args.num_classes,
            device=device,
            use_prior=(args.prior_ckpt is not None),
            prior_path=args.prior_ckpt,
            prior_alpha=0.4,
        )
        caft_predictor = NeuralPredictor(
            model=caft_model,
            processor=processor,
            device=device,
            model_name="CAFT",
            verbose=True,
        )
        all_results.append(evaluate_model(caft_predictor, dataset, "closed", "CAFT (closed)"))
        all_results.append(evaluate_model(caft_predictor, dataset, "open", "CAFT (open)"))
        caft_predictor.report_dropped_candidates()

    # ------------------------
    # Summary
    # ------------------------
    print("\n" + build_summary_table(all_results))
    export_results(all_results, args.output)


if __name__ == "__main__":
    main()