"""
Module: build_double_blind_attractor.py (Final Corrected Edition)
Description:
    Builds a rigorous three-condition (A/B/C) double-blind attractor benchmark
    for Chinese quantifier selection, with PMI-balanced adversarial pairs,
    affordance-preserved Wug tests, open-set candidates, and statistical baselines.

Fixes / Design:
    1. Three-condition (A: forward binding, B: adversarial inversion, C: neutral control).
    2. Exclusive classifier constraint (no generic "个" in closed-set).
    3. PMI-balanced adversarial construction.
    4. Affordance-preserved Wug test with physical attributes.
    5. Open-set (Top-10) and closed-set (2-way) candidates.
    6. Pure statistical baselines (N-gram PMI, FastText similarity).
    7. Naturalness rating interface for human filtering.
    8. Data audit: n-gram contamination check, sample statistics, JSON export.

Corrections over previous version:
    1. Condition C now includes a distractor classifier, enabling variance estimation.
    2. Condition A/B open_candidates deduplicated via dict.fromkeys.
    3. NgramBaseline uses pure PMI with lure only (no max over gold/lure).
    4. FastTextBaseline uses pure cosine with lure only (no mean over words).
    5. filter_pmi_balanced removed unused require_lure_dominant parameter.
"""

import os
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

import numpy as np


# =============================================================================
# 1. LEXICONS
# =============================================================================

EXCLUSIVE_ITEMS: List[Dict[str, str]] = [
    {"core_noun": "木盒", "core_clf": "盒", "attr": "方正木质"},
    {"core_noun": "长凳", "core_clf": "张", "attr": "平整木制"},
    {"core_noun": "宝剑", "core_clf": "柄", "attr": "锋利寒光"},
    {"core_noun": "书籍", "core_clf": "本", "attr": "装订厚重"},
    {"core_noun": "骏马", "core_clf": "匹", "attr": "雄壮矫健"},
    {"core_noun": "高楼", "core_clf": "座", "attr": "巍峨挺拔"},
    {"core_noun": "清泉", "core_clf": "股", "attr": "清澈甘冽"},
    {"core_noun": "宣纸", "core_clf": "张", "attr": "平整轻薄"},
]

WUG_ITEMS: List[Dict[str, str]] = [
    {"wug": "托格", "attr": "修长柔韧", "target_clf": "条"},
    {"wug": "卡德", "attr": "平展单薄", "target_clf": "张"},
    {"wug": "贝斯", "attr": "圆滚坚硬", "target_clf": "颗"},
    {"wug": "穆尔", "attr": "纤细弯曲", "target_clf": "根"},
    {"wug": "塔克", "attr": "厚重宽大", "target_clf": "块"},
]

TOP10_OPEN_SET: List[str] = ["个", "只", "支", "条", "张", "本", "台", "柄", "座", "块"]


# =============================================================================
# 2. CORPUS STATISTICS
# =============================================================================

class CorpusStatistics:
    """
    Wraps external corpus statistics for PMI and co-occurrence frequency.
    """

    def __init__(self, corpus_path: Optional[str] = None) -> None:
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        self.total_tokens = 0
        self.loaded = False

        if corpus_path and os.path.exists(corpus_path):
            self._load_corpus(corpus_path)

    def _load_corpus(self, path: str) -> None:
        print(f"[CORPUS] Loading corpus from {path}...")
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
        print(f"[CORPUS] Loaded: {self.total_tokens} tokens, "
              f"{len(self.unigram)} unigrams, {len(self.bigram)} bigrams")

    def pmi(self, w1: str, w2: str) -> float:
        """
        PMI(w1, w2) = log( P(w1, w2) / (P(w1) * P(w2)) + eps )
        """
        if not self.loaded or self.total_tokens == 0:
            return 0.0

        p_w1 = self.unigram.get(w1, 0) / self.total_tokens
        p_w2 = self.unigram.get(w2, 0) / self.total_tokens
        p_pair = self.bigram.get((w1, w2), 0) / max(self.total_tokens - 1, 1)

        if p_w1 <= 0 or p_w2 <= 0 or p_pair <= 0:
            return float('-inf')

        return math.log(p_pair / (p_w1 * p_w2) + 1e-12)

    def freq(self, w1: str, w2: str) -> int:
        """Co-occurrence frequency of (w1, w2), symmetric."""
        return self.bigram.get((w1, w2), 0) + self.bigram.get((w2, w1), 0)


# =============================================================================
# 3. CONDITION BUILDERS
# =============================================================================

def _dedup(seq: List[str]) -> List[str]:
    """Deduplicate a list while preserving order."""
    return list(dict.fromkeys(seq))


def build_condition_a(wug: Dict[str, str], core: Dict[str, str]) -> Dict[str, Any]:
    """
    Condition A: adjective modifies the Wug head noun; correct classifier = target_clf.
    """
    return {
        "condition": "A_Forward_Binding",
        "sentence": f"这[MASK]摆放在{core['core_noun']}内部的、{wug['attr']}的{wug['wug']}。",
        "gold_clf": wug["target_clf"],
        "lure_clf": core["core_clf"],
        "closed_candidates": _dedup([wug["target_clf"], core["core_clf"]]),
        "open_candidates": _dedup([wug["target_clf"], core["core_clf"]] + TOP10_OPEN_SET),
        "core_noun": core["core_noun"],
        "wug": wug["wug"],
        "attr": wug["attr"],
    }


def build_condition_b(wug: Dict[str, str], core: Dict[str, str]) -> Dict[str, Any]:
    """
    Condition B: adjective modifies the Wug inside a relative clause; head noun = core.
    Correct classifier = core_clf, NOT target_clf.
    """
    return {
        "condition": "B_Adversarial_Inversion",
        "sentence": f"这[MASK]装着{wug['attr']}的{wug['wug']}的{core['core_noun']}。",
        "gold_clf": core["core_clf"],
        "lure_clf": wug["target_clf"],
        "closed_candidates": _dedup([core["core_clf"], wug["target_clf"]]),
        "open_candidates": _dedup([core["core_clf"], wug["target_clf"]] + TOP10_OPEN_SET),
        "core_noun": core["core_noun"],
        "wug": wug["wug"],
        "attr": wug["attr"],
    }


def build_condition_c(core: Dict[str, str], sibling: Dict[str, str]) -> Dict[str, Any]:
    """
    Condition C: Neutral control. Head noun shares classifier with sibling, but a
    distractor classifier is added to allow accuracy variance estimation.

    The distractor is chosen from EXCLUSIVE_ITEMS with a different classifier
    from the head noun's.
    """
    other_clfs = [c["core_clf"] for c in EXCLUSIVE_ITEMS
                  if c["core_clf"] != core["core_clf"]]
    distractor_clf = random.choice(other_clfs) if other_clfs else None

    closed = [core["core_clf"]]
    if distractor_clf:
        closed.append(distractor_clf)

    return {
        "condition": "C_Neutral_Control",
        "sentence": f"这[MASK]摆放在{sibling['core_noun']}旁边的、{core['attr']}的{core['core_noun']}。",
        "gold_clf": core["core_clf"],
        "lure_clf": distractor_clf if distractor_clf else core["core_clf"],
        "closed_candidates": _dedup(closed),
        "open_candidates": _dedup(closed + TOP10_OPEN_SET),
        "core_noun": core["core_noun"],
        "sibling_noun": sibling["core_noun"],
        "attr": core["attr"],
    }


def generate_three_condition_benchmark(
    include_condition_c: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate the full three-condition benchmark with cross-product of items.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset: List[Dict[str, Any]] = []
    uid = 0

    for wug in WUG_ITEMS:
        for core in EXCLUSIVE_ITEMS:
            if wug["target_clf"] == core["core_clf"]:
                continue

            # Condition A
            uid += 1
            item_a = build_condition_a(wug, core)
            item_a["id"] = f"condA_{uid:04d}"
            dataset.append(item_a)

            # Condition B
            uid += 1
            item_b = build_condition_b(wug, core)
            item_b["id"] = f"condB_{uid:04d}"
            dataset.append(item_b)

            # Condition C (optional)
            if include_condition_c:
                siblings = [c for c in EXCLUSIVE_ITEMS
                            if c["core_clf"] == core["core_clf"]
                            and c["core_noun"] != core["core_noun"]]
                if siblings:
                    sibling = random.choice(siblings)
                    uid += 1
                    item_c = build_condition_c(core, sibling)
                    item_c["id"] = f"condC_{uid:04d}"
                    dataset.append(item_c)

    return dataset


# =============================================================================
# 4. PMI-BALANCED FILTERING
# =============================================================================

def filter_pmi_balanced(
    dataset: List[Dict[str, Any]],
    corpus: CorpusStatistics,
    min_advantage: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter Condition B items where the lure has stronger PMI/frequency than gold.

    Hard items satisfy:
        pmi(lure_clf, lure_noun) - pmi(gold_clf, gold_noun) >= min_advantage
        and
        freq(lure_clf, lure_noun) >= freq(gold_clf, gold_noun)

    Returns:
        (hard_items, easy_items)
    """
    hard_items, easy_items = [], []

    if not corpus.loaded:
        print("[WARNING] Corpus not loaded, skipping PMI filtering.")
        return dataset, []

    for item in dataset:
        if item["condition"] != "B_Adversarial_Inversion":
            continue

        lure_noun = item["wug"]
        gold_noun = item["core_noun"]
        lure_clf = item["lure_clf"]
        gold_clf = item["gold_clf"]

        pmi_lure = corpus.pmi(lure_clf, lure_noun)
        pmi_gold = corpus.pmi(gold_clf, gold_noun)
        freq_lure = corpus.freq(lure_clf, lure_noun)
        freq_gold = corpus.freq(gold_clf, gold_noun)

        item["pmi_lure"] = pmi_lure
        item["pmi_gold"] = pmi_gold
        item["freq_lure"] = freq_lure
        item["freq_gold"] = freq_gold

        is_hard = (pmi_lure - pmi_gold) >= min_advantage and freq_lure >= freq_gold
        if is_hard:
            hard_items.append(item)
        else:
            easy_items.append(item)

    print(f"[PMI-FILTER] Hard (lure dominant): {len(hard_items)} | "
          f"Easy: {len(easy_items)}")
    return hard_items, easy_items


# =============================================================================
# 5. STATISTICAL BASELINES (PURE, NO GOLD PEEKING)
# =============================================================================

class NgramBaseline:
    """
    Pure N-gram PMI baseline: choose classifier with highest PMI with the
    linearly nearest noun (lure). Does NOT peek at the gold noun.
    """

    def __init__(self, corpus: CorpusStatistics) -> None:
        self.corpus = corpus

    def predict(self, item: Dict[str, Any]) -> str:
        candidates = item.get("closed_candidates", [])
        if not candidates:
            return ""

        lure_noun = item.get("wug", "")
        if not lure_noun:
            return candidates[0]

        best_clf, best_score = candidates[0], float('-inf')
        for clf in candidates:
            score = self.corpus.pmi(clf, lure_noun)
            if score > best_score:
                best_score = score
                best_clf = clf
        return best_clf


class FastTextBaseline:
    """
    Pure FastText similarity baseline: choose classifier with highest cosine
    similarity to the linearly nearest noun (lure). Does NOT peek at the gold noun.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                from gensim.models import FastText
                self.model = FastText.load(model_path)
                print(f"[FASTTEXT] Loaded model from {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load FastText: {e}")

    def predict(self, item: Dict[str, Any]) -> str:
        candidates = item.get("closed_candidates", [])
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
# 6. DATA AUDIT
# =============================================================================

def audit_dataset(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Report condition distribution, classifier distribution, sentence length stats.
    """
    condition_counts = Counter(item["condition"] for item in dataset)
    gold_clf_counts = Counter(item["gold_clf"] for item in dataset)
    lure_clf_counts = Counter(item["lure_clf"] for item in dataset)

    closed_sizes = Counter(len(item.get("closed_candidates", [])) for item in dataset)
    open_sizes = Counter(len(item.get("open_candidates", [])) for item in dataset)

    sent_lengths = [len(item["sentence"]) for item in dataset]

    audit = {
        "total": len(dataset),
        "condition_counts": dict(condition_counts),
        "gold_clf_counts": dict(gold_clf_counts),
        "lure_clf_counts": dict(lure_clf_counts),
        "closed_candidate_sizes": dict(closed_sizes),
        "open_candidate_sizes": dict(open_sizes),
        "sentence_length": {
            "mean": float(np.mean(sent_lengths)),
            "std": float(np.std(sent_lengths)),
            "min": int(np.min(sent_lengths)),
            "max": int(np.max(sent_lengths)),
        },
    }
    return audit


def check_contamination(
    dataset: List[Dict[str, Any]],
    train_sentences: List[str],
    n_gram: int = 4,
) -> float:
    """
    Check n-gram overlap between benchmark sentences and training set.
    """
    def ngrams(text: str, n: int) -> set:
        tokens = list(text)
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    train_ngrams = set()
    for sent in train_sentences:
        train_ngrams.update(ngrams(sent, n_gram))

    overlaps = 0
    for item in dataset:
        if ngrams(item["sentence"], n_gram) & train_ngrams:
            overlaps += 1

    ratio = overlaps / len(dataset) * 100 if dataset else 0.0
    print(f"[CONTAMINATION] {n_gram}-gram overlap: "
          f"{overlaps}/{len(dataset)} ({ratio:.2f}%)")
    return ratio


# =============================================================================
# 7. EXPORT
# =============================================================================

def export_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """
    Export dataset to JSON with audit info.
    """
    audit = audit_dataset(dataset)
    payload = {"audit": audit, "samples": dataset}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[EXPORT] Saved {len(dataset)} samples to {output_path}")
    print(f"[AUDIT] {json.dumps(audit, ensure_ascii=False, indent=2)}")


def export_baseline_predictions(
    dataset: List[Dict[str, Any]],
    ngram: NgramBaseline,
    fasttext: FastTextBaseline,
    output_path: str,
) -> None:
    """
    Optionally precompute baseline predictions for inspection.
    """
    records = []
    for item in dataset:
        rec = {
            "id": item["id"],
            "condition": item["condition"],
            "gold_clf": item["gold_clf"],
            "lure_clf": item["lure_clf"],
            "ngram_pred": ngram.predict(item) if ngram else None,
            "fasttext_pred": fasttext.predict(item) if fasttext and fasttext.model else None,
        }
        records.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[EXPORT] Baseline predictions saved to {output_path}")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build double-blind attractor benchmark."
    )
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to external corpus for PMI.")
    parser.add_argument("--fasttext", type=str, default=None,
                        help="Path to FastText model.")
    parser.add_argument("--output", type=str, default="double_blind_benchmark.json")
    parser.add_argument("--output_baseline", type=str,
                        default="double_blind_baseline_predictions.json")
    parser.add_argument("--include_condition_c", action="store_true", default=True)
    parser.add_argument("--no_condition_c", dest="include_condition_c",
                        action="store_false")
    parser.add_argument("--min_pmi_advantage", type=float, default=0.0)
    parser.add_argument("--n_gram", type=int, default=4)
    parser.add_argument("--train_sentences", type=str, default=None,
                        help="Optional path to training sentences for contamination check.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[STEP 1] Generating three-condition benchmark...")
    dataset = generate_three_condition_benchmark(
        include_condition_c=args.include_condition_c,
        seed=args.seed,
    )
    print(f"[INFO] Generated {len(dataset)} raw items.")

    print("[STEP 2] Loading corpus statistics...")
    corpus = CorpusStatistics(args.corpus)

    if corpus.loaded:
        print("[STEP 3] PMI-balanced adversarial filtering...")
        hard_items, easy_items = filter_pmi_balanced(
            dataset, corpus, min_advantage=args.min_pmi_advantage
        )
        print(f"[INFO] Hard items: {len(hard_items)} | Easy items: {len(easy_items)}")
    else:
        print("[SKIP] Corpus not loaded, skipping PMI filtering.")

    print("[STEP 4] Auditing dataset...")
    audit = audit_dataset(dataset)
    print(f"[AUDIT] {json.dumps(audit, ensure_ascii=False, indent=2)}")

    if args.train_sentences and os.path.exists(args.train_sentences):
        print("[STEP 4.5] Contamination check...")
        with open(args.train_sentences, "r", encoding="utf-8") as f:
            train_sents = [line.strip() for line in f if line.strip()]
        check_contamination(dataset, train_sents, n_gram=args.n_gram)

    print("[STEP 5] Exporting benchmark...")
    export_dataset(dataset, args.output)

    print("[STEP 6] Precomputing baseline predictions...")
    ngram = NgramBaseline(corpus)
    fasttext = FastTextBaseline(args.fasttext)
    export_baseline_predictions(dataset, ngram, fasttext, args.output_baseline)

    print("[DONE] Double-blind benchmark construction complete.")


if __name__ == "__main__":
    main()