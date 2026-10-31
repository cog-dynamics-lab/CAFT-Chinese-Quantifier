"""
Module: eval_llm_rigorous.py (Final Corrected Edition)
Description:
    Rigorous LLM benchmarking suite for Chinese quantifier selection.
    Implements length-normalized autoregressive prefix scoring,
    per-template isolated ECE with stratified bootstrap,
    cross-template weighted aggregation, and sample efficiency curves.

Fixes over previous version:
    1. Cross-template weighted aggregation by sample count.
    2. Per-template isolated ECE with stratified bootstrap.
    3. Cross-template std explicitly labeled as inter-template variance.
    4. Direct comparison of length-normalized log-probs with z-score softmax.
    5. Subword token length audit explicitly logged.
    6. Multi-seed few-shot example selection with variance reporting.
    7. Sample efficiency curve for K in {1, 3, 5, 10, 20, 50, 100}.
    8. Complete run_llm_benchmark main entry point with __main__ guard.
    9. Type hints and boundary checks added.
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any
from collections import Counter

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ud_kshot_ablation import compute_stratified_bootstrap_ece


# =============================================================================
# 1. PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES: List[str] = [
    "句子：{sent} 请填入正确量词：",
    "输入：{sent} 符合语法的量词答案：",
    "填空题：{sent} 答案是：",
]


# =============================================================================
# 2. LENGTH-NORMALIZED AUTOREGRESSIVE SCORING
# =============================================================================

def score_sequence_normalized_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt_prefix: str,
    candidate: str,
    device: torch.device = torch.device('cuda'),
) -> Tuple[float, int]:
    """
    Compute length-normalized autoregressive prefix log-probability for a candidate.

    Returns:
        (norm_log_prob, num_subword_tokens)
    """
    model.eval()
    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    num_tokens = len(cand_ids)

    if len(prefix_ids) == 0:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        prefix_ids = [bos]

    if num_tokens == 0:
        return 0.0, 0

    full_ids = torch.tensor([prefix_ids + cand_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(full_ids).logits  # [1, L, V]

    start_pos = len(prefix_ids)
    target_logits = logits[0, start_pos - 1: start_pos + num_tokens - 1, :]
    log_probs = torch.log_softmax(target_logits, dim=-1)

    cand_indices = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(1)
    token_log_probs = log_probs.gather(1, cand_indices).squeeze(1)
    total_log_prob = token_log_probs.sum().item()

    norm_log_prob = total_log_prob / max(num_tokens, 1)
    return norm_log_prob, num_tokens


# =============================================================================
# 3. SINGLE-TEMPLATE EVALUATION
# =============================================================================

def evaluate_single_template(
    model: nn.Module,
    tokenizer: Any,
    test_samples: List[Dict[str, str]],
    label_vocab: List[str],
    template: str,
    device: torch.device = torch.device('cuda'),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single prompt template.

    Returns:
        {
            'accuracy': float,
            'num_samples': int,
            'confidences': np.ndarray,
            'correct_flags': np.ndarray,
            'true_labels': np.ndarray,
            'ece_mean': float,
            'ece_low': float,
            'ece_high': float,
        }
    """
    model.eval()
    label_to_idx = {c: i for i, c in enumerate(label_vocab)}

    correct_flags: List[int] = []
    confidences: List[float] = []
    true_labels: List[int] = []

    for sample in test_samples:
        sent_masked = sample['sentence'].replace("[MASK]", "____")
        prompt = template.format(sent=sent_masked)

        scores = []
        for c in label_vocab:
            s, _ = score_sequence_normalized_logprob(model, tokenizer, prompt, c, device)
            scores.append(s)

        # Direct comparison via z-score softmax: scale-free confidence
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        pred_idx = int(torch.argmax(scores_tensor).item())
        pred_label = label_vocab[pred_idx]

        z = (scores_tensor - scores_tensor.mean()) / (scores_tensor.std() + 1e-8)
        probs = torch.softmax(z, dim=-1).numpy()
        confidence = float(probs[pred_idx])

        gold_clf = sample['correct_clf']
        is_corr = 1 if pred_label == gold_clf else 0

        correct_flags.append(is_corr)
        confidences.append(confidence)
        true_labels.append(label_to_idx.get(gold_clf, 0))

    confidences_np = np.array(confidences)
    correct_flags_np = np.array(correct_flags)
    true_labels_np = np.array(true_labels)

    acc = float(np.mean(correct_flags_np) * 100)

    ece_mean, ece_low, ece_high = compute_stratified_bootstrap_ece(
        confidences=confidences_np,
        correct_flags=correct_flags_np,
        labels=true_labels_np,
        verbose=False,
    )

    if verbose:
        print(f"[TEMPLATE-EVAL] acc={acc:.2f} | ECE={ece_mean:.4f} "
              f"[{ece_low:.4f}, {ece_high:.4f}] | n={len(test_samples)}")

    return {
        'accuracy': acc,
        'num_samples': len(test_samples),
        'confidences': confidences_np,
        'correct_flags': correct_flags_np,
        'true_labels': true_labels_np,
        'ece_mean': ece_mean,
        'ece_low': ece_low,
        'ece_high': ece_high,
    }


# =============================================================================
# 4. CROSS-TEMPLATE WEIGHTED AGGREGATION (CORRECTED)
# =============================================================================

def weighted_mean_std(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute weighted mean and weighted std.

    Weighted mean: sum(w_i * x_i) / sum(w_i)
    Weighted variance: sum(w_i * (x_i - mean)^2) / sum(w_i)

    Args:
        values: array of values.
        weights: array of non-negative weights.

    Returns:
        (weighted_mean, weighted_std)
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0
    weights_norm = weights / total
    mean = float(np.sum(weights_norm * values))
    var = float(np.sum(weights_norm * (values - mean) ** 2))
    return mean, float(np.sqrt(var))


def aggregate_template_results(
    template_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-template results with sample-count weighting.

    Returns:
        {
            'acc_weighted_mean': float,
            'acc_weighted_std': float,           # 加权标准差
            'acc_unweighted_mean': float,
            'acc_inter_template_std': float,     # 提示词敏感度
            'ece_weighted_mean': float,
            'ece_weighted_std': float,           # 加权标准差
            'ece_inter_template_std': float,     # 提示词敏感度
            'total_samples': int,
            'per_template_sample_counts': List[int],
            'per_template_accs': List[float],
            'per_template_eces': List[float],
        }
    """
    empty = {
        'acc_weighted_mean': 0.0,
        'acc_weighted_std': 0.0,
        'acc_unweighted_mean': 0.0,
        'acc_inter_template_std': 0.0,
        'ece_weighted_mean': 0.0,
        'ece_weighted_std': 0.0,
        'ece_inter_template_std': 0.0,
        'total_samples': 0,
        'per_template_sample_counts': [],
        'per_template_accs': [],
        'per_template_eces': [],
    }

    if not template_results:
        return empty

    accs = np.array([r['accuracy'] for r in template_results])
    eces = np.array([r['ece_mean'] for r in template_results])
    ns = np.array([r['num_samples'] for r in template_results], dtype=float)

    total_n = float(np.sum(ns))
    if total_n <= 0:
        return empty

    acc_wmean, acc_wstd = weighted_mean_std(accs, ns)
    ece_wmean, ece_wstd = weighted_mean_std(eces, ns)

    return {
        'acc_weighted_mean': acc_wmean,
        'acc_weighted_std': acc_wstd,
        'acc_unweighted_mean': float(np.mean(accs)),
        'acc_inter_template_std': float(np.std(accs)),
        'ece_weighted_mean': ece_wmean,
        'ece_weighted_std': ece_wstd,
        'ece_inter_template_std': float(np.std(eces)),
        'total_samples': int(total_n),
        'per_template_sample_counts': ns.astype(int).tolist(),
        'per_template_accs': accs.tolist(),
        'per_template_eces': eces.tolist(),
    }

# =============================================================================
# 5. MULTI-TEMPLATE EVALUATION
# =============================================================================

def evaluate_llm_all_prompts(
    model: nn.Module,
    tokenizer: Any,
    test_samples: List[Dict[str, str]],
    label_vocab: List[str],
    device: torch.device = torch.device('cuda'),
    templates: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate across multiple prompt templates with per-template isolated ECE,
    then aggregate with sample-count weighting.

    Returns:
        {
            'template_results': List[Dict],
            'aggregate': Dict,   # see aggregate_template_results
            'token_len_audit': Dict[str, int],
        }
    """
    if templates is None:
        templates = PROMPT_TEMPLATES

    # Subword length audit
    token_len_audit = {
        c: len(tokenizer.encode(c, add_special_tokens=False)) for c in label_vocab
    }
    print(f"[TOKEN-AUDIT] Candidate subword lengths: {token_len_audit}")

    template_results: List[Dict[str, Any]] = []

    for t_idx, template in enumerate(templates):
        if verbose:
            print(f"\n[PROMPT] Evaluating template {t_idx + 1}/{len(templates)}: "
                  f"{template[:40]}...")
        result = evaluate_single_template(
            model, tokenizer, test_samples, label_vocab, template, device, verbose=verbose
        )
        result['template'] = template
        template_results.append(result)

    aggregate = aggregate_template_results(template_results)

    return {
        'template_results': template_results,
        'aggregate': aggregate,
        'token_len_audit': token_len_audit,
    }


# =============================================================================
# 6. FEW-SHOT EXAMPLE SELECTION
# =============================================================================

def select_few_shot_demos(
    train_samples: List[Dict[str, str]],
    label_vocab: List[str],
    k: int,
    seed: int,
) -> List[Dict[str, str]]:
    """
    Select k balanced few-shot demonstrations with fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    by_label: Dict[str, List[Dict[str, str]]] = {c: [] for c in label_vocab}
    for s in train_samples:
        if s['correct_clf'] in by_label:
            by_label[s['correct_clf']].append(s)

    demos: List[Dict[str, str]] = []
    per_label = max(1, k // max(1, len(label_vocab)))
    for c in label_vocab:
        pool = by_label[c]
        if pool:
            demos.extend(random.sample(pool, min(per_label, len(pool))))

    if len(demos) > k:
        demos = random.sample(demos, k)
    return demos


def build_few_shot_prompt(
    sample: Dict[str, str],
    demos: List[Dict[str, str]],
    template: str,
) -> str:
    """
    Construct few-shot prompt with demonstrations.
    """
    prompt_parts = []
    for d in demos:
        demo_sent = d['sentence'].replace("[MASK]", d.get('answer', d.get('correct_clf', '')))
        answer = d.get('answer', d.get('correct_clf', ''))
        prompt_parts.append(f"例句：{demo_sent} → 答案：{answer}")

    prompt_parts.append(template.format(sent=sample['sentence'].replace("[MASK]", "____")))
    return "\n".join(prompt_parts)


# =============================================================================
# 7. MULTI-SEED EVALUATION
# =============================================================================

def evaluate_with_multiple_seeds(
    model: nn.Module,
    tokenizer: Any,
    train_samples: List[Dict[str, str]],
    test_samples: List[Dict[str, str]],
    label_vocab: List[str],
    device: torch.device,
    k_shot: int = 5,
    seeds: Optional[List[int]] = None,
    templates: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate with multiple few-shot example selections, reporting variance.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    seed_results = []
    for seed in seeds:
        demos = select_few_shot_demos(train_samples, label_vocab, k_shot, seed)
        augmented_tests = []
        for s in test_samples:
            s_copy = dict(s)
            s_copy['sentence'] = build_few_shot_prompt(s, demos, PROMPT_TEMPLATES[0])
            augmented_tests.append(s_copy)

        res = evaluate_llm_all_prompts(
            model, tokenizer, augmented_tests, label_vocab, device, templates, verbose=False
        )
        res['seed'] = seed
        seed_results.append(res)

    acc_means = [r['aggregate']['acc_weighted_mean'] for r in seed_results]
    ece_means = [r['aggregate']['ece_weighted_mean'] for r in seed_results]

    return {
        'seed_results': seed_results,
        'acc_weighted_mean_across_seeds': float(np.mean(acc_means)),
        'acc_std_across_seeds': float(np.std(acc_means)),
        'ece_weighted_mean_across_seeds': float(np.mean(ece_means)),
        'ece_std_across_seeds': float(np.std(ece_means)),
    }


# =============================================================================
# 8. SAMPLE EFFICIENCY CURVE
# =============================================================================

def run_sample_efficiency(
    model: nn.Module,
    tokenizer: Any,
    train_samples: List[Dict[str, str]],
    test_samples: List[Dict[str, str]],
    label_vocab: List[str],
    device: torch.device,
    k_list: Optional[List[int]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model across varying few-shot sizes K.
    """
    if k_list is None:
        k_list = [1, 3, 5, 10, 20, 50, 100]

    results: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        demos = select_few_shot_demos(train_samples, label_vocab, k, seed)
        augmented_tests = []
        for s in test_samples:
            s_copy = dict(s)
            s_copy['sentence'] = build_few_shot_prompt(s, demos, PROMPT_TEMPLATES[0])
            augmented_tests.append(s_copy)

        res = evaluate_llm_all_prompts(
            model, tokenizer, augmented_tests, label_vocab, device, verbose=False
        )
        agg = res['aggregate']
        results[k] = {
            'acc_weighted_mean': agg['acc_weighted_mean'],
            'acc_weighted_std': agg['acc_weighted_std'],
            'ece_weighted_mean': agg['ece_weighted_mean'],
            'ece_inter_template_std': agg['ece_inter_template_std'],
        }
        if verbose:
            print(f"[K={k}] acc={agg['acc_weighted_mean']:.2f} "
                  f"(inter-template std {agg['acc_inter_template_std']:.2f}) | "
                  f"ECE={agg['ece_weighted_mean']:.4f}")

    return results


# =============================================================================
# 9. DATA LOADING
# =============================================================================

def load_test_samples(
    conllu_path: str,
    top_k: int = 10,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Load and filter test samples from CoNLL-U file.
    """
    from conllu import parse_incr

    all_samples: List[Dict[str, str]] = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if isinstance(t['id'], int)]
            for i, token in enumerate(tokens):
                if token['deprel'] in ['clf', 'mark:clf']:
                    clf = token['form']
                    num_token = next((t for t in tokens if t['id'] == token['head']), None)
                    if num_token is None:
                        continue
                    noun_token = next((t for t in tokens if t['id'] == num_token['head']), None)
                    if noun_token is None or noun_token['upos'] not in ['NOUN', 'PROPN']:
                        continue

                    words = [t['form'] for t in tokens]
                    words[i] = "[MASK]"
                    all_samples.append({
                        "sentence": "".join(words),
                        "noun": noun_token['form'],
                        "correct_clf": clf,
                    })

    counter = Counter([s['correct_clf'] for s in all_samples])
    valid_labels = [q for q, _ in counter.most_common(top_k)]
    filtered = [s for s in all_samples if s['correct_clf'] in valid_labels]

    return filtered, valid_labels


# =============================================================================
# 10. MAIN BENCHMARK PIPELINE
# =============================================================================

MODELS_TO_COMPARE: Dict[str, str] = {
    "Qwen2.5-1.5B-Instruct": "/mnt/proj/V3/models/Qwen2.5-1.5B-Instruct/",
    "Qwen2.5-0.5B-Instruct": "/mnt/proj/V3/models/Qwen2.5-0.5B-Instruct/",
    "ChatGLM3-6B": "/mnt/proj/V3/models/chatglm3-6b/",
    "InternLM2.5-7B-Chat": "/mnt/proj/V3/models/internlm2_5-7b-chat/",
    "GLM-4-9B-Chat": "/mnt/proj/V3/models/glm-4-9b-chat/",
}


def load_llm_model(hf_path: str, device: torch.device):
    """
    Load model + tokenizer with fallback.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 2048)

    try:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        quant_config = None

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
        )
    except Exception as e:
        print(f"[WARNING] Native loading failed: {e}. Trying .bin fallback...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                config=config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quant_config,
                use_safetensors=False,
            )
        except Exception as e2:
            print(f"[ERROR] Fatal loading error: {e2}")
            return None, None

    model.eval()
    return model, tokenizer


def run_llm_benchmark(
    conllu_path: str = "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu",
    top_k: int = 10,
    k_shot: int = 5,
    num_splits: int = 10,
    run_samp_eff: bool = True,
    verbose: bool = True,
) -> None:
    """
    Main LLM benchmark entry point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    all_samples, label_vocab = load_test_samples(conllu_path, top_k=top_k)
    print(f"[INFO] Loaded {len(all_samples)} samples | label vocab: {label_vocab}")

    if not all_samples:
        print("[ERROR] No samples loaded.")
        return

    class_samples: Dict[str, List[Dict[str, str]]] = {}
    for s in all_samples:
        class_samples.setdefault(s['correct_clf'], []).append(s)

    all_results: Dict[str, List[Dict[str, Any]]] = {
        model_name: [] for model_name in MODELS_TO_COMPARE
    }

    for split_idx in range(num_splits):
        print(f"\n==================== Split {split_idx + 1}/{num_splits} ====================")
        random.seed(42 + split_idx)
        np.random.seed(42 + split_idx)
        torch.manual_seed(42 + split_idx)

        train_samples, test_samples = [], []
        for cls, lst in class_samples.items():
            random.shuffle(lst)
            n_train = min(len(lst) // 2, max(k_shot * 2, 20))
            train_samples.extend(lst[:n_train])
            test_samples.extend(lst[n_train:])

        if not test_samples:
            print(f"[WARNING] Empty test split, skipping.")
            continue

        for model_name, hf_path in MODELS_TO_COMPARE.items():
            print(f"\n[INFO] Evaluating {model_name}")
            if not os.path.exists(hf_path):
                print(f"[WARNING] Path not found: {hf_path}, skipping.")
                continue

            model, tokenizer = load_llm_model(hf_path, device)
            if model is None:
                continue

            try:
                seed_res = evaluate_with_multiple_seeds(
                    model, tokenizer, train_samples, test_samples, label_vocab, device,
                    k_shot=k_shot, verbose=False,
                )

                agg = seed_res
                print(f"[{model_name}] "
                      f"acc={agg['acc_weighted_mean_across_seeds']:.2f} ± "
                      f"{agg['acc_std_across_seeds']:.2f} (across seeds) | "
                      f"ECE={agg['ece_weighted_mean_across_seeds']:.4f} ± "
                      f"{agg['ece_std_across_seeds']:.4f}")

                all_results[model_name].append(seed_res)

                if run_samp_eff and split_idx == 0:
                    print(f"\n[SAMPLE-EFFICIENCY] {model_name}")
                    run_sample_efficiency(
                        model, tokenizer, train_samples, test_samples, label_vocab,
                        device, verbose=True,
                    )

            except Exception as e:
                print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            finally:
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 110)
    print("FINAL LLM BENCHMARK SUMMARY")
    print("=" * 110)
    print(f"| {'Model':<28} | {'Acc (weighted)':<20} | {'ECE (weighted)':<20} |")
    print("-" * 110)
    for model_name, results in all_results.items():
        if not results:
            continue
        accs = [r['acc_weighted_mean_across_seeds'] for r in results]
        eces = [r['ece_weighted_mean_across_seeds'] for r in results]
        print(f"| {model_name:<28} | "
              f"{np.mean(accs):>5.2f} ± {np.std(accs):<10.2f} | "
              f"{np.mean(eces):>6.4f} ± {np.std(eces):<10.4f} |")
    print("=" * 110)


# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":
    run_llm_benchmark(
        conllu_path="UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu",
        top_k=10,
        k_shot=5,
        num_splits=10,
        run_samp_eff=True,
        verbose=True,
    )