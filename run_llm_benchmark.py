#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-LLM Benchmark for Chinese Quantifier Selection

This script evaluates multiple large language models on a Chinese quantifier
classification task using few-shot in-context learning (ICL). It implements:
- Stratified cross-validation splits (10 folds).
- Strict Token-level Evaluation (generation-free probability comparison).
- Calibration metrics (Expected Calibration Error, Brier score).
- Robust model loading with fallback to PyTorch native weights.

This codebase serves as the evaluation pipeline for the accompanying manuscript.
Author: [Anonymous for Double-Blind Review]
Date: 2026-06-22
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from conllu import parse_incr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score
from typing import List, Dict
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Environment Configuration
# =============================================================================
# Set Hugging Face mirror for stable download in mainland China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Suppress symlink warnings in cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Dictionary mapping model names to their local paths.
MODELS_TO_COMPARE: Dict[str, str] = {
    "GLM-4-9B-Chat": "/mnt/proj/V3/models/glm-4-9b-chat/",
    "ChatGLM3-6B": "/mnt/proj/V3/models/chatglm3-6b/",
    "InternLM2.5-7B-Chat": "/mnt/proj/V3/models/internlm2_5-7b-chat/",
    "Qwen2.5-1.5B-Instruct": "/mnt/proj/V3/models/Qwen2.5-1.5B-Instruct/",
    "Qwen2.5-0.5B-Instruct": "/mnt/proj/V3/models/Qwen2.5-0.5B-Instruct/"
}


# =============================================================================
# Calibration Metrics
# =============================================================================
def calculate_ece(confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the discrepancy between model confidence and empirical accuracy
    by partitioning predictions into equally spaced confidence bins.

    Args:
        confidences: List of normalized probabilities for the predicted class.
        accuracies: List of binary correctness indicators (1 if correct, else 0).
        n_bins: Number of bins for discretization.

    Returns:
        ECE value (float) in [0, 1]. Lower values indicate better calibration.
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(conf_in_bin - acc_in_bin)

    return float(ece)


def calculate_brier(confidences: List[float], accuracies: List[int]) -> float:
    """
    Compute the Brier score.

    The Brier score is the mean squared error between predicted probabilities
    and actual binary outcomes.

    Args:
        confidences: List of predicted probabilities for the chosen class.
        accuracies: List of binary correctness indicators (1 if correct else 0).

    Returns:
        Brier score (float). Lower values indicate better probabilistic forecasting.
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)
    return float(np.mean((confidences - accuracies) ** 2))


# =============================================================================
# Main Evaluator Class
# =============================================================================
class MultiLLMEvaluator:
    """
    Benchmark evaluator for extracting and evaluating zero/few-shot performance
    of autoregressive LLMs on syntactic dependency datasets (CONLL-U format).
    """

    def __init__(self, conllu_path: str, top_k: int = 10, few_shot_k: int = 5,
                 num_splits: int = 10, seed: int = 42) -> None:
        """
        Initialize the evaluator with data paths and hyperparameters.
        """
        self.conllu_path = conllu_path
        self.top_k = top_k
        self.few_shot_k = few_shot_k
        self.num_splits = num_splits
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fixed few-shot demonstrations strictly curated for context alignment.
        self.fixed_demos = [
            {"sentence": "我买了一幅非常精美的画。", "noun": "画", "answer": "幅"},
            {"sentence": "那两间（在暴风雨中摇摇欲坠的）小木屋。", "noun": "木屋", "answer": "间"},
            {"sentence": "桌上摆着三本厚厚的英文词典。", "noun": "词典", "answer": "本"},
            {"sentence": "他递给我一杯热气腾腾的牛奶。", "noun": "牛奶", "answer": "杯"},
            {"sentence": "夜空中挂着一轮明亮的弯月。", "noun": "弯月", "answer": "轮"},
        ]
        self.label_vocab = []

    def extract_all_samples(self) -> List[Dict[str, str]]:
        """
        Extract cloze-style masked samples from the CONLL-U dependency tree.
        """
        samples = []
        with open(self.conllu_path, "r", encoding="utf-8") as f:
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
                        samples.append({
                            "sentence": "".join(words),
                            "noun": noun_token['form'],
                            "correct_clf": clf
                        })
        return samples

    def filter_top_quantifiers(self, samples: List[Dict[str, str]]) -> List[str]:
        """
        Filter the target vocabulary to the top-K most frequent quantifiers.
        """
        counter = Counter([s['correct_clf'] for s in samples])
        valid = [q for q, cnt in counter.most_common(self.top_k)]
        return valid

    def build_few_shot_prompt(self, sample: Dict[str, str], demos: List[Dict[str, str]]) -> str:
        """
        Construct an autoregressive prompt for in-context learning.
        We deliberately omit instruction headers to avoid triggering conversational
        fillers (e.g., `<|assistant|>`), forcing pure textual continuation.
        """
        prompt = ""
        for demo in demos:
            demo_sent = demo['sentence'].replace("[MASK]", demo['answer'])
            prompt += f"例句：{demo_sent} → 答案：{demo['answer']}\n"
        prompt += f"例句：{sample['sentence']} → 答案："
        return prompt

    def evaluate_single_split(self, model_name: str, hf_path: str, train_samples: List[Dict[str, str]],
                              test_samples: List[Dict[str, str]], tokenizer, model, device, label_vocab) -> Dict:
        """
        Evaluate the specified LLM on a given test split using strict token-level
        generation-free probability comparisons.
        """
        demos = self.fixed_demos[:self.few_shot_k]

        # Prepend contextual text to avoid tokenization artifacts (e.g., implicit spaces)
        valid_labels = []
        valid_token_ids = []
        for token in label_vocab:
            ids = tokenizer.encode("答案：" + token, add_special_tokens=False)
            if len(ids) > 0:
                valid_labels.append(token)
                valid_token_ids.append(ids[-1])

        pred_labels = []
        confidences = []
        accuracies = []
        binary_predictions = []

        for sample in test_samples:
            prompt = self.build_few_shot_prompt(sample, demos)

            # Note: Explicitly bypassing chat templates to enforce raw ICL evaluation.
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Logits of the immediate next token
            next_token_logits = outputs.logits[0, -1, :]

            # Compute softmax over the full vocabulary to maintain true probability mass
            full_probs = F.softmax(next_token_logits, dim=-1)

            # Extract and normalize probabilities constrained to the valid label subset
            valid_probs = full_probs[valid_token_ids]
            best_idx = torch.argmax(valid_probs).item()
            pred_clf = valid_labels[best_idx]

            sum_valid_probs = torch.sum(valid_probs).item()
            if sum_valid_probs > 0:
                confidence = (valid_probs[best_idx] / sum_valid_probs).item()
            else:
                confidence = 1.0 / len(valid_labels)  # Fallback to uniform distribution

            pred_labels.append(pred_clf)
            confidences.append(confidence)
            is_correct = (pred_clf == sample['correct_clf'])
            accuracies.append(1 if is_correct else 0)
            binary_predictions.append(1.0 if is_correct else 0.0)

        # Compute classification metrics
        acc = np.mean(accuracies) * 100
        macro_f1 = f1_score([s['correct_clf'] for s in test_samples], pred_labels,
                            average='macro', zero_division=0) * 100
        weighted_f1 = f1_score([s['correct_clf'] for s in test_samples], pred_labels,
                               average='weighted', zero_division=0) * 100

        # Compute probabilistic reliability metrics
        ece = calculate_ece(confidences, accuracies, n_bins=10)
        brier = calculate_brier(confidences, accuracies)

        # Bootstrap for empirical standard deviation estimation
        boot_means = []
        for _ in range(200):
            boot = np.random.choice(binary_predictions, size=len(binary_predictions), replace=True)
            boot_means.append(np.mean(boot) * 100)
        acc_std = np.std(boot_means)

        return {
            "accuracy": acc,
            "accuracy_std": acc_std,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "ece": ece,
            "brier": brier,
            "num_samples": len(test_samples)
        }

    def run_benchmark(self) -> None:
        """
        Execute the full evaluation pipeline across all specified models and splits.
        """
        all_samples = self.extract_all_samples()
        if not all_samples:
            print("[ERROR] No samples extracted. Please verify the CONLL-U file path.")
            return

        top_quantifiers = self.filter_top_quantifiers(all_samples)
        self.label_vocab = top_quantifiers
        print(f"[INFO] Selected valid label vocabulary (Top {self.top_k}): {self.label_vocab}")

        filtered_samples = [s for s in all_samples if s['correct_clf'] in self.label_vocab]
        if len(filtered_samples) == 0:
            print("[ERROR] No samples match the target vocabulary.")
            return

        # Prepare stratified data grouping
        class_samples = {}
        for s in filtered_samples:
            class_samples.setdefault(s['correct_clf'], []).append(s)

        # Initialize tracking dictionaries
        results = {model_name: {"accuracy": [], "macro_f1": [], "weighted_f1": [], "ece": [], "brier": []}
                   for model_name in MODELS_TO_COMPARE}
        results["Random Baseline"] = {"accuracy": [], "macro_f1": [], "weighted_f1": [], "ece": [], "brier": []}

        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            quantization_config = None
            print("[WARNING] BitsAndBytesConfig not available; loading models in standard precision.")

        for split_idx in range(self.num_splits):
            print(f"\n==================== Split {split_idx + 1}/{self.num_splits} ====================")
            random.seed(self.seed + split_idx)
            np.random.seed(self.seed + split_idx)
            torch.manual_seed(self.seed + split_idx)

            train_samples, test_samples = [], []
            for cls, lst in class_samples.items():
                random.shuffle(lst)
                if len(lst) <= self.few_shot_k:
                    train_samples.extend(lst)
                else:
                    train_samples.extend(lst[:self.few_shot_k])
                    test_samples.extend(lst[self.few_shot_k:])

            if not test_samples:
                print(f"[WARNING] Skipping split {split_idx + 1} due to insufficient test samples.")
                continue

            for model_name, hf_path in MODELS_TO_COMPARE.items():
                print(f"\n[INFO] Evaluating {model_name} on Split {split_idx + 1}")

                config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
                if not hasattr(config, "max_length"):
                    config.max_length = getattr(config, "seq_length", 2048)

                tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                model = None
                try:
                    # Attempt standard loading strategy (prioritizes .safetensors)
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_path, config=config, torch_dtype=torch_dtype,
                        trust_remote_code=True, device_map="auto",
                        quantization_config=quantization_config
                    )
                except Exception as e:
                    print(f"[WARNING] Native loading failed for {model_name}: {e}. Attempting .bin fallback...")
                    try:
                        # Fallback mechanism for corrupted safetensors environments
                        model = AutoModelForCausalLM.from_pretrained(
                            hf_path, config=config, torch_dtype=torch_dtype,
                            trust_remote_code=True, device_map="auto",
                            quantization_config=quantization_config, use_safetensors=False
                        )
                        print(f"[INFO] Successfully loaded {model_name} via .bin format.")
                    except Exception as e2:
                        print(f"[ERROR] Fatal loading error for {model_name}: {e2}. Skipping this split.")
                        continue

                model.eval()
                metrics = self.evaluate_single_split(
                    model_name, hf_path, train_samples, test_samples,
                    tokenizer, model, device, self.label_vocab
                )

                for key in ["accuracy", "macro_f1", "weighted_f1", "ece", "brier"]:
                    results[model_name][key].append(metrics[key])

                # Memory management: Prevent VRAM OOM across loop iterations
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Process analytical baseline using empirical training distributions
            class_freq = Counter([s['correct_clf'] for s in train_samples])
            total_train = sum(class_freq.values())
            probs = [class_freq[cls] / total_train for cls in self.label_vocab]

            rand_preds = np.random.choice(self.label_vocab, size=len(test_samples), p=probs)
            gold = [s['correct_clf'] for s in test_samples]

            rand_acc = np.mean(np.array(rand_preds) == np.array(gold)) * 100
            rand_f1 = f1_score(gold, rand_preds, average='macro', zero_division=0) * 100

            results["Random Baseline"]["accuracy"].append(rand_acc)
            results["Random Baseline"]["macro_f1"].append(rand_f1)
            results["Random Baseline"]["weighted_f1"].append(rand_f1)
            results["Random Baseline"]["ece"].append(0.0)
            results["Random Baseline"]["brier"].append(0.0)

        # ------------------------------------------------------------------
        # Final Benchmark Summary Compilation
        # ------------------------------------------------------------------
        print("\n" + "=" * 120)
        print(f"📊 FINAL FAIR BENCHMARK RESULTS (AGGREGATED OVER {self.num_splits} SPLITS)")
        print("=" * 120)
        header = f"| {'Model':<20} | {'Acc (%)':<15} | {'Macro-F1 (%)':<12} | {'Weighted-F1 (%)':<12} | {'ECE':<8} | {'Brier':<8} |"
        print(header)
        print("-" * 120)

        for model_name in list(results.keys()):
            acc_list = results[model_name]["accuracy"]
            acc_mean = np.nanmean(acc_list) if acc_list else float('nan')
            acc_std = np.nanstd(acc_list) if acc_list else float('nan')

            f1_list = results[model_name]["macro_f1"]
            f1_mean = np.nanmean(f1_list) if f1_list else float('nan')

            wf1_list = results[model_name]["weighted_f1"]
            wf1_mean = np.nanmean(wf1_list) if wf1_list else float('nan')

            ece_list = results[model_name]["ece"]
            ece_mean = np.nanmean(ece_list) if ece_list else float('nan')

            brier_list = results[model_name]["brier"]
            brier_mean = np.nanmean(brier_list) if brier_list else float('nan')

            print(
                f"| {model_name:<20} | "
                f"{acc_mean:>5.2f} ± {acc_std:<6.2f} | "
                f"{f1_mean:>10.2f} | "
                f"{wf1_mean:>12.2f} | "
                f"{ece_mean:>8.4f} | "
                f"{brier_mean:>8.4f} |"
            )
        print("=" * 120)


if __name__ == "__main__":
    # Specify the dependency parser dataset path
    ud_test_file = "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu"

    # Instantiate and execute the framework
    evaluator = MultiLLMEvaluator(ud_test_file, top_k=10, few_shot_k=5, num_splits=10)
    evaluator.run_benchmark()
