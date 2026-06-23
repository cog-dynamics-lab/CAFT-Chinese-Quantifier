"""
Module: ud_kshot_ablation.py
Description: Implements the syntactic distance parsing, target quantifier extraction,
             robust semantic noise injection, calibration metrics (ECE), and
             evaluation pipelines under varying complexity groups.
             Aligned with the AlignedUDProcessor vocabulary and masking protocol.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from conllu import parse_incr
from meta_pretrain_full_strong import NeuralTransformerStudent, pad_batch

# =============================================================================
# 1. CORE QUANTITATIVE METRICS (ECE & NOISE PERTURBATION)
# =============================================================================

def calculate_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) to measure prediction confidence alignment.
    """
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def inject_syntactic_noise(X: torch.Tensor, vocab_size: int, noise_rate: float = 0.15,
                           pad_idx: int = 0) -> torch.Tensor:
    """
    Randomly perturbs sequences to test robustness against surface lexical shortcuts.
    """
    X_noisy = X.clone()
    mask = (torch.rand(X.shape, device=X.device) < noise_rate) & (X != pad_idx)
    replace_mask = mask & (torch.rand(X.shape, device=X.device) < 0.8)
    zero_mask = mask & (~replace_mask)

    random_tokens = torch.randint(2, max(3, vocab_size - 1), X.shape, device=X.device)
    X_noisy[replace_mask] = random_tokens[replace_mask]
    X_noisy[zero_mask] = pad_idx
    return X_noisy


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via single-parameter logit temperature scaling.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def calibrate_model(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device) -> float:
    """
    Optimizes temperature scaling parameter on validation data using LBFGS.
    """
    model.eval()
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    with torch.no_grad():
        logits = model(X_val)

    def eval_loss():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(scaler(logits), y_val)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return max(scaler.temperature.item(), 0.1)


# =============================================================================
# 2. UNIVERSAL DEPENDENCIES PROCESSOR WITH COMPLEXITY MAPPING
# =============================================================================

class UDCorpusProcessor:
    """
    Loads CoNLL-U format dependency trees, processes tokens, and maps syntax
    complexity (Simple vs Complex) based on head-dependent structural distance.
    Aligned 100% with AlignedUDProcessor to ensure vocabs and structures match.
    """
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "的": 2, "之": 3, "<MASK>": 4}
        self.n_id_counter = 100
        self.other_id_counter = 2000
        self.quantifier_to_id = {}

    def get_aligned_id(self, form: str, upos: str) -> int:
        if form in self.vocab:
            return self.vocab[form]
        if upos in ['NOUN', 'PROPN', 'PRON']:
            self.vocab[form] = self.n_id_counter
            self.n_id_counter = 100 + ((self.n_id_counter - 99) % 1899)
        else:
            self.vocab[form] = self.other_id_counter
            self.other_id_counter = 2000 + ((self.other_id_counter - 1999) % 22999)
        return self.vocab[form]

    def extract_quantifier_tasks(self, filepath: str, build_vocab: bool = True) -> list:
        """
        Parses sentence tasks matching the sequence preservation and masking logic.
        """
        tasks = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing UD dataset file at '{filepath}'")

        with open(filepath, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                for token in tokenlist:
                    if token['deprel'] == 'clf':
                        clf_word, clf_id, head_id = token['form'], token['id'], token['head']
                        if not head_id:
                            continue
                        head_token = tokenlist[head_id - 1]
                        if head_token['upos'] == 'NUM' and head_token['head'] != 0:
                            head_token = tokenlist[head_token['head'] - 1]
                        if head_token['upos'] not in ['NOUN', 'PROPN']:
                            continue
                        noun_id = head_token['id']
                        dist = abs(clf_id - noun_id)

                        seq_ids, mask_idx, noun_idx = [], -1, -1
                        for t in tokenlist:
                            if t['id'] == clf_id:
                                seq_ids.append(self.vocab["<MASK>"])
                                mask_idx = len(seq_ids) - 1
                            else:
                                if build_vocab:
                                    cid = self.get_aligned_id(t['form'], t['upos'])
                                else:
                                    cid = self.vocab.get(t['form'], self.vocab["<UNK>"])
                                seq_ids.append(cid)
                                if t['id'] == noun_id:
                                    noun_idx = len(seq_ids) - 1
                        tasks.append({
                            'quantifier': clf_word,
                            'seq_ids': seq_ids,
                            'mask_idx': mask_idx,
                            'noun_idx': noun_idx,
                            'complexity': "Simple" if dist == 1 else "Complex"
                        })
        return tasks

    def filter_top_quantifiers(self, tasks: list, top_k: int = 5) -> list:
        counts = Counter([t['quantifier'] for t in tasks])
        top_quants = [q for q, _ in counts.most_common(top_k)]
        self.quantifier_to_id = {q: i for i, q in enumerate(top_quants)}
        return [t for t in tasks if t['quantifier'] in self.quantifier_to_id]

    def encode_batch(self, tasks: list) -> tuple:
        X = [t['seq_ids'] for t in tasks]
        y = [self.quantifier_to_id.get(t['quantifier'], 0) for t in tasks]
        return pad_batch(X), torch.tensor(y, dtype=torch.long)


# =============================================================================
# 3. EVALUATION PROTOCOL (COMPLEXITY-WISE & METRICS SPLIT)
# =============================================================================

def evaluate_performance(model: nn.Module, processor: UDCorpusProcessor,
                         test_tasks: list, device: torch.device) -> tuple:
    """
    Evaluates accuracy and ECE metrics mapped across structural complexity partitions.
    """
    model.eval()
    X_test, y_test = processor.encode_batch(test_tasks)
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)

    overall_acc = preds.eq(y_test).float().mean().item() * 100
    overall_ece = calculate_ece(logits, y_test)

    stats = {
        "Simple": {"corr": 0, "tot": 0},
        "Complex": {"corr": 0, "tot": 0}
    }

    for idx, task in enumerate(test_tasks):
        comp = task.get("complexity", "Complex")
        group = "Simple" if comp == "Simple" else "Complex"
        stats[group]["tot"] += 1
        if preds[idx] == y_test[idx]:
            stats[group]["corr"] += 1

    return overall_acc, overall_ece, stats
