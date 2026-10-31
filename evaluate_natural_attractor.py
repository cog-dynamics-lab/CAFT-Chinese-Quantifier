"""
Natural Attractor Probe: R1-1 response.
筛选 linear_dist >= 3 且中间有干扰名词的句子，
对比 Baseline (MAML) vs CAFT (MAML) 在这些句子上的准确率。
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from run_rigorous_aligned_ablationTJ import (
    BulletproofUDProcessor,
    evaluate_model_bulletproof,
)
from meta_pretrain_full_strong import NeuralTransformerStudent
from train_sota_few_shot import load_prior_weights_soft, train_sota_few_shot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONLLU = "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu"
PRIOR = "meta_pretrained_prior.pth"


def extract_attractor_tasks(all_tasks):
    """筛选天然 attractor 样本：linear_dist >= 3 且中间有干扰名词。"""
    attractor = []
    for t in all_tasks:
        ld = t.get("linear_dist", -1)
        if ld < 3:
            continue
        m, n = t["mask_idx"], t["noun_idx"]
        if m < 0 or n < 0:
            continue
        lo, hi = (m + 1, n) if m < n else (n + 1, m)
        has_intervening_noun = False
        for pos in range(lo, hi):
            tid = t["seq_ids"][pos]
            if 100 <= tid < 2000:
                has_intervening_noun = True
                break
        if has_intervening_noun:
            attractor.append(t)
    return attractor


def main():
    print(f"[INFO] Device: {DEVICE}")
    processor = BulletproofUDProcessor()
    all_tasks = processor.extract_tasks(CONLLU)
    print(f"[INFO] Total tasks: {len(all_tasks)}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_t, val_t, test_t, num_classes = processor.split_three_way(
        all_tasks, top_k=6, k_shot=10, k_calib=10, max_train_linear_dist=1,
    )
    print(f"[INFO] Split 0: train={len(train_t)}, val={len(val_t)}, test={len(test_t)}")

    attractor = extract_attractor_tasks(test_t)
    print(f"[INFO] Natural Attractor tasks (linear_dist>=3 + intervening noun): {len(attractor)}")
    if len(attractor) == 0:
        print("[WARN] No attractor tasks found. Try relaxing threshold.")
        return

    X_train, y_train, _ = processor.encode(train_t)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

    fixed_T = 1.2

    has_prior = os.path.exists(PRIOR)
    if has_prior:
        print(f"[INFO] Prior found: {PRIOR}")
    else:
        print(f"[WARN] Prior NOT found: {PRIOR} -> will use random init (Scratch)")

    results = {}
    configs = [
        ("Baseline (MAML)" if has_prior else "Baseline (Scratch)", has_prior, "Baseline"),
        ("CAFT (MAML)" if has_prior else "CAFT (Scratch)", has_prior, "CAFT"),
    ]
    for config_name, use_prior, config_type in configs:
        print(f"\n[INFO] Training {config_name} ...")
        model = NeuralTransformerStudent(vocab_size=25000, num_classes=num_classes).to(DEVICE)
        if use_prior:
            load_prior_weights_soft(model, PRIOR, DEVICE, alpha=0.4)

        train_sota_few_shot(
            model=model, X=X_train, y=y_train,
            processor=processor, train_t=train_t,
            is_ours=(config_type == "CAFT"),
            has_prior=use_prior,
            device=DEVICE, config_type=config_type,
            phase1_epochs=20, phase2_epochs=60,
            verbose=False,
        )
        model.eval()

        acc_full, _, _, _ = evaluate_model_bulletproof(
            model, processor, test_t, fixed_T, DEVICE
        )
        acc_attr, _, _, _ = evaluate_model_bulletproof(
            model, processor, attractor, fixed_T, DEVICE
        )
        results[config_name] = {
            "acc_full": acc_full,
            "acc_attractor": acc_attr,
            "n_attractor": len(attractor),
        }
        print(f"[RESULT] {config_name}: full={acc_full:.2f}, attractor={acc_attr:.2f} (n={len(attractor)})")

    print("\n" + "=" * 60)
    print("NATURAL ATTRACTOR PROBE SUMMARY (Split 0)")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name:25s} | full: {r['acc_full']:6.2f} | attractor: {r['acc_attractor']:6.2f}")
    names = list(results.keys())
    delta = results[names[1]]["acc_attractor"] - results[names[0]]["acc_attractor"]
    print(f"\n{names[1]} - {names[0]} on attractor subset: {delta:+.2f} pp")

    with open("results/natural_attractor_probe.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[DUMP] results/natural_attractor_probe.json")


if __name__ == "__main__":
    main()
