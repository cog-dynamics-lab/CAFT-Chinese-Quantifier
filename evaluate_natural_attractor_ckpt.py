"""
Natural Attractor Probe (checkpoint-based, no training)
"""
import os, json, random, numpy as np, torch
from run_rigorous_aligned_ablationTJ import (
    BulletproofUDProcessor, evaluate_model_bulletproof,
)
from meta_pretrain_full_strong import NeuralTransformerStudent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONLLU = "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu"

def extract_attractor_tasks(all_tasks):
    attractor = []
    for t in all_tasks:
        ld = t.get("linear_dist", -1)
        if ld < 3: continue
        m, n = t["mask_idx"], t["noun_idx"]
        if m < 0 or n < 0: continue
        lo, hi = (m + 1, n) if m < n else (n + 1, m)
        has = any(100 <= t["seq_ids"][p] < 2000 for p in range(lo, hi))
        if has: attractor.append(t)
    return attractor

def main():
    processor = BulletproofUDProcessor()
    all_tasks = processor.extract_tasks(CONLLU)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train_t, val_t, test_t, num_classes = processor.split_three_way(
        all_tasks, top_k=6, k_shot=10, k_calib=10, max_train_linear_dist=1)
    attractor = extract_attractor_tasks(test_t)
    print(f"[INFO] Natural Attractor tasks: {len(attractor)}")
    fixed_T = 1.2
    results = {}
    for name, ckpt in [("Baseline (MAML)", "results/baseline_maml.pth"),
                       ("CAFT (MAML)", "results/caft_maml.pth")]:
        if not os.path.exists(ckpt):
            print(f"[SKIP] {ckpt} not found")
            continue
        model = NeuralTransformerStudent(vocab_size=25000, num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        acc_full, _, _, _ = evaluate_model_bulletproof(model, processor, test_t, fixed_T, DEVICE)
        acc_attr, _, _, _ = evaluate_model_bulletproof(model, processor, attractor, fixed_T, DEVICE)
        results[name] = {"acc_full": acc_full, "acc_attractor": acc_attr, "n": len(attractor)}
        print(f"[RESULT] {name}: full={acc_full:.2f}, attractor={acc_attr:.2f}")
    names = list(results.keys())
    if len(names) == 2:
        delta = results[names[1]]["acc_attractor"] - results[names[0]]["acc_attractor"]
        print(f"\n{names[1]} - {names[0]} on attractor: {delta:+.2f} pp")
    with open("results/natural_attractor_probe_ckpt.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[DUMP] results/natural_attractor_probe_ckpt.json")

if __name__ == "__main__":
    main()
