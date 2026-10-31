# Biologically Motivated Inductive Biases for Few-Shot Cognitive Generalization in Embodied AI 🚀

> **Anonymous Repository for Double-Blind Review**
>
> **Status:** 5/5 LLM models completed (Split 1, K=10); GLM-4-9B K=100 in progress.
>
> Official implementation, evaluation scripts, and replication guidelines for the above paper.

This repository contains the complete experimental pipeline, including:

- **Synthetic PCFG meta-learning** via first-order Reptile optimization;
- **CAFT** (**C**ontrastive **A**daptive **F**ine-**T**uning), the unified adaptation framework;
- **SIPAR** (**S**yntax-**I**nvariant **P**airwise **A**lignment **R**egularizer), the contrastive regularization term;
- **Natural Attractor Probe** (Split-0 diagnostic subset, N = 413) and **No-Noun masking**;
- **Two counterfactual scrambling variants** (`modifier` / `core_swap`) implemented in `run_rigorous_aligned_ablationTJ.py`;
- **Structural probing** via Hewitt & Manning (2019) rank-constrained bilinear probe;
- **Rigorous OOD evaluation** on the Universal Dependencies (UD) Chinese treebank (`UD_Chinese-GSDSimp`);
- **Leakage-free calibration** (temperature scaling fitted exclusively on validation sets).

> **Note on terminology**: The manuscript and Appendix C strictly use **CAFT** = *Contrastive Adaptive Fine-Tuning* and **SIPAR** = *Syntax-Invariant Pairwise Alignment Regularizer*. Earlier working drafts or low-level code comments may occasionally reference alternative operational terms (e.g., *"Causal Attention with Formal-grammar Topology"*, **present only in legacy code comments, not in the manuscript**); these are **deprecated implementation nomenclature** and are superseded by the paper's theoretical definitions.

---

## 📂 Repository Structure

### Phase 1 & 2 — Meta-Prior Distillation
- `meta_pretrain_full_strong.py` — Reptile meta-training on synthetic PCFG tasks. Produces `meta_pretrained_prior.pth`.
- `meta_pretrain_no_noise.py` — Structure-free ablation meta-training. Produces `meta_no_noise.pth` (lexical control group).

### Phase 3 — Few-Shot Adaptation
- `run_rigorous_aligned_ablationTJ.py` — **[Core script]** End-to-end 10-split ablation. Implements `BulletproofUDProcessor`, three-way split, leakage-free calibration, Monte Carlo causal head ablation, and both `modifier` and `core_swap` syntactic perturbation variants (per-split `fallback_ratio` audited via `reset_all_audit()`).
- `train_sota_few_shot.py` — Few-shot training pipeline (Soft Prior Interpolation + LLRD + SIPAR) for the 8 ablation configurations.

### Phase 4 — Behavioral Probes

#### Natural Attractor Probe (Main Text, Table 3)
- `evaluate_natural_attractor_ckpt.py` — Extracts the Split-0 attractor subset (N=413) using positional criteria (`linear_dist ≥ 3` + `has_intervening_noun`), evaluates CAFT vs. Baseline, writes `results/natural_attractor_probe_ckpt.json` (**Table 3 source**).
- `evaluate_natural_attractor.py` — Standalone version of the same probe.

#### Double-Blind Counterfactual Benchmark (Complementary, Repository Only)
- `build_double_blind_attractor.py` — Constructs three-condition (A/B/C) benchmark: Forward Binding / Adversarial Inversion / Neutral Control.
- `eval_double_blind_attractor.py` — Evaluates CAFT, baselines, and statistical controls on the double-blind benchmark.
- `visualize_double_blind_results.py` — Generates figures.

> **Note**: The main-text Table 3 reports the **Natural Attractor Probe (Split 0, N=413)**, not the three-condition double-blind benchmark. The latter is a complementary repository analysis.

### Phase 5 — Structural Probing & LLM Benchmark
- `representation_analyzer.py` — Effective Rank (SVD singular spectrum entropy), Cosine Anisotropy, and Hewitt & Manning (2019) probe. `compute_mst_and_hm_uuas` returns `{mst_uuas, hm_undirected_uuas, hm_directed_uuas, discarded_ratio}`. **Table 1's MST-UUAS column reports `mst_uuas`**, not `hm_undirected_uuas`.
- `ud_kshot_ablation.py` — Diagnostic & calibration suite: `fit_temperature_on_val` (leakage-free), `compute_stratified_bootstrap_ece` (class-stratified bootstrap ECE with 95% CI), `compute_dependency_hops_with_telemetry` (BFS dependency hop), `inject_syntactic_noise` (non-PAD controlled noise).
- `eval_llm_rigorous.py` — Generation-free token-level probability evaluation for causal LLMs: multi-token length-normalized prefix log-likelihood, per-template isolated ECE with class-stratified bootstrap, cross-template weighted aggregation, multi-seed few-shot selection, and **K ∈ {1,3,5,10,20,50,100} sample-efficiency support** (Qwen2.5-1.5B's full K=1..100 sweep is reported in Table A1; the remaining models' K-sweeps are archived in `results/llm_split1_ksweep_partial.json`).

### Phase 6 — Table Export
- `export_latex_tables.py` — Generates publication-ready booktabs tables (`--tables 1,2,3`).

### Utility / Orchestration (Repository Tooling, Not in Main Text)
- `run_all_experiments.py` — Convenience orchestrator (S0 env check → S1 encode verify → S2 meta-pretrain → S3 ablation → S4 double-blind → S5 figures → S6 LaTeX export). **Not referenced in the paper; provided for end-to-end reproduction.**
- `run_llm_benchmark.py` — Batch wrapper around `eval_llm_rigorous.py` (10-fold stratified CV, generation-free token-level comparison, ECE/Brier, robust weight loading). **Not referenced in the paper.**
- `verify_encode.py` — Unit regression for `NeuralPredictor._encode_sentence` (Max-Length Forward Matching; 12 hand-crafted cases). **Not referenced in the paper.**

---

## 🛠️ Environment Setup

**Hardware:** All experiments, including local inference of up to 9B-parameter LLMs, were conducted on a single **NVIDIA A40 GPU (48GB VRAM)**.

**Python Version:** 3.8+

```bash
pip install torch numpy scikit-learn conllu networkx transformers gensim matplotlib
```

Dataset: Download UD_Chinese-GSDSimp and place under ./UD_Chinese-GSDSimp-master/:

- zh_gsdsimp-ud-train.conllu
- zh_gsdsimp-ud-test.conllu


## ⚙️ Quick Start

One-shot full pipeline

```bash
python run_all_experiments.py
```

Runs S0–S6 (env check → encode verify → meta-pretrain → ablation → double-blind → figures → LaTeX export).

**Or step by step**

```bash
# Phase 1 & 2
python meta_pretrain_full_strong.py        # → meta_pretrained_prior.pth
python meta_pretrain_no_noise.py           # → meta_no_noise.pth (control)

# Phase 3 (main ablation)
python run_rigorous_aligned_ablationTJ.py  # → results/*_10split.json

# Phase 4 (Attractor probe, Table 3)
python evaluate_natural_attractor_ckpt.py  # → results/natural_attractor_probe_ckpt.json

# Phase 6
python export_latex_tables.py --results_dir ./results --output_dir ./tables --tables 1,2,3
```

**LLM benchmark**

```bash
pip install --upgrade huggingface_hub tiktoken sentencepiece
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
python run_llm_benchmark.py    # or: python eval_llm_rigorous.py
```

## 📊 Experimental Protocol & Artifact Data Flow 


[UD Chinese-GSDSimp]
        │
        ▼
run_rigorous_aligned_ablationTJ.py  (10 aligned splits, K=10)
        ├──► results/ablation_summary_10split.json ──────────► Table 1
        │    (source log: logs/clean_run_final_20261018_063218.log)
        │
        └──► results/head_knockout_results_10split.json ─────► Table 2
             (per-split values in Appendix C)
        │
        ▼
evaluate_natural_attractor_ckpt.py  (Split 0, N=413)
        └──► results/natural_attractor_probe_ckpt.json ──────► Table 3

> **Data Integrity Notice**
>
> The legacy files results/ablation_summary.json and results/head_knockout_results.json are single-split smoke-test artifacts (from logs/ckpt_save_run.log). They have been renamed to results/*_splits1_DEPRECATED.json and must not be used to reproduce Tables 1–2. The exact numerical sources are:
> 
> - logs/clean_run_final_20261018_063218.log — the 10-split run log. Table 1 aggregates parsed from its AGGREGATED RESULTS block; Table 2 mediation/knockout parsed from per-split [KNOCKOUT] / [MEDIATION] lines.
> - results/ablation_summary_10split.json — Table 1 archive.
> - results/head_knockout_results_10split.json — Table 2 archive.
> - results/natural_attractor_probe_ckpt.json — Table 3 archive (Split 0, N=413).
> - logs/eval_llm_rigorous_20261019_071711.log — the 5-model Split-1 LLM log; Table A2 source (Qwen2.5-1.5B, Qwen2.5-0.5B, ChatGLM3-6B, InternLM2.5-7B-Chat, GLM-4-9B-Chat).
> - Per-split values are listed in Appendix C.

## 🧪 Reproducibility Checklist

- [x] **No Test Leakage** : Three-way split (train / val / test) strictly isolated; temperature calibration T* fitted on validation sets only.
- [x] **Multi-Split Stability** : Table 1 aggregated across 10 random splits (grand-mean ± grand-std).
- [x] **Multi-Seed LLM Stability** : Appendix A averaged over 5 seeds × 3 prompt templates (Split 1, K=10; prompt-sensitivity std reported).
- [x] **Statistical Rigor** : Knockout significance via 1,000-iteration Monte Carlo permutation against data-driven null; causal-mediation intervals via 95% bootstrap CI.
- [x] **Strict Disentanglement** : UUAS via cross-validated bilinear probe; UUAS parity is reported candidly (14.73 vs. 14.71, Δ=+0.02 not meaningful), and the paper's structural claim rests on behavioral + causal evidence (Attractor Probe; IE=0.1961, 95% CI [0.129, 0.262]), not on geometric metrics alone.
- [x] **Perturbation Audit** : Both modifier (Distractor Scrambling) and core_swap (Core-Argument Scrambling) variants are implemented in run_rigorous_aligned_ablationTJ.py with per-split fallback_ratio logging. Main-text behavioral evidence uses the Natural Attractor Probe (Table 3) and No-Noun masking (Fig. 3b); the modifier/core_swap quantitative comparison is reported as future work in Appendix B.5.
- [x] **Subword Tokenization Audit** : Candidate classifier subword lengths logged to detect tokenizer-induced bias in LLM scoring.
- [x] **Causal Null Distributions**: Monte Carlo Random-K head ablation for null-hypothesis testing with conservative p-value lower bound.
- [x] **Length-Normalized Likelihoods** : LLM log-probabilities explicitly normalized by candidate-token length.
- [x] **Prompt-Isolated Calibration** : ECE calculated per prompt template before weighted aggregation.

## ⚠️ Known Scope & Scientific Limitations
- **Single Language Scope** : Primary experiments use UD_Chinese-GSDSimp. Cross-linguistic applicability (e.g., English subject–verb agreement with attractors) is discussed as future work in §Limitations; a full orthogonalized syntactic-complexity stratification is specified in Appendix B.

- **Treebank Sample Size** : ~1,640 classifier–noun instances extracted; after OOD filtering (d > 1) the diagnostic Natural Attractor subset further restricts to linear_dist ≥ 3 with intervening distractor nouns, yielding N=413 on Split 0. Tail-classifier breakdowns carry wider confidence intervals.

- **Pre-Trained vs. From-Scratch** : This study deliberately isolates from-scratch meta-inductive adaptation to assess sample-efficient structural inductive biases without confounding from massive web-scale pre-training. Extending CAFT to pre-trained backbones (e.g., RoBERTa, BERT) is stated in §Limitations as a distinct future scientific question.

- **Probe Decodability** : The rank-constrained bilinear probe (k=32) tests linear decodability of tree geometry; nonlinear representations may form an upper bound.

- **Prompt Sensitivity** : LLM few-shot results are reported over 3 fixed templates with prompt-sensitivity std; exhaustive prompt engineering is out of scope.

- **Single-Model Causal Scope** : Knockout and mediation are evaluated per trained architecture seed; cross-architecture generalization is future work.

## 🐛 Troubleshooting

| Issue / Error | Root Cause | Verified Solution |
| ---- | ---- | ---- |
| ModuleNotFoundError: conllu | Missing dependency | pip install conllu networkx gensim matplotlib |
| fit_temperature_on_val() takes 4 positional arguments... | Passing attention_mask | Signature is fit_temperature_on_val(model, X_val, y_val, device). Do not pass mask tensors. |
| ValueError: too many values to unpack (expected 2, got 3) | Temperature calibration returns 3-tuple | Returns (opt_T, converged, (ci_low, ci_high)). Unpack all 3. |
| TypeError: compute_stratified_bootstrap_ece() missing... | Passing 2D probability matrix | Pass confidences=probs.max(axis=-1), correct_flags=(preds==y).astype(int), labels=y. Returns 3-tuple. |
| AttributeError: ... has no attribute 'apply_head_mask' | Non-existent method | Use apply_attention_head_knockout_exact(model, layer_idx, head_idx) which zeros out_proj.weight / out_proj.bias. |
| `KeyError: 'quantifier'` / all `<UNK>` predictions | Misaligned vocabulary | Ensure `processor.quantifier_to_id` is initialized via `split_three_way()` and pass the identical `--train_corpus` to `eval_double_blind_attractor.py`. |
| KeyError: 'acc_weighted_std' | Duplicate aggregator in LLM script | Ensure only one definition of aggregate_template_results exists in eval_llm_rigorous.py. |
| AttributeError: 'BulletproofUDProcessor' has no 'reset_all_audit' | Missing reset method | Ensure reset_all_audit() is defined in BulletproofUDProcessor (prevents cross-seed counter leakage). |
| results/ablation_summary_splits1_DEPRECATED.json numbers do not match Table 1 | Reading deprecated single-split artifact | Use results/ablation_summary_10split.json (10-split aggregate). Legacy file is now results/ablation_summary_splits1_DEPRECATED.json |

## 📜 License

This repository is released under the MIT License (see LICENSE). During double-blind review, author identities and affiliations are omitted; the camera-ready repository will include the full author list and institutional acknowledgements.
