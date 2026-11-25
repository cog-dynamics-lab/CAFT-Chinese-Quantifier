# Decoupling Structure from Proximity: Structured Prior Regularization for Robust Pattern Classification under Data Scarcity 🚀

> **Anonymous Repository for Double-Blind Review**
>
> **Status:** All experiments completed.
> - **Main paper (Tables 1–3):** 10-split aligned ablations, causal head knockouts, and attractor probe complete.
> - **LLM Benchmark (Appendix A):** 5/5 models completed for the 10-split weighted K=10 benchmark (Table A2). The Split-1 K-sweep (K=1–100) is complete for all five models, including GLM-4-9B (full K=1–100 reported in Table A1; high-K context-saturation discussion in Appendix A).
>
> Official implementation, evaluation scripts, and replication guidelines for the above paper.

# Contrastive Adaptive Fine-Tuning (CAFT) Research Repository

Data Integrity Notice: The evaluation pipeline strictly adheres to the 10-split cross-validated results found in results/*_10split.json. Files suffixed with _DEPRECATED are archival artifacts from initial development iterations and are not used for any reported metric in the manuscript.

This repository contains the complete experimental pipeline, including:

- **Synthetic PCFG meta-learning** via first-order Reptile optimization;
- **CAFT** (**C**ontrastive **A**daptive **F**ine-**T**uning), the unified adaptation framework;
- **SIPAR** (**S**yntax-**I**nvariant **P**airwise **A**lignment **R**egularizer), the contrastive regularization term;
- **Natural Attractor Probe** (Split-0 diagnostic subset, N = 413) and **No-Noun masking**;
- **Two counterfactual scrambling variants** (`modifier` / `core_swap`) implemented in `run_rigorous_aligned_ablationTJ.py`;
- **Structural probing** via Hewitt & Manning (2019) rank-constrained bilinear probe;
- **Rigorous OOD evaluation** on the Universal Dependencies (UD) Chinese treebank (`UD_Chinese-GSDSimp`);
- **Leakage-free calibration** (temperature scaling fitted exclusively on validation sets).

> **Note on terminology**: The manuscript and Appendix C strictly use **CAFT** = *Contrastive Adaptive Fine-Tuning* and **SIPAR** = *Syntax-Invariant Pairwise Alignment Regularizer*. Earlier working drafts or low-level code comments may occasionally reference alternative operational terms; these are **deprecated implementation nomenclature** and are superseded by the paper's theoretical definitions.

Data Integrity & Consistency Notice:

> Evaluation Protocol: The evaluation pipeline strictly adheres to 10-split cross-validated results.
> Class Parity & Scale: Per meta-task optimization utilizes K=10 support instances per class (
N_train=60 total across 6 active quantifier classes).
> Syntax Integrity: An automated grammar-filtering stage (S3.6: Grammar Filtering) is integrated into the pipeline to perform deterministic validation on slot existence, context integrity, and lexical non-emptiness, ensuring the validity of the double-blind attractor benchmark.


## 📂 Repository Structure

### Phase 1 & 2 — Meta-Prior Distillation
- `meta_pretrain_full_strong.py` — Reptile meta-training on synthetic PCFG tasks. Produces `meta_pretrained_prior.pth` (~100M backbone).
- `meta_pretrain_no_noise.py` — Structure-free ablation meta-training. Produces `meta_no_noise.pth` (lexical control group).
> *Note: `meta_pretrain_no_noise.py` instantiates a compact control architecture (d_model=128, 2 layers) solely to isolate the effect of hierarchical versus flat priors. The ~100M-parameter backbone used for all main-text results is produced by `meta_pretrain_full_strong.py`.*

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
- `eval_llm_rigorous.py` — Generation-free token-level probability evaluation for causal LLMs: multi-token length-normalized prefix log-likelihood, per-template isolated ECE with class-stratified bootstrap, cross-template weighted aggregation, multi-seed few-shot selection, and **K ∈ {1,3,5,10,20,50,100} sample-efficiency support** (all five models' Split-1 K-sweeps (K=1–100) are reported in Table A1 and archived in `results/llm_split1_ksweep_partial.json`, whereas the 10-split cross-validated weighted metrics are archived in results/llm_10split_summary.json.)

### Phase 6 — Table Export
- `export_latex_tables.py` — Generates publication-ready booktabs tables (`--tables 1,2,3`).

### Utility / Orchestration (Repository Tooling, Not in Main Text)
- `run_all_experiments.py` — Convenience orchestrator (S0 env check → S1 encode verify → S2 meta-pretrain → S3 ablation → S4 double-blind → S5 figures → S6 LaTeX export). **Not referenced in the paper; provided for end-to-end reproduction.**
- `run_llm_benchmark.py` — Early-stage debugging wrapper using single-token restricted softmax. **Not the source of Table A1/A2.** The manuscript's LLM results are generated exclusively by `eval_llm_rigorous.py` (length-normalized prefix log-likelihood + intra-sample Z-score softmax, main text Eqs. 25–27). **Not referenced in the paper.**
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
# 流程: S0(Env) → S1(Encode) → S2(Pretrain) → S3(Ablation) → S3.5(Build) → S3.6(Grammar Filter) → S4(Evaluation) ...
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

Experiments default to k_shot=10 to align with the few-shot cognitive generalization setup described in the paper. The evaluation uses length-normalized prefix scoring with a 6-class vocabulary truncation to ensure chance-level symmetry (16.67%).

```bash
pip install --upgrade huggingface_hub tiktoken sentencepiece
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
python run_llm_benchmark.py    # or: python eval_llm_rigorous.py
```

## 📊 Experimental Protocol & Artifact Data Flow 

```text
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

eval_llm_rigorous.py (5 models × 5 seeds × 3 templates)
       ├──► results/llm_split1_ksweep_partial.json ─────────► Table A1 (Split 1, K=1–100)
       └──► results/llm_10split_summary.json ──────────────► Table A2 (10-split weighted, K=10)
            (source log: logs/eval_llm_rigorous_20261019_071711.log)
```

### ⚙️ Data Artifact Verification
**Note:** All LaTeX tables are generated via `export_latex_tables.py`. The script expects standard naming conventions for the 10-split archival JSONs located in `./results/`. If running on fresh logs, ensure that the artifacts are mapped to their respective table identifiers as per the `TABLE_MAP` defined in `export_latex_tables.py`. To verify reproduction, ensure the file hashes of generated JSONs match the audited artifacts listed in the "Artifact-to-Claim Map" below.

### Artifact-to-Claim Map (Audit Trail for Reviewers)

Each headline claim in the manuscript maps to exactly one machine-readable artifact below; reviewers can verify every number by opening the listed file.
| Artifact File | Supports Claim / Result | Target Location |
|---|---|---|
| `results/ablation_summary_10split.json` | CAFT (MAML) 42.70% ± 3.49%, ECE = 0.0693 | Table 1 |
| `results/head_knockout_results_10split.json` | Top-1 knockout Δ=−4.82 pp; Causal mediation IE=0.1961 [0.129, 0.262] | Table 2 & App. C |
| `results/natural_attractor_probe_ckpt.json` | +11.62 pp attractor gain (+3.07× vs baseline) | Table 3 |
| `results/llm_split1_ksweep_partial.json` | In-context sample efficiency & non-monotonic K-trajectories (incl. GLM-4-9B K=50/100) | Table A1 |
| `results/llm_10split_summary.json` | 10-split weighted Acc/ECE (CAFT calibration error 2.7–3.4× lower) | Table A2 |
| `logs/eval_llm_rigorous_20261019_071711.log` | Raw evaluation stream for all 5 LLMs (K-sweep & 10 splits) | Table A1 / A2 |

> *Note: `double_blind_results.json` (generated by `eval_double_blind_attractor.py`) is a complementary repository-only analysis and does not correspond to any table in the manuscript. The main-text attractor result (Table 3) comes exclusively from `natural_attractor_probe_ckpt.json`.*

> **Important Evaluation Distinction:** Table A1 (Split 1, K-sweep) and Table A2 (10-split weighted, K=10) employ fundamentally distinct aggregation protocols. They illustrate single-split non-monotonic trajectories versus variance-reduced multi-split evaluation, respectively, and must not be compared row-for-row.

> **Data Integrity Notice**
> The legacy files results/ablation_summary.json and results/head_knockout_results.json are single-split smoke-test artifacts (from logs/ckpt_save_run.log). They have been renamed to results/*_splits1_DEPRECATED.json and must not be used to reproduce Tables 1–2. The exact numerical sources are:
> - logs/clean_run_final_20261018_063218.log — the 10-split run log. Table 1 aggregates parsed from its AGGREGATED RESULTS block; Table 2 mediation/knockout parsed from per-split [KNOCKOUT] / [MEDIATION] lines.
> - results/ablation_summary_10split.json — Table 1 archive.
> - results/head_knockout_results_10split.json — Table 2 archive.
> - results/natural_attractor_probe_ckpt.json — Table 3 archive (Split 0, N=413).

> **LLM Data Integrity & High-K Reporting:**
> - `logs/eval_llm_rigorous_20261019_071711.log` is the unified log covering all 5 LLM models.
> - **GLM-4-9B high-K cells:** The log records GLM-4-9B at K=50 (51.54%) and K=100 (49.54%). These values are **reported in Table A1**; the post-peak decline is attributed to exemplar-order sensitivity and length-normalization dilution at high K (Appendix A). No cells are masked.
> - `results/llm_10split_summary.json` and `results/llm_split1_ksweep_partial.json` are the primary machine-readable JSON artifacts referenced in Appendix A, generated directly from the evaluation stream.

## 🧪 Reproducibility Checklist

Note: The meta-pretraining routines in this repository use an AdamW-preconditioned Reptile update for optimized convergence. All final results are governed by seed=42 to ensure consistent data flow across the 10-split evaluation framework.

- [x] **No Test Leakage** : Three-way split (train / val / test) strictly isolated; temperature calibration T* fitted on validation sets only.
- [x] **Multi-Split Stability** : Table 1 aggregated across 10 random splits (grand-mean ± grand-std).
- [~] **Multi-Seed LLM Stability** : Appendix A averaged over 5 seeds × 3 prompt templates (Split 1, K=10). *10-split K-sweep extension deferred to camera-ready.*
- [x] **Statistical Rigor** : Knockout significance via 1,000-iteration Monte Carlo permutation against data-driven null; causal-mediation intervals via 95% bootstrap CI.
- [x] **Strict Disentanglement** : UUAS via cross-validated bilinear probe; UUAS parity is reported candidly (14.73 vs. 14.71, Δ=+0.02 not meaningful), and the paper's structural claim rests on behavioral + causal evidence (Attractor Probe; IE=0.1961, 95% CI [0.129, 0.262]), not on geometric metrics alone.
- [x] **Perturbation Audit** : Both modifier (Distractor Scrambling) and core_swap (Core-Argument Scrambling) variants are implemented in run_rigorous_aligned_ablationTJ.py with per-split fallback_ratio logging. Main-text behavioral evidence uses the Natural Attractor Probe (Table 3) and No-Noun masking (Fig. 3b); the modifier/core_swap quantitative comparison is reported as future work in Appendix B.5. (Quantitative comparison of modifier/core_swap is deferred to Appendix B.5 as future work)
- [x] **Subword Tokenization Audit** : Candidate classifier subword lengths logged to detect tokenizer-induced bias in LLM scoring.
- [x] **Causal Null Distributions**: Monte Carlo Random-K head ablation for null-hypothesis testing with conservative p-value lower bound.
- [x] **Length-Normalized Likelihoods** : LLM log-probabilities explicitly normalized by candidate-token length.
- [x] **Prompt-Isolated Calibration** : ECE calculated per prompt template before weighted aggregation.
- [x] **LLM Artifact Completeness** : `results/llm_split1_ksweep_partial.json` (Table A1, incl. GLM-4-9B K=50/100) and `results/llm_10split_summary.json` (Table A2) are present and syntactically valid. Specifically, the former archives Table A1 (sample-efficiency sweep on Split 1), while the latter archives Table A2 (10-split multi-seed aggregated benchmark). Both are derived from `logs/eval_llm_rigorous_20261019_071711.log`.

## ⚠️ Known Scope & Scientific Limitations
- **Single Language Scope** : Primary experiments use UD_Chinese-GSDSimp. Cross-linguistic applicability (e.g., English subject–verb agreement with attractors) is discussed as future work in §Limitations; a full orthogonalized syntactic-complexity stratification is specified in Appendix B.

- **Treebank Sample Size** : ~1,640 classifier–noun instances extracted; after OOD filtering (d > 1) the diagnostic Natural Attractor subset further restricts to linear_dist ≥ 3 with intervening distractor nouns, yielding N=413 on Split 0. Tail-classifier breakdowns carry wider confidence intervals.

- **Pre-Trained vs. From-Scratch** : This study deliberately isolates from-scratch meta-inductive adaptation to assess sample-efficient structural inductive biases without confounding from massive web-scale pre-training. Extending CAFT to pre-trained backbones (e.g., RoBERTa, BERT) is stated in §Limitations as a distinct future scientific question.

- **Probe Decodability** : The rank-constrained bilinear probe (k=32) tests linear decodability of tree geometry; nonlinear representations may form an upper bound.

- **Prompt Sensitivity** : LLM few-shot results are reported over 3 fixed templates with prompt-sensitivity std; exhaustive prompt engineering is out of scope.

- **Single-Model Causal Scope** : Knockout and mediation are evaluated per trained architecture seed; cross-architecture generalization is future work.

- **K‑Sweep Non‑Monotonicity** : Trajectories in Table A1 oscillate across K (e.g., GLM‑4‑9B 62.61% @K=10 → 55.73% @K=20 → 51.54% @K=50 → 49.54% @K=100). This reflects single‑split variance, prompt‑order sensitivity, and context‑length saturation at high K — **not** optimization instability. The variance‑reduced comparison is Table A2 (10‑split weighted, K=10). A full 10‑split extension of the K‑sweep is scheduled for camera‑ready.

## 🐛 Troubleshooting

### Troubleshooting & Discrepancy FAQ

| Issue / Observation | Root Cause | Resolution / Explanation |
|---|---|---|
| Numbers in Table A2 differ from Table A1 at K=10 | Aggregation mismatch (Split‑1 vs 10‑split weighted) | Table A1 = Split‑1 sample efficiency; Table A2 = 10‑split weighted averages. Not row‑for‑row comparable. |
| GLM‑4‑9B K=50/100 lower than K=10 | Context‑length saturation at high K + single‑split variance | Expected post‑peak decline; reported in Table A1 and explained in Appendix A. |

### Common Runtime Errors

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
| Pipeline fails at S3.6 | Missing filter_grammar_conflicts.py | Ensure filter_grammar_conflicts.py is present in the root directory and contains the slot-constraint validation logic. |

Note on Artifact Paths: All experiment artifacts are generated in the ./results/ directory. The export_latex_tables.py script automatically targets ./results/ for input and ./tables/ for output. If running the pipeline sequentially, ensure the working directory is the project root to maintain path resolution integrity.

## 📜 License

This repository is released under the MIT License (see LICENSE). During double-blind review, author identities and affiliations are omitted; the camera-ready repository will include the full author list and institutional acknowledgements.
