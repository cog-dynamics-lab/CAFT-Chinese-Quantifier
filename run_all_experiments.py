#!/usr/bin/env python3
"""
Module: run_all_experiments.py
Description:
    One-shot orchestrator for the complete CAFT experimental pipeline.

    Stages:
        S0. Environment self-check (data + dependencies)
        S1. Encoding verification (verify_encode.py)
        S2. Meta-prior distillation (meta_pretrain_full_strong.py)
        S3. Main ablation (run_rigorous_aligned_ablationTJ.py)
        S4. Double-blind evaluation (eval_double_blind_attractor.py)
        S5. Figure generation (visualize_double_blind_results.py)
        S6. LaTeX table export (export_latex_tables.py)

    Fail-fast:
        Any non-zero exit code aborts the pipeline immediately.

Usage:
    # Full pipeline
    python run_all_experiments.py --config default

    # Skip meta-pretraining (if .pth already exists)
    python run_all_experiments.py --skip_s2

    # Only re-run double-blind eval + plotting
    python run_all_experiments.py --skip_s0 --skip_s1 --skip_s2 --skip_s3 --skip_s6
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG: Dict[str, Any] = {
    # ---------------- Paths ----------------
    "train_corpus": "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-train.conllu",
    "test_corpus": "UD_Chinese-GSDSimp-master/zh_gsdsimp-ud-test.conllu",
    "meta_prior_path": "meta_pretrained_prior.pth",
    "benchmark_path": "double_blind_benchmark.json",
    "external_corpus": None,        # e.g., "path/to/corpus.txt"; None = skip N-gram baseline
    "fasttext_path": None,          # e.g., "path/to/fasttext.model"; None = skip

    # ---------------- Directories ----------------
    "results_dir": "results",
    "figures_dir": "figures",
    "tables_dir": "tables",
    "checkpoints_dir": "results/checkpoints",

    # ---------------- Ablation hyperparameters ----------------
    "num_splits": 10,
    "top_k": 6,
    "k_shot": 10,
    "k_calib": 10,

    # ---------------- Meta-pretraining hyperparameters ----------------
    "meta_epochs": 3000,
    "meta_lr": 1.0,
    "inner_lr": 0.01,
    "inner_steps": 10,

    # ---------------- Double-blind eval ----------------
    "caft_ckpt": None,              # Will be auto-detected if None
    "baseline_ckpt": None,          # Will be auto-detected if None
    "num_classes": 6,

    # ---------------- Figure format ----------------
    "figure_format": "pdf",
}


# =============================================================================
# 2. UTILITIES
# =============================================================================

def print_banner(text: str, char: str = "=") -> None:
    width = 80
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def check_file(path: str, description: str) -> bool:
    if not os.path.exists(path):
        print(f"[MISSING] {description}: {path}")
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[OK] {description}: {path} ({size_mb:.2f} MB)")
    return True


def check_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        print(f"[OK] Module: {module_name}")
        return True
    except ImportError:
        print(f"[MISSING] Module: {module_name} -> pip install {module_name}")
        return False


def run_stage(
    stage_name: str,
    cmd: List[str],
    cwd: Optional[str] = None,
    skip: bool = False,
) -> Tuple[bool, float]:
    """
    Run a subprocess stage.

    Returns:
        (success, elapsed_seconds)
    """
    if skip:
        print(f"\n[SKIP] {stage_name}")
        return True, 0.0

    print_banner(f"STAGE: {stage_name}")
    print(f"[CMD] {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,             # We check returncode manually
            capture_output=False,    # Stream to stdout for live progress
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"\n[PASS] {stage_name} completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n[FAIL] {stage_name} exited with code {result.returncode} "
                  f"after {elapsed:.1f}s")
            return False, elapsed

    except FileNotFoundError as e:
        print(f"\n[FATAL] {stage_name}: command not found -> {e}")
        return False, 0.0
    except Exception as e:
        print(f"\n[FATAL] {stage_name}: unexpected error -> {e}")
        return False, 0.0


# =============================================================================
# 3. STAGE 0: ENVIRONMENT SELF-CHECK
# =============================================================================

def stage_0_env_check(config: Dict[str, Any]) -> bool:
    print_banner("STAGE 0: ENVIRONMENT SELF-CHECK")

    all_ok = True

    # 1. Check Python version
    print(f"[INFO] Python: {sys.version}")
    if sys.version_info < (3, 8):
        print("[FAIL] Python >= 3.8 required")
        all_ok = False

    # 2. Check required modules
    print("\n--- Module Check ---")
    for mod in ["torch", "numpy", "conllu", "networkx",
                "transformers", "gensim", "matplotlib", "sklearn"]:
        if not check_module(mod):
            all_ok = False

    # 3. Check data files
    print("\n--- Data Check ---")
    if not check_file(config["train_corpus"], "Training corpus"):
        all_ok = False

    if config.get("test_corpus") and not check_file(
        config["test_corpus"], "Test corpus"
    ):
        print("  (Test corpus is optional; only training corpus is required)")

    # 4. Check directories
    print("\n--- Directory Check ---")
    for d in [config["results_dir"], config["figures_dir"],
              config["tables_dir"], config["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)
        print(f"[OK] Directory: {d}")

    # 5. Check GPU
    print("\n--- Device Check ---")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[WARN] CUDA not available; running on CPU (slow)")
    except Exception as e:
        print(f"[WARN] Could not check CUDA: {e}")

    # 6. Check script files
    print("\n--- Script Check ---")
    scripts = [
        "verify_encode.py",
        "meta_pretrain_full_strong.py",
        "run_rigorous_aligned_ablationTJ.py",
        "build_double_blind_attractor.py",
        "eval_double_blind_attractor.py",
        "visualize_double_blind_results.py",
        "export_latex_tables.py",
    ]
    for s in scripts:
        if not check_file(s, f"Script {s}"):
            all_ok = False

    return all_ok


# =============================================================================
# 4. STAGE 1: ENCODING VERIFICATION
# =============================================================================

def stage_1_verify_encode(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    cmd = [
        sys.executable, "verify_encode.py",
        "--train_corpus", config["train_corpus"],
        "--benchmark", config["benchmark_path"],
        "--num_classes", str(config["num_classes"]),
    ]
    # If benchmark doesn't exist yet, skip benchmark audit
    if not os.path.exists(config["benchmark_path"]):
        cmd.append("--skip_benchmark")

    success, _ = run_stage("S1: Encoding Verification", cmd, skip=skip)
    return success


# =============================================================================
# 5. STAGE 2: META-PRIOR DISTILLATION
# =============================================================================

def stage_2_meta_pretrain(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    # If prior already exists and skip not forced, ask
    prior_path = config["meta_prior_path"]
    if os.path.exists(prior_path) and not skip:
        print(f"\n[INFO] Prior already exists: {prior_path}")
        print("[INFO] Skipping meta-pretraining. Use --force_s2 to overwrite.")
        return True

    cmd = [
        sys.executable, "meta_pretrain_full_strong.py",
        "--meta_epochs", str(config["meta_epochs"]),
        "--outer_lr", str(config["meta_lr"]),          # === PATCH 3 ===
        "--inner_lr", str(config["inner_lr"]),
        "--inner_steps", str(config["inner_steps"]),
        "--save_path", prior_path,
    ]
    success, _ = run_stage("S2: Meta-Prior Distillation", cmd, skip=skip)
    return success


# =============================================================================
# 6. STAGE 3: MAIN ABLATION
# =============================================================================

def stage_3_ablation(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    cmd = [
        sys.executable, "run_rigorous_aligned_ablationTJ.py",
        "--num_splits", str(config["num_splits"]),
        "--top_k", str(config["top_k"]),
        "--k_shot", str(config["k_shot"]),
        "--k_calib", str(config["k_calib"]),
    ]
    success, _ = run_stage("S3: Main Ablation (8 configs × N splits)", cmd, skip=skip)

    if success:
        # Verify outputs
        for f in ["ablation_summary.json", "head_knockout_results.json"]:
            path = os.path.join(config["results_dir"], f)
            if os.path.exists(path):
                print(f"[OUTPUT] {path} ({os.path.getsize(path)} bytes)")
            else:
                print(f"[WARN] Expected output missing: {path}")

    return success


# =============================================================================
# 7. STAGE 3.5: BUILD BENCHMARK (if not exists)
# =============================================================================

def stage_3_5_build_benchmark(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    """Build double-blind benchmark if it doesn't exist."""
    benchmark_path = config["benchmark_path"]
    if os.path.exists(benchmark_path) and not skip:
        print(f"\n[INFO] Benchmark already exists: {benchmark_path}")
        print("[INFO] Skipping benchmark construction.")
        return True

    cmd = [
        sys.executable, "build_double_blind_attractor.py",
        "--output", benchmark_path,
        "--seed", "42",
    ]
    if config.get("external_corpus"):
        cmd.extend(["--corpus", config["external_corpus"]])
    if config.get("fasttext_path"):
        cmd.extend(["--fasttext", config["fasttext_path"]])

    success, _ = run_stage("S3.5: Build Double-Blind Benchmark", cmd, skip=skip)
    return success


# =============================================================================
# 8. STAGE 4: DOUBLE-BLIND EVALUATION
# =============================================================================

def auto_detect_checkpoints(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect CAFT and Baseline checkpoints in results/ if not specified."""
    caft_ckpt = config.get("caft_ckpt")
    baseline_ckpt = config.get("baseline_ckpt")

    if caft_ckpt is None or baseline_ckpt is None:
        # Search for .pth files in results/
        results_dir = config["results_dir"]
        candidates = []
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                if f.endswith(".pth"):
                    candidates.append(os.path.join(results_dir, f))

        if caft_ckpt is None:
            for c in candidates:
                if "caft" in c.lower():
                    caft_ckpt = c
                    break
        if baseline_ckpt is None:
            for c in candidates:
                if "baseline" in c.lower():
                    baseline_ckpt = c
                    break

    return caft_ckpt, baseline_ckpt


def stage_4_double_blind(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    caft_ckpt, baseline_ckpt = auto_detect_checkpoints(config)

    if caft_ckpt is None or baseline_ckpt is None:
        print(f"\n[WARN] CAFT or Baseline checkpoint not found.")
        print(f"  CAFT:     {caft_ckpt}")
        print(f"  Baseline: {baseline_ckpt}")
        print("[INFO] Running statistical-only evaluation.")
    else:
        print(f"\n[INFO] CAFT ckpt:     {caft_ckpt}")
        print(f"[INFO] Baseline ckpt: {baseline_ckpt}")

    cmd = [
        sys.executable, "eval_double_blind_attractor.py",
        "--benchmark", config["benchmark_path"],
        "--train_corpus", config["train_corpus"],
        "--num_classes", str(config["num_classes"]),
        "--output", os.path.join(config["results_dir"], "double_blind_results.json"),
    ]
    if caft_ckpt:
        cmd.extend(["--caft_ckpt", caft_ckpt])
    if baseline_ckpt:
        cmd.extend(["--baseline_ckpt", baseline_ckpt])
    if os.path.exists(config["meta_prior_path"]):
        cmd.extend(["--prior_ckpt", config["meta_prior_path"]])
    if config.get("external_corpus"):
        cmd.extend(["--corpus", config["external_corpus"]])
    if config.get("fasttext_path"):
        cmd.extend(["--fasttext", config["fasttext_path"]])

    success, _ = run_stage("S4: Double-Blind Evaluation", cmd, skip=skip)

    if success:
        output_path = os.path.join(config["results_dir"], "double_blind_results.json")
        if os.path.exists(output_path):
            print(f"[OUTPUT] {output_path} ({os.path.getsize(output_path)} bytes)")

    return success


# =============================================================================
# 9. STAGE 5: FIGURE GENERATION
# =============================================================================

def stage_5_visualize(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    results_path = os.path.join(config["results_dir"], "double_blind_results.json")
    if not os.path.exists(results_path):
        print(f"\n[SKIP] {results_path} not found; cannot plot.")
        return True

    cmd = [
        sys.executable, "visualize_double_blind_results.py",
        "--results", results_path,
        "--output_dir", config["figures_dir"],
        "--format", config["figure_format"],
        "--figures", "1,2,3",
    ]
    success, _ = run_stage("S5: Figure Generation", cmd, skip=skip)

    if success:
        for fig in ["Fig1_main_benchmark", "Fig2_closed_vs_open", "Fig3_heatmap"]:
            path = os.path.join(config["figures_dir"],
                                f"{fig}.{config['figure_format']}")
            if os.path.exists(path):
                print(f"[OUTPUT] {path}")

    return success


# =============================================================================
# 10. STAGE 6: LATEX EXPORT
# =============================================================================

def stage_6_latex_export(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    cmd = [
        sys.executable, "export_latex_tables.py",
        "--results_dir", config["results_dir"],
        "--output_dir", config["tables_dir"],
        "--tables", "1,2,3,4",
    ]
    success, _ = run_stage("S6: LaTeX Table Export", cmd, skip=skip)

    if success:
        for t in ["table1_main_ablation.tex", "table2_double_blind.tex",
                  "table3_head_knockout.tex", "table4_llm_benchmark.tex"]:
            path = os.path.join(config["tables_dir"], t)
            if os.path.exists(path):
                print(f"[OUTPUT] {path}")

    return success


# =============================================================================
# 11. STAGE 7 (OPTIONAL): LLM BENCHMARK
# =============================================================================

def stage_7_llm_benchmark(
    config: Dict[str, Any],
    skip: bool,
) -> bool:
    """Optional LLM benchmark (requires local model paths in eval_llm_rigorous.py)."""
    cmd = [sys.executable, "eval_llm_rigorous.py"]
    success, _ = run_stage("S7: LLM Benchmark (optional)", cmd, skip=skip)
    return success


# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================

def print_final_summary(
    config: Dict[str, Any],
    stage_results: Dict[str, Tuple[bool, float]],
) -> None:
    print_banner("EXPERIMENT SUMMARY")

    total_time = sum(t for _, t in stage_results.values())
    all_passed = all(s for s, _ in stage_results.values())

    print(f"\n{'Stage':<40} {'Status':<10} {'Time':<10}")
    print("-" * 60)
    for name, (success, elapsed) in stage_results.items():
        status = "PASS" if success else "FAIL"
        print(f"{name:<40} {status:<10} {elapsed:>7.1f}s")
    print("-" * 60)
    print(f"{'TOTAL':<40} {'PASS' if all_passed else 'FAIL':<10} {total_time:>7.1f}s")

    # Output artifact inventory
    print(f"\n{'ARTIFACT INVENTORY':^60}")
    print("-" * 60)

    inventory = [
        (config["results_dir"], "ablation_summary.json"),
        (config["results_dir"], "head_knockout_results.json"),
        (config["results_dir"], "double_blind_results.json"),
        (config["results_dir"], "llm_benchmark_summary.json"),
        (config["tables_dir"], "table1_main_ablation.tex"),
        (config["tables_dir"], "table2_double_blind.tex"),
        (config["tables_dir"], "table3_head_knockout.tex"),
        (config["tables_dir"], "table4_llm_benchmark.tex"),
    ]
    for fig in ["Fig1_main_benchmark", "Fig2_closed_vs_open", "Fig3_heatmap"]:
        inventory.append(
            (config["figures_dir"], f"{fig}.{config['figure_format']}")
        )

    for d, f in inventory:
        path = os.path.join(d, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"[OK]     {path:<60} ({size_str})")
        else:
            print(f"[MISSING] {path:<60}")

    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] All stages completed successfully.")
        print("[NEXT]   Review figures/ and tables/ for manuscript integration.")
    else:
        print("\n[FAILURE] Some stages failed. Review above output.")
        print("[NEXT]   Fix errors and re-run with --skip_* to resume from failure point.")

    # Save summary as JSON
    summary_path = os.path.join(config["results_dir"], "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "stages": {
                name: {"success": s, "elapsed_sec": t}
                for name, (s, t) in stage_results.items()
            },
            "all_passed": all_passed,
            "total_time_sec": total_time,
        }, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot orchestrator for the complete CAFT experimental pipeline."
    )

    # Stage skip toggles
    parser.add_argument("--skip_s0", action="store_true", help="Skip environment check")
    parser.add_argument("--skip_s1", action="store_true", help="Skip encoding verification")
    parser.add_argument("--skip_s2", action="store_true", help="Skip meta-pretraining")
    parser.add_argument("--skip_s3", action="store_true", help="Skip main ablation")
    parser.add_argument("--skip_s3_5", action="store_true", help="Skip benchmark construction")
    parser.add_argument("--skip_s4", action="store_true", help="Skip double-blind evaluation")
    parser.add_argument("--skip_s5", action="store_true", help="Skip figure generation")
    parser.add_argument("--skip_s6", action="store_true", help="Skip LaTeX export")
    parser.add_argument("--skip_s7", action="store_true", default=True,
                        help="Skip LLM benchmark by default")

    # Force flags
    parser.add_argument("--force_s2", action="store_true",
                        help="Force re-run meta-pretraining even if .pth exists")
    parser.add_argument("--run_llm", action="store_true",
                        help="Run LLM benchmark (S7). Overrides --skip_s7")

    # Config overrides
    parser.add_argument("--train_corpus", type=str,
                        default=EXPERIMENT_CONFIG["train_corpus"])
    parser.add_argument("--num_splits", type=int,
                        default=EXPERIMENT_CONFIG["num_splits"])
    parser.add_argument("--top_k", type=int, default=EXPERIMENT_CONFIG["top_k"])
    parser.add_argument("--k_shot", type=int, default=EXPERIMENT_CONFIG["k_shot"])
    parser.add_argument("--figure_format", type=str,
                        default=EXPERIMENT_CONFIG["figure_format"],
                        choices=["pdf", "png", "svg"])
    parser.add_argument("--caft_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--external_corpus", type=str, default=None)
    parser.add_argument("--fasttext_path", type=str, default=None)

    args = parser.parse_args()

    # Merge args into config
    config = EXPERIMENT_CONFIG.copy()
    config.update({
        "train_corpus": args.train_corpus,
        "num_splits": args.num_splits,
        "top_k": args.top_k,
        "k_shot": args.k_shot,
        "num_classes": args.top_k,
        "figure_format": args.figure_format,
        "caft_ckpt": args.caft_ckpt,
        "baseline_ckpt": args.baseline_ckpt,
        "external_corpus": args.external_corpus,
        "fasttext_path": args.fasttext_path,
    })

    # Print config
    print_banner("CAFT FULL EXPERIMENT PIPELINE")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # Stage execution
    stage_results: Dict[str, Tuple[bool, float]] = {}

    # S0: Environment
    if not args.skip_s0:
        t0 = time.time()
        ok = stage_0_env_check(config)
        stage_results["S0: Environment Check"] = (ok, time.time() - t0)
        if not ok:
            print("\n[FATAL] Environment check failed. Aborting.")
            print_final_summary(config, stage_results)
            sys.exit(1)
    else:
        stage_results["S0: Environment Check"] = (True, 0.0)

    # S1: Encoding verification
    ok, t = run_stage(
        "S1: Encoding Verification",
        [sys.executable, "verify_encode.py",
         "--train_corpus", config["train_corpus"],
         "--benchmark", config["benchmark_path"],
         "--num_classes", str(config["num_classes"])],
        skip=args.skip_s1,
    )
    stage_results["S1: Encoding Verification"] = (ok, t)
    if not ok:
        print("\n[FATAL] Encoding verification failed. Aborting.")
        print_final_summary(config, stage_results)
        sys.exit(1)

    # S2: Meta-pretraining
    if args.force_s2:
        args.skip_s2 = False
    ok, t = run_stage(
        "S2: Meta-Prior Distillation",
        [sys.executable, "meta_pretrain_full_strong.py",
         "--meta_epochs", str(config["meta_epochs"]),
         "--outer_lr", str(config["meta_lr"]),         # === PATCH 3 ===
         "--inner_lr", str(config["inner_lr"]),
         "--inner_steps", str(config["inner_steps"]),
         "--save_path", config["meta_prior_path"]],
        skip=args.skip_s2,
    )
    stage_results["S2: Meta-Prior Distillation"] = (ok, t)
    if not ok:
        print("\n[FATAL] Meta-pretraining failed. Aborting.")
        print_final_summary(config, stage_results)
        sys.exit(1)

    # S3: Main ablation
    ok, t = run_stage(
        "S3: Main Ablation",
        [sys.executable, "run_rigorous_aligned_ablationTJ.py",
         "--num_splits", str(config["num_splits"]),
         "--top_k", str(config["top_k"]),
         "--k_shot", str(config["k_shot"]),
         "--k_calib", str(config["k_calib"])],
        skip=args.skip_s3,
    )
    stage_results["S3: Main Ablation"] = (ok, t)
    if not ok:
        print("\n[FATAL] Main ablation failed. Aborting.")
        print_final_summary(config, stage_results)
        sys.exit(1)

    # S3.5: Build benchmark (if not exists)
    ok, t = run_stage(
        "S3.5: Build Benchmark",
        [sys.executable, "build_double_blind_attractor.py",
         "--output", config["benchmark_path"],
         "--seed", "42"],
        skip=args.skip_s3_5,
    )
    stage_results["S3.5: Build Benchmark"] = (ok, t)

    # S4: Double-blind evaluation
    caft_ckpt, baseline_ckpt = auto_detect_checkpoints(config)
    cmd_s4 = [
        sys.executable, "eval_double_blind_attractor.py",
        "--benchmark", config["benchmark_path"],
        "--train_corpus", config["train_corpus"],
        "--num_classes", str(config["num_classes"]),
        "--output", os.path.join(config["results_dir"], "double_blind_results.json"),
    ]
    if caft_ckpt:
        cmd_s4.extend(["--caft_ckpt", caft_ckpt])
    if baseline_ckpt:
        cmd_s4.extend(["--baseline_ckpt", baseline_ckpt])
    if os.path.exists(config["meta_prior_path"]):
        cmd_s4.extend(["--prior_ckpt", config["meta_prior_path"]])
    if config.get("external_corpus"):
        cmd_s4.extend(["--corpus", config["external_corpus"]])
    if config.get("fasttext_path"):
        cmd_s4.extend(["--fasttext", config["fasttext_path"]])

    ok, t = run_stage(
        "S4: Double-Blind Evaluation",
        cmd_s4,
        skip=args.skip_s4,
    )
    stage_results["S4: Double-Blind Evaluation"] = (ok, t)
    if not ok:
        print("\n[FATAL] Double-blind evaluation failed. Aborting.")
        print_final_summary(config, stage_results)
        sys.exit(1)

    # S5: Figures
    ok, t = run_stage(
        "S5: Figure Generation",
        [sys.executable, "visualize_double_blind_results.py",
         "--results", os.path.join(config["results_dir"], "double_blind_results.json"),
         "--output_dir", config["figures_dir"],
         "--format", config["figure_format"],
         "--figures", "1,2,3"],
        skip=args.skip_s5,
    )
    stage_results["S5: Figure Generation"] = (ok, t)

    # S6: LaTeX export
    ok, t = run_stage(
        "S6: LaTeX Export",
        [sys.executable, "export_latex_tables.py",
         "--results_dir", config["results_dir"],
         "--output_dir", config["tables_dir"],
         "--tables", "1,2,3"],
        skip=args.skip_s6,
    )
    stage_results["S6: LaTeX Export"] = (ok, t)

    # S7: LLM benchmark (optional)
    if args.run_llm:
        ok, t = run_stage(
            "S7: LLM Benchmark",
            [sys.executable, "eval_llm_rigorous.py"],
            skip=False,
        )
        stage_results["S7: LLM Benchmark"] = (ok, t)

    # Final summary
    print_final_summary(config, stage_results)

    # Exit code
    all_passed = all(s for s, _ in stage_results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()