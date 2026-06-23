# Beyond Meta-Prior: Official Implementation 🚀

> **Anonymous Repository for Double-Blind Review**
> 
> Official implementation for the paper: *"Beyond Meta-Prior: Cognitively-Inspired Contrastive Adaptive Fine-Tuning and Counterfactual Probing for Few-Shot Chinese Linguistic Quantifier Calibrated Resolution under Extreme Data Scarcity"*

This repository contains the complete experimental pipeline, including Bayesian PCFG meta-learning, the **CAFT (Contrastive Adaptive Fine-Tuning)** framework, **SIPAR (Syntax-Invariant Pairwise Alignment Regularizer)**, and the rigorous Out-of-Distribution (OOD) testing suite evaluated on the Universal Dependencies (UD) Chinese treebank.

---

## 📂 Repository Structure

Our codebase is highly modular, reflecting the cognitive phases detailed in the manuscript:

- **`meta_pretrain_full_strong.py`**: Implementation of Phase 1 & 2. Distills Bayesian Probabilistic Context-Free Grammar (PCFG) structural rules into the Transformer backbone via first-order meta-learning (Reptile).
- **`meta_pretrain_no_noise.py`**: Baseline meta-training script without data perturbation/noise mechanisms.
- **`run_rigorous_aligned_ablationTJ.py`**: **[Core Script]** Implementation of Phase 3 (Naturalistic Adaptation). Contains the full CAFT framework, Soft Prior Interpolation, SIPAR contrastive loss, and Layer-wise Learning Rate Decay (LLRD).
- **`ud_kshot_ablation.py`**: Utility and data processing suite for handling Universal Dependencies CoNLL-U parsing, K-shot sampling, and syntactic distance computation ($(d=1$) vs. $(d>1$)).
- **`representation_analyzer.py`**: Evaluation module computing Effective Rank ($E_{\text{rank}}$) via SVD and Cosine Similarity to monitor representation collapse and manifold isotropy.
- **`run_llm_benchmark.py`**: Generation-free, token-level probability evaluation pipeline for native Large Language Models (LLMs).

---

## 🛠️ Environment Setup

**Hardware Specifications:** All experiments, including the inference of up to 9B-parameter LLMs, were conducted on a single **NVIDIA A40 GPU (48GB VRAM)**. Ensure your environment has sufficient VRAM when reproducing the large language model benchmarks.

Ensure you are using Python 3.8+ and install the necessary dependencies:

```bash
pip install torch numpy scikit-learn conllu transformers
```

Dataset Download:
Please download the Universal Dependencies Chinese GSD Simplified Treebank (zh_gsdsimp-ud-train.conllu, zh_gsdsimp-ud-test.conllu) from the official repository and place them in the ./UD_Chinese-GSDSimp-master/ directory.

⚙ Quick Start: Running the CAFT Pipeline
Step 1: Meta-Prior Distillation (Phase 1 & 2)

To generate the initialization meta-prior weights from the synthetic PCFG tasks, run:
```bash
python meta_pretrain_full_strong.py
```
This will generate a meta_pretrained_prior.pth file containing the structural priors.

(Optional) Step 1.5: Meta-Pretraining Ablation (Control Group)

To rigorously verify that the performance gains originate from the injection of hierarchical syntactic structures rather than merely the meta-learning optimization process itself, we provide an unstructured ablation baseline. This script performs meta-training using flat, structure-free nominal sequences:
```bash
python meta_pretrain_no_noise.py
```
This generates meta_no_noise.pth, establishing a purely lexical control group devoid of structural priors. You can load these weights in Phase 3 to reproduce the ablation results presented in the manuscript.

Step 2: Few-Shot Adaptation & OOD Evaluation (Phase 3)

To execute the extreme data-scarcity adaptation (K=10) and test the model against the syntactic complexity shift ($d>1$), execute the rigorous ablation script:
```bash
python run_rigorous_aligned_ablationTJ.py
```
This script automatically injects SIPAR, performs Soft Prior Interpolation, and calls representation_analyzer.py internally to track the Effective Rank of the latent manifold.

🤖 Reproducing the LLM Benchmark (Local Inference)
To comprehensively evaluate state-of-the-art Large Language Models (LLMs) via our strict generation-free ICL protocol natively without API costs, follow these guidelines:

Step 1: Upgrade Hugging Face CLI

Ensure you have the latest huggingface_hub installed to support large model fetching:
```bash
pip install --upgrade huggingface_hub
```
Step 2: Download Model Weights Locally

Use the hf download command to fetch the models from Hugging Face. For example, to download Qwen2.5-0.5B-Instruct (approx. 1GB), run in your terminal:
```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```
(Repeat this step for other models such as THUDM/glm-4-9b-chat or internlm/internlm2_5-7b-chat by changing the repo ID and target directory accordingly).

Step 3: Install LLM Tokenizer Dependencies

Ensure the necessary tokenization libraries required by specific LLMs are installed:
```bash
pip install tiktoken sentencepiece
```
Step 4: Configure the Evaluation Script

Open run_llm_benchmark.py and modify the script to point to your local directories.

Comment out the HF mirror (since you are reading from local storage):
```bash
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Comment out or remove
```
Update the model paths dictionary to reflect your local absolute paths:
```bash
MODELS_TO_COMPARE: Dict[str, str] = {
    "Qwen2.5-0.5B-Instruct": "./models/Qwen2.5-0.5B-Instruct",  # Point to local path
    # "GLM-4-9B-Chat": "./models/glm-4-9b-chat",
    # Add other models as downloaded...
}
```
Step 5: Run the Benchmark

Execute the Python script to start the strict generation-free, token-level probability evaluation across the stratified splits:
```bash
python run_llm_benchmark.py
```

📊 Evaluation Metrics Computed
- OOD Macro-F1 & Accuracy: Performance on structurally complex ($d>1$) instances.
- Expected Calibration Error (ECE): Confidence vs. Accuracy calibration via dynamic temperature scaling.
- Effective Rank ($E_{\text{rank}}$): Measurement of representation isotropy to prove the prevention of dimensional collapse.

Note: Due to the double-blind review policy, author identities, affiliations, and institutional acknowledgments have been removed from this repository. Code will be open-sourced under the MIT License upon publication.

