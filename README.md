# kt-research

Knowledge Tracing research: comparing probabilistic, factor-based, neural recurrent, and attention-based KT models on the ASSISTments 2017 dataset.

## Current Status

**Implementation phase** -- the codebase, configuration, tests, and documentation are implemented but no experiments have been executed on real data. All tests use synthetic toy data.

## Research Question

> Under the same ASSISTments 2017 preprocessing and evaluation protocol, which model best predicts the correctness of the next student interaction: BKT, LFA, DBKT, DKT, or Attention-based KT?

Secondary questions:
- How do neural models (DKT, Attention) compare against probabilistic models (BKT, DBKT)?
- Does adding dynamic forgetting (DBKT) improve over static BKT?
- Does adding auxiliary features improve any model family?

## Repository Structure

```
kt-research/
├── configs/                    # Hydra YAML configuration
│   ├── data/                   # Dataset configs
│   ├── experiment/             # Experiment configs (compose data+model+trainer)
│   ├── model/                  # Model hyperparameter configs
│   └── trainer/                # Trainer configs
├── data/
│   ├── raw/                    # Raw ASSISTments 2017 dataset (immutable)
│   ├── external/
│   ├── interim/
│   ├── processed/              # Output from preprocessing pipeline
│   └── splits/
├── docs/
│   └── agent/                  # Implementation guidelines and specs
├── notebooks/                  # Jupyter notebooks for EDA and analysis
├── outputs/                    # Generated reports, figures, metrics, artifacts
├── scripts/                    # Runnable entrypoints (thin, logic in src/)
├── src/                        # Reusable source code
│   ├── data/                   # Loading, preprocessing, splitting, sequences
│   │   ├── sequence_builder.py # Student sequence construction and padding
│   │   ├── sequence_dataset.py # PyTorch Dataset for sequence models
│   │   └── xes3g5m.py          # XES3G5M dataset loader & metadata parsing
│   ├── evaluation/             # Metrics, bootstrap, evaluators
│   ├── features/               # Feature engineering, encoders
│   ├── models/                 # BKT, LFA, DBKT, DKT, Attention, SFN-KT, baselines
│   │   ├── dbkt.py             # Dynamic Bayesian Knowledge Tracing
│   │   ├── dkt.py              # Deep Knowledge Tracing (LSTM/GRU)
│   │   ├── attention_kt.py     # Attention-based Contextual KT
│   │   ├── sfn_kt.py           # SFN-KT: Rasch, Backbone, SCDT, Q-Former, Adapter
│   │   ├── llm_reasoner.py     # Foundation LLM Reasoner & Counterfactual prompts
│   │   └── factory.py          # Model factory from config
│   ├── training/               # Training loop, early stopping, sequence trainer
│   │   ├── sequence_trainer.py # Generic trainer for DKT, Attention
│   │   └── sfn_kt_trainer.py   # 3-stage decoupled trainer for SFN-KT
│   └── utils/                  # Seed, logging, I/O helpers
├── tests/                      # Unit tests with synthetic & mock data
├── .env.example                # W&B configuration template
├── .gitignore
├── Makefile                    # Common commands
├── pyproject.toml              # Project metadata and tool config
└── requirements.txt            # Python dependencies
```

## Environment Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv

### Installation

```bash
# Create and activate conda environment
conda create -n kt-research python=3.11
conda activate kt-research

# Install dependencies
pip install -r requirements.txt
```

### GPU & Device Support

The default trainer config uses `device: cpu`. You can specify compute device via CLI overrides (`mps` for Apple Silicon, `cuda` for NVIDIA GPU, or `auto` for automatic detection):

```bash
# Run LFA-core training on Apple Silicon MPS
python scripts/train.py experiment=lfa_core_assistments trainer.device=mps

# Run LFA-core training on NVIDIA CUDA
python scripts/train.py experiment=lfa_core_assistments trainer.device=cuda

# Auto-detect available GPU/MPS device
python scripts/train.py experiment=lfa_core_assistments trainer.device=auto
```

Training progress is displayed in real-time using `tqdm` progress bars (per-skill progress for BKT and per-epoch progress for LFA models).

## Dataset

### Location

The raw ASSISTments 2017 dataset should be placed at:

```
data/raw/ASSISTments Data Mining Competition Dataset/Released Full Dataset/anonymized_full_release_competition_dataset.csv
```

### Key Columns

Required: `studentId`, `skill`, `problemId`, `correct`, `startTime`, `action_num`  
Recommended: `timeTaken`, `hintCount`, `attemptCount`, `scaffold`, `hintTotal`

## Configuration

This project uses [Hydra](https://hydra.cc) with OmegaConf for config-driven experiments.

### Config Structure

```
configs/
├── config.yaml                         # Main Hydra entrypoint (defaults to bkt_assistments)
├── compare.yaml                        # Model comparison experiment names
├── data/assistments2017.yaml           # Dataset path, columns, filtering thresholds
├── model/bkt.yaml                      # BKT parameters and optimization config
├── model/lfa.yaml                      # LFA architecture and training config
├── model/dbkt.yaml                     # DBKT dynamic HMM config
├── model/dkt.yaml                      # DKT LSTM/GRU config
├── model/attention_kt.yaml             # Attention-based KT config
├── trainer/base.yaml                   # Default trainer settings (device: cpu)
├── trainer/debug.yaml                  # Debug overrides (fewer epochs)
└── experiment/
    ├── bkt_assistments.yaml            # BKT experiment
    ├── lfa_core_assistments.yaml       # LFA-core experiment
    ├── lfa_aux_assistments.yaml        # LFA-aux experiment
    ├── dbkt_core_assistments.yaml      # DBKT-core experiment
    ├── dkt_core_assistments.yaml       # DKT-core experiment
    ├── attention_core_assistments.yaml # Attention-core experiment
    └── baseline_skill_average.yaml     # Skill average baseline
```

### Override from CLI

```bash
python scripts/train.py experiment=lfa_core_assistments trainer.batch_size=256 trainer.device=mps
python scripts/train.py experiment=bkt_assistments data.skill_handling.multi_skill_strategy=first
```

## How to Run

### Run EDA

```bash
python scripts/eda_assistments.py
```

Output: `outputs/reports/eda_assistments.json`, `outputs/figures/eda/*.png`

### Run Preprocessing

```bash
python scripts/preprocess_assistments.py
```

Output: `data/processed/assistments2017/v1/` containing `interactions.parquet`, `metadata.json`, `split_info.json`, `skill_map.json`, `student_map.json`

### Run Baselines

```bash
python scripts/baseline_skill_average.py
```

Output: `outputs/metrics/baseline_metrics.json`

### Train Models

```bash
python scripts/train.py experiment=bkt_assistments trainer.device=actual_device

python scripts/train.py experiment=lfa_core_assistments trainer.device=actual_device

python scripts/train.py experiment=lfa_aux_assistments trainer.device=actual_device

# DBKT (Dynamic Bayesian Knowledge Tracing)
python scripts/train.py experiment=dbkt_core_assistments trainer.device=actual_device

# DKT (Deep Knowledge Tracing via LSTM)
python scripts/train.py experiment=dkt_core_assistments trainer.device=actual_device

# Attention-based Contextual KT
python scripts/train.py experiment=attention_core_assistments trainer.device=actual_device
```

Device override examples:

```bash
python scripts/train.py experiment=dkt_core_assistments trainer.device=cuda
python scripts/train.py experiment=attention_core_assistments trainer.device=mps
python scripts/train.py experiment=dbkt_core_assistments trainer.device=auto
```

Output: `outputs/artifacts/{experiment_name}/` with model checkpoints, encoders, and training metadata.

### Evaluate Models

```bash
python scripts/evaluate.py experiment=bkt_assistments
python scripts/evaluate.py experiment=lfa_core_assistments
python scripts/evaluate.py experiment=dbkt_core_assistments
python scripts/evaluate.py experiment=dkt_core_assistments
python scripts/evaluate.py experiment=attention_core_assistments
```

### Compare Models

```bash
python scripts/compare_models.py
```

Output: `outputs/comparison/model_comparison.json`, `outputs/comparison/model_comparison.md`. Includes all trained models: baselines, BKT, LFA-core, LFA-aux, DBKT-core, DKT-core, Attention-core.

### SFN-KT on XES3G5M Dataset

SFN-KT (Selective Foundation-Neural Knowledge Tracing) is trained and evaluated on the large-scale **XES3G5M** dataset using a 3-stage decoupled pipeline.

#### 1. Preprocess & Validate XES3G5M Metadata

Verify and cache split integrity (0 student UID overlap between train/val folds and held-out test), question metadata, KC route maps, and RoBERTa embeddings:

```bash
conda run -n kt-research-env python scripts/preprocess_xes3g5m.py
```

Output: `data/processed/xes3g5m/dataset_summary.json`

#### 2. Train SFN-KT (3-Stage Decoupled Pipeline)

SFN-KT utilizes a decoupled 3-stage training pipeline designed for maximum efficiency:
- **Stage 1**: Train Fast Causal Backbone and freeze base sensor $w_{\text{base}}$.
- **Stage 2**: Scan anomalies via SCDT, generate pedagogical Chain-of-Thought (CoT) rationales via LLM (Ollama / Cloud APIs / Local HF), encode text into continuous representations via a frozen **Lightweight Text Encoder** (`sentence-transformers/all-MiniLM-L6-v2`), align Cognitive Q-Former, and store compressed memory tensors $\mathbf{Z}_t^{\text{cog}} \in \mathbb{R}^{M \times d}$ into an offline HDF5 cache.
- **Stage 3**: Train Multi-Anchor Causal Adapter with Residual Highway and joint Soft-ECE calibration.

##### Prerequisites & Configuration Setup Before Running

Before executing training with an LLM backend, prepare the corresponding provider:

| Provider / Backend | Prerequisites & Setup Steps | Default Endpoint | Example Command Flag |
| :--- | :--- | :--- | :--- |
| **`ollama`** (Local Free) | 1. Install Ollama: `brew install ollama`<br>2. Start daemon: `ollama serve`<br>3. Pull model: `ollama pull qwen2.5:7b` (or `deepseek-r1:8b`) | `http://localhost:11434` | `--llm-backend ollama --llm qwen2.5:7b` |
| **`google` / `gemini`** | 1. Obtain free key from [Google AI Studio](https://aistudio.google.com/)<br>2. Set: `export GEMINI_API_KEY="AIzaSy..."` | Google AI Studio REST v1beta | `--llm-backend google --llm gemini-1.5-flash` |
| **`deepseek` / `openai`** | 1. Obtain API key from DeepSeek / OpenAI / OpenRouter<br>2. Set: `export DEEPSEEK_API_KEY="..."` or `OPENAI_API_KEY` | `https://api.deepseek.com/v1` or OpenAI | `--llm-backend deepseek --llm deepseek-reasoner` |
| **`mock`** (Offline CI/Dev) | No setup required. Uses XES3G5M RoBERTa embeddings or deterministic pseudo-tokens. | Local CPU/MPS/CUDA | `--llm-backend mock` |
| **`huggingface`** | Requires local NVIDIA GPU with $\ge 24\text{GB}$ VRAM for models $\ge 7\text{B}$. | Local GPU | `--llm-backend huggingface --llm Qwen/Qwen2.5-Math-7B` |

> **Lightweight Text Encoder**: When using API or Ollama backends, generated text rationales are converted into token-level representations using `sentence-transformers/all-MiniLM-L6-v2` ($d_{\text{enc}} = 384$). Weights are permanently frozen, avoiding catastrophic forgetting and heavy VRAM consumption.
>
> **Two-Tier Text Caching**: Specify `--text-cache-path` to save raw generated text rationales to a JSON file. Subsequent runs reuse cached text instantly, incurring zero additional API calls or latency.

##### CLI Configuration Reference

| CLI Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--experiment-name` | `str` | `sfn_kt_xes3g5m_default` | Unique experiment identifier (alias `--exp-name`). Isolates checkpoints (`outputs/artifacts/<exp_name>/`), HDF5 cache, text cache, and evaluation metrics (`outputs/metrics/<exp_name>/`). |
| `--stage` | `str` | `all` | Training stage to execute: `1`, `2`, `3`, or `all`. |
| `--llm-backend` | `str` | `mock` | LLM backend: `ollama`, `google`, `gemini`, `deepseek`, `openai`, `openrouter`, `huggingface`, `mock`. |
| `--llm` | `str` | `Qwen/Qwen2.5-Math-7B` | LLM model name (e.g. `qwen2.5:7b`, `gemini-1.5-flash`, `deepseek-reasoner`). |
| `--encoder-name` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Pre-trained text encoder used to encode CoT rationales. |
| `--llm-d-llm` | `int` | `384` (for MiniLM) | Latent dimension fed into Cognitive Q-Former (MiniLM automatically projects if dimension differs). |
| `--api-key` | `str` | `None` | API key (or read from environment: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`). |
| `--base-url` | `str` | `None` | Base URL override for Ollama (e.g. `http://localhost:11434`) or custom OpenAI proxies. |
| `--text-cache-path`| `str` | Auto: `outputs/artifacts/<exp_name>/<backend>_text_cache.json` | Path to JSON file caching generated text rationales. If omitted, automatically isolated inside the experiment directory. |
| `--device` | `str` | `auto` | Compute device: `auto`, `mps` (Apple Silicon), `cuda`, or `cpu`. |
| `--batch-size` | `int` | `64` | Training batch size. |
| `--epochs` | `int` | `None` | Override epoch count for all stages. |
| `--max-triggers` | `int` | `None` | Maximum number of SCDT anomaly events to query LLM for (useful for strict API budgets). |
| `--smoke-test` | `flag` | `False` | Run rapid 1-epoch sanity check with small sample size. |

##### Example Commands

```bash
# 1. Run full end-to-end pipeline with custom experiment name and Ollama:
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --experiment-name sfn_kt_qwen2_5_7b \
    --stage all \
    --llm-backend ollama \
    --llm qwen2.5:7b \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --llm-d-llm 384 \
    --device auto

# 2. Run with Google AI Studio Gemini API (Ultra-fast cloud reasoning):
export GEMINI_API_KEY="your-gemini-api-key"
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --stage all \
    --llm-backend google \
    --llm gemini-1.5-flash \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --llm-d-llm 384 \
    --api-key $GEMINI_API_KEY \
    --text-cache-path outputs/artifacts/sfn_kt_xes3g5m_default/gemini_text_cache.json \
    --device auto

# 3. Run with DeepSeek-R1 via OpenAI-compatible REST endpoint:
export DEEPSEEK_API_KEY="your-deepseek-api-key"
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --stage all \
    --llm-backend deepseek \
    --llm deepseek-reasoner \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --llm-d-llm 384 \
    --api-key $DEEPSEEK_API_KEY \
    --text-cache-path outputs/artifacts/sfn_kt_xes3g5m_default/deepseek_text_cache.json

# 4. Run Stage 2 independently with strict trigger budget (e.g. top 500 anomalies):
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --stage 2 \
    --llm-backend ollama \
    --llm qwen2.5:7b \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --llm-d-llm 384 \
    --max-triggers 500

# 5. Rapid smoke test verifying the pipeline without external dependencies:
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --smoke-test \
    --stage all \
    --llm-backend ollama \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --llm-d-llm 384
```

Artifacts generated:
- Stage 1: `outputs/artifacts/{exp}/fast_backbone_best.pt`
- Stage 2: `outputs/artifacts/{exp}/cognitive_qformer_cache.h5` (offline compressed HDF5 tensor cache)
- Stage 2 text cache: `outputs/artifacts/{exp}/llm_text_cache.json` (raw generated CoT rationales)
- Stage 3: `outputs/artifacts/{exp}/sfn_kt_best.pt`

#### 3. Standalone SFN-KT Evaluation

Compare Base Fast Backbone against Calibrated SFN-KT on the held-out test set:

```bash
conda run -n kt-research-env python scripts/evaluate_sfn_kt.py --smoke-test
# Or full test set:
conda run -n kt-research-env python scripts/evaluate_sfn_kt.py --checkpoint outputs/artifacts/sfn_kt_xes3g5m_default/sfn_kt_best.pt
```

Outputs: `outputs/metrics/{exp}/test_eval_predictions.parquet`, `outputs/metrics/{exp}/evaluation_report.json`

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_bkt.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Expected Outputs

After a complete experiment run, the following outputs are expected:

### Reports
- `outputs/reports/eda_assistments.json`
- `outputs/metrics/baseline_metrics.json`
- `outputs/comparison/model_comparison.json`
- `outputs/comparison/model_comparison.md`

### Processed Data
- `data/processed/assistments2017/v1/interactions.parquet`
- `data/processed/assistments2017/v1/metadata.json`
- `data/processed/assistments2017/v1/split_info.json`

### Model Artifacts
- `outputs/artifacts/bkt_assistments2017_v1/bkt_params.csv`
- `outputs/artifacts/lfa_core_assistments2017_v1/best_model.pt`
- `outputs/artifacts/lfa_core_assistments2017_v1/encoder_student.json`
- `outputs/artifacts/lfa_core_assistments2017_v1/encoder_skill.json`
- `outputs/artifacts/dbkt_core_assistments2017_v1/best_model.pt`
- `outputs/artifacts/dbkt_core_assistments2017_v1/dbkt_params.csv`
- `outputs/artifacts/dkt_core_assistments2017_v1/best_model.pt`
- `outputs/artifacts/dkt_core_assistments2017_v1/encoder_skill.json`
- `outputs/artifacts/attention_core_assistments2017_v1/best_model.pt`
- `outputs/artifacts/attention_core_assistments2017_v1/encoder_skill.json`

## Models

### BKT (Bayesian Knowledge Tracing)

Per-skill model with 4 parameters: `p_init` (initial knowledge), `p_learn` (learning rate), `p_slip` (slip probability), `p_guess` (guess probability). Trained with L-BFGS-B with multiple random restarts. Falls back to global parameters for rare skills.

### LFA-core (Logistic Factor Analysis)

Logistic regression model using:
- Student identity (embedding + bias)
- Skill identity (embedding + bias)
- Student-skill interaction (dot product of embeddings)
- History features: success before, failure before, attempts before

### LFA-aux

Extends LFA-core with auxiliary features: `timeTaken`, `hintCount`, `attemptCount`, `scaffold`, `hintTotal`. Features are cleaned, imputed, and standardized (fitted on train only).

### DBKT (Dynamic Bayesian Knowledge Tracing)

Dynamic HMM extension of BKT with covariate-dependent transition probabilities:

- Per-skill parameters: `p_init`, `p_slip`, `p_guess` (same as BKT)
- Dynamic learning rate: `learn_t = sigmoid(alpha_learn + w_learn · f_t)`
- Dynamic forgetting rate: `forget_t = sigmoid(alpha_forget + w_forget · f_t)`
- Transition features: `log(1 + attempts_before)`, `log(1 + failure_before)`
- PyTorch-based differentiable forward algorithm in log-space
- Adam optimization (alternative to L-BFGS-B)
- Slip+guess constraint penalty

Reference: Dynamic HMM variant documented in `docs/research/kt_models_survey.md` (fallback `dbkt_dynamic_hmm_fallback`).

### DKT (Deep Knowledge Tracing)

Recurrent neural network (LSTM/GRU) mapping sequential student interactions to knowledge state:

- **Input encoding**: Learned skill + correctness embeddings (default) or one-hot vectors
- **Architecture**: Embedding -> LSTM/GRU -> Linear head over skills
- **Loss**: BCE with logits on target skill only (`loss_scope: target_only`)
- **Inference**: Online hidden state maintained per student
- Supports `max_seq_len` truncation and padding masks

Reference: Piech et al., "Deep Knowledge Tracing", NeurIPS 2015.

### Attention-based Contextual KT

SAKT/AKT hybrid with causal multi-head attention over past interactions:

- **Query**: Target skill embedding + position embedding
- **Keys/Values**: Historical interaction embeddings (skill + correctness + position)
- **Causal masking**: Strict lower-triangular mask prevents future leakage
- **Monotonic decay**: Learnable recency decay parameter (`softplus`-constrained)
- **Prediction head**: `LayerNorm(context + query) -> Linear -> GELU -> Linear -> sigmoid`
- Online history buffer with `max_seq_len` truncation

Reference: Hybrid of SAKT (Pandey & Karypis, 2019) and AKT (Ghosh et al., 2020).

### SFN-KT (Selective Foundation-Neural Knowledge Tracing)

Selective Foundation-Neural KT architecture designed for high throughput and deep pedagogical reasoning on multi-modal benchmark datasets (XES3G5M):

- **Module 1: Rasch-Parameterized Input Embedding**:
  - Incorporates question difficulty $\beta_q$, discrimination $\alpha_q$, KC embedding $\mathbf{e}_{c_t}$, and response correctness $\mathbf{r}_t$.
  - Strictly leak-free: interaction embedding $\mathbf{x}_t$ uses response $r_t$, while target query $\mathbf{q}_{t+1}$ contains only question identity and item characteristics without target response.
- **Module 2: Fast Sequential Backbone**:
  - 4-layer Causal Transformer Encoder ($d=128$, 8 heads) processing interaction sequences at $\mathcal{O}(T)$ inference step latency.
  - Strict causal masking: $M_{ij} = -\infty$ for $i < j$, guaranteeing zero future leakage.
- **Module 3: Selective Cognitive Dilemma Trigger (SCDT)**:
  - Anomaly regulator monitoring:
    1. Epistemic uncertainty: $\mathcal{H}(\hat{y}_t) = -[\hat{y}_t \log \hat{y}_t + (1-\hat{y}_t) \log (1-\hat{y}_t)]$.
    2. Consecutive failure momentum: $F_t = \sum_{k=0}^{\min(t, K)-1} \lambda^k \cdot \mathbb{I}(r_{t-k} = 0)$.
    3. Slip/Guess cognitive conflict score: $C_t = |\hat{y}_t - (1 - \sigma(\beta_{q_t}))|$.
  - Triggers foundation reasoning only on anomalous interactions ($\sim 5\text{--}10\%$ of steps), keeping $90\text{--}95\%$ on the fast neural backbone.
- **Module 4: Foundation Reasoning Core & Cognitive Q-Former**:
  - Track B: Sparse Context Counterfactual Error Hypothesis prompt using strictly past interactions ($1 \dots t-1$), question stem, options, and KC taxonomy.
  - Cognitive Q-Former with $M=4$ learnable queries compresses variable-length LLM hidden states ($\mathbb{R}^{L \times d_{\text{LLM}}}$) into compact cognitive representations $\mathbf{Z}_t^{\text{cog}} \in \mathbb{R}^{M \times d}$.
  - Offline HDF5 caching decouples expensive LLM inference from adapter tuning.
- **Module 5: Multi-Anchor Causal Adapter & Residual Highway**:
  - Dual causal/trigger masked cross-attention incorporates $\mathbf{Z}_k^{\text{cog}}$ only at triggered past steps ($k \le t, \tau_k = 1$).
  - Residual Highway with learnable scalar gating parameter $\gamma$ (initialized to $0.0$) ensures SFN-KT performance is strictly bounded below by the fast neural backbone.
- **Loss Functions**:
  - Cognitive-Weighted BCE Loss: $\mathcal{L}_{\text{BCE}}^{\text{cog}} = - \frac{1}{N} \sum_t w_t [y_t \log \hat{y}_t + (1 - y_t) \log (1 - \hat{y}_t)]$ where $w_t = 1 + \alpha \cdot \tau_t$ ($\alpha = 0.75$).
  - Soft-ECE Loss: Differentiable Expected Calibration Error using 10 soft Sigmoid bins to align predicted probabilities with empirical mastery.

Reference: `docs/sfn-kt/model_arch.md`.

### SFN-KT & XES3G5M (pyKT Benchmark Standard Pipeline)

This project strictly implements and follows the **pyKT benchmark standards** (NeurIPS 2022) for the XES3G5M dataset:

#### 1. pyKT Benchmark Protocol
- **Dataset Partitioning (5-Fold CV)**:
  - `train_valid_sequences.csv`: Folds 0, 1, 2, 3 are used for model training (14,453 students), and Fold 4 is used for validation.
  - Zero student leakage: strictly enforced across all 5 folds.
- **Official Test Split (`test_question_window_sequences.csv`)**:
  - 3,613 withheld test students split into sliding windows of length 200.
  - Exactly zero student UID overlap between training folds and test windows.
  - Contains pyKT-specific tracking columns: `qidxs` (question interaction index), `rest` (remaining KCs in question), and `orirow` (original student sequence index).
- **Question-Level Evaluation via Late Fusion**:
  - In educational benchmarks where questions map to multiple Knowledge Components (KCs), models predict at the KC sequence level where `selectmasks == 1`.
  - Predictions are aggregated across KCs sharing the same `(uid, qidx)` to form question-level predictions:
    - **Late-Mean** (pyKT default): $\hat{y}_q = \frac{1}{|K_q|} \sum_{k \in K_q} \hat{y}_k$
    - **Late-Vote**: Majority voting over binary KC decisions.
    - **Late-All**: Product of probabilities ($\hat{y}_q = \prod_{k \in K_q} \hat{y}_k$).
  - Evaluated against binary question ground-truth $y_q$ using AUC, Accuracy, LogLoss, and Brier score.

#### 2. Workflow & Execution Guide

> **Note**: All terminal commands must be executed using the Conda environment `kt-research-env`.

##### A. Preprocessing & pyKT Dataset Profiling
Inspects metadata, sequence lengths, verifies 0 student UID leakage, and validates pyKT schemas:
```bash
conda run -n kt-research-env python scripts/preprocess_xes3g5m.py
```
Outputs are saved to `data/processed/xes3g5m/dataset_summary.json` and `data/processed/xes3g5m/pykt_benchmark_info.json`.

##### B. Verification & Compliance Tests
Run the comprehensive test suite verifying late fusion, dataset loaders, and SFN-KT training:
```bash
conda run -n kt-research-env pytest tests/test_pykt_compliance.py tests/test_xes3g5m.py -v
```

##### C. Minimal End-to-End Smoke Test
Verify the complete 3-stage training and late fusion evaluation pipeline locally without requiring a GPU or LLM API:
```bash
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --smoke-test \
    --stage all \
    --llm-backend mock \
    --device cpu
```

##### D. Full 3-Stage Training (pyKT Benchmark)
Run the 3 decoupled stages sequentially or all together:
```bash
# Option 1: Run all 3 stages in one command on GPU
conda run -n kt-research-env python scripts/train_sfn_kt.py \
    --stage all \
    --test-mode question_window \
    --fusion-type mean \
    --llm Qwen/Qwen2.5-Math-7B \
    --llm-backend hf \
    --device cuda

# Option 2: Run stage-by-stage
# Stage 1: Fast Backbone pretraining (invariant sensor freezing)
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 1 --device cuda --epochs 10

# Stage 2: SCDT cognitive anomaly scan, Q-Former alignment & HDF5 tensor caching
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 2 --device cuda --llm Qwen/Qwen2.5-Math-7B --llm-backend hf

# Stage 3: Multi-Anchor Causal Adapter training & Soft-ECE joint calibration
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 3 --device cuda --epochs 10 --test-mode question_window --fusion-type mean
```

##### E. Standalone pyKT Evaluation
Evaluate a saved SFN-KT checkpoint on the official pyKT question-level window test set with late fusion:
```bash
conda run -n kt-research-env python scripts/evaluate_sfn_kt.py \
    --checkpoint-dir outputs/artifacts/sfn_kt/sfn_kt_xes3g5m_default \
    --test-mode question_window \
    --fusion-type mean \
    --device cuda
```
The evaluator reports both KC-level metrics and Question-level Late Fusion metrics (Late-Mean, Late-Vote, Late-All), and saves prediction records to `test_eval_predictions.parquet`.

### Baselines

- **Majority baseline**: predicts global train correct rate
- **Skill average baseline**: predicts per-skill correct rate (fallback to global)
- **Student average baseline**: predicts per-student correct rate (fallback to global)

## Evaluation Protocols

### Primary: Online Next-Step Prediction

For each interaction `t`, use only interactions before `t` to predict. After prediction, observe the true `correct` and update model state.

### Secondary: Held-Out Block Prediction

Hold out the final block of each student's interactions as test. Predict without updating using test responses.

## Metrics

- AUC (primary), LogLoss, Brier score, Accuracy
- Skill-level metrics (macro/weighted AUC)
- Bucket metrics by skill frequency and student history length
- Expected Calibration Error (ECE)
- Bootstrap confidence intervals
- Paired bootstrap comparison

## Leakage Prevention

Critical design rules enforced in tests:

- `success_before` excludes current `correct` (computed as `cumsum() - correct`)
- `attempts_before` excludes current interaction
- Encoders fitted on train split only
- Feature scalers fitted on train split only
- Split preserves temporal order per student
- DKT/Attention target shift: input `z_0..z_{t-1}` predicts target `z_t`; never `z_t` in input for predicting `z_t`
- Attention causal mask: strict lower-triangular, `-inf` for `j >= t`
- Sequence padding masks isolate valid positions from loss computation
- Loss computed only on train-split positions for neural sequence models

## Troubleshooting

### Processed data not found

All training scripts check for processed data first. Run preprocessing first:

```bash
python scripts/preprocess_assistments.py
```

### Hydra config errors

Ensure you are running from the project root. The `@hydra.main` decorator uses `config_path="../configs"` relative to each script.

### Test failures

If tests fail, ensure dependencies are installed:

```bash
pip install -r requirements.txt
python -m pytest tests/
```

## Decision Log

| Decision | Value | Rationale |
|----------|-------|-----------|
| Multi-skill strategy | `first` | Simple, aligns with BKT per-skill assumption |
| Min student interactions | 10 | Filter noisy short sequences |
| Min skill interactions | 50 | Ensure sufficient data for per-skill BKT training |
| Min sequence length | 3 | Minimum for meaningful temporal split |
| Split protocol | `temporal_per_student` | Preserves temporal order, no future leakage |
| Sorting order | `startTime`, then `action_num` | `action_num` as tie-breaker |
| BKT optimizer | L-BFGS-B | Standard for BKT, supports bounds |
| BKT fallback | Global parameters | Handles rare skills gracefully |
| DBKT formulation | `dbkt_dynamic_hmm_fallback` | No single canonical DBKT paper; dynamic HMM is principled and compatible with row-level data |
| DBKT transition features | `attempts_before`, `failure_before` | Leak-free per-skill history, no current-row data |
| DBKT optimizer | Adam (PyTorch) | Differentiable forward algorithm, flexible constraint penalties |
| DKT input mode | `embedding` (default) | More scalable than one-hot for large skill vocabularies |
| DKT loss scope | `target_only` | Aligns with observed student paths; avoids all-skill noise |
| DKT architecture | 1-layer LSTM, hidden 128 | Balance of expressiveness and overfitting prevention |
| Attention variant | SAKT/AKT hybrid | Lightweight, maintains causal masking + recency decay |
| Attention decay type | `position` | Robust against missing timestamps |
| Attention architecture | 1-layer, 2 heads, dim 64 | Efficient CPU debugging, sufficient for ~100 skills |
| Auxiliary feature policy | Past-history aggregates only | No current-row response metadata in core comparisons |
| Sequence max length | 200 | Covers >95% of ASSISTments student lengths |
| Sequence truncation | Keep most recent | Preserves recency for prediction |
| Rare skill handling (neural) | UNK embedding | Out-of-vocabulary skills map to trained UNK token |
| Rare skill handling (DBKT) | Global parameter fallback | Same strategy as BKT |
| Validation protocol | `online_next_step` | Matches primary evaluation protocol |
| Device default | `cpu` | Works everywhere, override via CLI for GPU/MPS |

## Code Quality

```bash
# Lint
make lint

# Format
make format

# Test
make test
```