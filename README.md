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

> [!NOTE]
> Always execute commands using the Conda environment:
> `conda run -n kt-research-env python <script>`

#### 1. Preprocess & Validate XES3G5M Metadata

Verify and cache split integrity (0 student UID overlap between train/val folds and held-out test), question metadata, KC route maps, and RoBERTa embeddings:

```bash
conda run -n kt-research-env python scripts/preprocess_xes3g5m.py
```

Output: `data/processed/xes3g5m/dataset_summary.json`

#### 2. Train SFN-KT (3-Stage Decoupled Pipeline)

You can train all stages end-to-end or run individual stages (`1`, `2`, `3`, or `all`). The LLM model name and backend can be passed directly from the terminal CLI:

```bash
# Run all stages end-to-end with Mock/RoBERTa reasoner (smoke test)
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage all --smoke-test

# Train Stage 1: Fast Causal Backbone
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 1 --epochs 20 --batch-size 64

# Train Stage 2: Selective Cognitive Dilemma Trigger (SCDT) scan & Cognitive Q-Former caching
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 2 \
  --llm Qwen/Qwen2.5-Math-7B \
  --llm-backend huggingface \
  --llm-d-llm 3584

# Alternatively with DeepSeek-R1 Distill or OpenAI API:
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 2 \
  --llm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --llm-backend huggingface \
  --llm-d-llm 1536

# Train Stage 3: Multi-Anchor Causal Adapter Warm-Up & Joint Calibration (Soft-ECE)
conda run -n kt-research-env python scripts/train_sfn_kt.py --stage 3 --epochs 15
```

Artifacts generated:
- Stage 1: `outputs/artifacts/{exp}/fast_backbone_best.pt`
- Stage 2: `outputs/artifacts/{exp}/cognitive_qformer_cache.h5` (offline compressed HDF5 tensor cache)
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