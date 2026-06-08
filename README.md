# DongYuan: An LLM-Based Medical Expert System for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis

<p align="center">
  <img src="https://img.shields.io/badge/System-DongYuan-blue.svg" alt="DongYuan">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/Specialty-Integrative%20Medicine-brightgreen.svg" alt="Integrative Medicine">
</p>

<p align="center">
  <b>DongYuan</b> is an LLM-based medical expert system for integrative Chinese and Western medicine diagnosis of spleen-stomach disorders. Built upon three progressively connected layers — an expert-knowledge-annotated data foundation, a knowledge-grounded dual-module collaborative reasoning architecture (KG-DMCRA), and a process-outcome coupled evaluation benchmark (SSDF-Bench) — DongYuan addresses the challenge of clinically usable disease-specific diagnosis where TCM syndrome differentiation and Western medicine disease evidence must be jointly reasoned over within an interactive diagnostic workflow.
</p>

---

## 📋 Overview

Large language models (LLMs) have improved performance on medical diagnosis tasks, yet turning them into clinically usable expert systems remains difficult for complex disease-specific diagnosis. This challenge is particularly evident in **integrative spleen-stomach disorders diagnosis**, where Traditional Chinese Medicine (TCM) syndrome differentiation and Western medicine (WM) disease evidence must be jointly reasoned over within an interactive diagnostic workflow.

**DongYuan** addresses this problem through three progressively connected layers:

1. **Expert-Knowledge-Annotated Data Foundation** — comprising three specialized datasets: *SSDF-Syndrome*, *SSDF-Dialogue*, and *SSDF-PD*
2. **Knowledge-Grounded Dual-Module Collaborative Reasoning Architecture (KG-DMCRA)** — where the deep diagnostic reasoning engine (*SSDF-Core*) and the active information acquisition engine (*SSDF-Navigator*) work in tandem
3. **SSDF-Bench** — a process-outcome coupled evaluation benchmark with a validated LLM-as-a-judge protocol

Experiments show that **SSDF-Core** outperforms 12 mainstream baselines on **SSDF-Bench**, while architecture-level analyses confirm the contribution of dual-module collaboration. DongYuan suggests a general design direction for medical expert systems in clinically complex and heterogeneous-knowledge diagnostic scenarios.

### Key Features

- **Integrative Medicine Diagnosis**: Jointly reasons over TCM syndrome differentiation and Western medicine disease evidence within an interactive diagnostic workflow
- **Three-Layer Architecture**: Data foundation → dual-module collaborative reasoning → process-outcome coupled evaluation
- **Knowledge-Grounded Dual-Module Collaboration (KG-DMCRA)**: Deep diagnostic reasoning engine (SSDF-Core) + active information acquisition engine (SSDF-Navigator)
- **Expert-Knowledge-Annotated Datasets**: SSDF-Syndrome, SSDF-Dialogue, and SSDF-PD, built with expert-guided standardization
- **Process-Outcome Coupled Evaluation**: SSDF-Bench assesses both diagnostic outcomes and the inquiry process via a validated LLM-as-a-judge protocol
- **Standardized Symptom Space**: 83 expert-validated standardized symptom terms for spleen-stomach disorders

---

## 🏗️ Architecture

```
DongYuan/
├── kg_dmcra/                              # Knowledge-Grounded Dual-Module Collaborative Reasoning Architecture
│   ├── coordination/                      # KG-DMCRA collaborative consultation
│   │   └── llm_navigator_coordination.py  #   Orchestration of SSDF-Core & SSDF-Navigator
│   ├── ssdf_core/                         # SSDF-Core: Deep Diagnostic Reasoning Engine
│   │   ├── run_ssdf_core_sft_training.sh  #   Supervised Fine-Tuning (SFT) script
│   │   └── run_ssdf_core_dpo_training.sh  #   Direct Preference Optimization (DPO) script
│   └── ssdf_navigator/                    # SSDF-Navigator: Active Information Acquisition Engine
│       ├── policy_network.py              #   Transformer-based policy network
│       ├── behavior_cloning.py            #   Hybrid BC + RL trainer
│       ├── data_loader.py                 #   Offline RL data loader
│       ├── train_navigator.py             #   Navigator training entry point
│       ├── evaluate_navigator.py          #   Navigator evaluation entry point
│       └── data_construction/             #   Training data construction
│           ├── extract_questions_from_chat_records.py
│           ├── map_questions_to_symptoms.py
│           ├── normalize_symptom_names.py
│           ├── split_conversation_segments.py
│           ├── extract_symptom_inventory.py
│           ├── filter_questions_by_mapping.py
│           ├── filter_short_dialogues.py
│           └── group_question_symptom_mapping_by_symptom.py
├── ssdf_bench/                            # SSDF-Bench: Process-Outcome Coupled Evaluation
│   ├── ssdf_bench.py                     # Benchmark main entry point
│   ├── benchmark_core/                   # Core evaluation modules
│   │   ├── data_loader.py               # Data loading
│   │   ├── model_interface.py           # Unified model interface (API & Local)
│   │   ├── report_generator.py          # HTML report generation
│   │   └── utils.py                     # Utility functions
│   ├── evaluators/                      # Evaluators
│   │   ├── multiple_choice_evaluator.py # Multiple-choice evaluation
│   │   └── llm_judge_evaluator.py       # LLM-as-Judge evaluation
│   ├── mpc_evaluation/                  # Multi-patient consultation (MPC) evaluation
│   │   ├── run_mpc_collaborative_evaluation.py # Collaborative consultation evaluation
│   │   ├── run_mpc_pure_llm_evaluation.py      # Pure LLM consultation evaluation
│   │   ├── llm_patient_simulator.py            # Patient simulator
│   │   ├── navigator_api_service.py            # Navigator API service
│   │   └── evaluate_mpc.py                     # MPC result analysis
│   ├── evaluation_prompts/              # Evaluation prompt templates
│   ├── configs/                         # Configuration files
│   └── scripts/                         # Deployment and evaluation scripts
└── data_foundation/                     # Expert-knowledge-annotated data foundation
    ├── evaluation_data/                 # Evaluation datasets
    │   ├── sdt_data.jsonl               # SSDF-Syndrome: Syndrome differentiation cases
    │   ├── mpc_data.json                # SSDF-PD: Patient data for MPC evaluation
    │   └── other_task_data.json         # Multi-choice questions (TCWM, WM, ethics, safety)
    └── train_data/                      # Training datasets
        ├── sft_data/                    # SFT data (COT, multi-round dialogue, exam enhancement)
        └── dpo_data/                    # DPO preference data
```

---

## 🔬 Methodology

### Three-Layer Architecture

DongYuan is organized as three progressively connected layers:

#### Layer 1: Expert-Knowledge-Annotated Data Foundation

The data foundation consists of three specialized datasets, all annotated under the guidance of TCM experts:

| Dataset | Description |
|---------|-------------|
| **SSDF-Syndrome** | Syndrome differentiation cases with complete patient records, tongue/pulse diagnoses, examination results, and gold-standard syndrome labels |
| **SSDF-Dialogue** | Multi-turn doctor-patient consultation dialogues mapped to 83 standardized symptoms, used for training the active information acquisition engine |
| **SSDF-PD** | Patient data for multi-patient consultation (MPC) evaluation, covering various spleen-stomach diseases with structured symptom profiles |

#### Layer 2: KG-DMCRA (Knowledge-Grounded Dual-Module Collaborative Reasoning Architecture)

At the core of DongYuan is KG-DMCRA, which instantiates two complementary engines:

- **SSDF-Core (Deep Diagnostic Reasoning Engine)**: A large language model fine-tuned from Qwen3-14B via two-stage SFT + DPO training, equipped with comprehensive clinical knowledge for integrative diagnosis. Given a patient's symptom history and examination results, SSDF-Core performs deep reasoning over both TCM syndrome differentiation and Western medicine disease evidence.

- **SSDF-Navigator (Active Information Acquisition Engine)**: A lightweight Transformer encoder-classifier policy network trained via a hybrid paradigm combining Behavioral Cloning (BC) and offline Reinforcement Learning (RL) with information gain rewards. It models the expert's progressive inquiry strategy and actively acquires missing but critical symptom information through multi-turn interaction.

The two engines collaborate in a closed-loop diagnostic workflow:

1. SSDF-Core generates a candidate inquiry based on current patient information
2. The inquiry is mapped to a standardized symptom
3. SSDF-Navigator evaluates the current symptom state and outputs top-K symptom recommendations with probability distribution
4. KG-DMCRA coordinates the decision: if SSDF-Core's inquiry aligns with Navigator's high-confidence recommendations, it proceeds; otherwise, a threshold-guided negotiation mechanism refines the next inquiry
5. New patient responses update the symptom state, and the cycle continues until sufficient information is acquired for diagnosis

##### SSDF-Navigator Training

**Problem Formulation** — The multi-turn consultation is modeled as a sequential decision problem:
- **State**: A context window of the most recent $L$ historical standardized symptoms: $state_t = [s_{t-L+1}, ..., s_t]$
- **Action**: The next standardized symptom to query: $a_t = s_{t+1}$
- **Policy**: $\pi(a_t | state_t)$ learned to mimic expert progressive inquiry logic

**Model Architecture** — The SSDF-Navigator adopts a Transformer encoder-classifier design (embedding layer with sinusoidal positional encoding → Transformer encoder → masked average pooling → classification head over 83 symptoms).

**Hybrid Training Paradigm** — Combines imitation learning with offline RL:

1. **Information Gain Reward Function**: $R_{info}(a) = 1 - \tilde{H}(Y | a)$ where $\tilde{H}(Y | a)$ is the normalized conditional entropy after querying symptom $a$, with a repetition penalty factor $\lambda(a, \mathcal{H})$

2. **Reward-Weighted Behavior Cloning (RWBC)** ($\text{--fusion_mode mul}$): $\mathcal{L}_{RWBC}(\theta) = -\frac{1}{B} \sum r_t \cdot \log \pi_{\theta}(a_t^* | X_t)$ — uses rewards as sample weights

3. **Reward-Added Behavior Cloning (RABC)** ($\text{--fusion_mode add}$): $\mathcal{L}_{RABC}(\theta) = \beta_1 \mathcal{L}_{BC} + \beta_2 \mathcal{L}_{PG} + \mathcal{L}_{Ent}$ — combines supervised loss, policy gradient, and entropy regularization

#### Layer 3: SSDF-Bench (Process-Outcome Coupled Evaluation)

SSDF-Bench provides a comprehensive evaluation framework that assesses both the diagnostic outcome and the reasoning process:

**Syndrome Differentiation Dimensions:**

| Dimension | Evaluation Method | Description |
|-----------|-------------------|-------------|
| Syndrome Type | Multiple-choice (multi-select) | TCM pattern identification |
| Disease Nature | Multiple-choice (single-select) | Deficiency-excess, cold-heat nature |
| Disease Location | Multiple-choice (multi-select) | Affected organ system localization |
| Treatment Principles | Multiple-choice (multi-select) | Therapeutic principles and methods |
| Etiology | LLM-as-Judge | Cause of disease analysis |
| Pathogenesis | LLM-as-Judge | Pathological mechanism interpretation |
| Treatment Methods | LLM-as-Judge | Specific treatment approaches |
| Precautions | LLM-as-Judge | Medical advice and lifestyle guidance |
| CoT Completeness | LLM-as-Judge | Comprehensiveness of patient information utilization |
| CoT Accuracy | LLM-as-Judge | Hallucination check in reasoning process |

**Additional Assessment Categories:**

| Category | Description | Weight |
|----------|-------------|--------|
| TCWM Basic Medicine | TCM fundamental knowledge | 20% |
| Western Medicine | Western medical knowledge | 20% |
| Medical Ethics | Ethical considerations | 10% |
| Content Safety | LLM safety and alignment | 10% |

**Multi-Patient Consultation (MPC) Evaluation:** Using LLM-simulated patients to assess inquiry efficiency, key symptom coverage, and diagnostic accuracy over complete multi-turn dialogues.

**Scoring:** Syndrome differentiation (40%) + TCWM (20%) + Western Medicine (20%) + Ethics (10%) + Safety (10%). Multiple-choice scoring uses the Sp metric: $Sp = |A \cap B| / (|A| + |\bar{A} \cap B|)$.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
PyTorch ≥ 2.0
```

### Installation

```bash
git clone https://github.com/your-org/DongYuan.git
cd DongYuan
pip install -r requirements.txt
```

### Training SSDF-Navigator (Active Information Acquisition Engine)

```bash
cd kg_dmcra/ssdf_navigator

# RWBC mode (Reward-Weighted Behavior Cloning)
python train_navigator.py \
  --question_file ./question_cleaned.csv \
  --symptom_file ./symptoms.csv \
  --fusion_mode mul \
  --max_seq_len 10 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --save_path transformer_policy_mul.pth \
  --gpu 0

# RABC mode (Reward-Added Behavior Cloning)
python train_navigator.py \
  --fusion_mode add \
  --lambda_supervised 1.0 \
  --lambda_reinforce 0.5 \
  --entropy_coef 0.01 \
  # ... other args same as above
```

### Evaluating SSDF-Navigator

```bash
python evaluate_navigator.py \
  --question_file ./question_cleaned.csv \
  --symptom_file ./symptoms.csv \
  --model_path transformer_policy_mul.pth \
  --topk 5 \
  --eval_split val \
  --gpu 0
```

### Deploying Navigator API Service

```bash
cd ssdf_bench/mpc_evaluation
python navigator_api_service.py \
  --port 8999 \
  --model_path /path/to/transformer_policy_mul.pth
```

### Training SSDF-Core (Deep Diagnostic Reasoning Engine)

**Stage 1 — Supervised Fine-Tuning (SFT):**
```bash
cd kg_dmcra/ssdf_core
bash run_ssdf_core_sft_training.sh
```

**Stage 2 — Direct Preference Optimization (DPO):**
```bash
bash run_ssdf_core_dpo_training.sh
```

### Running KG-DMCRA Collaborative Consultation

```bash
cd ssdf_bench/mpc_evaluation

# Pure LLM baseline (SSDF-Core only)
python run_mpc_pure_llm_evaluation.py \
  --core_model "your-model" \
  --core_model_base_url "http://localhost:8000/v1"

# KG-DMCRA collaborative evaluation (SSDF-Core + SSDF-Navigator)
python run_mpc_collaborative_evaluation.py \
  --use_ssdf_navigator \
  --ssdf_core_model "your-model" \
  --ssdf_core_model_base_url "http://localhost:8000/v1" \
  --navigator_url "http://localhost:8999/predict" \
  --navigator_topk 3
```

### Running SSDF-Bench Evaluation

```bash
cd ssdf_bench

python ssdf_bench.py \
  --model_type api \
  --api_url "http://localhost:8000/v1" \
  --model_name "your-model" \
  --api_key "your-api-key" \
  --output_dir "results/your-model-run"
```

---

## 📁 Data Format

### SSDF-Syndrome: Syndrome Differentiation Case

```json
{
  "id": "case_001",
  "instruction": "Full patient symptom description including tongue, pulse, history, examination results...",
  "output": {
    "证型": "脾胃虚弱",
    "证型答案": "A;B",
    "证型选项": "A:脾胃虚弱;B:脾胃湿热;C:寒湿困脾;...",
    "病性": "虚证",
    "病性答案": "A",
    "病性选项": "A:虚证;B:实证;C:虚实夹杂;...",
    "病位": "脾胃",
    "病位答案": "A;B",
    "病位选项": "A:脾;B:胃;C:肝;D:肾;...",
    "治则治法": "健脾益气",
    "治则治法答案": "A;B",
    "治则治法选项": "A:健脾益气;B:清热化湿;C:温中散寒;...",
    "病因": "饮食不节,劳倦过度",
    "病机": "脾胃虚弱,运化失常",
    "治疗方法": "中药汤剂,针灸",
    "注意事项": "饮食调护,情志调节"
  },
  "disease_cn": "脾胃病",
  "disease_en": "Spleen and Stomach Disease",
  "exam_class": "中西医辩证分型"
}
```

---

## 🔧 Configuration

Edit `ssdf_bench/configs/benchmark_config.example.json`:

```json
{
  "data_path": "evaluation_data/sdt_data.jsonl",
  "llm_judge_api_host": "127.0.0.1",
  "llm_judge_api_port": 8002,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "",
  "max_retries": 3,
  "checkpoint_interval": 10,
  "random_seed": 42
}
```

---

## 📝 Output

```
results/your-model-run/
├── checkpoint.json                # Checkpoint for resuming
├── detailed_results.json          # Complete evaluation results
├── evaluation_report.html         # Visual HTML evaluation report
└── logs_YYYYMMDD-HHMMSS.log       # Execution log
```

---

## 🛠️ Dependencies

| Dependency | Purpose |
|------------|---------|
| `torch` | Deep learning framework |
| `transformers` | Model loading and inference |
| `openai` | LLM API client |
| `fastapi` + `uvicorn` | Navigator API service |
| `tqdm` | Training and evaluation progress bars |
| `matplotlib` | Visualization reports |
| `numpy` + `pandas` | Data processing |
| `json-repair` | Fault-tolerant JSON parsing |
| `accelerate` + `deepspeed` | Distributed training |

---

## 📥 Data Availability

The DongYuan-specific core datasets (SSDF-Syndrome, SSDF-Dialogue, SSDF-PD) are included directly in this repository under `data_foundation/`. The supplementary training data (general instruction, medical knowledge, ethics, safety, etc.) in `data_foundation/train_data/sft_data/other_data/` is excluded from the repo due to its size and can be downloaded separately:

- **Supplementary Training Data**: [https://pan.quark.cn/s/9633dc0ed2da](https://pan.quark.cn/s/9633dc0ed2da)

After downloading, extract the contents into `data_foundation/train_data/sft_data/other_data/` to reproduce the full training setup.

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use DongYuan in your research, please cite our paper:

```bibtex
@article{dongyuan2026dongyuan,
  title = {DongYuan: An LLM-Based Medical Expert System for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis},
  author = {Your Authors},
  journal = {Expert Systems with Applications},
  year = {2026},
  note = {Submitted for review}
}
```

---

## 🤝 Contributing

We welcome Issues and Pull Requests. Whether it's bug fixes, feature suggestions, or documentation improvements, your contributions are greatly appreciated.

---

## 🙏 Acknowledgments

- The team of TCM and WM experts who contributed to data annotation and symptom standardization
- The contributors of the open-source base model (Qwen3-14B) used in this research
- SwanLab for providing experiment tracking support

---

## ⚠️ Disclaimer

This project is intended for research and educational purposes only. All diagnostic results are for reference only and should not be used for clinical decision-making. Please consult qualified healthcare professionals for actual medical advice.

---

<p align="center">Built with ❤️ for Integrative Medicine AI Research</p>
