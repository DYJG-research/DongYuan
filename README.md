# DongYuan: A Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis with a Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism

<p align="center">
  <img src="https://img.shields.io/badge/Framework-DongYuan-blue.svg" alt="DongYuan">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/Specialty-Integrative%20Medicine-brightgreen.svg" alt="Integrative Medicine">
</p>

<p align="center">
  <b>DongYuan</b> is a framework for integrative Chinese and Western medicine (ICWM) diagnosis of spleen-stomach disorders. It is centered on a Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism (KG-DMCRM), which grounds domain knowledge into two heterogeneous models and coordinates them for multi-turn proactive consultation: a core diagnostic large language model (<i>SSDF-Core</i>) and a lightweight consultation navigation model (<i>SSDF-Navigator</i>). The framework is evaluated on <i>SSDF-Bench</i>, a process-outcome coupled benchmark with a validated LLM-as-a-judge protocol.
</p>

---

## 📋 Overview

Large language models (LLMs) have improved performance on medical diagnosis tasks, yet systematically grounding domain knowledge into models and supporting the multi-turn proactive consultation that integrative diagnosis requires remain difficult. This challenge is particularly evident in **integrative spleen-stomach disorders diagnosis**, where Traditional Chinese Medicine (TCM) syndrome differentiation and Western medicine (WM) objective evidence must be jointly reasoned over within an interactive diagnostic workflow.

**DongYuan** is a framework centered on the Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism (KG-DMCRM) for ICWM spleen-stomach disorders diagnosis. KG-DMCRM grounds domain knowledge into two heterogeneous models and coordinates them for multi-turn proactive consultation: a core diagnostic LLM (*SSDF-Core*) and a lightweight consultation navigation model (*SSDF-Navigator*). The framework further introduces *SSDF-Bench*, a self-constructed process-outcome coupled benchmark with a validated LLM-as-a-judge protocol.

Experiments show that **SSDF-Core** outperforms 12 mainstream baselines on **SSDF-Bench**, and that coupling **SSDF-Navigator** with **SSDF-Core** improves critical symptom recall and consultation efficiency in multi-turn proactive consultation.

### Key Features

- **Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism (KG-DMCRM)**: Grounds domain knowledge into a core diagnostic LLM and a consultation navigation model, and coordinates them for multi-turn proactive consultation through candidate-based coordination
- **Core Diagnostic LLM (SSDF-Core)**: Trained on three expert-annotated datasets via two-stage SFT + DPO training, jointly reasoning over TCM syndrome differentiation and Western medicine objective evidence
- **Consultation Navigation Model (SSDF-Navigator)**: Provides a controllable and optimizable proactive consultation strategy through ontology-based state abstraction (83 standardized syndrome types) and hybrid imitation-reinforcement learning
- **Process-Outcome Coupled Benchmark (SSDF-Bench)**: Assesses both diagnostic outcomes and reasoning processes via a validated LLM-as-a-judge protocol

---

## 🏗️ Architecture

```
DongYuan/
├── kg_dmcrm/                              # Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism
│   ├── coordination/                      # KG-DMCRM dual-model coordination
│   │   └── llm_navigator_coordination.py  #   Orchestration of SSDF-Core & SSDF-Navigator
│   ├── ssdf_core/                         # SSDF-Core: Core Diagnostic Model
│   │   ├── run_ssdf_core_sft_training.sh  #   Supervised Fine-Tuning (SFT) script
│   │   └── run_ssdf_core_dpo_training.sh  #   Direct Preference Optimization (DPO) script
│   └── ssdf_navigator/                    # SSDF-Navigator: Consultation Navigation Model
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
│   ├── mpc_evaluation/                  # Multi-turn proactive consultation (MPC) evaluation
│   │   ├── run_mpc_collaborative_evaluation.py # Collaborative consultation evaluation
│   │   ├── run_mpc_pure_llm_evaluation.py      # Pure LLM consultation evaluation
│   │   ├── llm_patient_simulator.py            # Patient simulator
│   │   ├── navigator_api_service.py            # Navigator API service
│   │   └── evaluate_mpc.py                     # MPC result analysis
│   ├── evaluation_prompts/              # Evaluation prompt templates
│   ├── configs/                         # Configuration files
│   └── scripts/                         # Deployment and evaluation scripts
└── data_foundation/                     # Expert-annotated datasets
    ├── evaluation_data/                 # Evaluation datasets
    │   ├── sdt_data.jsonl               # SSDF-Syndrome: Syndrome differentiation cases
    │   ├── mpc_data.json                # SSDF-PD: Patient data for MPC evaluation
    │   └── other_task_data.json         # Multi-choice questions (TCM, WM, ethics, safety)
    └── train_data/                      # Training datasets
        ├── sft_data/                    # SFT data (COT, multi-round dialogue, exam enhancement)
        └── dpo_data/                    # DPO preference data
```

---

## 🔬 Methodology

### KG-DMCRM: Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism

At the core of DongYuan is KG-DMCRM, which systematically grounds domain knowledge into two heterogeneous models — a core diagnostic LLM (*SSDF-Core*) and a consultation navigation model (*SSDF-Navigator*) — and coordinates them for multi-turn proactive consultation through a candidate-based coordination mechanism. For each model, domain knowledge is grounded through a "knowledge carrier + injection method" pair:

#### SSDF-Core: Core Diagnostic Model

**Expert-annotated datasets (knowledge carrier).** The knowledge grounding of SSDF-Core rests on three self-constructed, expert-annotated datasets. These datasets explicitly encode expert clinical reasoning rather than simple input-output pairs, all built by ICWM physicians under double-blind annotation and third-party arbitration:

| Dataset | Description |
|---------|-------------|
| **SSDF-Syndrome** | Syndrome-differentiation reasoning chains (1,345 samples) with complete patient records, tongue/pulse diagnoses, examination results, and gold-standard syndrome labels |
| **SSDF-Dialogue** | Multi-turn doctor-patient consultation dialogues with structured reasoning templates (1,080 cases), mapped to 83 standardized syndrome types |
| **SSDF-PD** | Expert decision-preference data (12,800 chosen-rejected pairs) encoding expert preferences for proactive consultation and syndrome differentiation |

**Two-stage training (injection method).** To deeply ground the domain knowledge carried by these datasets into the model, SSDF-Core is trained on Qwen3-14B via a two-stage SFT + DPO pipeline: supervised fine-tuning (SFT) first learns expert diagnostic reasoning, and direct preference optimization (DPO) then aligns the model with expert decision preferences, enabling it to jointly reason over TCM syndrome differentiation and Western medicine objective evidence.

#### SSDF-Navigator: Consultation Navigation Model

**Ontology-based state abstraction (knowledge carrier).** To provide a controllable and optimizable consultation strategy, SSDF-Navigator maps raw natural-language consultation histories into a structured state space: under the guidance of the national standard *Clinic Terminology of Traditional Chinese Medical Diagnosis and Treatment* and ICWM physicians, physician-initiated inquiry questions are mapped to 83 standardized spleen-stomach syndrome types, converting a raw consultation history into a syndrome-state sequence.

**Hybrid imitation-reinforcement learning (injection method).** SSDF-Navigator is a lightweight Transformer encoder-classifier policy network trained via a hybrid paradigm combining Behavioral Cloning (BC) and reward-driven offline Reinforcement Learning (RL), learning the expert's progressive inquiry strategy:

- **Problem Formulation** — The multi-turn consultation is modeled as a sequential decision problem:
  - **State**: A context window of the most recent $L$ historical standardized syndrome types: $state_t = [s_{t-L+1}, ..., s_t]$
  - **Action**: The next standardized syndrome type to query: $a_t = s_{t+1}$
  - **Policy**: $\pi(a_t | state_t)$ learned to mimic expert progressive inquiry logic
- **Model Architecture** — SSDF-Navigator adopts a Transformer encoder-classifier design (embedding layer with sinusoidal positional encoding → Transformer encoder → masked average pooling → classification head over 83 syndrome types)
- **Information Gain Reward Function**: $R_{info}(a) = 1 - \tilde{H}(\mathcal{S} | a)$ where $\tilde{H}(\mathcal{S} | a)$ is the normalized conditional entropy after querying syndrome type $a$, with a repetition factor $\lambda(a, x, H_t, \mathcal{M}_t)$
- **Reward-Weighted Behavior Cloning (RWBC)** (`--fusion_mode mul`): $\mathcal{L}_{RWBC}(\theta) = -\frac{1}{B} \sum r_t \cdot \log \pi_{\theta}(a_t^* | X_t)$ — uses rewards as sample weights
- **Reward-Added Behavior Cloning (RABC)** (`--fusion_mode add`): $\mathcal{L}_{RABC}(\theta) = \beta_1 \mathcal{L}_{BC} + \beta_2 \mathcal{L}_{PO} + \mathcal{L}_{Ent}$ — combines supervised loss, reward-driven policy optimization, and entropy regularization

#### Dual-Model Collaboration: Candidate-Based Coordination

The two models collaborate in a closed-loop diagnostic workflow through a candidate-based coordination mechanism:

1. SSDF-Core generates a candidate inquiry based on the current patient information
2. The inquiry is mapped to a standardized syndrome type
3. SSDF-Navigator evaluates the current syndrome state and outputs top-K syndrome-type recommendations with a probability distribution
4. SSDF-Core selects the next inquiry from the combined candidate set (Navigator's top-K recommendations plus SSDF-Core's own candidate)
5. New patient responses update the syndrome state, and the cycle continues until sufficient information is acquired for diagnosis

### SSDF-Bench: Process-Outcome Coupled Evaluation

SSDF-Bench provides a comprehensive evaluation framework that assesses both the diagnostic outcome and the reasoning process. It encompasses six tasks, of which two are core clinical tasks for ICWM spleen-stomach disorders diagnosis (SDT and MPC) and four are conventional multiple-choice tasks assessing disease-general medical knowledge, safety, and ethics (TCM-BC, WM-BC, MS, ME).

**Syndrome Differentiation and Treatment (SDT) Dimensions:**

| Dimension | Evaluation Method | Description |
|-----------|-------------------|-------------|
| Syndrome | Multiple-choice (multi-select) | TCM pattern identification |
| Nature of Disease | Multiple-choice (single-select) | Deficiency-excess, cold-heat nature |
| Location of Disease | Multiple-choice (multi-select) | Affected organ system localization |
| Therapeutic Principles | Multiple-choice (multi-select) | Treatment principles and methods |
| Causative Factors | LLM-as-Judge | Cause of disease analysis |
| Pathogenesis | LLM-as-Judge | Pathological mechanism interpretation |
| Treatment Method | LLM-as-Judge | Specific treatment approaches |
| Precautions | LLM-as-Judge | Medical advice and lifestyle guidance |
| CoT Completeness | LLM-as-Judge | Comprehensiveness of patient information utilization |
| CoT Accuracy | LLM-as-Judge | Hallucination check in reasoning process |

**Additional Assessment Tasks:**

| Task | Description | Weight |
|----------|-------------|--------|
| TCM-BC (Basic Knowledge of TCM) | TCM fundamental knowledge | 20% |
| WM-BC (Basic Knowledge of Western Medicine) | Western medical knowledge | 20% |
| MS (Medical Safety) | Clinical boundary compliance and safety alignment | 10% |
| ME (Medical Ethics) | Medical ethics understanding and judgment | 10% |

**Multi-Turn Proactive Consultation (MPC) Evaluation:** Using LLM-simulated patients to assess critical symptom recall, inquiry efficiency, and diagnostic accuracy over complete multi-turn dialogues.

**Scoring:** SDT (40%) + TCM-BC (20%) + WM-BC (20%) + MS (10%) + ME (10%). Multiple-choice scoring uses the Sp metric: $Sp = |A \cap B| / (|A| + |\bar{A} \cap B|)$.

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

### Training SSDF-Navigator (Consultation Navigation Model)

```bash
cd kg_dmcrm/ssdf_navigator

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

### Training SSDF-Core (Core Diagnostic Model)

**Stage 1 — Supervised Fine-Tuning (SFT):**
```bash
cd kg_dmcrm/ssdf_core
bash run_ssdf_core_sft_training.sh
```

**Stage 2 — Direct Preference Optimization (DPO):**
```bash
bash run_ssdf_core_dpo_training.sh
```

### Running KG-DMCRM Collaborative Consultation

```bash
cd ssdf_bench/mpc_evaluation

# Pure LLM baseline (SSDF-Core only)
python run_mpc_pure_llm_evaluation.py \
  --core_model "your-model" \
  --core_model_base_url "http://localhost:8000/v1"

# KG-DMCRM collaborative evaluation (SSDF-Core + SSDF-Navigator)
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
@article{dongyuan2026framework,
  title = {DongYuan: A Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis with a Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism},
  author = {Your Authors},
  journal = {Journal of Biomedical Informatics},
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
