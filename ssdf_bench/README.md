# SSDF-Bench: Process-Outcome Coupled Evaluation for Integrative Chinese and Western Medicine Diagnosis

<p align="center">
  <img src="https://img.shields.io/badge/Framework-DongYuan-blue.svg" alt="DongYuan">
  <img src="https://img.shields.io/badge/Module-SSDF--Bench-orange.svg" alt="SSDF-Bench">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
</p>

A comprehensive process-outcome coupled evaluation benchmark for assessing LLMs on integrative Chinese and Western medicine spleen-stomach disorders diagnosis. **SSDF-Bench** is the evaluation layer of the **DongYuan** medical expert system.

> **📌 This module is part of the [DongYuan](https://github.com/your-org/DongYuan) project.**  
> Refer to the [root README](../README.md) for the full system overview and architecture.

## 📋 Overview

SSDF-Bench (Spleen and Stomach Disease Framework Benchmark) systematically assesses LLMs on integrative Chinese and Western medicine diagnosis, focusing on spleen and stomach diseases. It evaluates both the diagnostic **outcome** and the reasoning **process** via a validated LLM-as-a-judge protocol.

### Key Features

- **Multi-dimensional Evaluation**: Assesses 10 key dimensions of TCWM diagnosis
- **Bias Mitigation**: Uses option shuffling with 3-round evaluation to reduce position bias
- **Dual-mode Support**: Supports both API-based models and local model deployment
- **LLM-as-Judge**: Employs LLM-based evaluation for complex dimensions like etiology and pathogenesis
- **Chain-of-Thought Analysis**: Evaluates reasoning completeness and accuracy
- **Comprehensive Reporting**: Generates detailed HTML evaluation reports

## 🏗️ Architecture

```
ssdf_bench/
├── ssdf_bench.py                     # Main benchmark runner
├── benchmark_core/                   # Core evaluation modules
│   ├── data_loader.py               # Data loading
│   ├── model_interface.py           # Unified model interface (API & Local)
│   ├── report_generator.py          # HTML report generation
│   └── utils.py                     # Utility functions
├── evaluators/                      # Evaluators
│   ├── multiple_choice_evaluator.py # Multiple-choice evaluation
│   └── llm_judge_evaluator.py       # LLM-as-Judge evaluation
├── evaluation_prompts/              # Evaluation prompt templates
├── configs/                         # Configuration files
├── mpc_evaluation/                  # Multi-patient consultation evaluation
└── results/                         # Evaluation outputs
```

## 📊 Evaluation Dimensions

The benchmark evaluates models across the following TCWM diagnostic dimensions:

| Dimension | Chinese | Evaluation Method |
|-----------|---------|-------------------|
| Syndrome Type | 证型 | Multiple-choice (multi-select) |
| Disease Nature | 病性 | Multiple-choice (single-select) |
| Disease Location | 病位 | Multiple-choice (multi-select) |
| Treatment Principles | 治则治法 | Multiple-choice (multi-select) |
| Etiology | 病因 | LLM-as-Judge |
| Pathogenesis | 病机 | LLM-as-Judge |
| Treatment Methods | 治疗方法 | LLM-as-Judge |
| Precautions | 注意事项 | LLM-as-Judge |
| Completeness of COT | Comprehensiveness of patient information utilization | LLM Evaluation |
| Accuracy of COT | Hallucination check | LLM Evaluation |
### Additional Assessment Categories

- **TCWM Basic Medicine** (中医药学): TCWM fundamental knowledge
- **Western Medicine** (西医药学): Western medical knowledge
- **Medical Ethics** (医学伦理): Ethical considerations in medicine
- **Content Safety** (安全评估): LLM safety and alignment

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/your-org/DongYuan.git
cd DongYuan
pip install -r requirements.txt
```

### Running Evaluation

#### API Mode (Recommended for cloud models)

```bash
python ssdf_bench.py \
    --model_type api \
    --api_url "http://localhost:8000/v1" \
    --model_name "your-model-name" \
    --api_key "your-api-key" \
    --output_dir "results/your-model-name-run"
```

#### Local Mode (For self-hosted models)

```bash
python ssdf_bench.py \
    --model_type local \
    --model_path "./models/your-model" \
    --output_dir "results/local-model-run"
```

### Command-line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_type` | Model type: `api` or `local` | `api` |
| `--api_url` | API endpoint URL | `http://localhost:8000/v1` |
| `--model_name` | Model identifier | `lora_5e6` |
| `--api_key` | API authentication key | `-` |
| `--model_path` | Path to local model | `-` |
| `--output_dir` | Output directory for results | `results/benchmark_run` |
| `--config_file` | Path to configuration file | Config file path |
| `--resume` | Resume from checkpoint | `True` |
| `--skip_think` | Skip CoT evaluation (for non-CoT models) | `False` |

## 📁 Data Format

### Syndrome Differentiation Cases

```json
{
  "id": "case_001",
  "instruction": "Patient description with symptoms, tongue, pulse...",
  "output": {
    "证型": "脾胃虚弱",
    "证型答案": "A;B",
    "证型选项": "A:脾胃虚弱;B:脾胃湿热;...",
    "病性": "虚证",
    "病性答案": "A",
    "病性选项": "A:虚证;B:实证;...",
    "病位": "脾胃",
    "病位答案": "A;B",
    "病位选项": "A:脾;B:胃;...",
    "治则治法": "健脾益气",
    "治则治法答案": "A;B",
    "治则治法选项": "A:健脾益气;B:清热化湿;...",
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

### Multiple Choice Cases

```json
{
  "id": "question_001",
  "question": "The question text...",
  "answer": "A",
  "option": {"A": "Option 1", "B": "Option 2"},
  "question_type": "单项选择题",
  "exam_class": "中医药学"
}
```

## 📈 Scoring

The final score is computed using a weighted combination:

| Task Category | Weight |
|---------------|--------|
| Syndrome Differentiation (中西医辩证分型) | 40% |
| TCWM Basic Medicine (中医药学) | 20% |
| Western Medicine (西医药学) | 20% |
| Medical Ethics (医学伦理) | 10% |
| Content Safety (安全评估) | 10% |

### Syndrome Differentiation Scoring

- **Multiple-choice dimensions**: Uses Sp metric: `Sp = |A∩B| / (|A| + |Ā∩B|)`
  - Where A is the ground truth set, B is the model's selection
- **LLM-judged dimensions**: Scored 0-100, normalized to 0-1

## 🔧 Configuration

Edit `configs/config_example.json` to customize evaluation settings:

```json
{
  "data_path": "./data/evaluation_data/sdt_data.jsonl",
  "llm_judge_api_host": "127.0.0.1",
  "llm_judge_api_port": 8002,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "",
  "max_retries": 3,
  "checkpoint_interval": 10,
  "random_seed": 42
}
```

## 📝 Output

The benchmark generates comprehensive evaluation reports:

```
results/your-model-name-run/
├── checkpoint.json          # Resume checkpoint
├── detailed_results.json    # Complete evaluation results
├── evaluation_report.html  # Visual HTML report
└──logs_YYYYMMDD-HHMMSS.log # Execution logs
```

## 🛠️ Dependencies

- `transformers` - Model loading
- `torch` - Deep learning backend
- `openai` - API client
- `httpx` - HTTP client
- `tqdm` - Progress bars
- `matplotlib` - Visualization
- `numpy` - Numerical computing
- `json-repair` - JSON parsing

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Citation

If you use this benchmark in your research, please cite the DongYuan paper:

```bibtex
@article{dongyuan,
  title = {DongYuan: An LLM-Based Medical Expert System for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis},
  author = {Your Authors},
  journal = {Expert Systems with Applications},
  year = {2026}
}
```

## ⚠️ Disclaimer

This benchmark is intended for research and educational purposes only. The evaluation results should not be used as medical advice. Always consult qualified healthcare professionals for medical decisions.

---

<p align="center">Built with ❤️ for Traditional Chinese Medicine AI Research</p>
