# SSDF  Consultation Navigator Evaluation System
> A multi-turn medical consultation evaluation framework for testing the SSDF Transformer-based Navigation Model in spleen-stomach disease diagnosis.

## Introduction

This `exam` module provides a comprehensive evaluation framework for assessing the performance of the SSDF (Spleen-Stomach Disease Foundation) Consultation Navigator in conducting multi-turn Traditional Chinese Medicine (TCM) consultations for spleen-stomach diseases.

The evaluation system simulates real-world medical consultations by:
1. Using an **LLM-based Patient Simulator** to generate patient responses based on real medical records
2. Running **multi-turn conversations** between the doctor model and patient simulator
3. Assessing the doctor model's consultation capabilities including:
   - Inquiry logic and coherence
   - Symptom collection completeness
   - Diagnostic accuracy (TCM syndrome differentiation + Western medicine diagnosis)

## Project Structure

```
exam/
├── Multiround_exam.py          # Main evaluation script
├── LLM_as_Patient.py           # Patient simulator using LLM
├── Eval.py                     # Evaluation script
├── exam_datas/                 # Evaluation datasets
│   ├── exam_datas.json         # Main test cases (medical records)
│   └── clearned_3_100.json     # Cleaned test cases
├── datas/                      # Evaluation results
│   ├── eval_results/           # Evaluation outputs
│   └── exam_chat_records_*.json # Consultation records
└── logs/                      # Execution logs
```

## Core Components

### 1. Patient Simulator (`LLM_as_Patient.py`)
An LLM-based patient simulator that generates realistic patient responses based on:
- Medical record information (chief complaint, symptoms, history)
- Predefined response rules ensuring consistency and authenticity

**Features:**
- Supports two modes: default mode (chief complaint based) and medical record mode
- Maintains conversation consistency
- Generates colloquial Chinese responses mimicking real patients

### 2. Evaluation Engine (`Multiround_exam.py`)
Main evaluation script that orchestrates the consultation process:

**Key Functions:**
- `SSDFConsultationClient`: Doctor model client with Navigator collaboration
- Multi-turn consultation loop (up to 30 rounds)
- Automatic diagnosis generation when consultation ends
- Comprehensive conversation logging

### 3. Fusion Methods
Supports multiple collaboration strategies between the Navigator and doctor LLM:
- `llm_choose`: Navigator recommends symptoms, LLM decides whether to follow
- `PureLLM`: Direct LLM consultation without Navigator assistance

## Usage

### Prerequisites

1. **Running Services:**
   - SSDF-core model service (doctor model)
   - Navigator service (symptom prediction)
   - Patient simulator model service (e.g., Qwen3-32B)

2. **Required Files:**
   - Symptom vocabulary file (`symptoms.csv`)
   - Question standardization file (`question_cleaned.csv`)
   - Evaluation dataset (`exam_datas.json`)

### Running Evaluation

**Standard Evaluation (with Navigator):**
```bash
python Multiround_exam.py \
    --fusion_method llm_choose \
    --file-diagnose-records ./exam_datas/clearned_3_100.json \
    --file-save-path ./datas/exam_chat_records_llm_choose_20260305.json \
    --ssdf-core-model-base-url http://localhost:8008/v1 \
    --navigator-url http://localhost:8999/predict \
    --patient-base-url http://localhost:11118/v1 \
    --patient-model-name Qwen3-32B
```

**Pure LLM Evaluation (without Navigator):**
```bash
python Multiround_examPureLLM.py \
    --fusion_method PureLLM \
    --file-diagnose-records ./exam_datas/clearned_3_100.json \
    --file-save-path ./datas/exam_chat_records_pureLLM.json \
    --ssdf-core-model-base-url http://localhost:8008/v1 \
    --patient-base-url http://localhost:11118/v1
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fusion_method` | Collaboration method | `llm_choose` |
| `--file-diagnose-records` | Test dataset path | Required |
| `--file-save-path` | Output save path | Required |
| `--ssdf-core-model-base-url` | Doctor model API | `http://localhost:8008/v1` |
| `--navigator-url` | Navigator service URL | `http://localhost:8999/predict` |
| `--patient-base-url` | Patient simulator API | `http://localhost:11118/v1` |
| `--history-length` | Symptom history window | `5` |
| `--navigator-topk` | Navigator Top-K symptoms | `5` |
| `--low-threshold` | Navigator probability threshold | `0.1` |

## Evaluation Output

The evaluation generates:

1. **Consultation Records** (`exam_chat_records_*.json`):
   - Full multi-turn conversation history
   - Doctor queries and patient responses
   - Final diagnosis conclusions

2. **Evaluation Logs** (`logs/`):
   - Detailed execution logs
   - Timestamped conversation rounds
   - Error tracking

3. **Statistics** (`datas/statistics_results.ipynb`):
   - Conversation length analysis
   - Symptom coverage metrics
   - Diagnostic accuracy assessment

## Integration with Main Project

This evaluation module is part of the **SSDF-navigator** project:

```
SSDF-navigator/
├── exam/                      # Evaluation module (this folder)
├── train.py                   # Model training script
├── evaluate.py                # Navigator evaluation
└── [other modules]
```

For details on the Navigator model architecture and training, please refer to the main project README.

## Technical Details

### Consultation Flow

1. **Initialization**: Load patient medical record, initialize doctor client and patient simulator
2. **Chief Complaint**: Start with the patient's main symptom from the medical record
3. **Multi-turn Loop**:
   - Doctor generates inquiry question (with/without Navigator guidance)
   - Patient simulator responds based on medical record
   - Check if consultation should end
4. **Diagnosis**: Generate TCM syndrome and Western medicine diagnosis
5. **Recording**: Save complete consultation records

### Navigator Collaboration

The Navigator assists the doctor model by:
1. Predicting next K most relevant symptoms based on conversation history
2. Providing probability scores for each symptom
3. Doctor model decides whether to follow Navigator's recommendations

## License

This project is part of the SSDF-navigator research project. For license and usage terms, refer to the main project documentation.

## Acknowledgments

- TCM experts for consultation logic guidance
- Medical record data providers
- Open-source LLM projects (Qwen, etc.)
