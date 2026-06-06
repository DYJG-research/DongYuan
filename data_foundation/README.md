# Data Foundation

This directory contains the expert-knowledge-annotated datasets used by DongYuan, corresponding to **Layer 1** of the paper's three-layer architecture.

## Dataset Overview

| Dataset | File | Description | Format |
|---------|------|-------------|--------|
| **SSDF-Syndrome** | `evaluation_data/sdt_data.jsonl` | Syndrome differentiation cases with complete patient records, diagnostic results, and gold-standard syndrome labels | JSONL (messages format) |
| **SSDF-Dialogue** | `train_data/sft_data/ssdf_dialogue/` | Multi-turn doctor-patient consultation dialogues mapped to standardized symptoms | JSONL (messages format) |
| **SSDF-PD** | `evaluation_data/mpc_data.json` | Pairwise samples for direct preference optimization (DPO) | JSON |
| **Objective Evaluation** | `evaluation_data/other_task_data.json` | Multiple-choice questions for Mediccal ethics, safety assessment, etc | JSON |

## Data Privacy & Compliance

> **⚠️ IMPORTANT**: The datasets in this repository have been **de-identified** to remove direct personal identifiers. However, the SSDF-Syndrome data contains detailed clinical case descriptions derived from real patient encounters. Users must:
>
> 1. Use this data for **research purposes only**
> 2. Not attempt to re-identify individuals
> 3. Comply with all applicable privacy regulations (HIPAA, PIPL, etc.)
> 4. Cite the original paper when using the data in publications

## Training Data

The `train_data/` directory contains SFT and DPO training data in `messages` format. Key subsets:

- **`sft_data/ssdf_dialogue/`**: SSDF-Dialogue — multi-round consultation data with CoT reasoning
- **`sft_data/other_data/`**: Supplementary training data (medical knowledge, ethics, safety, general instruction)
- **`dpo_data/`**: Preference pairs for DPO training

> **Note on data availability**: If you need access to larger or raw versions of these datasets, please contact the authors or refer to the paper for data access instructions.

## Data Schema

### SSDF-Syndrome

```json
{
  "messages": [
    {"role": "system", "content": "System prompt for the diagnostic task..."},
    {"role": "user", "content": "Patient description with symptoms, history, examinations..."},
    {"role": "assistant", "content": "Structured diagnostic output with CoT reasoning..."}
  ]
}
```

### SSDF-PD 

```json
{
  "id": 0,
  "disease_type": "脾胃病",
  "disease_name": "胃食管反流病",
  "syndrome_name": "脾虚湿热证",
  "patient_summary": "...",
  "chat_rounds": {
    "messages": [
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }
}
```
