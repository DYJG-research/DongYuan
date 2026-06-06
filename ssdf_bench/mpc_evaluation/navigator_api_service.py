"""
FastAPI server for Transformer-based Medical Consultation Navigator.
Based on evaluate.py inference logic.
"""
import argparse
import os
import sys

# Ensure project root is importable when this file is run from exam/ or elsewhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from data_loader import QuestionDataset
from kg_dmcra.ssdf_navigator.behavior_cloning  import BehaviorCloning
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager
import warnings
import pandas as pd

# Default configuration (override via CLI arguments or environment variables)
DEFAULT_CONFIG = {
    "question_file": "./data/question_cleaned.csv",
    "symptom_file": "./data/symptoms.csv",
    "state_type": "transformer_seq",
    "d_model": 256,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_len": 10,
    "train_ratio": 0.8,
    "gpu": None,
    "reward_type": "information_gain",
    "lr": 1e-4,
    "model_path": "./models/transformer_policy.pth",
    "splitted_dataset_path": "./data/data_split.pkl",
    "topk": 5,
}

class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    symptoms: List[str]  # List of symptom names (strings)
    topk: Optional[int] = DEFAULT_CONFIG["topk"]  # Number of top predictions to return

class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    predicted_symptom: str  # Top-1 predicted symptom
    topk_symptoms: List[str]  # List of top-K predicted symptoms
    topk_probabilities: List[float]  # Corresponding probabilities (softmax)
    history_before_focus: List[str]  # History symptoms before the current focus
    current_focus: str  # Current focus symptom (last symptom in input list)

class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""
    eval_split: str = "val"  # "train", "val", or "all"
    topk: int = DEFAULT_CONFIG["topk"]

class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint."""
    total_transitions: int
    top1_accuracy: float
    topk_accuracy: float
    split: str

# Global variables to hold loaded model and dataset
bc_model = None
dataset_obj = None
device = None
app_args = None  # Stores the configuration arguments
symptoms_list = None  # List of all standard symptoms
questions_dict = {}  # List of all standard questions
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and dataset on startup."""
    global bc_model, dataset_obj, device, app_args, symptoms_list
    print("[INFO] Loading model and dataset...")

    # Parse arguments using default config
    args = argparse.Namespace(**DEFAULT_CONFIG)
    app_args = args  # Store globally for later use

    # Configure device
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = f"cuda:{args.gpu}"
            torch.cuda.set_device(args.gpu)
            print(f"[INFO] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"[WARNING] GPU {args.gpu} not available, using CPU")
            device = "cpu"
    else:
        device = None  # Let BehaviorCloning class auto-select

    # Load dataset
    try:
        dataset_obj = QuestionDataset(args)
        print("[INFO] Dataset loaded.")

        # Load standard symptoms list
        try:
            # Try to get symptoms from dataset object
            if hasattr(dataset_obj, 'symptoms'):
                symptoms_list = dataset_obj.symptoms
                print(f"[INFO] Loaded {len(symptoms_list)} standard symptoms from dataset.")
            else:
                # Fallback: load directly from symptoms.csv file
                symptom_df = pd.read_csv(args.symptom_file)
                symptoms_list = symptom_df["symptom"].tolist()
                print(f"[INFO] Loaded {len(symptoms_list)} standard symptoms from CSV file.")
        except Exception as e:
            print(f"[WARNING] Failed to load symptoms list: {e}")
            symptoms_list = []
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        raise

    # Load model
    try:
        bc_model = BehaviorCloning(
            dataset=dataset_obj,
            stage='test',
            vocab_size=dataset_obj.get_vocab_size(),
            action_dim=dataset_obj.get_action_dim(),
            args=args,
            device=device
        )
        bc_model.load(args.model_path)
        print(f"[INFO] Model loaded from {args.model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

    yield  # Server runs here

    # Cleanup on shutdown (if needed)
    print("[INFO] Shutting down...")

app = FastAPI(title="Medical Consultation Navigator API",
              description="API for Transformer-based symptom prediction",
              lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": bc_model is not None, "dataset_loaded": dataset_obj is not None}

@app.get("/symptoms")
async def get_symptoms():
    """Get all standard symptoms from symptoms.csv."""
    global symptoms_list
    if symptoms_list is None:
        raise HTTPException(status_code=503, detail="Symptoms list not loaded")
    return {
        "count": len(symptoms_list),
        "symptoms": symptoms_list
    }
@app.get("/questions")
async def get_questions():
    """Get all standard symptoms from questions_clearned.csv."""
    global questions_dict
    
    
    symptom_df = pd.read_csv(DEFAULT_CONFIG['question_file'])
    questions_raw_dict = symptom_df.to_dict(orient="records")
    
    for line in questions_raw_dict:
        print(line)
        if line['result'] in questions_dict:
            # questions_dict[line['result']].append(line['question'])
            pass
        else: 
            questions_dict[line['result']] = line
    print(f"[INFO] Loaded {len(questions_dict)} standard questions from CSV file.")
    if questions_dict is None:
        raise HTTPException(status_code=503, detail="Questions dict not loaded")
    return {
        "count": len(questions_dict),
        "questions": questions_dict
    }

@app.post("/predict", response_model=PredictResponse)
async def predict_symptom(request: PredictRequest):
    """
    Predict the next symptom given a sequence of symptoms.

    The input symptom list represents the current consultation session in chronological order.
    The model will treat the last symptom as the current focus, and all previous symptoms as history.
    """
    global bc_model, dataset_obj

    if bc_model is None or dataset_obj is None:
        raise HTTPException(status_code=503, detail="Model or dataset not loaded")

    # Validate input symptoms
    if len(request.symptoms) < 1:
        raise HTTPException(status_code=400, detail="At least one symptom is required")

    # Filter symptoms that are not in the dataset's vocabulary
    # Use question2idx since _build_state uses this mapping
    valid_symptoms = [s for s in request.symptoms if s in dataset_obj.question2idx]
    if len(valid_symptoms) < 1:
        raise HTTPException(status_code=400, detail="No valid symptoms found in vocabulary")

    # If there's only one symptom, we can't predict next (need at least one as history and one as focus)
    # However, the model can still predict from a single symptom (history empty, focus = that symptom)
    # We'll follow the evaluation logic: history_before_focus = [], current_focus = first symptom
    # But we need to predict the next symptom after the current focus.
    # Since we only have one symptom, we can treat it as current focus and predict what would come next.
    # This matches the evaluation where sequences can start with first symptom as focus.

    # Determine history and current focus
    # In evaluation, for a sequence [s1, s2, s3, ...], they start with:
    #   history_before_focus = [], current_focus = s1
    #   predict s2, then update: history_before_focus = [s1], current_focus = s2
    #   predict s3, etc.
    # For our API, we'll treat the entire input list as the sequence seen so far.
    # The last symptom is the current focus, all previous symptoms are history.
    # This matches the stepwise prediction at the last step.

    if len(valid_symptoms) == 1:
        # Only one symptom: history empty, focus = that symptom
        history_before_focus = []
        current_focus = valid_symptoms[0]
    else:
        # Multiple symptoms: all but last are history, last is current focus
        history_before_focus = valid_symptoms[:-1]
        current_focus = valid_symptoms[-1]

    # Build state indices (consistent with training)
    try:
        state_indices = dataset_obj._build_state(history_before_focus, current_focus)
    except AttributeError:
        raise HTTPException(status_code=500, detail="Dataset missing _build_state method")

    # Convert to tensor and add batch dimension
    state_tensor = torch.LongTensor(state_indices).unsqueeze(0)  # [1, seq_len]

    # Build state mask (no padding for single sample)
    state_mask = torch.zeros(1, len(state_indices), dtype=torch.bool)

    # Get prediction
    bc_model.policy.eval()
    with torch.no_grad():
        # Get Top-K predictions (similar to evaluate_sequence_prediction)
        logits = bc_model.policy(state_tensor.to(bc_model.device),
                                 src_key_padding_mask=state_mask.to(bc_model.device))
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()  # [action_dim]

        # Get Top-K
        topk = min(request.topk, len(probs))
        topk_values, topk_idxs = torch.topk(probs, topk)

        # Convert indices to symptom names
        topk_symptoms = [dataset_obj.idx2question.get(idx.item(), "") for idx in topk_idxs]
        topk_probs = topk_values.tolist()

        # Top-1 prediction
        pred_action_idx = torch.argmax(probs).item()
        predicted_symptom = dataset_obj.idx2question.get(pred_action_idx, "")

    return PredictResponse(
        predicted_symptom=predicted_symptom,
        topk_symptoms=topk_symptoms,
        topk_probabilities=topk_probs,
        history_before_focus=history_before_focus,
        current_focus=current_focus
    )

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate the model on a dataset split (train, val, or all).
    This replicates the evaluation logic from evaluate.py.
    """
    global bc_model, dataset_obj, app_args

    if bc_model is None or dataset_obj is None or app_args is None:
        raise HTTPException(status_code=503, detail="Model, dataset, or configuration not loaded")

    # Load dataset split
    try:
        train_dataset, val_dataset = dataset_obj.split_dataset(
            app_args, app_args.splitted_dataset_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset split: {e}")

    # Select dataset based on split
    if request.eval_split == "val":
        eval_dataset = val_dataset
    elif request.eval_split == "train":
        eval_dataset = train_dataset
    elif request.eval_split == "all":
        eval_dataset = dataset_obj
    else:
        raise HTTPException(status_code=400, detail="eval_split must be 'train', 'val', or 'all'")

    # Run evaluation (simplified version of evaluate_sequence_prediction)
    bc_model.policy.eval()

    hit1 = 0
    hitk = 0
    total = 0

    # Get original dataset (if Subset, access .dataset)
    if hasattr(eval_dataset, 'dataset'):
        original_dataset = eval_dataset.dataset
        indices = set(eval_dataset.indices)
        # Collect evaluation uuids (extracted from transitions)
        eval_uuids = set()
        for idx in indices:
            if idx < len(original_dataset.transitions):
                eval_uuids.add(original_dataset.transitions[idx]['uu_id'])
    else:
        original_dataset = eval_dataset
        eval_uuids = None  # use all data

    df = original_dataset.df.copy()
    if eval_uuids is not None:
        df = df[df['uu_id'].isin(eval_uuids)]

    for uu_id, group in df.groupby("uu_id", sort=False):
        seq = group["symptom"].tolist()
        # Filter: keep only symptoms present in symptom2idx (used by _build_state)
        seq = [s for s in seq if s in original_dataset.symptom2idx]

        if len(seq) < 2:
            continue

        # Initialization: start from the first symptom
        history_before_focus = []  # Symptoms before t are history
        current_focus = seq[0]     # current focus symptom

        # Stepwise prediction
        for t in range(len(seq) - 1): # seq[t] is the current focused symptom
            true_next = seq[t + 1]  # ground-truth next symptom

            # Build state (consistent with training)
            state_indices = original_dataset._build_state(history_before_focus, current_focus)

            # Convert to tensor and add batch dimension
            state_tensor = torch.LongTensor(state_indices).unsqueeze(0)  # [1, seq_len]

            # Build state mask: for a single sample there are no padding positions, so all False
            state_mask = torch.zeros(1, len(state_indices), dtype=torch.bool)

            # Get prediction
            with torch.no_grad():
                logits = bc_model.policy(state_tensor.to(bc_model.device),
                                         src_key_padding_mask=state_mask.to(bc_model.device))
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu()  # [action_dim]

                # Get Top-K (no action masking applied)
                topk_values, topk_idxs = torch.topk(probs, min(request.topk, len(probs)))

            # Check Top-1
            pred_action_idx = torch.argmax(probs).item()
            pred_symptom = original_dataset.idx2question.get(pred_action_idx, "")
            if pred_symptom == true_next:
                hit1 += 1

            # Check Top-K
            topk_symptoms = [original_dataset.idx2question.get(idx.item(), "") for idx in topk_idxs]
            if true_next in topk_symptoms:
                hitk += 1

            total += 1

            # Update state: append current symptom to history and set the next symptom as current focus
            history_before_focus.append(current_focus)
            current_focus = true_next

    # Calculate accuracies
    acc1 = hit1 / total if total > 0 else 0.0
    acck = hitk / total if total > 0 else 0.0

    return EvaluationResponse(
        total_transitions=total,
        top1_accuracy=acc1,
        topk_accuracy=acck,
        split=request.eval_split
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSDF-Navigator API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8999, help="Port to listen on")
    parser.add_argument("--question-file", type=str, default=DEFAULT_CONFIG["question_file"],
                        help="Path to question_cleaned.csv")
    parser.add_argument("--symptom-file", type=str, default=DEFAULT_CONFIG["symptom_file"],
                        help="Path to symptoms.csv")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG["model_path"],
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_CONFIG["splitted_dataset_path"],
                        help="Path to dataset split file (data_split.pkl)")
    parser.add_argument("--topk", type=int, default=DEFAULT_CONFIG["topk"],
                        help="Number of top-K predictions")
    args = parser.parse_args()

    # Override DEFAULT_CONFIG with CLI arguments
    DEFAULT_CONFIG["question_file"] = args.question_file
    DEFAULT_CONFIG["symptom_file"] = args.symptom_file
    DEFAULT_CONFIG["model_path"] = args.model_path
    DEFAULT_CONFIG["splitted_dataset_path"] = args.dataset_split
    DEFAULT_CONFIG["topk"] = args.topk

    uvicorn.run("navigator_api_service:app", host=args.host, port=args.port, reload=False)