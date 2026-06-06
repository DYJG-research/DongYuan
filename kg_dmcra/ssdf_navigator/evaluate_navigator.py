"""
Evaluation script: Evaluate trained Transformer policy
"""
import argparse
import torch
from data_loader import QuestionDataset
from behavior_cloning import BehaviorCloning
import torch.nn.functional as F

def evaluate_sequence_prediction(bc: BehaviorCloning,dataset,topk: int,device: str):
    """
    Evaluate sequence prediction performance.
    Directly iterate through sequences for evaluation.
    Args:
        bc: trained BC model
        dataset: dataset (QuestionDataset or Subset)
        topk: Top-K value
        device: device
    """
    bc.policy.eval()
    
    hit1 = 0
    hitk = 0
    total = 0
    
    # Get original dataset (if Subset, access .dataset)
    if hasattr(dataset, 'dataset'):
        original_dataset = dataset.dataset
        indices = set(dataset.indices)
        # Collect evaluation uuids (extracted from transitions)
        eval_uuids = set()
        for idx in indices:
            if idx < len(original_dataset.transitions):
                eval_uuids.add(original_dataset.transitions[idx]['uu_id'])
    else:
        original_dataset = dataset
        eval_uuids = None  # use all data
    df = original_dataset.df.copy()
    if eval_uuids is not None:
        df = df[df['uu_id'].isin(eval_uuids)]
    
    for uu_id, group in df.groupby("uu_id", sort=False):
        seq = group["symptom"].tolist()
        # Filter: keep only symptoms present in symptom2idx
        seq = [s for s in seq if s in original_dataset.symptom2idx]
        
        if len(seq) < 2:
            continue
        
        # Initialization: start from the first symptom
        history_before_focus = []  # Symptoms before t are history
        current_focus = seq[0]     # current focus symptom
        
        # Stepwise prediction: same transition logic as training
        for t in range(len(seq) - 1): # seq[t] is the current focused symptom
            true_next = seq[t + 1]  # ground-truth next symptom
            
            # Build state (consistent with training)
            state_indices = original_dataset._build_state(history_before_focus, current_focus)
            
            # Convert to tensor and add batch dimension
            state_tensor = torch.LongTensor(state_indices).unsqueeze(0)  # [1, seq_len]
            
            # Build state mask: for a single sample there are no padding positions, so all False
            state_mask = torch.zeros(1, len(state_indices), dtype=torch.bool)
            
            pred_action_idx = bc.predict(
                state_tensor, 
                mask=state_mask,  # state mask
            )
            
            # If pred_action_idx is a tensor, ensure it's an int
            if isinstance(pred_action_idx, torch.Tensor):
                pred_action_idx = pred_action_idx.item()
            
            # Get predicted symptom name
            pred_symptom = original_dataset.idx2question.get(pred_action_idx, "")
            
            # Compute Top-K predictions
            with torch.no_grad():
                # Get probability distribution
                logits = bc.policy(state_tensor.to(device), src_key_padding_mask=state_mask.to(device))
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu()  # [action_dim]
                
                # Get Top-K (no action masking applied)
                topk_values, topk_idxs = torch.topk(probs, min(topk, len(probs)))
            
            # Check Top-1
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
    
    print("=" * 60)
    print(f"Total transitions: {total}")
    if total > 0:
        acc1 = hit1 / total
        acck = hitk / total
        print(f"Top-1 Accuracy: {hit1}/{total} = {acc1:.4f}")
        print(f"Top-{topk} Accuracy: {hitk}/{total} = {acck:.4f}")
    else:
        acc1 = acck = 0.0
        print("No transitions to evaluate!")
    print("=" * 60)
    
    return acc1, acck


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer-Based Medical Consultation Navigator")
    # The following arguments must match training settings
    parser.add_argument("--question_file", type=str,default="./question_cleaned.csv",help="Question CSV file path")
    parser.add_argument("--symptom_file", type=str,default="./symptoms.csv",help="Symptom CSV file path")
    parser.add_argument("--state_type", type=str, default="transformer_seq",choices=["transformer_seq"],help="State representation type")
    parser.add_argument("--d_model", type=int, default=256,help="Transformer model dimension ")
    parser.add_argument("--nhead", type=int, default=8,help="Number of attention heads ")
    parser.add_argument("--num_layers", type=int, default=3,help="Number of transformer layers ")
    parser.add_argument("--dim_feedforward", type=int, default=512,help="Feedforward network dimension ")
    parser.add_argument("--dropout", type=float, default=0.1,help="Dropout rate ")
    parser.add_argument("--max_seq_len", type=int, default=10,help="The max length of sequence length ")
    parser.add_argument("--train_ratio", type=float, default=0.8,help="Training data ratio ")
    parser.add_argument("--gpu", type=int, default=None,help="GPU device ID (e.g., 0, 1). If not specified, auto-select")
    parser.add_argument("--reward_type", type=str, default="information_gain",choices=["information_gain"],help="Reward type")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    parser.add_argument("--model_path", type=str, default="transformer_policy_add_no_repeat_factor.pth",help="Path to trained model")
    parser.add_argument("--splitted_dataset_path", type=str, default="./data_split.pkl",help="Path to splitted dataset")
    parser.add_argument("--topk", type=int, default=5,help="Top-K for evaluation")  
    parser.add_argument("--eval_split", type=str, default="val",choices=["train", "val", "all"],help="Which split to evaluate: train (training), val (validation), all (all data)")
    
    
    args = parser.parse_args()
    
    # Configure GPU device
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
    
    # 1. Load data
    print("Loading dataset...")
    dataset = QuestionDataset(args)
    
    # 2. Load dataset split information (consistent with training)
    train_dataset,val_dataset=dataset.split_dataset(args,args.splitted_dataset_path)

    # 3. Load model
    print("Loading model...")
    bc = BehaviorCloning(
        dataset = dataset,
        stage='test',
        vocab_size=dataset.get_vocab_size(),
        action_dim=dataset.get_action_dim(),
        args=args,
        device=device
    )
    bc.load(args.model_path)
    
    # 4. Evaluate
    print("=" * 60)
    if args.eval_split == "val":
        print("Evaluating on VALIDATION set...")
        evaluate_sequence_prediction(bc, val_dataset, topk=args.topk,device=device)
    elif args.eval_split == "train":
        print("Evaluating on TRAIN set...")
        evaluate_sequence_prediction(bc, train_dataset, topk=args.topk,device=device)
    else:
        print("Evaluating on ALL data...")
        evaluate_sequence_prediction(bc, dataset, topk=args.topk,device=device)


if __name__ == "__main__":
    main()

