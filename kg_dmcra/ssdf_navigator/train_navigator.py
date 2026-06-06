"""
Training script: Transformer-based supervised + reinforcement hybrid training
"""
import argparse
import torch
from torch.utils.data import DataLoader
from data_loader import QuestionDataset,collate_fn
from behavior_cloning import BehaviorCloning


def main():
    parser = argparse.ArgumentParser(description="Train Transformer-Based Medical Consultation Navigator")
    # data related
    parser.add_argument("--question_file", type=str, default="./question_cleaned.csv",help="Question CSV file path")
    parser.add_argument("--symptom_file", type=str,default="./symptoms.csv",help="Symptom CSV file path")
    
    # State representation related
    parser.add_argument("--state_type", type=str, default="transformer_seq",choices=["transformer_seq"],help="State representation type")
    parser.add_argument("--max_seq_len", type=int, default=10,help="The max length of sequence length")

    # offline reinforcement learning related
    parser.add_argument("--reward_type", type=str, default="information_gain",choices=["information_gain"],help="Reward type")
    parser.add_argument("--reward_clip", type=float, default=5.0,help="Reward clipping factor")
    parser.add_argument("--use_reward_norm", type=bool, default=True,help="Whether to use  batch normalization for reward")
    parser.add_argument("--use_baseline", type=bool, default=True,help="Whether to caculate and remove reward baseline")
    parser.add_argument("--use_repeat_factor", type=bool, default=True,help="Whether to use repeat factor")


    # loss function related
    parser.add_argument("--fusion_mode", type=str, default="mul",choices=["mul","add"],help="Loss function fusion method")
    parser.add_argument("--entropy_coef", type=float, default=0.01,help="Entropy regularization coefficient (for add mode)")
    parser.add_argument("--lambda_supervised", type=float, default=1.0,help="Weight coefficient for supervision loss (used in add mode)")
    parser.add_argument("--lambda_reinforce", type=float, default=0.5,help="Weight coefficient for reinforcement loss (used in add mode)")
    
    # transformer structure related
    parser.add_argument("--d_model", type=int, default=256,help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=8,help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3,help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=512,help="Feedforward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1,help="Dropout rate")
    
    # trainning setting related
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Transformer models typically require a smaller learning rate)")
    parser.add_argument("--save_path", type=str, default="transformer_policy_mul3.pth",help="Path to save model")
    parser.add_argument("--train_ratio", type=float, default=0.8,help="Training data ratio")
    parser.add_argument("--gpu", type=int, default=None,help="GPU device ID (e.g., 0, 1). If not specified, auto-select")
    parser.add_argument("--topk", type=int, default=5,help="Top-K for validation")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    
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
    print("=" * 60)
    print("Loading dataset...")
    dataset = QuestionDataset(args)

    train_dataset,val_dataset=dataset.split_dataset(args)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn # Use custom collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    
    # 3. Initialize model
    print("=" * 60)
    print("Initializing Transformer model...")
    print(f"[INFO] Model config: d_model={args.d_model}, nhead={args.nhead}, "
          f"num_layers={args.num_layers}, dim_feedforward={args.dim_feedforward}")
    
    bc = BehaviorCloning(
        dataset = dataset,
        vocab_size=dataset.get_vocab_size(),
        action_dim=dataset.get_action_dim(),
        args=args,
        device=device
    )
    
    print(f"[INFO] Vocabulary size: {dataset.get_vocab_size()}")
    print(f"[INFO] Action dim: {dataset.get_action_dim()}")
    
    # Compute model parameter counts
    total_params = sum(p.numel() for p in bc.policy.parameters())
    trainable_params = sum(p.numel() for p in bc.policy.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    # 4. Train
    print("=" * 60)
    print("Training...")
    bc.train(dataloader=train_loader, epochs=args.epochs,val_loader=val_loader,args=args)
    
    print("=" * 60)
    print("Training completed!")


if __name__ == "__main__":
    main()

