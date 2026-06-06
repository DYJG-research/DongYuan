"""
Behavior Cloning (BC) - Behavior Cloning
Using Transformer architecture
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from policy_network import TransformerPolicyNetwork
import statistics

class BehaviorCloning:
    """
    Transformer-based Behavior Cloning algorithm.
    Trained with a combination of supervised and reinforcement signals.
    Supports two loss fusion modes.
    """
    def __init__(
        self,
        dataset: Dataset,
        device: str,
        args,
        vocab_size: int,
        action_dim: int,
        stage: str ='train'
    ):
        """
        Args:
            dataset: training dataset
            stage: stage, "train" or "test"
            vocab_size: number of symptoms
            action_dim: action space dimensionality
            learning_rate: learning rate
            d_model: Transformer model dimension
            nhead: number of attention heads
            num_layers: number of Transformer layers
            dim_feedforward: feedforward network hidden dimension
            dropout: dropout rate
            device: device
            fusion_mode: loss fusion method, multiplication ("mul") or addition ("add")
            lambda_supervised: weight for supervised loss in "add" mode (default 1.0)
            lambda_reinforce: weight for reinforcement loss in "add" mode (default 0.5)
            entropy_coef: entropy regularization coefficient (used in "add" mode) to encourage exploration (default 0.01)
            use_baseline: whether to use a baseline to reduce reward variance (used in "add" mode)
            reward_clip: reward clipping threshold to prevent outliers (default 5.0)
            use_reward_norm: whether to perform batch normalization on rewards
        """
        self.device = device
        self.stage = stage
        if self.stage == 'train':
            self.fusion_mode = args.fusion_mode
            self.lambda_supervised = args.lambda_supervised
            self.lambda_reinforce = args.lambda_reinforce
            self.entropy_coef = args.entropy_coef
            self.reward_clip = args.reward_clip
            self.use_reward_norm = args.use_reward_norm
            self.use_baseline = args.use_baseline

            # Store historical rewards (for computing moving-average baseline)
            self.reward_history = []
            self.baseline = 0.0
            self.baseline_alpha = 0.1  # Baseline update coefficient
            
            print(f"[INFO] Hybrid BC initialized with fusion_mode='{self.fusion_mode}'")
        
        self.dataset = dataset
        

        # Policy network
        self.policy = TransformerPolicyNetwork(
            vocab_size=vocab_size,
            action_dim=action_dim,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=args.lr,
            weight_decay=1e-4 # Weight decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=1e-6
        )
    
    def compute_batch_rewards(self,preds,batch):
        """
        Compute rewards for a batch of samples.
        Here, rewards are computed using actions sampled from model predictions (pred_question).

        """
        batch_rewards = []
        
        # Get raw data for a batch
        for i in range(len(batch["action"])):
            # The batch contains the following raw fields, accessible directly
            current_idx = batch["current_idxs"][i] if "current_idxs" in batch else None
            history_padded = batch["history"][i] if "history" in batch else None
            history = history_padded[history_padded!=-1]
            pred=preds[i]
            
            reward = self.dataset._compute_reward(
                current_idx.item(),
                pred.item(),
                history
            )
            batch_rewards.append(reward)
            
        
        return batch_rewards
    
    def compute_loss(self, states, actions, rewards, masks=None):
        """
        Compute loss according to the fusion mode.
        
        Args:
            states: history state sequences for a batch, shape (batch_size, seq_len)
            actions: ground-truth next actions (expert actions), shape (batch_size,)
            rewards: reward values, shape (batch_size,)
            masks: attention mask, shape (batch_size, seq_len)
        
        Returns:
            A dictionary of loss values with the following fields:
            "final_loss": final fused loss for the batch
            "supervised_loss": supervised loss
            "reinforce_loss": reinforcement loss
            "accuracy": accuracy on the current batch
            "avg_reward": final average reward
            "avg_raw_reward": raw average reward (equals avg_reward if reward normalization is not used)
        """
        batch_size = states.size(0)
        
        # 1. Forward pass
        logits = self.policy(states, src_key_padding_mask=masks)  # (batch_size, action_dim)
        
        # 2. Process rewards (clipping and normalization)
        rewards = rewards.clone()
        if self.reward_clip > 0:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        
        if self.use_reward_norm and batch_size > 1:
            # Batch normalization: subtract mean and divide by std (keeps values positive)
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards = (rewards - reward_mean) / reward_std
            # Optionally scale back to positive range
            rewards = torch.relu(rewards + 1.0)  # Most values will be around [0, 2]
        
        # 3. Compute loss according to fusion mode
        if self.fusion_mode == "mul":
            # === Multiplicative scheme: weighted cross-entropy ===
            # Compute cross-entropy loss per sample (no reduction)
            log_probs = F.log_softmax(logits, dim=-1)
            nll_loss = F.nll_loss(log_probs, actions, reduction='none')  # (batch_size,)
            
            # Use rewards as weights (assumes rewards are positive; higher reward -> higher weight)
            weighted_loss = nll_loss * rewards
            final_loss = weighted_loss.mean()
            
            # Record loss components
            supervised_loss = nll_loss.mean()
            reinforce_loss = torch.tensor(0.0)  # No separate reinforcement loss
        
        elif self.fusion_mode == "add":
            # === Additive scheme: supervised loss + policy gradient loss ===
            # Supervised loss
            supervised_loss = F.cross_entropy(logits, actions)
            
            # Policy gradient loss
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            if self.use_baseline:   # Update baseline (moving average)
                with torch.no_grad():
                    batch_mean_reward = rewards.mean().item()
                    self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * batch_mean_reward
                    advantages = rewards - self.baseline
            else:
                advantages = rewards
            
            reinforce_loss = -(action_log_probs * advantages).mean()
            

            # Entropy regularization (encourage exploration)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean() # Mean entropy
            entropy_penalty = -self.entropy_coef * entropy # Negative sign because we maximize entropy
            
            # Final loss
            final_loss = (
                self.lambda_supervised * supervised_loss +
                self.lambda_reinforce * reinforce_loss +
                entropy_penalty
            )
        
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        # 4. Compute additional statistics
        with torch.no_grad():
            # Prediction accuracy
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == actions).float().mean()
            
            # Average reward
            avg_reward = rewards.mean()
        
        return {
            "final_loss": final_loss,
            "supervised_loss": supervised_loss,
            "reinforce_loss": reinforce_loss if self.fusion_mode == "add" else torch.tensor(0.0),
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "avg_raw_reward": reward_mean if self.use_reward_norm else avg_reward,
        }
    
    def train_step(self, states, actions, rewards, masks=None):
        """Perform a single training step"""
        
        # Compute loss during forward pass
        losses = self.compute_loss(states, actions, rewards, masks)
        
        # Backpropagation
        self.optimizer.zero_grad()
        losses["final_loss"].backward()
        
        # Gradient clipping (to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def train(self, dataloader, epochs ,val_loader,args):
        """
        Train the policy network
        Args:
            dataloader: data loader
            epochs: number of training epochs
        """
        self.policy.train()
        
        # Update scheduler T_max
        self.scheduler.T_max = epochs * len(dataloader)

        end_ep = 0  # Record the epoch at which training stopped
        best_accuracy = 0 
        for epoch in range(epochs): # Training loop
        
            epoch_losses = []
            epoch_accs = []
            epoch_rewards = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                states = batch["state"].to(self.device)
                actions = batch["action"].squeeze(-1).to(self.device)
                masks = batch["state_mask"].to(self.device)
                sampled_question = self.policy.sample_action(states,src_key_padding_mask=masks)
                rewards = self.compute_batch_rewards(sampled_question,batch)
                rewards = torch.FloatTensor(rewards).to(self.device)
                
                # Train one step
                losses = self.train_step(states,actions,rewards,masks)
                curr_loss = losses["final_loss"]
                curr_acc = losses["accuracy"]
                curr_rewards = losses["avg_raw_reward"]
                epoch_losses.append(curr_loss)
                epoch_accs.append(curr_acc)
                epoch_rewards.append(curr_rewards)
                current_lr = self.scheduler.get_last_lr()[0]
                # Print training statistics per batch/iteration
                pbar.set_postfix({
                            "curr_loss": f"{curr_loss:.4f}",
                            "curr_acc": f"{curr_acc:.4f}",
                            "curr_rewards": f"{curr_rewards:.4f}",
                            "lr": f"{current_lr:.6f}"
                        })
            
            # Print training statistics per epoch
            avg_loss = statistics.mean(epoch_losses)
            avg_accs = statistics.mean(epoch_accs)
            avg_rewards = statistics.mean(epoch_rewards)
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accs:.4f}") 
            print(f"Average Rewards: {avg_rewards:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}") 
            end_ep = epoch + 1
            # Evaluate current model on validation set
            val_accuracy = self.val(val_loader)

            # Save model if validation performance improves
            if val_accuracy > best_accuracy :
                best_accuracy = val_accuracy
                no_improve_epochs = 0  # Reset no-improvement counter
                self.save(args.save_path)
                print(f"[INFO] New best model saved to {args.save_path} with Validation accuracy: {best_accuracy:.4f}")
            else:
                print(f"[INFO] Validation accuracy is not improved: {best_accuracy:.4f}.Don't save model")
                if val_accuracy !=0.0:
                    no_improve_epochs += 1
                else:
                    no_improve_epochs = 0  # Reset no-improvement counter
            # Check early stopping condition
            if no_improve_epochs >= args.patience:
                print(f"[INFO] Early stopping triggered after {epoch + 1} epochs.")
                break

        # Warn if maximum epochs reached without convergence
        if end_ep == args.epochs:
            print(f"[WARNING] Training reached MAX_EPOCHS ({args.epochs}) without convergence.")

            
    def val(self, val_loader):
        """
        Validate model at the end of each epoch during training
        """
        print("Validating...")
        self.policy.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"]
                masks = batch["state_mask"]  # Extract mask
                actions = batch["action"].squeeze(-1)
                
                # Pass mask to predict
                pred_actions = self.predict(states, mask=masks)
                correct += (pred_actions == actions).sum().item()
                total += actions.size(0)
        
        val_accuracy = correct / total
        print(f"[INFO] Validation Accuracy: {val_accuracy:.4f} ({correct}/{total})")
        return val_accuracy
    
    def predict(self, state, mask=None, temperature=1.0):
        """
        Predict actions (supports temperature sampling)
        Args:
            state: [batch_size, seq_len] index sequences
            mask: [batch_size, seq_len], True denotes masked (padding) positions
            temperature: temperature parameter
        Returns:
            action: [batch_size] action indices (on CPU)
        """
        self.policy.eval()
        state = state.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            
        with torch.no_grad():
            # Important: pass mask!
            probs = self.policy.get_action_probs(state, mask)
            if temperature != 1.0:
                probs = probs / temperature
            
            action = torch.argmax(probs, dim=-1)
        
        return action.cpu()



    def save(self, path: str):
        """Save model"""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"[INFO] Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[INFO] Model loaded from {path}")

