"""
Offline RL data loader
Builds transition dataset from question.csv
Supports Transformer architecture
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import pickle
import torch
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import math

def collate_fn(batch):
    """
    Custom collate function: dynamically pads variable-length sequences within a batch.
    """
    # Find the longest sequence length in the batch
    max_len = max(item['state_len'] for item in batch)
    max_history = max(item['history_len'] for item in batch)
    # Initialize padded tensors
    batch_size = len(batch)
    
    # 1. Pad states and history
    padded_states = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long) # Pad with 0
    state_masks = torch.zeros((batch_size, max_len), dtype=torch.bool) # Attention mask
    padded_historys = torch.full((batch_size, max_history), fill_value=-1, dtype=torch.long) # Pad history with -1
    # 2. Handle actions, rewards, etc.
    actions = []
    current_symptoms = []
    current_idxs=[]
    dones = []
    uu_ids = []


    for i, item in enumerate(batch):
        seq = item['state']
        seq_len = item['state_len']
        history_seq = item['history_idxs']
        history_len = item['history_len']
        # Pad state and history sequences
        padded_states[i, :seq_len] = torch.LongTensor(seq)
        padded_historys[i,:history_len] = torch.LongTensor(history_seq)
        
        # Create attention mask: False for real tokens (no masking), True for padding positions (masked)
        state_masks[i, seq_len:] = True  # Only padding positions are masked
        
        # Collect other information
        actions.append(item['action'])
        dones.append(item['done'])
        current_symptoms.append(item['current_symptom'])
        current_idxs.append(item['current_idx'])
        uu_ids.append(item['uu_id'])
        
    
    
    return {
        'state': padded_states,        # (batch_size, max_seq_len_in_this_batch)
        'state_mask': state_masks,     # (batch_size, max_seq_len_in_this_batch)
        'action': torch.cat(actions, dim=0),  # (batch_size,)
        'current_symptom':current_symptoms,  # (batch_size,)
        'current_idxs':torch.cat(current_idxs, dim=0),# (batch_size,)
        'done': torch.cat(dones, dim=0),      # (batch_size,)
        'history':padded_historys, 
        'uu_id':uu_ids
    }



class QuestionDataset(Dataset):
    """Offline RL dataset"""
    
    def __init__(self,args):
        """
        Args used:
            question_file: path to question.csv
            symptom_file: path to symptom.csv
            state_type: state representation type
            max_seq_len: maximum sequence length (for sequence-type states)
            reward_type: reward calculation method
        """
        self.state_type = args.state_type
        self.max_seq_len = args.max_seq_len
        self.reward_type = args.reward_type
        self.args = args
        # Load data
        self.df = pd.read_csv(args.question_file) 
        self.df["uu_id"] = self.df["uu_id"].astype(str)
        
        # Load symptom list
        symptom_df = pd.read_csv(args.symptom_file)
        self.symptoms = symptom_df["symptom"].tolist()
        self.symptom2idx = {s: i for i, s in enumerate(self.symptoms)}
        
        # Build mapping from question to index
        self._build_question_mapping()
        
        # Build offline transition dataset
        self.transitions = self._build_transitions()
        self._precompute_conditional_entropy() # Precompute statistics used for reward calculation

        print(f"[INFO] Loaded {len(self.transitions)} transition samples")
        print(f"[INFO] Number of questions (symptoms): {len(self.question2idx)}")
        print(f"[INFO] State type: {args.state_type}, Reward type: {args.reward_type}")
    

    def _precompute_conditional_entropy(self, smoothing=1.0):
        """Precompute normalized conditional entropy H(Y|X) for each symptom"""
        
        # Count symptom transition frequencies
        transition_counts = defaultdict(lambda: defaultdict(int))
        symptom_counts = defaultdict(int)
        
        # Iterate over all patient cases
        for _, group in self.df.groupby("uu_id", sort=False):
            seq = [s for s in group["symptom"].tolist() if s in self.symptom2idx]
            
            for i in range(len(seq) - 1):
                x = seq[i]
                y = seq[i + 1]
                
                x_idx = self.question2idx[x]
                y_idx = self.question2idx[y]
                
                transition_counts[x_idx][y_idx] += 1
                symptom_counts[x_idx] += 1
        
        # Compute normalized conditional entropy
        self.conditional_entropy = {}
        num_symptoms = len(self.symptoms)
        
        # Theoretical maximum entropy (natural log base e, consistent with math.log usage)
        max_entropy = math.log(num_symptoms)  # natural logarithm (consistent with math.log usage)
        
        for x_idx in range(num_symptoms):
            total = symptom_counts.get(x_idx, 0)
            if total == 0:
                self.conditional_entropy[x_idx] = 1.0  # set to normalized maximum entropy
                continue
            
            entropy = 0.0
            # Use Laplace smoothing (add 'smoothing')
            for y_idx in range(num_symptoms):
                count = transition_counts[x_idx].get(y_idx, 0)
                prob = (count + smoothing) / (total + smoothing * num_symptoms)
                if prob > 0:
                    entropy -= prob * math.log(prob)  # natural logarithm
            
            # Normalize: divide entropy by max_entropy to obtain a value in [0,1]
            # Note: due to smoothing, normalized entropy may slightly exceed 1; clip to [0,1]
            normalized_entropy = entropy / max_entropy
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))  # clip to [0,1]
            
            self.conditional_entropy[x_idx] = normalized_entropy
    
        # Print statistics
        entropy_values = list(self.conditional_entropy.values())
        print(f"[INFO] Normalized conditional entropy stats:")
        print(f"  Min: {min(entropy_values):.4f}")
        print(f"  Max: {max(entropy_values):.4f}")
        print(f"  Mean: {np.mean(entropy_values):.4f}")
        print(f"  Proportion entropy>0.9: {sum(1 for v in entropy_values if v > 0.9) / len(entropy_values):.4f}")

    def _build_question_mapping(self):
        """Build mapping from questions to indices"""
        # Use 'symptom' column as questions (action space)
        # Only use symptoms defined in symptom.csv
        unique_questions = [s for s in self.symptoms if s in self.df["symptom"].values]
        self.question2idx = {q: i for i, q in enumerate(unique_questions)}
        self.idx2question = {i: q for q, i in self.question2idx.items()}
        self.num_questions = len(unique_questions)
    
    def _build_transitions(self) -> List[Dict]:
        """
        Build transition samples: (state, action, history_idxs, done, uu_id, current_symptom)
            - state (state): previously asked history symptoms + current focus symptom (most recent)
            - action (action): next symptom to query
            - current_symptom: name of the current focus symptom
            - current_idx: index of the current focus symptom
            - history_idxs: history symptoms before the current focus (excluding current)
        """
        transitions = []
        
        # Group by case (uu_id denotes different conversation segments)
        for uu_id, group in self.df.groupby("uu_id", sort=False):
            # Use 'symptom' column as sequence of questions
            seq = group["symptom"].tolist()
            
            # Filter: keep only symptoms present in symptom2idx
            seq = [s for s in seq if s in self.symptom2idx]
            
            if len(seq) < 2: # Need at least two symptoms to build a transition
                continue
            
            # Initial state: empty history, first symptom as current focus
            # Note: state includes the current focused symptom
            for t in range(len(seq)-1):
                current_focus = seq[t] # Current focused symptom
                next_action = seq[t + 1]  # Next action (next symptom to ask)
                action_idx = self.question2idx.get(next_action, -1)
                current_idx = self.question2idx.get(current_focus, -1)

                # Build current state s_t: history is seq[:t], current focus is current_focus
                history_before_focus = seq[:t]  # Symptoms before t are history
                state = self._build_state(history_before_focus, current_focus)
    
                done = (t + 1 == len(seq) - 1) # Flag whether it's terminal

                # Build history indices
                history_idxs = [self.question2idx.get(symptom, -1) for symptom in history_before_focus]
                
                transitions.append({
                    "state": state,
                    "action": action_idx,
                    "done": done,
                    "uu_id": uu_id,
                    "current_symptom": current_focus, 
                    "current_idx": current_idx,
                    "history_idxs":history_idxs
                })
        
        return transitions
    
    def _build_state(self, history: List[str], current_question: str) -> np.ndarray:
        """
        Build state representation
        
        Args:
            history: list of previous questions
            current_question: current question
        
        Returns:
            state_indices: variable-length list of indices (history + current question)
        """
        if self.state_type == "transformer_seq":
            # Return symptom index sequence for Transformer embedding
            # Sequence = [indices of recent N history symptoms] + [index of current symptom]
            # This strictly preserves temporal order and repetition.
            window_size = self.max_seq_len-1  # Reuse parameter to control history length; reserve 1 slot for current focus
            recent_history = history[-window_size:]  # Keep the most recent N items
            
            # Convert symptom names to indices
            state_indices = []
            for symptom in recent_history:
                if symptom in self.question2idx:
                    state_indices.append(self.question2idx[symptom])
            # Append current symptom index
            if current_question in self.question2idx:
                state_indices.append(self.question2idx[current_question])
            
            # Return list of indices. Converted to Tensor in __getitem__.
            # Note: this returns a variable-length list, not a fixed-size vector.
            return state_indices 
        
        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")
    
    def _compute_reward(self, current_idx, pred_idx, history) :
        """
        Compute reward for a single sample - "base information gain" combined with a "repeat factor".
        Characteristics: depends only on current state, predicted action and history; not on future states.
        """
        if self.reward_type == "information_gain":
            
            # ===== 1. Base information gain reward =====
            # Conditional entropy of symptoms: uncertainty of transitioning from X to other symptoms
            base_entropy = self.conditional_entropy.get(pred_idx, 1.0)
            base_reward = 1.0 - base_entropy  # Lower entropy => higher reward
            
            # ===== 2. Repeat reward factor =====
            repeat_factor = 1.0
            
            if (history is not None and len(history) > 0) and self.args.use_repeat_factor:
                # a) Immediate repeat: predicted == current focus -> penalize
                if pred_idx == current_idx:
                    repeat_factor = 0.3  # Penalize immediate repeat
                
                # b) Non-consecutive repeat: predicted symptom appeared in history -> reward
                elif pred_idx in history:
                    repeat_factor = 1.5
    
            # ===== 3. Combine reward =====
            # Multiply base information gain by repeat factor
            final_reward = base_reward * repeat_factor
            
            # ===== 4. Normalization and scaling =====
            # Theoretical ranges: base_reward ∈ [0,1], repeat_factor ∈ [0.3, 1.3]
            # So final_reward ∈ [0, 1.3], scale to approx [0,2]
            final_reward = final_reward * (2.0 / 1.5)  # Scale to approx [0, 2]
            
            # Ensure reward is not negative (shouldn't happen theoretically)
            if final_reward < 0:
                raise ValueError("Invalid reward!")
            
            return float(final_reward)
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
    
    def get_action_dim(self) -> int:
        """Get action space size"""
        return self.num_questions
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (total number of symptoms)"""
        return self.num_questions

    def get_max_seq_len(self) -> int:
        """Return defined maximum sequence length (used for model initialization)"""
        # Note: this is the configured maximum window length; individual samples may be shorter
        return self.max_seq_len + 1  # because state = history + current symptom
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        sample = self.transitions[idx]
        return {
            "state": sample["state"],
            "action": torch.LongTensor([sample["action"]]),
            "uu_id": sample["uu_id"],
            "current_symptom": sample["current_symptom"],  # Symptom corresponding to the current question
            "current_idx": torch.LongTensor([sample["current_idx"]]),
            "history_idxs": sample["history_idxs"], # Indices of history symptoms (before current focus)
            "done": torch.BoolTensor([sample["done"]]),
            "state_len": len(sample["state"]),  # Record original length for subsequent masking
            "history_len": len(sample["history_idxs"]),
        }
    
    def split_dataset(self,args,splitted_dataset_path=None):
        # Split transitions by uu_id
        train_indices = []
        val_indices = []
        unique_uuids = self.df["uu_id"].unique()

        if splitted_dataset_path is not None:
            with open("./data_split.pkl", "rb") as f:
                split_info = pickle.load(f)
            train_uuids = set(split_info["train_uuids"])
            val_uuids = set(split_info["val_uuids"])
        else:
            # Split train/validation sets (by case)
            np.random.seed(42)
            np.random.shuffle(unique_uuids)
        
            train_size = int(len(unique_uuids) * args.train_ratio)
            train_uuids = set(unique_uuids[:train_size])
            val_uuids = set(unique_uuids[train_size:])
        
        for idx, transition in enumerate(self.transitions):
            if transition['uu_id'] in train_uuids:
                train_indices.append(idx)
            elif transition['uu_id'] in val_uuids:
                val_indices.append(idx)
        
        # Create Subset
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        
        print(f"[INFO] Total cases: {len(unique_uuids)}")
        print(f"[INFO] Train cases: {len(train_uuids)}, Val cases: {len(val_uuids)}")
        print(f"[INFO] Train transitions: {len(train_indices)}, Val transitions: {len(val_indices)}")

        print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        if splitted_dataset_path is None:
            # Save dataset split information
            split_info = {
                "train_uuids": list(train_uuids),
                "val_uuids": list(val_uuids),
                "train_ratio": args.train_ratio
            }
            with open("data_split.pkl", "wb") as f:
                pickle.dump(split_info, f)
            print(f"[INFO] Data split info saved to data_split.pkl")
        
        return train_dataset,val_dataset
    
