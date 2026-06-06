"""
Transformer policy network: process state using Transformer Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """Positional encoding"""
        super().__init__()
        # 1. Initialize positional encoding matrix (max_len positions, each of dimension d_model)
        pe = torch.zeros(max_len, d_model)
        # 2. Generate position indices (0 to max_len-1), expand to [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 3. Compute decay terms (avoid large values using exponential scaling)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 4. Use sin for even indices and cos for odd indices (covering all d_model dims)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 5. Adjust dimensions to match Transformer input format (seq_len, batch_size, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        # 6. Register as buffer (not a parameter, saves memory)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support two input formats: (seq_len, batch, d_model) or (batch, seq_len, d_model)
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            if x.size(1) <= self.pe.size(0):
                x = x.transpose(0, 1)  # Convert to (seq_len, batch, d_model)
                x = x + self.pe[:x.size(0), :]  # Take only the positional encodings for the needed length
                x = x.transpose(0, 1)  # Convert back to original format
            else:
                x = x + self.pe[:x.size(0), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return x

class TransformerPolicyNetwork(nn.Module):
    """Transformer-based policy network"""
    
    def __init__(
        self,
        vocab_size: int,  
        action_dim: int,  
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        padding_idx: int = 0     
    ):
        """
        Args:
            vocab_size: number of symptoms
            action_dim: action space size (number of symptoms)
            d_model: Transformer model dimension
            nhead: number of attention heads
            num_layers: number of Transformer layers
            dim_feedforward: feedforward network hidden dimension
            dropout: dropout rate
            max_seq_len: maximum sequence length (for positional encoding)
            padding_idx: padding index (optional)
        """
        super(TransformerPolicyNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.d_model = d_model
        
        # 1. Embedding layer: map symptom indices to dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model,
            padding_idx=padding_idx  # If padding is used, ensure padding index does not participate in gradient updates
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Use (batch, seq_len, d_model) format for better performance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: typically use the last position (or [CLS]) to predict actions
        # Here we use simple pooling: average outputs across sequence positions
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool (seq_len, d_model) to (1, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self, init_transformer:bool=True, init_range: float = 0.1):
        """Initialize weights
        args: init_range: absolute range for uniform initialization of embedding weights. The default 0.1 is an empirical value and is generally safe for discrete symptom IDs.
        """
        # 1. Special initialization for embedding layer
        if hasattr(self, 'embedding'):
            # Use uniform initialization instead of xavier_uniform_ for embeddings
            # xavier_uniform_ is designed for linear layers and uses input/output dims to compute a gain.
            # Embedding lookup is not a typical linear transform; using simple uniform initialization is more straightforward and common.
            nn.init.uniform_(self.embedding.weight, -init_range, init_range)
            
            # If padding_idx is set, freeze its corresponding vector (often initialized to 0)
            # Padding tokens (e.g., index 0) have no semantics; their vectors should not be updated during training
            # because there is no meaningful gradient signal. Initializing them to 0 and keeping them fixed is good practice.
            if self.embedding.padding_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[self.embedding.padding_idx].fill_(0)
            print(f"[INFO] Embedding layer initialized with uniform_(-{init_range}, {init_range})")
        
        # 2. Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 3. (Optional but recommended) Initialize Transformer encoder parameters
        # PyTorch's nn.TransformerEncoderLayer contains multiple linear layers and attention weights.
        # Although PyTorch sets default initializations, explicitly and consistently applying xavier_uniform_
        # to all parameters helps the model start training from a stable point, improving Transformer training stability,
        # especially for deeper stacks.
        if init_transformer:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            print("[INFO] Transformer encoder layers initialized with xavier_uniform_")

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: input symptom index sequences, shape (batch_size, seq_len)
            src_key_padding_mask: mask, shape (batch_size, seq_len), True indicates padding positions to be masked
        Returns:
            logits: unnormalized action scores, shape (batch_size, action_dim)
        """
        # 1. Embedding
        # x shape: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        
        # 2. Scale embeddings
        x = x * math.sqrt(self.d_model)
        
        # 3. Add positional encoding
        x = self.pos_encoder(x)
        
        # 4. Transformer encoding with attention mask
        # src_key_padding_mask tells the Transformer which positions are padding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 5. Pooling: a smarter pooling is needed to average only over non-padding positions
        if src_key_padding_mask is not None:
            # Invert mask: True indicates valid positions, False indicates padding
            valid_mask = ~src_key_padding_mask  # (batch_size, seq_len)
            # Zero-out outputs at padding positions
            x = x * valid_mask.unsqueeze(-1).float()
            # Compute valid lengths per sample
            valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
            # Average over valid positions
            x_pooled = x.sum(dim=1) / valid_lengths.clamp(min=1.0)  # (batch_size, d_model)
        else:
            # If no mask, average across all positions
            x_pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # 6. Classification head
        logits = self.classifier(x_pooled)  # (batch_size, action_dim)
        return logits
    
    def get_action_probs(self, state: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Get action probability distribution
        Args:
            state: [batch_size, seq_len]
            mask: [batch_size, seq_len], True denotes masked (padding) positions
        Returns:
            probs: [batch_size, action_dim] action probability distribution
        """
        logits = self.forward(state, src_key_padding_mask=mask)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor, src_key_padding_mask: torch.Tensor = None, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample actions from the policy
        
        Args:
            state: [batch_size, seq_len] index sequences
            src_key_padding_mask: [batch_size, seq_len], True denotes positions to be masked (padding)
            temperature: temperature parameter (controls exploration)
        
        Returns:
            actions: [batch_size] sampled action indices
        """
        # Important: pass mask!
        logits = self.forward(state, src_key_padding_mask=src_key_padding_mask) / temperature
        probs = F.softmax(logits, dim=-1)
        action_idx_batch = torch.multinomial(probs, 1).squeeze(-1)
        return action_idx_batch
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log-probabilities of actions
        
        Args:
            state: [batch_size, state_dim] or [state_dim]
            action: [batch_size] or scalar action indices
        
        Returns:
            log_probs: [batch_size] or scalar log-probabilities
        """
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        return log_probs.gather(-1, action.unsqueeze(-1) if action.dim() == 1 else action.unsqueeze(0)).squeeze(-1)

