import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium import spaces
import gymnasium as gym

from Tokenizers.tokenizer import BasePoetryTokenizer
from Configuration import CLASSIFIER_CONFIGURATION
from models.Transformers import PoetEmbedder
from models.RewardSystem import PoeticRewardSystem


# ============================================================
#  POLICY NETWORK (Multi-Layer 2D Attention Pooling)
# ============================================================

class MultiLayerAttentionMaskingPolicy(nn.Module):
    """Policy network with multi-layer 2D attention pooling for action selection.
    
    This module implements a reinforcement learning policy that processes hidden states 
    from multiple transformer layers using 2D attention pooling. It learns to:
    1. Compute attention scores across both layers and sequence positions
    2. Apply softmax normalization to create attention weights
    3. Pool hidden states using the computed weights
    4. Generate action logits via a feedforward network
    
    The design allows the policy to flexibly attend to different layers and positions,
    making it suitable for tasks where optimal representations vary across layers.
    
    Attributes:
        ctx_len: Maximum sequence length and number of possible actions.
        num_layers: Number of transformer layers from which hidden states are extracted.
        embed_dim: Dimension of the embedding/hidden state vectors.
        scorer: Neural network for computing raw attention scores.
        actor_head: Feedforward network producing final action logits.
    """
    
    def __init__(self, embed_dim: int, ctx_len: int, num_layers: int, dropout: float = 0.1):
        """Initialize the multi-layer attention masking policy.
        
        Args:
            embed_dim (int): Hidden embedding dimension. Must match the transformer's 
                hidden dimension.
            ctx_len (int): Maximum context length, which determines both the input 
                sequence length and the number of possible actions (output logits).
            num_layers (int): Number of transformer layers whose hidden states will 
                be used for attention pooling.
            dropout (float, optional): Dropout probability for regularization. 
                Defaults to 0.1.
        
        Returns:
            None. Initializes the module with learnable parameters.
        """
        super().__init__()
        self.ctx_len = ctx_len
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Scorer network: computes attention scores from hidden vectors
        # Input: (batch, layers, seq_len, embed_dim) -> Output: (batch, layers, seq_len, 1)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

        # Actor head: processes pooled representations -> action logits
        # Outputs logits for each position in the context (ctx_len actions)
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, ctx_len),
        )

    def forward(
        self, 
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[int, torch.Tensor], 
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute action logits from multi-layer hidden states using 2D attention pooling.
        
        The forward pass performs three main operations:
        1. Stack and select hidden states from specified layers
        2. Compute 2D attention weights across layers and positions
        3. Pool hidden states and generate action logits
        
        Args:
            hidden_states (tuple | list | dict): Container of hidden state tensors from
                multiple transformer layers. Each tensor has shape (batch, seq_len, embed_dim).
                - If dict: uses sorted keys to select first num_layers layers
                - If list/tuple: uses first num_layers elements
                
            padding_mask (torch.Tensor, optional): Binary mask of shape (batch, seq_len)
                where 1 indicates valid tokens and 0 indicates padding tokens. Used to
                prevent attention to padded positions. Defaults to None (no masking).

        Returns:
            torch.Tensor: Action logits of shape (batch, ctx_len) representing the
                probability distribution over positions to mask next.

        Raises:
            TypeError: If hidden_states is not a dictionary, list, or tuple.
        """
        # Stack and select hidden states from specified layers
        if isinstance(hidden_states, dict):
            # For dict input: sort keys and take first num_layers entries
            keys = sorted(hidden_states.keys())[:self.num_layers]
            stacked = torch.stack([hidden_states[k] for k in keys], dim=1)
        elif isinstance(hidden_states, (list, tuple)):
            # For list/tuple input: take first num_layers elements
            selected = hidden_states[:self.num_layers]
            stacked = torch.stack(selected, dim=1)
        else:
            raise TypeError(f"Unsupported type for hidden_states: {type(hidden_states)}")

        # Tensor shape after stacking: (batch, num_layers, seq_len, embed_dim)
        B, N, L, d = stacked.shape

        # Compute raw attention scores: (batch, num_layers, seq_len, 1) -> (batch, num_layers, seq_len)
        raw_scores = self.scorer(stacked).squeeze(-1)

        # Apply padding mask if provided to prevent attention to padded positions
        if padding_mask is not None:
            # Move mask to correct device and add layer dimension
            # mask shape: (batch, seq_len) -> (batch, 1, seq_len)
            mask = padding_mask.to(stacked.device).unsqueeze(1).bool()
            # Mask out padding by setting scores to very negative value
            # This ensures padding positions get ~0 attention after softmax
            min_val = torch.finfo(raw_scores.dtype).min
            raw_scores = raw_scores.masked_fill(~mask, min_val)

        # Flatten scores for softmax: (batch, num_layers, seq_len) -> (batch, num_layers * seq_len)
        flat_scores = raw_scores.view(B, -1)
        # Compute attention weights via softmax (ensures they sum to 1)
        flat_weights = F.softmax(flat_scores, dim=-1)
        # Apply dropout for regularization
        flat_weights = self.dropout(flat_weights)
        # Reshape weights back: (batch, num_layers, seq_len)
        weights = flat_weights.view(B, N, L)

        # Pool hidden states using attention weights
        # Element-wise multiply: (B, N, L, d) * (B, N, L, 1) -> (B, N, L, d)
        # Sum across layers (dim=1) and positions (dim=2): (B, d)
        pooled = (stacked * weights.unsqueeze(-1)).sum(dim=(1, 2))
        
        # Generate action logits from pooled representation
        return self.actor_head(pooled)


# ============================================================
#  MANAGER
# ============================================================

class ParallelMaskingPoets:
    """Coordinator for dual-modality masked poetry generation using RL policies.
    
    This manager orchestrates two separate poetry embedders (syllable and WordPiece)
    and two corresponding masking policies. It supports autoregressive text generation
    by alternating between:
    - First half of episode: WordPiece policy masks and selects word subunits
    - Second half of episode: Syllable policy masks and selects syllable units
    
    The alternating approach allows the model to refine generated text at different
    linguistic granularities. Each policy is trained to maximize poetic quality rewards.
    
    Attributes:
        syl_emb: Transformer embedder for syllable tokenization.
        word_emb: Transformer embedder for WordPiece tokenization.
        syl_tok: Syllable tokenizer.
        word_tok: WordPiece tokenizer.
        syl_policy: RL policy network for syllable-level masking decisions.
        word_policy: RL policy network for WordPiece-level masking decisions.
        device: Torch device for tensor computation.
        ctx_len: Maximum context length for both modalities.
    """
    
    def __init__(
        self,
        syl_embedder: PoetEmbedder,
        word_embedder: PoetEmbedder,
        ctx_len: int,
        dropout: float,
        device: torch.device,
        num_layers: int = 6,
    ):
        """Initialize the parallel masking poetry manager.
        
        Args:
            syl_embedder (PoetEmbedder): Pre-trained transformer embedder for 
                syllable-level token processing. Must have a .tokenizer attribute.
            word_embedder (PoetEmbedder): Pre-trained transformer embedder for 
                WordPiece token processing. Must have a .tokenizer attribute.
            ctx_len (int): Maximum sequence length for both modalities.
            dropout (float): Dropout probability for policy network layers.
            device (torch.device): Device (CPU/GPU) for tensor operations.
            num_layers (int, optional): Number of transformer layers to use for 
                attention pooling in both policies. Defaults to 6.
        
        Returns:
            None. Initializes embedders, tokenizers, and policy networks.
        """
        self.syl_emb = syl_embedder
        self.word_emb = word_embedder
        self.syl_tok = syl_embedder.tokenizer
        self.word_tok = word_embedder.tokenizer

        self.ctx_len = ctx_len
        self.device = device
        self.num_layers = num_layers

        # Set embedders to evaluation mode (no gradient updates)
        self.syl_emb.eval()
        self.word_emb.eval()

        # Define which transformer layers to use for attention pooling
        # Note: These can be modified to use different layer combinations
        self.syl_layer_indices = [0, 1, 2, 3, 4, 5]
        self.word_layer_indices = [0, 1, 2, 3, 4, 5]

        # Initialize learnable policy networks
        # Syllable policy: converts syllable hidden states -> position logits
        self.syl_policy = MultiLayerAttentionMaskingPolicy(
            embed_dim=syl_embedder.d,
            ctx_len=ctx_len,
            num_layers=len(self.syl_layer_indices),
            dropout=dropout,
        ).to(device)

        # WordPiece policy: converts WordPiece hidden states -> position logits
        self.word_policy = MultiLayerAttentionMaskingPolicy(
            embed_dim=word_embedder.d,
            ctx_len=ctx_len,
            num_layers=len(self.word_layer_indices),
            dropout=dropout,
        ).to(device)

    def _get_selected_hidden_states(
        self, 
        embedder: PoetEmbedder, 
        input_ids: torch.Tensor, 
        padding_mask: torch.Tensor,
        layer_indices: list[int]
    ) -> list[torch.Tensor]:
        """Extract and return hidden states from specified transformer layers.
        
        This helper runs the embedder in inference mode and retrieves intermediate
        representations from selected layers. The returned states serve as input
        to the policy networks.
        
        Args:
            embedder (PoetEmbedder): Transformer embedder to run.
            input_ids (torch.Tensor): Token IDs of shape (batch, seq_len).
            padding_mask (torch.Tensor): Binary mask of shape (batch, seq_len)
                where 1 = valid, 0 = padding.
            layer_indices (list[int]): List of layer indices to extract.
                Example: [0, 1, 2, 3, 4, 5] for all 6 layers.

        Returns:
            list[torch.Tensor]: List of hidden state tensors, one per requested layer.
                Each tensor has shape (batch, seq_len, embed_dim).
        """
        # Run embedder in forward pass with input tokens and padding information
        embedder(input_ids, padding_mask=padding_mask)
        
        # Extract hidden states from specified layers
        return [embedder.get_layer_output(i) for i in layer_indices]

    def autoregressive_forward(
        self,
        text: str,
        quality_threshold_syl: float,
        quality_threshold_word: float,
        reward_system: PoeticRewardSystem,
        steps_per_episode: int = 12,
        num_episodes: int = 5,
        top_k: int = 3,
        verbose: bool = False,
    ) -> tuple[str, list[dict]]:
        """Generate text autoregressively using alternating masking policies.
        
        This is the main inference loop that:
        1. Runs multiple episodes of generation
        2. In first half of each episode: uses WordPiece policy to select positions
        3. In second half: uses syllable policy to refine selected positions
        4. Evaluates top-k candidates using the reward system
        5. Stops when quality thresholds are met or max episodes reached
        
        The process is iterative: each step masks one position, evaluates alternatives,
        and keeps the best one before moving to the next step.
        
        Args:
            text (str): Initial text to begin generation from.
            quality_threshold_syl (float): Minimum syllable quality score (0-1) 
                to stop generation.
            quality_threshold_word (float): Minimum WordPiece quality score (0-1) 
                to stop generation.
            reward_system (PoeticRewardSystem): System for evaluating candidate 
                text quality.
            steps_per_episode (int, optional): Total masking steps per episode. 
                Defaults to 12.
            num_episodes (int, optional): Maximum number of episodes to run. 
                Defaults to 5.
            top_k (int, optional): Number of top candidate tokens to evaluate 
                per action. Defaults to 3.
            verbose (bool, optional): Whether to print progress information. 
                Defaults to False.

        Returns:
            tuple: (final_text, history) where:
                - final_text (str): Generated text after final iteration
                - history (list[dict]): Step-by-step generation log with keys:
                    - episode, step: Episode and step indices
                    - text: Current generated text
                    - syl_action, word_action: Position indices
                    - exp_reward_syl, exp_reward_word: Expected rewards
                    - abs_score_syl, abs_score_word: Quality scores
                    - topk_probs_syl, topk_probs_word: Top-k probability distributions
                    - rewards_syl, rewards_word: Actual rewards for candidates
        """
        current_text = text
        pad_id_syl = self.syl_tok.vocab.get("[PAD]", 0)
        pad_id_word = self.word_tok.vocab.get("[PAD]", 0)

        history: list[dict] = []
        stop_all = False
        half_steps = steps_per_episode // 2

        # Loop over episodes
        for episode in range(num_episodes):
            if stop_all:
                break

            # Tokenize initial text and prepare for reward tracking
            s_init = self.pad_tokens(self.syl_tok.tokenize(current_text), self.syl_tok)
            w_init = self.pad_tokens(self.word_tok.tokenize(current_text), self.word_tok)
            reward_system.init_reward(s_init, w_init)

            if verbose:
                print(f"\n===== EPISODE {episode} | Starting text =====")
                print(current_text)

            # Loop over steps within the episode
            for step in range(steps_per_episode):
                # Tokenize current text for both modalities
                syl_toks = self.pad_tokens(self.syl_tok.tokenize(current_text), self.syl_tok)
                word_toks = self.pad_tokens(self.word_tok.tokenize(current_text), self.word_tok)

                # Convert token lists to tensor batches of shape (1, ctx_len)
                s_input = torch.tensor([syl_toks], dtype=torch.long, device=self.device)
                w_input = torch.tensor([word_toks], dtype=torch.long, device=self.device)
                
                # Create padding masks: 1 for valid tokens, 0 for padding
                # Shape: (1, ctx_len)
                s_mask = (s_input != pad_id_syl).long()
                w_mask = (w_input != pad_id_word).long()

                if step < half_steps:
                    # ===== PHASE 1: WordPiece Policy (first half of episode) =====
                    with torch.no_grad():
                        # Extract hidden states from multiple transformer layers
                        word_states = self._get_selected_hidden_states(
                            self.word_emb, w_input, w_mask, self.word_layer_indices
                        )
                        # Get action distribution from WordPiece policy
                        _, word_dist = self.evaluate([], word_states, None, w_mask)

                    # Sample action from policy distribution, respecting padding
                    word_logits = word_dist.logits.clone()
                    min_val = torch.finfo(word_logits.dtype).min
                    # Mask out padding positions by setting their logits to very negative
                    word_logits.masked_fill_(w_mask.view_as(word_logits) == 0, min_val)
                    masked_word_dist = Categorical(logits=word_logits)
                    
                    word_action = int(masked_word_dist.sample().item())
                    syl_action = 0

                    # Get top-k candidate tokens for the WordPiece policy action
                    topk_probs_word, cand_w_word = self.get_topk_candidate_tokens(
                        tokens=word_toks,
                        action=word_action,
                        tokenizer=self.word_tok,
                        embedder=self.word_emb,
                        top_k=top_k,
                        device=self.device,
                    )
                    
                    # Build dual batch: for each WordPiece candidate, keep syllables unchanged
                    cand_s_word = [list(syl_toks) for _ in range(len(cand_w_word))]

                    # Evaluate all candidates using the reward system
                    with torch.no_grad():
                        rewards_word = reward_system.evaluate_batch(cand_s_word, cand_w_word)
                    
                    # Compute expected reward as probability-weighted average
                    exp_reward_word = float(torch.dot(topk_probs_word, rewards_word).item())
                    exp_reward_syl = 0.0
                    exp_reward = exp_reward_word

                    # Select best candidates based on reward
                    best_s_toks = cand_s_word[0]
                    best_w_toks = cand_w_word[0]
                    topk_probs_syl = torch.zeros(top_k)
                    rewards_syl = torch.zeros(top_k)

                else:
                    # ===== PHASE 2: Syllable Policy (second half of episode) =====
                    with torch.no_grad():
                        # Extract hidden states for syllable modality
                        syl_states = self._get_selected_hidden_states(
                            self.syl_emb, s_input, s_mask, self.syl_layer_indices
                        )
                        # Get action distribution from syllable policy
                        syl_dist, _ = self.evaluate(syl_states, [], s_mask, None)

                    # Sample action from policy, respecting padding
                    syl_logits = syl_dist.logits.clone()
                    min_val = torch.finfo(syl_logits.dtype).min
                    syl_logits.masked_fill_(s_mask.view_as(syl_logits) == 0, min_val)
                    masked_syl_dist = Categorical(logits=syl_logits)
                    
                    syl_action = int(masked_syl_dist.sample().item())
                    word_action = 0

                    # Get top-k candidate tokens for syllable policy action
                    topk_probs_syl, cand_s_syl = self.get_topk_candidate_tokens(
                        tokens=syl_toks,
                        action=syl_action,
                        tokenizer=self.syl_tok,
                        embedder=self.syl_emb,
                        top_k=top_k,
                        device=self.device,
                    )
                    
                    # Build dual batch: for each syllable candidate, keep WordPiece unchanged
                    cand_w_syl = [list(word_toks) for _ in range(len(cand_s_syl))]

                    # Evaluate all candidates
                    with torch.no_grad():
                        rewards_syl = reward_system.evaluate_batch(cand_s_syl, cand_w_syl)
                    
                    # Compute expected reward
                    exp_reward_syl = float(torch.dot(topk_probs_syl, rewards_syl).item())
                    exp_reward_word = 0.0
                    exp_reward = exp_reward_syl

                    best_s_toks = cand_s_syl[0]
                    best_w_toks = cand_w_syl[0]
                    topk_probs_word = torch.zeros(top_k)
                    rewards_word = torch.zeros(top_k)

                # Update current text to the best candidate (highest reward)
                clean_w = [t for t in best_w_toks if t != pad_id_word]
                new_text = self.word_tok.detokenize(clean_w, as_text=True, control_tokens=True)
                current_text = new_text

                # Evaluate final text quality using reward system
                with torch.no_grad():
                    _, abs_score_syl, abs_score_word = reward_system.evaluate(
                        best_s_toks, best_w_toks, update_scores=True,
                    )

                # Record step information for training/analysis
                history.append({
                    "episode": episode,
                    "step": step,
                    "text": current_text,
                    "syl_action": syl_action,
                    "word_action": word_action,
                    "exp_reward_syl": exp_reward_syl,
                    "exp_reward_word": exp_reward_word,
                    "exp_reward_selected": exp_reward,
                    "abs_score_syl": float(abs_score_syl),
                    "abs_score_word": float(abs_score_word),
                    "avg_score": float((abs_score_syl + abs_score_word) / 2.0),
                    # Convert to list for JSON serialization
                    "topk_probs_syl": topk_probs_syl.detach().cpu().tolist() if torch.is_tensor(topk_probs_syl) else topk_probs_syl,
                    "topk_probs_word": topk_probs_word.detach().cpu().tolist() if torch.is_tensor(topk_probs_word) else topk_probs_word,
                    "rewards_syl": rewards_syl.detach().cpu().tolist() if torch.is_tensor(rewards_syl) else rewards_syl,
                    "rewards_word": rewards_word.detach().cpu().tolist() if torch.is_tensor(rewards_word) else rewards_word,
                })

                # Check stopping condition: both quality thresholds met
                if abs_score_syl >= quality_threshold_syl and abs_score_word >= quality_threshold_word:
                    stop_all = True
                    break

        return current_text, history

    def get_topk_candidate_tokens(
        self,
        tokens: list[int],
        action: int,
        tokenizer: BasePoetryTokenizer,
        embedder: PoetEmbedder,
        top_k: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Generate top-k candidate token sequences by masking a position.
        
        The process:
        1. Replaces token at action position with [MASK] token
        2. Runs embedder to predict probabilities at masked position
        3. Selects top-k tokens by probability
        4. Creates candidate sequences with each top-k token
        
        Args:
            tokens (list[int]): Current token sequence (length <= ctx_len).
            action (int): Position index to mask and replace.
            tokenizer (BasePoetryTokenizer): Tokenizer with [MASK] token ID.
            embedder (PoetEmbedder): Transformer embedder for prediction.
            top_k (int): Number of top candidates to return.
            device (torch.device): Device for tensor computations.

        Returns:
            tuple: (topk_probs, cand_batch) where:
                - topk_probs (torch.Tensor): Softmax probabilities of shape (top_k,)
                - cand_batch (list[list[int]]): List of top_k candidate token sequences
        """
        # Clamp action to valid range (some actions may exceed sequence length)
        target_idx = min(action, self.ctx_len - 1)
        mask_id = tokenizer.vocab.get("[MASK]", 4)

        # Create input with mask token at target position
        padded_tokens = list(tokens)
        padded_tokens[target_idx] = mask_id
        
        # Convert to tensor for model inference: shape (1, ctx_len)
        input_tensor = torch.tensor([padded_tokens], dtype=torch.long, device=device)
        # Create padding mask for transformer attention
        padding_mask = (input_tensor != tokenizer.vocab.get("[PAD]", 0)).long()

        # Run embedder and extract logits at masked position
        with torch.no_grad():
            logits = embedder(input_tensor, padding_mask=padding_mask)
            # Get probabilities for position being replaced: shape (vocab_size,)
            probs = torch.softmax(logits[0, target_idx, :], dim=-1)
            # Select top-k probabilities and their token indices
            topk_probs, topk_indices = torch.topk(probs, k=top_k)

        # Build candidate sequences with each top-k token
        cand_batch = []
        for tok_id in topk_indices.tolist():
            # Create new sequence with top-k token at masked position
            t_modified = list(padded_tokens)
            t_modified[target_idx] = tok_id
            cand_batch.append(t_modified)

        return topk_probs, cand_batch

    def pad_tokens(self, tokens: list[int], tokenizer: BasePoetryTokenizer) -> list[int]:
        """Pad or truncate token sequence to configured context length.
        
        This utility ensures all sequences have fixed length ctx_len, which is
        required for batched tensor operations.
        
        Args:
            tokens (list[int]): Raw token sequence from tokenizer.
            tokenizer (BasePoetryTokenizer): Tokenizer to get padding token ID.

        Returns:
            list[int]: Padded/truncated sequence of length ctx_len.
        """
        # Truncate if longer than context length
        tokens = tokens[:self.ctx_len]
        # Pad with [PAD] tokens if shorter
        pad_id = tokenizer.vocab.get("[PAD]", 0)
        return tokens + [pad_id] * (self.ctx_len - len(tokens))

    def evaluate(
        self,
        syl_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        word_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        syl_pad_mask: torch.Tensor = None,
        word_pad_mask: torch.Tensor = None,
    ) -> tuple[Categorical | None, Categorical | None]:
        """Evaluate both policy networks to produce action distributions.
        
        This is a utility method to run both policies in parallel. Used during
        generation when only one modality is active, the other returns None.
        
        Args:
            syl_states (tuple | list): Hidden state tensors for syllable policy.
                Can be empty list if only evaluating word policy.
            word_states (tuple | list): Hidden state tensors for WordPiece policy.
                Can be empty list if only evaluating syllable policy.
            syl_pad_mask (torch.Tensor, optional): Padding mask for syllable inputs.
            word_pad_mask (torch.Tensor, optional): Padding mask for WordPiece inputs.

        Returns:
            tuple: (syl_dist, word_dist) where each is a torch.distributions.Categorical
                or None if the corresponding hidden states were empty.
        """
        syl_dist = None
        word_dist = None

        # Evaluate syllable policy if hidden states provided
        if len(syl_states) > 0:
            syl_logits = self.syl_policy(syl_states, padding_mask=syl_pad_mask)
            syl_dist = Categorical(logits=syl_logits)

        # Evaluate WordPiece policy if hidden states provided
        if len(word_states) > 0:
            word_logits = self.word_policy(word_states, padding_mask=word_pad_mask)
            word_dist = Categorical(logits=word_logits)

        return syl_dist, word_dist

    @staticmethod
    def from_config(
        syl_embedder: PoetEmbedder,
        word_embedder: PoetEmbedder,
        syl_policy_weights_path: str,
        word_policy_weights_path: str,
        device: torch.device,
    ) -> "ParallelMaskingPoets":
        """Factory method to create ParallelMaskingPoets from saved weights.
        
        This static method provides a convenient way to load a complete system from
        configuration files and pre-trained policy weights.
        
        Args:
            syl_embedder (PoetEmbedder): Initialized syllable embedder.
            word_embedder (PoetEmbedder): Initialized WordPiece embedder.
            syl_policy_weights_path (str): Path to syllable policy state dict.
            word_policy_weights_path (str): Path to WordPiece policy state dict.
            device (torch.device): Device for placing tensors/models.

        Returns:
            ParallelMaskingPoets: Fully initialized manager with loaded weights.

        Raises:
            Exception: If either policy weight file fails to load.
        """
        # Create base instance with embedders
        poets = ParallelMaskingPoets(
            syl_embedder=syl_embedder,
            word_embedder=word_embedder,
            ctx_len=CLASSIFIER_CONFIGURATION.PARAMETERS()["MAX_SEQ_LEN"],
            device=device,
            dropout=CLASSIFIER_CONFIGURATION.PARAMETERS()["DROPOUT"],
            num_layers=6
        )

        # Load syllable policy weights if file exists
        if syl_policy_weights_path and os.path.exists(syl_policy_weights_path):
            try:
                state = torch.load(syl_policy_weights_path, map_location=device)
                poets.syl_policy.load_state_dict(state.get("state_dict", state))
                print(f"[INFO] Loaded syllable policy from '{syl_policy_weights_path}'")
            except Exception as e:
                print(f"[ERROR] Error loading syllable policy: {e}")
                raise

        # Load WordPiece policy weights if file exists
        if word_policy_weights_path and os.path.exists(word_policy_weights_path):
            try:
                state = torch.load(word_policy_weights_path, map_location=device)
                poets.word_policy.load_state_dict(state.get("state_dict", state))
                print(f"[INFO] Loaded WordPiece policy from '{word_policy_weights_path}'")
            except Exception as e:
                print(f"[ERROR] Error loading WordPiece policy: {e}")
                raise

        return poets


# ============================================================
#  ENVIRONMENT (Sequential: Phase 1: WordPiece -> Phase 2: Syllable)
# ============================================================

class ParallelMaskingPoetsEnv(gym.Env):
    """Gymnasium environment for reinforcement learning with masked poetry generation.
    
    This environment implements the RL loop for training policies to generate poetry
    by alternating between two modalities:
    - Phase 1 (first half): WordPiece masking policy decides which word subunits to refine
    - Phase 2 (second half): Syllable masking policy decides which syllables to refine
    
    At each step, the environment:
    1. Gets policy action (position to mask)
    2. Evaluates top-k candidate replacements
    3. Selects best candidate based on reward
    4. Returns reward signal for policy training
    5. Resets when quality thresholds met or max steps reached
    
    Attributes:
        action_space: Tuple of two Discrete spaces (one per modality).
        observation_space: Tuple of two Box spaces for hidden states.
        poets: ParallelMaskingPoets manager.
        reward_system: Reward evaluation system.
    """
    
    def __init__(
        self,
        poets: ParallelMaskingPoets,
        reward_system: PoeticRewardSystem,
        device: torch.device,
        top_k: int = 3,
        threshold_quality_word: float = 0.82,
        threshold_quality_syl: float = 0.72,
        max_steps_per_episode: int = 12,
    ):
        """Initialize the masked poetry Gymnasium environment.
        
        Args:
            poets (ParallelMaskingPoets): Manager for poetry generation policies.
            reward_system (PoeticRewardSystem): System for scoring generated text.
            device (torch.device): Device for tensor computations (CPU or GPU).
            top_k (int, optional): Number of candidate tokens per action. 
                Defaults to 3.
            threshold_quality_word (float, optional): WordPiece quality target (0-1). 
                Defaults to 0.82.
            threshold_quality_syl (float, optional): Syllable quality target (0-1). 
                Defaults to 0.72.
            max_steps_per_episode (int, optional): Maximum steps before forced reset. 
                Defaults to 12.
        """
        super().__init__()
        self.poets = poets
        self.reward_system = reward_system
        self.device = device
        self.top_k = top_k
        self.threshold_quality_word = threshold_quality_word
        self.threshold_quality_syl = threshold_quality_syl
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # Action space: two discrete action spaces, one per modality
        # Each action is a position index in range [0, ctx_len)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.poets.ctx_len),
            spaces.Discrete(self.poets.ctx_len),
        ))
        
        # Observation space: two hidden state tensors
        # (syllable_states, word_states) where each state is continuous
        self.observation_space = spaces.Box(
            low=-float("inf"), 
            high=float("inf"), 
            shape=(2,), 
            dtype=float
        )
        
        self.current_text = ""

    def reset(self, seed=None, options=None):
        """Reset environment to initial state.
        
        Can initialize from:
        - Empty text (default)
        - Text string provided in options["text"]
        - Pre-tokenized sequences in options["syl_tokens"] and options["word_tokens"]
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Configuration dictionary with possible keys:
                - "text": Initial text string
                - "syl_tokens": Pre-tokenized syllable sequence
                - "word_tokens": Pre-tokenized WordPiece sequence

        Returns:
            tuple: ((word_states, syl_states), info_dict) where:
                - word_states, syl_states: Lists of hidden state tensors
                - info_dict: Metadata with "word_mask" and "syl_mask"
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize tokens from various sources
        if options and "text" in options:
            self.current_text = options["text"]
            raw_syl_toks = self.poets.syl_tok.tokenize(self.current_text)
            raw_word_toks = self.poets.word_tok.tokenize(self.current_text)
        elif options and "syl_tokens" in options and "word_tokens" in options:
            raw_syl_toks = options["syl_tokens"]
            raw_word_toks = options["word_tokens"]
            self.current_text = self.poets.word_tok.detokenize(
                raw_word_toks, as_text=True, control_tokens=True
            )
        else:
            self.current_text = ""
            raw_syl_toks, raw_word_toks = [], []

        # Pad token sequences to context length
        s_toks = self.poets.pad_tokens(raw_syl_toks, self.poets.syl_tok)
        w_toks = self.poets.pad_tokens(raw_word_toks, self.poets.word_tok)

        # Initialize reward tracking with starting tokens
        self.reward_system.init_reward(s_toks, w_toks)

        # Get hidden states and masks for initial observation
        word_states, syl_states, w_mask, s_mask = self._get_env_states(w_toks, s_toks)

        return (word_states, syl_states), {"word_mask": w_mask, "syl_mask": s_mask}

    def step(self, actions: tuple[int, int]):
        """Advance environment by one step.
        
        The step function:
        1. Routes to correct policy based on episode phase (first half = word, second = syl)
        2. Selects top-k candidates for the active modality
        3. Computes expected reward from candidate probabilities
        4. Updates current text and scores
        5. Checks termination conditions
        
        Args:
            actions (tuple[int, int]): (syl_action, word_action) indices where
                actions for inactive policy are ignored.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: ((word_states, syl_states))
                - reward: (exp_reward_syl, exp_reward_word)
                - terminated (bool): True if quality thresholds met
                - truncated (bool): Always False (Gymnasium requirement)
                - info (dict): Metadata including scores, masks, and current text
        """
        syl_action, word_action = actions
        half_steps = self.max_steps_per_episode // 2

        # Tokenize current text for evaluation
        current_s_toks = self.poets.pad_tokens(
            self.poets.syl_tok.tokenize(self.current_text), self.poets.syl_tok
        )
        current_w_toks = self.poets.pad_tokens(
            self.poets.word_tok.tokenize(self.current_text), self.poets.word_tok
        )

        # Get padding token IDs for both modalities
        syl_pad_id = self.poets.syl_tok.vocab.get("[PAD]", 0)
        word_pad_id = self.poets.word_tok.vocab.get("[PAD]", 0)

        if self.current_step < half_steps:
            # ===== PHASE 1: WordPiece Policy =====
            # Check if word action hits padding
            hit_pad_word = (word_action >= len(current_w_toks)) or (current_w_toks[word_action] == word_pad_id)
            hit_pad_syl = False

            # Compute expected reward and best candidate for word action
            exp_reward_word, win_s_toks, win_w_toks = self._compute_expected_reward(
                current_w_toks, current_s_toks, self.poets.word_tok, self.poets.word_emb, word_action
            )
            exp_reward_syl = 0.0

        else:
            # ===== PHASE 2: Syllable Policy =====
            # Check if syllable action hits padding
            hit_pad_syl = (syl_action >= len(current_s_toks)) or (current_s_toks[syl_action] == syl_pad_id)
            hit_pad_word = False

            # Compute expected reward and best candidate for syl action
            exp_reward_syl, win_s_toks, win_w_toks = self._compute_expected_reward(
                current_w_toks, current_s_toks, self.poets.syl_tok, self.poets.syl_emb, syl_action
            )
            exp_reward_word = 0.0

        # Increment step counter
        self.current_step += 1

        # Evaluate best candidate to get quality scores
        _, abs_score_syl, abs_score_word = self.reward_system.evaluate(
            win_s_toks, win_w_toks, update_scores=True
        )
        
        # Get next observation (hidden states for both modalities)
        next_word_states, next_syl_states, w_mask, s_mask = self._get_env_states(win_w_toks, win_s_toks)

        # Check termination: either average score high or both thresholds met
        avg_score = (abs_score_syl + abs_score_word) / 2.0
        terminated = (avg_score >= 0.9) or (
            abs_score_syl >= self.threshold_quality_syl
            and abs_score_word >= self.threshold_quality_word
        )

        # Update current text (remove padding before detokenizing)
        clean_w = [t for t in win_w_toks if t != word_pad_id]
        self.current_text = self.poets.word_tok.detokenize(clean_w, as_text=True, control_tokens=True)

        # Assemble info dictionary
        info = {
            "syl_expected_reward": exp_reward_syl,
            "word_expected_reward": exp_reward_word,
            "abs_score_syl": abs_score_syl,
            "abs_score_word": abs_score_word,
            "hit_pad_syl": hit_pad_syl,
            "hit_pad_word": hit_pad_word,
            "current_text": self.current_text,
            "word_mask": w_mask,
            "syl_mask": s_mask,
        }

        return (next_word_states, next_syl_states), (exp_reward_syl, exp_reward_word), terminated, False, info

    def _get_env_states(self, w_toks: list[int], s_toks: list[int]):
        """Compute hidden states and padding masks for both modalities.
        
        This helper extracts transformer hidden states for the current text,
        which serve as observations for the policies.
        
        Args:
            w_toks (list[int]): WordPiece token sequence (length ctx_len).
            s_toks (list[int]): Syllable token sequence (length ctx_len).

        Returns:
            tuple: (word_states, syl_states, w_pad_mask, s_pad_mask) where:
                - word_states, syl_states: Lists of hidden tensors from selected layers
                - w_pad_mask, s_pad_mask: Binary padding masks (1=valid, 0=padding)
        """
        # Convert token lists to batch tensors of shape (1, ctx_len)
        s_input = torch.tensor([s_toks], dtype=torch.long, device=self.device)
        w_input = torch.tensor([w_toks], dtype=torch.long, device=self.device)

        # Create padding masks by comparing to padding token ID
        # Mask shape: (1, ctx_len)
        s_pad_mask = (s_input != self.poets.syl_tok.vocab.get("[PAD]", 0)).long()
        w_pad_mask = (w_input != self.poets.word_tok.vocab.get("[PAD]", 0)).long()

        # Extract hidden states from specified transformer layers (no gradients)
        with torch.no_grad():
            syl_states = self.poets._get_selected_hidden_states(
                self.poets.syl_emb, s_input, s_pad_mask, self.poets.syl_layer_indices
            )
            word_states = self.poets._get_selected_hidden_states(
                self.poets.word_emb, w_input, w_pad_mask, self.poets.word_layer_indices
            )

        return word_states, syl_states, w_pad_mask, s_pad_mask

    def _compute_expected_reward(
        self,
        current_w_toks: list[int],
        current_s_toks: list[int],
        tokenizer: BasePoetryTokenizer,
        embedder: PoetEmbedder,
        action: int,
    ) -> tuple[float, list[int], list[int]]:
        """Compute expected reward for a masking action.
        
        This method implements the core reward computation:
        1. Gets top-k candidate tokens for the active modality
        2. Builds dual batch (varying one modality, keeping other fixed)
        3. Evaluates each candidate using reward system
        4. Computes probability-weighted expected reward
        5. Returns expected reward and best candidate sequence
        
        Args:
            current_w_toks (list[int]): Current WordPiece token sequence.
            current_s_toks (list[int]): Current syllable token sequence.
            tokenizer (BasePoetryTokenizer): Tokenizer for active modality.
            embedder (PoetEmbedder): Embedder for active modality.
            action (int): Position to mask in active modality.

        Returns:
            tuple: (expected_reward, best_win_s_toks, best_win_w_toks) where:
                - expected_reward: Probability-weighted average of top-k rewards
                - best_win_s_toks, best_win_w_toks: Best candidate sequences
        """
        # Determine which modality is active
        is_word = (tokenizer is self.poets.word_tok)
        active_toks = current_w_toks if is_word else current_s_toks

        # Get top-k candidates for the active modality
        topk_probs, cand_active = self.poets.get_topk_candidate_tokens(
            tokens=active_toks,
            action=action,
            tokenizer=tokenizer,
            embedder=embedder,
            top_k=self.top_k,
            device=self.device,
        )

        # Build dual batch for reward evaluation
        # Vary one modality, keep the other fixed across the batch
        if is_word:
            cand_w_batch = cand_active
            cand_s_batch = [list(current_s_toks) for _ in range(len(cand_w_batch))]
        else:
            cand_s_batch = cand_active
            cand_w_batch = [list(current_w_toks) for _ in range(len(cand_s_batch))]

        # Evaluate all candidates in batch
        rewards = self.reward_system.evaluate_batch(cand_s_batch, cand_w_batch)
        
        # Compute expected reward as dot product of probabilities and rewards
        expected_reward = float(torch.dot(topk_probs, rewards).item())

        # Return best candidate (highest reward)
        best_win_s_toks = cand_s_batch[0]
        best_win_w_toks = cand_w_batch[0]

        return expected_reward, best_win_s_toks, best_win_w_toks