import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Configuration import CLASSIFIER_CONFIGURATION, PATH_CONFIGURATION
from models.Pooling import MultiLayerPoemClassifier
from models.Transformers import PoetEmbedder


class PoeticRewardSystem:
    """Reward system for evaluating and reinforcing poetic quality through dual classifiers.
    
    This system evaluates generated poetry across two linguistic dimensions:
    1. Syllable-level quality: evaluated by a syllable-trained classifier
    2. WordPiece-level quality: evaluated by a WordPiece-trained classifier
    
    The core design uses the harmonic mean of syllable and word scores as the reward signal,
    encouraging balanced quality across both modalities. Rewards are normalized as changes
    (deltas) from previous states, making the signal suitable for RL training.

    """
    
    def __init__(
        self,
        syl_embedder: PoetEmbedder,
        syl_classifier: MultiLayerPoemClassifier,
        word_embedder: PoetEmbedder,
        word_classifier: MultiLayerPoemClassifier,
        reward_multiplier: float = 10.0,
        num_layers: int = 6,
    ):
        """Initialize the poetic reward system with dual embedders and classifiers.
        
        Args:
            syl_embedder (PoetEmbedder): Pre-trained transformer embedder for 
                syllable-level representations. Should be in eval mode after init.
            syl_classifier (MultiLayerPoemClassifier): Classifier trained to score
                poetic quality at syllable level. Takes multi-layer hidden states as input.
            word_embedder (PoetEmbedder): Pre-trained transformer embedder for 
                WordPiece-level representations. Should be in eval mode after init.
            word_classifier (MultiLayerPoemClassifier): Classifier trained to score
                poetic quality at WordPiece level. Takes multi-layer hidden states as input.
            reward_multiplier (float, optional): Scaling factor for delta rewards.
                Higher values amplify reward signal for policy learning. Defaults to 10.0.
            num_layers (int, optional): Number of transformer layers whose outputs
                are used by the classifiers. Defaults to 6.
        
        Returns:
            None. Initializes the reward system in evaluation mode.
        """
        # Store embedders and set to evaluation mode (no batch norm, no dropout)
        self.syl_embedder = syl_embedder
        self.syl_embedder.eval()
        self.syl_classifier = syl_classifier
        self.syl_classifier.eval()

        # Store WordPiece embedder and classifier
        self.word_embedder = word_embedder
        self.word_embedder.eval()
        self.word_classifier = word_classifier
        self.word_classifier.eval()

        # Configuration parameters
        self.num_layers = num_layers
        self.eps = 1e-4  # Small constant to prevent division by zero
        self.previous_scores = None  # Will be initialized on first call to init_reward
        self.reward_multiplier = reward_multiplier

    def init_reward(
        self, syl_token_ids: list[int], word_token_ids: list[int]
    ) -> tuple[float, float]:
        """Initialize reward system with baseline scores from starting text.
        
        This method must be called before the first step() call to establish a
        baseline. The baseline scores are used to compute delta rewards during
        generation. Typically called at the start of each episode.
        
        Args:
            syl_token_ids (list[int]): Initial syllable token sequence
                (length typically equals ctx_len after padding).
            word_token_ids (list[int]): Initial WordPiece token sequence
                (length typically equals ctx_len after padding).

        Returns:
            tuple[float, float]: (initial_syl_score, initial_word_score) where each
                score is in range [0, 1] representing initial poetic quality.
        """
        # Compute baseline quality scores for both modalities
        init_syl = self._compute_single_score(
            self.syl_embedder, self.syl_classifier, syl_token_ids
        )
        init_word = self._compute_single_score(
            self.word_embedder, self.word_classifier, word_token_ids
        )
        
        # Store baseline scores for delta computation in evaluate()
        self.previous_scores = (init_syl, init_word)
        return self.previous_scores

    def evaluate(
        self,
        current_syl_token_ids: list[int],
        current_word_token_ids: list[int],
        update_scores: bool = True,
    ) -> tuple[float, float, float]:
        """Evaluate current text and compute reward as delta from baseline.
        
        The reward computation:
        1. Computes current quality scores for both modalities
        2. Applies penalty for detected subword hallucinations (repeated chars)
        3. Computes harmonic mean of scores to encourage balanced quality
        4. Returns reward as change in harmonic mean from previous state
        5. Optionally updates baseline for next evaluation
        
        Args:
            current_syl_token_ids (list[int]): Current syllable token sequence.
            current_word_token_ids (list[int]): Current WordPiece token sequence.
            update_scores (bool, optional): If True, updates baseline scores for
                next delta computation. Defaults to True.

        Returns:
            tuple[float, float, float]: (final_reward, abs_score_syl, abs_score_word) where:
                - final_reward: Delta reward (change in harmonic mean * multiplier)
                  Positive when quality improved, negative when degraded.
                - abs_score_syl: Absolute syllable quality score (0-1)
                - abs_score_word: Absolute WordPiece quality score (0-1)
                
        Raises:
            ValueError: If init_reward() not called before this method.
        """
        if self.previous_scores is None:
            raise ValueError("RewardSystem not initialized. Call init_reward() first.")

        # Compute absolute quality scores for current text
        abs_score_syl = self._compute_single_score(
            self.syl_embedder, self.syl_classifier, current_syl_token_ids
        )
        abs_score_word = self._compute_single_score(
            self.word_embedder, self.word_classifier, current_word_token_ids
        )

        # Get previous scores for delta computation
        prev_syl, prev_word = self.previous_scores

        # Compute harmonic mean of current scores: H(s, w) = 2*s*w / (s + w)
        # Harmonic mean ensures both scores must be high for the result to be high
        h_current = (2.0 * abs_score_syl * abs_score_word) / (abs_score_syl + abs_score_word + self.eps)
        
        # Compute harmonic mean of previous scores
        h_prev = (2.0 * prev_syl * prev_word) / (prev_syl + prev_word + self.eps)

        # Reward is change in harmonic mean, scaled by multiplier
        delta_h = h_current - h_prev
        final_reward = delta_h * self.reward_multiplier

        # Update baseline scores if requested (typically True during training)
        if update_scores:
            self.previous_scores = (abs_score_syl, abs_score_word)

        return final_reward, abs_score_syl, abs_score_word

    def evaluate_batch(
        self,
        syl_token_batch: list[list[int]],
        word_token_batch: list[list[int]],
    ) -> torch.Tensor:
        """Evaluate multiple candidate texts and return vectorized rewards.
        
        This method efficiently scores a batch of candidates (e.g., top-k alternatives)
        using vectorized operations. All candidates are evaluated against the same
        baseline scores established by init_reward().
        
        The process:
        1. Creates batched tensors from token lists
        2. Runs both embedders and classifiers in batch mode
        3. Applies subword penalty elementwise
        4. Computes delta rewards as batch operation
        
        Args:
            syl_token_batch (list[list[int]]): Batch of syllable token sequences.
                Shape: (batch_size, ctx_len).
            word_token_batch (list[list[int]]): Batch of WordPiece token sequences.
                Shape: (batch_size, ctx_len).

        Returns:
            torch.Tensor: Batch of delta rewards, shape (batch_size,).
                Each element is a scalar reward for the corresponding candidate.
                
        Raises:
            ValueError: If init_reward() not called before this method.
        """
        if self.previous_scores is None:
            raise ValueError("RewardSystem not initialized. Call init_reward() first.")

        # Get device for each embedder (may differ in distributed setups)
        device_s = next(self.syl_embedder.parameters()).device
        device_w = next(self.word_embedder.parameters()).device

        # Convert token lists to batched tensors
        # Shape: (batch_size, seq_len)
        s_input = torch.tensor(syl_token_batch, dtype=torch.long, device=device_s)
        w_input = torch.tensor(word_token_batch, dtype=torch.long, device=device_w)

        # Create padding masks: 1 for valid tokens, 0 for padding (token_id == 0)
        # Shape: (batch_size, seq_len)
        s_mask = (s_input != 0).long()
        w_mask = (w_input != 0).long()

        # Run embedders and classifiers on batches (no gradients needed)
        with torch.no_grad():
            # Process syllable modality: embedder -> extract layers -> classify
            self.syl_embedder(s_input, padding_mask=s_mask)
            syl_layers = {l: self.syl_embedder.get_layer_output(l) for l in range(self.num_layers)}
            # Output shape: (batch_size, 1) -> squeeze to (batch_size,)
            scores_syl = self.syl_classifier(syl_layers, padding_mask=s_mask).squeeze(-1)

            # Process WordPiece modality: embedder -> extract layers -> classify
            self.word_embedder(w_input, padding_mask=w_mask)
            word_layers = {l: self.word_embedder.get_layer_output(l) for l in range(self.num_layers)}
            # Output shape: (batch_size, 1) -> squeeze to (batch_size,)
            scores_word = self.word_classifier(word_layers, padding_mask=w_mask).squeeze(-1)

            # Apply subword hallucination penalty elementwise across batch
            # This detects repeated characters that indicate token collapse


        # Get baseline scores from previous state
        prev_syl, prev_word = self.previous_scores

        # Compute harmonic means for current batch scores
        # Broadcasting: (batch_size,) o (batch_size,) -> (batch_size,)
        h_current = (2.0 * scores_syl * scores_word) / (scores_syl + scores_word + self.eps)
        
        # Compute harmonic mean from baseline (scalars broadcast to batch)
        h_prev = (2.0 * prev_syl * prev_word) / (prev_syl + prev_word + self.eps)

        # Compute vectorized delta rewards
        # Shape: (batch_size,)
        final_rewards = (h_current - h_prev) * self.reward_multiplier
        return final_rewards

    def _compute_single_score(
        self,
        embedder: PoetEmbedder,
        classifier: MultiLayerPoemClassifier,
        token_ids: list[int],
    ) -> float:
        """Compute quality score for a single text sample.
        
        This is a helper method that:
        1. Converts token list to batch tensor (batch_size=1)
        2. Extracts hidden states from all transformer layers
        3. Passes multi-layer representations to classifier
        4. Returns scalar quality score (0-1)
        
        Args:
            embedder (PoetEmbedder): Transformer embedder to extract representations.
            classifier (MultiLayerPoemClassifier): Classifier that scores the text.
            token_ids (list[int]): Token sequence to score.

        Returns:
            float: Quality score in range [0, 1]. Higher values indicate better poetry.
        """
        # Get device from embedder parameters
        device = next(embedder.parameters()).device
        
        # Convert token list to batch tensor: (1, seq_len)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        # Create padding mask: 1 for valid tokens, 0 for padding (token_id == 0)
        # Shape: (1, seq_len)
        padding_mask = (input_tensor != 0).long()

        # Extract representation and compute score (no gradients)
        with torch.no_grad():
            # Run embedder to get hidden states from all layers
            embedder(input_tensor, padding_mask=padding_mask)
            
            # Extract hidden states from each layer: dictionary of {layer_idx: tensor}
            # Each tensor has shape (1, seq_len, hidden_dim)
            layer_outputs = {l: embedder.get_layer_output(l) for l in range(self.num_layers)}
            
            # Pass multi-layer states to classifier
            # Returns: (1, 1) tensor
            score = classifier(layer_outputs, padding_mask=padding_mask).item()

        return float(score)


    @staticmethod
    def from_config(
        syl_embedder: PoetEmbedder,
        word_embedder: PoetEmbedder,
        device: torch.device,
    ) -> "PoeticRewardSystem":
        """Factory method to create reward system from configuration and weights.
        
        This static method provides a convenient way to instantiate a complete
        reward system from pre-trained models and configuration files. It:
        1. Infers number of layers from embedder architecture
        2. Loads pre-trained classifiers from disk
        3. Creates and returns initialized PoeticRewardSystem
        
        Args:
            syl_embedder (PoetEmbedder): Pre-initialized syllable embedder.
            word_embedder (PoetEmbedder): Pre-initialized WordPiece embedder.
            device (torch.device): Device for placing classifiers (CPU or GPU).

        Returns:
            PoeticRewardSystem: Fully initialized reward system with loaded weights.
            
        Raises:
            Exception: If loading classifier weights from disk fails.
        """
        # Infer number of layers from embedder architecture
        num_layers = len(syl_embedder.layers)

        # Load pre-trained syllable classifier from weights file
        syl_classifier = MultiLayerPoemClassifier.from_config(
            device=device,
            num_layers=num_layers,
            weight_path=PATH_CONFIGURATION.MODELS_WEIGHTS_PATH()["SYLLABLE_CLASSIFIER"],
        )
        
        # Load pre-trained WordPiece classifier from weights file
        word_classifier = MultiLayerPoemClassifier.from_config(
            device=device,
            num_layers=num_layers,
            weight_path=PATH_CONFIGURATION.MODELS_WEIGHTS_PATH()["WORDPIECE_CLASSIFIER"],
        )

        # Create and return reward system with loaded components
        return PoeticRewardSystem(
            syl_embedder=syl_embedder,
            syl_classifier=syl_classifier,
            word_embedder=word_embedder,
            word_classifier=word_classifier,
            num_layers=num_layers,
        )