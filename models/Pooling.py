# models/Pooling.py
"""Pooling and classification models for poetry quality scoring.

This module provides an attention-based poem classifier and a multi-layer
classifier that performs joint 2D attention pooling over transformer layers and
tokens.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from Configuration import CLASSIFIER_CONFIGURATION

import os
import torch
import torch.nn as nn
from Configuration import CLASSIFIER_CONFIGURATION


class AttentionPoemClassifier(nn.Module):
    """
    Wrapper classifier for poetry quality scoring.

    This module wraps ``MultiLayerPoemClassifier`` and exposes a simplified
    interface for the reward system. It uses joint 2D attention over all selected
    transformer layers and tokens.

    Args:
        embed_dim: Hidden embedding dimension.
        dropout: Dropout probability. Defaults to 0.1.
        attention_hidden: Hidden dimension of the attention scorer. If ``None``,
            ``embed_dim`` is used.
        num_layers: Number of transformer layers. Defaults to 6.
        device: Device on which the model is placed.

    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        attention_hidden: int = None,
        num_layers: int = 6,
        device: torch.device = None,
    ):
        """Initialize the attention poem classifier.

        Args:
            embed_dim: Hidden embedding dimension.
            dropout: Dropout probability. Defaults to 0.1.
            attention_hidden: Hidden dimension of the attention scorer. If
                ``None``, ``embed_dim`` is used.
            num_layers: Number of transformer layers. Defaults to 6.
            device: Device on which the model is placed.
        """
        super().__init__()
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")

        # Inizializzazione della nuova testa di pooling multi-layer (2D Attention)
        self.model = MultiLayerPoemClassifier(
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_hidden=attention_hidden,
            device=self.device,
        )

    def forward(
        self,
        layer_outputs: dict[int, torch.Tensor],
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute poetry quality scores.

        Args:
            layer_outputs: Layer outputs from the embedder. It can be a
                dictionary, list, or tuple of tensors, each of shape
                ``(B, L, d)``.
            padding_mask: Optional padding mask of shape ``(B, L)`` where 1
                indicates a valid token and 0 indicates padding.

        Returns:
            A tensor of shape ``(B, 1)`` containing poetry scores in ``[0, 1]``.
        """
        return self.model(layer_outputs, padding_mask=padding_mask)

    def save_checkpoint(self, file_path: str):
        """Save the classifier's weights to the specified path."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save_state_dict(file_path)

    def load_checkpoint(self, file_path: str):
        """ Load the classifier's weights from the specified path."""
        self.model.load_state_dict_from_path(file_path, device=str(self.device))

    @staticmethod
    def from_config(
        device: torch.device,
        weight_path: str = None,
        num_layers: int = 6,
    ) -> "AttentionPoemClassifier":
        """Create an ``AttentionPoemClassifier`` from configuration.

        Args:
            device: Device on which the classifier is placed.
            weight_path: Optional path to pretrained weights.
            num_layers: Number of transformer layers. Defaults to 6.

        Returns:
            An initialized ``AttentionPoemClassifier`` instance.
        """
        classifier = AttentionPoemClassifier(
            embed_dim=CLASSIFIER_CONFIGURATION.PARAMETERS()["EMBED_DIM"],
            dropout=CLASSIFIER_CONFIGURATION.PARAMETERS()["DROPOUT"],
            attention_hidden=CLASSIFIER_CONFIGURATION.PARAMETERS()["HIDDEN_DIM"],
            num_layers=num_layers,
            device=device,
        )

        if weight_path and os.path.exists(weight_path):
            classifier.load_checkpoint(weight_path)
            print(f"[INFO] Caricati pesi Multi-Layer Classifier da '{weight_path}'")
        elif weight_path:
            print(f"[WARNING] Checkpoint non trovato in '{weight_path}'. Inizializzazione casuale.")

        return classifier


class MultiLayerPoemClassifier(nn.Module):
    """Poem classifier with joint 2D attention pooling over layers and tokens.

    The classifier computes an independent scalar attention weight for each
    ``(layer, token)`` pair, performs softmax over the flattened layer-token
    dimension, and pools the weighted representations before classification.

    Args:
        embed_dim: Hidden embedding dimension.
        num_layers: Number of transformer layers. Defaults to 6.
        dropout: Dropout probability. Defaults to 0.1.
        attention_hidden: Hidden dimension of the attention scorer. If ``None``,
            ``embed_dim`` is used.
        device: Device on which the model is placed."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        attention_hidden: int = None,
        device: torch.device = None,
    ):
        """Initialize the multi-layer poem classifier.

        Args:
            embed_dim: Hidden embedding dimension.
            num_layers: Number of transformer layers. Defaults to 6.
            dropout: Dropout probability. Defaults to 0.1.
            attention_hidden: Hidden dimension of the attention scorer. If
                ``None``, ``embed_dim`` is used.
            device: Device on which the model is placed.
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        attention_hidden = attention_hidden or embed_dim

        # Scorer for 2D attention pooling: generates a scalar score for each (layer, token) pair
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

        # Final classification head: maps pooled representation to a single score
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(
        self, 
        layer_outputs: dict[int, torch.Tensor], 
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute poetry quality scores using 2D attention pooling.

        Args:
            layer_outputs: Layer outputs from the embedder. It is a
                dictionary of tensors, each of shape ``(B, L, d)``, where ``B``
                is the batch size, ``L`` is the sequence length, and ``d`` is the
                embedding dimension.
            padding_mask: Optional padding mask of shape ``(B, L)`` where 1
                indicates a valid token and 0 indicates padding.

        Returns:
            A tensor of shape ``(B, 1)`` containing sigmoid-activated poetry
            scores in ``[0, 1]``.

            TypeError: If ``layer_outputs`` is not a dictionary.
        """
        # 1. Normalization and stacking of layer outputs into a tensor of shape (B, N, L, d)
        keys = sorted(layer_outputs.keys())[:self.num_layers]
        stacked = torch.stack([layer_outputs[k] for k in keys], dim=1)
            

        B, N, L, d = stacked.shape  # Batch, num_layers, seq_len, embed_dim

        # 2. 2D Attention Scoring: Compute raw attention scores for each (layer, token) pair
        raw_scores = self.scorer(stacked).squeeze(-1)  # Shape: (B, N, L)

        # 3. Apply padding mask if provided
        if padding_mask is not None:
            mask = padding_mask.to(stacked.device).unsqueeze(1).bool()  # (B, 1, L)
            min_value = torch.finfo(raw_scores.dtype).min
            raw_scores = raw_scores.masked_fill(~mask, min_value)

        # 4. Flatten and apply softmax over the (layer, token) dimension to get attention weights
        flat_scores = raw_scores.view(B, -1)                     # (B, N * L)
        flat_weights = F.softmax(flat_scores, dim=-1)            # (B, N * L)
        flat_weights = self.dropout(flat_weights)
        weights = flat_weights.view(B, N, L)                     # (B, N, L)

        # 5. Weighted pooling of the stacked representations using the attention weights
        pooled = (stacked * weights.unsqueeze(-1)).sum(dim=(1, 2))

        # 6. Final classification: Pass the pooled representation through the classification head
        logits = self.net(pooled)
        return torch.sigmoid(logits)

    def save_state_dict(self, file_path: str):
        """Save the classifier's state dictionary to the specified path."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({"classifier_state_dict": self.state_dict()}, file_path)

    def load_state_dict_from_path(self, file_path: str, device: str = "cpu"):
        """
        Load the classifier's state dictionary from the specified path.

        Args:
            file_path: Path to the checkpoint file.
            device: Device on which to load the model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nessun checkpoint trovato in {file_path}")
        
        checkpoint = torch.load(file_path, map_location=device)
        
        # Estragga lo state_dict se salvato in un dict di checkpoint
        state_dict = checkpoint.get("classifier_state_dict", checkpoint)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # RIMUOVE IL PREFISSO 'model.' SE PRESENTE
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                cleaned_state_dict[key[6:]] = value  # Taglia "model."
            else:
                cleaned_state_dict[key] = value

        self.load_state_dict(cleaned_state_dict)
        self.eval()

    @staticmethod
    def from_config(
        device: torch.device,
        weight_path: str = None,
        num_layers: int = 6,
    ) -> "MultiLayerPoemClassifier":
        """Create a ``MultiLayerPoemClassifier`` from configuration.

        Args:
            device: Device on which the classifier is placed.
            weight_path: Optional path to pretrained weights.
            num_layers: Number of transformer layers. Defaults to 6.

        Returns:
            An initialized ``MultiLayerPoemClassifier`` instance.
        """
        classifier = MultiLayerPoemClassifier(
            embed_dim=CLASSIFIER_CONFIGURATION.PARAMETERS()["EMBED_DIM"],
            num_layers=num_layers,
            attention_hidden=CLASSIFIER_CONFIGURATION.PARAMETERS()["HIDDEN_DIM"],
            dropout=CLASSIFIER_CONFIGURATION.PARAMETERS()["DROPOUT"],
            device=device,
        )

        if weight_path and os.path.exists(weight_path):
            classifier.load_state_dict_from_path(weight_path, device=str(device))
        return classifier