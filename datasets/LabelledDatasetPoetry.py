import os
import random
import torch
from torch.utils.data import Dataset
from datasets.MaskedDatasetPoetry import DynamicMaskedPoetryDataset
from tokenizers.tokenizer import BasePoetryTokenizer


class PoetryLabelledDataset(Dataset):
    """PyTorch Dataset for binary poetry classification (Real vs Generated/Fake).
    
    
    Args:
        true_poems: List of authentic poetry texts.
        fake_poems: List of generated/fake poetry texts.
        tokenizer: BasePoetryTokenizer instance for tokenization.
        max_len: Maximum sequence length.
        strip_structural_tokens_prob: Probability of removing structural tokens
            from fake samples to augment diversity. Defaults to 0.5.
    """
    
    def __init__(
        self, 
        true_poems: list[str], 
        fake_poems: list[str], 
        tokenizer: BasePoetryTokenizer, 
        max_len: int,
        strip_structural_tokens_prob: float = 0.5
    ):
        """Initialize classification dataset with balanced classes.
        
        Process:
        1. Balance classes by truncating to min length
        2. Extract structural token IDs for stochastic removal
        3. Tokenize and chunk real poems (label=1)
        4. Tokenize and chunk fake poems (label=0)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.strip_structural_tokens_prob = strip_structural_tokens_prob

        # Balance classes to prevent class imbalance bias
        min_len = min(len(true_poems), len(fake_poems))
        if len(true_poems) != len(fake_poems):
            print(f"[INFO] Class balancing: truncating to {min_len} samples per class.")
            true_poems = true_poems[:min_len]
            fake_poems = fake_poems[:min_len]

        # Extract token IDs for special tokens
        self.pad_id = tokenizer.vocab.get("[PAD]", 0)
        self.stanza_id = tokenizer.vocab.get("[STANZA]")
        self.verse_id = tokenizer.vocab.get("[VERSE]")

        # Set of structural token IDs (used for stochastic removal in fake samples)
        self.structural_token_ids = {
            idx for idx in [self.stanza_id, self.verse_id] if idx is not None
        }

        self.samples = []

        # Process real poems with label=1
        print("[INFO] Processing real poems (label=1)...")
        for poem in true_poems:
            # Tokenize and chunk while preserving poetic structure
            token_ids = self.tokenizer.tokenize(poem)
            chunks = DynamicMaskedPoetryDataset._poem_chunking(
                token_ids, max_len=self.max_len, stanza_id=self.stanza_id, verse_id=self.verse_id  # type: ignore
            )
            for chunk in chunks:
                self.samples.append((chunk, 1))

        # Process fake poems with label=0
        print("[INFO] Processing fake poems (label=0)...")
        for poem in fake_poems:
            # Tokenize and chunk
            token_ids = self.tokenizer.tokenize(poem)
            chunks = DynamicMaskedPoetryDataset._poem_chunking(
                token_ids, max_len=self.max_len, stanza_id=self.stanza_id, verse_id=self.verse_id  # type: ignore
            )
            for chunk in chunks:
                self.samples.append((chunk, 0))

    def __len__(self) -> int:
        """Return total number of samples (chunks) in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single labeled sample with padding and masks.
        
        Process:
        1. Retrieve token chunk and label
        2. Stochastically remove structural tokens from fake samples
        3. Truncate to max_len if needed
        4. Pad with [PAD] tokens and create attention mask
        5. Convert to PyTorch tensors
        
        Args:
            idx: Sample index.
        
        Returns:
            dict with keys:
                - "input_ids": token sequence, padded to max_len
                - "padding_mask": attention mask (1=valid, 0=padding)
                - "labels": class label (1=real, 0=fake)
        """
        token_ids, label = self.samples[idx]

        # Stochastic removal of structural tokens from fake samples (augmentation)
        if label == 0 and random.random() < self.strip_structural_tokens_prob:
            token_ids = [tok for tok in token_ids if tok not in self.structural_token_ids]

        # Truncate and pad to max_len
        seq_len = len(token_ids)

        if seq_len < self.max_len:
            # Pad with [PAD] tokens
            padding_length = self.max_len - seq_len
            input_ids = token_ids + [self.pad_id] * padding_length
            # Mask: 1 for valid tokens, 0 for padding
            padding_mask = [1] * seq_len + [0] * padding_length
        else:
            # Truncate if longer than max_len
            input_ids = token_ids[:self.max_len]
            padding_mask = [1] * self.max_len

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        padding_mask_tensor = torch.tensor(padding_mask, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "padding_mask": padding_mask_tensor,
            "labels": label_tensor
        }