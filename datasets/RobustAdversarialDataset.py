import torch
from torch.utils.data import Dataset
from datasets.MaskedDatasetPoetry import DynamicMaskedPoetryDataset


class RobustAdversarialDataset(Dataset):
    """Memory-efficient Dataset with pre-allocated contiguous tensors.
    
    Optimization: Instead of storing N dictionaries of tensors (expensive serialization
    with multiprocessing), uses 3 contiguous tensors:
    - input_ids: (num_samples, max_len) of token IDs
    - padding_mask: (num_samples, max_len) of attention masks
    - labels: (num_samples,) of float labels
    
    This reduces tensor objects from ~230K to 3 when using num_workers > 0.
    Result: 10-100x faster DataLoader with multiprocessing.
    
    Args:
        texts_with_labels: Sequence of (text, label) tuples.
        tokenizer: BasePoetryTokenizer for tokenization.
        domain: Name of domain (for logging only).
        max_len: Maximum sequence length. Defaults to 256.
    """

    def __init__(self, texts_with_labels, tokenizer, domain: str, max_len: int = 256):
        """Initialize with pre-tokenization and tensor pre-allocation.
        
        Process:
        1. Extract pad/stanza/verse token IDs
        2. Accumulate chunks into Python lists (cheap, primitive types)
        3. Vectorized torch.tensor() conversion (single C++ call, not N calls)
        4. Delete intermediate lists to free memory
        
        Key optimization: torch.tensor(list_of_lists) is vectorized in C++,
        much faster than looping and calling torch.tensor(item) N times.
        """
        self.pad_id = tokenizer.vocab.get("[PAD]", 0)
        self.stanza_id = tokenizer.vocab.get("[STANZA]")
        self.verse_id = tokenizer.vocab.get("[VERSE]")

        print(f"[INFO] Tokenizing and chunking for domain '{domain}'...")

        # Accumulate chunks in Python lists (fast, primitives only: int/float)
        # These are NOT tensor objects yet, so very memory-efficient
        input_ids_list = []
        mask_list = []
        labels_list = []

        for text, label in texts_with_labels:
            # Tokenize and chunk while preserving poetic structure
            token_ids = tokenizer.tokenize(text)
            chunks = DynamicMaskedPoetryDataset._poem_chunking(
                token_ids, max_len, self.stanza_id, self.verse_id
            )

            for chunk in chunks:
                # Skip very short chunks (likely noise)
                if len(chunk) < 5:
                    continue

                # Truncate and pad to max_len
                seq_len = len(chunk)
                if seq_len < max_len:
                    padded_chunk = chunk + [self.pad_id] * (max_len - seq_len)
                    # Mask: 1 for valid tokens, 0 for padding
                    mask = [1] * seq_len + [0] * (max_len - seq_len)
                else:
                    padded_chunk = chunk[:max_len]
                    mask = [1] * max_len

                # Append to Python lists (no tensor overhead yet)
                input_ids_list.append(padded_chunk)
                mask_list.append(mask)
                labels_list.append(float(label))

        # Single vectorized torch.tensor() call per array
        # This is MUCH faster than looping: torch handles it with C++ backend
        # Result: (num_samples, max_len) or (num_samples,) tensors
        self.input_ids    = torch.tensor(input_ids_list, dtype=torch.long)     # (N, max_len)
        self.padding_mask = torch.tensor(mask_list,      dtype=torch.long)     # (N, max_len)
        self.labels       = torch.tensor(labels_list,    dtype=torch.float32)  # (N,)

        # Delete Python lists to free intermediate memory
        # RAM now lives only in tensor objects (contiguous allocation)
        del input_ids_list, mask_list, labels_list

        print(f"[SUCCESS] Pre-allocated {self.input_ids.shape[0]} chunks in RAM.")

    def __len__(self):
        """Return total number of samples."""
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        """Return sample at index via tensor slicing (O(1), returns view).
        
        Tensor slicing returns a view (not a copy), so minimal overhead.
        DataLoader's default_collate() will stack views into contiguous batch.
        
        Args:
            idx: Sample index.
        
        Returns:
            dict with keys:
                - "input_ids": token sequence, shape (max_len,)
                - "padding_mask": attention mask, shape (max_len,)
                - "labels": float label, shape ()
        """
        return {
            "input_ids":    self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
            "labels":       self.labels[idx],
        }