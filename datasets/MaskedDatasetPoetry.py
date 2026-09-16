import os
import random
import torch
from torch.utils.data import Dataset

from Tokenizers.tokenizer import BasePoetryTokenizer


class DynamicMaskedPoetryDataset(Dataset):
    """PyTorch Dataset with on-the-fly dynamic masking for Masked Language Modeling (MLM).
    
    Features:
    - Generates new random masks for each sample per epoch (dynamic)
    - BERT 80/10/10 masking strategy: 80% [MASK], 10% random token, 10% unchanged
    - Protects special tokens ([VERSE], [STANZA], [PAD], etc.) from masking
    - Poetic structure-aware chunking (respects stanzas and verses)
    - Compatible with any BasePoetryTokenizer subclass
    
    Args:
        poems: List of poetry texts.
        tokenizer: BasePoetryTokenizer instance.
        max_len: Maximum sequence length. Defaults to 256.
        mask_prob: Probability of masking each valid token. Defaults to 0.15.
    """
    
    def __init__(self, poems: list[str], tokenizer: BasePoetryTokenizer, max_len: int = 256, mask_prob: float = 0.15):
        """Initialize dataset with pre-tokenization and structural chunking.
        
        Process:
        1. Extract special token IDs (never mask these)
        2. Extract valid token IDs (can be used as random replacements)
        3. Pre-tokenize all poems and chunk while preserving structure
        4. Store chunks in memory for fast access
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

        # Extract all special tokens (control tokens that should never be masked)
        self.special_tokens = set(tokenizer.get_special_tokens())
        self.special_token_ids = {
            idx for tok, idx in tokenizer.vocab.items() 
            if tok in self.special_tokens
        }

        # Extract key token IDs for masking and structure-aware operations
        self.pad_id = tokenizer.vocab.get("[PAD]", 0)
        self.mask_id = tokenizer.vocab.get("[MASK]", 4)
        
        # Structural markers for intelligent chunking
        self.stanza_id = tokenizer.vocab.get("[STANZA]")
        self.verse_id = tokenizer.vocab.get("[VERSE]")

        # Valid token IDs = all tokens except special tokens (used for random replacement)
        self.valid_token_ids = [
            idx for idx in tokenizer.vocab.values()
            if idx not in self.special_token_ids
        ]

        # Pre-tokenize and chunk all poems
        print("[INFO] Pre-tokenizing and chunking poems...")
        self.encoded_poems = []

        for poem in poems:
            token_ids = self.tokenizer.tokenize(poem)
            # Chunk while respecting poetic structure
            chunks = self._poem_chunking(token_ids, self.max_len, self.stanza_id, self.verse_id)
            self.encoded_poems.extend(chunks)

    @staticmethod
    def _poem_chunking(token_ids: list[int], max_len: int, stanza_id: int, verse_id: int) -> list[list[int]]:
        """Chunk token sequence while respecting poetic structure.
        
        Strategy (4 steps):
        1. Split on stanza markers first (preserve stanzas as units)
        2. Split long stanzas on verse markers
        3. Force-split any remaining segments > max_len
        4. Pack segments into chunks to maximize context
        
        Args:
            token_ids: Complete token sequence.
            max_len: Maximum tokens per chunk.
            stanza_id: Token ID for [STANZA] marker.
            verse_id: Token ID for [VERSE] marker.
        
        Returns:
            list[list[int]]: Chunks of max length, preserving structure.
        """
        # Return as-is if already short enough
        if len(token_ids) <= max_len:
            return [token_ids]

        # Step 1: Split on stanza boundaries
        stanzas, curr = [], []
        for tok in token_ids:
            curr.append(tok)
            if tok == stanza_id:
                stanzas.append(curr)
                curr = []
        if curr: 
            stanzas.append(curr)

        # Step 2: Split long stanzas on verse boundaries
        segments = []
        for stanza in stanzas:
            if len(stanza) <= max_len:
                segments.append(stanza)
            else:
                # Verse-level split
                curr_verse = []
                for tok in stanza:
                    curr_verse.append(tok)
                    if tok == verse_id:
                        segments.append(curr_verse)
                        curr_verse = []
                if curr_verse: 
                    segments.append(curr_verse)
        
        # Step 3: Force-split any segment still > max_len
        final_segments = []
        for seg in segments:
            for i in range(0, len(seg), max_len):
                final_segments.append(seg[i:i+max_len])

        # Step 4: Pack segments into chunks to maximize context utilization
        chunks = []
        curr_chunk = []
        for seg in final_segments:
            if len(curr_chunk) + len(seg) <= max_len:
                # Fits in current chunk, extend it
                curr_chunk.extend(seg)
            else:
                # Doesn't fit, save current and start new
                if curr_chunk:
                    chunks.append(curr_chunk)
                curr_chunk = seg
        
        # Save final chunk
        if curr_chunk:
            chunks.append(curr_chunk)

        return chunks

    def __len__(self) -> int:
        """Return total number of chunks."""
        return len(self.encoded_poems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a masked sample with dynamic random masking.
        
        Process:
        1. Truncate/pad to max_len, create attention mask
        2. Initialize labels to -100 (ignore index for CrossEntropyLoss)
        3. For each valid token: randomly mask with probability mask_prob
        4. Apply BERT 80/10/10 strategy: 80% [MASK], 10% random, 10% unchanged
        5. Return input_ids, padding_mask, and labels
        
        Returns:
            dict with keys:
                - "input_ids": masked token sequence
                - "padding_mask": attention mask (1=valid, 0=padding)
                - "labels": original token IDs (for masked positions only, -100 elsewhere)
        """
        token_ids = list(self.encoded_poems[idx])
        seq_len = len(token_ids)

        # Truncate and pad to max_len
        if seq_len < self.max_len:
            padding_length = self.max_len - seq_len
            input_ids = token_ids + [self.pad_id] * padding_length
            # Mask: 1 for valid tokens, 0 for padding
            padding_mask = [1] * seq_len + [0] * padding_length
        else:
            input_ids = token_ids[:self.max_len]
            padding_mask = [1] * self.max_len

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.long)

        # Initialize labels: -100 means "ignore this token" in CrossEntropyLoss
        labels = torch.full_like(input_ids, fill_value=-100)
        
        # Apply dynamic masking with BERT 80/10/10 strategy
        for i in range(len(input_ids)):
            if self.mask_prob <= 0.0:
                break  # Skip masking if prob is 0 or negative
            
            token_id = input_ids[i].item()

            # Skip special tokens and padding tokens
            if token_id in self.special_token_ids or padding_mask[i] == 0:
                continue

            # Random decision: mask this token?
            if random.random() < self.mask_prob:
                # Store original token ID in labels
                labels[i] = token_id

                # BERT 80/10/10 masking strategy
                prob = random.random()
                if prob < 0.8:
                    # 80%: Replace with [MASK] token
                    input_ids[i] = self.mask_id
                elif prob < 0.9:
                    # 10%: Replace with random valid token from vocabulary
                    input_ids[i] = random.choice(self.valid_token_ids)
                # 10%: Keep original token unchanged (no action needed)

        return {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "labels": labels
        }