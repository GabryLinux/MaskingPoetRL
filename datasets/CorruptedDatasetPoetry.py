import os
import random
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset

from Tokenizers.tokenizer import SyllableTokenizer, WordPieceTokenizer
from datasets.MaskedDatasetPoetry import DynamicMaskedPoetryDataset

# ============================================================
#  GLOBAL TOKENIZERS FOR MULTIPROCESSING WORKERS
# ============================================================

# These globals are initialized once per CPU worker process
_worker_word_tok : Union[WordPieceTokenizer, None] = None
_worker_syl_tok : Union[SyllableTokenizer, None] = None


def _init_tokenizer_worker():
    """Initialize tokenizers for a single multiprocessing worker.
    
    This initializer is called once per worker process (not per task).
    Ensures each worker has its own tokenizer instances to avoid pickling issues.
    """
    global _worker_word_tok, _worker_syl_tok
    _worker_word_tok = WordPieceTokenizer.from_config()
    _worker_syl_tok = SyllableTokenizer.from_config()


def _process_single_poem_task(args: tuple[str, int]) -> list[dict[str, Any]]:
    """Tokenize and chunk a single poem in a worker process.
    
    Process:
    1. Tokenize poem with WordPiece
    2. Chunk based on stanza/verse boundaries (preserves structure)
    3. Reconstruct text from each chunk
    4. Tokenize reconstructed text with Syllable tokenizer
    5. Return list of chunks with both tokenization formats
    
    Args:
        args: Tuple of (poem_text, max_len) where max_len is chunk size limit.
    
    Returns:
        list[dict]: Each dict contains:
            - "text": reconstructed chunk text
            - "word_tokens": WordPiece token sequence
            - "syl_tokens": Syllable token sequence
    
    Raises:
        RuntimeError: If tokenizers not initialized in worker.
    """
    poem, max_len = args
    global _worker_word_tok, _worker_syl_tok

    if _worker_word_tok is None or _worker_syl_tok is None:
        raise RuntimeError(
            "Tokenizer worker not initialized. Ensure _init_tokenizer_worker() "
            "is called in worker process."
        )

    # 1. Tokenize with WordPiece and identify structural markers
    word_tokens = _worker_word_tok.tokenize(poem)
    word_stanza_id = _worker_word_tok.vocab.get("[STANZA]", 0)
    word_verse_id = _worker_word_tok.vocab.get("[VERSE]", 0)
    
    # Chunk respecting poetic structure (stanzas and verses)
    word_chunks = DynamicMaskedPoetryDataset._poem_chunking(
        word_tokens, max_len, word_stanza_id, word_verse_id
    )

    processed_chunks = []

    for w_chunk in word_chunks:
        # 2. Reconstruct exact chunk text from tokens
        chunk_text = _worker_word_tok.detokenize(w_chunk, as_text=True, control_tokens=True)

        # 3. Tokenize reconstructed text with Syllable tokenizer
        # Remove control tokens before syllable tokenization (different vocab)
        clean_text_for_syl = chunk_text.replace("[VERSE]", "").replace("[STANZA]", "")
        syl_tokens = _worker_syl_tok.tokenize(clean_text_for_syl)

        processed_chunks.append({
            "text": chunk_text,
            "word_tokens": w_chunk,
            "syl_tokens": syl_tokens,
        })

    return processed_chunks


# ============================================================
#  DATASET CLASS
# ============================================================

class CorruptedDatasetPoetry(Dataset):
    """PyTorch Dataset for poetry with parallel pre-tokenization and dual modality support 
    (Single-threaded or Multi-threaded).
    
    
    Args:
        poems: List of poetry texts.
        word_tokenizer: WordPiece tokenizer instance.
        syl_tokenizer: Syllable tokenizer instance.
        max_len: Maximum chunk length. Defaults to 256.
        max_chunks: Limit total chunks. Defaults to None (no limit).
        num_workers: Number of CPU workers for parallel processing.
            Defaults to CPU count or 4.
    """
    
    def __init__(
        self, 
        poems: list[str], 
        word_tokenizer,
        syl_tokenizer,
        max_len: int = 256,
        max_chunks: int | None = None, 
        num_workers: Optional[int] = None
    ):
        """Initialize dataset with parallel pre-tokenization."""
        super().__init__()
        
        self.word_tokenizer = word_tokenizer
        self.syl_tokenizer = syl_tokenizer
        self.max_len = max_len
        self.max_chunks = max_chunks

        # Extract special token IDs for WordPiece (e.g., [PAD], [MASK], [CLS])
        word_specials = set(word_tokenizer.get_special_tokens())
        self.word_special_ids = {idx for tok, idx in word_tokenizer.vocab.items() if tok in word_specials}
        # Valid tokens = all tokens except special tokens (used for corruption)
        self.word_valid_ids = [idx for tok, idx in word_tokenizer.vocab.items() if idx not in self.word_special_ids]
        self.word_pad_id = word_tokenizer.vocab.get("[PAD]", 0)

        # Extract special token IDs for Syllable tokenizer
        syl_specials = set(syl_tokenizer.get_special_tokens())
        self.syl_special_ids = {idx for tok, idx in syl_tokenizer.vocab.items() if tok in syl_specials}
        # Valid tokens for syllable modality
        self.syl_valid_ids = [idx for tok, idx in syl_tokenizer.vocab.items() if idx not in self.syl_special_ids]
        self.syl_pad_id = syl_tokenizer.vocab.get("[PAD]", 0)

        # Parallel pre-tokenization and chunking
        workers = num_workers or (os.cpu_count() or 4)
        print(f"[INFO] Parallel pre-tokenization on {workers} CPU cores...")

        tasks = [(poem, self.max_len) for poem in poems]
        self.encoded_poems = []

        if workers == 1:
            # Single-threaded: initialize in main process
            _init_tokenizer_worker()
            for task in tasks:
                poem_chunks = _process_single_poem_task(task)
                self.encoded_poems.extend(poem_chunks)
        else:
            # Multi-threaded: each worker gets tokenizers via initializer
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer_worker) as executor:
                results = executor.map(_process_single_poem_task, tasks)
                for poem_chunks in results:
                    self.encoded_poems.extend(poem_chunks)

        # Shuffle and limit chunks
        random.shuffle(self.encoded_poems)
        self.encoded_poems = self.encoded_poems[:self.max_chunks] if self.max_chunks else self.encoded_poems
        print(f"[SUCCESS] Dataset ready! Total chunks: {len(self.encoded_poems)}")

    @staticmethod
    def _poem_chunking_static(
        token_ids: list[int], 
        max_len: int, 
        stanza_id: int, 
        verse_id: int
    ) -> list[list[int]]:
        """Chunk token sequence respecting poetic structure.
        
        Strategy:
        1. Split on stanza markers first (preserve stanzas)
        2. Split stanzas on verse markers if too long
        3. Further split by max_len if needed
        4. Merge adjacent segments to fill chunks efficiently
        
        Args:
            token_ids: Complete token sequence.
            max_len: Maximum tokens per chunk.
            stanza_id: Token ID for [STANZA] marker.
            verse_id: Token ID for [VERSE] marker.
        
        Returns:
            list[list[int]]: Chunks of max length, respecting structure.
        """
        # If already short enough, return as single chunk
        if len(token_ids) <= max_len:
            return [token_ids]

        # Step 1: Split on stanzas
        stanzas, curr = [], []
        for tok in token_ids:
            curr.append(tok)
            if tok == stanza_id:
                stanzas.append(curr)
                curr = []
        if curr: 
            stanzas.append(curr)

        # Step 2: Split long stanzas on verses
        segments = []
        for stanza in stanzas:
            if len(stanza) <= max_len:
                segments.append(stanza)
            else:
                # Split stanza by verses
                curr_verse = []
                for tok in stanza:
                    curr_verse.append(tok)
                    if tok == verse_id:
                        segments.append(curr_verse)
                        curr_verse = []
                if curr_verse: 
                    segments.append(curr_verse)
        
        # Step 3: Further split segments longer than max_len
        final_segments = []
        for seg in segments:
            for i in range(0, len(seg), max_len):
                final_segments.append(seg[i:i+max_len])

        # Step 4: Merge segments into chunks, packing efficiently
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
        """Return total number of chunks in dataset."""
        return len(self.encoded_poems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample with padding and masks.
        
        For each modality (WordPiece and Syllable):
        1. Truncate to max_len
        2. Pad with [PAD] tokens to reach max_len
        3. Create attention mask (1 for valid, 0 for padding)
        
        Returns:
            dict with keys:
                - "text": original chunk text
                - "word_input_ids": corrupted WordPiece tokens, padded
                - "word_padding_mask": attention mask (1=valid, 0=pad)
                - "word_labels": original WordPiece tokens, padded
                - "syl_input_ids": corrupted syllable tokens, padded
                - "syl_padding_mask": attention mask
                - "syl_labels": original syllable tokens, padded
        """
        item = self.encoded_poems[idx]
        
        # Get original and corrupted token sequences
        word_orig = item["word_tokens"]
        word_corrupted = item.get("word_corrupted", word_orig)
        
        syl_orig = item["syl_tokens"]
        syl_corrupted = item.get("syl_corrupted", syl_orig)

        # Helper: truncate and pad sequence to max_len
        def pad_sequence(seq: list[int], pad_id: int):
            # Truncate to max_len
            seq = seq[:self.max_len]
            pad_len = self.max_len - len(seq)
            
            if pad_len > 0:
                # Mask: 1 for original tokens, 0 for padding
                mask = [1] * len(seq) + [0] * pad_len
                seq_padded = seq + [pad_id] * pad_len
            else:
                # No padding needed
                mask = [1] * self.max_len
                seq_padded = seq
                
            return seq_padded, mask

        # Pad both modalities
        word_corrupted_padded, word_mask = pad_sequence(word_corrupted, self.word_pad_id)
        word_orig_padded, _ = pad_sequence(word_orig, self.word_pad_id)

        syl_corrupted_padded, syl_mask = pad_sequence(syl_corrupted, self.syl_pad_id)
        syl_orig_padded, _ = pad_sequence(syl_orig, self.syl_pad_id)

        # Convert to PyTorch tensors for DataLoader
        return {
            "text": item.get("text", ""),
            # WordPiece modality
            "word_input_ids": torch.tensor(word_corrupted_padded, dtype=torch.long),
            "word_padding_mask": torch.tensor(word_mask, dtype=torch.long),
            "word_labels": torch.tensor(word_orig_padded, dtype=torch.long),
            # Syllable modality
            "syl_input_ids": torch.tensor(syl_corrupted_padded, dtype=torch.long),
            "syl_padding_mask": torch.tensor(syl_mask, dtype=torch.long),
            "syl_labels": torch.tensor(syl_orig_padded, dtype=torch.long),
        }