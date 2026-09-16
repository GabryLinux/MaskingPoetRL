import math
import os
import random
from abc import ABC, abstractmethod
from typing import Sequence, Union
from concurrent.futures import ProcessPoolExecutor
import numpy as np


# ============================================================
#  DISTRIBUTIONS FOR CORRUPTION WEIGHTING
# ============================================================

class BaseDistribution(ABC):
    """
    Abstract base for probability distributions used in corruption weighting.
    
    Distributions weight tokens based on position (0 to 1 normalized), enabling
    non-uniform corruption rates across sequences (e.g., more corruption at beginning).
    """
    
    @abstractmethod
    def sample(self, x: float) -> float: 
        """Sample probability weight for normalized position x in [0, 1]."""
        pass
    
    def __call__(self, x: float) -> float: 
        return self.sample(x)


class UniformDistribution(BaseDistribution):
    """Uniform distribution: always returns 1.0 (no position bias)."""
    
    def sample(self, x: float) -> float: 
        return 1.0


class CustomDistribution(BaseDistribution):
    """Custom distribution: wraps arbitrary callable with safe fallback.
    
    Args:
        func: Callable(normalized_position) -> weight. Should return float >= 0.
            On exception, returns 1.0 as fallback.
    """
    
    def __init__(self, func): 
        self.func = func
    
    def sample(self, x: float) -> float:
        try: 
            return max(float(self.func(x)), 0.0)
        except Exception: 
            return 1.0


# ============================================================
#  ABSTRACT CORRUPTION METHOD
# ============================================================

class CorruptionMethod(ABC):
    """
    Abstract interface for all poetry corruption strategies.
    
    Subclasses implement corrupt() which transforms text by:
    - Tokenizing input
    - Selecting positions to corrupt (based on vocab-aware weights)
    - Replacing tokens with random valid alternatives
    - Detokenizing back to text (optional)
    """

    @abstractmethod
    def corrupt(self, text: str) -> Union[str, list[int]]:
        """Apply corruption to text and return as text or token IDs."""
        pass

    def __call__(self, text: str) -> Union[str, list[int]]:
        return self.corrupt(text)

    @staticmethod
    def _get_vocab_info(tokenizer) -> tuple[set, list]:
        """Extract special token IDs and valid token IDs from tokenizer.
        
        Returns:
            (special_ids, valid_ids) where:
            - special_ids: set of token IDs for [PAD], [MASK], [VERSE], etc.
            - valid_ids: list of regular vocabulary token IDs (never special)
        """
        specials = set(tokenizer.get_special_tokens())
        special_ids = {idx for tok, idx in tokenizer.vocab.items() if tok in specials}
        valid_ids = [idx for tok, idx in tokenizer.vocab.items() if idx not in special_ids]
        return special_ids, valid_ids

    @staticmethod
    def _apply_corruption_logic(
        chunk: list[int],
        chunk_idx: int,
        total_chunks: int,
        percentage: float,
        special_ids: set,
        valid_ids: list,
        h_dist: BaseDistribution,
        v_dist: BaseDistribution
    ) -> list[int]:
        """Apply weighted token replacement to a chunk.
        
        Corruption logic:
        1. Get eligible indices (non-special tokens)
        2. Compute vertical weight from chunk position in sequence
        3. Compute horizontal weights from token position in chunk
        4. Compute weighted probability for each eligible token
        5. Select top tokens to corrupt and replace with random valid tokens
        
        Args:
            chunk: Token ID sequence.
            chunk_idx: Position of this chunk (0-based).
            total_chunks: Total chunks in sequence.
            percentage: Fraction of eligible tokens to corrupt.
            special_ids: Token IDs that must never be corrupted.
            valid_ids: Token IDs available for replacement.
            h_dist: Horizontal distribution (position-within-chunk weighting).
            v_dist: Vertical distribution (chunk-position weighting).
        
        Returns:
            Corrupted token sequence (original chunk modified in place).
        """
        corrupted = list(chunk)
        # Find indices of tokens that can be corrupted (non-special)
        eligible_indices = [idx for idx, tok_id in enumerate(chunk) if tok_id not in special_ids]

        if not eligible_indices or percentage <= 0.0:
            return corrupted

        # Vertical weight: chunk position affects corruption rate
        # (e.g., corrupt more in middle chunks, less at edges)
        norm_v_pos = chunk_idx / max(total_chunks, 1)
        v_weight = v_dist.sample(norm_v_pos)

        # Number of tokens to corrupt: percentage * eligible count * vertical weight
        num_to_corrupt = int(len(eligible_indices) * percentage * v_weight)
        num_to_corrupt = min(max(num_to_corrupt, 1 if percentage > 0 else 0), len(eligible_indices))

        if num_to_corrupt == 0:
            return corrupted

        # Horizontal weights: token position within chunk affects selection probability
        chunk_len = len(chunk)
        h_weights = [h_dist.sample(idx / max(chunk_len - 1, 1)) for idx in eligible_indices]

        # Normalize weights to probabilities
        total_weight = sum(h_weights)
        h_probs = [w / total_weight for w in h_weights] if total_weight > 0 else [1.0 / len(eligible_indices)] * len(eligible_indices)

        # Select top tokens using weighted probability sampling
        chosen_indices = np.random.choice(eligible_indices, size=num_to_corrupt, replace=False, p=h_probs)

        # Replace selected tokens with random valid tokens
        for idx in chosen_indices:
            corrupted[idx] = random.choice(valid_ids)

        return corrupted


# ============================================================
#  CONCRETE CORRUPTION METHODS
# ============================================================

class WordCorruptionMethod(CorruptionMethod):
    """Corrupt WordPiece-level tokens with weighted position-based selection."""
    
    def __init__(
        self,
        word_tokenizer,
        percentage: float,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        horizontal_dist: BaseDistribution = UniformDistribution(),
        vertical_dist: BaseDistribution = UniformDistribution(),
        as_text: bool = True
    ):
        """
        Initialize WordPiece corruption.
        
        Args:
            word_tokenizer: WordPiece tokenizer instance.
            percentage: Fraction of tokens to corrupt.
            chunk_idx: Index of this chunk in sequence.
            total_chunks: Total chunks in sequence.
            horizontal_dist: Position-within-token weighting.
            vertical_dist: Chunk-position weighting.
            as_text: If True, detokenize result; else return token IDs.
        """
        self.word_tokenizer = word_tokenizer
        self.percentage = percentage
        self.chunk_idx = chunk_idx
        self.total_chunks = total_chunks
        self.horizontal_dist = horizontal_dist
        self.vertical_dist = vertical_dist
        self.as_text = as_text
        self.special_ids, self.valid_ids = self._get_vocab_info(word_tokenizer)

    def corrupt(self, text: str) -> Union[str, list[int]]:
        """Tokenize, corrupt, and return as text or token IDs."""
        tokens = self.word_tokenizer.tokenize(text)
        corrupted_tokens = self._apply_corruption_logic(
            tokens, self.chunk_idx, self.total_chunks, self.percentage,
            self.special_ids, self.valid_ids, self.horizontal_dist, self.vertical_dist
        )
        if self.as_text:
            return self.word_tokenizer.detokenize(corrupted_tokens, as_text=True, control_tokens=True)
        return corrupted_tokens


class SyllableCorruptionMethod(CorruptionMethod):
    """Corrupt syllable-level tokens with weighted position-based selection."""
    
    def __init__(
        self,
        syl_tokenizer,
        percentage: float,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        horizontal_dist: BaseDistribution = UniformDistribution(),
        vertical_dist: BaseDistribution = UniformDistribution(),
        as_text: bool = True
    ):
        """
        Initialize syllable corruption (same params as WordCorruptionMethod).
        """
        self.syl_tokenizer = syl_tokenizer
        self.percentage = percentage
        self.chunk_idx = chunk_idx
        self.total_chunks = total_chunks
        self.horizontal_dist = horizontal_dist
        self.vertical_dist = vertical_dist
        self.as_text = as_text
        self.special_ids, self.valid_ids = self._get_vocab_info(syl_tokenizer)

    def corrupt(self, text: str) -> Union[str, list[int]]:
        """Tokenize, corrupt, and return as text or token IDs."""
        tokens = self.syl_tokenizer.tokenize(text)
        corrupted_tokens = self._apply_corruption_logic(
            tokens, self.chunk_idx, self.total_chunks, self.percentage,
            self.special_ids, self.valid_ids, self.horizontal_dist, self.vertical_dist
        )
        if self.as_text:
            return self.syl_tokenizer.detokenize(corrupted_tokens, as_text=True, control_tokens=True)
        return corrupted_tokens


class WordAndSyllableCorruptionMethod(CorruptionMethod):
    """Chain corruption: corrupt WordPiece, then syllable level on result.
    
    This creates cascading corruption: words are replaced, then syllables
    within the new words are also replaced, for aggressive augmentation.
    """
    
    def __init__(
        self,
        word_tokenizer,
        syl_tokenizer,
        word_percentage: float,
        syl_percentage: float,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        horizontal_dist: BaseDistribution = UniformDistribution(),
        vertical_dist: BaseDistribution = UniformDistribution(),
        as_text: bool = True
    ):
        """Initialize dual-level corruption.
        
        Args:
            word_tokenizer: WordPiece tokenizer.
            syl_tokenizer: Syllable tokenizer.
            word_percentage: Percentage of words to corrupt.
            syl_percentage: Percentage of syllables to corrupt.
            chunk_idx: Index of the current data chunk.
            total_chunks: Total number of data chunks.
            horizontal_dist: Distribution for horizontal corruption.
            vertical_dist: Distribution for vertical corruption.
            as_text: If True, return text; else return token IDs.
        """
        self.word_method = WordCorruptionMethod(
            word_tokenizer, word_percentage, chunk_idx, total_chunks,
            horizontal_dist, vertical_dist, as_text=True
        )
        self.syl_method = SyllableCorruptionMethod(
            syl_tokenizer, syl_percentage, chunk_idx, total_chunks,
            horizontal_dist, vertical_dist, as_text=as_text
        )

    def corrupt(self, text: str) -> Union[str, list[int]]:
        """Apply word-level corruption, then syllable-level on result."""
        intermediate_text = self.word_method.corrupt(text)
        return self.syl_method.corrupt(intermediate_text)


class ControlTokenDropoutMethod(CorruptionMethod):
    """Randomly remove structural tokens ([VERSE], [STANZA]) from text.
    
    Used to augment fake/corrupted samples by removing poetic markers,
    forcing the model to learn content-based discrimination.
    """
    
    def __init__(self, word_tokenizer, dropout_prob: float = 0.2, as_text: bool = True):
        """
        Initialize dropout for control tokens.
        
        Args:
            word_tokenizer: WordPiece tokenizer.
            dropout_prob: Probability of removing each control token.
            as_text: If True, return text; else return token IDs.
        """
        self.word_tokenizer = word_tokenizer
        self.dropout_prob = dropout_prob
        self.as_text = as_text

        # Collect control token IDs to remove
        control_ids = {word_tokenizer.vocab.get("[VERSE]"), word_tokenizer.vocab.get("[STANZA]")}
        control_ids.discard(None)
        self.control_ids = control_ids

    def corrupt(self, text: str) -> Union[str, list[int]]:
        """Tokenize and randomly drop control tokens."""
        tokens = self.word_tokenizer.tokenize(text)
        # Keep token if: (1) not a control token OR (2) random() > dropout_prob
        corrupted_tokens = [
            tok for tok in tokens
            if tok not in self.control_ids or random.random() > self.dropout_prob
        ]
        if self.as_text:
            return self.word_tokenizer.detokenize(corrupted_tokens, as_text=True, control_tokens=True)
        return corrupted_tokens


# ============================================================
#  MULTIPROCESSING: SERIALIZATION & WORKER MANAGEMENT
# ============================================================

# Global worker state: initialized once per process, not per task
_WORKER_TOKENIZERS: dict = {"word": None, "syl": None}


def _worker_initializer(word_tokenizer_cls, syl_tokenizer_cls):
    """Initialize tokenizers once per worker process (not per task).
    
    This runs once at pool startup, creating tokenizer instances from their
    class constructors. This avoids pickling tokenizers with every task.
    
    Args:
        word_tokenizer_cls: WordPiece tokenizer class (has from_config()).
        syl_tokenizer_cls: Syllable tokenizer class (has from_config()).
    """
    global _WORKER_TOKENIZERS
    _WORKER_TOKENIZERS["word"] = (
        word_tokenizer_cls.from_config() if word_tokenizer_cls is not None else None
    )
    _WORKER_TOKENIZERS["syl"] = (
        syl_tokenizer_cls.from_config() if syl_tokenizer_cls is not None else None
    )


def _method_to_primitive(method: CorruptionMethod) -> tuple[str, dict]:
    """Convert CorruptionMethod instance to primitives for serialization.
    
    Returns tuple (method_type_string, kwargs_dict) containing only:
    - Strings, floats, ints, bools (fully serializable)
    
    Avoids pickling tokenizers, distributions, or any complex objects.
    
    Note: CustomDistribution lambdas cannot be serialized; use UniformDistribution
    in parallel mode.
    
    Args:
        method: CorruptionMethod instance to serialize.
    
    Returns:
        (method_type, kwargs) where method_type is used by worker to reconstruct.
    """
    if isinstance(method, ControlTokenDropoutMethod):
        return "dropout", {
            "dropout_prob": float(method.dropout_prob),
            "as_text": bool(method.as_text),
        }

    if isinstance(method, WordAndSyllableCorruptionMethod):
        return "word_syl", {
            "word_percentage": float(method.word_method.percentage),
            "syl_percentage": float(method.syl_method.percentage),
            "chunk_idx": int(method.word_method.chunk_idx),
            "total_chunks": int(method.word_method.total_chunks),
            "as_text": bool(method.syl_method.as_text),
        }

    if isinstance(method, WordCorruptionMethod):
        return "word", {
            "percentage": float(method.percentage),
            "chunk_idx": int(method.chunk_idx),
            "total_chunks": int(method.total_chunks),
            "as_text": bool(method.as_text),
        }

    if isinstance(method, SyllableCorruptionMethod):
        return "syl", {
            "percentage": float(method.percentage),
            "chunk_idx": int(method.chunk_idx),
            "total_chunks": int(method.total_chunks),
            "as_text": bool(method.as_text),
        }

    raise TypeError(f"Unsupported CorruptionMethod type: {type(method).__name__}")


def _infer_tokenizer_classes(tasks: Sequence[tuple]) -> tuple:
    """Infer tokenizer classes from tasks to auto-detect worker requirements.
    
    Iterates tasks and extracts concrete tokenizer classes used by methods.
    Used when tokenizer classes not explicitly provided to parallel functions.
    
    Args:
        tasks: Sequence of (text, CorruptionMethod) tuples.
    
    Returns:
        (word_tokenizer_cls, syl_tokenizer_cls) - one or both may be None.
    """
    word_cls = None
    syl_cls = None
    for _, method in tasks:
        if isinstance(method, WordAndSyllableCorruptionMethod):
            word_cls = word_cls or type(method.word_method.word_tokenizer)
            syl_cls = syl_cls or type(method.syl_method.syl_tokenizer)
        elif isinstance(method, (WordCorruptionMethod, ControlTokenDropoutMethod)):
            word_cls = word_cls or type(method.word_tokenizer)
        elif isinstance(method, SyllableCorruptionMethod):
            syl_cls = syl_cls or type(method.syl_tokenizer)
        if word_cls is not None and syl_cls is not None:
            break
    return word_cls, syl_cls


def _worker_process_task(task: tuple) -> Union[str, list[int]]:
    """Execute corruption task in worker process.
    
    Receives only primitives (text, method_type string, kwargs dict).
    Reconstructs CorruptionMethod from globals and kwargs, then executes.
    
    Args:
        task: (text, method_type, kwargs) - all primitives, fully serializable.
    
    Returns:
        Corrupted text (as text or token IDs).
    """
    text, method_type, kwargs = task
    word_tok = _WORKER_TOKENIZERS["word"]
    syl_tok = _WORKER_TOKENIZERS["syl"]

    # Reconstruct method from type and kwargs
    if method_type == "dropout":
        method = ControlTokenDropoutMethod(word_tok, **kwargs)
    elif method_type == "word":
        method = WordCorruptionMethod(word_tok, **kwargs)
    elif method_type == "syl":
        method = SyllableCorruptionMethod(syl_tok, **kwargs)
    elif method_type == "word_syl":
        method = WordAndSyllableCorruptionMethod(word_tok, syl_tok, **kwargs)
    else:
        raise ValueError(f"Unknown method type: {method_type!r}")

    return method.corrupt(text)


def parallelized_corruption_generation(
    tasks: Sequence[tuple[str, CorruptionMethod]],
    num_workers: int | None = None,
    word_tokenizer_cls=None,
    syl_tokenizer_cls=None,
) -> list[Union[str, list[int]]]:
    """Execute corruption in parallel on multiple processes.
    
    Design:
    - Tokenizers: not serialized per task, created once per worker
    - Tasks: contain only primitives (strings, floats, ints, bools)
    - CorruptionMethod instances: converted to primitive descriptors
    
    Args:
        tasks: Sequence of (text, CorruptionMethod) tuples.
        num_workers: Number of processes. Defaults to CPU count or 4.
        word_tokenizer_cls: WordPiece tokenizer class. Auto-inferred if None.
        syl_tokenizer_cls: Syllable tokenizer class. Auto-inferred if None.
    
    Returns:
        list: Corrupted texts/tokens in same order as input tasks.
    """
    workers = num_workers or (os.cpu_count() or 4)

    # Auto-infer tokenizer classes if not provided
    if word_tokenizer_cls is None or syl_tokenizer_cls is None:
        inferred_word, inferred_syl = _infer_tokenizer_classes(tasks)
        word_tokenizer_cls = word_tokenizer_cls or inferred_word
        syl_tokenizer_cls = syl_tokenizer_cls or inferred_syl

    # Convert CorruptionMethod instances to primitives
    primitive_tasks = [
        (text, *_method_to_primitive(method))
        for text, method in tasks
    ]

    print(
        f"[INFO] Starting parallel corruption on {len(primitive_tasks)} items "
        f"with {workers} workers (word_tok={word_tokenizer_cls.__name__ if word_tokenizer_cls else None}, "
        f"syl_tok={syl_tokenizer_cls.__name__ if syl_tokenizer_cls else None})..."
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initializer,
        initargs=(word_tokenizer_cls, syl_tokenizer_cls),
    ) as executor:
        results_list = list(executor.map(_worker_process_task, primitive_tasks))

    print(f"[SUCCESS] Parallel corruption completed! {len(results_list)} items processed.")
    return results_list


def sequential_corruption_generation(
    tasks: Sequence[tuple[str, CorruptionMethod]]
) -> list[Union[str, list[int]]]:
    """Execute corruption sequentially in main process (single-threaded).
    
    Advantage: supports any CorruptionMethod including custom lambdas
    in CustomDistribution (not picklable).
    
    Args:
        tasks: Sequence of (text, CorruptionMethod) tuples.
    
    Returns:
        list: Corrupted texts/tokens in same order as input tasks.
    """
    results = []
    for idx, (text, method) in enumerate(tasks, 1):
        print(f"[INFO] Sequential processing: {idx}/{len(tasks)}", end="\r")
        results.append(method.corrupt(text))
    print(f"\n[SUCCESS] Sequential corruption completed! {len(results)} items processed.")
    return results


# ============================================================
#  UTILITY FUNCTIONS: LABELLING & CONTROL TOKEN INJECTION
# ============================================================

def controlTokenGeneration(
    max_token_per_sequence: int,
    num_sequences: int,
    control_tokens: Sequence[str] = ("[VERSE]", "[STANZA]"),
    min_token_per_sequence: int = 1,
    verse_ratio: float = 0.8,
    seed: int | None = None,
) -> list[str]:
    """Generate sequences of control tokens for hard negative mining.
    
    Creates synthetic "fake" sequences containing only control tokens.
    Trains classifier that control tokens alone are insufficient for poetry.
    
    Args:
        max_token_per_sequence: Maximum tokens per generated sequence.
        num_sequences: Number of sequences to generate.
        control_tokens: Available control tokens. First is primary (frequent).
        min_token_per_sequence: Minimum tokens per sequence.
        verse_ratio: Probability of primary token (vs others).
        seed: Optional seed for reproducibility.
    
    Returns:
        list: Strings of space-separated control tokens.
        
    Raises:
        ValueError: If parameters invalid.
    """
    if max_token_per_sequence < 1:
        raise ValueError("max_token_per_sequence must be >= 1")
    if min_token_per_sequence < 1:
        raise ValueError("min_token_per_sequence must be >= 1")
    if min_token_per_sequence > max_token_per_sequence:
        raise ValueError("min_token_per_sequence > max_token_per_sequence")
    if num_sequences < 0:
        raise ValueError("num_sequences must be >= 0")
    if not control_tokens:
        raise ValueError("control_tokens cannot be empty")
    if not 0.0 <= verse_ratio <= 1.0:
        raise ValueError("verse_ratio must be in [0, 1]")

    rng = random.Random(seed)
    primary, *others = control_tokens

    sequences: list[str] = []
    for _ in range(num_sequences):
        n = rng.randint(min_token_per_sequence, max_token_per_sequence)
        toks = []
        for _ in range(n):
            if others and rng.random() > verse_ratio:
                # Sample from secondary tokens with probability (1 - verse_ratio)
                toks.append(rng.choice(others))
            else:
                # Use primary token with probability verse_ratio
                toks.append(primary)
        sequences.append(" ".join(toks))
    return sequences


def nonLinearLabelling(
    corruption_percentage: float,
    rate: float = 2.0,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> float:
    """Map corruption percentage to poeticity label via exponential decay.
    
    Label = floor + (ceil - floor) * exp(-rate * corruption_percentage)
    
    Examples (rate=2, floor=0, ceil=1):
    - corruption=0.0 → label=1.00 (pristine)
    - corruption=0.2 → label=0.67
    - corruption=0.5 → label=0.37
    - corruption=1.0 → label=0.14 (heavily corrupted)
    
    Args:
        corruption_percentage: Fraction corrupted (0 to 1).
        rate: Decay rate (higher = faster decay).
        floor: Label value at 100% corruption.
        ceil: Label value at 0% corruption.
    
    Returns:
        float: Label in [floor, ceil].
        
    Raises:
        ValueError: If parameters invalid.
    """
    if not 0.0 <= corruption_percentage <= 1.0:
        raise ValueError("corruption_percentage must be in [0, 1]")
    if rate < 0:
        raise ValueError("rate must be >= 0")
    if not 0.0 <= floor <= ceil <= 1.0:
        raise ValueError("Required: 0 <= floor <= ceil <= 1")

    decay = math.exp(-rate * corruption_percentage)
    return float(floor + (ceil - floor) * decay)


def dropControlTokens(
    text: str,
    word_tokenizer,
    dropout_prob: float = 0.5,
) -> str:
    """Randomly remove control tokens from text.
    
    Args:
        text: Input text.
        word_tokenizer: WordPiece tokenizer.
        dropout_prob: Probability of removing each control token (0 to 1).
    
    Returns:
        Text with control tokens stochastically removed.
    """
    if dropout_prob <= 0.0:
        return text
    method = ControlTokenDropoutMethod(
        word_tokenizer, dropout_prob=min(dropout_prob, 1.0), as_text=True
    )
    return method.corrupt(text)


def injectControlTokens(
    text: str,
    word_tokenizer,
    n_insert: int | None = None,
    control_token: str = "[VERSE]",
    seed: int | None = None,
) -> str:
    """Insert control tokens at random positions in text.
    
    Args:
        text: Input text.
        word_tokenizer: WordPiece tokenizer.
        n_insert: Number of insertions. If None, inserts ~1 per 8 tokens.
        control_token: Which control token to inject.
        seed: Optional seed for reproducibility.
    
    Returns:
        Text with injected control tokens.
    """
    rng = random.Random(seed)
    tokens = list(word_tokenizer.tokenize(text))
    if not tokens:
        return control_token

    control_id = word_tokenizer.vocab.get(control_token)
    if control_id is None:
        return text

    if n_insert is None:
        # Default: ~1 insertion per 8 tokens
        n_insert = max(1, len(tokens) // 8)

    # Insert control token at random positions
    for _ in range(n_insert):
        pos = rng.randint(0, len(tokens))
        tokens.insert(pos, control_id)

    return word_tokenizer.detokenize(tokens, as_text=True, control_tokens=True)