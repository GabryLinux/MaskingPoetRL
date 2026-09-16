from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set

import torch
from torch.distributions import Categorical

from Tokenizers.tokenizer import BasePoetryTokenizer
from models.MaskingPoetRL import ParallelMaskingPoets, ParallelMaskingPoetsEnv


@dataclass
class TrajectoryData:
    """Stores trajectory details collected during a single RL episode.

    Attributes:
        syl_log_probs (list): Scalar tensors (with gradients) for syllable action log probabilities.
        word_log_probs (list): Scalar tensors (with gradients) for WordPiece action log probabilities.
        syl_actions (list): Integer actions chosen by syllable policy.
        word_actions (list): Integer actions chosen by WordPiece policy.
        syl_dist (list[Categorical]): Syllable policy Categorical distributions per step.
        word_dist (list[Categorical]): WordPiece policy Categorical distributions per step.
        syl_rewards (list): Float rewards received by syllable policy.
        word_rewards (list): Float rewards received by WordPiece policy.
        steps_data (list): Per-step diagnostic and telemetry dictionaries.
        terminated (bool): True if episode ended via terminal signal.
        max_steps_per_episode (int): Maximum step limit configuration.
        trajectory_len (int): Total steps executed in trajectory.
        final_syl_state (Any): Final hidden state for syllable policy.
        final_word_state (Any): Final hidden state for WordPiece policy.
        word_wins (int): Count of steps won by WordPiece model.
        syl_wins (int): Count of steps won by syllable model.
    """
    syl_log_probs: list = field(default_factory=list)
    word_log_probs: list = field(default_factory=list)
    syl_actions: list = field(default_factory=list)
    word_actions: list = field(default_factory=list)
    syl_dist: list[Categorical] = field(default_factory=list)
    word_dist: list[Categorical] = field(default_factory=list)

    syl_rewards: list = field(default_factory=list)
    word_rewards: list = field(default_factory=list)

    steps_data: list = field(default_factory=list)

    terminated: bool = False
    max_steps_per_episode: int = 0
    trajectory_len: int = 0
    final_syl_state: Any = None
    final_word_state: Any = None

    word_wins: int = 0
    syl_wins: int = 0


@dataclass
class UpdateResult:
    """Stores computed returns, total losses, and per-step loss tensors after a trajectory update.

    Attributes:
        syl_loss (float): Scalar loss value for syllable policy update.
        word_loss (float): Scalar loss value for WordPiece policy update.
        syl_returns (list): List of computed return values for syllable policy steps.
        word_returns (list): List of computed return values for WordPiece policy steps.
        syl_loss_terms (Any): Detached PyTorch tensor with individual step losses for syllable policy.
        word_loss_terms (Any): Detached PyTorch tensor with individual step losses for WordPiece policy.
        trajectory_len (int): Total number of steps in trajectory.
    """
    syl_loss: float = 0.0
    word_loss: float = 0.0
    syl_returns: list = field(default_factory=list)
    word_returns: list = field(default_factory=list)
    syl_loss_terms: Any = None
    word_loss_terms: Any = None
    trajectory_len: int = 0


@dataclass
class TrainingParameters:
    """Configuration container holding parameters for RL training runs.

    Attributes:
        max_steps_per_episode (int): Maximum step limit per episode.
        max_chunks_per_training (int): Maximum training iterations.
        epochs (int): Epoch count.
        lr_wordpiece (float): Optimizer learning rate for WordPiece policy.
        lr_syllable (float): Optimizer learning rate for Syllable policy.
        gamma (float): Discount factor for reward computation.
        batch_size_rl (int): RL trajectory batch size.
        log_interval (int): Logging frequency threshold.
        cutoff_threshold_word (float): Stopping condition threshold for WordPiece classifier.
        cutoff_threshold_syllable (float): Stopping condition threshold for syllable classifier.
        top_k (int): Top-K sampling/masking parameter.
    """
    max_steps_per_episode: int
    max_chunks_per_training: int
    epochs: int
    lr_wordpiece: float
    lr_syllable: float
    gamma: float
    batch_size_rl: int
    log_interval: int
    cutoff_threshold_word: float
    cutoff_threshold_syllable: float
    top_k: int


def unpadding_text(tokens: list[int], tokenizer: BasePoetryTokenizer) -> list[int]:
    """Strips padding token IDs from a given token sequence.

    Args:
        tokens (list[int]): Raw sequence of token IDs including padding.
        tokenizer (BasePoetryTokenizer): Tokenizer containing vocabulary map.

    Returns:
        list[int]: Sequence of token IDs excluding padding tokens.
    """
    pad_id = tokenizer.vocab.get("[PAD]", 0)
    return [t for t in tokens if t != pad_id]


def _move_state_to_device(state: Any, device: torch.device) -> Any:
    """Transfers hidden state representations to a specified target compute device.

    Handles single PyTorch tensors or multi-layer hidden state structures (lists/tuples of tensors).

    Args:
        state (Any): Hidden state tensor or list/tuple of tensors representing multi-layer states.
        device (torch.device): PyTorch compute target device (e.g., 'cuda', 'cpu').

    Returns:
        Any: Hidden state structured identically to input with tensors transferred to target device.
    """
    if isinstance(state, (list, tuple)):
        # Recursively move multi-layer hidden states (list of tensors) asynchronously
        return [t.to(device, non_blocking=True) for t in state]
    elif isinstance(state, torch.Tensor):
        # Move single hidden state tensor asynchronously
        return state.to(device, non_blocking=True)
    return state


def _init_trajectory(
    batch: dict[str, torch.Tensor],
    word_tokenizer: BasePoetryTokenizer,
    syl_tokenizer: BasePoetryTokenizer,
    env: ParallelMaskingPoetsEnv,
    device: torch.device,
) -> dict[str, Any]:
    """Initializes environment, unpads input tokens, and transfers initial multi-layer states and masks to device.

    Args:
        batch (dict[str, torch.Tensor]): Dictionary containing token inputs ('word_input_ids', 'syl_input_ids').
        word_tokenizer (BasePoetryTokenizer): Tokenizer for WordPiece model.
        syl_tokenizer (BasePoetryTokenizer): Tokenizer for Syllable model.
        env (ParallelMaskingPoetsEnv): Reinforcement Learning environment.
        device (torch.device): PyTorch compute device target.

    Returns:
        dict[str, Any]: Dictionary containing device-allocated multi-layer states, mask tensors,
            unpadded sequences, reset dictionary, and control token IDs.
    """
    # 1. Retrieve control token IDs for masking logic
    syl_verse_id = syl_tokenizer.vocab.get("[VERSE]", 1)
    word_verse_id = word_tokenizer.vocab.get("[VERSE]", 1)

    # 2. Extract sequences from batch tensors and remove padding
    clean_word_toks = unpadding_text(batch["word_input_ids"][0].tolist(), word_tokenizer)
    clean_syl_toks = unpadding_text(batch["syl_input_ids"][0].tolist(), syl_tokenizer)

    # 3. Environment reset
    (word_state, syl_state), reset_info = env.reset(options={
        "word_tokens": clean_word_toks,
        "syl_tokens": clean_syl_toks,
    })

    # 4. Asynchronously transfer multi-layer states (Shape: list of (1, seq_len, hidden_dim) tensors) to target device
    word_state = _move_state_to_device(word_state, device)
    syl_state = _move_state_to_device(syl_state, device)

    # 5. Transfer binary attention masks (Shape: (1, seq_len)) to target device
    word_mask = reset_info["word_mask"].to(device, non_blocking=True)
    syl_mask = reset_info["syl_mask"].to(device, non_blocking=True)

    return {
        "word_state": word_state,
        "syl_state": syl_state,
        "word_mask": word_mask,
        "syl_mask": syl_mask,
        "clean_word_toks": clean_word_toks,
        "clean_syl_toks": clean_syl_toks,
        "reset_info": reset_info,
        "syl_verse_id": syl_verse_id,
        "word_verse_id": word_verse_id,
    }


def _masked_distribution(
    dist: Categorical,
    padding_mask: torch.Tensor | None,
    current_tokens: list[int],
    control_ids: set[int],
    ctx_len: int,
) -> Categorical:
    """Applies logit masking (-1e9) to prevent model from selecting padding tokens or protected control tokens.

    Args:
        dist (Categorical): Unmasked Categorical probability distribution over action indices.
        padding_mask (torch.Tensor | None): Binary tensor mask (1 for valid, 0 for padded tokens). Shape: (1, seq_len).
        current_tokens (list[int]): Sequence of active sequence token IDs.
        control_ids (set[int]): Set containing special control token IDs (e.g. [VERSE]) to mask out.
        ctx_len (int): Length of current input context window.

    Returns:
        Categorical: New Categorical distribution constructed with masked logits.
    """
    # Clone underlying logits tensor to prevent in-place operations on computational graph
    logits = dist.logits.clone()

    # 1. Mask PAD positions: Set logits to -1e9 where padding_mask == 0
    if padding_mask is not None:
        logits.masked_fill_(padding_mask.view_as(logits) == 0, -1e9)

    # 2. Mask control token positions located within current active sequence context
    if control_ids:
        for idx, tok_id in enumerate(current_tokens[:ctx_len]):
            if tok_id in control_ids:
                logits[0, idx] = -1e9

    return Categorical(logits=logits)


def collect_trajectory(
    env: ParallelMaskingPoetsEnv,
    poets: ParallelMaskingPoets,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    max_steps_per_episode: int,
    gamma: float,
    word_tokenizer: BasePoetryTokenizer,
    syl_tokenizer: BasePoetryTokenizer,
) -> TrajectoryData:
    """Executes a full RL rollout episode with distribution action masking and multi-layer state propagation.

    Args:
        env (ParallelMaskingPoetsEnv): RL environment instance managing step dynamics.
        poets (ParallelMaskingPoets): Combined policy network model.
        batch (dict[str, torch.Tensor]): Dictionary of input batch tensors.
        device (torch.device): PyTorch target device.
        max_steps_per_episode (int): Maximum allowed steps before termination.
        gamma (float): Reward discount factor.
        word_tokenizer (BasePoetryTokenizer): WordPiece tokenizer instance.
        syl_tokenizer (BasePoetryTokenizer): Syllable tokenizer instance.

    Returns:
        TrajectoryData: Completed trajectory container containing step histories, action distributions, and states.
    """
    # Initialize trajectory environment and state data
    traj_data = _init_trajectory(batch, word_tokenizer, syl_tokenizer, env, device)
    word_state = traj_data["word_state"]
    syl_state = traj_data["syl_state"]

    word_mask = traj_data["word_mask"]
    syl_mask = traj_data["syl_mask"]

    syl_verse_id = traj_data["syl_verse_id"]
    word_verse_id = traj_data["word_verse_id"]
    
    traj = TrajectoryData()
    terminated = False

    for step in range(max_steps_per_episode):
        # A. Model forward pass using multi-layer states -> returns unmasked action distributions
        syl_dist, word_dist = poets.evaluate( 
            syl_states=syl_state,
            word_states=word_state,
            syl_pad_mask=syl_mask,
            word_pad_mask=word_mask
        )

        # B. Apply masking for padding and special control tokens
        masked_syl_dist = _masked_distribution(
            dist=syl_dist,
            padding_mask=syl_mask,
            current_tokens=traj_data["clean_syl_toks"],
            control_ids={syl_verse_id},
            ctx_len=len(traj_data["clean_syl_toks"]),
        )
        masked_word_dist = _masked_distribution(
            dist=word_dist,
            padding_mask=word_mask,
            current_tokens=traj_data["clean_word_toks"],
            control_ids={word_verse_id},
            ctx_len=len(traj_data["clean_word_toks"]),
        )

        # C. Sample action indices from masked Categorical distributions (0D Tensors)
        syl_action_t = masked_syl_dist.sample() 
        word_action_t = masked_word_dist.sample()

        # Extract log-probabilities retaining Autograd computational graph
        syl_log_prob = masked_syl_dist.log_prob(syl_action_t)  
        word_log_prob = masked_word_dist.log_prob(word_action_t)

        # Convert 0D action tensors to scalar integers for environment interface
        syl_action = int(syl_action_t.item())
        word_action = int(word_action_t.item())

        # D. Step environment
        (next_word_state, next_syl_state), (reward_syl, reward_word), terminated, _, info = env.step(
            (syl_action, word_action)
        )

        # Record step output data into trajectory container
        traj.syl_log_probs.append(syl_log_prob)
        traj.word_log_probs.append(word_log_prob)
        traj.syl_actions.append(syl_action)
        traj.word_actions.append(word_action)
        traj.syl_rewards.append(reward_syl)
        traj.word_rewards.append(reward_word)
        traj.syl_dist.append(masked_syl_dist)
        traj.word_dist.append(masked_word_dist)

        # Step diagnostics telemetry
        syl_score = float(info["abs_score_syl"])
        word_score = float(info["abs_score_word"])
        traj.steps_data.append({
            "step": step,
            "syl_action": syl_action,
            "word_action": word_action,
            "reward_syl": float(reward_syl),
            "reward_word": float(reward_word),
            "current_text": info["current_text"],
            "syl_classifier_score": syl_score,
            "word_classifier_score": word_score,
            "avg_classifier_score": (syl_score + word_score) / 2.0,
        })

        # E. Move updated multi-layer hidden states and masks to target device
        word_state = _move_state_to_device(next_word_state, device)
        syl_state = _move_state_to_device(next_syl_state, device)

        word_mask = info.get("word_mask")
        syl_mask = info.get("syl_mask")
        if word_mask is not None:
            word_mask = word_mask.to(device, non_blocking=True)
        if syl_mask is not None:
            syl_mask = syl_mask.to(device, non_blocking=True)

        if terminated:
            break

    traj.trajectory_len = len(traj.syl_rewards)
    traj.terminated = terminated
    traj.max_steps_per_episode = max_steps_per_episode
    traj.final_syl_state = syl_state
    traj.final_word_state = word_state

    return traj


class EntropyScheduler:
    """Schedules policy entropy regularization weight (beta) decay over training episodes."""

    def __init__(self, beta_start: float, beta_end: float, total_episodes: int, decay_type: str = "linear"):
        """Initializes the entropy scheduler.

        Args:
            beta_start (float): Initial starting value for entropy coefficient beta.
            beta_end (float): Final minimum target value for entropy coefficient beta.
            total_episodes (int): Total number of episode steps for decay progression.
            decay_type (str): Schedule decay method ('linear' or 'exponential').
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_episodes = total_episodes
        self.decay_type = decay_type

    def get_beta(self, current_episode: int) -> float:
        """Computes current episode's entropy coefficient value.

        Args:
            current_episode (int): Current episode step index.

        Returns:
            float: Calculated beta coefficient.
        """
        if current_episode >= self.total_episodes:
            return self.beta_end
        
        progress = current_episode / self.total_episodes
        
        if self.decay_type == "linear":
            return self.beta_start + progress * (self.beta_end - self.beta_start)
        elif self.decay_type == "exponential":
            return self.beta_start * ((self.beta_end / self.beta_start) ** progress)
        return self.beta_start


def compute_returns(rewards: list, gamma: float) -> list:
    """Computes standard discounted returns-to-go G_t = r_t + gamma * G_{t+1}.

    Args:
        rewards (list): Sequence of scalar step rewards [r_0, r_1, ..., r_T].
        gamma (float): Reward discount factor in [0, 1].

    Returns:
        list: Sequence of computed discounted return values [G_0, G_1, ..., G_T].
    """
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns


def compute_trajectory_loss(
    traj: TrajectoryData,
    gamma: float,
    device: torch.device,
    beta_syl: float = 0.04,
    beta_word: float = 0.06,
    global_baseline_syl: float | None = None,
    global_baseline_word: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, UpdateResult]:
    """Computes GPOMDP policy loss with discounted returns-to-go and policy entropy regularization.

    Args:
        traj (TrajectoryData): Episode trajectory container.
        gamma (float): Reward discount factor.
        device (torch.device): PyTorch compute target device.
        beta_syl (float): Entropy coefficient weight for syllable policy.
        beta_word (float): Entropy coefficient weight for WordPiece policy.
        global_baseline_syl (float | None): Optional baseline value (unused in standard version).
        global_baseline_word (float | None): Optional baseline value (unused in standard version).

    Returns:
        tuple[torch.Tensor, torch.Tensor, UpdateResult]: Tuple containing:
            - Scalar loss tensor for syllable policy (with autograd graph).
            - Scalar loss tensor for WordPiece policy (with autograd graph).
            - UpdateResult container containing detached loss terms and step statistics.
    """
    result = UpdateResult()
    result.trajectory_len = traj.trajectory_len

    if traj.trajectory_len == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), result

    # 1. Compute discounted returns-to-go
    syl_returns = compute_returns(list(traj.syl_rewards), gamma)
    word_returns = compute_returns(list(traj.word_rewards), gamma)
    result.syl_returns = syl_returns
    result.word_returns = word_returns

    # Convert returns lists into 1D PyTorch tensors: Shape (T,)
    syl_returns_t = torch.tensor(syl_returns, dtype=torch.float32, device=device).view(-1)
    word_returns_t = torch.tensor(word_returns, dtype=torch.float32, device=device).view(-1)

    # Stack scalar log-prob tensors from trajectory into a 1D sequence tensor: Shape (T,)
    syl_log_probs_t = torch.stack(traj.syl_log_probs).view(-1)
    word_log_probs_t = torch.stack(traj.word_log_probs).view(-1)

    # Extract policy entropy values across trajectory steps into 1D tensors: Shape (T,)
    syl_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.syl_dist]).view(-1)
    word_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.word_dist]).view(-1)

    # Element-wise calculation: Loss_t = - log_prob_t * Return_t - beta * Entropy_t -> Shape (T,)
    loss_syl_terms = -syl_log_probs_t * syl_returns_t - (beta_syl * syl_entropy_t)
    loss_word_terms = -word_log_probs_t * word_returns_t - (beta_word * word_entropy_t)

    # Aggregate step loss terms across entire trajectory into scalar values
    total_loss_syl = loss_syl_terms.sum()
    total_loss_word = loss_word_terms.sum()

    result.syl_loss = total_loss_syl.item()
    result.word_loss = total_loss_word.item()
    result.syl_loss_terms = loss_syl_terms.detach()
    result.word_loss_terms = loss_word_terms.detach()

    return total_loss_syl, total_loss_word, result


def compute_trajectory_loss_REINFORCE(
    traj: TrajectoryData,
    gamma: float,
    device: torch.device,
    beta_syl: float = 0.04,
    beta_word: float = 0.06,
    global_baseline_syl: float | None = None,
    global_baseline_word: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, UpdateResult]:
    """Computes classical REINFORCE policy loss using total discounted episode trajectory reward R(tau).

    Args:
        traj (TrajectoryData): Episode trajectory container.
        gamma (float): Reward discount factor.
        device (torch.device): PyTorch target compute device.
        beta_syl (float): Syllable policy entropy weight.
        beta_word (float): WordPiece policy entropy weight.
        global_baseline_syl (float | None): Optional baseline parameter (unused).
        global_baseline_word (float | None): Optional baseline parameter (unused).

    Returns:
        tuple[torch.Tensor, torch.Tensor, UpdateResult]: Scalar syllable loss tensor,
            scalar WordPiece loss tensor, and detailed update result object.
    """
    result = UpdateResult()
    result.trajectory_len = traj.trajectory_len

    if traj.trajectory_len == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), result

    # Vector of powers of gamma [gamma^0, gamma^1, ..., gamma^(T-1)]: Shape (T,)
    gamma_powers = [gamma ** i for i in range(traj.trajectory_len)]
    gamma_tensor = torch.tensor(gamma_powers, dtype=torch.float32, device=device)

    # Calculate total discounted episode return R(tau) = sum_t (gamma^t * r_t) as scalar float
    R_tau_syl = float(torch.sum(torch.tensor(traj.syl_rewards, device=device) * gamma_tensor))
    R_tau_word = float(torch.sum(torch.tensor(traj.word_rewards, device=device) * gamma_tensor))

    result.syl_returns = [R_tau_syl] * traj.trajectory_len
    result.word_returns = [R_tau_word] * traj.trajectory_len

    # Scale total return by discount weights vector: Shape (T,)
    syl_returns_t = gamma_tensor * R_tau_syl
    word_returns_t = gamma_tensor * R_tau_word

    # Stack log probabilities and entropy across trajectory steps: Shape (T,)
    syl_log_probs_t = torch.stack(traj.syl_log_probs).view(-1)
    word_log_probs_t = torch.stack(traj.word_log_probs).view(-1)

    syl_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.syl_dist]).view(-1)
    word_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.word_dist]).view(-1)

    # Element-wise loss computation: Shape (T,)
    loss_syl_terms = -syl_log_probs_t * syl_returns_t - (beta_syl * syl_entropy_t)
    loss_word_terms = -word_log_probs_t * word_returns_t - (beta_word * word_entropy_t)

    total_loss_syl = loss_syl_terms.sum()
    total_loss_word = loss_word_terms.sum()

    result.syl_loss = total_loss_syl.item()
    result.word_loss = total_loss_word.item()
    result.syl_loss_terms = loss_syl_terms.detach()
    result.word_loss_terms = loss_word_terms.detach()

    return total_loss_syl, total_loss_word, result


def compute_trajectory_loss_REINFORCE_baseline(
    traj: TrajectoryData,
    gamma: float,
    device: torch.device,
    beta_syl: float = 0.04,
    beta_word: float = 0.06,
    global_baseline_syl: float | None = None,
    global_baseline_word: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, UpdateResult]:
    """Computes REINFORCE policy loss using total episode trajectory return minus global baseline.

    Args:
        traj (TrajectoryData): Trajectory containing actions, step rewards, and log probabilities.
        gamma (float): Discount factor.
        device (torch.device): Computation target device.
        beta_syl (float): Syllable entropy penalty factor.
        beta_word (float): WordPiece entropy penalty factor.
        global_baseline_syl (float | None): Baseline subtractor value for syllable return.
        global_baseline_word (float | None): Baseline subtractor value for WordPiece return.

    Returns:
        tuple[torch.Tensor, torch.Tensor, UpdateResult]: Syllable loss tensor, WordPiece loss tensor, and update summary.
    """
    result = UpdateResult()
    result.trajectory_len = traj.trajectory_len

    if traj.trajectory_len == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), result

    # Compute exponential discount weights tensor: Shape (T,)
    gamma_powers = [gamma ** i for i in range(traj.trajectory_len)]
    gamma_tensor = torch.tensor(gamma_powers, dtype=torch.float32, device=device)

    # Calculate total discounted episode return R(tau)
    R_tau_syl = float(torch.sum(torch.tensor(traj.syl_rewards, device=device) * gamma_tensor))
    R_tau_word = float(torch.sum(torch.tensor(traj.word_rewards, device=device) * gamma_tensor))

    # Subtract global baseline to compute policy advantage
    b_syl = global_baseline_syl if global_baseline_syl is not None else 0.0
    b_word = global_baseline_word if global_baseline_word is not None else 0.0

    advantage_syl = R_tau_syl - b_syl
    advantage_word = R_tau_word - b_word

    result.syl_returns = [advantage_syl] * traj.trajectory_len
    result.word_returns = [advantage_word] * traj.trajectory_len

    # Scale advantage by discount powers: Shape (T,)
    syl_returns_t = gamma_tensor * advantage_syl
    word_returns_t = gamma_tensor * advantage_word

    # Stack log probabilities and entropy tensors: Shape (T,)
    syl_log_probs_t = torch.stack(traj.syl_log_probs).view(-1)
    word_log_probs_t = torch.stack(traj.word_log_probs).view(-1)

    syl_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.syl_dist]).view(-1)
    word_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.word_dist]).view(-1)

    # Compute individual step loss terms: Shape (T,)
    loss_syl_terms = -syl_log_probs_t * syl_returns_t - (beta_syl * syl_entropy_t)
    loss_word_terms = -word_log_probs_t * word_returns_t - (beta_word * word_entropy_t)

    total_loss_syl = loss_syl_terms.sum()
    total_loss_word = loss_word_terms.sum()

    result.syl_loss = total_loss_syl.item()
    result.word_loss = total_loss_word.item()
    result.syl_loss_terms = loss_syl_terms.detach()
    result.word_loss_terms = loss_word_terms.detach()

    return total_loss_syl, total_loss_word, result


def compute_trajectory_loss_baseline(
    traj: TrajectoryData,
    gamma: float,
    device: torch.device,
    beta_syl: float = 0.01,
    beta_word: float = 0.04,
    global_baseline_syl: float | None = None,
    global_baseline_word: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, UpdateResult]:
    """Computes discounted GPOMDP loss with baseline subtraction for variance reduction.

    Args:
        traj (TrajectoryData): Trajectory container for a single episode rollout.
        gamma (float): Reward discount factor.
        device (torch.device): Compute target device.
        beta_syl (float): Syllable policy entropy multiplier.
        beta_word (float): WordPiece policy entropy multiplier.
        global_baseline_syl (float | None): Global baseline subtracted from syllable returns.
        global_baseline_word (float | None): Global baseline subtracted from WordPiece returns.

    Returns:
        tuple[torch.Tensor, torch.Tensor, UpdateResult]: Syllable loss tensor, WordPiece loss tensor, update container.
    """
    result = UpdateResult()
    result.trajectory_len = traj.trajectory_len

    if traj.trajectory_len == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), result

    # 1. Compute discounted returns-to-go G_t
    syl_returns = compute_returns(list(traj.syl_rewards), gamma)
    word_returns = compute_returns(list(traj.word_rewards), gamma)
    result.syl_returns = syl_returns
    result.word_returns = word_returns

    # Convert returns lists to 1D tensors: Shape (T,)
    syl_returns_t = torch.tensor(syl_returns, dtype=torch.float32, device=device).view(-1)
    word_returns_t = torch.tensor(word_returns, dtype=torch.float32, device=device).view(-1)

    # Stack step action log probabilities: Shape (T,)
    syl_log_probs_t = torch.stack(traj.syl_log_probs).view(-1)
    word_log_probs_t = torch.stack(traj.word_log_probs).view(-1)

    # Prepare baseline tensors
    baseline_syl = 0.0
    baseline_word = 0.0

    if global_baseline_syl is not None:
        baseline_syl = torch.tensor(global_baseline_syl, dtype=torch.float32, device=device)

    if global_baseline_word is not None:
        baseline_word = torch.tensor(global_baseline_word, dtype=torch.float32, device=device)

    # Calculate step advantage vectors A_t = G_t - b: Shape (T,)
    syl_advantages = syl_returns_t - baseline_syl
    word_advantages = word_returns_t - baseline_word

    # Stack distributions entropy: Shape (T,)
    syl_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.syl_dist]).view(-1)
    word_entropy_t = torch.stack([dist.entropy().squeeze() for dist in traj.word_dist]).view(-1)

    # Compute step loss using advantages: Shape (T,)
    loss_syl_terms = -syl_log_probs_t * syl_advantages - (beta_syl * syl_entropy_t)
    loss_word_terms = -word_log_probs_t * word_advantages - (beta_word * word_entropy_t)

    total_loss_syl = loss_syl_terms.sum()
    total_loss_word = loss_word_terms.sum()

    result.syl_loss = total_loss_syl.item()
    result.word_loss = total_loss_word.item()
    result.syl_loss_terms = loss_syl_terms.detach()
    result.word_loss_terms = loss_word_terms.detach()

    return total_loss_syl, total_loss_word, result


class RunningBaseline:
    """Tracks exponential moving average (EMA) of trajectory returns to maintain a dynamic baseline."""

    def __init__(self, decay: float = 0.99):
        """Initializes EMA baseline params.

        Args:
            decay (float): Exponential moving average momentum factor in (0, 1).
        """
        self.decay = decay
        self.syl_mean = 0.0
        self.word_mean = 0.0
        self.is_initialized = False

    def update(self, syl_returns: list[float], word_returns: list[float]) -> None:
        """Updates internal running EMA return estimates given a batch of trajectory returns.

        Args:
            syl_returns (list[float]): Return values from syllable trajectory.
            word_returns (list[float]): Return values from WordPiece trajectory.
        """
        if not syl_returns or not word_returns:
            return
            
        syl_batch_mean = sum(syl_returns) / len(syl_returns)
        word_batch_mean = sum(word_returns) / len(word_returns)

        if not self.is_initialized:
            self.syl_mean = syl_batch_mean
            self.word_mean = word_batch_mean
            self.is_initialized = True
        else:
            self.syl_mean = self.decay * self.syl_mean + (1 - self.decay) * syl_batch_mean
            self.word_mean = self.decay * self.word_mean + (1 - self.decay) * word_batch_mean


def apply_policy_optimization(
    total_loss_syl: torch.Tensor,
    total_loss_word: torch.Tensor,
    syl_optimizer: torch.optim.Optimizer,
    word_optimizer: torch.optim.Optimizer,
    retain_graph: bool = False,
    accumulate_steps: int = 1,
) -> None:
    """Scales losses by accumulation step factor, backpropagates gradients, and updates optimizers.

    Args:
        total_loss_syl (torch.Tensor): Calculated total scalar loss tensor for syllable model.
        total_loss_word (torch.Tensor): Calculated total scalar loss tensor for WordPiece model.
        syl_optimizer (torch.optim.Optimizer): PyTorch optimizer for syllable model parameters.
        word_optimizer (torch.optim.Optimizer): PyTorch optimizer for WordPiece model parameters.
        retain_graph (bool): Flag to keep Autograd computational graph after backward pass.
        accumulate_steps (int): Number of gradient accumulation steps.
    """
    # Scale loss tensors by accumulation steps to keep gradient magnitude consistent
    scaled_loss_syl = total_loss_syl / accumulate_steps
    scaled_loss_word = total_loss_word / accumulate_steps

    # Optimize Syllable network parameters
    syl_optimizer.zero_grad()
    scaled_loss_syl.backward(retain_graph=retain_graph)
    syl_optimizer.step()

    # Optimize WordPiece network parameters
    word_optimizer.zero_grad()
    scaled_loss_word.backward(retain_graph=retain_graph)
    word_optimizer.step()


def build_episode_log(
    traj: TrajectoryData,
    update: UpdateResult,
    epoch: int,
    episode_idx: int,
) -> dict:
    """Constructs a structured logging dictionary combining step metrics, trajectory rewards, and losses.

    Args:
        traj (TrajectoryData): Trajectory container containing episode step data.
        update (UpdateResult): Update summary containing loss terms and return lists.
        epoch (int): Current epoch index.
        episode_idx (int): Current episode index.

    Returns:
        dict: Aggregated episode log dictionary.
    """
    steps_log = []
    for step_idx, step_info in enumerate(traj.steps_data):
        entry = dict(step_info)
        entry["loss_syl"] = update.syl_loss_terms[step_idx].item()
        entry["loss_word"] = update.word_loss_terms[step_idx].item()
        entry["return_syl"] = update.syl_returns[step_idx]
        entry["return_word"] = update.word_returns[step_idx]
        steps_log.append(entry)

    last = steps_log[-1] if steps_log else {}
    n = max(traj.trajectory_len, 1)

    return {
        "epoch": epoch,
        "episode": episode_idx,
        "trajectory_length": traj.trajectory_len,
        "avg_syl_reward": sum(traj.syl_rewards) / n,
        "avg_word_reward": sum(traj.word_rewards) / n,
        "total_loss_syl": update.syl_loss,
        "total_loss_word": update.word_loss,
        "final_score": last.get("avg_classifier_score"),
        "final_syl_classifier_score": last.get("syl_classifier_score"),
        "final_word_classifier_score": last.get("word_classifier_score"),
        "word_win_count": traj.word_wins,
        "syl_win_count": traj.syl_wins,
        "total_steps": traj.trajectory_len,
        "terminated_before": traj.trajectory_len < traj.max_steps_per_episode,
        "steps": steps_log,
    }