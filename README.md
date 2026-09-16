# MaskingPoets

### Enhancing Poetic Creativity through Alternative Tokenizations

---

## 1. Introduction

Poetry is a highly structured linguistic artefact: it obeys constraints of rhyme, rhythm, rhetorical figuration, and prosody that distinguish it sharply from ordinary prose. Yet the question of whether a machine can *poetify* arbitrary text — iteratively transforming it into something that satisfies poetic constraints — remains largely open.

**MaskingPoets** addresses this problem directly. Given an input text, the system iteratively identifies the segments that are least "poetic" and replaces them with more suitable alternatives, until a desired poetic threshold is reached.

Formally, we frame *poetization* as a **sequential decision process** over token replacements, guided by two complementary linguistic signals and optimised through Reinforcement Learning.

The entire project is grounded in the *Biblioteca Italiana* corpus — a collection of over 25,000 Italian poems — which provides the empirical basis for both representation learning and reward modelling.

---

## 2. Architecture Overview

Poeticity is not a monolithic property. It manifests along **two orthogonal linguistic dimensions**, each governed by a distinct set of regularities:

| Dimension     | Governs                                                         | Signal               |
| ------------- | --------------------------------------------------------------- | -------------------- |
| **Syntactic** | Grammatical structure, lexical compatibility, phrasal coherence | WordPiece embeddings |
| **Syllabic**  | Rhythm, metre, rhyme, stress patterns                           | Syllable embeddings  |

The system is therefore **dual-branched**: two encoders, two scorers, and two policies operate in parallel, each specialising on one dimension. Their outputs are combined through a harmonic aggregation that prevents either dimension from dominating the other.

```text
                        ┌──────────────────────┐
                        │   Input Text (X₀)     │
                        └──────────┬───────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
        ┌────────────────┐                  ┌────────────────┐
        │  WordPiece     │                  │   Syllabic     │
        │  Encoder       │                  │   Encoder      │
        │  (6L · 8H · 512)│                 │  (6L · 8H · 512)│
        └────────┬───────┘                  └────────┬───────┘
                 │                                   │
                 ▼                                   ▼
        ┌────────────────┐                  ┌────────────────┐
        │  Syntactic     │                  │   Metric       │
        │  Scorer  S_W   │                  │   Scorer  S_S  │
        └────────┬───────┘                  └────────┬───────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Harmonic Score  H   │
                        │  H = 2·S_S·S_W/(S_S+S_W) │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Policy → Mask → MLM │
                        │  (iterative loop)    │
                        └──────────────────────┘
```

---

## 3. Encoders

The first computational task is to learn the statistical structure underlying each dimension. To this end, two **bidirectional attention-based encoders** are trained independently via Masked Language Modelling (MLM), one per dimension. Both adopt a BERT-like architecture:

- **6 attention layers**
- **8 attention heads**
- **512 hidden dimensions**

Their design departs from standard BERT in two deliberate ways:

1. **Poetry-specific control tokens** — `[STANZA]` and `[VERSE]` — allow the model to encode structural boundaries inherent to poetic form.
2. **Task-specific tokenizations** — the syntactic encoder uses a WordPiece vocabulary trained on the Italian poetry corpus; the syllabic encoder uses a syllabic tokenizer (via *Pyphen*) trained on the same corpus.

These choices yield representations that are **maximally informative for poetic structure**, rather than for general-purpose language modelling.

```text
  Raw verse:  "Nel mezzo del cammin di nostra vita"

  WordPiece:  [VERSE] Nel  ##mezzo  del  cam  ##min  di  nostra  vita
  Syllabic:   [VERSE] Nel  mez-zo  del  cam-min  di  no-stra  vi-ta
```

---

## 4. The Poetic Scorer (Reward Model)

Given the encoders, the next task is to assign a scalar **poeticity score** to any text. This is not a simple classification problem: empirical analysis revealed that the information relevant to poeticity is **distributed across all attention layers**, not concentrated in any single one.

A naïve approach — using only the final layer, or averaging layers uniformly — discards this distributed structure. Instead, the scorer must **learn to aggregate** token embeddings across layers in a principled, data-driven way.

### 4.1 Attention-Based Layer-Token Aggregation

We adopt a **Bahdanau-style attention** mechanism operating jointly over the *layer* and *token* axes:

1. A shared 3-layer MLP **scorer** maps each embedding vector `v ∈ ℝ^d` — at any `(layer, token)` position — to a scalar logit:
   
   ```text
   score(v) = MLP(v)   ∈ ℝ
   ```
   
   Because the same MLP is applied position-wise across all `(layer, token)` pairs, the operation is formally equivalent to a **2D 1×1 convolution** over the layer–token grid.

2. The resulting logit matrix of shape `(n_layers × seq_len)` is **linearised** and normalised via a single joint softmax:
   
   ```text
   α_{ℓ,t} = exp(score(v_{ℓ,t})) / Σ_{ℓ',t'} exp(score(v_{ℓ',t'}))
   ```
   
   This yields a **global attention distribution** over all `(layer, token)` positions.

3. The final aggregated representation is the weighted sum:
   
   ```text
   h = Σ_{ℓ,t} α_{ℓ,t} · v_{ℓ,t}
   ```

```text
        Layer × Token grid                    Attention heatmap
        ┌─────────────────┐                   ┌─────────────────┐
        │ v₀₀ v₀₁ v₀₂ ... │                   │ α₀₀ α₀₁ α₀₂ ... │
        │ v₁₀ v₁₁ v₁₂ ... │  ── scorer ──▶    │ α₁₀ α₁₁ α₁₂ ... │
        │ v₂₀ v₂₁ v₂₂ ... │  ── softmax ──▶   │ α₂₀ α₂₁ α₂₂ ... │
        │ ...             │                   │ ...             │
        └─────────────────┘                   └─────────────────┘
                                                       │
                                                       ▼
                                            h = Σ α · v   ∈ ℝ^d
```

### 4.2 Classification Head

The aggregated vector `h` is passed to a 3-layer MLP terminating in a logistic activation, producing the final poetic score in `[0, 1]`.

### 4.3 Training Objective

The scorer is trained as a **regression model** on a dataset constructed as follows:

1. Starting from *Biblioteca Italiana*, a corpus of **paraphrases** of Italian poems is generated via an LLM.

2. **Corruption methods** at varying rates are applied to both poems and paraphrases.

3. The expected score is defined as a min–max normalisation of:
   
   $$
   score = e^{−ρ · rate}
   $$

4. The dataset is constructed so that the scorer learns the ordering:
   
   ```text
   corrupted paraphrase  <  paraphrase  <  corrupted poem  <  poem
   ```

Two independent scorers are trained — one per encoder — yielding a **syntactic scorer** `S_W` and a **metric scorer** `S_S`.

---

## 5. The Deep Policy

The bidirectional encoders, thanks to MLM pre-training, already provide a strong prior over the most likely token for any masked position. The remaining problem is **which** token to mask, so that a small number of replacements yields the largest gain in poeticity.

Two policies are trained in parallel, mirroring the dual-branch architecture:

- **Syntactic policy** — targets grammatical and lexical disruptions.
- **Syllabic policy** — targets rhythmic and metrical disruptions.

Each policy is a 3-layer MLP operating on the **same attention-based aggregation** described in §4.1. Its output is a **probability distribution over the context length**, indicating which token positions are the most poetically disruptive.

```text
   Aggregated h  ──▶  MLP (3 layers)  ──▶  π(t) ∈ Δ^(L−1)
                                            │
                                            ▼
                                  argmax / sample position
                                            │
                                            ▼
                                       mask & MLM
```

---

## 6. Execution Pipeline

The full inference loop proceeds as follows:

```text
   X₀ ──▶ Encoder ──▶ Scorer ──▶ H(X₀)
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │  Phase 1 — Syntactic Restoration    │
              │  π_W proposes a mask position       │
              │  MLM fills it                       │
              │  Repeat for 6 steps                 │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │  Phase 2 — Metric Refinement        │
              │  π_S proposes a mask position       │
              │  MLM fills it                       │
              │  Repeat for 6 steps                 │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                              X_T  (poetified)
```

### 6.1 Harmonic Reward

At each iteration, the two scores are combined via the **harmonic mean**:

$$

             
H(X_t)  = \frac{2 · S_S(X_t) · S_W(X_t)}{S_S(X_t) + S_W(X_t)}
             

$$

The reward is the **harmonic difference**:

$$
\Delta H = H(X_t) − H(X_{t−1})
$$


Where $S_s$ is the syllable scorer while $S_W$ is the syntactic scorer.

The harmonic mean has a desirable property for this setting: it is **strictly dominated by the weaker of the two scores**. A text cannot achieve a high harmonic score by excelling in only one dimension; both must improve together. This discourages degenerate solutions in which, for instance, syntactic correctness is sacrificed for rhythmic gain.

### 6.2 Two-Phase Decomposition

Multi-Agent RL in this stochastic environment is notoriously difficult to stabilise. To mitigate this, each episode is decomposed into **two sequential phases**:

1. **Syntactic phase** — the syntactic policy restores grammatical correctness.
2. **Metric phase** — the syllabic policy refines rhythm and rhyme.

Although the two policies act separately, the harmonic reward couples them: each phase is constrained to avoid catastrophic degradation of the other dimension.

---

## 7. Summary of Components

| Component              | Role                      | Architecture                             |
| ---------------------- | ------------------------- | ---------------------------------------- |
| WordPiece Encoder      | Syntactic representations | 6L · 8H · 512d, BERT-like                |
| Syllabic Encoder       | Metric representations    | 6L · 8H · 512d, BERT-like                |
| Attention Aggregation  | Layer–token pooling       | 2D 1×1 Conv (shared MLP) + joint softmax |
| Syntactic Scorer `S_W` | Poeticity (syntax)        | MLP + sigmoid, regression                |
| Metric Scorer `S_S`    | Poeticity (metre)         | MLP + sigmoid, regression                |
| Syntactic Policy `π_W` | Mask position (syntax)    | MLP over aggregated `h`                  |
| Syllabic Policy `π_S`  | Mask position (metre)     | MLP over aggregated `h`                  |
| Reward                 | Combined signal           | Harmonic mean of `S_S`, `S_W`            |

---

## 8. Reinforcement Learning Training

The models explained above are used to train the policies using actor-only gradient-based algorithm: REINFORCE and GPOMDP. The training process is based on 12 steps long trajectories and a total number of episodes equal to 5000, sufficient enough to show clear differences between the two algorithms.

Experiments results reveal that, due to the high variance, REINFORCE (even with the baseline technique) is unable to properly learn. In GPOMDP results there are some improving trends in average rewards per episode (even though they are so little). 

# 
