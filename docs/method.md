# GAC: Method Notes

This document expands on the mathematical derivation and proxy semantics of
the Guided Adaptive Controller (GAC). It is intended as a companion to
Sec. 3 of the paper and to the docstrings in
[`gac/controller.py`](../gac/controller.py).

## 1. Setting

Hybrid post-training combines two update signals:

- an **SFT branch** that minimises a negative log-likelihood loss
  `L_SFT(θ)` on expert demonstrations, and
- an **RL branch** that maximises a scalar reward via a policy-gradient
  surrogate such as PPO or GRPO, denoted `L_RL(θ)`.

The mixed loss is `L(θ; µ) = (1 − µ) · L_RL(θ) + µ · L_SFT(θ)` with
mixing weight `µ ∈ [0, 1]`. Choosing `µ` well is important because SFT
provides expert anchoring but no reward-driven adaptation, while RL
provides reward maximisation but suffers from high-variance advantage
estimates that can destabilise policy behaviour. Fixed schedules cannot
respond when the noise budgets of the two branches shift during training.

## 2. Closed form

Model the two branch gradients as `ĝ_s = g_s* + ε_s` and `ĝ_r = g_r* + ε_r`
with independent zero-mean noise of variances `σ_s²` and `σ_r²`. Given a
local target `g* = α · g_s* + (1 − α) · g_r*`, the mixed gradient
`ĝ(µ) = µ · ĝ_s + (1 − µ) · ĝ_r` has expected squared error

```
E(µ) = (µ − α)² · Δg²  +  µ² · σ_s²  +  (1 − µ)² · σ_r²
```

where `Δg² = ‖g_s* − g_r*‖²`. Minimising `E(µ)` yields the unique
optimum

```
µ* = (α · Δg² + σ_r²) / (Δg² + σ_s² + σ_r²).
```

This is the core estimator implemented by
`AdaptiveMuController.compute_mu`.

### Boundary behaviour

- **Δg² → 0** — the branches agree on the ideal update; `µ*` reduces to
  inverse-variance weighting, `σ_r² / (σ_s² + σ_r²)`.
- **Δg² → ∞** — the branches disagree strongly; `µ*` defaults to the
  user-specified target `α`, recovering the operator's high-level
  preference.

### Regime of applicability

The closed form assumes unbiased estimators and variance-dominated noise.
When estimator bias becomes dominant (e.g., aggressive PPO clipping or a
learned reward model with systematic drift), the formula loses its global
optimality interpretation and should be read as a **variance-aware
stabiliser** rather than an exact optimum. Sec. 3 of the paper carries the
derivation through a bias-augmented MSE upper bound (`Eq. 6` in the paper),
which reduces to the equation above when both biases vanish. The
implementation supports this interpretation transparently: the same code
path is used in both regimes.

## 3. Proxies (coefficient space, not gradient space)

Gradient-level quantities are prohibitively expensive to compute every
step. The controller instead consumes three online proxies derived from
tensors that any PPO/GRPO trainer already materialises.

### σ_r² — post-normalisation advantage dispersion

`AdaptiveMuController.update_stats` computes `σ_r²` as the batch variance
of *sequence-level* GRPO-normalised advantages,

```
σ_r² = Var_i [ mean_t A_{i,t} ]
```

evaluated over on-policy rollouts. Because GRPO normalises within each
group before the loss step, this is a dispersion of *coefficients* rather
than of raw rewards. Length invariance is automatic: each rollout
contributes one scalar regardless of its token count.

### σ_s² — length-normalised NLL variance

For SFT sequences, `σ_s²` is the two-sided trimmed variance of
length-normalised per-sequence NLL:

```
nll_i    = mean_t [ −log π_θ(y_{i,t} | y_{i,<t}) ]
σ_s²     = TrimmedVar_i [ nll_i ]  (10% tail trim by default).
```

Trimming robustifies against occasional degenerate expert samples without
discarding informative mass in the body of the distribution.

### Δg̃² — coefficient-space disagreement

The disagreement proxy compares SFT and RL gradient *coefficients* on
shared response tokens. Since both objectives share a policy-gradient
structure

```
∇_θ L = E [ Σ_t c_t · ∇_θ log π_θ(a_t | s_t) ],
```

the SFT coefficient is `c_t^s = −1` (or `−φ(p_t)` when the CHORD
reweighting is enabled), and the RL coefficient is `c_t^r = −A_t`. After
within-batch z-normalisation, we compute

```
Δg̃² = mean_i [ mean_t (c̃_{s,t} − c̃_{r,t})² ]
```

on tokens selected by the action mask. This is a coefficient-space
mismatch, not a parameter-space gradient inner product, and it does not
require an additional backward pass. Falsification-style perturbations
(shuffling advantages or `φ`-weights, replacing coefficients with
constants) reduce the proxy's correlation with the true `Δg²` from
`r ≈ 0.84` to near zero, indicating that the signal comes from genuine
token-level structure rather than from marginal-statistic capture (see
paper Sec. 4).

## 4. Guardrails

The raw closed-form estimate `µ_raw` is passed through three standard
control mechanisms before it is applied:

1. **EMA smoothing** — `µ_ada = β · µ_prev + (1 − β) · µ_raw` with
   `β = 0.99`.
2. **Cosine-prior blending** — `µ_blend = (1 − λ) · µ_prior(t) + λ · µ_ada`,
   where `µ_prior(t)` is a warmup + cosine-decay schedule and
   `λ = 0.5` by default.
3. **Per-step change cap** — `µ_t = clip(µ_t−1 + clip(µ_blend − µ_t−1,
   ±c̄), [µ_min, µ_max])` with `c̄ = 0.01`.

These are analogous to PPO surrogate clipping and Adam-style momentum:
they stabilise, but do not replace, the underlying estimator. Empirically,
the noise-aware estimator alone contributes +3.7 pp on AMC, whereas EMA
alone contributes +0.2 pp, so gains should be attributed to the full
stack with the estimator as the dominant component (see paper Table 5).

## 5. Complexity and integration

The controller adds:

- **three EMA scalars** (`σ_r²`, `σ_s²`, `Δg̃²`) with no additional
  forward or backward passes,
- **one all-reduce** of first and second moments for `σ_r²` in
  distributed mode,
- **one scalar division** per `µ` update.

Total wall-time overhead measured on 8 A800 GPUs is below 1% across
sequence lengths 2k–8k (see paper App. F). Peak GPU memory is
statistically indistinguishable from a Trinity-RFT baseline without
GAC.

## 6. Where GAC is validated (and where it is not)

*In scope for empirical claims.* GAC is designed for the interface at
which an SFT loss and an on-policy RL loss are co-optimised within the
same optimiser step. All results in the paper use `L_SFT-φ` + on-policy
GRPO with verifiable, rule-based rewards.

*Not validated in this paper.* Multi-stage pipelines that chain
preference-based and on-policy phases (e.g., DPO → GRPO), non-co-mixing
pipelines (staged SFT → RL without shared updates), and open-ended
alignment with learned reward models. GAC is naturally applied only
within the stage where two update sources are actively mixed; earlier
stages affect initialisation but do not change the controller's role.
Transferring to those settings is an interesting direction for future
work but is not part of the current claims.
