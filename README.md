# GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blueviolet.svg)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-forthcoming-b31b1b.svg)]()

Reference implementation of the **Guided Adaptive Controller (GAC)**, a
noise-aware controller for the global mixing weight between supervised
fine-tuning (SFT) and on-policy reinforcement learning (RL) in hybrid
post-training of large language models.

> **GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training**
> Yuelin Hu, Zhenbo Yu, Wei Liu, Zhengxue Cheng, Li Song
> *Proceedings of EMNLP 2026 (Main Conference), Budapest.*

## Overview

Hybrid post-training that combines SFT and RL is the standard paradigm for
aligning LLMs, but a fixed schedule for the mixing weight µ cannot adapt when
the relative *noise* of the two signals evolves during training. GAC casts
SFT–RL mixing as a **minimum-mean-squared-error gradient estimation problem**
and derives a closed-form optimal mixing weight

<p align="center">
  <em>&mu;<sup>*</sup> = ( &alpha;<sub>tgt</sub> &middot; &Delta;g<sup>2</sup> + &sigma;<sub>r</sub><sup>2</sup> )
   &nbsp;/&nbsp; ( &Delta;g<sup>2</sup> + &sigma;<sub>s</sub><sup>2</sup> + &sigma;<sub>r</sub><sup>2</sup> )</em>
</p>

that balances SFT noise, RL noise, and SFT–RL disagreement. Because
gradient-level quantities are prohibitively expensive at every step, the
deployed controller consumes three **coefficient-space proxies** estimated
online from tensors that any GRPO/PPO trainer already computes, and wraps
them in EMA smoothing, a cosine-schedule prior, and per-step change capping.

The noise-aware controller alone outperforms the best rule-based baseline by
**+3.0 pp on AMC**; the full system reaches **+3.8 pp over HPT** across math,
code, science, and logic benchmarks, while reducing KL-drift area by 28%
and large |Δµ| events by more than 70%, at **less than 1% wall-time
overhead**. Gains grow with model size from 1.5B to 14B.

## Repository layout

```
gac/
├── gac/                     # Reference implementation
│   ├── controller.py        # AdaptiveMuController (Sec. 3 of the paper)
│   ├── sft_loss.py          # Standard SFT loss and the CHORD φ(p)=p(1-p) variant
│   ├── hybrid_loss.py       # Reference hybrid loss (µ · L_SFT + (1-µ) · L_RL)
│   ├── utils.py             # Masked reductions (length-invariant statistics)
│   └── __init__.py
├── configs/
│   └── gac_default.yaml     # Default hyperparameters (matches paper Sec. 4.1)
├── docs/
│   └── method.md            # Long-form derivation and proxy semantics
├── tests/
│   └── test_controller.py   # Unit tests for the closed-form estimator
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/huyuelin/GAC.git
cd GAC
pip install -r requirements.txt
pip install -e .        # optional editable install
```

The controller has no framework-specific dependencies beyond PyTorch and
optionally `torch.distributed` for multi-GPU training.

## Quick start

The controller is a small, stateful object that you call inside your
existing training step. A minimal integration reads:

```python
import torch
from gac import AdaptiveMuController
from gac.sft_loss import SFTPhiLossFn
from gac.hybrid_loss import compute_hybrid_loss
from gac.controller import mu_schedule

controller = AdaptiveMuController(
    ema_beta=0.99,
    mu_change_cap=0.01,
    trim_ratio_sft=0.1,
    mu_update_freq=10,
    alpha_static=0.5,
)
sft_loss_fn = SFTPhiLossFn(token_level=True)

def training_step(batch, global_step, rl_loss_fn):
    prior = mu_schedule(
        global_step,
        warmup_steps=200, decay_steps=800,
        mu_peak=0.85, mu_valley=0.15,
    )
    loss, logs = compute_hybrid_loss(
        logprob=batch["logprob"],
        action_mask=batch["action_mask"],
        advantages=batch["advantages"],
        expert_mask=batch["expert_mask"],
        rl_loss_fn=rl_loss_fn,
        sft_loss_fn=sft_loss_fn,
        controller=controller,
        global_step=global_step,
        schedule_prior=prior,
        blend_weight=0.5,
    )
    return loss, logs
```

The `rl_loss_fn` argument is the caller's preferred policy-optimisation
surrogate (PPO, GRPO, or any compatible variant); GAC does not modify the
RL loss itself, only its weighting relative to the SFT branch.

## Method summary

The controller performs three operations per update:

1. **Proxy estimation** (Sec. 3.2 of the paper).
   * `σ_r²` = variance of *sequence-level* GRPO-normalised advantages
     across the batch. Post-normalisation dispersion, **not** raw reward
     variance.
   * `σ_s²` = tail-trimmed variance of length-normalised per-sequence NLL
     on expert samples.
   * `Δg̃²` = mean squared coefficient mismatch between the SFT and RL
     token-level gradient coefficients on shared response tokens (after
     within-batch z-normalisation).
2. **Closed-form aggregation** into µ* using the formula above.
3. **Guarded update**: EMA smoothing → cosine-prior blend → per-step
   change cap → hard clipping to `[µ_min, µ_max]`.

By construction each proxy is length-invariant: every sequence contributes
exactly one scalar, so longer rollouts do not receive extra weight.

## Reproducibility

The paper's experiments were run on `Qwen2.5-{1.5B, 7B, 14B}-Instruct` with a
Trinity-RFT-based hybrid trainer. Base config matches
`configs/gac_default.yaml`:

| Hyperparameter | Value | Paper Sec. |
|---|---|---|
| `ema_beta` (β) | 0.99 | 3.4 |
| `mu_change_cap` (c̄) | 0.01 | 3.4 |
| `blend_weight` (λ) | 0.5 | 3.4 |
| `mu_update_freq` (f_µ) | 10 | 3.3 |
| `trim_ratio_sft` | 0.10 | 3.3 |
| KL target (KL_tgt) | 0.02 | 3.3 |
| α range | [0.1, 0.95] | 3.3 |

Data: OpenR1-Math-220k (5k SFT + 20k RL prompts), MBPP, HumanEval, GPQA,
SciBench, BBH logical subsets. All main results report mean ± std over three
random seeds with joint `p<0.05` and Cohen's `d>0.8` significance thresholds.

## Citation

If you use GAC in your research, please cite:

```bibtex
@inproceedings{hu2026gac,
  title     = {GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training},
  author    = {Hu, Yuelin and Yu, Zhenbo and Liu, Wei and Cheng, Zhengxue and Song, Li},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  address   = {Budapest, Hungary},
}
```

Machine-readable metadata is also provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

This work builds on the token-wise reweighting introduced by
[CHORD](https://arxiv.org/abs/2508.11408) and integrates with the
[Trinity-RFT](https://github.com/modelscope/Trinity-RFT) hybrid post-training
framework. We thank the anonymous ARR reviewers and the EMNLP 2026 Area
Chair for detailed feedback that shaped the camera-ready version of the
paper.

## License

Released under the [Apache License 2.0](LICENSE).
