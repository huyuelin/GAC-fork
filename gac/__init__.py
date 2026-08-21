"""GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training.

This package provides the reference implementation of the Guided Adaptive
Controller (GAC), a lightweight noise-aware controller for the global mixing
weight between supervised fine-tuning (SFT) and on-policy reinforcement
learning (RL) losses in hybrid post-training of large language models.

The public surface exposes:

* :class:`AdaptiveMuController` -- the closed-form estimator with EMA
  smoothing, prior blending, and per-step change capping described in
  Section 3 of the paper.
* :func:`masked_mean` -- the length-invariant averaging utility used by all
  coefficient-space proxies (borrowed, with minor typing changes, from the
  VERL codebase).
* :class:`SFTLossFn` -- the token-wise weighted SFT loss with
  :math:`\\varphi(p) = p(1-p)` weighting adopted from CHORD.
* :class:`ChordPolicyLossFn` -- the full hybrid loss that couples RL and SFT
  branches through the GAC-controlled mixing weight :math:`\\mu`.

Refer to :mod:`gac.controller` for the mathematical details and to the paper
Algorithm 1 for the deployed procedure.
"""

from gac.controller import AdaptiveMuController
from gac.utils import masked_mean, masked_var

__all__ = [
    "AdaptiveMuController",
    "masked_mean",
    "masked_var",
]

__version__ = "0.1.0"
