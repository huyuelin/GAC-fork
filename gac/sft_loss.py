"""Token-wise weighted SFT loss with the CHORD :math:`\\varphi(p) = p(1-p)` reweighting.

This module provides two SFT loss variants used by the hybrid loss in
:mod:`gac.hybrid_loss`:

* :class:`SFTLossFn` -- vanilla token-level or sequence-level negative
  log-likelihood.
* :class:`SFTPhiLossFn` -- the token-wise reweighted variant adopted from
  CHORD, in which the per-token contribution is scaled by
  :math:`\\varphi(p_t) = p_t(1 - p_t)`. This weighting concentrates the SFT
  signal on uncertain tokens and is orthogonal to the global mixing weight
  produced by :class:`~gac.controller.AdaptiveMuController`.

Both classes are simple ``nn``-free callables and can be plugged into any
training loop that supplies per-token log-probabilities together with an
action mask.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from gac.utils import masked_mean


class SFTLossFn:
    """Standard supervised fine-tuning (NLL) loss.

    Parameters
    ----------
    token_level : bool, default True
        If ``True``, the loss is a token-level mean of NLL over all response
        tokens in the batch. If ``False``, per-sequence NLLs are computed
        first and then averaged, which corresponds to a length-normalised
        objective.
    """

    def __init__(self, token_level: bool = True) -> None:
        self.token_level = token_level

    def __call__(
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.token_level:
            loss = masked_mean(-logprob, action_mask)
        else:
            loss = masked_mean(-logprob, action_mask, axis=1).mean()
        return loss, {"sft_loss": float(loss.detach().item())}


class SFTPhiLossFn:
    """SFT loss reweighted by :math:`\\varphi(p_t) = p_t(1 - p_t)`.

    The reweighting is adopted from CHORD (Zhang et al., 2025) and provides
    a bounded, uncertainty-aware token weight that vanishes at both extremes
    of the softmax probability. Empirically, adding
    :class:`SFTPhiLossFn` on top of the GAC controller contributes a further
    +0.4--1.4 pp on the AMC benchmark (Sec. 4 of the paper), independent of
    the closed-form estimator.

    Parameters
    ----------
    token_level : bool, default True
        Same semantics as :class:`SFTLossFn`.
    detach_weights : bool, default True
        If ``True``, the :math:`\\varphi(p_t)` weights are detached from the
        computation graph so that gradients do not flow through the
        reweighting; this matches the paper's implementation.
    """

    def __init__(self, token_level: bool = True, detach_weights: bool = True) -> None:
        self.token_level = token_level
        self.detach_weights = detach_weights

    def __call__(
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p = torch.exp(logprob)
        phi = p * (1.0 - p)
        if self.detach_weights:
            phi = phi.detach()
        weighted_nll = -phi * logprob
        if self.token_level:
            loss = masked_mean(weighted_nll, action_mask)
        else:
            loss = masked_mean(weighted_nll, action_mask, axis=1).mean()
        return loss, {
            "sft_phi_loss": float(loss.detach().item()),
            "sft_phi_mean_weight": float(masked_mean(phi, action_mask).detach().item()),
        }
