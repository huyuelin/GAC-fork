"""Reference training loop entry point.

This script sketches how :class:`~gac.controller.AdaptiveMuController` is
integrated with a supervised loss and an on-policy RL loss inside a single
optimiser step. It is intentionally framework-agnostic: the actual RL
gradient (advantages, importance ratios, clipping) is left to the caller's
policy-optimisation routine.

Usage
-----
This is a template, not a runnable training entry. To reproduce the paper's
results, integrate the ``compute_hybrid_loss`` routine below with your
GRPO/PPO trainer of choice (we used a fork of Trinity-RFT for the
experiments in the paper).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from gac.controller import AdaptiveMuController, mu_schedule
from gac.sft_loss import SFTPhiLossFn


def compute_hybrid_loss(
    *,
    logprob: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    expert_mask: torch.Tensor,
    rl_loss_fn: Callable[..., Tuple[torch.Tensor, Dict[str, float]]],
    sft_loss_fn: Callable[..., Tuple[torch.Tensor, Dict[str, float]]],
    controller: AdaptiveMuController,
    global_step: int,
    schedule_prior: Optional[float] = None,
    alpha_ctrl: Optional[float] = None,
    blend_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the GAC-mixed hybrid loss for a single optimiser step.

    Parameters
    ----------
    logprob, action_mask, advantages, expert_mask
        See :meth:`AdaptiveMuController.update_stats`.
    rl_loss_fn, sft_loss_fn
        Callables returning ``(loss, log_dict)`` for the on-policy RL and
        supervised branches respectively. The RL callable should implement
        the user's preferred surrogate (e.g. PPO or GRPO with clipping).
    controller
        The :class:`AdaptiveMuController` instance whose state persists
        across calls.
    global_step
        Current optimiser step, forwarded to both the controller (for
        update throttling) and the schedule prior.
    schedule_prior
        Value of the prior schedule at this step. If ``None``, no prior is
        blended in and :math:`\\mu` follows the pure adaptive estimate.
    alpha_ctrl
        Target ratio :math:`\\alpha_\\mathrm{ctrl}` from the outer KL
        controller. If ``None``, the controller uses its ``alpha_static``
        default.
    blend_weight
        Weight :math:`\\lambda` on the adaptive estimate in the prior blend
        (see :meth:`AdaptiveMuController.compute_mu`).

    Returns
    -------
    loss : torch.Tensor
        Scalar hybrid loss :math:`L = (1 - \\mu) L_\\mathrm{RL} + \\mu L_\\mathrm{SFT}`.
    log_dict : dict
        Merged diagnostic scalars from both branches plus the controller.
    """
    controller.update_stats(
        logprob=logprob,
        action_mask=action_mask,
        advantages=advantages,
        expert_mask=expert_mask,
        current_step=global_step,
    )
    mu, mu_stats = controller.compute_mu(
        alpha=alpha_ctrl,
        mu_prior=schedule_prior,
        blend_weight=blend_weight,
    )

    rl_sel = ~expert_mask
    sft_sel = expert_mask

    log_dict: Dict[str, float] = dict(mu_stats)

    if rl_sel.any():
        rl_loss, rl_log = rl_loss_fn(
            logprob=logprob[rl_sel],
            action_mask=action_mask[rl_sel],
            advantages=advantages[rl_sel],
        )
        log_dict.update({f"rl/{k}": v for k, v in rl_log.items()})
    else:
        rl_loss = torch.zeros((), device=logprob.device, dtype=logprob.dtype)

    if sft_sel.any():
        sft_loss, sft_log = sft_loss_fn(
            logprob=logprob[sft_sel],
            action_mask=action_mask[sft_sel],
        )
        log_dict.update({f"sft/{k}": v for k, v in sft_log.items()})
    else:
        sft_loss = torch.zeros((), device=logprob.device, dtype=logprob.dtype)

    loss = (1.0 - mu) * rl_loss + mu * sft_loss
    log_dict["hybrid/loss"] = float(loss.detach().item())
    log_dict["hybrid/mu"] = float(mu)
    return loss, log_dict


__all__ = ["compute_hybrid_loss", "AdaptiveMuController", "mu_schedule", "SFTPhiLossFn"]
