"""Unit tests for :class:`gac.AdaptiveMuController`.

These tests verify the analytical properties of the closed-form estimator
and the guardrail pipeline. They do not exercise the distributed
:func:`_global_variance_1d` path, which requires a live process group.
"""

from __future__ import annotations

import math

import torch

from gac.controller import AdaptiveMuController, mu_schedule


def _make_batch(
    batch_size: int = 8,
    seq_len: int = 32,
    rl_fraction: float = 0.5,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    logprob = -torch.rand(batch_size, seq_len, generator=g).abs()
    action_mask = torch.ones(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len, generator=g)
    expert_mask = torch.zeros(batch_size, dtype=torch.bool)
    n_expert = int(batch_size * (1.0 - rl_fraction))
    expert_mask[:n_expert] = True
    return dict(
        logprob=logprob,
        action_mask=action_mask,
        advantages=advantages,
        expert_mask=expert_mask,
    )


def test_closed_form_matches_manual_computation():
    """With injected proxies, `compute_mu` matches the analytical µ*."""
    ctrl = AdaptiveMuController(ema_beta=0.0, mu_change_cap=1.0, init_mu=0.5)
    ctrl.ema_sigma_s2 = 1.0
    ctrl.ema_sigma_r2 = 4.0
    ctrl.ema_delta_g2 = 0.0
    mu, stats = ctrl.compute_mu(alpha=0.5, mu_prior=None, blend_weight=1.0)
    expected = (0.5 * 0.0 + 4.0) / (0.0 + 1.0 + 4.0)
    assert math.isclose(mu, expected, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(stats["mu/raw"], expected, rel_tol=1e-6, abs_tol=1e-6)


def test_mu_clipped_to_bounds():
    """The final µ never leaves the configured [mu_min, mu_max] interval."""
    ctrl = AdaptiveMuController(
        ema_beta=0.0, mu_change_cap=1.0, mu_min=0.2, mu_max=0.8, init_mu=0.5
    )
    ctrl.ema_sigma_s2 = 0.0
    ctrl.ema_sigma_r2 = 1e6
    ctrl.ema_delta_g2 = 0.0
    mu, _ = ctrl.compute_mu(alpha=0.5, blend_weight=1.0)
    assert mu <= 0.8 + 1e-9


def test_change_cap_limits_step_size():
    """A single call cannot move µ by more than mu_change_cap."""
    ctrl = AdaptiveMuController(
        ema_beta=0.0, mu_change_cap=0.02, init_mu=0.5
    )
    ctrl.ema_sigma_s2 = 0.0
    ctrl.ema_sigma_r2 = 1e6
    ctrl.ema_delta_g2 = 0.0
    mu, _ = ctrl.compute_mu(alpha=0.5, blend_weight=1.0)
    assert abs(mu - 0.5) <= 0.02 + 1e-9


def test_update_stats_throttling():
    """update_stats is a no-op when called before mu_update_freq elapses."""
    ctrl = AdaptiveMuController(mu_update_freq=10)
    batch = _make_batch()
    ctrl.update_stats(current_step=0, **batch)
    snapshot = (ctrl.ema_sigma_r2, ctrl.ema_sigma_s2, ctrl.ema_delta_g2)
    # Second call at step 5 (< 10) should not change EMA state.
    batch2 = _make_batch(seed=1)
    ctrl.update_stats(current_step=5, **batch2)
    assert (ctrl.ema_sigma_r2, ctrl.ema_sigma_s2, ctrl.ema_delta_g2) == snapshot


def test_schedule_endpoints_and_monotone_decay():
    """`mu_schedule` respects its boundary conditions."""
    assert mu_schedule(0, warmup_steps=10, decay_steps=100, mu_peak=0.9, mu_valley=0.1) == 0.0
    peak = mu_schedule(10, warmup_steps=10, decay_steps=100, mu_peak=0.9, mu_valley=0.1)
    assert math.isclose(peak, 0.9, rel_tol=1e-6)
    end = mu_schedule(200, warmup_steps=10, decay_steps=100, mu_peak=0.9, mu_valley=0.1)
    assert math.isclose(end, 0.1, rel_tol=1e-6)
    mid = mu_schedule(60, warmup_steps=10, decay_steps=100, mu_peak=0.9, mu_valley=0.1)
    assert 0.1 < mid < 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("All tests passed.")
