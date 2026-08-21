"""Masked reduction utilities used by the GAC controller and proxies.

These helpers compute length-invariant statistics on padded token tensors and
are adapted from the VERL codebase
(https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py).
They are intentionally self-contained so that the controller does not depend
on the parent training framework.
"""

from __future__ import annotations

from typing import Optional

import torch


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: Optional[int] = None,
) -> torch.Tensor:
    """Sum ``values`` over positions where ``mask`` is non-zero.

    The mask is cast to the dtype of ``values`` to prevent unintended
    upcasting (e.g., a ``float32`` mask combined with ``bfloat16`` activations
    would otherwise promote the result to ``float32``).
    """
    if mask.dtype != values.dtype:
        mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(axis=axis)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Length-invariant mean of ``values`` over positions where ``mask`` is 1.

    A small ``eps`` is added to the denominator to guard against fully masked
    sequences that may arise when a batch contains only expert or only
    on-policy samples.
    """
    if mask.dtype != values.dtype:
        mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + eps)


def masked_var(
    values: torch.Tensor,
    mask: torch.Tensor,
    unbiased: bool = True,
) -> torch.Tensor:
    """Masked variance with an optional Bessel correction.

    Raises
    ------
    ValueError
        If the mask has zero or one active elements when ``unbiased=True``,
        which would cause a division-by-zero after the Bessel correction.
    """
    mean = masked_mean(values, mask)
    centered = values - mean
    variance = masked_mean(centered ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("Empty mask: at least one active token is required.")
        if mask_sum == 1:
            raise ValueError(
                "Mask contains a single active token; unbiased variance is undefined. "
                "Increase batch size or set unbiased=False."
            )
        variance = variance * (mask_sum / (mask_sum - 1))
    return variance
