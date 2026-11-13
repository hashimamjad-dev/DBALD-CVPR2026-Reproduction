"""Unit tests for the L_lane / L_env diffusion stealth losses (src/lane_repro/diffusion_losses.py).

Pure math checks — no diffusion model is loaded anywhere here. These establish that the loss terms
behave as the paper requires (zero on identical images, differentiable, mask-restricted, SSIM in
range) so they can be trusted once wired into the GPU generation path.
"""
import numpy as np
import pytest
import torch

from lane_repro import diffusion_losses as dl


def _rand_img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


# ----- to_tensor ------------------------------------------------------------

def test_to_tensor_uint8_scales_to_unit_range():
    arr = (np.random.default_rng(0).integers(0, 256, (8, 8, 3))).astype(np.uint8)
    t = dl.to_tensor(arr)
    assert t.shape == (1, 3, 8, 8)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    assert t.dtype == torch.float32


# ----- masked_mse / L_lane --------------------------------------------------

def test_masked_mse_zero_on_identical():
    x = _rand_img()
    mask = torch.ones(1, 1, 32, 32)
    assert float(dl.masked_mse(x, x.clone(), mask)) == pytest.approx(0.0, abs=1e-7)


def test_masked_mse_only_counts_masked_pixels():
    x = torch.zeros(1, 3, 4, 4)
    y = x.clone()
    y[:, :, 0, 0] = 1.0            # one pixel differs, value 1 in every channel
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 0, 0] = 1           # mask covers exactly that pixel
    # averaged over masked pixels×channels = (1^2 * 3) / (1 * 3) = 1.0
    assert float(dl.masked_mse(y, x, mask)) == pytest.approx(1.0, abs=1e-6)
    # a mask that excludes the differing pixel -> zero loss
    other = torch.zeros(1, 1, 4, 4); other[:, :, 3, 3] = 1
    assert float(dl.masked_mse(y, x, other)) == pytest.approx(0.0, abs=1e-7)


def test_masked_mse_empty_mask_is_zero_not_nan():
    x = _rand_img(); y = _rand_img(seed=1)
    empty = torch.zeros(1, 1, 32, 32)
    out = dl.masked_mse(y, x, empty)
    assert float(out) == pytest.approx(0.0) and np.isfinite(float(out))


def test_lane_loss_differentiable():
    x = _rand_img()
    gen = _rand_img(seed=2).clone().requires_grad_(True)
    mask = torch.ones(1, 1, 32, 32)
    loss = dl.lane_consistency_loss(gen, x, mask)
    loss.backward()
    assert gen.grad is not None and torch.isfinite(gen.grad).all()
    assert gen.grad.abs().sum() > 0


# ----- SSIM / L_env ---------------------------------------------------------

def test_ssim_identical_is_one():
    x = _rand_img()
    assert float(dl.ssim(x, x.clone())) == pytest.approx(1.0, abs=1e-4)


def test_ssim_in_range_and_lower_for_different_images():
    x = _rand_img(seed=3)
    y = _rand_img(seed=4)
    s_same = float(dl.ssim(x, x.clone()))
    s_diff = float(dl.ssim(x, y))
    assert -1.0 - 1e-4 <= s_diff <= 1.0 + 1e-4
    assert s_diff < s_same


def test_env_loss_zero_on_identical_and_positive_on_difference():
    x = _rand_img(seed=5)
    mask = torch.ones(1, 1, 32, 32)
    assert float(dl.env_consistency_loss(x, x.clone(), mask)) == pytest.approx(0.0, abs=1e-4)
    y = _rand_img(seed=6)
    assert float(dl.env_consistency_loss(y, x, mask)) > 0.0


def test_env_loss_differentiable():
    x = _rand_img(seed=7)
    gen = _rand_img(seed=8).clone().requires_grad_(True)
    mask = torch.ones(1, 1, 32, 32)
    dl.env_consistency_loss(gen, x, mask).backward()
    assert gen.grad is not None and torch.isfinite(gen.grad).all()


# ----- combined -------------------------------------------------------------

def test_stealth_loss_weights_and_total():
    x = _rand_img(seed=9); gen = _rand_img(seed=10)
    lane_mask = torch.ones(1, 1, 32, 32)
    env_mask = torch.ones(1, 1, 32, 32)
    out = dl.stealth_loss(gen, x, lane_mask, env_mask, w_lane=2.0, w_env=3.0)
    expected = 2.0 * out['l_lane'] + 3.0 * out['l_env']
    assert float(out['total']) == pytest.approx(float(expected), rel=1e-6)


def test_stealth_loss_zero_when_identical():
    x = _rand_img(seed=11)
    m = torch.ones(1, 1, 32, 32)
    out = dl.stealth_loss(x, x.clone(), m, m)
    assert float(out['total']) == pytest.approx(0.0, abs=1e-4)


# ----- mask builders --------------------------------------------------------

def test_lane_mask_from_lanes_marks_only_near_lane():
    lanes = [[(10, 5), (10, 25)]]       # a vertical segment at x=10
    mask = dl.lane_mask_from_lanes(lanes, height=32, width=32, thickness=4)
    assert mask.shape == (32, 32)
    assert mask[15, 10] == 1            # on the lane
    assert mask[15, 30] == 0            # far from it


def test_env_mask_proxy_excludes_trigger_and_lanes():
    lane_mask = np.zeros((32, 32), np.uint8); lane_mask[0, 0] = 1
    env = dl.env_mask_proxy(32, 32, dict(y0=4, x0=4, size=8), lane_mask)
    assert env[20, 20] == 1            # ordinary scene pixel kept (outside trigger & lanes)
    assert env[5, 5] == 0             # inside trigger footprint [4:12,4:12] -> excluded
    assert env[0, 0] == 0            # lane pixel -> excluded
