"""DBALD's two stealthiness loss terms, L_lane and L_env (paper §4.3; supplementary §1).

Paper definitions (main paper §4.3, verbatim):
    L_lane = MSE(gen_img ⊙ lane_mask,  clean_img ⊙ lane_mask)
    L_env  = SSIM(gen_img ⊙ env_mask,  clean_img ⊙ env_mask)
where ⊙ is element-wise masking, lane_mask marks lane-marking pixels (keep lane geometry
intact) and env_mask marks surrounding traffic/environment pixels (keep the scene coherent).
Supplementary §1 adds that the two terms are applied in *image space after decoding the latent*
and "back-propagate through the denoising steps", i.e. they are a guidance signal on the diffusion
trajectory, not a post-hoc filter.

This module implements the loss MATH only (pure torch, CPU-testable, no diffusion model). Two
deliberate, documented choices, both recorded in SOURCES.md §7:

1. SIGN OF L_env. The paper writes "L_env = SSIM(...)", but SSIM is a *similarity* in [-1, 1]
   (higher = more similar), and the paper minimises its losses. Minimising raw SSIM would push the
   edit to be maximally DISsimilar — the opposite of "preserve the environment". So the quantity we
   actually minimise is `env_consistency_loss = 1 - SSIM`, i.e. we maximise SSIM. `ssim()` below
   returns the raw similarity if a caller wants the paper's literal written quantity; the loss
   helper returns 1 - SSIM. This sign choice is an interpretation, flagged as such.

2. LOSS WEIGHTS. The paper gives no weights for combining L_lane and L_env (nor a diffusion-step
   schedule for applying them). `stealth_loss` defaults to w_lane = w_env = 1.0; these are an
   assumption, overridable, and must be reported with any number produced using them.

Masking note: SSIM is computed with an 11×11 Gaussian window (Wang et al. 2004 defaults), so the
per-pixel SSIM map near a mask boundary mixes in a few pixels from just outside the mask. We average
the SSIM map over the mask; this is the standard windowed-SSIM-with-mask approximation, not a
pixel-exact restriction. Documented, not hidden.
"""
import numpy as np
import torch
import torch.nn.functional as F

# ----- tensor helpers -------------------------------------------------------

def to_tensor(image):
    """RGB uint8 HxWx3 (or float HxWx3 in [0,1]) -> (1,3,H,W) float32 in [0,1]. A torch tensor that
    is already (N,C,H,W) is returned unchanged (detached dtype/float32 enforced)."""
    if torch.is_tensor(image):
        t = image.float()
        if t.dim() == 3: t = t.unsqueeze(0)
        if t.dim() != 4: raise ValueError('tensor image must be (C,H,W) or (N,C,H,W)')
        return t
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3: raise ValueError('image must be HxWx3')
    if arr.dtype == np.uint8: arr = arr.astype(np.float32) / 255.0
    else: arr = arr.astype(np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()

def _broadcast_mask(mask, like):
    """mask: (H,W) / (1,H,W) / (N,1,H,W) truthy array or tensor -> (N,1,H,W) float on like's device."""
    if not torch.is_tensor(mask): mask = torch.from_numpy(np.asarray(mask))
    mask = mask.to(device=like.device, dtype=like.dtype)
    while mask.dim() < 4: mask = mask.unsqueeze(0)
    if mask.shape[-2:] != like.shape[-2:]:
        raise ValueError(f'mask spatial size {tuple(mask.shape[-2:])} != image {tuple(like.shape[-2:])}')
    return (mask > 0).to(like.dtype) if mask.dtype != torch.bool else mask.to(like.dtype)

# ----- L_lane: masked MSE ---------------------------------------------------

def masked_mse(gen, clean, mask):
    """Mean squared error between gen and clean, averaged over masked pixels only (all channels).
    gen, clean: (N,C,H,W) float in [0,1]. mask: truthy, broadcast to (N,1,H,W). Differentiable in
    gen. Returns a scalar tensor; 0 (still graph-connected via gen) when the mask is empty, so a
    frame with no lane pixels contributes nothing rather than crashing a batch."""
    if gen.shape != clean.shape: raise ValueError('gen/clean shape mismatch')
    m = _broadcast_mask(mask, gen)
    diff2 = (gen - clean) ** 2 * m                      # zero outside the mask
    denom = (m.sum() * gen.shape[1]).clamp(min=1.0)     # masked pixels × channels
    return diff2.sum() / denom

def lane_consistency_loss(gen, clean, lane_mask):
    """L_lane (paper §4.3): MSE inside the lane mask. Minimise to keep lane markings intact."""
    return masked_mse(gen, clean, lane_mask)

# ----- SSIM and L_env -------------------------------------------------------

def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w2d = (g[:, None] * g[None, :])
    return w2d.expand(channels, 1, window_size, window_size).contiguous()

def ssim_map(x, y, window_size=11, sigma=1.5, data_range=1.0):
    """Per-pixel SSIM map (N,C,H,W) between x and y (both (N,C,H,W) in [0, data_range]), using a
    Gaussian window (Wang et al. 2004). 'same' padding keeps the map the image size."""
    if x.shape != y.shape: raise ValueError('x/y shape mismatch')
    c = x.shape[1]
    win = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2
    def flt(t): return F.conv2d(t, win, padding=pad, groups=c)
    mu_x, mu_y = flt(x), flt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = flt(x * x) - mu_x2
    sigma_y2 = flt(y * y) - mu_y2
    sigma_xy = flt(x * y) - mu_xy
    c1 = (0.01 * data_range) ** 2; c2 = (0.03 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))

def ssim(x, y, window_size=11, sigma=1.5, data_range=1.0):
    """Mean SSIM similarity over the whole image (scalar tensor in [-1, 1]; 1 == identical). This is
    the paper's literal written L_env quantity; see module docstring for why the LOSS negates it."""
    return ssim_map(x, y, window_size, sigma, data_range).mean()

def masked_ssim(x, y, mask, window_size=11, sigma=1.5, data_range=1.0):
    """Mean SSIM over masked pixels only (the env mask). See module docstring on the windowed-mask
    approximation. Returns 1.0 (perfect similarity, zero env loss) for an empty mask."""
    smap = ssim_map(x, y, window_size, sigma, data_range)
    m = _broadcast_mask(mask, x)
    denom = (m.sum() * x.shape[1]).clamp(min=1.0)
    return (smap * m).sum() / denom

def env_consistency_loss(gen, clean, env_mask, window_size=11, sigma=1.5, data_range=1.0):
    """L_env (paper §4.3), as a quantity to MINIMISE: 1 - SSIM inside the env mask. Minimising this
    maximises structural similarity to the clean scene. See module docstring, sign note (1)."""
    if gen.shape != clean.shape: raise ValueError('gen/clean shape mismatch')
    return 1.0 - masked_ssim(gen, clean, env_mask, window_size, sigma, data_range)

# ----- combined -------------------------------------------------------------

def stealth_loss(gen, clean, lane_mask, env_mask, w_lane=1.0, w_env=1.0,
                 window_size=11, sigma=1.5, data_range=1.0):
    """Combined DBALD stealth objective on a decoded edit. Returns a dict of scalar tensors
    {l_lane, l_env, total} with total = w_lane·L_lane + w_env·L_env. Weights default to 1.0 each
    (unspecified by the paper; see SOURCES.md §7). All terms differentiable in `gen`."""
    l_lane = lane_consistency_loss(gen, clean, lane_mask)
    l_env = env_consistency_loss(gen, clean, env_mask, window_size, sigma, data_range)
    return dict(l_lane=l_lane, l_env=l_env, total=w_lane * l_lane + w_env * l_env)

# ----- mask construction ----------------------------------------------------

def lane_mask_from_lanes(lanes_xy, height, width, thickness=12):
    """Binary (H,W) uint8 mask of lane-marking pixels: each lane's valid (x,y) points drawn as a
    polyline of the given thickness. lanes_xy: list of lanes, each a list of (x,y) in pixel coords,
    x<0 == absent. Matches the lane-geometry convention in poison_pipeline / label_attacks."""
    import cv2
    mask = np.zeros((height, width), np.uint8)
    for lane in lanes_xy:
        pts = [[int(x), int(y)] for x, y in lane if x is not None and x >= 0]
        if len(pts) > 1:
            cv2.polylines(mask, [np.asarray(pts, np.int32)], False, 1, thickness)
    return mask

def env_mask_proxy(height, width, trigger_window, lane_mask):
    """PROXY env mask, pending a real object segmenter (SAM2 / a detector), documented in SOURCES.md
    §7: everything EXCEPT the trigger footprint and the lane markings. The paper's env_mask is the
    surrounding *traffic objects*; without object masks we fall back to "the rest of the scene",
    which is a superset. trigger_window: dict(y0,x0,size) or (y0,x0,size) tuple. Returns (H,W) uint8."""
    if isinstance(trigger_window, dict): y0, x0, size = trigger_window['y0'], trigger_window['x0'], trigger_window['size']
    else: y0, x0, size = trigger_window
    mask = np.ones((height, width), np.uint8)
    mask[y0:y0 + size, x0:x0 + size] = 0                 # exclude the trigger itself
    mask[np.asarray(lane_mask) > 0] = 0                  # exclude lane markings (covered by L_lane)
    return mask
