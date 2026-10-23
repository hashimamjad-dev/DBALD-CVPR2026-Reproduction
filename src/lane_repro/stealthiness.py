"""Stealthiness proxies for a poisoned image, relative to DBALD's Table 2 (forensic-detector
evasion rates) and the paper's own LPIPS-based trigger-diversity check (Section on Clarification
of Trigger Diversity, threshold 0.15).

These are NOT the paper's forensic detectors (UniDetection, LGrad, DIRE): those need each
detector's own pretrained weights, which are not available in this project. The functions here are
lighter, honestly-labelled stand-ins: LPIPS similarity to the clean image (the same perceptual
metric the paper itself uses for trigger diversity, just applied to clean-vs-poisoned rather than
trigger-vs-trigger), and a frequency-domain energy ratio (poisoned region vs a same-size clean
reference region), since diffusion/compositing edits are reported in the literature to leave high-
frequency artefacts that a simple spectral check can sometimes surface. Neither claims to predict
what UniDetection/LGrad/DIRE would output.
"""
import numpy as np

def lpips_distance(clean,poisoned,box=None,net='squeeze'):
    """clean,poisoned: RGB uint8 arrays, same shape. box=(y0,x0,y1,x1) crops both to that region
    first (e.g. the trigger window plus context, matching this project's existing Quality class
    convention in scripts/v2_common.py). Returns a float; lower is more similar/stealthier. Loads
    the LPIPS network once per call -- callers scoring many images should hold a LPIPSScorer
    instance instead (see below) to avoid repeated network construction."""
    return LPIPSScorer(net).distance(clean,poisoned,box)

class LPIPSScorer:
    """Loads the LPIPS network once; reuse across many images instead of calling lpips_distance
    repeatedly. torch.hub's cache directory is left at its default unless the caller has already
    set one (matching scripts/v2_common.py's Quality class, which points it at
    models/metrics/torch/hub so the small SqueezeNet weights are not re-downloaded per run)."""
    def __init__(self,net='squeeze'):
        import lpips
        self.model=lpips.LPIPS(net=net).eval()
    def distance(self,clean,poisoned,box=None):
        import torch
        if clean.shape!=poisoned.shape: raise ValueError('clean/poisoned shape mismatch')
        if box is not None:
            y0,x0,y1,x1=box
            clean=clean[y0:y1,x0:x1]; poisoned=poisoned[y0:y1,x0:x1]
        if clean.size==0: raise ValueError('Empty crop box')
        def tensor(image):
            return torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0).float()/127.5-1
        with torch.no_grad():
            return float(self.model(tensor(clean),tensor(poisoned)).item())

def frequency_energy_ratio(clean_region,poisoned_region):
    """Ratio of high-frequency to total spectral energy (2D FFT magnitude, grayscale), poisoned
    over clean, for two same-shape regions (e.g. the trigger window from each image). >1 means the
    poisoned region carries relatively more high-frequency energy than the same spot in the clean
    image -- a coarse, NOT validated, proxy for compositing/edit artefacts; see module docstring."""
    if clean_region.shape!=poisoned_region.shape: raise ValueError('clean/poisoned shape mismatch')
    def high_freq_fraction(region):
        gray=region.astype(float).mean(axis=2) if region.ndim==3 else region.astype(float)
        spectrum=np.abs(np.fft.fftshift(np.fft.fft2(gray)))**2
        h,w=spectrum.shape; cy,cx=h//2,w//2
        yy,xx=np.ogrid[:h,:w]
        radius=np.sqrt((yy-cy)**2+(xx-cx)**2)
        low=spectrum[radius<=min(h,w)*0.1].sum()
        total=spectrum.sum()
        if total<=0: raise ValueError('Degenerate (all-zero) region')
        return 1.-low/total
    clean_fraction=high_freq_fraction(clean_region)
    if clean_fraction<=0: raise ValueError('Degenerate clean region (no high-frequency content to compare against)')
    return high_freq_fraction(poisoned_region)/clean_fraction
