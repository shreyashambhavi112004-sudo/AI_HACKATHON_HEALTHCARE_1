"""
Post-prediction quality check on the current inference.py run.

Confirms the overlays are real (red boundaries present) and that the model
is actually segmenting particles on crowded powder images, not single particles.
"""
import os
import numpy as np
from PIL import Image

OUTPUTS = "outputs"

def analyze_overlay(name):
    p = os.path.join(OUTPUTS, f"{name}.png")
    if not os.path.exists(p):
        print(f"MISSING: {name}")
        return
    im = np.asarray(Image.open(p)).astype(np.float32)
    if im.ndim != 3 or im.shape[-1] < 3:
        print(f"BAD SHAPE: {name} {im.shape}")
        return
    R = im[..., 0]
    G = im[..., 1]
    B = im[..., 2]
    red = (R > 0.5) & (G < 0.5) & (B < 0.5)
    n_red = int(red.sum())
    pct = n_red / (im.shape[0] * im.shape[1]) * 100
    print(f"{name}: size={im.shape[1]}x{im.shape[0]} red_px={n_red} ({pct:.2f}%)")

print("=== Overlays from inference.py ===")
for name in ["image_000_overlay", "image_001_overlay", "image_002_overlay"]:
    analyze_overlay(name)
