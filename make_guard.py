#!/usr/bin/env python3
"""make_guard.py - Compute foreground fractions for all 500 real images
and choose a threshold that flags image_000 and image_218 with <=1% false positives.

Uses inference.segment() to get labels, then computes (labels > 0).sum() / labels.size.
"""

from inference import segment
import glob, os
import numpy as np
import json
import sys

# Get all 500 real images from dataset/images
real_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "images")
paths = sorted(glob.glob(os.path.join(real_dir, "*.tif")))
print(f"Total real images: {len(paths)}")

foreground_fracs = {}
for p in paths:
    labels, particles = segment(p)
    pixel_frac = (labels > 0).sum() / labels.size
    foreground_fracs[os.path.basename(p)] = pixel_frac

# Print image_000 and image_218
print(f"image_000 fraction: {foreground_fracs['image_000.tif']:.6f} ({foreground_fracs['image_000.tif']*100:.4f}%)")
print(f"image_218 fraction: {foreground_fracs['image_218.tif']:.6f} ({foreground_fracs['image_218.tif']*100:.4f}%)")

# Compute percentiles
fracs = sorted(foreground_fracs.values())
import numpy as np
print(f"Min: {min(fracs):.6f} ({min(fracs)*100:.4f}%)")
print(f"1st percentile: {np.percentile(fracs, 1):.6f} ({np.percentile(fracs, 1)*100:.4f}%)")
print(f"5th percentile: {np.percentile(fracs, 5):.6f} ({np.percentile(fracs, 5)*100:.4f}%)")
print(f"Median: {np.median(fracs):.6f} ({np.median(fracs)*100:.4f}%)")
print(f"Max: {max(fracs):.6f} ({max(fracs)*100:.4f}%)")

# How many images have fraction < 0.01 (1%)?
below_1pct = sum(1 for f in foreground_fracs.values() if f < 0.01)
print(f"Images with fraction < 1%: {below_1pct}/{len(foreground_fracs)}")

# Check image_000 and image_218 specifically
print(f"image_000 < 1%: {foreground_fracs['image_000.tif'] < 0.01}")
print(f"image_218 < 1%: {foreground_fracs['image_218.tif'] < 0.01}")

# Find threshold that flags image_000 and image_218 and <=5 images total
# (1% of 500 = 5)
sorted_fracs = sorted(foreground_fracs.items(), key=lambda x: x[1])
print("\nChoosing threshold...")
third_smallest_frac = sorted_fracs[2][1] if len(sorted_fracs) > 2 else None
print(f"3rd smallest fraction: {third_smallest_frac:.6f}")

# Choose threshold between image_218 and 3rd smallest
threshold = (foreground_fracs['image_218.tif'] + third_smallest_frac) / 2
print(f"Suggested threshold: {threshold:.6f}")

# Verify
flagged = [name for name, f in sorted_fracs if f < threshold]
print(f"Flagged images: {flagged}")
print(f"Count: {len(flagged)}")

# Save threshold to JSON
result = {
    "threshold": threshold,
    "flagged_images": flagged,
    "total_images": len(foreground_fracs),
    "image_000_frac": foreground_fracs['image_000.tif'],
    "image_218_frac": foreground_fracs['image_218.tif']
}
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "guard_threshold.json")
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved threshold to {output_path}")
print(f"JSON content: {json.dumps(result, indent=2)}")