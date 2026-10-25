#!/usr/bin/env python3
"""
generate_dense.py

Reuses the same particle-generating logic as synthetic_particle_generator_v2.ipynb
(transform_particle / sample_size-style transforms and freeze-frame placement with
an overlap probability), but writes to a NEW output tree:

    synthetic_dense/images  -> synthetic powder tifs (float32 0..1, 512x512)
    synthetic_dense/masks   -> matching instance masks (uint16)

Parameters controlled from the bottom of this file (the "SETS TO GENERATE" section):
  - particles_per_image : how many particles are placed per image
  - size_scale          : multiplicative factor on the notebook's random transform
                           sizes (>1 = bigger particles, 1.0 = same size distributions
                           as the notebook)
  - overlap_probability : notebook's OVERLAP_PROBABILITY; LOWER -> more clumping
                           (fewer placements skipped when they touch existing particles)

It generates three 30-image sets:
  (A) matching the real/synthetic powder reference (about 65 particles/image,
      median area ~570 px, ~19% dark pixels)
  (B) denser AND clumpier than real
  (C) larger particles than real (same-ish count as A)

After generation it measures each set with the SAME segment() used in inference.py,
and compares set A against the existing synthetic powder set in images/overlap_images
on: particle count, median area, and dark-pixel fraction.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
import warnings

import numpy as np
from PIL import Image
from tifffile import imwrite
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import resize, rotate
from scipy.ndimage import gaussian_filter as gaussian

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Reuse the notebook's single-particle bank + transform logic
# ---------------------------------------------------------------------------

SINGLE_PARTICLE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "images", "single_particles", "single"
)


def _load_particle_bank():
    """Return list of (img_float32_01, mask_bool) extracted from the BMP bank,
    mirroring the notebook's single-particle extraction. Each item is one
    individual particle."""
    files = []
    for root, _dirs, files_in_dir in os.walk(SINGLE_PARTICLE_FOLDER):
        for name in files_in_dir:
            if name.lower().endswith(".bmp"):
                files.append(os.path.join(root, name))
    files.sort()

    particles = []
    for path in files:
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        # Dark particles on light background.
        thr = threshold_otsu(img)
        mask = img < thr
        if mask.sum() < 20:
            continue
        with contextlib.suppress(Exception):
            mask = remove_small_objects(mask, min_size=20)
            mask = remove_small_holes(mask, area_threshold=20)
        if mask.sum() < 20:
            continue
        particles.append((img, mask))

    if not particles:
        raise RuntimeError(
            f"No single particles extracted from {SINGLE_PARTICLE_FOLDER}."
        )
    return particles


def sample_size(lo=3, hi=150):
    if lo <= 0 or hi < lo:
        raise ValueError("Require 0 < lo <= hi")
    return math.exp(random.uniform(math.log(lo), math.log(hi)))


def transform_particle(img, mask, size_scale=1.0):
    """Mirror the notebook's transform_particle, with an extra size_scale knob.

    The notebook picks a random longer-axis size with sample_size() and then
    scales the source particle up/down to that size (plus per-axis randomness).
    Here size_scale multiplies that longer-axis target size.
    """
    h, w = img.shape
    longer_target = sample_size() * size_scale
    longer = float(max(h, w))
    s = longer_target / longer

    # Per-axis randomness around 1.0, mirroring the notebook's independent
    # sx/sy stretch but keeping the intended scale.
    sy = random.uniform(0.75, 1.35)
    sx = random.uniform(0.75, 1.35)
    nh = max(2, round(h * s * sy))
    nw = max(2, round(w * s * sx))

    img_r = resize(img, (nh, nw), preserve_range=True, anti_aliasing=True)
    mask_r = resize(mask.astype(float), (nh, nw), preserve_range=True,
                    order=0, anti_aliasing=False) > 0.5

    angle = random.randint(0, 360)
    img_r = rotate(img_r, angle, resize=True, preserve_range=True,
                   mode="constant", cval=0.9)
    mask_r = rotate(mask_r.astype(float), angle, resize=True,
                    preserve_range=True, order=0, mode="constant", cval=0) > 0.5

    coords = np.argwhere(mask_r)
    if len(coords) < 3:
        return None

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    img_r = img_r[y1:y2 + 1, x1:x2 + 1]
    mask_r = mask_r[y1:y2 + 1, x1:x2 + 1]

    if random.random() < 0.3:
        sigma = random.uniform(0.6, 1.2) + longer_target * random.uniform(0, 0.05)
        img_r = gaussian(img_r, sigma=sigma)
        img_r = np.clip(img_r, 0, 1)

    return img_r, mask_r


def create_synthetic_image(
    particles,
    *,
    particles_per_image,
    size_scale=1.0,
    overlap_probability=0.1,
    image_size=512,
    background_level=0.9,
    background_noise_std=0.02,
):
    """Mirror the notebook's create_synthetic_image(), parameterized.

    Returns (image_float32_01, instance_mask_uint16).
    """
    canvas = np.random.normal(
        background_level, background_noise_std, (image_size, image_size)
    ).astype(np.float32)
    canvas = np.clip(canvas, 0, 1)
    instance_mask = np.zeros((image_size, image_size), dtype=np.uint16)

    n_target = particles_per_image
    particle_id = 1
    attempts = 0

    while particle_id <= n_target:
        attempts += 1
        if attempts > 40000:
            break

        img_p, mask_p = random.choice(particles)
        result = transform_particle(img_p, mask_p, size_scale=size_scale)
        if result is None:
            continue
        img_p, mask_p = result

        h, w = mask_p.shape
        if h >= image_size or w >= image_size:
            continue

        y = random.randint(0, image_size - h - 1)
        x = random.randint(0, image_size - w - 1)

        existing = instance_mask[y:y + h, x:x + w]
        overlap = int(np.sum(existing[mask_p] > 0))
        if overlap > 0 and random.random() > overlap_probability:
            continue

        canvas_region = canvas[y:y + h, x:x + w]
        canvas_region[mask_p] = img_p[mask_p]
        canvas[y:y + h, x:x + w] = canvas_region

        mask_region = instance_mask[y:y + h, x:x + w]
        mask_region[mask_p] = particle_id
        instance_mask[y:y + h, x:x + w] = mask_region

        particle_id += 1

    return canvas, instance_mask


# ---------------------------------------------------------------------------
# Stats helpers (dark-pixel fraction mirrors the notebook's polarity: dark
# particles on light background -> fraction of pixels darker than Otsu-level)
# ---------------------------------------------------------------------------


def dark_fraction(img):
    thr = threshold_otsu(img)
    return float(np.mean(img < thr))


# ---------------------------------------------------------------------------
# Segment using the existing inference.py pipeline (so generated images are
# evaluated exactly as the real/synthetic powder images are).
# ---------------------------------------------------------------------------

from inference import segment, _load_image  # noqa: E402


def measure_image(path):
    img = _load_image(path)
    labels, particles = segment(path)
    n_particles = len(particles)
    if n_particles:
        areas = [p["area"] for p in particles]
        median_area = float(np.median(areas))
    else:
        median_area = float("nan")
    df = float(dark_fraction(img))
    return {
        "n_particles": n_particles,
        "median_area": median_area,
        "dark_fraction": df,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SETS = [
    {
        "name": "A",
        "desc": "matching real powder reference",
        "n_images": 30,
        "particles_per_image": 66,       # target real count ~67 (mean)
        "size_scale": 1.55,            # larger base particles -> bigger median area
        "overlap_probability": 0.08,    # slightly more overlap -> higher dark frac
    },
    {
        "name": "B",
        "desc": "denser and clumpier than real",
        "n_images": 30,
        "particles_per_image": 105,     # clearly more than real
        "size_scale": 1.0,
        "overlap_probability": 0.0,      # no overlap rejection -> clumpier
    },
    {
        "name": "C",
        "desc": "larger particles than real",
        "n_images": 30,
        "particles_per_image": 40,      # fewer large particles than real count
        "size_scale": 2.6,             # clearly larger particles than real
        "overlap_probability": 0.06,
    },
]


def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading single-particle bank from:", SINGLE_PARTICLE_FOLDER)
    particles = _load_particle_bank()
    print("Particles in bank:", len(particles))

    base = os.path.dirname(os.path.abspath(__file__))
    out_image_dir = os.path.join(base, "synthetic_dense", "images")
    out_mask_dir = os.path.join(base, "synthetic_dense", "masks")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for cfg in SETS:
        print()
        print(f"=== Generating set {cfg['name']}: {cfg['desc']} ===")
        print(f"   particles/image={cfg['particles_per_image']}, "
              f"size_scale={cfg['size_scale']}, "
              f"overlap_probability={cfg['overlap_probability']}")

        img_dir = os.path.join(out_image_dir, f"set_{cfg['name']}")
        msk_dir = os.path.join(out_mask_dir, f"set_{cfg['name']}")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for i in range(cfg["n_images"]):
            image, mask = create_synthetic_image(
                particles,
                particles_per_image=cfg["particles_per_image"],
                size_scale=cfg["size_scale"],
                overlap_probability=cfg["overlap_probability"],
            )
            imwrite(
                os.path.join(img_dir, f"image_{i:03d}.tif"),
                image.astype(np.float32),
            )
            imwrite(
                os.path.join(msk_dir, f"mask_{i:03d}.tif"),
                mask.astype(np.uint16),
            )

            if (i + 1) % 10 == 0 or i == cfg["n_images"] - 1:
                print(f"   generated {i + 1}/{cfg['n_images']}")

    # ---------------------------------------------------------------------------
    # Measure generated sets + compare set A to images/overlap_images
    # ---------------------------------------------------------------------------
    import glob as _glob

    print()
    print("===== MEASURING GENERATED SETS (with inference.segment) =====")

    set_stats = {}
    for cfg in SETS:
        img_dir = os.path.join(out_image_dir, f"set_{cfg['name']}")
        paths = sorted(_glob.glob(os.path.join(img_dir, "*.tif")))
        stats = [measure_image(p) for p in paths]
        set_stats[cfg["name"]] = stats

        ns = [s["n_particles"] for s in stats]
        meds = [s["median_area"] for s in stats]
        dfs = [s["dark_fraction"] for s in stats]
        print()
        print(f"Set {cfg['name']} ({cfg['desc']}):")
        print(f"  n_images       = {len(stats)}")
        print(f"  particles      = mean={np.mean(ns):.1f}, "
              f"median={np.median(ns):.1f}, min={np.min(ns)}, max={np.max(ns)}")
        print(f"  median area    = mean={np.mean(meds):.1f}, "
              f"median={np.median(meds):.1f}, min={np.min(meds):.1f}, max={np.max(meds):.1f}")
        print(f"  dark fraction  = mean={np.mean(dfs):.3f}, "
              f"median={np.median(dfs):.3f}")

    print()
    print("===== COMPARE SET A vs images/overlap_images =====")

    overlap_dir = os.path.join(base, "images", "overlap_images")
    overlap_paths = sorted(_glob.glob(os.path.join(overlap_dir, "*.tif")))
    overlap_stats = [measure_image(p) for p in overlap_paths]

    def summarize(label, stats):
        ns = [s["n_particles"] for s in stats]
        meds = [s["median_area"] for s in stats]
        dfs = [s["dark_fraction"] for s in stats]
        return {
            "label": label,
            "n_images": len(stats),
            "particles_mean": float(np.mean(ns)),
            "particles_median": float(np.median(ns)),
            "median_area_mean": float(np.mean(meds)),
            "median_area_median": float(np.median(meds)),
            "dark_fraction_mean": float(np.mean(dfs)),
            "dark_fraction_median": float(np.median(dfs)),
        }

    ref = summarize("images/overlap_images", overlap_stats)
    a = summarize("synthetic_dense/set_A", set_stats["A"])

    print()
    print("Reference (images/overlap_images):")
    print(f"  n_images        = {ref['n_images']}")
    print(f"  particles       = mean={ref['particles_mean']:.1f}, "
          f"median={ref['particles_median']:.1f}")
    print(f"  median area     = mean={ref['median_area_mean']:.1f}, "
          f"median={ref['median_area_median']:.1f}")
    print(f"  dark fraction   = mean={ref['dark_fraction_mean']:.3f}, "
          f"median={ref['dark_fraction_median']:.3f}")

    print()
    print("Set A (synthetic_dense/set_A):")
    print(f"  n_images        = {a['n_images']}")
    print(f"  particles       = mean={a['particles_mean']:.1f}, "
          f"median={a['particles_median']:.1f}")
    print(f"  median area     = mean={a['median_area_mean']:.1f}, "
          f"median={a['median_area_median']:.1f}")
    print(f"  dark fraction   = mean={a['dark_fraction_mean']:.3f}, "
          f"median={a['dark_fraction_median']:.3f}")

    print()
    print("Comparison (Set A vs reference):")
    print(f"  particles: {a['particles_mean']:.1f} vs {ref['particles_mean']:.1f} "
          f"(mean), {a['particles_median']:.1f} vs {ref['particles_median']:.1f} (median)")
    print(f"  median area: {a['median_area_mean']:.1f} vs {ref['median_area_mean']:.1f} "
          f"(mean), {a['median_area_median']:.1f} vs {ref['median_area_median']:.1f} (median)")
    print(f"  dark fraction: {a['dark_fraction_mean']:.3f} vs {ref['dark_fraction_mean']:.3f} "
          f"(mean), {a['dark_fraction_median']:.3f} vs {ref['dark_fraction_median']:.3f} (median)")

    print()
    print("Done. Generated images in:", out_image_dir)
    print("Generated masks in:", out_mask_dir)


if __name__ == "__main__":
    main()
