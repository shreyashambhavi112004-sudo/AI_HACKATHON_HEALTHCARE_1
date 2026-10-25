#!/usr/bin/env python3
"""generate_dense_v2.py - One-shot attempt to match real powder statistics.

Generates set A (30 images) in synthetic_dense_v2/A with particles whose
areas are matched to real particle sizes, background brightness matched
to real powder, and particle brightness adjusted if dark fraction is too high.

Uses the same segment() from inference.py for measurement.
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
# Configuration
# ---------------------------------------------------------------------------

SINGLE_PARTICLE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "single_particles",
    "single",
)

# Real powder statistics (computed from inference run above)
# Area distribution from real powder (particles >20 px): mean=658, median=458, min=27, max=2746
REAL_MEAN_AREA = 658.0        # mean particle area in real powder (px)
REAL_MEDIAN_AREA = 458.0      # median particle area in real powder (px)
REAL_AREA_MIN = 27            # min real particle area (px)
REAL_AREA_MAX = 2746          # max real particle area (px)
REAL_BG_BRIGHTNESS = 0.975    # background mode/median intensity (0=black, 1=white)
REAL_DARK_FRACTION = 0.160    # fraction of pixels darker than Otsu threshold

# Generation parameters (set A only, 30 images)
particles_per_image = 66
image_size = 512
background_level = REAL_BG_BRIGHTNESS   # match real powder background
background_noise_std = 0.02

# ---------------------------------------------------------------------------
# Reuse the notebook's single-particle bank + transform logic
# ---------------------------------------------------------------------------

def _load_particle_bank():
    """Return list of (img_float32_01, mask_bool) extracted from the BMP bank."""
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


def match_area_scale(mask, target_area):
    """Resize mask so its area matches target_area.

    Scale factor = sqrt(target_area / current_mask_area).
    Returns resized mask (bool).
    """
    current_area = mask.sum()
    if current_area == 0 or target_area <= 0:
        return mask
    scale = math.sqrt(target_area / current_area)
    new_h = max(2, int(mask.shape[0] * scale))
    new_w = max(2, int(mask.shape[1] * scale))
    mask_r = resize(mask.astype(float), (new_h, new_w),
                    preserve_range=True, order=0, anti_aliasing=False) > 0.5
    return mask_r


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

        # Pick a particle and resize it to match a target real-particle area
        img_p, mask_p = random.choice(particles)

        # --- NEW: resize mask to match a target real-particle area ---
        # Sample a target area from the real distribution for each particle.
        # Use log-uniform sampling between min and max real areas.
        target_area = float(np.random.uniform(
            math.log(REAL_AREA_MIN), math.log(REAL_AREA_MAX)))
        target_area = math.exp(target_area)
        mask_p_scaled = match_area_scale(mask_p, target_area)
        img_p_scaled = resize(img_p, mask_p_scaled.shape,
                              preserve_range=True, anti_aliasing=True)

        result = transform_particle(img_p_scaled, mask_p_scaled, size_scale=size_scale)
        if result is None:
            continue
        img_p, mask_p = result

        # --- NEW: brightness normalization to match real powder ---
        # The single-particle BMPs have mean ~0.45-0.61; real powder has
        # background 0.975 and dark fraction 0.160. Brighten particles so the
        # generated dark fraction matches real.
        if img_p.mean() > 0:
            # Adaptive factor based on how bright the particle is after area scaling.
            # Brighter particles (after upscaling) need less correction; dimmer need more.
            target_particle_mean = 0.75  # empirical target for dark fraction ~0.16-0.18
            brightness_factor = target_particle_mean / max(img_p.mean(), 0.01)
            img_p = np.clip(img_p * brightness_factor, 0, 1)
        # -------------------------------------------------------

        h, w = mask_p.shape
        if h >= image_size or w >= image_size:
            continue

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
# Main: generate set A only, measure, compare, save figure
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading single-particle bank from:",
          SINGLE_PARTICLE_FOLDER)
    particles = _load_particle_bank()
    print("Particles in bank:", len(particles))

    base = os.path.dirname(os.path.abspath(__file__))

    # ---- Output directories ----
    out_img_dir = os.path.join(base, "synthetic_dense_v2", "images")
    out_msk_dir = os.path.join(base, "synthetic_dense_v2", "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    # ---- Generate set A (30 images) ----
    print()
    print("=== Generating set A: matching real powder reference ===")
    print(f"  particles/image={particles_per_image}, "
          f"size_scale=1.55, overlap_probability=0.08")

    img_dir = os.path.join(out_img_dir, "set_A")
    msk_dir = os.path.join(out_msk_dir, "set_A")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    generated_stats = []
    for i in range(30):
        image, mask = create_synthetic_image(
            particles,
            particles_per_image=particles_per_image,
            size_scale=1.55,
            overlap_probability=0.08,
        )
        imwrite(
            os.path.join(img_dir, f"image_{i:03d}.tif"),
            image.astype(np.float32),
        )
        imwrite(
            os.path.join(msk_dir, f"mask_{i:03d}.tif"),
            mask.astype(np.uint16),
        )

        if (i + 1) % 10 == 0 or i == 29:
            print(f"   generated {i + 1}/30")

        # Measure this image immediately
        stats = measure_image(os.path.join(img_dir, f"image_{i:03d}.tif"))
        generated_stats.append(stats)

    # ---- Measure real images (images/overlap_images) ----
    print()
    print("===== MEASURING REAL IMAGES (images/overlap_images) =====")
    import glob as _glob

    real_dir = os.path.join(base, "images", "overlap_images")
    real_paths = sorted(_glob.glob(os.path.join(real_dir, "*.tif")))

    real_stats = []
    for p in real_paths:
        # Only measure images with actual content
        try:
            s = measure_image(p)
            real_stats.append(s)
        except Exception as e:
            print(f"  Skipping {os.path.basename(p)}: {e}")

    # ---- Compute summaries ----
    def summarize(stats_list, label):
        ns = [s["n_particles"] for s in stats_list]
        meds = [s["median_area"] for s in stats_list if not math.isnan(s["median_area"])]
        dfs = [s["dark_fraction"] for s in stats_list]
        return {
            "label": label,
            "n_images": len(stats_list),
            "particles_mean": float(np.mean(ns)),
            "particles_median": float(np.median(ns)),
            "median_area_mean": float(np.mean(meds)) if meds else float("nan"),
            "median_area_median": float(np.median(meds)) if meds else float("nan"),
            "dark_fraction_mean": float(np.mean(dfs)),
            "dark_fraction_median": float(np.median(dfs)),
        }

    ref = summarize(real_stats, "images/overlap_images")
    a = summarize(generated_stats, "synthetic_dense_v2/set_A")

    # ---- Print comparison table ----
    print()
    print("=== COMPARISON: Real vs Generated Set A ===")
    print(f"{'':>12} | Real          | Generated     ")
    print(f"{'Images':>12} | {ref['n_images']:>11} | {a['n_images']:>13}")
    print(f"{'Particles/img mean':>12} | {ref['particles_mean']:>11.1f} | {a['particles_mean']:>13.1f}")
    print(f"{'Particles/img median':>12} | {ref['particles_median']:>11.1f} | {a['particles_median']:>13.1f}")
    print(f"{'Median area mean':>12} | {ref['median_area_mean']:>11.1f} | {a['median_area_mean']:>13.1f}")
    print(f"{'Median area median':>12} | {ref['median_area_median']:>11.1f} | {a['median_area_median']:>13.1f}")
    print(f"{'Dark fraction mean':>12} | {ref['dark_fraction_mean']:>11.3f} | {a['dark_fraction_mean']:>13.3f}")
    print(f"{'Dark fraction median':>12} | {ref['dark_fraction_median']:>11.3f} | {a['dark_fraction_median']:>13.3f}")

    # ---- Save compare_A.png: a real image next to a generated set A image ----
    print()
    print("Saving compare_A.png ...")

    # Pick a real image and a generated set A image
    real_img_path = real_paths[0] if real_paths else None
    gen_img_dir = os.path.join(out_img_dir, "set_A")
    gen_paths = sorted([p for p in _glob.glob(os.path.join(gen_img_dir, "*.tif"))])
    gen_img_path = gen_paths[0] if gen_paths else None

    if real_img_path and gen_img_path:
        from skimage.segmentation import find_boundaries
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Load and normalize both images
        def load_norm(path):
            arr = np.array(Image.open(path))
            if arr.dtype.kind == "f":
                img = arr.astype(np.float32, copy=False)
            else:
                img = (arr.astype(np.float32, copy=False) / 255.0)
            if img.ndim == 3:
                img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
            return normalize(img.astype(np.float32, copy=False), 1, 99.8)

        from csbdeep.utils import normalize

        real_img = load_norm(real_img_path)
        gen_img = load_norm(gen_img_path)

        # Segment both
        real_labels, real_particles = segment(real_img_path, refine=True)
        gen_labels, gen_particles = segment(gen_img_path, refine=True)

        # Build overlays: gray background + red boundaries
        real_overlay = np.stack([real_img] * 3, axis=-1)
        real_overlay[find_boundaries(real_labels, mode="outer")] = [1, 0, 0]

        gen_overlay = np.stack([gen_img] * 3, axis=-1)
        gen_overlay[find_boundaries(gen_labels, mode="outer")] = [1, 0, 0]

        # Side-by-side figure
        h, w = real_overlay.shape[:2]
        combined = np.hstack([real_overlay, gen_overlay])

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(combined)
        ax.set_title(
            f"Real (left) vs Generated Set A (right)\n"
            f"Real: {len(real_particles)} particles, med area={np.median([p['area'] for p in real_particles]):.0f} px | "
            f"Gen: {len(gen_particles)} particles, med area={np.median([p['area'] for p in gen_particles]):.0f} px",
            fontsize=10
        )
        ax.axis("off")
        plt.tight_layout()
        compare_path = os.path.join(base, "compare_A.png")
        fig.savefig(compare_path, dpi=150)
        plt.close(fig)
        print(f"  Saved to: {compare_path}")
    else:
        print("  Could not save compare_Apng: missing real or generated images.")

    print()
    print("Done. Generated images in:", out_img_dir)
    print("Generated masks in:", out_msk_dir)


if __name__ == "__main__":
    main()