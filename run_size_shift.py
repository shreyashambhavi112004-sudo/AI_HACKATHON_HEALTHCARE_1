#!/usr/bin/env python3
"""Fix size_shift.csv by running segmentation and fingerprinting on zoomed real images.

Runs scales 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 in background with nohup,
using 60 half-B real images (seed 42). Caches per-image fingerprints per scale.
Builds 300 batches of 10 per scale. Reports statistics.

Usage: python3 run_size_shift.py &  (run in background)
     or: nohup python3 run_size_shift.py > size_shift.log 2>&1 &
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import csv
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter as gaussian

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Project root - always use the absolute path
# ============================================================================

PROJECT_ROOT = "/Users/ishaan/AI_HACKATHON_HEALTHCARE"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
REFERENCE_PATH = os.path.join(PROJECT_ROOT, "reference.json")
SINGLE_PARTICLE_FOLDER = os.path.join(PROJECT_ROOT, "images", "single_particles", "single")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Background particle bank loading (same as original script)
def _load_particle_bank():
    """Return list of (img_float32_01, mask_bool) from the BMP bank."""
    files = []
    for root, _dirs, files_in_dir in os.walk(SINGLE_PARTICLE_FOLDER):
        for name in files_in_dir:
            if name.lower().endswith(".bmp"):
                files.append(os.path.join(root, name))
    files.sort()

    particles = []
    for path in files:
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        thr = __import__("skimage.filters").threshold_otsu(img)
        mask = img < thr
        if mask.sum() < 20:
            continue
        with contextlib.suppress(Exception):
            mask = __import__("skimage.morphology").remove_small_objects(mask, min_size=20)
            mask = __import__("skimage.morphology").remove_small_holes(mask, area_threshold=20)
        if mask.sum() < 20:
            continue
        particles.append((img, mask))

    if not particles:
        raise RuntimeError(f"No single particles extracted from {SINGLE_PARTICLE_FOLDER}.")
    return particles


# ============================================================================
# Zoom function - zoom about centre, same output size
# ============================================================================

def zoom_about_centre(img, scale, background_level=0.9):
    """Zoom image about its centre by `scale` factor, same output size.

    scale < 1: zoom out, fill border with background level
    scale > 1: zoom in, crop centre
    scale = 1: no change (but goes through same code path)
    """
    h, w = img.shape

    if scale == 1.0:
        return img.copy()

    new_h = int(h * scale)
    new_w = int(w * scale)

    if new_h < 2 or new_w < 2:
        new_h = max(2, int(h * scale))
        new_w = max(2, int(w * scale))

    if scale < 1.0:
        # Zoom out: resize smaller, pad with background to original size
        resized = resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)
        if resized.dtype != np.float32:
            resized = resized.astype(np.float32)
        padded = np.full((h, w), background_level, dtype=np.float32)
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        padded[y_start:y_start + new_h, x_start:x_start + new_w] = resized
        return padded
    else:
        # Zoom in: resize larger, crop centre to original size
        resized = resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)
        if resized.dtype != np.float32:
            resized = resized.astype(np.float32)
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        return resized[y_start:y_start + h, x_start:x_start + w]


# ============================================================================
# Main
# ============================================================================

def main():
    import glob as _glob
    import json

    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("SIZE SHIFT TEST: scale variation study with zoomed real images")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Select 60 half-B real images (seed 42)
    # ------------------------------------------------------------------
    print("\n[1/7] Selecting 60 half-B real images (seed 42)...")

    all_real_paths = sorted(_glob.glob(os.path.join(IMAGES_DIR, "*.tif")))
    assert len(all_real_paths) == 500, f"Expected 500 real images, got {len(all_real_paths)}"

    # Split like mix_curve.py: seed 42, first 250 = half A, last 250 = half B
    rng = random.Random(42)
    shuffled = all_real_paths[:]
    rng.shuffle(shuffled)
    half_a_paths = shuffled[:250]
    half_b_paths = shuffled[250:]

    # Use first 60 from half B
    sample_60 = half_b_paths[:60]
    print(f"  Half A (ref): {len(half_a_paths)} images")
    print(f"  Half B (test): {len(half_b_paths)} images")
    print(f"  Selected 60 images from half B (indices 250-309 in shuffled list)")

    # Print first 5 image paths as required
    print("  First 5 image paths:")
    for p in sample_60[:5]:
        print(f"    {os.path.basename(p)}")

    # ------------------------------------------------------------------
    # 2. Load reference from reference.json (from half A)
    # ------------------------------------------------------------------
    print("\n[2/7] Loading reference from reference.json...")
    with open(REFERENCE_PATH, "r") as f:
        reference = json.load(f)
    ref_sample = reference["ref_sample"]
    print(f"  Reference: {reference['n_images']} images")
    print(f"  ref_sample: particles_per_image={ref_sample['particles_per_image']:.1f}, "
          f"median_area={ref_sample['median_area']:.1f}")

    # Fixed thresholds (do not change)
    GREEN = 18.9
    RED = 6.4

    # Features used in scoring (same as mix_curve.py)
    features_list = ["particles_per_image", "median_area", "area_iqr",
                     "mean_circularity", "clumping_score"]

    weights = {
        "particles_per_image": 1.0,
        "median_area": 0.02,
        "area_iqr": 0.5,
        "mean_circularity": 0.1,
        "clumping_score": 0.5,
    }

    # ------------------------------------------------------------------
    # 3. Run each scale: zoom images, fingerprint, cache, compute stats
    # ------------------------------------------------------------------
    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    all_scale_results = {}

    for scale_idx, scale in enumerate(scales):
        print(f"\n{'='*60}")
        print(f"[4/7] Running scale = {scale} (scale {scale_idx+1}/{len(scales)})")
        print(f"{'='*60}")

        scale_start = time.time()

        # Output directories for this scale
        scale_dir = os.path.join(RESULTS_DIR, f"scale_{scale}")
        img_dir = os.path.join(scale_dir, "images")
        cache_dir = os.path.join(scale_dir, "cache")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Fingerprint all 60 images and cache per-image results
        print(f"  Generating {len(sample_60)} zoomed images with size_scale={scale}...")
        print(f"  (progress printed every 10 images)")

        generated_stats = []
        for i in range(len(sample_60)):
            # Load original real image
            orig_path = sample_60[i]
            orig_img = np.array(Image.open(orig_path))
            if orig_img.dtype.kind == "f":
                img = orig_img.astype(np.float32, copy=False)
            else:
                img = (orig_img.astype(np.float32, copy=False) / 255.0)
            if img.ndim == 3:
                img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
            img = np.float32(img)

            # Zoom about centre
            zoomed_img = zoom_about_centre(img, scale)

            # Save zoomed image as TIFF
            zoomed_path = os.path.join(img_dir, f"image_{i:03d}.tif")
            # Save using PIL - convert to uint8 if needed for PIL, or use tifffile
            if zoomed_img.max() <= 1.0 and zoomed_img.dtype == np.float32:
                pil_img = Image.fromarray((zoomed_img * 255).astype(np.uint8))
            else:
                pil_img = Image.fromarray(zoomed_img.astype(np.uint8))
            pil_img.save(zoomed_path)

            # Measure (fingerprint) this image using inference.segment
            # Add project root to path and import
            sys.path.insert(0, PROJECT_ROOT)
            from inference import segment

            labels, particles = segment(zoomed_path)

            # Compute features manually (same as measure_image in run_size_shift.py)
            valid = [p for p in particles if p["area"] > 20]

            if not valid:
                stats = {
                    "n_particles": 0,
                    "median_area": float("nan"),
                    "area_iqr": float("nan"),
                    "mean_circularity": 0.0,
                    "clumping_score": 0.0,
                }
            else:
                areas = np.array([p["area"] for p in valid])
                circs = np.array([max(0.0, min(1.0, p["circularity"])) for p in valid])

                n = len(valid)
                median_area = float(np.median(areas))
                area_iqr = float(np.subtract(*np.percentile(areas, [75, 25])))
                mean_circ = float(np.mean(circs))

                lbl_max = int(labels.max())
                if lbl_max >= 2:
                    from scipy.ndimage import maximum_filter, minimum_filter
                    labels_np = np.asarray(labels)
                    big = lbl_max + 1
                    mx = maximum_filter(labels_np, size=3)
                    mn = minimum_filter(np.where(labels_np > 0, labels_np, big), size=3)
                    touch = (labels_np > 0) & ((mx != labels_np) | (mn != labels_np))
                    touching = np.unique(labels_np[touch])
                    areas_arr = np.bincount(labels_np.ravel(), minlength=lbl_max + 1)
                    total = areas_arr[1:].sum()
                    clump = float(areas_arr[touching].sum() / total) if total else 0.0
                else:
                    clump = 0.0

                stats = {
                    "n_particles": n,
                    "median_area": median_area,
                    "area_iqr": area_iqr,
                    "mean_circularity": mean_circ,
                    "clumping_score": clump,
                }

            generated_stats.append(stats)

            # Progress: print every 10 images
            if (i + 1) % 10 == 0 or i == len(sample_60) - 1:
                print(f"  processed {i + 1}/{len(sample_60)}")

        # Save per-image stats for this scale
        scale_stats_path = os.path.join(scale_dir, "per_image_stats.json")
        with open(scale_stats_path, "w") as fOut:
            json.dump(generated_stats, fOut, indent=2)
        print(f"  Saved per-image stats to {scale_stats_path}")

        # ------------------------------------------------------------------
        # 4. Build 300 batches of 10 by resampling from the 60 cached images
        # ------------------------------------------------------------------
        print(f"\n  Building 300 batches of 10 from 60 cached images...")

        # Load the cached features for all 60 images
        with open(scale_stats_path, "r") as f:
            all_60_features = json.load(f)

        print(f"  Loaded {len(all_60_features)} features")

        # ------------------------------------------------------------------
        # 5. Build 300 batches of 10 by resampling from the 60 images (with replacement)
        # ------------------------------------------------------------------
        print(f"  Building 300 batches of 10 from 60 images (with replacement)...")

        batch_results = []
        for b in range(300):
            # Resample 10 images with replacement from the 60
            batch_indices = random.choices(range(len(sample_60)), k=10)
            batch_features = [all_60_features[idx] for idx in batch_indices]

            # Aggregate features (same as aggregate_sample in fingerprint.py)
            total_n = 0
            med_areas = []
            iqrs = []
            all_circ = []
            clumps = []

            for f in batch_features:
                if f["n_particles"] > 0:
                    total_n += f["n_particles"]
                    med_areas.append(f["median_area"])
                    iqrs.append(f["area_iqr"])
                    all_circ.extend([f["mean_circularity"]] * f["n_particles"])
                    clumps.append(f["clumping_score"])

            particles_per_image = float(np.mean([f["n_particles"] for f in batch_features]))
            median_area = float(np.mean(med_areas)) if med_areas else float("nan")
            area_iqr = float(np.mean(iqrs)) if iqrs else float("nan")
            mean_circularity = float(np.mean(all_circ)) if all_circ else 0.0
            mean_clumping = float(np.mean(clumps)) if clumps else 0.0

            sample = {
                "particles_per_image": particles_per_image,
                "median_area": median_area,
                "area_iqr": area_iqr,
                "mean_circularity": mean_circularity,
                "clumping_score": mean_clumping,
            }

            # Compare to reference using compute_score logic (same as mix_curve.py)
            deviations = {}
            for feat in features_list:
                fp_val = sample.get(feat, float("nan"))
                ref_val = ref_sample.get(feat, float("nan"))
                if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
                    deviations[feat] = float("inf")
                else:
                    rel = abs(fp_val - ref_val) / ref_val
                    deviations[feat] = rel

            weighted_sum = sum(weights[feat] * deviations.get(feat, 0) for feat in features_list)
            raw_score = 100.0 / (1.0 + weighted_sum * 10)
            match_score = max(0.0, min(100.0, raw_score))

            batch_results.append({
                "batch_idx": b,
                "match_score": match_score,
                "sample": sample,
                "deviations": deviations,
            })

        # ------------------------------------------------------------------
        # 6. Compute statistics across 300 batches
        # ------------------------------------------------------------------
        print(f"  Computing statistics across 300 batches...")

        scores = [br["match_score"] for br in batch_results]
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        p5_score = float(np.percentile(scores, 5))
        p95_score = float(np.percentile(scores, 95))

        # Determine verdict distribution using FIXED thresholds:
        # GREEN = score > 18.9, RED = score < 6.4, Amber = between
        green_share = sum(1 for s in scores if s > GREEN) / 300
        red_share = sum(1 for s in scores if s < RED) / 300
        amber_share = 1.0 - green_share - red_share

        # Mean particles_per_image across batches
        mean_pp_image = float(np.mean([br["sample"]["particles_per_image"] for br in batch_results]))

        # Median particle area vs reference (percent)
        median_area_ratio = float(np.median([br["sample"]["median_area"] for br in batch_results]) /
                                  ref_sample["median_area"] * 100)

        # Each feature's weighted contribution
        feature_contributions = {}
        for feat in features_list:
            avg_dev = float(np.mean([br["deviations"][feat] for br in batch_results]))
            weight = weights[feat]
            feature_contributions[feat] = avg_dev * weight

        # Weighted contribution sum
        weighted_contrib_sum = sum(feature_contributions.values())

        scale_elapsed = time.time() - scale_start

        all_scale_results[scale] = {
            "mean_score": mean_score,
            "median_score": median_score,
            "p5_score": p5_score,
            "p95_score": p95_score,
            "green_share": green_share,
            "amber_share": amber_share,
            "red_share": red_share,
            "mean_particles_per_image": mean_pp_image,
            "median_area_vs_ref_percent": median_area_ratio,
            "weighted_contribution_sum": weighted_contrib_sum,
            "feature_contributions": feature_contributions,
            "scale": scale,
            "elapsed_seconds": scale_elapsed,
            "n_batches": 300,
            "n_images": len(sample_60),
        }

        print(f"  Scale {scale} complete in {scale_elapsed:.1f}s")
        print(f"    median_score={median_score:.1f}, p5={p5_score:.1f}, p95={p95_score:.1f}")
        print(f"    green={green_share:.1%}, amber={amber_share:.1%}, red={red_share:.1%}")
        print(f"    mean_particles={mean_pp_image:.1f}, area_vs_ref={median_area_ratio:.1f}%")
        print(f"    weighted_contrib_sum={weighted_contrib_sum:.4f}")

    # ------------------------------------------------------------------
    # 7. Write results/size_shift.csv
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[7/7] Writing results/size_shift.csv")
    print(f"{'='*60}")

    csv_path = os.path.join(RESULTS_DIR, "size_shift.csv")

    with open(csv_path, "w", newline="") as fOut:
        w = csv.writer(fOut)
        w.writerow([
            "scale",
            "median_score",
            "p5_score",
            "p95_score",
            "green_share",
            "amber_share",
            "red_share",
            "mean_particles_per_image",
            "median_area_vs_ref_percent",
            "weighted_contribution_sum",
        ])

        # Sort scales: 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
        sorted_scales = sorted(all_scale_results.keys())

        for scale in sorted_scales:
            r = all_scale_results[scale]
            w.writerow([
                r["scale"],
                r["median_score"],
                r["p5_score"],
                r["p95_score"],
                r["green_share"],
                r["amber_share"],
                r["red_share"],
                r["mean_particles_per_image"],
                r["median_area_vs_ref_percent"],
                r["weighted_contribution_sum"],
            ])

    print(f"\nSaved: {csv_path}")

    # Print the full contents
    print("\nFull size_shift.csv contents:")
    with open(csv_path, "r") as fIn:
        content = fIn.read()
        print(content)

    # Sanity checks
    print("\nSanity checks:")
    print(f"  Scale 1.0 median_score should be close to 30 (real control): {all_scale_results[1.0]['median_score']:.1f}")
    print(f"  Scale 1.0 green_share should be close to 96%: {all_scale_results[1.0]['green_share']:.1%}")
    print(f"  Scale 1.0 red_share should be close to 0%: {all_scale_results[1.0]['red_share']:.1%}")

    # Scale area factor check: median_area_vs_ref_percent should roughly follow scale^2
    print("\nScale area factor check (should roughly follow scale^2):")
    for scale in sorted_scales:
        r = all_scale_results[scale]
        expected_factor = scale ** 2
        print(f"  Scale {scale}: area_vs_ref={r['median_area_vs_ref_percent']:.1f}%, "
              f"expected scale^2={expected_factor:.2f}x, ratio={r['median_area_vs_ref_percent']/expected_factor:.2f}")

    # Final summary
    print("\n" + "=" * 60)
    print("SIZE SHIFT TEST COMPLETE")
    print("=" * 60)
    for scale in sorted_scales:
        r = all_scale_results[scale]
        print(f"Scale {scale:>4}: median={r['median_score']:.1f} "
              f"green={r['green_share']:.1%} red={r['red_share']:.1%} "
              f"particles={r['mean_particles_per_image']:.1f} "
              f"area_vs_ref={r['median_area_vs_ref_percent']:.1f}% "
              f"weighted_contrib={r['weighted_contribution_sum']:.4f}")

    return all_scale_results


if __name__ == "__main__":
    results = main()