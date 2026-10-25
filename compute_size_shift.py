#!/usr/bin/env python3
"""Compute size_shift.csv from existing per_image_stats data.

Uses the reference.json built from 120 half-B images and the per-image
fingerprint stats from each scale to build 300 batches of 10 and compute
statistics.

Columns: scale, median_score, p5_score, p95_score, green_share, amber_share, red_share,
         mean_particles_per_image, median_area_vs_ref_percent, weighted_contribution_sum
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys

import numpy as np

# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = "/Users/ishaan/AI_HACKATHON_HEALTHCARE"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "fingerprint_cache")
SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

# ============================================================================
# Load reference
# ============================================================================

with open(os.path.join(PROJECT_ROOT, "reference.json"), "r") as f:
    reference = json.load(f)

ref_sample = reference["ref_sample"]
print(f"Reference: {reference['n_images']} images")
print(f"  ref_sample: particles_per_image={ref_sample['particles_per_image']:.1f}, "
      f"median_area={ref_sample['median_area']:.1f}")

# ============================================================================
# Feature weights (same as mix_curve.py)
# ============================================================================

features_list = ["particles_per_image", "median_area", "area_iqr",
                 "mean_circularity", "clumping_score"]

weights = {
    "particles_per_image": 1.0,
    "median_area": 0.02,
    "area_iqr": 0.5,
    "mean_circularity": 0.1,
    "clumping_score": 0.5,
}
total_weight = sum(weights.values())


# ============================================================================
# Aggregate sample from feature list
# ============================================================================

def aggregate_sample(feature_list_items):
    """Aggregate per-image features from a list of feature dicts."""
    total_n = 0
    med_areas = []
    iqrs = []
    all_circ = []
    clumps = []

    for f in feature_list_items:
        if f["n_particles"] > 0:
            total_n += f["n_particles"]
            med_areas.append(f["median_area"])
            iqrs.append(f["area_iqr"])
            all_circ.extend([f["mean_circularity"]] * f["n_particles"])
            clumps.append(f["clumping_score"])

    particles_per_image = float(np.mean([f["n_particles"] for f in feature_list_items]))
    median_area = float(np.mean(med_areas)) if med_areas else float("nan")
    area_iqr = float(np.mean(iqrs)) if iqrs else float("nan")
    mean_circularity = float(np.mean(all_circ)) if all_circ else 0.0
    mean_clumping = float(np.mean(clumps)) if clumps else 0.0

    return {
        "particles_per_image": particles_per_image,
        "median_area": median_area,
        "area_iqr": area_iqr,
        "mean_circularity": mean_circularity,
        "clumping_score": mean_clumping,
    }


# ============================================================================
# Compute match score from aggregated sample
# ============================================================================

def compute_match_score(sample, reference_sample):
    """Compute match score 0-100 for a sample vs reference."""
    deviations = {}
    for feat in features_list:
        fp_val = sample.get(feat, float("nan"))
        ref_val = reference_sample.get(feat, float("nan"))
        if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
            deviations[feat] = float("inf")
        else:
            rel = abs(fp_val - ref_val) / ref_val
            deviations[feat] = rel

    weighted_sum = sum(weights[feat] * deviations.get(feat, 0) for feat in features_list)
    raw_score = 100.0 / (1.0 + weighted_sum * 10)
    return max(0.0, min(100.0, raw_score))


# ============================================================================
# Main: compute statistics for each scale
# ============================================================================

all_scale_results = {}

for scale in SCALES:
    print(f"\n{'='*60}")
    print(f"Processing scale = {scale}")
    print(f"{'='*60}")

    # Load per-image stats
    per_image_path = os.path.join(RESULTS_DIR, f"scale_{scale}", "per_image_stats.json")
    
    if not os.path.exists(per_image_path):
        print(f"  WARNING: No per_image_stats.json found for scale {scale}, skipping")
        continue

    with open(per_image_path, "r") as f:
        all_features = json.load(f)

    print(f"  Loaded {len(all_features)} per-image feature sets")

    # Set random seed for reproducibility (same as mix_curve.py uses seed 42)
    rng = random.Random(42)

    # Build 300 batches of 10 by resampling with replacement from the 120 images
    n_batches = 300
    batch_scores = []
    batch_mean_pp = []
    batch_median_area_ratio = []
    batch_deviations = {feat: [] for feat in features_list}
    verdict_counts = {"green": 0, "amber": 0, "red": 0}

    for b in range(n_batches):
        # Resample 10 images with replacement
        batch_indices = rng.choices(range(len(all_features)), k=10)
        batch_features = [all_features[idx] for idx in batch_indices]

        # Aggregate
        sample = aggregate_sample(batch_features)

        # Compute match score
        score = compute_match_score(sample, ref_sample)
        batch_scores.append(score)

        # Mean particles_per_image
        mean_pp = sample["particles_per_image"]
        batch_mean_pp.append(mean_pp)

        # Median particle area vs reference (percent)
        if not math.isnan(sample["median_area"]) and ref_sample["median_area"] > 0:
            median_area_ratio = sample["median_area"] / ref_sample["median_area"] * 100
        else:
            median_area_ratio = float("nan")
        batch_median_area_ratio.append(median_area_ratio)

        # Store deviations for weighted contribution
        for feat in features_list:
            fp_val = sample.get(feat, float("nan"))
            ref_val = ref_sample.get(feat, float("nan"))
            if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
                dev = float("inf")
            else:
                dev = abs(fp_val - ref_val) / ref_val
            batch_deviations[feat].append(dev * weights[feat])

        # Determine verdict using FIXED thresholds: GREEN=18.9, RED=6.4
        if score > 18.9:
            verdict = "green"
        elif score < 6.4:
            verdict = "red"
        else:
            verdict = "amber"
        verdict_counts[verdict] += 1

    # Compute statistics across 300 batches
    mean_score = float(np.mean(batch_scores))
    median_score = float(np.median(batch_scores))
    p5_score = float(np.percentile(batch_scores, 5))
    p95_score = float(np.percentile(batch_scores, 95))

    mean_pp_image = float(np.mean(batch_mean_pp))
    # Median of median_area_vs_ref (skip NaNs)
    valid_area_ratios = [r for r in batch_median_area_ratio if not math.isnan(r)]
    median_area_vs_ref = float(np.median(valid_area_ratios)) if valid_area_ratios else float("nan")

    # Weighted contribution sum (average across batches)
    avg_weighted_contrib = float(np.mean([sum(d) for d in batch_deviations.values()]))

    green_share = verdict_counts["green"] / n_batches
    amber_share = verdict_counts["amber"] / n_batches
    red_share = verdict_counts["red"] / n_batches

    all_scale_results[scale] = {
        "mean_score": mean_score,
        "median_score": median_score,
        "p5_score": p5_score,
        "p95_score": p95_score,
        "green_share": green_share,
        "amber_share": amber_share,
        "red_share": red_share,
        "mean_particles_per_image": mean_pp_image,
        "median_area_vs_ref_percent": median_area_vs_ref,
        "weighted_contribution_sum": avg_weighted_contrib,
        "scale": scale,
    }

    print(f"  Results:")
    print(f"    median_score={median_score:.1f}, p5={p5_score:.1f}, p95={p95_score:.1f}")
    print(f"    green={green_share:.1%}, amber={amber_share:.1%}, red={red_share:.1%}")
    print(f"    mean_particles={mean_pp_image:.1f}, area_vs_ref={median_area_vs_ref:.1f}%")
    print(f"    weighted_contrib_sum={avg_weighted_contrib:.4f}")

# ============================================================================
# Write results/size_shift.csv
# ============================================================================

print(f"\n{'='*60}")
print("Writing results/size_shift.csv")
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
    print(fIn.read())

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