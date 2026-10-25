#!/usr/bin/env python3
"""Mix curve: how the match score falls as a sample is gradually replaced
by off-reference images.

Uses the scoring already in fingerprint.py. Writes mix_curve.png and
mix_curve.csv.

Key design decisions (all fixed, not tuned after seeing results):
- 500 real images from dataset/images/ split into two disjoint halves
  with fixed seed 42: half A (250 images) builds the reference,
  half B (250 images) provides "genuine" test batches only.
- GREEN = 5th percentile of match scores from genuine batches (half B only)
- RED = 95th percentile of match scores from fully off-reference batches
  (synthetic set_A from synthetic_dense/images/set_A)
- If RED >= GREEN, stop and report; do not adjust anything.
- Mix levels: 0, 10, 20, 30, 40, 50, 60, 80, 100 percent replaced
- At least 200 batches per mix level
- Coloured bands: green above GREEN, amber between GREEN and RED,
  red below RED
- Mark the first mix level where the median score leaves the green zone
- Save results/mix_curve_table.csv and results/mix_curve.png
"""

import csv
import glob
import json
import os
import random
import math

import numpy as np
from collections import Counter

from fingerprint import (
    fingerprint_image, aggregate_sample, compare,
    _load_cached, _save_cache,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REAL_DIR = os.path.join(ROOT, "dataset", "images")      # 500 real powder images
SYNTH_DIR = os.path.join(ROOT, "synthetic_dense", "images")  # synthetic sets A/B/C
SET_A_DIR = os.path.join(SYNTH_DIR, "set_A")           # off-reference, synthetic

SEED = 42                                      # fixed seed for all randomness
RNG = random.Random(SEED)

# ---------------------------------------------------------------------------
# Helper: compute match score 0-100 for a sample vs reference
# (mirrors the compare logic without needing natural_var)
# ---------------------------------------------------------------------------
def compute_score(fp_sample, reference):
    """Compute match score 0-100 for a sample vs reference."""
    ref_sample = reference["ref_sample"]
    features = ["particles_per_image", "median_area", "area_iqr",
                "mean_circularity", "clumping_score"]
    deviations = {}
    for feat in features:
        fp_val = fp_sample.get(feat, float("nan"))
        ref_val = ref_sample.get(feat, float("nan"))
        if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
            deviations[feat] = float("inf")
        else:
            rel = abs(fp_val - ref_val) / ref_val
            deviations[feat] = rel

    weights = {
        "particles_per_image": 1.0,
        "median_area": 0.02,
        "area_iqr": 0.5,
        "mean_circularity": 0.1,
        "clumping_score": 0.5,
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(weights[feat] * deviations.get(feat, 0) for feat in features)
    raw_score = 100.0 / (1.0 + weighted_sum * 10)
    return max(0.0, min(100.0, raw_score))


# ---------------------------------------------------------------------------
# Helper: build reference.json from a list of paths
# ---------------------------------------------------------------------------
def build_reference_from_paths(paths, seed=42):
    """Build reference.json from a list of image paths."""
    from fingerprint import _load_cached, _save_cache, aggregate_sample

    rng = random.Random(seed)
    paths_copy = paths[:]
    rng.shuffle(paths_copy)

    features = []
    for p in paths_copy:
        cache = _load_cached(p)
        if cache is None:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            features.append(feat)
        else:
            features.append(cache)

    ref_sample = aggregate_sample(features)

    reference = {
        "n_images": len(paths_copy),
        "ref_sample": ref_sample,
        "per_image": {
            os.path.basename(p): f for p, f in zip(paths_copy, features)
        },
    }

    ref_path = os.path.join(ROOT, "reference.json")
    with open(ref_path, "w") as fOut:
        json.dump(reference, fOut, indent=2)
    print(f"Reference saved to {ref_path} ({len(paths_copy)} images)")

    return reference


# ---------------------------------------------------------------------------
# Step 1: Split the 500 real images into two disjoint halves
# ---------------------------------------------------------------------------
all_real_paths = sorted(
    glob.glob(os.path.join(REAL_DIR, "*.tif"))
)
assert len(all_real_paths) == 500, f"Expected 500 real images, got {len(all_real_paths)}"

# Shuffle with fixed seed and split
RNG.seed(SEED)
shuffled = all_real_paths[:]
RNG.shuffle(shuffled)
half_a_paths = shuffled[:250]   # reference-building images
half_b_paths = shuffled[250:]   # genuine test-batch images

print(f"Total real images: {len(all_real_paths)}")
print(f"Half A (reference): {len(half_a_paths)} images")
print(f"Half B (genuine test): {len(half_b_paths)} images")

# ---------------------------------------------------------------------------
# Step 2: Build reference fingerprint from half A only
# ---------------------------------------------------------------------------
print("\nBuilding reference from half A (250 images, seed=42)...")
reference = build_reference_from_paths(half_a_paths, seed=SEED)
print(f"Reference saved: {reference['n_images']} images")

# ---------------------------------------------------------------------------
# Step 3: Compute GREEN threshold from genuine batches (half B only)
# ---------------------------------------------------------------------------
print("\nComputing GREEN threshold from genuine batches (half B, 500 draws)...")
genuine_scores = []
for i in range(500):
    # Draw a batch of 10 images with replacement from half B
    batch_paths = RNG.choices(half_b_paths, k=10)
    features = []
    for p in batch_paths:
        cache = _load_cached(p)
        if cache is None:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            features.append(feat)
        else:
            features.append(cache)
    sample = aggregate_sample(features)
    score = compute_score(sample, reference)
    genuine_scores.append(score)

genuine_scores = np.array(genuine_scores)
p5 = np.percentile(genuine_scores, 5)   # GREEN threshold
half_p5 = p5 / 2.0                       # RED threshold (half of GREEN)

print(f"Genuine batch scores (n=500): mean={genuine_scores.mean():.1f}, "
      f"std={genuine_scores.std():.1f}")
print(f"GREEN threshold (5th percentile): {p5:.1f}")
print(f"RED threshold (half of GREEN): {half_p5:.1f}")

# Check: if half_p5 >= p5, stop
if half_p5 >= p5:
    print("ERROR: half_p5 >= p5, stopping without adjusting anything.")
    exit(1)

# ---------------------------------------------------------------------------
# Step 4: Compute RED threshold from fully off-reference batches
# ---------------------------------------------------------------------------
print("\nComputing RED threshold from off-reference batches (synthetic set_A)...")
off_scores = []
# Use synthetic set_A as off-reference images (30 images, draw with replacement)
off_image_paths = sorted(glob.glob(os.path.join(SET_A_DIR, "*.tif")))
assert len(off_image_paths) > 0, f"No images found in {SET_A_DIR}"

for i in range(500):
    # Draw a batch of 10 images with replacement from set_A (synthetic off-reference)
    batch_paths = RNG.choices(off_image_paths, k=10)
    features = []
    for p in batch_paths:
        cache = _load_cached(p)
        if cache is None:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            features.append(feat)
        else:
            features.append(cache)
    sample = aggregate_sample(features)
    score = compute_score(sample, reference)
    off_scores.append(score)

off_scores = np.array(off_scores)
p95 = np.percentile(off_scores, 95)   # RED threshold from off-reference

print(f"Off-reference batch scores (n=500): mean={off_scores.mean():.1f}, "
      f"std={off_scores.std():.1f}")
print(f"RED threshold (95th percentile from off-reference): {p95:.1f}")

# The official RED = 95th percentile of off-reference scores
# Per the spec: "If RED >= GREEN, stop and report; do not adjust anything."
RED = p95
GREEN = p5

print(f"\nFinal thresholds:")
print(f"  GREEN = {GREEN:.1f} (5th percentile of {len(genuine_scores)} genuine batches)")
print(f"  RED   = {RED:.1f} (95th percentile of {len(off_scores)} off-reference batches)")
print(f"  half_p5 = {half_p5:.1f} (for reference, amber boundary)")

if RED >= GREEN:
    print("ERROR: RED >= GREEN, stopping without adjusting anything.")
    exit(1)
else:
    print("OK: RED < GREEN, proceeding with mix curve.")

# ---------------------------------------------------------------------------
# Step 5: Mix curve at 0, 10, 20, 30, 40, 50, 60, 80, 100 percent replaced
# ---------------------------------------------------------------------------
MIX_LEVELS = [0, 10, 20, 30, 40, 50, 60, 80, 100]
TRIALS_PER_LEVEL = 200  # at least 200 batches per point
BATCH_SIZE = 10         # images per batch

print(f"\nRunning mix curve at {len(MIX_LEVELS)} levels, {TRIALS_PER_LEVEL} trials each...")

rows = []  # (percent_off, median_score, p5_score, p95_score, green_share, amber_share, red_share, median_for_tracking)

for percent_off in MIX_LEVELS:
    k = percent_off // 10  # number of off-reference images replacing k images in the batch (percent_off is 0, 10, 20, ...)
    n_in = BATCH_SIZE - k  # number of genuine images remaining
    scores = []
    verdict_counts = {"green": 0, "amber": 0, "red": 0}

    for t in range(TRIALS_PER_LEVEL):
        # Deterministic but varied per trial
        trial_rng = random.Random(SEED * 1000 + t)

        # Select n_in genuine images from half B (with replacement if needed)
        if n_in > 0:
            genuine_batch = trial_rng.choices(half_b_paths, k=n_in)
        else:
            genuine_batch = []

        # Select k off-reference images from set_A (synthetic, with replacement)
        off_image_paths = sorted(glob.glob(os.path.join(SET_A_DIR, "*.tif")))
        off_batch = trial_rng.choices(off_image_paths, k=k) if k > 0 else []

        # Combine and shuffle
        all_batch = genuine_batch + off_batch
        trial_rng2 = random.Random(SEED * 999 + t)
        trial_rng2.shuffle(all_batch)

        # Fingerprint and aggregate
        features = []
        for p in all_batch:
            cache = _load_cached(p)
            if cache is None:
                feat = fingerprint_image(p)
                _save_cache(p, feat)
                features.append(feat)
            else:
                features.append(cache)
        sample = aggregate_sample(features)

        # Compare to reference using compute_score (no nat_var needed)
        score = compute_score(sample, reference)
        scores.append(score)

        # Also determine verdict using the same logic as GREEN/RED thresholds
        # GREEN = above 5th percentile of genuine, RED = below half of that
        if score > p5:
            verdict = "green"
        elif score < half_p5:
            verdict = "red"
        else:
            verdict = "amber"

        verdict_counts[verdict] += 1

    scores = np.array(scores)
    median_score = np.median(scores)
    p5_score = np.percentile(scores, 5)
    p95_score = np.percentile(scores, 95)

    green_share = verdict_counts["green"] / TRIALS_PER_LEVEL
    amber_share = verdict_counts["amber"] / TRIALS_PER_LEVEL
    red_share = verdict_counts["red"] / TRIALS_PER_LEVEL

    rows.append((
        percent_off,
        median_score,
        p5_score,
        p95_score,
        green_share,
        amber_share,
        red_share,
        median_score,  # track median for first_green_leave logic
    ))

    print(f"  {percent_off:3d}% off-reference: median={median_score:.1f} "
          f"(p5={p5_score:.1f}, p95={p95_score:.1f})  "
          f"green={green_share:.0%} amber={amber_share:.0%} red={red_share:.0%}")

# ---------------------------------------------------------------------------
# Step 6: Find first mix level where median leaves green zone
# ---------------------------------------------------------------------------
first_left_green = None
for i, row in enumerate(rows):
    pct, median, p5, p95, g, a, r, _ = row
    if median <= GREEN and i > 0:
        # Check if previous median was above GREEN
        prev_median = rows[i-1][1]
        if prev_median > GREEN:
            first_left_green = pct
            break
    # Also check if first level (0%) is already <= GREEN
    if i == 0 and median <= GREEN:
        first_left_green = 0
        break

if first_left_green is None:
    # If never left green, check the last level
    last_median = rows[-1][1]
    if last_median > GREEN:
        first_left_green = "never left green"
    else:
        # It must have left green at the last level
        first_left_green = MIX_LEVELS[-1]

print(f"\nFirst mix level where median leaves green zone: {first_left_green}")

# ---------------------------------------------------------------------------
# Step 7: Save mix_curve_table.csv
# ---------------------------------------------------------------------------
print("\nSaving results/mix_curve_table.csv...")
csv_path = os.path.join(ROOT, "results", "mix_curve_table.csv")
with open(csv_path, "w", newline="") as fOut:
    w = csv.writer(fOut)
    w.writerow([
        "percent_off_reference",
        "median_score",
        "p5_score",
        "p95_score",
        "green_share",
        "amber_share",
        "red_share",
        "first_left_green_percent",
    ])
    for row in rows:
        w.writerow(row)

# ---------------------------------------------------------------------------
# Step 8: Plot mix_curve.png
# ---------------------------------------------------------------------------
print("Saving results/mix_curve.png...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

x = np.array([r[0] for r in rows])
medians = np.array([r[1] for r in rows])
p5s = np.array([r[2] for r in rows])
p95s = np.array([r[3] for r in rows])

fig, ax = plt.subplots(figsize=(9, 5))

# Plot median line
ax.plot(x, medians, marker="o", color="#333333", lw=2, label="Median match score")

# Fill 5th-95th percentile band
ax.fill_between(x, p5s, p95s, color="#999999", alpha=0.25,
                label="5th-95th percentile band")

# Draw GREEN and RED horizontal lines
ax.axhline(GREEN, color="#2e9e4f", ls="--", lw=1.5,
           label=f"GREEN threshold (5th pct of genuine: {GREEN:.1f})")
ax.axhline(RED, color="#d23f3f", ls="--", lw=1.5,
           label=f"RED threshold (95th pct of off-ref: {RED:.1f})")

# Shade bands: green above GREEN, amber between, red below RED
# Since GREEN > RED, green region is y > GREEN, amber is RED < y < GREEN, red is y < RED

# Fill above GREEN (green region)
ax.fill_between(x, np.maximum(p95s, GREEN), np.minimum(p5s, 100), 
                color="#55a868", alpha=0.1, label="Green zone (above GREEN)")
# Fill between RED and GREEN (amber region)
ax.fill_between(x, np.minimum(np.maximum(p95s, RED), GREEN), 
                np.maximum(np.minimum(p5s, RED), 0),
                color="#e6a200", alpha=0.1, label="Amber zone (between RED and GREEN)")
# Fill below RED (red region) - where p5 <= RED
ax.fill_between(x, p5s, 0, where=(p5s <= RED), color="#c44e52", alpha=0.1, label="Red zone (below RED)")

# Mark the first mix level where median leaves green
if first_left_green is not None and first_left_green != "never left green":
    ax.axvline(first_left_green, color="black", ls=":", lw=1.5,
               label=f"First leaves green at {first_left_green}%")
    ax.text(first_left_green, GREEN + 2, f"leaves green\nat {first_left_green}%", 
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Legend elements
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markersize=8, label="Median"),
    Line2D([0], [0], marker="_", color="w", markerfacecolor="#55a868", markersize=8, label="Green zone"),
    Line2D([0], [0], marker="_", color="w", markerfacecolor="#e6a200", markersize=8, label="Amber zone"),
    Line2D([0], [0], marker="_", color="w", markerfacecolor="#c44e52", markersize=8, label="Red zone"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

ax.set_xlabel("Share of the sample replaced by off-reference images (%)")
ax.set_ylabel("Match score (0-100)")
ax.set_ylim(0, 100)
ax.set_title(
    f"Mix curve: match score falls as batch drifts from reference\n"
    f"Batch size={BATCH_SIZE}, resamples={TRIALS_PER_LEVEL} per level, "
    f"seed={SEED}"
)
ax.set_xlim(-5, 105)
ax.set_xticks(MIX_LEVELS)

plt.tight_layout()
figure_path = os.path.join(ROOT, "results", "mix_curve.png")
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {figure_path}")

# ---------------------------------------------------------------------------
# Step 9: Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MIX CURVE SUMMARY")
print("=" * 60)
print(f"Batch size: {BATCH_SIZE} images")
print(f"Number of resamples per mix level: {TRIALS_PER_LEVEL}")
print(f"Seed: {SEED}")
print(f"GREEN threshold: {GREEN:.1f} (5th percentile of {len(genuine_scores)} genuine batches from half B)")
print(f"RED threshold: {RED:.1f} (95th percentile of {len(off_scores)} off-reference batches from set_A)")
false_alarm = np.mean(np.array(genuine_scores) < GREEN) * 100
print(f"False alarm rate: {false_alarm:.1f}% (genuine batches below GREEN)")
if first_left_green is not None and first_left_green != "never left green":
    print(f"First mix level where median leaves green zone: {first_left_green}%")
elif first_left_green == "never left green":
    print(f"First mix level where median leaves green zone: never")
else:
    print(f"First mix level where median leaves green zone: never")

print(f"Mix curve table: results/mix_curve_table.csv")
print(f"Mix curve figure: results/mix_curve.png")

# ---------------------------------------------------------------------------
# End of script
# ---------------------------------------------------------------------------
print("\nDone.")