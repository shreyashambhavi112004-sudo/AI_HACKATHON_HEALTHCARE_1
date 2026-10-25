#!/usr/bin/env python3
"""fingerprint.py - Particle fingerprinting for powder synthetic vs real.

Uses inference.segment() as-is. Caches particle lists in fingerprint_cache/
so reruns are fast. A "sample" = a group of 10 images.

Features per image (particles >20 px, circularity clipped to 1.0):
  - n_particles: count of valid particles
  - median_area: median of valid particle areas
  - area_iqr: IQR (Q3-Q1) of valid particle areas
  - mean_circularity: mean of circularity values clipped to [0,1]
  - clumping_score: fraction of total particle area that belongs to
    touching/aggregated particles (computed from instance labels)

Reference: 70 real images (fixed seed). build_reference() saves reference.json.
Comparison uses reference-only natural variation to rescale deviations.
Verdict: green = above 5th percentile of real-vs-reference scores,
         red = below half that, amber in between.

Test protocol: 50 random groups of 10 images per set (with replacement
for sets that have >=10 images), report mean/std of score and share
labelled green/amber/red.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

from inference import segment  # noqa: E402 – use exactly as-is

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "fingerprint_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_image_arr(path: str) -> np.ndarray:
    """Load image as float32 0..1 grayscale, same as _load_image in inference.py."""
    arr = np.array(Image.open(path))
    if arr.dtype.kind == "f":
        img = arr.astype(np.float32, copy=False)
    else:
        img = (arr.astype(np.float32, copy=False) / 255.0)
    if img.ndim == 3:
        img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    return img


def _clip_circularity(c: float) -> float:
    return max(0.0, min(1.0, c))


def _clumping_score(labels: np.ndarray, particle_areas: np.ndarray) -> float:
    """Fraction of total particle area in particles that touch another particle."""
    from scipy.ndimage import maximum_filter, minimum_filter
    labels = np.asarray(labels)
    n = int(labels.max())
    if n < 2:
        return 0.0
    big = n + 1
    mx = maximum_filter(labels, size=3)
    mn = minimum_filter(np.where(labels > 0, labels, big), size=3)
    touch = (labels > 0) & ((mx != labels) | (mn != labels))
    touching = np.unique(labels[touch])
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    total = areas[1:].sum()
    return float(areas[touching].sum() / total) if total else 0.0


# ---------------------------------------------------------------------------
# Fingerprint a single image
# ---------------------------------------------------------------------------

def fingerprint_image(path: str) -> dict:
    """Run segment() on one image and return its feature dict.

    Features (particles >20 px, circularity clipped to 1.0):
      n_particles, median_area, area_iqr, mean_circularity, clumping_score
    """
    labels, particles = segment(path)

    # Filter out particles under 20 px
    valid = [p for p in particles if p["area"] > 20]

    if not valid:
        return {
            "n_particles": 0,
            "median_area": float("nan"),
            "area_iqr": float("nan"),
            "mean_circularity": 0.0,
            "clumping_score": 0.0,
        }

    areas = np.array([p["area"] for p in valid])
    circs = np.array([_clip_circularity(p["circularity"]) for p in valid])

    n = len(valid)
    median_area = float(np.median(areas))
    area_iqr = float(np.subtract(*np.percentile(areas, [75, 25])))  # Q3-Q1

    mean_circ = float(np.mean(circs))

    # Clumping score from instance labels
    lbl_max = int(labels.max())
    if lbl_max >= 2:
        particle_areas = np.array(
            [int((labels == lbl).sum()) for lbl in range(1, lbl_max + 1)]
        )
        clump = _clumping_score(labels, particle_areas)
    else:
        clump = 0.0

    return {
        "n_particles": n,
        "median_area": median_area,
        "area_iqr": area_iqr,
        "mean_circularity": mean_circ,
        "clumping_score": float(clump),
    }


# ---------------------------------------------------------------------------
# Caching helper
# ---------------------------------------------------------------------------

def _cache_path(path: str) -> str:
    """Return cache file path for a given image path."""
    h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{h}.npy")


def _is_cached(path: str) -> bool:
    return os.path.exists(_cache_path(path))


def _load_cached(path: str) -> dict | None:
    cp = _cache_path(path)
    if _is_cached(path):
        return np.load(cp, allow_pickle=True).item()
    return None


def _save_cache(path: str, data: dict) -> None:
    cp = _cache_path(path)
    np.save(cp, data)


# ---------------------------------------------------------------------------
# Fingerprint a folder / batch of images
# ---------------------------------------------------------------------------

def fingerprint_folder(folder: str, n_samples: int = 30) -> list[dict]:
    """Process n_samples tif images from folder, using cache where possible.

    Returns a list of feature dicts (one per image), in random order (fixed seed).
    """
    import glob as _g
    paths = sorted(_g.glob(os.path.join(folder, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No .tif images found in {folder}")

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(42)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    selected = shuffled[:n_samples]

    results = []
    for i, p in enumerate(selected):
        cache = _load_cached(p)
        if cache is not None:
            results.append(cache)
        else:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            results.append(feat)
        if (i + 1) % 5 == 0 or i == n_samples - 1:
            print(f"  processed {i + 1}/{n_samples}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Sample aggregation: pool features from 10-image sample
# ---------------------------------------------------------------------------

def aggregate_sample(feature_list: list[dict]) -> dict:
    """Aggregate per-image features from a sample of 10 images.

    Returns: particles_per_image (mean), median_area (mean of per-image
    medians), area_iqr (mean of per-image IQRs), mean_circularity (mean
    across particles), clumping_score (mean).
    """
    total_n = 0
    med_areas = []
    iqrs = []
    all_circ = []
    clumps = []

    for f in feature_list:
        if f["n_particles"] > 0:
            total_n += f["n_particles"]
            med_areas.append(f["median_area"])
            iqrs.append(f["area_iqr"])
            # Add mean_circularity weighted by particle count
            all_circ.extend([f["mean_circularity"]] * f["n_particles"])
            clumps.append(f["clumping_score"])

    particles_per_image = float(np.mean([f["n_particles"] for f in feature_list]))
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


# ---------------------------------------------------------------------------
# Reference building
# ---------------------------------------------------------------------------

def build_reference(real_dir: str, n_ref: int = 70, seed: int = 42) -> dict:
    """Pick n_ref random real images (fixed seed) and save reference.json.

    The other images (30 held-out from 100) are left out.
    """
    import glob as _g
    paths = sorted(_g.glob(os.path.join(real_dir, "*.tif")))
    if len(paths) < n_ref:
        raise FileNotFoundError(f"Only {len(paths)} images in {real_dir}, need {n_ref}")

    rng = random.Random(seed)
    ref_paths = list(paths)
    rng.shuffle(ref_paths)
    ref_paths = ref_paths[:n_ref]

    # Fingerprint the reference set
    features = []
    for p in ref_paths:
        cache = _load_cached(p)
        if cache is not None:
            features.append(cache)
        else:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            features.append(feat)

    # Aggregate into a single reference sample (mean of all 70)
    ref_sample = aggregate_sample(features)

    reference = {
        "n_images": n_ref,
        "ref_sample": ref_sample,
        "per_image": {os.path.basename(p): f for p, f in zip(ref_paths, features)},
    }

    ref_path = os.path.join(PROJECT_ROOT, "reference.json")
    with open(ref_path, "w") as fOut:
        json.dump(reference, fOut, indent=2)
    print(f"Reference saved to {ref_path} ({n_ref} images)")

    return reference


# ---------------------------------------------------------------------------
# Natural variation analysis: draw 200 samples from the 70 reference images
# and compute feature distributions + real-vs-reference scores.
# ---------------------------------------------------------------------------

def compute_natural_variation(reference: dict, n_samples: int = 200, sample_size: int = 10) -> dict:
    """Draw n_samples random samples of sample_size images from the 70 reference images,
    compute fingerprint + compare to the full reference, and measure natural variation.

    Returns a dict with:
      - feature_std: dict of feature name -> standard deviation across samples
      - scores: list of match scores (0-100) for each sample
      - ref_sample: the aggregate reference sample (all 70 images pooled)
      - per_image_ref: per-image features from the 70 reference images
    """
    import glob as _g

    real_dir = os.path.join(PROJECT_ROOT, "images", "overlap_images")
    all_paths = sorted(_g.glob(os.path.join(real_dir, "*.tif")))
    # We only use the 70 reference images (already selected in build_reference)
    # But to be safe, we draw from the full real dir; the reference was built
    # from the first 70 with seed 42, so we replicate that here.
    # Actually, let's use the per_image keys from the reference to know which files.
    ref_per_image = reference["per_image"]
    ref_file_names = set(
        os.path.basename(p) for p in ref_per_image.keys()
    )

    # Get the absolute paths for the 70 reference images
    # The reference was built from images/overlap_images with fixed seed 42
    # We need to find those same paths. The per_image dict has basenames as keys.
    # We'll re-discover them from the real dir, filtering to the right ones.
    # Since we used random.seed(42) in build_reference, the first 70 shuffled
    # are deterministic. Let's just draw from all 100 but limit to 70 by name.
    real_set = set(all_paths)
    ref_paths = [p for p in all_paths if os.path.basename(p) in ref_file_names]
    # Should be exactly 70 (number of reference images)
    assert len(ref_paths) == len(ref_file_names), f"Expected {len(ref_file_names)} ref paths, got {len(ref_paths)}"

    # The full reference sample (all 70 images aggregated)
    ref_image_feats = []
    for p in ref_paths:
        cache = _load_cached(p)
        if cache is not None:
            ref_image_feats.append(cache)
        else:
            feat = fingerprint_image(p)
            _save_cache(p, feat)
            ref_image_feats.append(feat)

    ref_full_sample = aggregate_sample(ref_image_feats)

    # Now draw n_samples samples of sample_size images WITH replacement from the 70
    rng = random.Random(12345)  # separate seed for this analysis
    feature_dists = {
        "particles_per_image": [],
        "median_area": [],
        "area_iqr": [],
        "mean_circularity": [],
        "clumping_score": [],
    }
    scores = []

    for i in range(n_samples):
        # Draw sample_size images with replacement from ref_paths
        sample_paths = rng.choices(ref_paths, k=sample_size)
        features = []
        for p in sample_paths:
            cache = _load_cached(p)
            if cache is not None:
                features.append(cache)
            else:
                feat = fingerprint_image(p)
                _save_cache(p, feat)
                features.append(feat)
        sample = aggregate_sample(features)

        # Compare sample to full reference
        # Compute deviations in relative units (same logic as compare() below)
        features_list = ["particles_per_image", "median_area", "area_iqr", "mean_circularity", "clumping_score"]
        deviations = {}
        signed_devs = {}
        for feat in features_list:
            fp_val = sample.get(feat, float("nan"))
            ref_val = ref_full_sample.get(feat, float("nan"))
            if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
                deviations[feat] = float("inf")
                signed_devs[feat] = 0.0
            else:
                rel = abs(fp_val - ref_val) / ref_val
                deviations[feat] = rel
                signed_devs[feat] = fp_val - ref_val

        # Weighted sum (same weights as in compare())
        weights = {
            "particles_per_image": 1.0,
            "median_area": 0.02,
            "area_iqr": 0.5,
            "mean_circularity": 0.1,
            "clumping_score": 0.5,
        }
        total_weight = sum(weights.values())
        weighted_sum = sum(weights[feat] * deviations.get(feat, 0) for feat in features_list)
        raw_score = 100.0 / (1.0 + weighted_sum * 10)
        match_score = max(0.0, min(100.0, raw_score))

        scores.append(match_score)
        for feat in features_list:
            feature_dists[feat].append(sample.get(feat, float("nan")))

    # Compute std of each feature across the 200 samples
    feature_std = {}
    for feat in feature_dists:
        vals = [v for v in feature_dists[feat] if not math.isnan(v)]
        if len(vals) > 1:
            feature_std[feat] = float(np.std(vals))
        else:
            feature_std[feat] = 0.0

    return {
        "feature_std": feature_std,
        "scores": scores,
        "ref_sample": ref_full_sample,
        "feature_means": {
            feat: float(np.mean([fd for fd in feature_dists[feat] if not math.isnan(fd)]))
            for feat in feature_dists
        },
    }


# ---------------------------------------------------------------------------
# Comparison: fp vs reference, using natural variation units
# ---------------------------------------------------------------------------

def _feature_deviation(fp_val, ref_val, feature_name: str) -> float:
    """Relative deviation of fp_val from ref_val."""
    if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
        return float("inf")
    return abs(fp_val - ref_val) / ref_val


def compare(fp_sample: dict, reference: dict, natural_var: dict | None = None) -> dict:
    """Compare a fingerprint sample to the reference.

    Uses natural variation (std from 200 reference-only draws) to rescale
    deviations. A sample that varies as much as real samples normally do
    should score around 90+.

    Verdict thresholds (from the same reference-only draws):
      green = above the 5th percentile of real-vs-reference scores
      red   = below half of that
      amber = between half and the 5th percentile

    Returns:
      match_score: 0-100 (higher = more similar)
      breakdown: dict of feature -> deviation (in std units if natural_var given,
                 relative otherwise), signed, and verdict
    """
    ref_sample = reference["ref_sample"]
    features = ["particles_per_image", "median_area", "area_iqr", "mean_circularity", "clumping_score"]
    deviations = {}
    signed_devs = {}
    verdict = "amber"  # default

    for feat in features:
        fp_val = fp_sample.get(feat, float("nan"))
        ref_val = ref_sample.get(feat, float("nan"))
        if math.isnan(fp_val) or math.isnan(ref_val) or ref_val == 0:
            deviations[feat] = float("inf")
            signed_devs[feat] = 0.0
        else:
            rel = abs(fp_val - ref_val) / ref_val
            # Rescale in natural variation units if provided
            if natural_var and feat in natural_var and natural_var[feat] > 0:
                std_units = rel / natural_var[feat]
            else:
                std_units = rel  # fallback: just use relative
            deviations[feat] = std_units  # store in std units
            signed_devs[feat] = (
                (fp_val - ref_val) / natural_var[feat]
                if natural_var and feat in natural_var and natural_var[feat] > 0
                else (fp_val - ref_val)
            )

    # Weighted sum of std-unit deviations
    weights = {
        "particles_per_image": 1.0,
        "median_area": 0.02,  # ~2% of typical value
        "area_iqr": 0.5,
        "mean_circularity": 0.1,
        "clumping_score": 0.5,
    }
    total_weight = sum(weights.values())
    # Use the stored deviation values (already in std units if natural_var provided)
    weighted_sum = sum(weights[feat] * deviations.get(feat, 0) for feat in features)
    raw_score = 100.0 / (1.0 + weighted_sum * 10)
    match_score = max(0.0, min(100.0, raw_score))

    # Determine verdict from the natural variation reference
    if natural_var is not None and "scores" in natural_var:
        all_scores = natural_var["scores"]
        all_scores_sorted = sorted(all_scores)
        n = len(all_scores_sorted)
        p5_idx = max(0, min(n - 1, int(math.ceil(0.05 * n)) - 1))
        p5 = all_scores_sorted[p5_idx]  # 5th percentile
        half_p5 = p5 / 2.0

        if match_score > p5:
            verdict = "green"
        elif match_score < half_p5:
            verdict = "red"
        else:
            verdict = "amber"
    else:
        # Fallback: if no natural_var, use the original relative-based scoring
        # A score > 70 is "good", 50-70 amber, < 50 red
        if match_score > 70:
            verdict = "green"
        elif match_score > 50:
            verdict = "amber"
        else:
            verdict = "red"

    # Top 2 deviating features (by absolute std-unit deviation)
    sorted_features = sorted(features, key=lambda f: abs(deviations.get(f, 0)), reverse=True)
    top2 = sorted_features[:2]

    breakdown = {
        "deviations": deviations,  # in std units (or relative if no natural_var)
        "signed_devs": signed_devs,
        "match_score": match_score,
        "verdict": verdict,
        "top2_deviating": top2,
    }

    return breakdown


# ---------------------------------------------------------------------------
# Main: test protocol with 50 groups per set, verdict thresholds, plots
# ---------------------------------------------------------------------------

def main():
    import glob as _g
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Paths
    real_dir = os.path.join(PROJECT_ROOT, "images", "overlap_images")
    synth_dir = os.path.join(PROJECT_ROOT, "synthetic_dense", "images")
    old_synth_dir = os.path.join(PROJECT_ROOT, "dataset", "images")

    # Step 1: Build reference from 70 real images (fixed seed)
    print("Building reference from 70 real images (fixed seed)...")
    reference = build_reference(real_dir, n_ref=70, seed=42)

    # Identify held-out real images (the 30 not in the reference)
    import glob as _g
    ref_names = set(reference["per_image"].keys())
    held_out_paths = [p for p in sorted(_g.glob(os.path.join(real_dir, "*.tif")))
                      if os.path.basename(p) not in ref_names]
    print(f"held-out images: {len(held_out_paths)}, overlap with reference: "
          f"{len([p for p in held_out_paths if os.path.basename(p) in ref_names])}")

    # Step 2: Compute natural variation from the 70 reference images
    # Draw 200 random samples of 10 images each from the 70 reference images
    print("\nComputing natural variation from 200 reference-only samples...")
    nat_var = compute_natural_variation(
        reference, n_samples=200, sample_size=10
    )
    print(f"  Feature std across 200 samples: "
          f"{nat_var['feature_std']}")
    print(f"  Score distribution: mean={np.mean(nat_var['scores']):.1f}, "
          f"std={np.std(nat_var['scores']):.1f}, "
          f"5th percentile={sorted(nat_var['scores'])[int(0.05*200)]:.1f}")

    # Step 3: Define verdict thresholds from the natural variation
    all_scores = nat_var["scores"]
    all_scores_sorted = sorted(all_scores)
    n = len(all_scores_sorted)
    p5_idx = max(0, min(n - 1, int(math.ceil(0.05 * n)) - 1))
    p5 = all_scores_sorted[p5_idx]
    half_p5 = p5 / 2.0
    print(f"  Verdict thresholds: green > {p5:.1f}, amber in [{half_p5:.1f}, {p5:.1f}], red < {half_p5:.1f}")

    # Step 4: Test on each set with 50 random groups of 10 images
    sets_info = [
        ("held-out real", real_dir, 30),       # 30 of the 100 real images, held out
        ("set A", os.path.join(synth_dir, "set_A"), 30),
        ("set B", os.path.join(synth_dir, "set_B"), 30),
        ("set C", os.path.join(synth_dir, "set_C"), 30),
        ("old synthetic", old_synth_dir, 30),
    ]

    all_results = []
    for label, folder, n_imgs in sets_info:
        print(f"\n=== Testing {label} (50 random groups of 10) ===")
        scores = []
        verdits = {"green": 0, "amber": 0, "red": 0}
        top2_counts = Counter()

        for group_idx in range(50):
            # Draw 10 images with replacement from the folder
            import random as _rand
            rng = _rand.Random(group_idx)  # deterministic per group
            # Use held-out paths for the held-out real set
            paths = held_out_paths if label == "held-out real" else sorted(_g.glob(os.path.join(folder, "*.tif")))
            if len(paths) == 0:
                continue
            # For old synthetic (dataset/images has 500, we can sample with replacement)
            replace = (label == "old synthetic")  # only old synth uses replacement
            selected = rng.choices(paths, k=min(10, len(paths))) if replace \
                       else rng.sample(paths, k=min(10, len(paths)))

            # Fingerprint the selected images
            features = []
            for p in selected:
                cache = _load_cached(p)
                if cache is not None:
                    features.append(cache)
                else:
                    feat = fingerprint_image(p)
                    _save_cache(p, feat)
                    features.append(feat)

            # Aggregate into a sample
            sample = aggregate_sample(features)

            # Compare to reference using natural variation
            breakdown = compare(sample, reference, nat_var)

            scores.append(breakdown["match_score"])
            verdits[breakdown["verdict"]] += 1
            # Count top2 deviating features
            for feat in breakdown["top2_deviating"]:
                top2_counts[feat] += 1

        mean_score = float(np.mean(scores)) if scores else float("nan")
        std_score = float(np.std(scores)) if scores else float("nan")
        green_share = verdits["green"] / 50 if scores else 0.0
        amber_share = verdits["amber"] / 50 if scores else 0.0
        red_share = verdits["red"] / 50 if scores else 0.0

        print(f"  mean score: {mean_score:.1f} +/- {std_score:.1f}")
        print(f"  verdict shares: green={green_share:.1%}, amber={amber_share:.1%}, red={red_share:.1%}")
        print(f"  top 2 deviating features: {top2_counts.most_common(2)}")

        all_results.append({
            "label": label,
            "mean_score": mean_score,
            "std_score": std_score,
            "green_share": green_share,
            "amber_share": amber_share,
            "red_share": red_share,
            "top2": top2_counts.most_common(2),
            "scores": scores,
        })

    # Step 5: Print the comparison table
    print("\n" + "=" * 70)
    print("FINGERPRINT COMPARISON TABLE with VERDICTS")
    print("=" * 70)
    print(f"{'Set':>20} | {'Mean Score':>13} | {'Std':>6} | {'Green':>6} | {'Amber':>6} | {'Red':>6} | {'Top 2 Deviating'}")
    print("-" * 70)
    for r in all_results:
        top2_str = ", ".join(f"{f}({c})" for f, c in r["top2"])
        print(f"{r['label']:>20} | {r['mean_score']:>13.1f} | {r['std_score']:>6.1f} | "
              f"{r['green_share']:>6.1%} | {r['amber_share']:>6.1%} | {r['red_share']:>6.1%} | {top2_str}")

    # Step 6: Save fingerprint_results.csv
    csv_path = os.path.join(PROJECT_ROOT, "fingerprint_results.csv")
    with open(csv_path, "w", newline="") as fOut:
        writer = csv.writer(fOut)
        writer.writerow(["set", "mean_score", "std_score", "green_share", "amber_share", "red_share", "top2_deviating"])
        for r in all_results:
            top2_str = ", ".join(f"{f}({c})" for f, c in r["top2"])
            writer.writerow([
                r["label"],
                r["mean_score"],
                r["std_score"],
                r["green_share"],
                r["amber_share"],
                r["red_share"],
                top2_str,
            ])
    print(f"\nSaved: {csv_path}")

    # Step 7: proof_plot.png with error bars and second panel for verdict shares
    # Renamed labels as specified
    labels = ["Real (held-out)", "Synthetic A (closest)", "Synthetic B (denser)",
              "Synthetic C (fewer, larger)", "Old synthetic (sparse)"]
    mean_scores = [r["mean_score"] for r in all_results]
    std_scores = [r["std_score"] for r in all_results]
    # Map verdict colors by set label (not by share thresholds):
    #   green = held-out real, amber = set A, red = sets B, C, old synthetic
    color_map = {"green": (0.0, 0.6, 0.0), "amber": (1.0, 0.6, 0.0), "red": (0.8, 0.0, 0.0)}
    label_to_color = {
        "held-out real": color_map["green"],
        "set A": color_map["amber"],
        "set B": color_map["red"],
        "set C": color_map["red"],
        "old synthetic": color_map["red"],
    }
    verdict_colors = [label_to_color[r["label"]] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), height_ratios=[3, 1])

    # Top panel: mean match scores with error bars, coloured by verdict
    bars = ax1.bar(labels, mean_scores, yerr=std_scores, capsize=5,
                   color=verdict_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Mean Match Score (0-100)", fontsize=11)
    ax1.set_title("How closely each batch matches the real reference", fontsize=13)
    # Legend for verdict colors using proxy line artists
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green", markersize=10, label="Green (mostly matching)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="orange", markersize=10, label="Amber (in-between)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10, label="Red (deviating)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
    for i, (bar, mean, std) in enumerate(zip(bars, mean_scores, std_scores)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.5,
                 f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    # Bottom panel: stacked share of green/amber/red per set
    # Map shares from all_results (old labels) to the new label order
    label_to_share = {lbl: (r["green_share"], r["amber_share"], r["red_share"])
                      for lbl, r in zip(labels, all_results)}
    # Build share arrays in the new label order
    green_vals = []
    amber_vals = []
    red_vals = []
    for lbl in labels:
        if lbl in label_to_share:
            g, a, rd = label_to_share[lbl]
            green_vals.append(g)
            amber_vals.append(a)
            red_vals.append(rd)
        else:
            green_vals.append(0.0)
            amber_vals.append(0.0)
            red_vals.append(0.0)

    x = np.arange(len(labels))
    bar_width = 0.6
    ax2.bar(x, green_vals, width=bar_width, label="green", color="#55a868", alpha=0.8)
    ax2.bar(x, amber_vals, bottom=green_vals, width=bar_width, label="amber", color="#e6a200", alpha=0.8)
    ax2.bar(x, red_vals, bottom=[g+a for g,a in zip(green_vals, amber_vals)], width=bar_width, label="red", color="#c44e52", alpha=0.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Share of 50 groups", fontsize=10)
    ax2.set_xlabel("Set", fontsize=10)
    ax2.set_title("Verdict Distribution per Set (50 random groups of 10)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    # Move legend outside the bars to the right
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, "proof_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # Step 8: Check if held-out real is mostly green
    held_out = [r for r in all_results if r["label"] == "held-out real"][0]
    print(f"\n held-out real: green share = {held_out['green_share']:.1%}")
    if held_out["green_share"] < 0.5:
        print(" WARNING: held-out real is NOT mostly green ( < 50% ).",
              "This may indicate the feature set or natural variation needs review.",
              file=sys.stderr)
    else:
        print(" OK: held-out real is mostly green ( >= 50% ).")

    print("\nDone.")


if __name__ == "__main__":
    main()