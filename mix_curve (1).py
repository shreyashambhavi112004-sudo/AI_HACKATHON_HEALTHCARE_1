"""Drift curve: how the match score falls as a sample is gradually replaced by off-reference images.
Uses the scoring already in fingerprint.py. Run:  python3 mix_curve.py
Writes mix_curve.png and mix_curve.csv
"""
import csv, glob, json, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint import (fingerprint_image, aggregate_sample, compare,
                         compute_natural_variation, _load_cached, _save_cache)

ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "images", "overlap_images")
OFF_DIR = os.path.join(ROOT, "synthetic_dense", "images", "set_A")   # change to set_B / set_C to try others
TRIALS = 30
SIZE = 10

with open(os.path.join(ROOT, "reference.json")) as f:
    reference = json.load(f)

nv_path = os.path.join(ROOT, "app_cache", "nat_var.json")
if os.path.exists(nv_path):
    with open(nv_path) as f:
        nat_var = json.load(f)
else:
    nat_var = compute_natural_variation(reference, n_samples=200, sample_size=10)

def feats(p):
    c = _load_cached(p)
    if c is None:
        c = fingerprint_image(p)
        _save_cache(p, c)
    return c

ref_names = set(reference["per_image"].keys())
real = [p for p in sorted(glob.glob(os.path.join(REAL_DIR, "*.tif"))) if os.path.basename(p) not in ref_names]
off = sorted(glob.glob(os.path.join(OFF_DIR, "*.tif")))
print(f"unseen real images: {len(real)}, off-reference images: {len(off)}")

rows = []
for k in range(0, SIZE + 1, 2):          # k off-reference images out of 10
    scores, verdicts = [], []
    for t in range(TRIALS):
        rng = random.Random(1000 * k + t)
        paths = rng.sample(real, SIZE - k) if SIZE - k <= len(real) else rng.choices(real, k=SIZE - k)
        paths += rng.sample(off, k)
        res = compare(aggregate_sample([feats(p) for p in paths]), reference, nat_var)
        scores.append(res["match_score"]); verdicts.append(res["verdict"])
    rows.append((k * 10, np.mean(scores), np.std(scores),
                 verdicts.count("green") / TRIALS, verdicts.count("amber") / TRIALS, verdicts.count("red") / TRIALS))
    print(f"{k*10:3d}% off-reference: score {rows[-1][1]:.1f} +/- {rows[-1][2]:.1f}  "
          f"green {rows[-1][3]:.0%} amber {rows[-1][4]:.0%} red {rows[-1][5]:.0%}")

with open(os.path.join(ROOT, "mix_curve.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["percent_off_reference", "mean_score", "std", "green", "amber", "red"]); w.writerows(rows)

x = [r[0] for r in rows]; m = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, m, marker="o", color="#333333", lw=2)
ax.fill_between(x, m - s, m + s, color="#999999", alpha=0.25)
scores = nat_var.get("scores", [])
if scores:
    p5 = sorted(scores)[max(0, int(np.ceil(0.05 * len(scores))) - 1)]
    ax.axhline(p5, color="#2e9e4f", ls="--"); ax.text(100, p5 + 1.5, "green above this", ha="right", color="#2e9e4f")
    ax.axhline(p5 / 2, color="#d23f3f", ls="--"); ax.text(100, p5 / 2 + 1.5, "red below this", ha="right", color="#d23f3f")
ax.set_xlabel("Share of the sample replaced by off-reference images (%)")
ax.set_ylabel("Match score (0-100)")
ax.set_ylim(0, 100)
ax.set_title("Match score falls as a batch drifts away from the reference")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "mix_curve.png"), dpi=200)
print("Saved mix_curve.png and mix_curve.csv")
