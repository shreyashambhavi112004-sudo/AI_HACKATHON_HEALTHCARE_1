# AI_HACKATHON_HEALTHCARE_1# PowderPrint

**Screens a pharmaceutical powder sample against a trusted reference batch, using microscope images.**

Built for the Leeds AI Society *AI in Healthcare* hackathon (24-25 Oct 2026).

> **Prototype.** PowderPrint is a screening aid. It measures how much a powder sample differs from a reference batch. It does **not** certify a medicine, prove that a sample is genuine or counterfeit, or replace laboratory testing.

## What it does

1. You give it 10 microscope images of a powder.
2. A StarDist segmentation model finds and outlines every particle.
3. The batch is summarised as a five-number **fingerprint**: particles per image, median particle area, spread of particle sizes, roundness, and clumping (the share of particle area in particles that touch another particle).
4. The fingerprint is compared with a reference batch and the app gives a verdict with a reason, for example "smaller particles (-41%)":
   - **Green**: consistent with the reference
   - **Amber**: inconclusive, send to lab
   - **Red**: deviates from the reference

### Two modes (switch at the top of the app)

| Mode | What you do | Output | Status |
|---|---|---|---|
| **Built-in reference batch** | Upload (or pick) 10 images | Match score 0-100 and green / amber / red, with a reason | Validated (see Results) |
| **My own reference batch** | Upload a trusted batch (box A) and a sample batch (box B), at least 5 images each | Percent difference per feature against a tolerance you set. No pass/fail verdict: that decision is yours | Prototype, **not validated** |

## Quick start

Requires Python and the project's model and data files (see below). Developed on macOS with Python 3.14 (Homebrew).

```bash
cd AI_HACKATHON_HEALTHCARE

# 1. (recommended) create a clean environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install packages
python3 -m pip install -r requirements.txt

# 3. run the app
python3 -m streamlit run app.py
```

The app opens at http://localhost:8501.

- Always start it with `python3 -m streamlit run app.py` so it uses the same Python that has the model libraries.
- The **first start takes a few minutes**: it measures the reference images and computes how much real batches normally vary. Later starts are fast, because results are cached in `fingerprint_cache/` and `app_cache/`.
- If installing TensorFlow or StarDist fails, your Python version may not be supported by TensorFlow. Try a Python version TensorFlow supports (3.10 to 3.12 is usually the safest choice).
- On macOS with Homebrew Python you may see `externally-managed-environment`. Using the virtual environment above avoids it.

### Pin exact versions (recommended before sharing)

`requirements.txt` is unpinned. Once the app runs on your machine, record exactly what works:

```bash
python3 -m pip freeze | grep -i -E "^(numpy|scipy|pandas|pillow|matplotlib|scikit-image|tifffile|tensorflow|keras|csbdeep|stardist|streamlit)==" > requirements-lock.txt
```

## Project structure

```
app.py                  Streamlit app (both modes)
inference.py            segment(path): StarDist inference on one image
fingerprint.py          per-image features, batch fingerprint, comparison and verdicts, evaluation
reference.json          fingerprint of the 70 reference images
plot_from_csv.py        draws the results chart from fingerprint_results.csv
mix_curve.py            optional: score as a sample is gradually replaced by off-reference images
models/                 trained StarDist model (config.json, weights_best.h5, thresholds.json)
images/overlap_images/  100 real microscope images of powder (70 reference + 30 held out)
images/single_particles/ single-particle photos used by the synthetic generator
dataset/images/         500 synthetic powder images (masks in dataset/masks)
synthetic_dense/        synthetic sets A, B, C used as clearly different test batches
*.ipynb                 StarDist training/evaluation and synthetic-data generator notebooks
fingerprint_cache/      cached per-image measurements (safe to delete, but re-measuring takes a while)
app_cache/              cached app data (safe to delete)
```

**Watch the folder names.** Real images are in `images/overlap_images`. The folder `dataset/images` holds **synthetic** images. The file names overlap (`image_000.tif` exists in both), so mixing them up does not cause an error but gives wrong results.

## How it works

- **Training data.** Labelled real microscopy images are scarce, so the model is trained on synthetic powder images made by compositing real single-particle photos onto a background. This gives exact particle masks for free.
- **Fingerprint.** Particles under 20 px are ignored. A batch fingerprint is the average of the per-image features over a group of 10 images, because single images contain too few particles.
- **Match score.** With feature values $x_i$, reference values $r_i$ and fixed weights $w_i$, the relative deviation is $d_i = |x_i - r_i| / r_i$ and the score is

  $$S = \frac{100}{1 + 10\sum_i w_i d_i}$$

- **Verdict thresholds** come from the reference images only: green above the 5th percentile of scores from 200 random 10-image batches of reference images, red below half of that value, amber in between. Nothing was tuned on the test sets.
- **Reference.** 70 real images (seed 42) form `reference.json`. The other 30 real images were held out and never used to build it.

## Results (built-in mode)

Evaluated on 50 random groups of 10 images per set.

| Test set | Mean score | Verdicts |
|---|---|---|
| 30 unseen real images | 64.2 | 98% green, 2% amber |
| Synthetic set A | 17.2 | 100% red |
| Synthetic set B | 17.6 | 100% red |
| Synthetic set C | 9.8 | 100% red |
| Original synthetic set | 12.5 | 100% red |

**What this shows:** the tool recognises unseen real powder and rejects clearly different batches.
**What it does not show:** that it detects subtle drift, or real counterfeit or substandard medicine. The synthetic sets differ from real powder in particle size, density and darkness, and no real counterfeit samples were available.

To reproduce the evaluation (this rewrites `reference.json` and `fingerprint_results.csv`; keep a backup first):

```bash
python3 fingerprint.py
python3 plot_from_csv.py     # writes proof_plot_final.png
```

## Limitations

- Tested on one powder and one microscope setup.
- The synthetic generator does not match real particle sizes and darkness closely. The synthetic sets are used as controlled, clearly different test batches, not as copies of real powder.
- The score is calibrated for batches of 10 images.
- The two-batch mode has no validation and no pass/fail threshold.
- Images alone cannot detect every counterfeit: two powders can look alike under a microscope and differ chemically.

## Next steps

- Validation on real batches and real substandard samples, with a pharmaceutical partner.
- A richer fingerprint: D10/D50/D90 size percentiles, aspect ratio, particle opacity.
- Sensitivity testing on gradual, subtle drift.
- Validation of the two-batch mode.

## Tech

Python, StarDist, TensorFlow, scikit-image, NumPy, pandas, Matplotlib, Streamlit.
