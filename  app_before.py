"""PowderPrint - screens a powder sample against a reference batch.

Run with:  python3 -m streamlit run app.py

All scoring and verdicts come from fingerprint.py, exactly as in the evaluation behind the results
slide (reference.json, 70 real images; thresholds from reference-only draws). Nothing is re-implemented
or hard-coded here.
"""
import glob
import hashlib
import json
import math
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from inference import segment
from fingerprint import (
    fingerprint_image,
    aggregate_sample,
    compare,
    build_reference,
    compute_natural_variation,
    _load_cached,
    _save_cache,
    _load_image_arr,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- the only settings you may need to change ------------------------------------------------
REAL_DIR = os.path.join(ROOT, "images", "overlap_images")                 # REAL microscope images
SET_A_DIR = os.path.join(ROOT, "synthetic_dense", "images", "set_A")      # synthetic "clearly off" demo
REF_PATH = os.path.join(ROOT, "reference.json")                           # built by fingerprint.py
BATCH = 10          # the score is calibrated for batches of 10 images
ZOOM = 1.3          # zoom used for the "mildly off" demo batch
DEMO_SEED = 42      # change to draw different demo batches
# -----------------------------------------------------------------------------------------------

APP_CACHE = os.path.join(ROOT, "app_cache")
UPLOAD_DIR = os.path.join(APP_CACHE, "uploads")
ZOOM_DIR = os.path.join(APP_CACHE, "zoomed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ZOOM_DIR, exist_ok=True)

COLOURS = {"green": "#2e9e4f", "amber": "#f2a31b", "red": "#d23f3f"}
TITLES = {
    "green": "Consistent with reference",
    "amber": "Inconclusive, send to lab",
    "red": "Deviates from reference",
}
FEATURES = ["particles_per_image", "median_area", "area_iqr", "mean_circularity", "clumping_score"]
NICE = {
    "particles_per_image": "Particles per image",
    "median_area": "Median particle area (px²)",
    "area_iqr": "Spread of particle sizes",
    "mean_circularity": "Roundness",
    "clumping_score": "Clumping",
}
UP_DOWN = {
    "particles_per_image": ("more particles per image", "fewer particles per image"),
    "median_area": ("larger particles", "smaller particles"),
    "area_iqr": ("a wider spread of particle sizes", "a narrower spread of particle sizes"),
    "mean_circularity": ("rounder particles", "less round particles"),
    "clumping_score": ("more clumping", "less clumping"),
}

st.set_page_config(page_title="PowderPrint", layout="wide")
st.markdown(
    "<style>html, body, [class*='css'] {font-size: 18px;} "
    ".verdict {padding: 28px; border-radius: 16px; color: white; text-align: center;} "
    ".verdict h1 {margin: 0; font-size: 2.2rem; color: white;} "
    ".verdict p {margin: 6px 0 0 0; font-size: 1.4rem;}</style>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the reference batch (the first start can take a few minutes)...")
def load_reference_and_natvar():
    if os.path.exists(REF_PATH):
        with open(REF_PATH) as f:
            reference = json.load(f)
        stamp = int(os.path.getmtime(REF_PATH))
    else:
        reference = build_reference(REAL_DIR, n_ref=70, seed=42)
        stamp = int(os.path.getmtime(REF_PATH)) if os.path.exists(REF_PATH) else 0
    # the cache file name contains the reference's timestamp, so a rebuilt reference never reuses stale numbers
    nv_path = os.path.join(APP_CACHE, f"nat_var_{stamp}.json")
    nat_var = None
    if os.path.exists(nv_path):
        try:
            with open(nv_path) as f:
                nat_var = json.load(f)
        except Exception:
            nat_var = None
    if nat_var is None:
        nat_var = compute_natural_variation(reference, n_samples=200, sample_size=BATCH)
        try:
            with open(nv_path, "w") as f:
                json.dump(nat_var, f)
        except Exception:
            pass
    return reference, nat_var


def thresholds(nat_var):
    """Same rule as compare(): green above the 5th percentile of reference-only scores, red below half of it."""
    s = sorted(nat_var.get("scores", []))
    if not s:
        return None, None
    p5 = s[max(0, min(len(s) - 1, int(math.ceil(0.05 * len(s))) - 1))]
    return p5, p5 / 2.0


def _zoom_image(src, dst, scale):
    img = _load_image_arr(src).astype(np.float32)
    h, w = img.shape[:2]
    nh, nw = int(round(h * scale)), int(round(w * scale))
    big = resize(img, (nh, nw), preserve_range=True, anti_aliasing=False).astype(np.float32)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    Image.fromarray(np.ascontiguousarray(big[y0:y0 + h, x0:x0 + w])).save(dst)


@st.cache_resource(show_spinner=False)
def demo_samples(_reference):
    """Fixed 10-image demo batches built from images the reference never saw."""
    ref_names = set(_reference["per_image"].keys())
    held_out = [p for p in sorted(glob.glob(os.path.join(REAL_DIR, "*.tif")))
                if os.path.basename(p) not in ref_names]
    out = {}
    if len(held_out) >= BATCH:
        genuine = random.Random(DEMO_SEED).sample(held_out, BATCH)
        out["Genuine (unseen real images)"] = genuine
        try:
            zoomed = []
            for p in genuine:
                dst = os.path.join(ZOOM_DIR, f"zoom{int(ZOOM * 10)}_" + os.path.basename(p))
                if not os.path.exists(dst):
                    _zoom_image(p, dst, ZOOM)
                zoomed.append(dst)
            out[f"Mildly off (same images, zoomed {ZOOM}x)"] = zoomed
        except Exception:
            pass
    set_a = sorted(glob.glob(os.path.join(SET_A_DIR, "*.tif")))
    if len(set_a) >= BATCH:
        out["Clearly off (synthetic dense powder)"] = random.Random(DEMO_SEED).sample(set_a, BATCH)
    return out


def save_uploads(files):
    paths = []
    for f in files:
        data = f.getvalue()
        ext = os.path.splitext(f.name)[1].lower()
        path = os.path.join(UPLOAD_DIR, hashlib.md5(data).hexdigest()[:16] + ext)
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(data)
        paths.append(path)
    return paths


def features_for(path):
    cached = _load_cached(path)
    if cached is not None:
        return cached
    feat = fingerprint_image(path)
    _save_cache(path, feat)
    return feat


@st.cache_data(show_spinner=False, persist="disk")
def get_segmentation(path):
    labels, particles = segment(path)
    return labels, [p["area"] for p in particles if p["area"] > 20]


@st.cache_data(show_spinner=False, persist="disk")
def reference_areas(names, real_dir):
    out = {}
    for n in names:
        p = os.path.join(real_dir, n)
        if os.path.exists(p):
            out[n] = get_segmentation(p)[1]
    return out


def make_overlay(path, labels):
    img = _load_image_arr(path)
    lo, hi = float(np.min(img)), float(np.max(img))
    disp = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.stack([disp] * 3, axis=-1)
    rgb[find_boundaries(labels, mode="outer")] = [1.0, 0.0, 0.0]
    return (disp * 255).astype(np.uint8), (rgb * 255).astype(np.uint8)


def size_panel(sample_lists, ref_lists):
    def diam(a):
        return 2.0 * np.sqrt(np.asarray(a, float) / np.pi)

    s_d = [diam(a) for a in sample_lists if len(a)]
    r_d = [diam(a) for a in ref_lists if len(a)]
    s_all, r_all = np.concatenate(s_d), np.concatenate(r_d)
    lo, hi = np.percentile(np.concatenate([s_all, r_all]), [1, 99])
    edges = np.linspace(lo, hi, 21)
    centres = (edges[:-1] + edges[1:]) / 2

    def stack(ds):
        return np.array([np.histogram(np.clip(d, lo, hi), bins=edges)[0] / max(len(d), 1) * 100 for d in ds])

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for H, colour, label in ((stack(r_d), "#4c78a8", "Reference"), (stack(s_d), "#e45756", "Your sample")):
        a, b = np.percentile(H, [10, 90], axis=0)
        ax.fill_between(centres, a, b, color=colour, alpha=0.2, step="mid")
        ax.step(centres, H.mean(axis=0), where="mid", color=colour, linewidth=2, label=label)
    ax.set_xlabel("Particle diameter (px, equivalent circle)")
    ax.set_ylabel("% of particles in each size bin")
    ax.set_title("Mean per image, shaded = 10th to 90th percentile across images")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    s_med, r_med = float(np.median(s_all)), float(np.median(r_all))
    d = (s_med - r_med) / r_med * 100 if r_med > 0 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Your sample", f"{len(s_d)} images", f"{len(s_all)} particles", delta_color="off")
    c2.metric("Reference (shown)", f"{len(r_d)} images", f"{len(r_all)} particles", delta_color="off")
    c3.metric("Median diameter", f"{s_med:.1f} px", f"{d:+.1f}% vs reference ({r_med:.1f} px)", delta_color="off")


# ----------------------------------------------------------------- page
st.title("PowderPrint")
st.write("Upload microscope images of a powder sample. The app checks whether the sample "
         "is consistent with a trusted reference batch, and says why if it is not.")
st.caption("Prototype: screens samples against a reference batch. It does not replace lab testing.")

reference, nat_var = load_reference_and_natvar()
demos = demo_samples(reference)
green_t, red_t = thresholds(nat_var)

n_ref = len(reference["per_image"])
st.caption(f"Reference batch: {n_ref} real microscope images. "
           + (f"Green above {green_t:.1f}, red below {red_t:.1f} (set from reference-only samples)."
              if green_t is not None else ""))

left, right = st.columns([1, 1.2])
with left:
    st.subheader("1. Choose a sample")
    uploads = st.file_uploader(f"Upload {BATCH} microscope images",
                               type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)
    demo_name = None
    if demos:
        demo_name = st.selectbox("...or pick a demo sample (used when nothing is uploaded)", list(demos.keys()))
    else:
        st.info("No demo samples found. Check the image folders at the top of app.py.")

if uploads:
    sample_paths, sample_name = save_uploads(uploads), "Your upload"
elif demo_name:
    sample_paths, sample_name = list(demos[demo_name]), demo_name
else:
    sample_paths, sample_name = [], ""

if not sample_paths:
    with right:
        st.info("Upload images or pick a demo sample to start.")
    st.stop()

if len(sample_paths) < BATCH:
    with right:
        st.error(f"The score is calibrated for {BATCH}-image batches. Please upload {BATCH} images "
                 f"(you gave {len(sample_paths)}).")
    st.stop()
if len(sample_paths) > BATCH:
    with left:
        st.info(f"Using the first {BATCH} of {len(sample_paths)} images.")
    sample_paths = sample_paths[:BATCH]

t0 = time.time()
feats, good_paths = [], []
with st.spinner("Finding and measuring particles..."):
    for p in sample_paths:
        try:
            feats.append(features_for(p))
            good_paths.append(p)
        except Exception as e:
            st.warning(f"Skipped {os.path.basename(p)}: could not be read ({type(e).__name__}).")

if len(feats) < BATCH:
    with right:
        st.error(f"Only {len(feats)} of {BATCH} images could be analysed. "
                 "Please upload readable microscope images (.tif, .png, .jpg).")
    st.stop()

sample = aggregate_sample(feats)
if sample["particles_per_image"] < 5:
    with right:
        st.error("Too few particles found to judge this sample. Please upload a denser sample.")
    st.stop()

result = compare(sample, reference, nat_var)
verdict = result["verdict"]
score = float(result["match_score"])
ref_sample = reference["ref_sample"]
elapsed = time.time() - t0

rows, pct = [], {}
for f in FEATURES:
    s, r = sample.get(f, float("nan")), ref_sample.get(f, float("nan"))
    d = (s - r) / r * 100 if r not in (0, None) and not (np.isnan(s) or np.isnan(r)) else float("nan")
    pct[f] = d
    rows.append({"Feature": NICE[f], "This sample": round(s, 3), "Reference": round(r, 3),
                 "Difference (%)": None if np.isnan(d) else round(d, 1)})
table = pd.DataFrame(rows)

top = [f for f in result["top2_deviating"] if f in pct and not np.isnan(pct[f])][:2]
parts = [f"{UP_DOWN[f][0] if pct[f] > 0 else UP_DOWN[f][1]} ({pct[f]:+.0f}%)" for f in top]
if verdict == "green":
    why = "Consistent with the reference on size, shape and clumping. This does not prove the sample is genuine."
    if parts:
        why += f" Largest difference: {parts[0]}."
else:
    why = ("Compared with the reference, the sample has " + " and ".join(parts) + "."
           if parts else "The sample differs from the reference batch.")
    why += " Worth re-testing." if verdict == "amber" else " Send for laboratory testing."
why += " Screening tool, not proof of counterfeit medicine."

with right:
    st.subheader("2. Result")
    st.markdown(f"<div class='verdict' style='background:{COLOURS[verdict]}'>"
                f"<h1>{TITLES[verdict]}</h1><p>Match score: {score:.0f} / 100</p></div>",
                unsafe_allow_html=True)
    st.markdown(f"**Why:** {why}")
    st.caption(f"Sample: {sample_name} ({len(feats)} images). Analysed in {elapsed:.1f} s.")

st.subheader("3. Fingerprint compared with the reference")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(table, hide_index=True, width="stretch")
with c2:
    st.bar_chart(table.dropna(subset=["Difference (%)"]).set_index("Feature")["Difference (%)"])
    st.caption("Difference from the reference batch, in percent.")

st.subheader("4. Particle size distribution")
try:
    ref_names = tuple(sorted(reference["per_image"].keys())[:20])
    with st.spinner("Preparing reference sizes (first time only, up to a few minutes)..."):
        ref_areas = reference_areas(ref_names, REAL_DIR)
    sample_areas = [get_segmentation(p)[1] for p in good_paths]
    if ref_areas and any(len(a) for a in sample_areas):
        size_panel(sample_areas, list(ref_areas.values()))
        st.caption("Reference curve uses 20 of the reference images.")
    else:
        st.info("Not enough particle data to draw the size distribution.")
except Exception as e:
    st.warning(f"Could not draw the size panel ({type(e).__name__}).")

st.subheader("5. What the model saw")
pick = st.selectbox("Image", good_paths, format_func=os.path.basename)
try:
    with st.spinner("Outlining particles..."):
        labels, areas = get_segmentation(pick)
        original, overlay = make_overlay(pick, labels)
    a, b = st.columns(2)
    a.image(original, caption="Original", width="stretch")
    b.image(overlay, caption=f"Particles found: {len(areas)}", width="stretch")
except Exception as e:
    st.warning(f"Could not draw this image ({type(e).__name__}).")

st.caption("Screening tool, not a replacement for lab testing.")
