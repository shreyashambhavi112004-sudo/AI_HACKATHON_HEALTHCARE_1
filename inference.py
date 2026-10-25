"""
Inference script for the v3 StarDist particle model.

Loads the model once and provides a `segment(image_path)` function that returns
the particle instance mask plus a list of particles with area and circularity.
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import fill_label_holes

# ---------------------------------------------------------------------------
# Paths – mirror the v3 notebook's config so the model loads from the right place
# ---------------------------------------------------------------------------
MODEL_NAME = "stardist_particle_model"
MODEL_BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

# The v3 config (from config.json / notebook cell 7):
#   n_rays=64, grid=(1,1), unet_n_depth=4, n_channel_in=1, axes='YXC'
# StarDist2D re-reads config.json + weights_best.h5 from <basedir>/<name>/ automatically.

# Directory layout (the 3 datasets for this project):
#   Single particles : /images/single_particles/single/*.bmp  (individual building-block particles)
#   Real powder      : /dataset/images/*.tif                  (500 real microscope powder frames)
#   Synthetic powder  : /images/overlap_images/*.tif          (100 synthetic powder images)
_model = None


def _load_model():
    """Lazily load the StarDist2D model (singleton)."""
    global _model
    if _model is None:
        _model = StarDist2D(
            None,  # config is read from disk via name/basedir
            name=MODEL_NAME,
            basedir=MODEL_BASEDIR,
        )
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(image_path):
    """Load an image and normalize like the v3 notebook (load_real_image): 0..255 -> 0..1, then normalize(1, 99.8).

    Robust to both uint8 images (single-particle bmps) and 0..1 float tifs
    (dataset powder images and overlap_images)."""
    arr = np.array(Image.open(image_path))  # keep original mode/depth
    if arr.dtype.kind == "f":
        img = arr.astype(np.float32, copy=False)
    else:
        img = (arr.astype(np.float32, copy=False) / 255.0)
    if img.ndim == 3:
        img = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
    return normalize(img.astype(np.float32, copy=False), 1, 99.8)


def _refine_labels(img_gray01, labels, grow=4, thresh_offset=0.0):
    """Tighten StarDist label boundaries to the real particle pixels.

    Mirrors v3 notebook cell ``refine_labels`` (``Section 8``): particles are
    dark on a light background, so we expand each label a few px, thresh with
    Otsu, and fill holes back in."""
    from skimage.filters import threshold_otsu
    from skimage.segmentation import expand_labels
    from scipy.ndimage import binary_fill_holes

    thr = threshold_otsu(img_gray01) + thresh_offset
    fg = img_gray01 < thr
    refined = np.where(fg, expand_labels(labels, distance=grow), 0)
    out = np.zeros_like(refined)
    for lbl in np.unique(refined):
        if lbl == 0:
            continue
        out[binary_fill_holes(refined == lbl)] = lbl
    return out


def _refine_path(image_path, labels):
    """Mirror the v3 notebook's refine_path: re-load raw pixels and refine."""
    arr = np.array(Image.open(image_path))
    if arr.dtype.kind == "f":
        gray01 = arr.astype(np.float32, copy=False)
    else:
        gray01 = (arr.astype(np.float32, copy=False) / 255.0)
    if gray01.ndim == 3:
        gray01 = (0.299*gray01[...,0] + 0.587*gray01[...,1] + 0.114*gray01[...,2])
    return _refine_labels(gray01.astype(np.float32, copy=False), labels)
    thr = threshold_otsu(img_gray01) + thresh_offset
    fg = img_gray01 < thr
    refined = np.where(fg, expand_labels(labels, distance=grow), 0)
    out = np.zeros_like(refined)
    for lbl in np.unique(refined):
        if lbl == 0:
            continue
        out[binary_fill_holes(refined == lbl)] = lbl
    return out


def _particle_properties(labels):
    """Return a list of dicts with area (px) and circularity for each instance.

    Circularity = 4 * pi * area / perimeter^2  (1.0 = perfect circle).
    Label 0 (background) is skipped.
    """
    from skimage.measure import label as sk_label, regionprops

    props = regionprops(labels)
    particles = []
    for p in props:
        area = p.area
        perimeter = p.perimeter
        circ = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        particles.append(
            {"label": p.label, "area": float(area), "circularity": float(circ)}
        )
    return particles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(image_path, refine=True):
    """Run StarDist v3 instance segmentation on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image (PNG, BMP, JPG, TIFF – anything PIL handles).
    refine : bool
        Whether to tighten label boundaries with Otsu thresholding (matches the
        v3 notebook's real-image pipeline). Default True.

    Returns
    -------
    labels : ndarray (int32)
        Instance segmentation mask. 0 = background, 1..N = particle IDs.
    particles : list of dict
        Each dict has keys ``label``, ``area`` (px), ``circularity`` (0..1).
    """
    model = _load_model()
    img = _load_image(image_path)

    # StarDist expects float32 images in [0, 1] (or close) with shape (H, W).
    # predict_instances returns (labels, details).
    labels, details = model.predict_instances(
        img, n_tiles=model._guess_n_tiles(img)
    )
    labels = fill_label_holes(labels)

    if refine:
        labels = _refine_path(image_path, labels)

    particles = _particle_properties(labels)
    return labels, particles


# ---------------------------------------------------------------------------
# CLI demo: segment 3 images and write overlay PNGs into /outputs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import glob as _glob
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    print(f"Using model from: {MODEL_DIR}/{MODEL_NAME}")

    import glob as _glob_module
    def _glob_imgs(root):
        exts = ("*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg", "*.jpeg")
        out = []
        for e in exts:
            out.extend(sorted(_glob_module.glob(os.path.join(root, e))))
        seen = set()
        res = []
        for p in out:
            if p not in seen:
                seen.add(p)
                res.append(p)
        return res

    def _has_real_content(path):
        try:
            im = _load_any_image(path)
            return float(im.max()) > 0.02
        except Exception as e:
            print("    _has_real_content err for", os.path.basename(path), type(e), e)
            return False

    def _load_any_image(path):
        """Load an image robust to float-0..1 tif (dataset powder) vs uint8-0..255 bmp (single particles).
        Returns a float32 0..255 image so downstream code can normalize consistently."""
        arr = np.array(Image.open(path))  # keep original mode/depth
        if arr.dtype.kind == "f":
            # 0..1 float tif (dataset powder or overlap images).  Model wants 0..1 so
            # pass through as-is rather than scaling to 0..255.
            arr = arr.astype(np.float32, copy=False)
        else:
            # 0..255 uint (single-particle bmp etc.) normalize to 0..1.
            arr = (arr.astype(np.float32, copy=False) / 255.0)
        if arr.ndim == 3:
            # RGB -> luminance (matches convert('L') behaviour but on raw pixels)
            arr = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
        return arr.astype(np.float32, copy=False)


    # Dataset inventory (mirrors the folder summary printed above).
    real_powder_dir = os.path.join(os.path.dirname(__file__), "dataset", "images")
    real_powder_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset", "images"))
    synthetic_powder_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "images", "overlap_images"))

    print(f"\nREAL powder dir  : {real_powder_dir}")
    print(f"SYNTH powder dir : {synthetic_powder_dir}")

    real_paths = _glob_imgs(real_powder_dir)
    synth_paths_all = _glob_imgs(synthetic_powder_dir)

    real_paths = [p for p in real_paths if _has_real_content(p)]
    synth_paths = [p for p in synth_paths_all if _has_real_content(p)]

    real_with_content = sum(1 for p in real_paths if _has_real_content(p))
    synth_with_content = sum(1 for p in synth_paths if _has_real_content(p))

    print(f"REAL powder images (all tif)         : {len(real_paths)}")
    print(f"REAL powder images with content        : {real_with_content}")
    print(f"SYNTH powder images (all tif)         : {len(synth_paths_all)}")
    print(f"SYNTH powder images with content       : {synth_with_content}")

    if len(real_paths) < 3:
        raise FileNotFoundError(
            f"Found only {len(real_paths)} content images in REAL powder dir."
        )
    if len(synth_paths) < 3:
        raise FileNotFoundError(
            f"Found only {len(synth_paths)} content images in SYNTH powder dir.\n"
            "The overlap_images/*.tif files look like 0..1 float tifs with content when read with PIL.\n"
            "If _has_real_content returns False it is likely a dtype/contrast threshold issue."
        )

    test_real = real_paths[:3]
    test_synth = synth_paths[:3]
    test_paths = test_real + test_synth

    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    print()
    print("===== REAL POWDER (dataset/images/*.tif) =====")

    for path in test_real:
        labels, particles = segment(path)

        n_particles = len(particles)

        basename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outputs_dir, f"{basename}_overlay.png")

        if n_particles == 0:
            print(
                f"{os.path.basename(path)}: NO PARTICLES FOUND | "
                f"(labels.max()={int(labels.max()) if labels.size else 0})"
            )
            print(f"  -> (no overlay saved for {os.path.basename(path)})")
            continue

        # Build an RGB overlay: normalized image + red label boundaries.
        img_norm = _load_image(path)
        overlay = np.stack([img_norm] * 3, axis=-1)
        from skimage.segmentation import find_boundaries
        overlay[find_boundaries(labels, mode="outer")] = [1, 0, 0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlay, cmap="gray" if overlay.shape[-1] == 1 else None)
        ax.set_title(f"{basename}\nparticles={n_particles}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        areas = [p["area"] for p in particles]
        circs = [p["circularity"] for p in particles]
        print(
            f"{os.path.basename(path)}: {n_particles} particles | "
            f"area median = {np.median(areas):.0f} px | "
            f"area range = {min(areas):.0f}-{max(areas):.0f} px | "
            f"circularity mean = {np.mean(circs):.3f}"
        )
        print(f"  -> {out_path}")

    print()
    print("===== SYNTHETIC POWDER (images/overlap_images/*.tif) =====")

    for path in test_synth:
        labels, particles = segment(path)

        n_particles = len(particles)

        basename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outputs_dir, f"{basename}_overlay.png")

        if n_particles == 0:
            print(
                f"{os.path.basename(path)}: NO PARTICLES FOUND | "
                f"(labels.max()={int(labels.max()) if labels.size else 0})"
            )
            print(f"  -> (no overlay saved for {os.path.basename(path)})")
            continue

        # Build an RGB overlay: normalized image + red label boundaries.
        img_norm = _load_image(path)
        overlay = np.stack([img_norm] * 3, axis=-1)
        from skimage.segmentation import find_boundaries
        overlay[find_boundaries(labels, mode="outer")] = [1, 0, 0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlay, cmap="gray" if overlay.shape[-1] == 1 else None)
        ax.set_title(f"{basename}\nparticles={n_particles}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        areas = [p["area"] for p in particles]
        circs = [p["circularity"] for p in particles]
        print(
            f"{os.path.basename(path)}: {n_particles} particles | "
            f"area median = {np.median(areas):.0f} px | "
            f"area range = {min(areas):.0f}-{max(areas):.0f} px | "
            f"circularity mean = {np.mean(circs):.3f}"
        )
        print(f"  -> {out_path}")

    print()
    print("Done. Open the PNGs in /outputs to inspect the overlays.")
