#!/usr/bin/env python3
"""
Digital Image Processing - Elementary (point-processing) methods
================================================================
1. Image Negative            4. Contrast Stretching
2. Gamma Correction          5. Histogram Equalization
3. Logarithmic Transform     6. Intensity Level Slicing
                             7. Bit-Plane Slicing

Every method is implemented from scratch with NumPy (OpenCV is used only for
reading/writing files and for the RGB <-> YCrCb colour conversion).
All methods accept BOTH grayscale (H x W) and RGB (H x W x 3) uint8 images.

Usage
-----
    python dip_elementary_methods.py                     # built-in sample images
    python dip_elementary_methods.py --images a.jpg b.png  # your own images
    python dip_elementary_methods.py --show              # also open the windows

Results (figures + processed images + run_log.txt) go to ./results/
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")           # no GUI needed when just saving figures
import matplotlib.pyplot as plt
import cv2
from skimage import data            # only used for the built-in sample images

L = 256                              # number of gray levels (8-bit images)
LOG_LINES = []                       # everything printed is also saved to run_log.txt


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)


def is_rgb(img):
    return img.ndim == 3 and img.shape[2] == 3


def to_uint8(x):
    """Round and clip any numeric array to the valid 0..255 range."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def to_gray(img):
    """RGB -> gray using ITU-R BT.601 luminance weights."""
    if not is_rgb(img):
        return img
    return to_uint8(0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def display_u8(img):
    """Linearly scale a float array (e.g. an FFT spectrum) to 0..255 for display."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    return to_uint8(255 * (img - img.min()) / (img.max() - img.min() + 1e-12))


def hist_counts(img):
    return np.bincount(img.ravel(), minlength=L)


def describe(name, img):
    kind = "RGB " if is_rgb(img) else "GRAY"
    log(f"  [{kind}] {name:<18} shape={img.shape}  dtype={img.dtype}  "
        f"min={img.min():.0f}  max={img.max():.0f}  mean={img.mean():.1f}")


# ----------------------------------------------------------------------------
# 1. IMAGE NEGATIVE          s = (L-1) - r
# ----------------------------------------------------------------------------
def negative(img):
    return (L - 1) - img            # uint8 arithmetic is safe: 255 - [0..255]


# ----------------------------------------------------------------------------
# 2. GAMMA (POWER-LAW) CORRECTION     s = c * r^gamma   (r normalised to 0..1)
# ----------------------------------------------------------------------------
def gamma_correction(img, gamma, c=1.0):
    r = np.arange(L) / (L - 1)                          # all 256 possible inputs
    lut = to_uint8((L - 1) * c * np.power(r, gamma))    # look-up table
    return lut[img]                                     # works for gray and RGB


# ----------------------------------------------------------------------------
# 3. LOGARITHMIC TRANSFORM     s = c * log(1 + r),   c = 255 / log(1 + r_max)
# ----------------------------------------------------------------------------
def log_transform(img):
    r = img.astype(np.float64)
    c = (L - 1) / np.log1p(r.max())
    return to_uint8(c * np.log1p(r))


# ----------------------------------------------------------------------------
# 4. CONTRAST STRETCHING
# ----------------------------------------------------------------------------
def stretch_minmax(img):
    """Full-range stretch: (r - rmin) / (rmax - rmin) * 255 (same for all channels)."""
    rmin, rmax = float(img.min()), float(img.max())
    return to_uint8((img.astype(np.float64) - rmin) / (rmax - rmin) * (L - 1))


def stretch_percentile(img, p_low=2, p_high=98):
    """Robust stretch: ignore the darkest/brightest outliers."""
    lo, hi = np.percentile(img, [p_low, p_high])
    return to_uint8((np.clip(img, lo, hi) - lo) / (hi - lo) * (L - 1))


def stretch_piecewise(img, r1, s1, r2, s2):
    """Piece-wise linear: (0,0) -> (r1,s1) -> (r2,s2) -> (255,255)."""
    lut = to_uint8(np.interp(np.arange(L), [0, r1, r2, L - 1], [0, s1, s2, L - 1]))
    return lut[img]


# ----------------------------------------------------------------------------
# 5. HISTOGRAM EQUALIZATION
# ----------------------------------------------------------------------------
def equalization_lut(channel):
    """s_k = round( (cdf(k) - cdf_min) / (N - cdf_min) * 255 )"""
    hist = hist_counts(channel)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    n = channel.size
    return to_uint8((cdf - cdf_min) / max(n - cdf_min, 1) * (L - 1))


def equalize_gray(img):
    return equalization_lut(img)[img]


def equalize_rgb_per_channel(img):
    """Naive version: equalize R, G and B separately (can shift the colours)."""
    return np.dstack([equalize_gray(img[..., c]) for c in range(3)])


def equalize_rgb_luminance(img):
    """Better: equalize only the luminance (Y) and keep the chroma untouched."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = equalize_gray(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def equalize(img):
    return equalize_rgb_luminance(img) if is_rgb(img) else equalize_gray(img)


# ----------------------------------------------------------------------------
# 6. INTENSITY LEVEL SLICING
# ----------------------------------------------------------------------------
def level_mask(img, lo, hi):
    g = to_gray(img)                                    # decide on luminance
    return (g >= lo) & (g <= hi)


def slice_highlight(img, lo, hi):
    """Approach A: pixels in [lo, hi] are highlighted, the rest is preserved."""
    out = img.copy()
    out[level_mask(img, lo, hi)] = (255, 40, 40) if is_rgb(img) else 255
    return out


def slice_binary(img, lo, hi):
    """Approach B: pixels in [lo, hi] -> white, everything else -> black."""
    out = np.zeros_like(img)
    out[level_mask(img, lo, hi)] = 255
    return out


# ----------------------------------------------------------------------------
# 7. BIT-PLANE SLICING
# ----------------------------------------------------------------------------
def bit_plane(img, k):
    """Plane k (0 = LSB ... 7 = MSB) as a black/white image (per channel for RGB)."""
    return (((img >> k) & 1) * 255).astype(np.uint8)


def bit_reconstruct(img, n_planes):
    """Rebuild the image from only the n most-significant planes."""
    mask = (0xFF << (8 - n_planes)) & 0xFF
    return (img & mask).astype(np.uint8)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def show(ax, img, title=""):
    img = display_u8(img)
    ax.imshow(img, cmap=None if is_rgb(img) else "gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_hist(ax, img, title="", cdf=False, band=None):
    img = display_u8(img)
    if is_rgb(img):
        for ch, col in enumerate("rgb"):
            ax.plot(hist_counts(img[..., ch]), color=col, lw=1, alpha=0.85)
    else:
        ax.bar(np.arange(L), hist_counts(img), width=1.0, color="0.35")
    if band:
        ax.axvspan(band[0], band[1], color="orange", alpha=0.35)
    if cdf:
        ax2 = ax.twinx()
        g = to_gray(img)
        c = np.cumsum(hist_counts(g)) / g.size
        ax2.plot(c, color="crimson", lw=1.5, ls="--")
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(labelsize=7, colors="crimson")
    ax.set_xlim(-1, 256)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("intensity", fontsize=7)
    ax.tick_params(labelsize=7)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    log(f"      saved figure -> {path}")
    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


def save_img(img, path):
    img = display_u8(img)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if is_rgb(img) else img)


# ----------------------------------------------------------------------------
# Sample images (works offline - they ship with scikit-image)
# ----------------------------------------------------------------------------
def build_samples():
    s = {
        "camera": data.camera(),            # grayscale
        "moon": data.moon(),                # grayscale, naturally dark / low contrast
        "coins": data.coins(),              # grayscale
        "astronaut": data.astronaut(),      # RGB
        "chelsea": data.chelsea(),          # RGB
        "coffee": data.coffee(),            # RGB
    }
    # Synthetic degraded versions so the effect of each method is easy to see
    s["lowcon_camera"] = to_uint8(0.3 * s["camera"] + 90)          # washed-out gray
    s["lowcon_chelsea"] = to_uint8(0.3 * s["chelsea"] + 90)        # washed-out RGB
    s["dark_camera"] = to_uint8(255 * (s["camera"] / 255.0) ** 3 * 0.6)       # dark gray
    s["dark_astronaut"] = to_uint8(255 * (s["astronaut"] / 255.0) ** 3 * 0.6)  # dark RGB
    # Fourier spectrum of 'camera': huge dynamic range -> the classic log-transform demo
    spec = np.abs(np.fft.fftshift(np.fft.fft2(s["camera"].astype(np.float64))))
    s["spectrum_camera"] = spec                                     # float, not uint8
    return s


def load_custom(paths):
    out = {}
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            log(f"  !! could not read {p}, skipping")
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out[name + "_rgb"] = rgb
        out[name + "_gray"] = to_gray(rgb)
    return out


# ----------------------------------------------------------------------------
# Experiments (one function per method)
# ----------------------------------------------------------------------------
def run_transfer_functions(out):
    r = np.arange(L)
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    ax = ax.ravel()
    ax[0].plot(r, 255 - r, color="C3"); ax[0].set_title("1. Negative  s = 255 - r")
    for g in (0.3, 0.5, 1, 2, 3):
        ax[1].plot(r, 255 * (r / 255) ** g, label=f"g={g}")
    ax[1].legend(fontsize=7); ax[1].set_title("2. Gamma  s = c.r^g")
    ax[2].plot(r, 255 / np.log(256) * np.log1p(r), color="C2")
    ax[2].set_title("3. Log  s = c.log(1+r)")
    ax[3].plot(r, np.interp(r, [0, 90, 166, 255], [0, 0, 255, 255]), label="min-max (90..166)")
    ax[3].plot(r, np.interp(r, [0, 100, 150, 255], [0, 40, 215, 255]), label="piece-wise")
    ax[3].legend(fontsize=7); ax[3].set_title("4. Contrast stretching")
    cdf = np.cumsum(np.exp(-((r - 90) ** 2) / (2 * 25 ** 2))); cdf = cdf / cdf[-1]
    ax[4].plot(r, 255 * cdf, color="C4"); ax[4].set_title("5. Hist. equalization  s = 255.CDF(r)")
    ax[5].plot(r, np.where((r >= 100) & (r <= 180), 255, r), color="C1")
    ax[5].set_title("6a. Slicing (keep background)")
    ax[6].plot(r, np.where((r >= 100) & (r <= 180), 255, 0), color="C1")
    ax[6].set_title("6b. Slicing (binary)")
    ax[7].step(r, (r >> 7) * 255, color="C5", where="post", label="plane 7 (MSB)")
    ax[7].step(r, ((r >> 6) & 1) * 255, color="C0", lw=0.8, where="post", label="plane 6")
    ax[7].legend(fontsize=7); ax[7].set_title("7. Bit-plane slicing")
    for a in ax:
        a.set_xlim(0, 255); a.set_ylim(0, 260); a.grid(alpha=0.3)
        a.set_xlabel("input r", fontsize=8); a.set_ylabel("output s", fontsize=8)
        a.set_aspect("equal", adjustable="box")
    save_fig(fig, f"{out}/00_transfer_functions.png")


def run_negative(imgs, out):
    log("\n=== 1. IMAGE NEGATIVE  (s = 255 - r) ===")
    for name, img in imgs:
        describe(name, img)
        res = negative(img)
        log(f"      output min={res.min()} max={res.max()} mean={res.mean():.1f} "
            f"(mean(in)+mean(out) = {img.mean() + res.mean():.1f} -> 255 as expected)")
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        show(ax[0, 0], img, "Input"); show(ax[0, 1], res, "Negative")
        plot_hist(ax[1, 0], img, "Histogram - input"); plot_hist(ax[1, 1], res, "Histogram - negative (mirrored)")
        save_fig(fig, f"{out}/01_negative/{name}.png")
        save_img(res, f"{out}/01_negative/outputs/{name}_negative.png")


def run_gamma(imgs, out, gammas=(0.3, 0.5, 1.5, 2.5)):
    log("\n=== 2. GAMMA CORRECTION  (s = c * r^gamma) ===")
    for name, img in imgs:
        describe(name, img)
        fig, ax = plt.subplots(1, len(gammas) + 1, figsize=(3.3 * (len(gammas) + 1), 3.6))
        show(ax[0], img, f"Input (mean={img.mean():.0f})")
        for i, g in enumerate(gammas, 1):
            res = gamma_correction(img, g)
            kind = "brightens" if g < 1 else "darkens"
            log(f"      gamma={g:<4} -> mean={res.mean():6.1f}  ({kind})")
            show(ax[i], res, f"gamma = {g}  (mean={res.mean():.0f})")
            save_img(res, f"{out}/02_gamma/outputs/{name}_gamma_{g}.png")
        save_fig(fig, f"{out}/02_gamma/{name}.png")


def run_log(imgs, out):
    log("\n=== 3. LOGARITHMIC TRANSFORM  (s = c * log(1 + r)) ===")
    for name, img in imgs:
        shown = display_u8(img)
        describe(name, shown)
        res = log_transform(img)
        log(f"      c = {255 / np.log1p(float(img.max())):.4f}   output mean={res.mean():.1f}")
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        show(ax[0, 0], shown, "Input (linear display)"); show(ax[0, 1], res, "Log transform")
        plot_hist(ax[1, 0], shown, "Histogram - input"); plot_hist(ax[1, 1], res, "Histogram - log output")
        save_fig(fig, f"{out}/03_log/{name}.png")
        save_img(res, f"{out}/03_log/outputs/{name}_log.png")


def run_contrast(imgs, out):
    log("\n=== 4. CONTRAST STRETCHING ===")
    for name, img in imgs:
        describe(name, img)
        lo, hi = int(img.min()), int(img.max())
        r1, r2 = np.percentile(img, [10, 90])           # data-adaptive knee points
        results = [
            ("Input", img),
            ("Min-max stretch", stretch_minmax(img)),
            ("Percentile 2-98", stretch_percentile(img)),
            (f"Piece-wise ({r1:.0f},40)-({r2:.0f},215)", stretch_piecewise(img, r1, 40, r2, 215)),
        ]
        for t, r in results:
            log(f"      {t:<34} min={r.min():3d} max={r.max():3d} std={r.std():6.2f}")
        fig, ax = plt.subplots(2, 4, figsize=(15, 6.8))
        for i, (t, r) in enumerate(results):
            show(ax[0, i], r, t); plot_hist(ax[1, i], r, "Histogram")
        save_fig(fig, f"{out}/04_contrast/{name}.png")
        for t, r in results[1:]:
            save_img(r, f"{out}/04_contrast/outputs/{name}_{t.split()[0].lower()}.png")


def run_histeq(imgs, out):
    log("\n=== 5. HISTOGRAM EQUALIZATION ===")
    for name, img in imgs:
        describe(name, img)
        if is_rgb(img):
            results = [("Input", img),
                       ("Per-channel equalization", equalize_rgb_per_channel(img)),
                       ("Luminance (Y) equalization", equalize_rgb_luminance(img))]
        else:
            results = [("Input", img), ("Equalized", equalize_gray(img))]
        for t, r in results:
            log(f"      {t:<28} min={r.min():3d} max={r.max():3d} std={to_gray(r).std():6.2f}")
        n = len(results)
        fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 7))
        for i, (t, r) in enumerate(results):
            show(ax[0, i], r, t)
            plot_hist(ax[1, i], r, "Histogram (dashed = CDF)", cdf=True)
        save_fig(fig, f"{out}/05_histeq/{name}.png")
        save_img(results[-1][1], f"{out}/05_histeq/outputs/{name}_equalized.png")


# Intensity ranges that isolate something meaningful in each built-in image
SLICE_RANGES = {"camera": (150, 255), "coins": (120, 200), "astronaut": (170, 255)}


def run_slicing(imgs, out):
    log("\n=== 6. INTENSITY LEVEL SLICING ===")
    for name, img in imgs:
        describe(name, img)
        lo, hi = SLICE_RANGES.get(name, (150, 255))
        mask = level_mask(img, lo, hi)
        log(f"      range [{lo}, {hi}] selects {mask.mean() * 100:.1f}% of the pixels")
        a, b = slice_highlight(img, lo, hi), slice_binary(img, lo, hi)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        show(ax[0], img, "Input")
        show(ax[1], a, f"A: highlight [{lo},{hi}], keep background")
        show(ax[2], b, f"B: binary, [{lo},{hi}] -> white")
        plot_hist(ax[3], img, "Selected range (orange)", band=(lo, hi))
        save_fig(fig, f"{out}/06_slicing/{name}.png")
        save_img(a, f"{out}/06_slicing/outputs/{name}_A_highlight.png")
        save_img(b, f"{out}/06_slicing/outputs/{name}_B_binary.png")


def run_bitplanes(imgs, out):
    log("\n=== 7. BIT-PLANE SLICING ===")
    for name, img in imgs:
        describe(name, img)
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        show(ax[0, 0], img, "Input (8 bits)")
        for k, a in zip(range(7, -1, -1), ax.ravel()[1:]):
            plane = bit_plane(img, k)
            show(a, plane, f"Bit plane {k}" + (" (MSB)" if k == 7 else " (LSB)" if k == 0 else ""))
            save_img(plane, f"{out}/07_bitplanes/outputs/{name}_plane{k}.png")
        save_fig(fig, f"{out}/07_bitplanes/{name}_planes.png")
        counts = (1, 2, 4, 6)
        fig, ax = plt.subplots(1, 5, figsize=(17, 3.8))
        show(ax[0], img, "Original (8 planes)")
        for i, n in enumerate(counts, 1):
            rec = bit_reconstruct(img, n)
            mse = np.mean((img.astype(float) - rec.astype(float)) ** 2)
            psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
            log(f"      top {n} plane(s): MSE={mse:8.2f}  PSNR={psnr:5.1f} dB")
            show(ax[i], rec, f"Top {n} plane(s)  PSNR={psnr:.1f} dB")
        save_fig(fig, f"{out}/07_bitplanes/{name}_reconstruction.png")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Elementary digital image processing methods")
    ap.add_argument("--images", nargs="*", help="your own image files (each runs as RGB and gray)")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--show", action="store_true", help="also display each figure")
    args = ap.parse_args()
    out = args.out

    log("Digital Image Processing - elementary methods")
    log(f"numpy {np.__version__} | opencv {cv2.__version__} | matplotlib {matplotlib.__version__}")

    if args.images:
        pool = load_custom(args.images)
        if not pool:
            sys.exit("No readable images given.")
        pick = lambda names: list(pool.items())         # same inputs for every method
        sets = {k: pick(None) for k in ("neg", "gam", "log", "con", "eq", "sl", "bp")}
    else:
        s = build_samples()
        for n, im in s.items():                         # keep inputs on disk for the report
            save_img(im, f"images/{n}.png")
        pick = lambda names: [(n, s[n]) for n in names]
        sets = {
            "neg": pick(["camera", "astronaut", "coins"]),
            "gam": pick(["camera", "coffee", "chelsea"]),
            "log": pick(["spectrum_camera", "dark_camera", "dark_astronaut"]),
            "con": pick(["lowcon_camera", "moon", "lowcon_chelsea"]),
            "eq": pick(["moon", "coins", "lowcon_chelsea"]),
            "sl": pick(["camera", "coins", "astronaut"]),
            "bp": pick(["camera", "coins", "astronaut"]),
        }

    run_transfer_functions(out)
    run_negative(sets["neg"], out)
    run_gamma(sets["gam"], out)
    run_log(sets["log"], out)
    run_contrast(sets["con"], out)
    run_histeq(sets["eq"], out)
    run_slicing(sets["sl"], out)
    run_bitplanes(sets["bp"], out)

    log("\nDone. All results are in: " + os.path.abspath(out))
    with open(os.path.join(out, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))


if __name__ == "__main__":
    main()
