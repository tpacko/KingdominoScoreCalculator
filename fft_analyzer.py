#!/usr/bin/env python3
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_PATH = "boards/game12_board.jpg"

TILES_OPTIONS = (5, 7)

# Visualization
SHOW_CV_STEPS = True
SHOW_MPL_SUMMARY = True
CV_WAITKEY_MS = 0  # 0 = waitKey(0)
DISPLAY_MAX_W = 1200  # 0 = no resizing for display
DISPLAY_MAX_H = 900   # 0 = no resizing for display

# Save debug images
DEBUG_DIR = "out_dbg"  # set "" to disable saving debug outputs

# Preprocessing / line evidence extraction
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
GRAD_PERCENTILE = 92.0

KLEN_DIV = 18.0
CLOSE_KSIZE = 3
OPEN_ITERS = 1
DILATE_ITERS = 1

# Classifier fusion weights (per axis)
AXIS_WEIGHTS = {
    "comb": 0.55,
    "autocorr": 0.35,
    "peaks": 0.10,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass(frozen=True)
class AxisScores:
    tiles: int
    autocorr: float
    comb: float
    peaks: float
    combined_loglik: float


@dataclass(frozen=True)
class AxisResult:
    axis_name: str
    n: int
    profile_raw: np.ndarray
    profile_hp: np.ndarray
    scores: List[AxisScores]
    probs: Dict[int, float]
    chosen_tiles: int


@dataclass(frozen=True)
class GridResult:
    tiles: int
    probs: Dict[int, float]
    x_axis: AxisResult
    y_axis: AxisResult
    debug: Dict[str, float]


# ============================================================================
# VIS HELPERS
# ============================================================================
def _resize_for_display(img: np.ndarray) -> np.ndarray:
    if DISPLAY_MAX_W <= 0 and DISPLAY_MAX_H <= 0:
        return img
    h, w = img.shape[:2]
    scale_w = DISPLAY_MAX_W / w if DISPLAY_MAX_W > 0 else 1.0
    scale_h = DISPLAY_MAX_H / h if DISPLAY_MAX_H > 0 else 1.0
    s = min(scale_w, scale_h, 1.0)
    if s >= 1.0:
        return img
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _cv_show(title: str, img: np.ndarray) -> None:
    disp = _resize_for_display(img)
    cv2.imshow(title, disp)
    cv2.waitKey(CV_WAITKEY_MS)


def _overlay_masks(bgr: np.ndarray, h_mask: np.ndarray, v_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    red = (h_mask > 0)
    green = (v_mask > 0)
    out[red] = (0, 0, 255)
    out[green] = (0, 255, 0)
    out[red & green] = (0, 255, 255)
    return out


# ============================================================================
# UTILITIES
# ============================================================================
def _ensure_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)


def _clahe(gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def _scharr_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def _gaussian_blur_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    x = x.astype(np.float32)
    if sigma <= 0:
        return x
    k = int(max(3, round(sigma * 6)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(x.reshape(1, -1), (k, 1), sigmaX=float(sigma), sigmaY=0).reshape(-1)


def _highpass_profile(profile: np.ndarray) -> np.ndarray:
    profile = profile.astype(np.float32)
    n = len(profile)
    sigma_small = max(1.5, n / 220.0)
    sigma_large = max(6.0, n / 30.0)

    sm = _gaussian_blur_1d(profile, sigma_small)
    base = _gaussian_blur_1d(sm, sigma_large)
    hp = sm - base
    hp = hp - np.median(hp)
    return hp.astype(np.float32)


def _percentile_threshold(img_u8: np.ndarray, percentile: float) -> np.ndarray:
    thr = float(np.percentile(img_u8, percentile))
    return (img_u8 >= thr).astype(np.uint8) * 255


# ============================================================================
# LINE EVIDENCE EXTRACTION
# ============================================================================
def extract_line_masks(
    gray: np.ndarray,
    clahe_clip: float,
    clahe_tile: int,
    grad_percentile: float,
    klen_div: float,
    close_ksize: int,
    open_iters: int,
    dilate_iters: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    gray = _ensure_uint8(gray)

    clahe_img = _clahe(gray, clahe_clip, clahe_tile)
    blur_img = cv2.GaussianBlur(clahe_img, (3, 3), 0)

    mag = _scharr_magnitude(blur_img)
    bw_thr = _percentile_threshold(mag, grad_percentile)

    close_k = max(3, int(close_ksize))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    bw_close = cv2.morphologyEx(bw_thr, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    h, w = gray.shape[:2]
    klen = int(max(25, round(min(h, w) / float(klen_div))))
    klen = int(min(klen, min(h, w) - 1 if min(h, w) > 1 else 25))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))

    h_mask = cv2.morphologyEx(bw_close, cv2.MORPH_OPEN, h_kernel, iterations=int(open_iters))
    v_mask = cv2.morphologyEx(bw_close, cv2.MORPH_OPEN, v_kernel, iterations=int(open_iters))

    if dilate_iters > 0:
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        h_mask = cv2.dilate(h_mask, d_kernel, iterations=int(dilate_iters))
        v_mask = cv2.dilate(v_mask, d_kernel, iterations=int(dilate_iters))

    stats = {
        "klen": float(klen),
        "mag_mean": float(np.mean(mag)),
        "bw_density": float(np.mean(bw_close > 0)),
        "h_density": float(np.mean(h_mask > 0)),
        "v_density": float(np.mean(v_mask > 0)),
    }

    steps = {
        "clahe": clahe_img,
        "blur": blur_img,
        "mag": mag,
        "bw_thr": bw_thr,
        "bw_close": bw_close,
        "h_mask": h_mask,
        "v_mask": v_mask,
    }

    return h_mask, v_mask, stats, steps


# ============================================================================
# PERIODICITY SCORERS (NO FFT)
# ============================================================================
def _autocorr_score(sig_hp: np.ndarray, tiles: int) -> float:
    s = sig_hp.astype(np.float32)
    s = s - np.mean(s)
    n = len(s)
    if n < 8:
        return 0.0

    F = np.fft.rfft(s)
    ac = np.fft.irfft(np.abs(F) ** 2, n=n)

    ac0 = float(ac[0]) if float(ac[0]) != 0.0 else 1.0
    period = float(n) / float(tiles)
    lo = int(max(1, math.floor(period * 0.85)))
    hi = int(min(n - 1, math.ceil(period * 1.15)))

    peak = float(np.max(ac[lo:hi + 1]))
    return float(peak / ac0)


def _comb_window_score(sig_hp: np.ndarray, tiles: int) -> float:
    p = sig_hp.astype(np.float32)
    p = p - np.mean(p)
    n = len(p)
    if n < 8:
        return 0.0

    period = float(n) / float(tiles)
    half = int(max(2, round(period * 0.08)))

    c = np.cumsum(np.concatenate([[0.0], p.astype(np.float64)]), dtype=np.float64)

    def win_sum(a: int, b: int) -> float:
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        return float(c[b] - c[a])

    max_off = int(max(1, round(period)))
    max_off = int(min(max_off, 512))

    best = -1e30
    for off in range(max_off):
        s = 0.0
        for i in range(tiles + 1):
            x = int(round(off + i * period))
            s += win_sum(x - half, x + half + 1)
        if s > best:
            best = s

    norm = float(np.sum(np.abs(p))) + 1e-6
    return float(best / norm)


def _peak_count_score(sig_hp: np.ndarray, tiles: int) -> Tuple[float, int]:
    s = sig_hp.astype(np.float32)
    s = s - np.mean(s)
    n = len(s)
    if n < 8:
        return 0.0, 0

    period = float(n) / float(tiles)
    sm = _gaussian_blur_1d(s, max(1.0, n / 200.0))

    thr = float(np.percentile(sm, 75))
    cand = [i for i in range(1, n - 1) if sm[i] > sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > thr]
    cand.sort(key=lambda i: float(sm[i]), reverse=True)

    min_dist = int(max(1, round(period * 0.50)))
    sel: List[int] = []
    for i in cand:
        if all(abs(i - j) >= min_dist for j in sel):
            sel.append(i)

    cnt = len(sel)
    expected = tiles + 1
    score = math.exp(-abs(cnt - expected))
    return float(score), cnt


def _combine_scores(
    raw: Dict[int, Dict[str, float]],
    weights: Dict[str, float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    loglik: Dict[int, float] = {}
    for t, feats in raw.items():
        ll = 0.0
        for k, w in weights.items():
            v = float(feats.get(k, 0.0))
            v = max(v, 1e-8)
            ll += float(w) * math.log(v)
        loglik[t] = float(ll)

    m = max(loglik.values())
    ex = {t: math.exp(v - m) for t, v in loglik.items()}
    z = sum(ex.values())
    probs = {t: float(ex[t] / z) for t in ex}
    return loglik, probs


def _score_axis_from_mask(
    mask_u8: np.ndarray,
    axis_name: str,
    tiles_options: Tuple[int, int],
    weights: Dict[str, float],
) -> AxisResult:
    if axis_name == "x":
        profile = np.sum(mask_u8.astype(np.float32), axis=0)
    elif axis_name == "y":
        profile = np.sum(mask_u8.astype(np.float32), axis=1)
    else:
        raise ValueError("axis_name must be 'x' or 'y'")

    hp = _highpass_profile(profile)

    raw_feats: Dict[int, Dict[str, float]] = {}
    for t in tiles_options:
        peaks_score, _ = _peak_count_score(hp, t)
        raw_feats[t] = {
            "autocorr": _autocorr_score(hp, t),
            "comb": _comb_window_score(hp, t),
            "peaks": peaks_score,
        }

    loglik, probs = _combine_scores(raw_feats, weights)

    scores_list: List[AxisScores] = []
    for t in tiles_options:
        feats = raw_feats[t]
        scores_list.append(
            AxisScores(
                tiles=t,
                autocorr=float(feats["autocorr"]),
                comb=float(feats["comb"]),
                peaks=float(feats["peaks"]),
                combined_loglik=float(loglik[t]),
            )
        )

    chosen = max(probs.items(), key=lambda kv: kv[1])[0]
    return AxisResult(
        axis_name=axis_name,
        n=int(len(profile)),
        profile_raw=profile.astype(np.float32),
        profile_hp=hp.astype(np.float32),
        scores=scores_list,
        probs=probs,
        chosen_tiles=int(chosen),
    )


# ============================================================================
# CLASSIFIER
# ============================================================================
def classify_grid(
    bgr: np.ndarray,
    tiles_options: Tuple[int, int],
    clahe_clip: float,
    clahe_tile: int,
    grad_percentile: float,
    klen_div: float,
    close_ksize: int,
    open_iters: int,
    dilate_iters: int,
    axis_weights: Dict[str, float],
) -> Tuple[GridResult, Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_mask, v_mask, stats, steps = extract_line_masks(
        gray=gray,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        grad_percentile=grad_percentile,
        klen_div=klen_div,
        close_ksize=close_ksize,
        open_iters=open_iters,
        dilate_iters=dilate_iters,
    )

    x_axis = _score_axis_from_mask(v_mask, "x", tiles_options, axis_weights)
    y_axis = _score_axis_from_mask(h_mask, "y", tiles_options, axis_weights)

    global_loglik: Dict[int, float] = {}
    for t in tiles_options:
        ll = 0.0
        ll += math.log(max(x_axis.probs.get(t, 1e-8), 1e-8))
        ll += math.log(max(y_axis.probs.get(t, 1e-8), 1e-8))
        quality = max(0.15, float(stats["h_density"] + stats["v_density"]) / 2.0)
        ll += math.log(quality)
        global_loglik[t] = float(ll)

    m = max(global_loglik.values())
    ex = {t: math.exp(v - m) for t, v in global_loglik.items()}
    z = sum(ex.values())
    probs = {t: float(ex[t] / z) for t in ex}
    chosen = max(probs.items(), key=lambda kv: kv[1])[0]

    debug = dict(stats)
    debug.update(
        {
            "x_p5": float(x_axis.probs.get(5, 0.0)),
            "x_p7": float(x_axis.probs.get(7, 0.0)),
            "y_p5": float(y_axis.probs.get(5, 0.0)),
            "y_p7": float(y_axis.probs.get(7, 0.0)),
        }
    )

    steps = dict(steps)
    steps["gray"] = gray
    steps["overlay"] = _overlay_masks(bgr, h_mask, v_mask)

    res = GridResult(
        tiles=int(chosen),
        probs=probs,
        x_axis=x_axis,
        y_axis=y_axis,
        debug=debug,
    )
    return res, steps


# ============================================================================
# DEBUG OUTPUTS
# ============================================================================
def save_debug(out_dir: str, bgr: np.ndarray, steps: Dict[str, np.ndarray], res: GridResult) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "0_original.png"), bgr)
    for k in ("gray", "clahe", "blur", "mag", "bw_thr", "bw_close", "h_mask", "v_mask", "overlay"):
        if k in steps:
            cv2.imwrite(os.path.join(out_dir, f"{k}.png"), steps[k])


def show_all_steps_cv(bgr: np.ndarray, steps: Dict[str, np.ndarray], res: GridResult) -> None:
    _cv_show("0) Original (BGR)", bgr)
    _cv_show("1) Grayscale", steps["gray"])
    _cv_show("2) CLAHE", steps["clahe"])
    _cv_show("3) Blur", steps["blur"])
    _cv_show("4) Scharr magnitude", steps["mag"])
    _cv_show(f"5) Percentile threshold (p={GRAD_PERCENTILE})", steps["bw_thr"])
    _cv_show("6) After CLOSE", steps["bw_close"])
    _cv_show("7) Horizontal-line mask (OPEN h-kernel)", steps["h_mask"])
    _cv_show("8) Vertical-line mask (OPEN v-kernel)", steps["v_mask"])
    _cv_show("9) Overlay (R=H lines, G=V lines, Y=both)", steps["overlay"])

    banner = np.zeros((200, 800), dtype=np.uint8)
    msg = f"Decision: {res.tiles}x{res.tiles} | P5={res.probs.get(5,0):.4f} P7={res.probs.get(7,0):.4f}"
    cv2.putText(banner, msg, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    _cv_show("10) Result", banner)


def show_summary_mpl(bgr: np.ndarray, steps: Dict[str, np.ndarray], res: GridResult) -> None:
    fig = plt.figure(figsize=(18, 12))

    ax = plt.subplot(3, 4, 1)
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Original")
    ax.axis("off")

    ax = plt.subplot(3, 4, 2)
    ax.imshow(steps["clahe"], cmap="gray")
    ax.set_title("CLAHE")
    ax.axis("off")

    ax = plt.subplot(3, 4, 3)
    ax.imshow(steps["mag"], cmap="gray")
    ax.set_title("Scharr magnitude")
    ax.axis("off")

    ax = plt.subplot(3, 4, 4)
    ax.imshow(steps["bw_close"], cmap="gray")
    ax.set_title("Edges (thr+close)")
    ax.axis("off")

    ax = plt.subplot(3, 4, 5)
    ax.imshow(steps["h_mask"], cmap="gray")
    ax.set_title("H mask")
    ax.axis("off")

    ax = plt.subplot(3, 4, 6)
    ax.imshow(steps["v_mask"], cmap="gray")
    ax.set_title("V mask")
    ax.axis("off")

    ax = plt.subplot(3, 4, 7)
    ax.imshow(cv2.cvtColor(steps["overlay"], cv2.COLOR_BGR2RGB))
    ax.set_title("Overlay")
    ax.axis("off")

    ax = plt.subplot(3, 4, 9)
    ax.plot(res.x_axis.profile_raw, linewidth=1.0, label="raw")
    ax.plot(res.x_axis.profile_hp, linewidth=1.0, label="highpass")
    ax.set_title(f"X profile (from V mask)  probs={res.x_axis.probs}")
    ax.legend()

    ax = plt.subplot(3, 4, 10)
    ax.plot(res.y_axis.profile_raw, linewidth=1.0, label="raw")
    ax.plot(res.y_axis.profile_hp, linewidth=1.0, label="highpass")
    ax.set_title(f"Y profile (from H mask)  probs={res.y_axis.probs}")
    ax.legend()

    ax = plt.subplot(3, 4, 11)
    xs = [s.tiles for s in res.x_axis.scores]
    comb = [s.comb for s in res.x_axis.scores]
    ac = [s.autocorr for s in res.x_axis.scores]
    pk = [s.peaks for s in res.x_axis.scores]
    ax.plot(xs, comb, marker="o", label="comb")
    ax.plot(xs, ac, marker="o", label="autocorr")
    ax.plot(xs, pk, marker="o", label="peaks")
    ax.set_xticks(xs)
    ax.set_title("X axis feature scores")
    ax.legend()

    ax = plt.subplot(3, 4, 12)
    ys = [s.tiles for s in res.y_axis.scores]
    comb = [s.comb for s in res.y_axis.scores]
    ac = [s.autocorr for s in res.y_axis.scores]
    pk = [s.peaks for s in res.y_axis.scores]
    ax.plot(ys, comb, marker="o", label="comb")
    ax.plot(ys, ac, marker="o", label="autocorr")
    ax.plot(ys, pk, marker="o", label="peaks")
    ax.set_xticks(ys)
    ax.set_title("Y axis feature scores")
    ax.legend()

    fig.suptitle(
        f"Decision: {res.tiles}x{res.tiles} | P(5)={res.probs.get(5,0):.4f} P(7)={res.probs.get(7,0):.4f} | "
        f"klen={res.debug.get('klen',0):.0f} bw={res.debug.get('bw_density',0):.4f} "
        f"h={res.debug.get('h_density',0):.4f} v={res.debug.get('v_density',0):.4f}"
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    res, steps = classify_grid(
        bgr=bgr,
        tiles_options=TILES_OPTIONS,
        clahe_clip=CLAHE_CLIP,
        clahe_tile=CLAHE_TILE,
        grad_percentile=GRAD_PERCENTILE,
        klen_div=KLEN_DIV,
        close_ksize=CLOSE_KSIZE,
        open_iters=OPEN_ITERS,
        dilate_iters=DILATE_ITERS,
        axis_weights=AXIS_WEIGHTS,
    )

    print(f"GRID: {res.tiles}x{res.tiles}")
    print(f"  P(5x5)={res.probs.get(5,0.0):.6f}  P(7x7)={res.probs.get(7,0.0):.6f}")
    print(f"  X axis probs: {res.x_axis.probs} chosen={res.x_axis.chosen_tiles}")
    for s in res.x_axis.scores:
        print(f"    X tiles={s.tiles}: comb={s.comb:.6f} ac={s.autocorr:.6f} peaks={s.peaks:.6f} ll={s.combined_loglik:.6f}")
    print(f"  Y axis probs: {res.y_axis.probs} chosen={res.y_axis.chosen_tiles}")
    for s in res.y_axis.scores:
        print(f"    Y tiles={s.tiles}: comb={s.comb:.6f} ac={s.autocorr:.6f} peaks={s.peaks:.6f} ll={s.combined_loglik:.6f}")
    print(
        "  Diagnostics: "
        f"klen={res.debug.get('klen'):.0f} "
        f"bw_density={res.debug.get('bw_density'):.6f} "
        f"h_density={res.debug.get('h_density'):.6f} "
        f"v_density={res.debug.get('v_density'):.6f}"
    )

    if DEBUG_DIR:
        save_debug(DEBUG_DIR, bgr, steps, res)
        print(f"Saved debug outputs to: {DEBUG_DIR}")

    if SHOW_CV_STEPS:
        show_all_steps_cv(bgr, steps, res)
        cv2.destroyAllWindows()

    if SHOW_MPL_SUMMARY:
        show_summary_mpl(bgr, steps, res)


if __name__ == "__main__":
    main()
