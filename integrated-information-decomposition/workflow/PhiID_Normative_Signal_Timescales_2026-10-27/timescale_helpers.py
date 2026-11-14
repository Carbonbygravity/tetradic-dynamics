"""Original signal-timescale definitions, isolated from historical launch code.

The five functions below retain their original numerical definitions. The
historical source is preserved separately as historical_helper_reference.py.
"""

import numpy as np

MIN_VALID_POINTS = 20
MI_BINS = 16


def normalized_autocorr(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < MIN_VALID_POINTS:
        return None

    x = x - np.nanmean(x)
    var = np.nanvar(x)

    if not np.isfinite(var) or var <= 0:
        return None

    max_lag = int(min(max_lag, x.size - 2))

    if max_lag < 2:
        return None

    ac = np.full(max_lag + 1, np.nan, dtype=float)
    ac[0] = 1.0

    for lag in range(1, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]

        if a.size < MIN_VALID_POINTS:
            continue

        denom = np.sqrt(np.nanvar(a) * np.nanvar(b))
        if np.isfinite(denom) and denom > 0:
            ac[lag] = np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b))) / denom

    return ac


def first_zero_crossing(ac):
    if ac is None:
        return np.nan

    for lag in range(1, len(ac)):
        if np.isfinite(ac[lag]) and ac[lag] <= 0:
            return float(lag)

    return np.nan


def first_local_minimum(vals):
    vals = np.asarray(vals, dtype=float)

    for lag in range(1, len(vals) - 1):
        if not np.isfinite(vals[lag - 1: lag + 2]).all():
            continue
        if vals[lag] <= vals[lag - 1] and vals[lag] <= vals[lag + 1]:
            return float(lag)

    return np.nan


def histogram_mi(a, b, bins=MI_BINS):
    ok = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a[ok], dtype=float)
    b = np.asarray(b[ok], dtype=float)

    if a.size < MIN_VALID_POINTS or np.nanvar(a) <= 0 or np.nanvar(b) <= 0:
        return np.nan

    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / np.sum(hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nz = pxy > 0
    denom = px[:, None] * py[None, :]

    return float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))


def mi_curve(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < MIN_VALID_POINTS:
        return None

    max_lag = int(min(max_lag, x.size - 2))

    if max_lag < 2:
        return None

    out = np.full(max_lag + 1, np.nan, dtype=float)

    for lag in range(1, max_lag + 1):
        out[lag] = histogram_mi(x[:-lag], x[lag:], bins=MI_BINS)

    return out
