"""
Script: 05_te_surrogate_controls_targeted.py

Purpose
-------
Evaluate the targeted transfer-entropy (TE) hypothesis for frontal and midline
homotopic pairs in the pre-response window using gated surrogate controls.

Contrast
--------
Delta TE = (Leader RH->LH) - (Follower RH->LH)

Scope
-----
- Pairwise trial-level source time series referenced through a manifest JSON
- ROI subset restricted by frontal and midline pair-name filters
- Time window sliced to pre-response (-1000 to 0 ms)

Estimator
---------
- JIDT Kraskov transfer entropy with k = 4
- Explicit TEK initialise signature: (kHist, lHist, kTau, lTau, delay)

Surrogate strategy
------------------
- Run trial-shuffle surrogates first
- Stop early if the interim trial-shuffle result is not promising
- Run time-shift surrogates only if the full trial-shuffle result passes gating

Outputs
-------
- subject_deltas_TARGET_pairwise.csv
- pair3d_TARGET_summary.json
- pair3d_TARGET_shuffleX.npy
- pair3d_TARGET_timeshift.npy
- pair3d_TARGET_meta.json
"""

import json
import time
from datetime import datetime
from pathlib import Path

import jpype
import jpype.imports
from jpype.types import JArray, JDouble, JInt
import numpy as np
import pandas as pd


MANIFEST_PATH = Path("PATH_TO_TE_MEMMAP_PAIRS") / "manifest_prepost_pairs.json"
JAR_PATH = Path("PATH_TO_JIDT_JAR") / "infodynamics.jar"
OUT_DIR = Path("PATH_TO_TE_SURROGATE_OUTPUT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = ("PRE", "POST")

K_KRASKOV = 4
KHIST, LHIST = 1, 1
KTAU, LTAU = 1, 1
DELAY = 1

RNG_SEED = 202

N_SURR = 100
EARLY_STOP_AT = 50
EARLY_STOP_P = 0.20
RUN_TIMESHIFT_IF_PASS = True

FS_HZ = 1000
TIME_SLICE_MS = (-1000, 0)

PAIR_FILTER_SUBSTR = (
    "frontal",
    "frontalpole",
    "precentral",
    "rostral",
    "acc",
    "cingulate",
    "orbitofrontal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
)

HEARTBEAT_EVERY = 1
NULLS_PER_PAIR = False


def checkpoint_path(stem):
    return OUT_DIR / f"pair3d_TARGET_{stem}.npy"


CKPT_TS = checkpoint_path("timeshift")
CKPT_SH = checkpoint_path("shuffleX")
CKPT_META = OUT_DIR / "pair3d_TARGET_meta.json"


def cohen_dz(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    std = values.std(ddof=1)
    return values.mean() / (std if std > 0 else np.nan)


def circ_time_shift_3d(array_3d, rng, min_shift=20):
    """Shift along time while preserving within-trial autocorrelation."""
    n_time = array_3d.shape[-1]
    max_safe = max(min_shift + 1, n_time // 2)
    if max_safe <= min_shift:
        lag = min(5, max(1, n_time // 4))
    else:
        lag = int(rng.integers(min_shift, max_safe))
    return np.roll(array_3d, shift=lag, axis=2)


def trial_shuffle_3d(array_3d, rng):
    """Shuffle trials while preserving within-trial temporal structure."""
    indices = rng.permutation(array_3d.shape[0])
    return array_3d[indices, :, :]


def ms_to_idx(ms, fs_hz, n_time):
    """Map milliseconds to sample index assuming -1000..+1000 ms trials."""
    zero = n_time // 2
    return zero + int(np.round(ms * fs_hz / 1000.0))


def slice_time_3d_ms(array_3d, fs_hz, ms_window):
    """Slice a (trials, pairs, time) array using a millisecond window."""
    n_time = array_3d.shape[-1]
    start_ms, stop_ms = ms_window
    start_idx = max(0, ms_to_idx(start_ms, fs_hz, n_time))
    stop_idx = min(n_time, ms_to_idx(stop_ms, fs_hz, n_time))
    return array_3d[..., start_idx:stop_idx]


def start_jvm_and_get_calculator():
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", f"-Djava.class.path={JAR_PATH}")
    from infodynamics.measures.continuous.kraskov import TransferEntropyCalculatorKraskov as TEK

    calculator = TEK()
    calculator.setProperty("k", str(K_KRASKOV))
    return calculator


def te_trialwise(calculator, source_2d, target_2d):
    """Compute trial-wise TE for arrays shaped (n_trials, n_time)."""
    n_trials = source_2d.shape[0]
    out = np.empty(n_trials, float)
    for trial_idx in range(n_trials):
        source = np.ascontiguousarray(source_2d[trial_idx], dtype=np.float64)
        target = np.ascontiguousarray(target_2d[trial_idx], dtype=np.float64)
        calculator.initialise(JInt(KHIST), JInt(LHIST), JInt(KTAU), JInt(LTAU), JInt(DELAY))
        try:
            calculator.setObservations(source, target)
        except TypeError:
            j_double_array = JArray(JDouble)
            calculator.setObservations(j_double_array(source.tolist()), j_double_array(target.tolist()))
        out[trial_idx] = float(calculator.computeAverageLocalOfObservations())
    return out


def te_subject_pairmeans(calculator, leader_rh, leader_lh, follower_rh, follower_lh):
    """Compute per-pair Delta TE in the configured millisecond window."""
    leader_rh = slice_time_3d_ms(leader_rh, FS_HZ, TIME_SLICE_MS)
    leader_lh = slice_time_3d_ms(leader_lh, FS_HZ, TIME_SLICE_MS)
    follower_rh = slice_time_3d_ms(follower_rh, FS_HZ, TIME_SLICE_MS)
    follower_lh = slice_time_3d_ms(follower_lh, FS_HZ, TIME_SLICE_MS)

    n_pairs = leader_rh.shape[1]
    out = np.empty(n_pairs, float)
    for pair_idx in range(n_pairs):
        te_leader = te_trialwise(calculator, leader_rh[:, pair_idx, :], leader_lh[:, pair_idx, :]).mean()
        te_follower = te_trialwise(
            calculator, follower_rh[:, pair_idx, :], follower_lh[:, pair_idx, :]
        ).mean()
        out[pair_idx] = te_leader - te_follower
    return out


def keep_target_pairs(pair_names, substrings):
    substrings = tuple(substr.lower() for substr in substrings)
    keep_idx = []
    for pair_idx, name in enumerate(pair_names):
        lowered = str(name).lower()
        if any(substr in lowered for substr in substrings):
            keep_idx.append(pair_idx)
    return np.array(keep_idx, dtype=int)


def load_or_init_nulls(size, n_pairs):
    have_shuffle = CKPT_META.exists() and CKPT_SH.exists()
    if have_shuffle:
        meta = json.loads(CKPT_META.read_text())
        shuffle = np.load(CKPT_SH)
        done = int(meta.get("done", shuffle.shape[0]))
        return shuffle, done
    shape = (0, 2, n_pairs) if NULLS_PER_PAIR else (0, 2)
    return np.empty(shape, float), 0


def grow_nulls(array, size, n_pairs):
    if NULLS_PER_PAIR:
        out = np.full((size, 2, n_pairs), np.nan, float)
        if array.shape[0] > 0:
            out[: array.shape[0], :, :] = array
    else:
        out = np.full((size, 2), np.nan, float)
        if array.shape[0] > 0:
            out[: array.shape[0], :] = array
    return out


def collapse_if_needed(array):
    if array is None:
        return None
    return np.nanmean(array, axis=2) if array.ndim == 3 else array


def perwin_p(null_2d, observed_mean):
    out = {}
    for window_idx, window in enumerate(WINDOWS):
        null_values = null_2d[:, window_idx]
        valid = np.isfinite(null_values)
        if np.isfinite(observed_mean[window]) and valid.any():
            out[window] = float(
                (np.sum(null_values[valid] >= observed_mean[window]) + 1) / (np.sum(valid) + 1)
            )
        else:
            out[window] = np.nan
    return out


def save_meta(done, observed_mean, dz_pairs, pair_names):
    meta = {
        "done": done,
        "observed_means": observed_mean,
        "dz_pair_avg": dz_pairs,
        "pair_names": pair_names,
        "rng_seed": RNG_SEED,
        "n_surrogates": N_SURR,
        "k_kraskov": K_KRASKOV,
        "kHist": KHIST,
        "lHist": LHIST,
        "kTau": KTAU,
        "lTau": LTAU,
        "delay": DELAY,
        "windows": WINDOWS,
        "manifest_path": str(MANIFEST_PATH),
        "nulls_per_pair": NULLS_PER_PAIR,
        "fs_hz": FS_HZ,
        "time_slice_ms": TIME_SLICE_MS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    CKPT_META.write_text(json.dumps(meta, indent=2))


if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}\n"
        "Edit MANIFEST_PATH to point to manifest_prepost_pairs.json."
    )
if not JAR_PATH.exists():
    raise FileNotFoundError(
        f"JIDT jar not found: {JAR_PATH}\n"
        "Edit JAR_PATH to point to infodynamics.jar."
    )


manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
if "subjects" not in manifest:
    raise ValueError("Manifest is missing the required top-level key 'subjects'.")

pair_names = manifest.get("pair_names", [f"pair{idx:02d}" for idx in range(10)])
keep_idx = keep_target_pairs(pair_names, PAIR_FILTER_SUBSTR)
if keep_idx.size == 0:
    print("[WARN] Pair filter matched 0 pairs; proceeding with all pairs.")
    keep_idx = np.arange(len(pair_names))
pair_names = [pair_names[idx] for idx in keep_idx]

calculator = start_jvm_and_get_calculator()

observed = {window: [] for window in WINDOWS}
subject_ids = {window: [] for window in WINDOWS}
for window in WINDOWS:
    if window not in manifest["subjects"]:
        raise ValueError(f"Manifest is missing subject entries for window '{window}'.")
    for entry in manifest["subjects"][window]:
        leader_rh = np.load(entry["Leader_RH3d"], mmap_mode="r")[:, keep_idx, :]
        leader_lh = np.load(entry["Leader_LH3d"], mmap_mode="r")[:, keep_idx, :]
        follower_rh = np.load(entry["Follower_RH3d"], mmap_mode="r")[:, keep_idx, :]
        follower_lh = np.load(entry["Follower_LH3d"], mmap_mode="r")[:, keep_idx, :]

        observed[window].append(
            te_subject_pairmeans(calculator, leader_rh, leader_lh, follower_rh, follower_lh)
        )
        subject_ids[window].append(entry["sid"])

observed = {window: np.stack(observed[window], axis=0) for window in WINDOWS}
n_pairs = observed[WINDOWS[0]].shape[1]

mean_pairs = {window: float(np.nanmean(observed[window])) for window in WINDOWS}
dz_pairs = {
    window: float(cohen_dz(np.nanmean(observed[window], axis=1))) for window in WINDOWS
}
per_pair = {window: np.nanmean(observed[window], axis=0) for window in WINDOWS}
dz_pair = {
    window: np.array([cohen_dz(observed[window][:, pair_idx]) for pair_idx in range(n_pairs)])
    for window in WINDOWS
}

print("TEK initialise expects 5 ints: (kHist, lHist, kTau, lTau, delay)")
print("[INFO] Computing observed Delta TE (pairwise, TARGETED, pre-response -1000..0 ms) ...")
for window in WINDOWS:
    print(f" {window}: mean Delta TE={mean_pairs[window]:.6f}  dz={dz_pairs[window]:.3f}")
for window in WINDOWS:
    order = np.argsort(per_pair[window])[::-1][: min(5, n_pairs)]
    print(f"[INFO] Top pairs (observed, {window}):")
    for pair_idx in order:
        print(
            f"   {pair_names[pair_idx]}: mean Delta TE={per_pair[window][pair_idx]:.6f} "
            f"dz={dz_pair[window][pair_idx]:.3f}"
        )

rows = []
for window in WINDOWS:
    for sid, vec in zip(subject_ids[window], observed[window]):
        for pair_idx, value in enumerate(vec):
            rows.append(
                {
                    "subject_id": sid,
                    "window": window,
                    "pair_index": pair_idx,
                    "pair_name": pair_names[pair_idx],
                    "delta_TE": float(value),
                }
            )
subject_csv = OUT_DIR / "subject_deltas_TARGET_pairwise.csv"
pd.DataFrame(rows).to_csv(subject_csv, index=False)

shuffle_nulls, done = load_or_init_nulls(N_SURR, n_pairs)
shuffle_nulls = grow_nulls(shuffle_nulls, N_SURR, n_pairs)

rng_master = np.random.default_rng(RNG_SEED)
start_time = time.time()
interim_checked = False
last_shuffle_idx = done - 1

for shuffle_idx in range(done, N_SURR):
    rng = np.random.default_rng(rng_master.integers(1 << 31))

    window_values = []
    for window in WINDOWS:
        subject_values = []
        for entry in manifest["subjects"][window]:
            leader_rh = np.load(entry["Leader_RH3d"], mmap_mode="r")[:, keep_idx, :]
            leader_lh = np.load(entry["Leader_LH3d"], mmap_mode="r")[:, keep_idx, :]
            follower_rh = np.load(entry["Follower_RH3d"], mmap_mode="r")[:, keep_idx, :]
            follower_lh = np.load(entry["Follower_LH3d"], mmap_mode="r")[:, keep_idx, :]

            leader_rh_s = trial_shuffle_3d(leader_rh, rng)
            follower_rh_s = trial_shuffle_3d(follower_rh, rng)

            subject_values.append(
                te_subject_pairmeans(
                    calculator, leader_rh_s, leader_lh, follower_rh_s, follower_lh
                )
            )

        subject_values = np.stack(subject_values, axis=0)
        if NULLS_PER_PAIR:
            window_values.append(np.nanmean(subject_values, axis=0))
        else:
            window_values.append(float(np.nanmean(subject_values)))

    if NULLS_PER_PAIR:
        shuffle_nulls[shuffle_idx, 0, :] = window_values[0]
        shuffle_nulls[shuffle_idx, 1, :] = window_values[1]
    else:
        shuffle_nulls[shuffle_idx, :] = window_values

    last_shuffle_idx = shuffle_idx

    if ((shuffle_idx + 1) % HEARTBEAT_EVERY) == 0:
        elapsed = (time.time() - start_time) / 60.0
        print(f"[HB] trial-shuffle {shuffle_idx + 1}/{N_SURR}  elapsed={elapsed:.1f}m", flush=True)
        np.save(CKPT_SH, shuffle_nulls[: shuffle_idx + 1])
        save_meta(shuffle_idx + 1, mean_pairs, dz_pairs, pair_names)

    if (shuffle_idx + 1) == EARLY_STOP_AT and not interim_checked:
        interim_nulls = collapse_if_needed(shuffle_nulls[: shuffle_idx + 1])
        p_interim = perwin_p(interim_nulls, mean_pairs)
        print(f"[INTERIM] p_trial_shuffle after {EARLY_STOP_AT}: {p_interim}")
        if any(np.isfinite(p) and p > EARLY_STOP_P for p in p_interim.values()):
            print("[EARLY-STOP] Trial-shuffle p too large; concluding no robust directionality.")
            break
        interim_checked = True

ran_full_shuffle = last_shuffle_idx + 1 >= N_SURR
timeshift_nulls = None

if RUN_TIMESHIFT_IF_PASS and (ran_full_shuffle or interim_checked):
    shuffle_final = collapse_if_needed(shuffle_nulls[: last_shuffle_idx + 1])
    p_shuffle_final = (
        perwin_p(shuffle_final, mean_pairs) if shuffle_final is not None else {window: np.nan for window in WINDOWS}
    )
    proceed_timeshift = all(np.isfinite(p) and p <= 0.10 for p in p_shuffle_final.values())
    print(f"[DECISION] p_trial_shuffle final: {p_shuffle_final} -> proceed time-shift? {proceed_timeshift}")

    if proceed_timeshift:
        if CKPT_TS.exists():
            timeshift_nulls = np.load(CKPT_TS)
            done_ts = timeshift_nulls.shape[0]
        else:
            shape = (0, 2, n_pairs) if NULLS_PER_PAIR else (0, 2)
            timeshift_nulls = np.empty(shape, float)
            done_ts = 0
        timeshift_nulls = grow_nulls(timeshift_nulls, N_SURR, n_pairs)

        rng_master_ts = np.random.default_rng(RNG_SEED + 17)
        start_time_ts = time.time()
        for ts_idx in range(done_ts, N_SURR):
            rng = np.random.default_rng(rng_master_ts.integers(1 << 31))

            window_values = []
            for window in WINDOWS:
                subject_values = []
                for entry in manifest["subjects"][window]:
                    leader_rh = np.load(entry["Leader_RH3d"], mmap_mode="r")[:, keep_idx, :]
                    leader_lh = np.load(entry["Leader_LH3d"], mmap_mode="r")[:, keep_idx, :]
                    follower_rh = np.load(entry["Follower_RH3d"], mmap_mode="r")[:, keep_idx, :]
                    follower_lh = np.load(entry["Follower_LH3d"], mmap_mode="r")[:, keep_idx, :]

                    leader_lh_s = circ_time_shift_3d(leader_lh, rng)
                    follower_lh_s = circ_time_shift_3d(follower_lh, rng)

                    subject_values.append(
                        te_subject_pairmeans(
                            calculator, leader_rh, leader_lh_s, follower_rh, follower_lh_s
                        )
                    )

                subject_values = np.stack(subject_values, axis=0)
                if NULLS_PER_PAIR:
                    window_values.append(np.nanmean(subject_values, axis=0))
                else:
                    window_values.append(float(np.nanmean(subject_values)))

            if NULLS_PER_PAIR:
                timeshift_nulls[ts_idx, 0, :] = window_values[0]
                timeshift_nulls[ts_idx, 1, :] = window_values[1]
            else:
                timeshift_nulls[ts_idx, :] = window_values

            if ((ts_idx + 1) % HEARTBEAT_EVERY) == 0:
                elapsed = (time.time() - start_time_ts) / 60.0
                print(f"[HB] time-shift {ts_idx + 1}/{N_SURR}  elapsed={elapsed:.1f}m", flush=True)
                np.save(CKPT_TS, timeshift_nulls[: ts_idx + 1])

shuffle_final = collapse_if_needed(shuffle_nulls[: last_shuffle_idx + 1]) if last_shuffle_idx >= 0 else None
timeshift_final = collapse_if_needed(timeshift_nulls) if timeshift_nulls is not None else None

p_shuffle = perwin_p(shuffle_final, mean_pairs) if shuffle_final is not None else {window: np.nan for window in WINDOWS}
p_timeshift = (
    perwin_p(timeshift_final, mean_pairs) if timeshift_final is not None else {window: np.nan for window in WINDOWS}
)

summary = {
    "pair_names": pair_names,
    "observed": {
        "mean_delta_TE_pair_avg": mean_pairs,
        "dz_pair_avg": dz_pairs,
        "per_pair_mean": {window: per_pair[window].tolist() for window in WINDOWS},
        "per_pair_dz": {window: dz_pair[window].tolist() for window in WINDOWS},
    },
    "p_trial_shuffle": p_shuffle,
    "p_time_shift": p_timeshift,
    "rng_seed": RNG_SEED,
    "n_surrogates": N_SURR,
    "k_kraskov": K_KRASKOV,
    "kHist": KHIST,
    "lHist": LHIST,
    "kTau": KTAU,
    "lTau": LTAU,
    "delay": DELAY,
    "windows": WINDOWS,
    "manifest_path": str(MANIFEST_PATH),
    "nulls_per_pair": NULLS_PER_PAIR,
    "fs_hz": FS_HZ,
    "time_slice_ms": TIME_SLICE_MS,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
}
summary_path = OUT_DIR / "pair3d_TARGET_summary.json"
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n=== TARGETED TE Surrogate Summary (pre-response -1000..0 ms) ===")
for window in WINDOWS:
    print(
        f" {window}: mean Delta TE (pair-avg)={summary['observed']['mean_delta_TE_pair_avg'][window]:.6f}  "
        f"dz={summary['observed']['dz_pair_avg'][window]:.3f}"
    )
    print(
        f"     p(one-sided): trial-shuffle={summary['p_trial_shuffle'][window]:.4f}  "
        f"time-shift={summary['p_time_shift'][window]:.4f}"
    )

print(f"\n[Saved] {subject_csv}")
print(f"[Saved] {summary_path}")
print("[Done]")
