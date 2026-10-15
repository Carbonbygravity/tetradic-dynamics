"""
Script: 02_trial_level_lmm_rt.py

Purpose
-------
Fit the main trial-level frequentist linear mixed model for reaction time.

Inputs
------
- reshaped_rt_data_dyads_cleaned.csv
  Required columns: dyad, participant_id, role, condition, rt_ms

Outputs
-------
- lmm_predicted_cell_means.csv
- lmm_planned_contrasts.csv

Model
-----
- Outcome: log(rt_ms)
- Fixed effects: Role x broad condition with sum coding (-0.5, +0.5)
- Random effects: participant-level random structure, with dyad variance
  components when supported

Notes
-----
- This script preserves the original fallback strategy for singular or
  unsupported mixed-model fits.
- Planned contrasts are back-transformed to milliseconds and percent change.
"""

import argparse
import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import chi2

# ------------------------- User config -----------------------------------------
from pathlib import Path

parser = argparse.ArgumentParser(description="Fit the trial-level behavioural RT mixed model.")
parser.add_argument("--input-csv", required=True, type=Path,
                    help="Cleaned trial-level RT CSV")
parser.add_argument("--output-dir", required=True, type=Path,
                    help="New output directory; must not already exist")
args = parser.parse_args()

DATA_PATH = args.input_csv
OUT_DIR = args.output_dir
if not DATA_PATH.is_file():
    parser.error(f"Input CSV does not exist: {DATA_PATH}")
if OUT_DIR.exists():
    parser.error(f"Output path already exists; choose a new directory: {OUT_DIR}")

MIN_EXPECTED_PARTICIPANTS = 32
REQUIRED_COLUMNS = {"dyad", "participant_id", "role", "condition", "rt_ms"}

# ------------------------- Helpers ---------------------------------------------
def normalize_condition_to_cell(role: str, condition: str) -> str:
    """Map to {LMR,LMA,FMR,FMA} from role + raw condition."""
    role_norm = str(role).strip().title()
    base = re.sub(r'[^A-Z]', '', str(condition).strip().upper())
    base3 = base[:3] if len(base) >= 3 else base
    if base3 in {"LMR","LMA","FMR","FMA"}:
        return base3
    if "MIRROR" in base or "MR" in base:
        return "LMR" if role_norm == "Leader" else "FMR"
    if "MATCH" in base or "MA" in base:
        return "LMA" if role_norm == "Leader" else "FMA"
    return ("LMR" if role_norm == "Leader" else "FMR") if "R" in base else ("LMA" if role_norm=="Leader" else "FMA")

def ensure_labels(df):
    """Create 'cell' and 'broad_condition'; standardise Role labels."""
    df = df.copy()
    df["role"] = df["role"].astype(str).str.title().str.strip()
    if "cell" not in df:
        df["cell"] = [normalize_condition_to_cell(r, c) for r, c in zip(df["role"], df["condition"])]
    cond_map = {"LMR":"Mirroring","FMR":"Mirroring","LMA":"Matching","FMA":"Matching"}
    df["broad_condition"] = df["cell"].map(cond_map)
    return df

def sum_code(series, mapping):
    """Sum-code a categorical series to −0.5/+0.5."""
    return series.map(mapping).astype(float)

def vc_formula_for_dyad(df):
    """Variance component for dyad random intercepts."""
    return {"dyad": "0 + C(dyad)"}

def fit_with_fallbacks(df):
    """
    Fit MixedLM with:
      1) vc_formula for dyad + maximal random slopes per participant
      2) simplify random slopes if singular
      3) fall back to dyad fixed effects if vc_formula unsupported/failing
    Returns: (result, used_structure_label)
    """
    re_forms = [
        "~ role_c + cond_c + role_c:cond_c",
        "~ role_c + cond_c",
        "~ cond_c",
        "1",
    ]
    # --- Attempt A: VC for dyad ---
    last_err = None
    for re_form in re_forms:
        try:
            model = smf.mixedlm(
                "log_rt ~ role_c * cond_c",
                data=df,
                groups=df["participant_id"],
                re_formula=re_form,
                vc_formula=vc_formula_for_dyad(df),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                result = model.fit(method="lbfgs", reml=True, maxiter=2000, disp=False)
            if hasattr(result, "cov_re") and result.cov_re is not None:
                if np.any(np.diag(result.cov_re) < 1e-8):
                    raise RuntimeError("Near-singular random-effects covariance (diag≈0).")
            print(f"[INFO] Fit OK with vc_formula; re_formula={re_form}")
            return result, re_form + " + VC(dyad)"
        except TypeError as e:
            last_err = e
            break  # vc_formula not supported in this statsmodels → use FE fallback
        except Exception as e:
            last_err = e
            continue
    # --- Attempt B: dyad fixed effects ---
    for re_form in re_forms:
        try:
            model = smf.mixedlm(
                "log_rt ~ role_c * cond_c + C(dyad)",
                data=df,
                groups=df["participant_id"],
                re_formula=re_form,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                result = model.fit(method="lbfgs", reml=True, maxiter=2000, disp=False)
            if hasattr(result, "cov_re") and result.cov_re is not None:
                if np.any(np.diag(result.cov_re) < 1e-8):
                    raise RuntimeError("Near-singular random-effects covariance (diag≈0).")
            print(f"[INFO] Fit OK with dyad fixed effects; re_formula={re_form}")
            return result, re_form + " + FE(dyad)"
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All model fits failed. Last error: {last_err}")

def linear_contrast(beta, cov, L):
    """Estimate contrast L·beta and its SE."""
    est = float(L @ beta)
    se = float(np.sqrt(L @ cov @ L))
    return est, se

def planned_contrasts(result):
    """
    Compute planned contrasts on log scale and back-transform to ratios, Δ ms, and %.
    Returns: cell_table, contrast_table
    """
    params = result.fe_params
    cov = result.cov_params().loc[params.index, params.index].values
    beta = params.values
    names = list(params.index)

    br  = names.index("role_c")
    bc  = names.index("cond_c")
    brc = names.index("role_c:cond_c")

    def L_vec(coefs):
        L = np.zeros_like(beta)
        for idx, val in coefs.items():
            L[idx] = val
        return L

    # Predicted fixed-effect cell means on log scale
    def mu_log(role_c, cond_c):
        return (params["Intercept"]
                + params["role_c"] * role_c
                + params["cond_c"] * cond_c
                + params["role_c:cond_c"] * (role_c * cond_c))

    cells = {
        "Leader_MATCHING":   (-0.5, -0.5),
        "Leader_MIRRORING":  (-0.5, +0.5),
        "Follower_MATCHING": (+0.5, -0.5),
        "Follower_MIRRORING":(+0.5, +0.5),
    }
    mu_ms = {k: float(np.exp(mu_log(*v))) for k, v in cells.items()}

    # Contrasts
    L_Foll      = L_vec({bc: 1.0, brc: +0.5})  # Foll Mir−Match
    L_Lead      = L_vec({bc: 1.0, brc: -0.5})  # Lead Mir−Match
    L_RoleMatch = L_vec({br: 1.0, brc: -0.5})  # F−L in Match
    L_RoleMirr  = L_vec({br: 1.0, brc: +0.5})  # F−L in Mirror

    contrast_defs = {
        "Follower: Mirroring vs Matching": (L_Foll, "Follower_MATCHING"),
        "Leader: Mirroring vs Matching":   (L_Lead, "Leader_MATCHING"),
        "Role in Matching (Follower−Leader)": (L_RoleMatch, "Leader_MATCHING"),
        "Role in Mirroring (Follower−Leader)":(L_RoleMirr,  "Leader_MIRRORING"),
    }

    rows = []
    for label, (L, base_key) in contrast_defs.items():
        est_log, se_log = linear_contrast(beta, cov, L)
        ci_lo, ci_hi = est_log - 1.96*se_log, est_log + 1.96*se_log
        ratio, ratio_lo, ratio_hi = np.exp(est_log), np.exp(ci_lo), np.exp(ci_hi)

        base_ms = mu_ms[base_key]
        d_ms    = (ratio - 1.0) * base_ms
        d_lo    = (ratio_lo - 1.0) * base_ms
        d_hi    = (ratio_hi - 1.0) * base_ms
        pct     = 100.0 * (ratio - 1.0)
        pct_lo, pct_hi = 100.0*(ratio_lo - 1.0), 100.0*(ratio_hi - 1.0)

        rows.append({
            "Contrast": label,
            "Ratio": ratio, "Ratio 95% CI": f"[{ratio_lo:.3f}, {ratio_hi:.3f}]",
            "Δ ms": d_ms, "Δ ms 95% CI": f"[{d_lo:.1f}, {d_hi:.1f}]",
            "% change": pct, "% 95% CI": f"[{pct_lo:.1f}%, {pct_hi:.1f}%]",
            "Base (ms)": base_ms
        })

    cell_table = (pd.DataFrame({"Cell": list(mu_ms.keys()),
                                "Predicted mean (ms)": list(mu_ms.values())})
                    .sort_values("Cell"))
    contrast_table = pd.DataFrame(rows)
    return cell_table, contrast_table

def nakagawa_r2_mixedlm(result, df):
    """Approximate Nakagawa R² (marginal, conditional)."""
    X = result.model.exog
    fe = result.fe_params.values
    var_fixed = np.var(X @ fe)
    re_var = 0.0
    if hasattr(result, "cov_re") and result.cov_re is not None:
        re_var += float(np.trace(result.cov_re))
    if hasattr(result, "vcomp") and result.vcomp is not None:
        re_var += float(np.sum(result.vcomp))
    resid_var = float(result.scale)
    denom = var_fixed + re_var + resid_var
    r2_marg = var_fixed / denom if denom > 0 else np.nan
    r2_cond = (var_fixed + re_var) / denom if denom > 0 else np.nan
    return r2_marg, r2_cond

# ------------------------- Load & prepare data ---------------------------------
df = pd.read_csv(DATA_PATH)
missing = REQUIRED_COLUMNS.difference(df.columns)
if missing:
    raise ValueError(f"DATA_PATH is missing required columns: {sorted(missing)}")

OUT_DIR.mkdir(parents=True)

df = ensure_labels(df)
df = df[df["role"].isin(["Leader","Follower"]) & df["broad_condition"].isin(["Matching","Mirroring"])].copy()
n_participants = df["participant_id"].nunique()
print(f"[INFO] Participants detected: {n_participants}")
if n_participants != MIN_EXPECTED_PARTICIPANTS:
    print(f"[WARN] Expected {MIN_EXPECTED_PARTICIPANTS}, found {n_participants}. Proceeding.")

df["role_c"] = sum_code(df["role"], {"Leader": -0.5, "Follower": +0.5})
df["cond_c"] = sum_code(df["broad_condition"], {"Matching": -0.5, "Mirroring": +0.5})
df["log_rt"] = np.log(df["rt_ms"].astype(float))
df["participant_id"] = df["participant_id"].astype(str)
df["dyad"] = df["dyad"].astype(str)

# ------------------------- Fit LMM ---------------------------------------------
result, used_structure = fit_with_fallbacks(df)
print("\n=== MixedLM (trial-level log RT) ===")
print(result.summary())
print(f"\n[INFO] Random-effects used: {used_structure}")

# ------------------------- LRT for Role×Condition ------------------------------
# Refit full and reduced models with ML for the fixed-effects comparison.
try:
    # Determine whether we fitted with vc or FE dyad:
    used_fe_dyad = "+ FE(dyad)" in used_structure
    re_form_used = used_structure.split(" + ")[0].strip()  # the re_formula part we recorded

    if used_fe_dyad:
        full = smf.mixedlm("log_rt ~ role_c * cond_c + C(dyad)",
                           data=df, groups=df["participant_id"], re_formula=re_form_used)
        reduced = smf.mixedlm("log_rt ~ role_c + cond_c + C(dyad)",
                              data=df, groups=df["participant_id"], re_formula=re_form_used)
    else:
        full = smf.mixedlm("log_rt ~ role_c * cond_c",
                           data=df, groups=df["participant_id"], re_formula=re_form_used,
                           vc_formula=vc_formula_for_dyad(df))
        reduced = smf.mixedlm("log_rt ~ role_c + cond_c",
                              data=df, groups=df["participant_id"], re_formula=re_form_used,
                              vc_formula=vc_formula_for_dyad(df))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        full_res = full.fit(method="lbfgs", reml=False, maxiter=2000, disp=False)
        reduced_res = reduced.fit(method="lbfgs", reml=False, maxiter=2000, disp=False)

    ll_full, ll_red = full_res.llf, reduced_res.llf
    LRT = 2.0*(ll_full - ll_red)
    p_LRT = 1.0 - chi2.cdf(LRT, df=1)
    print(f"\n[Interaction LRT] 2ΔLL = {LRT:.3f}, df=1, p = {p_LRT:.6f}")
except Exception as e:
    print(f"\n[WARN] LRT for interaction failed: {e}")

# ------------------------- Planned contrasts -----------------------------------
cell_table, contrast_table = planned_contrasts(result)
print("\n=== Predicted cell means (fixed-effects, back-transformed) ===")
print(cell_table.to_string(index=False, formatters={"Predicted mean (ms)": "{:.2f}".format}))

print("\n=== Planned contrasts (back-transformed) ===")
print(contrast_table.to_string(index=False, formatters={
    "Ratio": "{:.3f}".format, "Δ ms": "{:.1f}".format, "% change": "{:.1f}".format, "Base (ms)": "{:.1f}".format
}))

# ------------------------- Approx. R² ------------------------------------------
r2_marg, r2_cond = nakagawa_r2_mixedlm(result, df)
print(f"\n[Model fit] Approx. Nakagawa R²: marginal = {r2_marg:.3f}, conditional = {r2_cond:.3f}")

# ------------------------- Save tables -----------------------------------------
cell_out = os.path.join(OUT_DIR, "lmm_predicted_cell_means.csv")
con_out  = os.path.join(OUT_DIR, "lmm_planned_contrasts.csv")
cell_table.to_csv(cell_out, index=False)
contrast_table.to_csv(con_out, index=False)
print(f"\n[Saved] {cell_out}")
print(f"[Saved] {con_out}")

