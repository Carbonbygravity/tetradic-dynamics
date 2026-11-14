"""Combined variance-sensitive Bayesian log-RT model across dyad cohorts.

Compares normative HC participants in HC-HC dyads (NormHC), healthy partners
in HC-SZ dyads (MixedHC), and schizophrenia patients in HC-SZ dyads (SZ).
The NormHC group is the fixed-effect reference. Exact 0.95--1.05 ratio bounds
are retained for within-role Mirroring/Matching equivalence assessments.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OVERLAY = SCRIPT_DIR / "_runtime" / "pytensor_overlay"
if OVERLAY.exists():
    sys.path.insert(0, str(OVERLAY))

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm

REFERENCE_RT_MS = 500.0
ACTORS = ["NormHC", "MixedHC", "SZ"]
ROLES = ["Leader", "Follower"]
CONDITIONS = ["MATCHING", "MIRRORING"]
FIXED = [
    "Intercept", "role_c", "cond_c", "role_c:cond_c",
    "mixed_hc_c", "mixed_hc_c:role_c", "mixed_hc_c:cond_c",
    "mixed_hc_c:role_c:cond_c", "sz_c", "sz_c:role_c",
    "sz_c:cond_c", "sz_c:role_c:cond_c",
]
SIGMA = [f"sigma_{x}" for x in FIXED]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hchc-input", type=Path,
                   required=True)
    p.add_argument("--hcsz-input", type=Path,
                   required=True)
    p.add_argument("--output-dir", type=Path,
                   required=True)
    p.add_argument("--qc-dir", type=Path,
                   required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=.99)
    p.add_argument("--max-treedepth", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--ppc-thin", type=int, default=20)
    return p.parse_args()


def prepare(hchc_path: Path, hcsz_path: Path) -> pd.DataFrame:
    a, b = pd.read_csv(hchc_path), pd.read_csv(hcsz_path)
    required = {"participant_id", "role", "condition", "rt_ms"}
    for label, df in [("HC-HC", a), ("HC-SZ", b)]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
    a = a.copy()
    a["actor_group"] = "NormHC"
    a["dyad_id"] = "HCHC_" + a["dyad_id"].astype(str).str.replace(
        "HCHC_", "", regex=False)
    a["participant_id"] = "HCHC_" + a["participant_id"].astype(str).str.replace(
        "HCHC_", "", regex=False)
    b = b.copy()
    group = b["group"].astype(str).str.strip().str.upper()
    if not set(group).issubset({"HC", "PT", "SZ"}):
        raise ValueError(f"Unexpected HC-SZ groups: {sorted(group.unique())}")
    b["actor_group"] = np.where(group.eq("HC"), "MixedHC", "SZ")
    dyad_source = b["dyad_id"] if "dyad_id" in b else b["dyad"]
    b["dyad_id"] = "HCSZ_" + dyad_source.astype(str).str.replace(
        "HC_SZ_", "", regex=False).str.replace("HCSZ_", "", regex=False)
    b["participant_id"] = "HCSZ_" + b["participant_id"].astype(str).str.replace(
        "HC_SZ_", "", regex=False).str.replace("HCSZ_", "", regex=False)
    df = pd.concat([a, b], ignore_index=True, sort=False)
    df["role"] = df["role"].astype(str).str.strip().str.title()
    df["condition"] = df["condition"].astype(str).str.strip().str.upper()
    df["rt_ms"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    if df.rt_ms.isna().any() or df.rt_ms.le(0).any():
        raise ValueError("RT must be positive and nonmissing.")
    if set(df.role) != set(ROLES) or set(df.condition) != set(CONDITIONS):
        raise ValueError("Unexpected role or condition labels.")
    df["role_c"] = df.role.map({"Leader": -.5, "Follower": .5})
    df["cond_c"] = df.condition.map({"MATCHING": -.5, "MIRRORING": .5})
    df["mixed_hc_c"] = df.actor_group.eq("MixedHC").astype(float)
    df["sz_c"] = df.actor_group.eq("SZ").astype(float)
    df["log_rt_centered"] = np.log(df.rt_ms) - np.log(REFERENCE_RT_MS)
    counts = df.groupby(["actor_group", "role", "condition"]).size()
    if len(counts) != 12 or (counts == 0).any():
        raise ValueError(f"Incomplete 3 x 2 x 2 design:\n{counts}")
    overlap = set(a.participant_id) & set(b.participant_id)
    if overlap:
        raise ValueError(f"Participant ID overlap after prefixing: {sorted(overlap)[:5]}")
    return df


def summarize(x):
    lo, med, hi = np.quantile(x, [.025, .5, .975])
    return {"median": float(med), "hdi95_lo": float(lo),
            "hdi95_hi": float(hi), "p_gt_0": float(np.mean(x > 0)),
            "p_lt_0": float(np.mean(x < 0))}


def draws_dict(idata, names):
    post = az.extract(idata, var_names=names, combined=True)
    return {n: np.asarray(post[n]).reshape(-1) for n in names}


def linear_predictor(d, actor, role, condition, prefix=""):
    r = -.5 if role == "Leader" else .5
    c = -.5 if condition == "MATCHING" else .5
    m, s = float(actor == "MixedHC"), float(actor == "SZ")
    def v(name): return d[prefix + name]
    return (v("Intercept") + v("role_c")*r + v("cond_c")*c
            + v("role_c:cond_c")*r*c + v("mixed_hc_c")*m
            + v("mixed_hc_c:role_c")*m*r
            + v("mixed_hc_c:cond_c")*m*c
            + v("mixed_hc_c:role_c:cond_c")*m*r*c + v("sz_c")*s
            + v("sz_c:role_c")*s*r + v("sz_c:cond_c")*s*c
            + v("sz_c:role_c:cond_c")*s*r*c)


def posterior_results(idata):
    d = draws_dict(idata, FIXED)
    cells, cell_rows = {}, []
    for actor in ACTORS:
        for role in ROLES:
            for condition in CONDITIONS:
                log_mu = linear_predictor(d, actor, role, condition) + np.log(REFERENCE_RT_MS)
                cells[(actor, role, condition)] = log_mu
                cell_rows.append({"actor_group": actor, "role": role,
                                  "condition": condition,
                                  "estimand": "population conditional geometric mean RT (ms)",
                                  **summarize(np.exp(log_mu))})
    effects, rows = {}, []
    rope_lo, rope_hi = np.log(.95), np.log(1.05)
    for actor in ACTORS:
        for role in ROLES:
            x = cells[(actor, role, "MIRRORING")] - cells[(actor, role, "MATCHING")]
            effects[(actor, role)] = x
            for suffix, val, scale in [
                ("dlog", x, "log ratio"), ("ratio", np.exp(x), "Mirroring/Matching ratio"),
                ("percent", 100*(np.exp(x)-1), "percent change"),
                ("dms", np.exp(cells[(actor, role, "MIRRORING")])-
                 np.exp(cells[(actor, role, "MATCHING")]), "conditional geometric-mean difference (ms)"),
            ]:
                row = {"contrast": f"{actor}_{role}_Mirroring_minus_Matching_{suffix}",
                       "scale": scale, **summarize(val)}
                if suffix == "dlog":
                    lo, hi = np.quantile(val, [.025, .975])
                    row.update({"rope_ratio_lo": .95, "rope_ratio_hi": 1.05,
                                "p_in_exact_rope": float(np.mean((val >= rope_lo) & (val <= rope_hi))),
                                "hdi95_entirely_in_exact_rope": bool(lo >= rope_lo and hi <= rope_hi)})
                rows.append(row)
        dod = effects[(actor, "Follower")] - effects[(actor, "Leader")]
        effects[(actor, "DoD")] = dod
        rows.append({"contrast": f"{actor}_Role_by_Condition_DoD",
                     "scale": "log ratio difference", **summarize(dod)})
    for actor, reference in [
        ("MixedHC", "NormHC"), ("SZ", "NormHC"), ("SZ", "MixedHC")
    ]:
        for key in ["Leader", "Follower", "DoD"]:
            x = effects[(actor, key)] - effects[(reference, key)]
            rows.append({"contrast": f"{actor}_minus_{reference}_{key}_effect",
                         "scale": "log ratio difference", **summarize(x)})
        for role in ROLES:
            for condition in CONDITIONS:
                x = cells[(actor, role, condition)] - cells[(reference, role, condition)]
                rows.append({"contrast": f"{actor}_over_{reference}_{role}_{condition}_ratio",
                             "scale": "RT ratio", **summarize(np.exp(x))})
    return pd.DataFrame(cell_rows), pd.DataFrame(rows)


def sigma_cells(idata):
    d = draws_dict(idata, SIGMA)
    d = {k.removeprefix("sigma_"): v for k, v in d.items()}
    rows = []
    for actor in ACTORS:
        for role in ROLES:
            for condition in CONDITIONS:
                x = np.exp(linear_predictor(d, actor, role, condition))
                q = np.quantile(x, [.025, .5, .975])
                rows.append({"actor_group": actor, "role": role, "condition": condition,
                             "residual_log_rt_sd_median": float(q[1]),
                             "residual_log_rt_sd_hdi95_lo": float(q[0]),
                             "residual_log_rt_sd_hdi95_hi": float(q[2])})
    return pd.DataFrame(rows)


def diagnostics(idata):
    fixed = az.summary(idata, var_names=FIXED, hdi_prob=.95)
    stats, bfmi = idata.sample_stats, np.asarray(az.bfmi(idata), float)
    out = {"divergences": int(stats.diverging.sum()),
           "maximum_tree_depth_observed": int(stats.tree_depth.max()),
           "draws_reported_at_maximum_tree_depth": int(stats.reached_max_treedepth.sum()),
           "bfmi_by_chain": bfmi.tolist(), "bfmi_min": float(bfmi.min()),
           "fixed_rhat_max": float(fixed.r_hat.max()),
           "fixed_ess_bulk_min": float(fixed.ess_bulk.min()),
           "fixed_ess_tail_min": float(fixed.ess_tail.min())}
    return fixed.reset_index(names="parameter"), out


def ppc_summary(model, idata, data, thin):
    ppc = idata.sel(draw=slice(None, None, max(1, thin))).copy()
    model.predict(ppc, kind="response", inplace=True)
    y = REFERENCE_RT_MS * np.exp(np.asarray(ppc.posterior_predictive.log_rt_centered))
    y = y.reshape(-1, y.shape[-1]); obs = data.rt_ms.to_numpy()
    masks = [("overall", np.ones(len(data), bool))]
    masks += [(f"{a}_{r}_{c}", (data.actor_group.eq(a)&data.role.eq(r)&data.condition.eq(c)).to_numpy())
              for a in ACTORS for r in ROLES for c in CONDITIONS]
    rows = []
    for label, mask in masks:
        o, p = obs[mask], y[:, mask]
        specs = [("mean", np.mean(o), np.mean(p,1)), ("sd", np.std(o,ddof=1), np.std(p,1,ddof=1)),
                 ("q10", np.quantile(o,.1), np.quantile(p,.1,axis=1)),
                 ("median", np.median(o), np.median(p,axis=1)),
                 ("q90", np.quantile(o,.9), np.quantile(p,.9,axis=1))]
        for stat, oval, pred in specs:
            lo, med, hi = np.quantile(pred,[.025,.5,.975])
            rows.append({"cell":label,"statistic":stat,"observed":float(oval),
                         "predicted_median":float(med),"predicted_hdi95_lo":float(lo),
                         "predicted_hdi95_hi":float(hi),
                         "observed_inside_predictive_interval":bool(lo<=oval<=hi)})
    return pd.DataFrame(rows)


def main():
    args = parse_args(); tag = "pilot" if args.pilot else "primary_variance_ta099"
    out, qc = args.output_dir/tag, args.qc_dir/tag
    out.mkdir(parents=True,exist_ok=True); qc.mkdir(parents=True,exist_ok=True)
    draws = 250 if args.pilot else args.draws; tune = 250 if args.pilot else args.tune
    chains = 2 if args.pilot else args.chains; cores = 1 if args.pilot else min(chains,args.cores)
    thin = 10 if args.pilot else args.ppc_thin
    data = prepare(args.hchc_input,args.hcsz_input)
    base = "role_c * cond_c + mixed_hc_c * role_c * cond_c + sz_c * role_c * cond_c"
    mean_formula = (f"log_rt_centered ~ {base} + (1|dyad_id) + (1|participant_id) + "
                    "(0+role_c|participant_id) + (0+cond_c|participant_id) + "
                    "(0+role_c:cond_c|participant_id)")
    sigma_formula = f"sigma ~ {base}"
    priors = {name:bmb.Prior("Normal",mu=0,sigma=.3) for name in FIXED if name!="Intercept"}
    priors["Intercept"] = bmb.Prior("Normal",mu=0,sigma=.5)
    for name in SIGMA:
        priors[name] = bmb.Prior("Normal",mu=(-1 if name=="sigma_Intercept" else 0),
                                 sigma=(.5 if name=="sigma_Intercept" else .3))
    model = bmb.Model(bmb.Formula(mean_formula,sigma_formula),data=data,family="gaussian",priors=priors)
    idata = model.fit(draws=draws,tune=tune,chains=chains,cores=cores,
                      target_accept=args.target_accept,random_seed=args.seed,
                      idata_kwargs={"log_likelihood":True},nuts={"max_treedepth":args.max_treedepth})
    idata.to_netcdf(out/"posterior.nc")
    fixed, sampler = diagnostics(idata); fixed.to_csv(out/"mean_fixed_effect_summary.csv",index=False)
    az.summary(idata,var_names=SIGMA,hdi_prob=.95).reset_index(names="parameter").to_csv(
        out/"sigma_fixed_effect_summary.csv",index=False)
    cells, contrasts = posterior_results(idata); cells.to_csv(out/"population_cell_estimates.csv",index=False)
    contrasts.to_csv(out/"planned_posterior_contrasts.csv",index=False)
    sigma_cells(idata).to_csv(out/"residual_sigma_cell_estimates.csv",index=False)
    ppc_summary(model,idata,data,thin).to_csv(qc/"posterior_predictive_summary.csv",index=False)
    with (qc/"sampler_diagnostics.json").open("w") as f: json.dump(sampler,f,indent=2)
    meta={"hchc_input":str(args.hchc_input.resolve()),"hcsz_input":str(args.hcsz_input.resolve()),
          "rows":len(data),"dyads":int(data.dyad_id.nunique()),
          "participants":int(data.participant_id.nunique()),
          "actor_group_counts":data.groupby("actor_group").size().to_dict(),
          "mean_formula":mean_formula,"sigma_formula":sigma_formula,"draws":draws,"tune":tune,
          "chains":chains,"cores":cores,"target_accept":args.target_accept,
          "max_treedepth":args.max_treedepth,"seed":args.seed,"exact_rope_ratio":[.95,1.05]}
    with (qc/"model_metadata.json").open("w") as f: json.dump(meta,f,indent=2)
    with (qc/"package_versions.json").open("w") as f: json.dump({"python":sys.version,
        "platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
        "arviz":az.__version__,"bambi":bmb.__version__,"pymc":pm.__version__},f,indent=2)
    print(json.dumps({"metadata":meta,"diagnostics":sampler},indent=2))


if __name__ == "__main__":
    main()
