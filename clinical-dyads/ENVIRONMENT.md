# Analysis environments

Use a separate environment for each workflow. The following versions have different sources.

| Workflow | Recorded packages and requirements |
|---|---|
| Behavioural | Python 3.11.11, NumPy 1.26.4, pandas 2.2.2, ArviZ 0.21.0, Bambi 0.15.0 and PyMC 5.19.0. The measured-only fit used a PyTensor 2.26.4 overlay with h5py 3.12.1 and h5netcdf 1.4.1, a working C++ compiler and a short writable compilation-cache path. |
| Tonic | The model specification records NumPy 1.26.4, pandas 2.2.2, SciPy 1.15.1 and statsmodels 0.13.5. Keep the nested-intercept engine alongside the clinical fit script. |
| Phasic | MNE, NumPy, h5py, SciPy and threadpoolctl. For the historical participant analysis, Python 3.12.3, MNE 1.9.0, NumPy 1.26.4, h5py 3.11.0 and SciPy 1.15.2 were reconstructed from historical environment evidence. Use the paired-run specification to identify the corresponding package versions for the paired analysis. |
| Clinical associations | Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2, pandas 2.2.3 and an Excel reader compatible with pandas. Use one numerical thread and retain the source RNG stream assignments. |
| ΦID production | The clinical specification records Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2 and h5py 3.11.0. The actor mapping and pinned estimator implementation are additional dependencies. |

Posterior predictive draws in the original behavioural fit had no explicit posterior predictive seed. Compare predictive distributions rather than requiring identical predictive draws. These records do not constitute a tested cross-platform environment lock.
