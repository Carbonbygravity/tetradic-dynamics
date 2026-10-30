# ΦID environment

The recorded production specification identifies Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2, MNE 1.9.0, and h5py 3.11.0. `environment.yml` pins these versions and additional runtime dependencies. The analysis uses `phyid` revision `f7fe791217ef5ee16cf9222a3ed089e66f423a1f` of [integrated-info-decomp](https://github.com/Imperial-MIND-lab/integrated-info-decomp). `check_environment.py` checks the recorded package versions and five `phyid` source-file hashes.

From `integrated-information-decomposition/`:

```powershell
conda env create -f environment.yml
conda activate normative-phiid
python -m pip install --no-deps "git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git@f7fe791217ef5ee16cf9222a3ed089e66f423a1f"
python check_environment.py
```

This installation sequence passed in a fresh Windows Conda environment. A match to the recorded package versions and source hashes establishes the tested software configuration only; it does not reproduce the reported scientific results. The 30 source HDF5 inputs must be obtained separately and checked against the example manifest before computation.
