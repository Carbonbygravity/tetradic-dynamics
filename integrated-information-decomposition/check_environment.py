"""Read-only check of the runtime recorded by the normative PhiID run."""

import hashlib
import sys
from pathlib import Path

import h5py
import mne
import numpy
import scipy
from phyid import calculate

EXPECTED_VERSIONS = {
    "python": "3.12.3",
    "numpy": "1.26.4",
    "scipy": "1.15.2",
    "mne": "1.9.0",
    "h5py": "3.11.0",
}
EXPECTED_PHYID_HASHES = {
    "calculate.py": "b7aa23ff11c9b17587be3d537789cfc034a2a154e20daf101eb557a028d165f1",
    "measures.py": "074cb1d9d4f0317607946f34aa4c61b1696612ada9d1b217131db50c6f859ee8",
    "utils.py": "bf7b5d41b9a48e3aa8c7eecc3d9b9c0a04cdfc20d6043be92c489f6c8e3ee0d6",
    "_version.py": "0d5fb7b0b8d7eb591a7fda26639ea6909b0073d6a5a4822ac17f16358c1cd3ea",
    "__init__.py": "c17bdc01ba3b83e87bfc28a1e9c36e86882a2a6bb2123a3fd4230aac4f5db58d",
}


def main():
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mne": mne.__version__,
        "h5py": h5py.__version__,
    }
    root = Path(calculate.__file__).resolve().parent
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in EXPECTED_PHYID_HASHES
    }
    for key, expected in EXPECTED_VERSIONS.items():
        if actual[key] != expected:
            raise RuntimeError(f"{key}: expected {expected}, got {actual[key]}")
    for name, expected in EXPECTED_PHYID_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(f"phyid/{name}: source hash differs")
    print("Recorded core runtime versions and phyid source hashes match.")


if __name__ == "__main__":
    main()
