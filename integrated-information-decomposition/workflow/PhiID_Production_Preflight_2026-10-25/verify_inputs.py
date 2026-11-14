"""Compare a freshly hashed runtime input manifest with frozen source hashes.

This verifies file identity and cohort membership only. It does not establish
participant provenance, source reconstruction validity, or PhiID estimates.
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "input_manifest.example.json",
    )
    args = parser.parse_args()
    runtime = read_json(args.runtime_manifest)
    frozen = read_json(args.frozen_manifest)
    observed = {}
    for row in runtime["inputs"]:
        key = (int(row["dyad"]), int(row["partner"]))
        assert key not in observed, f"Duplicate runtime input {key}"
        observed[key] = (
            row["relative_path"],
            int(row["size_bytes"]),
            row["sha256"].lower(),
        )
    expected = {}
    for row in frozen["inputs"]:
        key = (int(row["dyad"]), int(row["partner"]))
        assert key not in expected, f"Duplicate frozen input {key}"
        expected[key] = (
            row["path"],
            int(row["size_bytes"]),
            row["sha256"].lower(),
        )
    assert len(expected) == 30, f"Expected 30 frozen inputs, found {len(expected)}"
    assert observed == expected, "Runtime source hashes, sizes or membership differ from frozen manifest"
    print(json.dumps({"status": "passed", "files": len(observed), "total_bytes": sum(row[1] for row in observed.values())}))


if __name__ == "__main__":
    main()
