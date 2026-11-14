"""Verify the approved 19-target handoff against newly finalized primary tiers.

This script does not calculate PhiID or select targets. It compares the
direction-aware final tier outputs with the preserved target inventory, then
copies that inventory to the historical handoff path without overwriting it.
"""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parent
PRIVATE_INVENTORY_SHA256 = "ebb2f292748303ab37af29e300159106efa40a9325079f17062f57d9754b7cb6"
DESTINATION = HERE / "frozen_targets.json"
SOURCES = (
    "PhiID_Normative_Controls_2026-10-26",
    "PhiID_Normative_Interaction_Controls_2026-10-26",
)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-targets", type=Path, required=True,
        help="Original private 19-target inventory with dyad-level values",
    )
    inventory_bytes = parser.parse_args().frozen_targets.read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != PRIVATE_INVENTORY_SHA256:
        raise ValueError("Expected the original private 19-target inventory")
    frozen = json.loads(inventory_bytes)
    assert len(frozen) == 19
    expected = {item["uid"]: item for item in frozen}
    assert len(expected) == len(frozen), "Duplicate frozen target UID"

    actual = {}
    counts = {}
    for folder_name in SOURCES:
        path = WORKFLOW / folder_name / "directional_tiers.json"
        rows = read_json(path)
        selected = [row for row in rows if row["tier"] in ("A", "B")]
        counts[folder_name] = len(selected)
        for row in selected:
            item = dict(row)
            item["source"] = f"{folder_name}/directional_tiers.json"
            item["uid"] = f"{folder_name}:{row['id']}"
            assert item["uid"] not in actual, "Duplicate generated target UID"
            actual[item["uid"]] = item

    assert counts == {
        SOURCES[0]: 16,
        SOURCES[1]: 3,
    }, f"Unexpected Tier A/B target counts: {counts}"
    assert actual == expected, "Finalized Tier A/B targets differ from frozen inventory"
    assert not DESTINATION.exists(), f"Refusing to overwrite {DESTINATION}"
    with DESTINATION.open("xb") as stream:
        stream.write(inventory_bytes)
    assert digest(DESTINATION) == PRIVATE_INVENTORY_SHA256
    print(json.dumps({
        "verified_targets": len(actual),
        "main_targets": counts[SOURCES[0]],
        "interaction_targets": counts[SOURCES[1]],
        "sha256": digest(DESTINATION),
        "destination": str(DESTINATION),
    }))


if __name__ == "__main__":
    main()
