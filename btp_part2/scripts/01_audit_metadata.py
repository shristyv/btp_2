"""
Download annotation JSONs only and audit:
- frame counts
- environment counts
- valid object-box counts
"""

import json
from collections import Counter

import pandas as pd

from common import (
    ROOT,
    config,
    BudgetSession,
    REPOS,
    list_shards,
    list_files_in_shard,
    local_label_path,
    candidate_boxes,
    parse_annotation,
)

def main():
    cfg = config()
    dl = BudgetSession(cfg)

    rows = []
    weather_counter = Counter()
    scene_counter = Counter()
    tod_counter = Counter()
    class_counter = Counter()

    splits = ["train", "val"]   # enough for audit; skip test for now

    for split in splits:
        print(f"\nScanning split: {split}")

        shard_dirs = list_shards(split, kind="labels")
        print(f"Found {len(shard_dirs)} shards")

        for shard_dir in shard_dirs:
            files = list_files_in_shard(shard_dir, limit=200)
            json_files = [f for f in files if f.endswith(".json")]

            print(f"  {shard_dir} -> {len(json_files)} json files (checking sampled subset)")

            # sample only first 50 per shard for fast audit
            for remote in json_files[:50]:
                dest = local_label_path(remote)

                try:
                    dl.get(REPOS["labels"], remote, dest)
                except Exception as ex:
                    raise RuntimeError(f"Failed label file {remote}: {ex}") from ex

                with open(dest, "r", encoding="utf-8") as f:
                    record = json.load(f)

                parsed = parse_annotation(record)
                boxes = candidate_boxes(record, cfg["min_box_side"])

                weather_counter[parsed["weather"]] += 1
                scene_counter[parsed["scene"]] += 1
                tod_counter[parsed["timeofday"]] += 1

                for b in boxes:
                    class_counter[b["class"]] += 1

                rows.append({
                    "split": split,
                    "name": parsed["name"],
                    "weather": parsed["weather"],
                    "scene": parsed["scene"],
                    "timeofday": parsed["timeofday"],
                    "valid_boxes": len(boxes),
                    "classes_present": ",".join(sorted(set(b["class"] for b in boxes))),
                })

    out_dir = ROOT / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows collected. Check internet or repository paths.")
        return

    df.to_csv(out_dir / "annotation_audit_detailed.csv", index=False)

    summary = df.groupby("split", as_index=False).agg(
        frames=("name", "size"),
        valid_boxes=("valid_boxes", "sum")
    )
    summary.to_csv(out_dir / "annotation_audit.csv", index=False)

    pd.DataFrame(weather_counter.items(), columns=["weather", "count"]).to_csv(
        out_dir / "weather_counts.csv", index=False
    )
    pd.DataFrame(scene_counter.items(), columns=["scene", "count"]).to_csv(
        out_dir / "scene_counts.csv", index=False
    )
    pd.DataFrame(tod_counter.items(), columns=["timeofday", "count"]).to_csv(
        out_dir / "timeofday_counts.csv", index=False
    )
    pd.DataFrame(class_counter.items(), columns=["class", "count"]).to_csv(
        out_dir / "class_counts.csv", index=False
    )

    print("\n=== AUDIT COMPLETE ===")
    print(summary)
    print(f"\nMetadata transferred so far: {dl.spent/1e6:.2f} MB")
    print(f"Saved results in: {out_dir}")


if __name__ == "__main__":
    main()