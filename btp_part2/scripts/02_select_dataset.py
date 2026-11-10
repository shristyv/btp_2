"""Phase 1 Step 2: select metadata-diverse frames from locally cached BDD100K JSON annotations.

Compatible with the updated common.py that provides ROOT, config, candidate_boxes,
parse_annotation, and local_label_path. No images are downloaded by this script.

Original BDD train -> train/internal validation; original BDD val -> held-out test.
The selection deliberately favors environment diversity and uncommon classes.
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

from common import ROOT, config, candidate_boxes, parse_annotation

CLASSES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]


def load_cached_annotations(cfg):
    base = ROOT / "data" / "raw" / "labels"
    files = sorted(base.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No cached JSON annotations found in {base}. Run 01_audit_metadata.py first.")

    frames = []
    for file in files:
        # Current common.py caches remote paths as flattened names, for example:
        # data_labels_100k_train_shard_000_<image-id>.json
        # Also support paths in nested folders.
        normalized = str(file.relative_to(base)).replace("\\", "/")
        tokens = normalized.replace("/", "_")
        if "labels_100k_train_shard_" in tokens:
            original_split = "train"
        elif "labels_100k_val_shard_" in tokens:
            original_split = "val"
        else:
            continue
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            parsed = parse_annotation(record)
            name = parsed["name"] or file.stem.split("_")[-1]
            boxes = candidate_boxes(record, cfg.get("min_box_side", 20))
            if not boxes:
                continue
            # Determine the shard from the local filename (or directory structure).
            import re
            match = re.search(r"shard_\d{3}", tokens)
            if not match:
                continue
            shard = match.group(0)
            box_items = []
            for index, box in enumerate(boxes):
                box_items.append({
                    "box_index": index,
                    "class": box["class"],
                    "x": float(box["x1"]), "y": float(box["y1"]),
                    "w": float(box["x2"] - box["x1"]),
                    "h": float(box["y2"] - box["y1"]),
                })
            frames.append({
                "source_split": original_split,
                "source_image": f"{name}.jpg" if not name.endswith(".jpg") else name,
                "weather": parsed["weather"],
                "timeofday": parsed["timeofday"],
                "scene": parsed["scene"],
                "remote_path": f"data/images_100k/100k/{original_split}/{shard}/{name}.jpg" if not name.endswith(".jpg") else f"data/images_100k/100k/{original_split}/{shard}/{name}",
                "boxes": box_items,
            })
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print(f"Skipping unreadable annotation {file.name}: {exc}")
    # Avoid accidental duplicate records. Frame IDs are unique per original split.
    unique = {}
    for row in frames:
        unique[(row["source_split"], row["source_image"])] = row
    return list(unique.values())


def env_group(row):
    w = str(row["weather"]).lower()
    t = str(row["timeofday"]).lower()
    if w in {"rainy", "snowy", "foggy"}:
        return "adverse_weather"
    if t == "night":
        return "night"
    if t == "dawn/dusk":
        return "dawn_dusk"
    if w in {"overcast", "partly cloudy"}:
        return "cloudy"
    if t == "daytime" and w == "clear":
        return "day_clear"
    return "other"


def select_frames(pool, crop_target, rng, max_per_image=8, reserved=None):
    """Greedy, coverage-aware selection; each original image stays in exactly one split."""
    remaining = [r for r in pool if (r["source_split"], r["source_image"]) not in (reserved or set())]
    rng.shuffle(remaining)
    counts_class = Counter()
    counts_env = Counter()
    chosen = []
    crop_count = 0
    while remaining and crop_count < crop_target:
        # Subsample candidates per iteration so the 550-frame pilot stays fast.
        # The reward penalizes already common classes/environments.
        candidates = remaining if len(remaining) < 1000 else rng.sample(remaining, 1000)
        def score(frame):
            box_counts = Counter(b["class"] for b in frame["boxes"])
            rare_reward = sum(min(v, 2) / (1.0 + counts_class[c])**0.65 for c, v in box_counts.items())
            env_reward = 3.0 / (1.0 + counts_env[frame["environment"]])**0.65
            return rare_reward + env_reward + rng.random() * 0.0001
        pick = max(candidates, key=score)
        remaining.remove(pick)
        # Select rarer classes within a frame first, but keep <= max_per_image.
        boxes = sorted(pick["boxes"], key=lambda b: (counts_class[b["class"]], rng.random()))
        take = min(max_per_image, len(boxes), crop_target - crop_count)
        if not take:
            break
        selected = dict(pick)
        selected["boxes"] = boxes[:take]
        selected["expected_crops"] = take
        chosen.append(selected)
        counts_class.update(b["class"] for b in selected["boxes"])
        counts_env[selected["environment"]] += take
        crop_count += take
    return chosen, counts_class, counts_env, crop_count


def main():
    cfg = config()
    rng = random.Random(cfg.get("seed", 42))
    rows = load_cached_annotations(cfg)
    for r in rows:
        r["environment"] = env_group(r)

    print(f"Cached usable source frames: {len(rows)}")
    print("Environment availability:", dict(Counter(r["environment"] for r in rows)))
    print("Note: audit sampled the FIRST 50 labels per shard; pilot results are not representative of full BDD100K.")

    train_pool = [r for r in rows if r["source_split"] == "train"]
    test_pool = [r for r in rows if r["source_split"] == "val"]
    if not train_pool or not test_pool:
        raise RuntimeError("Need cached JSONs from both original train and val partitions. Run script 01 first.")

    targets = cfg.get("target_crops", {"train": 300, "val": 60, "test": 60})
    max_per = cfg.get("max_crops_per_image", 8)
    reserved = set()
    selected = []
    summary = []
    # Internal validation before train; preserves some variety for model selection.
    for final_split, pool in [("val", train_pool), ("train", train_pool), ("test", test_pool)]:
        chosen, cls, env, crops = select_frames(pool, targets[final_split], rng, max_per, reserved)
        for item in chosen:
            item["split"] = final_split
            reserved.add((item["source_split"], item["source_image"]))
        selected += chosen
        print(f"{final_split}: {len(chosen)} frames; {crops}/{targets[final_split]} planned crops")
        print("  classes:", dict(cls))
        print("  environments:", dict(env))
        for key, count in cls.items():
            summary.append({"split": final_split, "group": "class", "value": key, "count": count})
        for key, count in env.items():
            summary.append({"split": final_split, "group": "environment", "value": key, "count": count})

    out = ROOT / "data" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "selected_frames.jsonl").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    results = ROOT / "results" / "phase1"
    results.mkdir(parents=True, exist_ok=True)
    with (results / "selection_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "group", "value", "count"])
        writer.writeheader()
        writer.writerows(summary)
    print("Saved data/manifests/selected_frames.jsonl")
    print("Saved results/phase1/selection_summary.csv")
    print("No JPEGs downloaded. Next step: update 03_extract_crops.py for this manifest format.")

if __name__ == "__main__":
    main()
