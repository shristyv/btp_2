
"""
BTP Part 2 - Phase 1, Step 3
Download selected BDD100K images and extract object crops.

Input:
    data/manifests/selected_frames.jsonl

Output:
    data/processed/crops/{train,val,test}/
    data/manifests/all_crops.csv
    data/manifests/image_download_ledger.json
"""

import csv
import json
import math
import os
import time

from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError

from common import ROOT, config


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_URL = (
    "https://huggingface.co/datasets/"
    "dronefreak/BDD100K/resolve/main/"
)

FIELDS = [
    "crop_id",
    "crop_path",
    "split",
    "source_image",
    "source_split",
    "class",
    "weather",
    "timeofday",
    "scene",
    "environment",
    "box_index",
    "x",
    "y",
    "w",
    "h",
]


# --------------------------------------------------
# READ SELECTED FRAMES
# --------------------------------------------------

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------
# SAVE AND LOAD CROP MANIFEST
# --------------------------------------------------

def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".csv.tmp")

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in FIELDS
            })

    os.replace(temp_path, path)


def load_existing(path):
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------
# GENERATE UNIQUE CROP ID
# --------------------------------------------------

def make_id(item, box):
    return (
        f"{item['source_split']}_"
        f"{Path(item['source_image']).stem}_"
        f"{int(box['box_index'])}"
    )


# --------------------------------------------------
# DOWNLOAD BUDGET
# --------------------------------------------------

def cached_label_bytes():
    label_dir = ROOT / "data" / "raw" / "labels"

    if not label_dir.exists():
        return 0

    return sum(
        p.stat().st_size
        for p in label_dir.rglob("*.json")
    )


def used_bytes(ledger):
    return int(ledger.get("download_bytes", 0))


def save_ledger(path, ledger):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(ledger, indent=2),
        encoding="utf-8"
    )


# --------------------------------------------------
# DOWNLOAD ONE IMAGE
# --------------------------------------------------

def download_image(
    session,
    remote_path,
    destination,
    ledger,
    ledger_path,
    cap_bytes
):
    """
    Download one JPEG using streaming.
    Record received bytes and enforce the budget.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    url = BASE_URL + quote(
        remote_path,
        safe="/"
    )

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    # Do not reuse incomplete files.
    temp_path.unlink(missing_ok=True)

    with session.get(
        url,
        stream=True,
        timeout=(15, 60),
        allow_redirects=True
    ) as response:

        response.raise_for_status()

        declared_size = response.headers.get(
            "Content-Length"
        )

        if declared_size:
            if (
                used_bytes(ledger) + int(declared_size)
                > cap_bytes
            ):
                raise StopIteration(
                    "Download budget would be exceeded."
                )

        try:
            with temp_path.open("wb") as f:

                for chunk in response.iter_content(
                    chunk_size=64 * 1024
                ):

                    if not chunk:
                        continue

                    ledger["download_bytes"] = (
                        used_bytes(ledger) + len(chunk)
                    )

                    if used_bytes(ledger) > cap_bytes:
                        raise StopIteration(
                            "Download budget reached."
                        )

                    f.write(chunk)

        finally:
            save_ledger(ledger_path, ledger)

    temp_path.replace(destination)


# --------------------------------------------------
# MAIN DATASET EXTRACTION
# --------------------------------------------------

def main():

    cfg = config()

    selection_path = (
        ROOT / "data" / "manifests"
        / "selected_frames.jsonl"
    )

    manifest_path = (
        ROOT / "data" / "manifests"
        / "all_crops.csv"
    )

    ledger_path = (
        ROOT / "data" / "manifests"
        / "image_download_ledger.json"
    )

    if not selection_path.exists():
        raise FileNotFoundError(
            "selected_frames.jsonl not found. "
            "Run 02_select_dataset.py first."
        )

    # Maximum 850 MB overall allowance.
    cap_mb = min(
        float(cfg.get("download_limit_mb", 850)),
        850.0
    )

    cap_bytes = int(cap_mb * 1_000_000)

    if ledger_path.exists():
        ledger = json.loads(
            ledger_path.read_text(encoding="utf-8")
        )
    else:
        ledger = {"download_bytes": 0}

    # Reserve space for previously downloaded labels.
    ledger["metadata_bytes_reserved"] = (
        cached_label_bytes()
    )

    # Resume only crops that exist.
    previous_rows = load_existing(manifest_path)

    rows = [
        row
        for row in previous_rows
        if (ROOT / row["crop_path"]).is_file()
    ]

    completed_ids = {
        row["crop_id"]
        for row in rows
    }

    frames = list(load_jsonl(selection_path))

    session = requests.Session()

    session.headers.update({
        "User-Agent": "BTP-Research-Subset-Builder/1.0"
    })

    failed = []
    counters = Counter()

    max_dim = int(
        cfg.get("crop_max_side", 160)
    )

    jpeg_quality = int(
        cfg.get("jpeg_quality", 85)
    )

    min_side = int(
        cfg.get("min_box_side", 20)
    )

    keep_raw = bool(
        cfg.get("keep_raw_images", False)
    )

    print("\n====================================")
    print(" BTP PART 2 - OBJECT CROP EXTRACTION")
    print("====================================")

    print("Selected frames:", len(frames))
    print("Download limit:", cap_mb, "MB")

    print(
        "Cached annotation size:",
        round(
            ledger["metadata_bytes_reserved"] / 1e6,
            2
        ),
        "MB"
    )

    # ----------------------------------------------
    # PROCESS ONE FRAME AT A TIME
    # ----------------------------------------------

    for index, item in enumerate(frames, start=1):

        required = [
            (box, make_id(item, box))
            for box in item["boxes"]
        ]

        # Skip already completed frames.
        if all(
            crop_id in completed_ids
            for _, crop_id in required
        ):
            counters["already_completed"] += 1
            continue

        if (
            ledger["metadata_bytes_reserved"]
            + used_bytes(ledger)
            >= cap_bytes
        ):
            print("STOP: Download budget reached.")
            break

        raw_path = (
            ROOT
            / "data"
            / "raw"
            / "images"
            / item["source_split"]
            / item["source_image"]
        )

        try:

            # Download image if not already cached.
            if not raw_path.is_file():

                download_image(
                    session,
                    item["remote_path"],
                    raw_path,
                    ledger,
                    ledger_path,
                    cap_bytes - ledger[
                        "metadata_bytes_reserved"
                    ]
                )

            # Open image and extract selected boxes.
            with Image.open(raw_path) as image:

                rgb = image.convert("RGB")

                for box, crop_id in required:

                    if crop_id in completed_ids:
                        continue

                    x = float(box["x"])
                    y = float(box["y"])
                    w = float(box["w"])
                    h = float(box["h"])

                    values = (x, y, w, h)

                    if not all(
                        math.isfinite(v)
                        for v in values
                    ):
                        continue

                    # Clip bounding box to image borders.
                    left = max(0, math.floor(x))
                    top = max(0, math.floor(y))

                    right = min(
                        rgb.width,
                        math.ceil(x + w)
                    )

                    bottom = min(
                        rgb.height,
                        math.ceil(y + h)
                    )

                    if (
                        right - left < min_side
                        or bottom - top < min_side
                    ):

                        counters[
                            "invalid_after_clipping"
                        ] += 1

                        continue

                    crop = rgb.crop((
                        left,
                        top,
                        right,
                        bottom
                    ))

                    # Resize while keeping aspect ratio.
                    crop.thumbnail(
                        (max_dim, max_dim),
                        Image.Resampling.LANCZOS
                    )

                    output_path = (
                        ROOT
                        / "data"
                        / "processed"
                        / "crops"
                        / item["split"]
                        / f"{crop_id}.jpg"
                    )

                    output_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    crop.save(
                        output_path,
                        format="JPEG",
                        quality=jpeg_quality
                    )

                    # Record sample information.
                    rows.append({
                        "crop_id": crop_id,

                        "crop_path": (
                            output_path
                            .relative_to(ROOT)
                            .as_posix()
                        ),

                        "split": item["split"],

                        "source_image": (
                            item["source_image"]
                        ),

                        "source_split": (
                            item["source_split"]
                        ),

                        "class": box["class"],

                        "weather": item.get(
                            "weather",
                            "undefined"
                        ),

                        "timeofday": item.get(
                            "timeofday",
                            "undefined"
                        ),

                        "scene": item.get(
                            "scene",
                            "undefined"
                        ),

                        "environment": (
                            item["environment"]
                        ),

                        "box_index": (
                            box["box_index"]
                        ),

                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    })

                    completed_ids.add(crop_id)

                    counters["crops"] += 1

            counters["frames"] += 1

        except StopIteration as exc:

            print("STOP:", exc)
            break

        except (
            requests.RequestException,
            OSError,
            ValueError,
            KeyError,
            UnidentifiedImageError
        ) as exc:

            failed.append({
                "source_image": (
                    item["source_image"]
                ),
                "error": str(exc)
            })

            print(
                "WARNING:",
                item["source_image"],
                str(exc)[:110]
            )

            counters["failed_frames"] += 1

        finally:

            # Delete original image to save disk space.
            if raw_path.is_file() and not keep_raw:
                raw_path.unlink()

            if index % 10 == 0:

                save_rows(
                    manifest_path,
                    rows
                )

                print(
                    f"{index}/{len(frames)} frames | "
                    f"{len(rows)} crops | "
                    f"{(ledger['metadata_bytes_reserved'] + used_bytes(ledger))/1e6:.2f} MB used"
                )

            time.sleep(0.02)

    # ----------------------------------------------
    # FINAL OUTPUTS
    # ----------------------------------------------

    save_rows(manifest_path, rows)

    save_ledger(
        ledger_path,
        ledger
    )

    if failed:

        failure_path = (
            ROOT
            / "results"
            / "phase1"
            / "failed_downloads.csv"
        )

        failure_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with failure_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_image",
                    "error"
                ]
            )

            writer.writeheader()
            writer.writerows(failed)

    print("\n====================================")
    print(" EXTRACTION COMPLETE")
    print("====================================")

    print("Total extracted crops:", len(rows))

    print(
        "Crops by split:",
        dict(
            Counter(
                row["split"]
                for row in rows
            )
        )
    )

    print(
        "Transfer allowance used:",
        round(
            (
                ledger["metadata_bytes_reserved"]
                + used_bytes(ledger)
            ) / 1e6,
            2
        ),
        "MB"
    )

    print("Manifest saved:", manifest_path)

    if failed:
        print(
            "Failed downloads:",
            len(failed)
        )


if __name__ == "__main__":
    main()
