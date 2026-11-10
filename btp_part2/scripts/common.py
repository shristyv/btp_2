import json
import time
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]

TARGET_CLASSES = {
    "car",
    "bus",
    "truck",
    "person",
    "rider",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
}

REPOS = {
    "images": "https://huggingface.co/datasets/dronefreak/BDD100K/resolve/main",
    "labels": "https://huggingface.co/datasets/dronefreak/BDD100K/resolve/main",
}

API_BASE = "https://huggingface.co/api/datasets/dronefreak/BDD100K/tree/main"


def config():
    cfg_path = ROOT / "configs" / "phase1.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class BudgetSession:
    """
    Keeps track of total downloaded bytes so we stay under budget.
    """
    def __init__(self, cfg):
        self.limit_bytes = int(cfg.get("download_limit_mb", 850) * 1024 * 1024)
        self.spent = 0
        self.session = requests.Session()

    def get(self, base_url, remote_path, local_path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            self.spent += local_path.stat().st_size
            return local_path

        url = f"{base_url}/{quote(remote_path)}"
        r = self.session.get(url, timeout=60)
        r.raise_for_status()

        content = r.content
        size = len(content)

        if self.spent + size > self.limit_bytes:
            raise RuntimeError(
                f"Download budget exceeded. Tried to add {size/1e6:.2f} MB "
                f"with {self.spent/1e6:.2f} MB already spent."
            )

        local_path.write_bytes(content)
        self.spent += size
        time.sleep(0.05)
        return local_path


def hf_tree(path, limit=1000):
    """
    List files/directories in a Hugging Face dataset folder.
    """
    r = requests.get(API_BASE + "/" + path, params={"limit": limit}, timeout=60)
    r.raise_for_status()
    return r.json()


def list_shards(split, kind="labels"):
    """
    kind = 'labels' or 'images'
    """
    if kind == "labels":
        base = f"data/labels/100k/{split}"
    else:
        base = f"data/images_100k/100k/{split}"

    return [item["path"] for item in hf_tree(base) if item["type"] == "directory"]


def list_files_in_shard(shard_path, limit=1000):
    return [item["path"] for item in hf_tree(shard_path, limit=limit) if item["type"] == "file"]


def local_label_path(remote_path):
    return ROOT / "data" / "raw" / "labels" / remote_path.replace("/", "_")


def local_image_path(remote_path):
    return ROOT / "data" / "raw" / "images" / remote_path.replace("/", "_")


def parse_annotation(record):
    """
    Parse one BDD100K annotation JSON.
    """
    attrs = record.get("attributes", {}) or {}
    frames = record.get("frames", []) or []

    objects = []
    if frames:
        objects = frames[0].get("objects", []) or []

    return {
        "name": record.get("name", ""),
        "weather": attrs.get("weather", "undefined"),
        "scene": attrs.get("scene", "undefined"),
        "timeofday": attrs.get("timeofday", "undefined"),
        "objects": objects,
    }


def candidate_boxes(record, min_box_side=20):
    """
    Extract only the nine classes we care about and only valid box2d objects.
    """
    parsed = parse_annotation(record)
    valid = []

    for obj in parsed["objects"]:
        category = obj.get("category", "").strip()
        if category not in TARGET_CLASSES:
            continue

        box = obj.get("box2d")
        if not box:
            continue

        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])

        w = x2 - x1
        h = y2 - y1

        if w < min_box_side or h < min_box_side:
            continue

        valid.append({
            "class": category,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return valid


def environment(record):
    parsed = parse_annotation(record)
    return {
        "weather": parsed["weather"],
        "scene": parsed["scene"],
        "timeofday": parsed["timeofday"],
    }