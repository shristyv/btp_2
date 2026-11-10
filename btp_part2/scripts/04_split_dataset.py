
"""
BTP Part 2 - Phase 1, Step 4

Verify existing train/val/test assignments and export manifests.

Input:
    data/manifests/all_crops.csv

Outputs:
    data/manifests/train.csv
    data/manifests/val.csv
    data/manifests/test.csv
    results/phase1/split_validation.json

The splits were assigned in Step 2.
We do NOT randomly split the crops again.
"""

import json
from pathlib import Path

import pandas as pd

from common import ROOT


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

SPLITS = ("train", "val", "test")

REQUIRED_COLUMNS = [
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
]


def validate_dataset(df):

    errors = []
    warnings = []

    # -----------------------------------------
    # 1. Check required columns
    # -----------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------
    # 2. Check empty values
    # -----------------------------------------

    core_columns = [
        "crop_id",
        "crop_path",
        "split",
        "source_image",
        "class",
    ]

    for col in core_columns:
        missing = int(
            df[col].isna().sum()
            + df[col].astype(str).str.strip().eq("").sum()
        )

        if missing > 0:
            errors.append(
                f"{col} contains {missing} missing values"
            )

    # -----------------------------------------
    # 3. Check duplicate crop IDs
    # -----------------------------------------

    duplicates = int(
        df["crop_id"].duplicated().sum()
    )

    if duplicates:
        errors.append(
            f"Found {duplicates} duplicate crop IDs"
        )

    # -----------------------------------------
    # 4. Validate split names
    # -----------------------------------------

    invalid_splits = set(
        df["split"].dropna().unique()
    ) - set(SPLITS)

    if invalid_splits:
        errors.append(
            f"Invalid split names: {invalid_splits}"
        )

    # -----------------------------------------
    # 5. Validate object classes
    # -----------------------------------------

    actual_classes = set(
        df["class"].dropna().unique()
    )

    invalid_classes = (
        actual_classes - TARGET_CLASSES
    )

    missing_classes = (
        TARGET_CLASSES - actual_classes
    )

    if invalid_classes:
        errors.append(
            f"Unexpected classes: {sorted(invalid_classes)}"
        )

    if missing_classes:
        warnings.append(
            f"Classes absent from dataset: "
            f"{sorted(missing_classes)}"
        )

    # -----------------------------------------
    # 6. Check that crop files exist
    # -----------------------------------------

    missing_files = []

    for path in df["crop_path"].dropna():

        crop_path = ROOT / Path(path)

        if not crop_path.is_file():
            missing_files.append(str(path))

    if missing_files:
        errors.append(
            f"{len(missing_files)} crop images are missing"
        )

    # -----------------------------------------
    # 7. Check source-image leakage
    # -----------------------------------------

    source_sets = {}

    for split in SPLITS:

        subset = df[df["split"] == split]

        source_sets[split] = set(
            zip(
                subset["source_split"],
                subset["source_image"]
            )
        )

    overlap_counts = {}

    for i, split_a in enumerate(SPLITS):

        for split_b in SPLITS[i + 1:]:

            overlap = (
                source_sets[split_a]
                & source_sets[split_b]
            )

            key = f"{split_a}_vs_{split_b}"

            overlap_counts[key] = len(overlap)

            if overlap:
                errors.append(
                    f"Source-image leakage: {key} "
                    f"has {len(overlap)} shared frames"
                )

    # -----------------------------------------
    # 8. Check environment metadata
    # -----------------------------------------

    for col in [
        "weather",
        "timeofday",
        "scene",
        "environment"
    ]:

        missing = int(
            df[col].isna().sum()
        )

        undefined = int(
            df[col].astype(str)
            .str.lower()
            .eq("undefined")
            .sum()
        )

        if missing or undefined:
            warnings.append(
                f"{col}: {missing} missing and "
                f"{undefined} undefined values"
            )

    # -----------------------------------------
    # 9. Summarize distributions
    # -----------------------------------------

    split_counts = {
        split: int(
            (df["split"] == split).sum()
        )
        for split in SPLITS
    }

    source_counts = {
        split: len(source_sets[split])
        for split in SPLITS
    }

    class_counts = {
        split: {
            str(k): int(v)
            for k, v in (
                df[df["split"] == split]["class"]
                .value_counts()
                .items()
            )
        }
        for split in SPLITS
    }

    environment_counts = {
        split: {
            str(k): int(v)
            for k, v in (
                df[df["split"] == split]["environment"]
                .value_counts()
                .items()
            )
        }
        for split in SPLITS
    }

    report = {
        "status": "PASS" if not errors else "FAIL",
        "total_crops": int(len(df)),
        "split_counts": split_counts,
        "source_frame_counts": source_counts,
        "unique_classes": sorted(actual_classes),
        "missing_target_classes": sorted(missing_classes),
        "duplicate_crop_ids": duplicates,
        "missing_crop_files": len(missing_files),
        "source_overlap_counts": overlap_counts,
        "class_counts": class_counts,
        "environment_counts": environment_counts,
        "errors": errors,
        "warnings": warnings,
    }

    return report


def main():

    input_path = (
        ROOT / "data" / "manifests"
        / "all_crops.csv"
    )

    output_dir = (
        ROOT / "data" / "manifests"
    )

    results_dir = (
        ROOT / "results" / "phase1"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. "
            "Run 03_extract_crops.py first."
        )

    df = pd.read_csv(input_path)

    print("\n====================================")
    print(" BTP PART 2 - SPLIT VALIDATION")
    print("====================================")

    report = validate_dataset(df)

    report_path = (
        results_dir / "split_validation.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("\nValidation status:", report["status"])

    print("\nDataset split counts:")
    for split, count in report["split_counts"].items():
        print(f"  {split}: {count}")

    print("\nUnique source frames:")
    for split, count in report["source_frame_counts"].items():
        print(f"  {split}: {count}")

    print("\nSource overlap checks:")
    for key, count in report["source_overlap_counts"].items():
        print(f"  {key}: {count}")

    print("\nClasses found:")
    print(", ".join(report["unique_classes"]))

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(" -", warning)

    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print(" -", error)

        print("\nValidation failed. No split CSVs exported.")
        raise SystemExit(1)

    # Only export split files after validation passes.
    for split in SPLITS:

        subset = (
            df[df["split"] == split]
            .sort_values("crop_id")
            .reset_index(drop=True)
        )

        output_path = (
            output_dir / f"{split}.csv"
        )

        subset.to_csv(
            output_path,
            index=False
        )

        print(
            f"Saved {output_path.name}: "
            f"{len(subset)} crops"
        )

    print("\n====================================")
    print(" STEP 4 COMPLETED SUCCESSFULLY")
    print("====================================")

    print("Validation report:", report_path)


if __name__ == "__main__":
    main()
