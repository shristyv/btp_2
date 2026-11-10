
"""
BTP Part 2 - Phase 1, Step 6
Final dataset validation and Non-IID statistical analysis.

Inputs:
    data/manifests/train.csv
    data/manifests/val.csv
    data/manifests/test.csv
    data/manifests/labeled_ids.csv
    data/manifests/iid_assignments.csv
    data/manifests/environment_non_iid_assignments.csv
    data/clients/client_*/

Outputs:
    results/phase1/final_validation.json
    results/phase1/heterogeneity_metrics.csv
    results/phase1/client_label_counts.csv
    results/phase1/test_coverage.csv
    results/phase1/*_heatmap.png
    results/phase1/js_divergence_comparison.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import ROOT


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

N_CLIENTS = 4

STRATEGIES = [
    "iid",
    "environment_non_iid"
]

CLASSES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign",
    "truck"
]

CLIENTS = [
    f"client_{i}"
    for i in range(1, N_CLIENTS + 1)
]

MANIFEST_DIR = ROOT / "data" / "manifests"
CLIENT_DIR = ROOT / "data" / "clients"
RESULTS_DIR = ROOT / "results" / "phase1"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

errors = []
warnings = []


# --------------------------------------------------
# 1. LOAD CSV FILES
# --------------------------------------------------

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

    return pd.read_csv(
        path,
        keep_default_na=False
    )


def add_error(message):
    errors.append(message)
    print("ERROR:", message)


def add_warning(message):
    warnings.append(message)
    print("WARNING:", message)


# --------------------------------------------------
# 2. JENSEN-SHANNON DIVERGENCE
# --------------------------------------------------

def js_divergence(p, q):
    """
    Calculate Jensen-Shannon divergence
    using base-2 logarithms.

    Range: 0 to 1
    """

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError(
            "Cannot calculate JSD for empty distributions."
        )

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = a > 0

        return np.sum(
            a[mask] * np.log2(
                a[mask] / b[mask]
            )
        )

    return float(
        0.5 * kl_divergence(p, m)
        + 0.5 * kl_divergence(q, m)
    )


# --------------------------------------------------
# 3. CHECK DATASET SPLITS
# --------------------------------------------------

def validate_splits(train, val, test):

    print("\n=== DATASET VALIDATION ===")

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    for name, df in splits.items():

        print(f"{name}: {len(df)} crops")

        if df.empty:
            add_error(f"{name} is empty")

        if df["crop_id"].duplicated().any():
            add_error(
                f"Duplicate crop IDs in {name}"
            )

        if not df["split"].eq(name).all():
            add_error(
                f"Incorrect split values in {name}"
            )

        invalid = set(df["class"]) - set(CLASSES)

        if invalid:
            add_error(
                f"Unexpected classes in {name}: {invalid}"
            )

        for crop_path in df["crop_path"]:
            if not (ROOT / crop_path).is_file():
                add_error(
                    f"Missing crop: {crop_path}"
                )

    # Check all crop IDs are unique across splits.
    combined = pd.concat(
        splits.values(),
        ignore_index=True
    )

    if combined["crop_id"].duplicated().any():
        add_error(
            "Crop IDs overlap across dataset splits"
        )

    # Check original frame overlap.
    source_sets = {}

    for name, df in splits.items():

        source_sets[name] = set(
            zip(
                df["source_split"].astype(str),
                df["source_image"].astype(str)
            )
        )

    names = list(splits)

    for i, a in enumerate(names):
        for b in names[i + 1:]:

            overlap = source_sets[a] & source_sets[b]

            print(
                f"{a} vs {b} source overlap:",
                len(overlap)
            )

            if overlap:
                add_error(
                    f"Source-image leakage: {a} vs {b}"
                )

    return combined


# --------------------------------------------------
# 4. VALIDATE FEDERATED CLIENT FILES
# --------------------------------------------------

def validate_clients(train, labeled_ids):

    print("\n=== FEDERATED CLIENT VALIDATION ===")

    training_ids = set(train["crop_id"])

    if len(labeled_ids) != len(set(labeled_ids)):
        add_error(
            "Duplicate IDs in labeled_ids.csv"
        )

    labeled_set = set(labeled_ids)

    if not labeled_set.issubset(training_ids):
        add_error(
            "Labeled IDs not contained in training set"
        )

    expected_columns = {
        "crop_id",
        "crop_path",
        "client_id",
        "label_status"
    }

    results = {}

    for strategy in STRATEGIES:

        print(f"\nStrategy: {strategy}")

        path = (
            MANIFEST_DIR
            / f"{strategy}_assignments.csv"
        )

        df = read_csv(path)

        if df["crop_id"].duplicated().any():
            add_error(
                f"Duplicate crop ownership in {strategy}"
            )

        if set(df["crop_id"]) != training_ids:
            add_error(
                f"Incomplete training coverage in {strategy}"
            )

        if set(df["client_id"]) != set(CLIENTS):
            add_error(
                f"Incorrect client IDs in {strategy}"
            )

        missing = expected_columns - set(df.columns)

        if missing:
            add_error(
                f"Missing columns in {strategy}: {missing}"
            )
            continue

        actual_labeled = set(
            df.loc[
                df["label_status"] == "labeled",
                "crop_id"
            ]
        )

        if actual_labeled != labeled_set:
            add_error(
                f"Labeled subset mismatch in {strategy}"
            )

        if not df["label_status"].isin(
            ["labeled", "unlabeled"]
        ).all():
            add_error(
                f"Invalid label status in {strategy}"
            )

        # Every source frame must remain in one client.
        grouped = df.groupby(
            ["source_split", "source_image"]
        )["client_id"].nunique()

        if (grouped > 1).any():
            add_error(
                f"Source frames shared across {strategy} clients"
            )

        # Verify each client's exported CSVs.
        for client in CLIENTS:

            folder = CLIENT_DIR / client

            labeled_file = (
                folder / f"{strategy}_labeled.csv"
            )

            unlabeled_file = (
                folder / f"{strategy}_unlabeled.csv"
            )

            labeled_df = read_csv(labeled_file)
            unlabeled_df = read_csv(unlabeled_file)

            expected = df[
                df["client_id"] == client
            ]

            actual_ids = (
                labeled_df["crop_id"].tolist()
                + unlabeled_df["crop_id"].tolist()
            )

            if len(actual_ids) != len(set(actual_ids)):
                add_error(
                    f"Repeated IDs in {client}/{strategy}"
                )

            if set(actual_ids) != set(
                expected["crop_id"]
            ):
                add_error(
                    f"Client file mismatch: {client}/{strategy}"
                )

            expected_labeled = set(
                expected.loc[
                    expected["label_status"] == "labeled",
                    "crop_id"
                ]
            )

            if set(labeled_df["crop_id"]) != expected_labeled:
                add_error(
                    f"Wrong labeled IDs: {client}/{strategy}"
                )

            if "class" not in labeled_df.columns:
                add_error(
                    f"Labeled targets missing: {client}/{strategy}"
                )

            # Unlabeled files must not include targets.
            forbidden = {
                "class",
                "label",
                "target",
                "class_id"
            }

            exposed = (
                forbidden & set(unlabeled_df.columns)
            )

            if exposed:
                add_error(
                    f"Labels exposed in unlabeled CSV: "
                    f"{client}/{strategy}: {exposed}"
                )

            print(
                f"  {client}: "
                f"{len(expected)} total, "
                f"{len(labeled_df)} labeled, "
                f"{len(unlabeled_df)} unlabeled"
            )

        results[strategy] = df

    return results


# --------------------------------------------------
# 5. CALCULATE CLIENT DISTRIBUTIONS
# --------------------------------------------------

def calculate_distributions(assignments, train):

    records = []
    matrices = {}

    all_environments = sorted(
        set(train["env_group"])
    )

    for strategy, df in assignments.items():

        print(f"\n=== {strategy.upper()} DISTRIBUTION ===")

        class_table = pd.crosstab(
            df["client_id"],
            df["class"]
        ).reindex(
            index=CLIENTS,
            columns=CLASSES,
            fill_value=0
        )

        environment_table = pd.crosstab(
            df["client_id"],
            df["env_group"]
        ).reindex(
            index=CLIENTS,
            columns=all_environments,
            fill_value=0
        )

        matrices[strategy] = {
            "class": class_table,
            "environment": environment_table
        }

        global_class = (
            train["class"]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
            .to_numpy()
        )

        global_env = (
            train["env_group"]
            .value_counts()
            .reindex(
                all_environments,
                fill_value=0
            )
            .to_numpy()
        )

        for client in CLIENTS:

            client_data = df[
                df["client_id"] == client
            ]

            class_js = js_divergence(
                class_table.loc[client].to_numpy(),
                global_class
            )

            env_js = js_divergence(
                environment_table.loc[client].to_numpy(),
                global_env
            )

            records.append({
                "strategy": strategy,
                "client_id": client,
                "samples": len(client_data),
                "class_jsd": class_js,
                "environment_jsd": env_js
            })

            print(
                f"{client}: "
                f"class JSD={class_js:.4f}, "
                f"environment JSD={env_js:.4f}"
            )

    metrics = pd.DataFrame(records)

    metrics.to_csv(
        RESULTS_DIR / "heterogeneity_metrics.csv",
        index=False
    )

    return metrics, matrices


# --------------------------------------------------
# 6. PLOT DISTRIBUTION HEATMAPS
# --------------------------------------------------

def plot_heatmap(table, title, output_path):

    values = table.to_numpy(dtype=float)

    row_sums = values.sum(
        axis=1,
        keepdims=True
    )

    percentages = np.divide(
        values * 100,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(
        figsize=(max(9, len(table.columns) * 1.2), 5)
    )

    image = ax.imshow(
        percentages,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(table.columns))
    )

    ax.set_xticklabels(
        table.columns,
        rotation=40,
        ha="right"
    )

    ax.set_yticks(
        range(len(table.index))
    )

    ax.set_yticklabels(
        table.index
    )

    for i in range(percentages.shape[0]):
        for j in range(percentages.shape[1]):

            value = percentages[i, j]

            ax.text(
                j, i,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white" if value > 50 else "black"
                )
            )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label="Percentage of client crops"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200
    )

    plt.close(fig)


# --------------------------------------------------
# 7. PLOT JSD COMPARISON
# --------------------------------------------------

def plot_js_comparison(metrics):

    summary = (
        metrics.groupby("strategy")[
            ["class_jsd", "environment_jsd"]
        ]
        .mean()
        .reindex(STRATEGIES)
    )

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        summary["class_jsd"],
        width,
        label="Class JSD"
    )

    ax.bar(
        x + width / 2,
        summary["environment_jsd"],
        width,
        label="Environment JSD"
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        "IID",
        "Environment Non-IID"
    ])

    ax.set_ylabel(
        "Mean client-to-global JSD"
    )

    ax.set_title(
        "Federated Data Heterogeneity Comparison"
    )

    ax.set_ylim(
        0,
        max(0.1, summary.max().max() * 1.2)
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "js_divergence_comparison.png",
        dpi=200
    )

    plt.close(fig)

    return summary


# --------------------------------------------------
# 8. LABELED DATA AND TEST COVERAGE
# --------------------------------------------------

def analyze_coverage(assignments, test):

    records = []

    for strategy, df in assignments.items():

        labeled = df[
            df["label_status"] == "labeled"
        ]

        counts = pd.crosstab(
            labeled["client_id"],
            labeled["class"]
        ).reindex(
            index=CLIENTS,
            columns=CLASSES,
            fill_value=0
        )

        for client in CLIENTS:

            for cls in CLASSES:

                records.append({
                    "strategy": strategy,
                    "client_id": client,
                    "class": cls,
                    "labeled_count": int(
                        counts.loc[client, cls]
                    )
                })

    pd.DataFrame(records).to_csv(
        RESULTS_DIR / "client_label_counts.csv",
        index=False
    )

    coverage = (
        test["class"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .rename_axis("class")
        .reset_index(name="test_count")
    )

    coverage.to_csv(
        RESULTS_DIR / "test_coverage.csv",
        index=False
    )

    for _, row in coverage.iterrows():

        if row["test_count"] < 10:

            add_warning(
                f"Only {row['test_count']} test examples "
                f"for class '{row['class']}'"
            )


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():

    print("\n======================================")
    print(" BTP PART 2 - FINAL DATASET VALIDATION")
    print("======================================")

    train = read_csv(
        MANIFEST_DIR / "train.csv"
    )

    val = read_csv(
        MANIFEST_DIR / "val.csv"
    )

    test = read_csv(
        MANIFEST_DIR / "test.csv"
    )

    labeled_df = read_csv(
        MANIFEST_DIR / "labeled_ids.csv"
    )

    # Reconstruct the environment groups using
    # the Step 5 assignment manifest.
    iid_df = read_csv(
        MANIFEST_DIR / "iid_assignments.csv"
    )

    env_mapping = (
        iid_df[["crop_id", "env_group"]]
        .drop_duplicates("crop_id")
    )

    train = train.merge(
        env_mapping,
        on="crop_id",
        how="left",
        validate="one_to_one"
    )

    if train["env_group"].isna().any():
        add_error(
            "Some training crops lack environment groups"
        )

    # 1. Original dataset validation
    validate_splits(train, val, test)

    # 2. Client files and assignments
    assignments = validate_clients(
        train,
        labeled_df["crop_id"].tolist()
    )

    # 3. Distribution analysis
    metrics, matrices = calculate_distributions(
        assignments,
        train
    )

    # 4. Distribution heatmaps
    for strategy, tables in matrices.items():

        for name, table in tables.items():

            output = (
                RESULTS_DIR
                / f"{strategy}_{name}_heatmap.png"
            )

            plot_heatmap(
                table,
                f"{strategy.upper()} - {name.title()} Distribution",
                output
            )

    # 5. Mean divergence comparison
    mean_jsd = plot_js_comparison(metrics)

    print("\n=== MEAN JSD COMPARISON ===")
    print(mean_jsd.to_string())

    # 6. Label and test-set coverage
    analyze_coverage(
        assignments,
        test
    )

    # 7. Research readiness checks
    iid_env = float(
        mean_jsd.loc[
            "iid", "environment_jsd"
        ]
    )

    noniid_env = float(
        mean_jsd.loc[
            "environment_non_iid",
            "environment_jsd"
        ]
    )

    if noniid_env <= iid_env:
        add_warning(
            "Non-IID environment divergence is not "
            "greater than IID divergence."
        )

    if len(train) < 1000:
        add_warning(
            "Pilot dataset is too small for "
            "final research conclusions."
        )

    if len(labeled_df) < 100:
        add_warning(
            "Very small labeled training budget."
        )

    # 8. Save final report
    report = {
        "validation_status": (
            "PASS" if not errors else "FAIL"
        ),
        "pilot_research_ready": False,
        "training_crops": len(train),
        "validation_crops": len(val),
        "test_crops": len(test),
        "clients": N_CLIENTS,
        "labeled_training_crops": len(labeled_df),
        "mean_jsd": mean_jsd.to_dict(
            orient="index"
        ),
        "errors": errors,
        "warnings": warnings
    }

    output_path = (
        RESULTS_DIR / "final_validation.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n======================================")
    print(" VALIDATION STATUS:",
          report["validation_status"])
    print("======================================")

    print("Errors:", len(errors))
    print("Warnings:", len(warnings))
    print("Saved report:", output_path)

    if errors:
        raise SystemExit(1)

    print("\nStep 6 completed successfully.")


if __name__ == "__main__":
    main()
