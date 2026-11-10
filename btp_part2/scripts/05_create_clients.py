
"""
BTP Part 2 - Phase 1, Step 5
Create four simulated vehicle clients for federated learning.

Input:
    data/manifests/train.csv

Outputs:
    data/clients/client_1 ... client_4/
    data/manifests/labeled_ids.csv
    data/manifests/iid_assignments.csv
    data/manifests/environment_non_iid_assignments.csv
    results/phase1/client_summary.json

No images are copied or downloaded.
"""

from collections import Counter
import json

import numpy as np
import pandas as pd

from common import ROOT, config


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

N_CLIENTS = 4
LABEL_FRACTION = 0.10

STRATEGIES = (
    "iid",
    "environment_non_iid"
)

CLASSES = (
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "truck",
)


# --------------------------------------------------
# 1. NORMALIZE ENVIRONMENT CATEGORIES
# --------------------------------------------------

def normalized_environment(row):
    """
    Convert weather and time-of-day metadata into
    mutually exclusive driving-environment groups.
    """

    tod = str(row["timeofday"]).strip().lower()
    weather = str(row["weather"]).strip().lower()

    if tod == "night":
        return "night"

    if weather in {"rainy", "snowy", "foggy"}:
        return "adverse"

    if weather == "undefined" or tod == "undefined":
        return "unknown"

    if tod == "dawn/dusk":
        return "dawn_dusk"

    if tod == "daytime" and weather in {
        "clear", "partly cloudy"
    }:
        return "day_clear"

    if weather == "overcast":
        return "overcast"

    return "other"


# --------------------------------------------------
# 2. SELECT FIXED LABELED TRAINING SAMPLES
# --------------------------------------------------

def label_ids_fixed(train, rng, fraction):
    """
    Select a shared labeled subset for both
    IID and Non-IID experiments.

    Gives each available class at least one label,
    then distributes the remaining label budget
    approximately by class frequency.
    """

    budget = max(
        1,
        round(len(train) * fraction)
    )

    by_class = {
        cls: train.loc[
            train["class"] == cls,
            "crop_id"
        ].tolist()
        for cls in CLASSES
        if (train["class"] == cls).any()
    }

    if budget < len(by_class):
        raise ValueError(
            "Label budget is smaller than "
            "the number of represented classes."
        )

    quota = {
        cls: 1
        for cls in by_class
    }

    weights = {
        cls: len(by_class[cls]) / len(train)
        for cls in by_class
    }

    while sum(quota.values()) < budget:

        options = [
            cls for cls in by_class
            if quota[cls] < len(by_class[cls])
        ]

        if not options:
            break

        cls = max(
            options,
            key=lambda c: (
                budget * weights[c] - quota[c],
                weights[c],
                c
            )
        )

        quota[cls] += 1

    chosen = set()

    for cls, ids in by_class.items():

        selected = rng.choice(
            ids,
            size=quota[cls],
            replace=False
        ).tolist()

        chosen.update(selected)

    assert len(chosen) == budget

    return chosen, quota


# --------------------------------------------------
# 3. GROUP CROPS BY ORIGINAL SOURCE FRAME
# --------------------------------------------------

def grouped_frames(train):
    """
    Keep all selected crops originating from
    the same driving image together.
    """

    grouped = []

    for (source_split, source_image), group in train.groupby(
        ["source_split", "source_image"],
        sort=True
    ):

        grouped.append({
            "key": (
                str(source_split),
                str(source_image)
            ),

            "crop_ids": group["crop_id"].tolist(),

            "size": len(group),

            "environments": Counter(
                group["env_group"]
            ),

            "classes": Counter(
                group["class"]
            ),

            "primary_env": (
                group["env_group"]
                .value_counts()
                .index[0]
            ),
        })

    return grouped


# --------------------------------------------------
# 4. CREATE IID CLIENT ASSIGNMENTS
# --------------------------------------------------

def assign_iid(frames, train, rng):
    """
    Distribute source frames across clients while
    approximately balancing data size, environment
    distribution, and object-class distribution.
    """

    total = len(train)
    target_n = total / N_CLIENTS

    env_total = (
        train["env_group"]
        .value_counts(normalize=True)
        .to_dict()
    )

    class_total = (
        train["class"]
        .value_counts(normalize=True)
        .to_dict()
    )

    counts = [0] * N_CLIENTS

    env_counts = [
        Counter()
        for _ in range(N_CLIENTS)
    ]

    class_counts = [
        Counter()
        for _ in range(N_CLIENTS)
    ]

    assignment = {}

    order = list(
        rng.permutation(len(frames))
    )

    # Assign larger source-frame groups first.
    order.sort(
        key=lambda i: -frames[i]["size"]
    )

    for index in order:

        group = frames[index]

        scores = []

        for client in range(N_CLIENTS):

            n_after = (
                counts[client] + group["size"]
            )

            e_after = (
                env_counts[client]
                + group["environments"]
            )

            y_after = (
                class_counts[client]
                + group["classes"]
            )

            size_error = (
                n_after / target_n
            ) ** 2

            env_error = sum(
                (
                    e_after[k] - n_after * p
                ) ** 2
                for k, p in env_total.items()
            ) / target_n ** 2

            class_error = sum(
                (
                    y_after[k] - n_after * p
                ) ** 2
                for k, p in class_total.items()
            ) / target_n ** 2

            score = (
                2.0 * size_error
                + env_error
                + 0.6 * class_error
            )

            scores.append(score)

        best = min(scores)

        candidates = [
            c for c, score in enumerate(scores)
            if abs(score - best) < 1e-12
        ]

        client = int(
            rng.choice(candidates)
        )

        assignment[group["key"]] = client

        counts[client] += group["size"]

        env_counts[client].update(
            group["environments"]
        )

        class_counts[client].update(
            group["classes"]
        )

    return assignment


# --------------------------------------------------
# 5. CREATE ENVIRONMENT NON-IID ASSIGNMENTS
# --------------------------------------------------

def assign_environment_non_iid(frames, train, rng):
    """
    Assign different environment preferences
    to each simulated autonomous vehicle.
    """

    total = len(train)
    target_n = total / N_CLIENTS

    # Experimental client profiles.
    preference = [
        # Vehicle 1: daytime driving
        {
            "day_clear": 5,
            "dawn_dusk": 2,
            "overcast": 1
        },

        # Vehicle 2: nighttime driving
        {
            "night": 5,
            "dawn_dusk": 2
        },

        # Vehicle 3: adverse weather
        {
            "adverse": 5,
            "overcast": 3
        },

        # Vehicle 4: mixed conditions
        {
            "unknown": 3,
            "other": 3,
            "overcast": 2,
            "day_clear": 1,
            "night": 1,
            "dawn_dusk": 1,
            "adverse": 1
        },
    ]

    counts = [0] * N_CLIENTS

    assignment = {}

    order = list(
        rng.permutation(len(frames))
    )

    # Prioritize frames with strong environment matches.
    order.sort(
        key=lambda i: (
            -max(
                preference[c].get(
                    frames[i]["primary_env"], 0
                )
                for c in range(N_CLIENTS)
            ),
            -frames[i]["size"]
        )
    )

    for index in order:

        group = frames[index]

        scores = []

        for client in range(N_CLIENTS):

            affinity = sum(
                preference[client].get(env, 0)
                * count
                for env, count in
                group["environments"].items()
            ) / group["size"]

            new_size = (
                counts[client] + group["size"]
            )

            excess = max(
                0.0,
                new_size - target_n
            ) / target_n

            balance = (
                new_size / target_n
            )

            score = (
                affinity
                - 7.0 * excess
                - 0.35 * balance
            )

            scores.append(score)

        best = max(scores)

        candidates = [
            c for c, score in enumerate(scores)
            if abs(score - best) < 1e-12
        ]

        client = int(
            rng.choice(candidates)
        )

        assignment[group["key"]] = client

        counts[client] += group["size"]

    return assignment


# --------------------------------------------------
# 6. GENERATE CLIENT CSV FILES
# --------------------------------------------------

def generate_client_csvs(
    train,
    assignment,
    labeled,
    strategy,
    out_dir
):
    """
    Save labeled/unlabeled files for each client.
    Unlabeled CSVs do not expose ground-truth classes.
    """

    assigned = train.copy()

    assigned["client_id"] = [
        f"client_{assignment[
            (str(row.source_split), str(row.source_image))
        ] + 1}"
        for row in assigned.itertuples(index=False)
    ]

    assigned["label_status"] = np.where(
        assigned["crop_id"].isin(labeled),
        "labeled",
        "unlabeled"
    )

    hidden_fields = [
        "class",
        "box_index",
        "x",
        "y",
        "w",
        "h",
    ]

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    assigned.to_csv(
        out_dir / f"{strategy}_assignments.csv",
        index=False
    )

    for i in range(1, N_CLIENTS + 1):

        client = f"client_{i}"

        folder = (
            ROOT / "data" / "clients" / client
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        part = assigned[
            assigned["client_id"] == client
        ]

        labeled_df = part[
            part["label_status"] == "labeled"
        ].copy()

        unlabeled_df = part[
            part["label_status"] == "unlabeled"
        ].drop(
            columns=[
                c for c in hidden_fields
                if c in part.columns
            ]
        )

        labeled_df.to_csv(
            folder / f"{strategy}_labeled.csv",
            index=False
        )

        unlabeled_df.to_csv(
            folder / f"{strategy}_unlabeled.csv",
            index=False
        )

    return assigned


# --------------------------------------------------
# 7. VALIDATE CLIENT ASSIGNMENTS
# --------------------------------------------------

def validate(assignments, train, labeled):

    ids = assignments["crop_id"]

    assert len(ids) == len(train)

    assert ids.is_unique

    assert set(ids) == set(train["crop_id"])

    expected_clients = {
        f"client_{i}"
        for i in range(1, N_CLIENTS + 1)
    }

    assert (
        set(assignments["client_id"])
        == expected_clients
    )

    actual_labeled = set(
        assignments.loc[
            assignments["label_status"] == "labeled",
            "crop_id"
        ]
    )

    assert actual_labeled == labeled

    # Every original frame belongs to only one client.
    frame_client_counts = assignments.groupby(
        ["source_split", "source_image"]
    )["client_id"].nunique()

    assert not (
        frame_client_counts > 1
    ).any()


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    cfg = config()

    rng = np.random.default_rng(
        int(cfg.get("seed", 42))
    )

    train_path = (
        ROOT / "data" / "manifests" / "train.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            "Run 04_split_dataset.py before Step 5."
        )

    train = pd.read_csv(
        train_path,
        keep_default_na=False
    )

    required = {
        "crop_id",
        "crop_path",
        "source_image",
        "source_split",
        "class",
        "weather",
        "timeofday",
        "scene",
        "split",
    }

    missing = required - set(train.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if (
        train.empty
        or not train["crop_id"].is_unique
        or not train["split"].eq("train").all()
    ):
        raise ValueError(
            "Expected nonempty training data "
            "with unique crop IDs."
        )

    if not set(train["class"]).issubset(CLASSES):
        raise ValueError(
            "Unexpected object class found."
        )

    # Create environment categories.
    train["env_group"] = train.apply(
        normalized_environment,
        axis=1
    )

    # Shared labeled sample IDs.
    labeled, quota = label_ids_fixed(
        train,
        rng,
        LABEL_FRACTION
    )

    frames = grouped_frames(train)

    out_dir = (
        ROOT / "data" / "manifests"
    )

    pd.DataFrame({
        "crop_id": sorted(labeled)
    }).to_csv(
        out_dir / "labeled_ids.csv",
        index=False
    )

    print("\n====================================")
    print(" BTP PART 2 - FEDERATED CLIENT SETUP")
    print("====================================")

    print("Training crops:", len(train))
    print("Source frames:", len(frames))

    print(
        "Shared labeled budget:",
        len(labeled)
    )

    print(
        "Labeled class allocation:",
        quota
    )

    report_dir = (
        ROOT / "results" / "phase1"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {}

    # ------------------------------------------
    # Generate both federated configurations.
    # ------------------------------------------

    for strategy in STRATEGIES:

        if strategy == "iid":

            assignment = assign_iid(
                frames, train, rng
            )

        else:

            assignment = assign_environment_non_iid(
                frames, train, rng
            )

        result = generate_client_csvs(
            train,
            assignment,
            labeled,
            strategy,
            out_dir
        )

        validate(
            result,
            train,
            labeled
        )

        counts = (
            result.groupby("client_id")
            .size()
            .to_dict()
        )

        env = pd.crosstab(
            result["client_id"],
            result["env_group"]
        )

        labels = (
            result.groupby("client_id")[
                "label_status"
            ]
            .value_counts()
            .unstack(fill_value=0)
        )

        print(
            f"\n=== {strategy.upper()} ==="
        )

        print(
            "Crops per client:",
            counts
        )

        print(
            "\nEnvironment counts:\n",
            env.to_string()
        )

        print(
            "\nLabel counts:\n",
            labels.to_string()
        )

        env.to_csv(
            report_dir
            / f"{strategy}_environment_counts.csv"
        )

        summary[strategy] = {
            "crops_per_client": counts,

            "source_frames_per_client": (
                result.groupby("client_id")[
                    "source_image"
                ].nunique().to_dict()
            ),

            "labeled_per_client": (
                labels.get(
                    "labeled",
                    pd.Series(dtype=int)
                ).to_dict()
            ),
        }

    # Save experiment summary.
    summary_path = (
        report_dir / "client_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n====================================")
    print(" STEP 5 COMPLETED")
    print("====================================")

    print(
        "PASS: All training crops assigned "
        "exactly once per strategy."
    )

    print(
        "Source frames remain within "
        "individual clients."
    )

    print(
        "Labeled IDs are identical "
        "across IID and Non-IID setups."
    )

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()
