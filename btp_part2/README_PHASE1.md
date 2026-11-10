# BTP Part 2 / Phase 1

This continues Part 1's nine-class BDD100K object-*crop classification* using **one computer** and four sequentially simulated vehicle clients. It creates labeled/unlabeled splits for later FSSL, but does **not** train a model or claim improved accuracy.

## Installation (Windows PowerShell, from `btp_2/btp_part2`)

Use the existing `btp_2/.venv`:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If that doesn't work, activate the parent's environment with `..\.venv\Scripts\Activate.ps1` and run `python -m pip install -r requirements.txt`.

## Run in this exact order

```powershell
..\.venv\Scripts\python.exe scripts/01_audit_metadata.py
..\.venv\Scripts\python.exe scripts/02_select_dataset.py
..\.venv\Scripts\python.exe scripts/03_extract_crops.py
..\.venv\Scripts\python.exe scripts/04_split_dataset.py
..\.venv\Scripts\python.exe scripts/05_create_clients.py
..\.venv\Scripts\python.exe scripts/06_validate_dataset.py
```

**Pilot**: `configs/phase1.yaml` initially targets 300 train, 60 validation, 60 test crops. After the pilot, inspect the weather/time labels and crop images. Then change targets to `train: 3200`, `val: 600`, `test: 600` and rerun 02 → 06. The script counts persistent downloaded HTTP response bytes towards 850 MB, excluding installation and unrelated requests. Cached metadata is reused. If the budget is exhausted, 03 writes progress and stops. The code does **not** guarantee enough crops before that point.

## Outputs

- `data/raw/labels/`: third-party dataset annotation metadata
- `data/processed/crops/`: extracted 160 px max-side JPEG object crops
- `data/manifests/all_crops.csv`: one record per crop (class and environment)
- `data/manifests/train.csv`, `val.csv`, `test.csv`: held-out splits
- `data/clients/client_1..4/iid` and `/environment`: `labeled.csv` and `unlabeled.csv`
- `data/manifests/*_hidden_targets_RESEARCH_ONLY.csv`: **must never be accessible to training**
- `results/phase1/`: validation, audit counts, heatmaps, divergence

## Important limitations

1. The repos are *third-party Hugging Face redistributions*. Metadata directory layout and label encodings can change. If 01 errors, inspect the repository tree. Do not guess mappings for numeric `weather` or `timeofday` labels.
2. The dataset is a controlled object-*classification* benchmark, **not** full-image object detection or a proof of vehicle-level safety.
3. Train/val/test use distinct `source_image` values, but `source_video` is not supplied in these manifests. Similar frames from a common driving sequence might require an extra deduplication audit.
4. The selector estimates crop counts from box metadata. Actual counts may be lower after image integrity and box clipping checks.
5. The client allocations are metadata-weighted *proxies* for independent vehicles, not actual vehicle IDs. Use the IID and environment partitions over the same source training pool for comparisons.
6. `unlabeled.csv` contains no class label; diagnostic hidden targets must stay outside model training code.
7. We do not automatically enforce rare-class or environmental minimum sample counts. Evaluate those counts and adjust before training.
8. `test` split availability depends on third-party source metadata. If a source split is missing, STOP and use verified official held-out annotations; never relabel train images as an official test set.

## Scientific question for Phase 2+

Does environment-aware federated semi-supervised learning help low-label road-object classification under non-IID conditions? Baselines and proposed algorithms need equal source splits, label budgets and seeds. Report macro-F1, per-class recall, environment-level metrics and communication bytes, with repeated seeds.
