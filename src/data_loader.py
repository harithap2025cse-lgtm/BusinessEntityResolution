from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct

    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} below {root.resolve()}"
        )
    return matches[0]


def _read_source(path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=limit,
    )
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[SOURCE_COLUMNS].copy()
    for col in SOURCE_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    if df["entity_id"].duplicated().any():
        raise ValueError(f"Duplicate entity_id values found in {path}")

    return df


def _read_ground_truth(
    path: Path,
    selected_s1_ids: set[str],
    selected_target_ids: set[str],
) -> dict[str, set[str]]:
    result = {sid: set() for sid in selected_s1_ids}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=50_000,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]

        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(chunk.columns):
            raise ValueError(
                f"{path} must contain source1_entity_id and matched_entity_ids"
            )

        relevant = chunk[chunk["source1_entity_id"].isin(selected_s1_ids)]

        for row in relevant.itertuples(index=False):
            sid = str(row.source1_entity_id)
            raw = str(row.matched_entity_ids or "")
            result[sid] = {
                item.strip()
                for item in raw.split(",")
                if item.strip() and item.strip() in selected_target_ids
            }

    return result


def load_dataset(data_dir: str | Path, limit: int) -> dict:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root.resolve()}"
        )

    names = {
        "train_s1": "train_source1.tsv",
        "train_s2": "train_source2.tsv",
        "train_s3": "train_source3.tsv",
        "ground_truth": "train_ground_truth.tsv",
        "test_s1": "test_source1.tsv",
        "test_s2": "test_source2.tsv",
        "test_s3": "test_source3.tsv",
    }

    paths = {key: _find_file(root, filename) for key, filename in names.items()}

    train_s1 = _read_source(paths["train_s1"], limit)
    train_s2 = _read_source(paths["train_s2"], limit)
    train_s3 = _read_source(paths["train_s3"], limit)

    test_s1 = _read_source(paths["test_s1"], limit)
    test_s2 = _read_source(paths["test_s2"], limit)
    test_s3 = _read_source(paths["test_s3"], limit)

    selected_s1_ids = set(train_s1["entity_id"])
    selected_target_ids = set(train_s2["entity_id"]) | set(train_s3["entity_id"])

    truth = _read_ground_truth(
        paths["ground_truth"],
        selected_s1_ids=selected_s1_ids,
        selected_target_ids=selected_target_ids,
    )

    return {
        "train": {
            "s1": train_s1,
            "s2": train_s2,
            "s3": train_s3,
            "truth": truth,
        },
        "test": {
            "s1": test_s1,
            "s2": test_s2,
            "s3": test_s3,
        },
        "paths": paths,
    }


def combine_targets(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    a = s2.copy()
    b = s3.copy()
    a["source"] = "S2"
    b["source"] = "S3"
    result = pd.concat([a, b], ignore_index=True)
    result["target_index"] = range(len(result))
    return result
