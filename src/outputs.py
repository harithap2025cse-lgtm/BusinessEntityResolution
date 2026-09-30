from pathlib import Path

import pandas as pd


def unique_ordered(values):
    seen = set()
    result = []

    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def write_submission(
    output_dir,
    source1,
    candidate_pairs,
    scored_pairs,
    threshold,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidate_map = {}
    for sid, group in candidate_pairs.groupby(
        "source1_entity_id", sort=False
    ):
        candidate_map[str(sid)] = unique_ordered(
            group["candidate_entity_id"].tolist()
        )

    match_map = {}
    accepted = scored_pairs[
        scored_pairs["score"] >= threshold
    ].copy()

    for sid, group in accepted.groupby(
        "source1_entity_id", sort=False
    ):
        ordered = (
            group.sort_values(
                ["score", "target_index"],
                ascending=[False, True],
            )["candidate_entity_id"]
            .tolist()
        )
        match_map[str(sid)] = unique_ordered(ordered)

    matching_rows = []
    candidate_rows = []

    for sid in source1["entity_id"].astype(str):
        matching_rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(
                match_map.get(sid, [])
            ),
        })

        candidate_rows.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(
                candidate_map.get(sid, [])
            ),
        })

    matching_path = out / "matching_results.tsv"
    candidate_path = out / "candidate_pairs.tsv"

    pd.DataFrame(matching_rows).to_csv(
        matching_path,
        sep="\t",
        index=False,
    )

    pd.DataFrame(candidate_rows).to_csv(
        candidate_path,
        sep="\t",
        index=False,
    )

    return matching_path, candidate_path


def write_report(output_dir, lines):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "validation_report.txt"
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return path
