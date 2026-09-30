from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path, limit: int | None = None):
    """
    Read a TSV file.

    If limit is provided, only the first `limit` data rows are read.
    """
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        if limit is None:
            return list(reader)

        rows = []
        for row_number, row in enumerate(reader):
            if row_number >= limit:
                break
            rows.append(row)

        return rows


def source_ids(path: Path, limit: int = 10000):
    """
    Return entity IDs from only the first `limit` rows.
    """
    rows = read_tsv(path, limit=limit)

    if not rows or "entity_id" not in rows[0]:
        raise ValueError(f"{path} does not contain entity_id")

    return {
        str(row["entity_id"]).strip()
        for row in rows
        if str(row["entity_id"]).strip()
    }


def split_ids(value):
    """
    Convert a comma-separated ID string into a list of IDs.
    """
    if not value:
        return []

    return [
        part.strip()
        for part in str(value).split(",")
        if part.strip()
    ]


def validate(
    matching,
    candidates,
    s1,
    s2,
    s3,
    limit: int = 10000,
):
    problems = []

    required_matching = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    required_candidates = {
        "source1_entity_id",
        "candidate_entity_ids",
    }

    # ---------------------------------------------------------
    # Basic file checks
    # ---------------------------------------------------------

    if not matching:
        problems.append("matching_results.tsv is empty")
        return problems

    if not candidates:
        problems.append("candidate_pairs.tsv is empty")
        return problems

    if set(matching[0].keys()) != required_matching:
        problems.append(
            f"matching_results.tsv columns are "
            f"{list(matching[0].keys())}; "
            f"expected {sorted(required_matching)}"
        )

    if set(candidates[0].keys()) != required_candidates:
        problems.append(
            f"candidate_pairs.tsv columns are "
            f"{list(candidates[0].keys())}; "
            f"expected {sorted(required_candidates)}"
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Validate only the first `limit` test rows because the
    # project is intentionally restricted to 10,000 records.
    # ---------------------------------------------------------

    s1_ids = source_ids(s1, limit=limit)
    s2_ids = source_ids(s2, limit=limit)
    s3_ids = source_ids(s3, limit=limit)

    target_ids = s2_ids | s3_ids

    # ---------------------------------------------------------
    # Row-count checks
    # ---------------------------------------------------------

    if len(matching) != len(s1_ids):
        problems.append(
            f"matching rows={len(matching)}, "
            f"expected={len(s1_ids)}"
        )

    if len(candidates) != len(s1_ids):
        problems.append(
            f"candidate rows={len(candidates)}, "
            f"expected={len(s1_ids)}"
        )

    # ---------------------------------------------------------
    # Build lookup dictionaries
    # ---------------------------------------------------------

    match_map = {}
    candidate_map = {}

    for row in matching:
        sid = str(row["source1_entity_id"]).strip()

        if sid in match_map:
            problems.append(
                f"duplicate Source 1 ID in matching_results.tsv: {sid}"
            )

        match_map[sid] = split_ids(
            row.get("matched_entity_ids", "")
        )

    for row in candidates:
        sid = str(row["source1_entity_id"]).strip()

        if sid in candidate_map:
            problems.append(
                f"duplicate Source 1 ID in candidate_pairs.tsv: {sid}"
            )

        candidate_map[sid] = split_ids(
            row.get("candidate_entity_ids", "")
        )

    # ---------------------------------------------------------
    # Validate every Source 1 entity
    # ---------------------------------------------------------

    for sid in s1_ids:

        # Must exist in both output files
        if sid not in match_map:
            problems.append(
                f"missing Source 1 ID in matching_results.tsv: {sid}"
            )

        if sid not in candidate_map:
            problems.append(
                f"missing Source 1 ID in candidate_pairs.tsv: {sid}"
            )

        matches = set(match_map.get(sid, []))
        cands = set(candidate_map.get(sid, []))

        # -----------------------------------------------------
        # IDs must exist in Source 2 / Source 3
        # -----------------------------------------------------

        invalid_matches = sorted(matches - target_ids)
        invalid_candidates = sorted(cands - target_ids)

        if invalid_matches:
            problems.append(
                f"invalid matches for {sid}: "
                f"{invalid_matches[:5]}"
            )

        if invalid_candidates:
            problems.append(
                f"invalid candidates for {sid}: "
                f"{invalid_candidates[:5]}"
            )

        # -----------------------------------------------------
        # Every final match must be a candidate
        # -----------------------------------------------------

        if not matches.issubset(cands):
            missing_from_candidates = sorted(
                matches - cands
            )

            problems.append(
                f"final match not present in candidate set "
                f"for {sid}: "
                f"{missing_from_candidates[:5]}"
            )

        # -----------------------------------------------------
        # No Source 1 self-match
        # -----------------------------------------------------

        if sid in matches:
            problems.append(
                f"Source 1 ID matched to itself: {sid}"
            )

        if sid in cands:
            problems.append(
                f"Source 1 ID included in candidate set: {sid}"
            )

        # -----------------------------------------------------
        # Duplicate IDs inside an individual output list
        # -----------------------------------------------------

        raw_matches = match_map.get(sid, [])
        raw_candidates = candidate_map.get(sid, [])

        if len(raw_matches) != len(set(raw_matches)):
            problems.append(
                f"duplicate entity IDs in matched list for {sid}"
            )

        if len(raw_candidates) != len(set(raw_candidates)):
            problems.append(
                f"duplicate entity IDs in candidate list for {sid}"
            )

    return problems


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 10K Business Entity "
            "Resolution submission."
        )
    )

    parser.add_argument(
        "--matching",
        required=True,
        help="Path to matching_results.tsv",
    )

    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate_pairs.tsv",
    )

    parser.add_argument(
        "--test-s1",
        required=True,
        help="Path to test_source1.tsv",
    )

    parser.add_argument(
        "--test-s2",
        required=True,
        help="Path to test_source2.tsv",
    )

    parser.add_argument(
        "--test-s3",
        required=True,
        help="Path to test_source3.tsv",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of test rows to validate. Default: 10000",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        print("ERROR: --limit must be greater than 0")
        return 1

    # ---------------------------------------------------------
    # Read output files.
    # These should contain exactly the rows produced by our
    # 10K pipeline.
    # ---------------------------------------------------------

    matching = read_tsv(
        Path(args.matching),
        limit=None,
    )

    candidates = read_tsv(
        Path(args.candidate),
        limit=None,
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    problems = validate(
        matching=matching,
        candidates=candidates,
        s1=Path(args.test_s1),
        s2=Path(args.test_s2),
        s3=Path(args.test_s3),
        limit=args.limit,
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    if not problems:
        print("PASS")
        print(
            f"Validated first {args.limit:,} "
            f"test records successfully."
        )
        return 0

    print(
        f"FAIL — {len(problems):,} issue(s)"
    )

    for i, problem in enumerate(problems, start=1):
        print(f"{i}. {problem}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())