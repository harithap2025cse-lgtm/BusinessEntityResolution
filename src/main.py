from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from .blocking import Blocker
from .config import Config
from .data_loader import combine_targets, load_dataset
from .evaluation import candidate_recall, entity_f05, predictions, tune_threshold
from .model import train, score
from .normalize import enrich
from .outputs import write_report, write_submission


def parse_args():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution — 10K pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def subset_by_ids(df, ids):
    return df[
        df["entity_id"].astype(str).isin(ids)
    ].reset_index(drop=True)


def main():
    args = parse_args()

    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        limit=max(1, args.limit),
        seed=args.seed,
    )

    print("=== Business — Entity Resolution ===")
    print(f"Python pipeline row limit: {config.limit}")
    print()

    print("[1/8] Loading the 10K dataset...")
    data = load_dataset(
        config.data_dir,
        config.limit,
    )

    train_s1 = enrich(data["train"]["s1"])
    train_s2 = enrich(data["train"]["s2"])
    train_s3 = enrich(data["train"]["s3"])
    train_truth = data["train"]["truth"]

    test_s1 = enrich(data["test"]["s1"])
    test_s2 = enrich(data["test"]["s2"])
    test_s3 = enrich(data["test"]["s3"])

    train_target = combine_targets(train_s2, train_s3)
    test_target = combine_targets(test_s2, test_s3)

    print(
        f"Train: S1={len(train_s1)}, "
        f"S2={len(train_s2)}, S3={len(train_s3)}"
    )
    print(
        f"Test : S1={len(test_s1)}, "
        f"S2={len(test_s2)}, S3={len(test_s3)}"
    )

    print()
    print("[2/8] Building training blocker...")
    blocker = Blocker.fit(
        train_target,
        top_k_tfidf=config.top_k_tfidf,
        max_candidates=config.max_candidates_per_source1,
        block_key_limit=config.block_key_limit,
    )

    print("[3/8] Generating training candidates...")
    all_pairs = blocker.generate(train_s1)
    print(f"Training candidate pairs: {len(all_pairs)}")

    source1_ids = train_s1["entity_id"].astype(str).tolist()
    train_ids, valid_ids = train_test_split(
        source1_ids,
        test_size=config.validation_fraction,
        random_state=config.seed,
        shuffle=True,
    )

    train_id_set = set(train_ids)
    valid_id_set = set(valid_ids)

    train_source1 = subset_by_ids(train_s1, train_id_set)
    valid_source1 = subset_by_ids(train_s1, valid_id_set)

    # Regenerate candidates so source1_index values match the subset DataFrame.
    train_pairs = blocker.generate(train_source1)
    valid_pairs = blocker.generate(valid_source1)

    train_truth = {
        sid: ids for sid, ids in train_truth.items()
        if sid in train_id_set
    }

    valid_truth = {
        sid: ids for sid, ids in data["train"]["truth"].items()
        if sid in valid_id_set
    }

    print(
        f"Training entities: {len(train_source1)} | "
        f"Validation entities: {len(valid_source1)}"
    )

    print()
    print("[4/8] Training matcher...")
    model, train_used_pairs = train(
        train_pairs,
        train_source1,
        train_target,
        train_truth,
        seed=config.seed,
        max_negative_per_entity=config.max_negative_pairs_per_entity,
    )

    print(
        f"Training pairs used: {len(train_used_pairs)} | "
        f"positives: {int(train_used_pairs['label'].sum())}"
    )

    print()
    print("[5/8] Tuning F0.5 threshold...")
    valid_scored = valid_pairs.copy()
    valid_scored["score"] = score(
        model,
        valid_source1,
        train_target,
        valid_pairs,
    )

    validation_threshold, validation_score = tune_threshold(
        valid_scored,
        valid_source1["entity_id"].astype(str).tolist(),
        valid_truth,
        minimum=config.threshold_min,
        maximum=config.threshold_max,
        count=config.threshold_count,
    )

    valid_predictions = predictions(
        valid_scored,
        validation_threshold,
    )

    validation_score = entity_f05(
        valid_source1["entity_id"].astype(str).tolist(),
        valid_predictions,
        valid_truth,
    )

    pair_recall, entity_recall = candidate_recall(
        valid_pairs,
        valid_truth,
    )

    print(f"Validation F0.5: {validation_score:.6f}")
    print(f"Selected threshold: {validation_threshold:.6f}")
    print(f"Candidate pair recall: {pair_recall:.6f}")
    print(f"Candidate entity recall: {entity_recall:.6f}")

    print()
    print("[6/8] Retraining on all 10K training entities...")
    final_blocker = Blocker.fit(
        train_target,
        top_k_tfidf=config.top_k_tfidf,
        max_candidates=config.max_candidates_per_source1,
        block_key_limit=config.block_key_limit,
    )

    final_train_pairs = final_blocker.generate(train_s1)

    final_model, final_train_used = train(
        final_train_pairs,
        train_s1,
        train_target,
        data["train"]["truth"],
        seed=config.seed,
        max_negative_per_entity=config.max_negative_pairs_per_entity,
    )

    print(
        f"Final training pairs used: {len(final_train_used)} | "
        f"positives: {int(final_train_used['label'].sum())}"
    )

    print()
    print("[7/8] Generating and scoring test candidates...")
    test_blocker = Blocker.fit(
        test_target,
        top_k_tfidf=config.top_k_tfidf,
        max_candidates=config.max_candidates_per_source1,
        block_key_limit=config.block_key_limit,
    )

    test_pairs = test_blocker.generate(test_s1)
    test_scored = test_pairs.copy()
    test_scored["score"] = score(
        final_model,
        test_s1,
        test_target,
        test_pairs,
    )

    print(f"Test candidate pairs: {len(test_pairs)}")

    print()
    print("[8/8] Writing required outputs...")
    matching_path, candidate_path = write_submission(
        config.output_dir,
        test_s1,
        test_pairs,
        test_scored,
        validation_threshold,
    )

    report_lines = [
        "Business — 10K Entity Resolution",
        "=================================================",
        f"Row limit per source: {config.limit}",
        f"Train S1: {len(train_s1)}",
        f"Train S2: {len(train_s2)}",
        f"Train S3: {len(train_s3)}",
        f"Test S1: {len(test_s1)}",
        f"Test S2: {len(test_s2)}",
        f"Test S3: {len(test_s3)}",
        f"Training candidate pairs: {len(all_pairs)}",
        f"Validation candidate pairs: {len(valid_pairs)}",
        f"Test candidate pairs: {len(test_pairs)}",
        f"Validation F0.5: {validation_score:.6f}",
        f"Selected threshold: {validation_threshold:.6f}",
        f"Validation candidate pair recall: {pair_recall:.6f}",
        f"Validation candidate entity recall: {entity_recall:.6f}",
        f"matching_results.tsv: {matching_path}",
        f"candidate_pairs.tsv: {candidate_path}",
    ]

    write_report(
        config.output_dir,
        report_lines,
    )

    print()
    print("DONE")
    print(f"Final submission: {matching_path}")
    print(f"Candidate file:   {candidate_path}")
    print(f"Report:           {config.output_dir / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
