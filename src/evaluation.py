from collections import defaultdict

import numpy as np
import pandas as pd


def f05(predicted, truth):
    if not predicted and not truth:
        return 1.0
    if not predicted:
        return 0.0

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth) if truth else 0.0

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def predictions(scored_pairs, threshold):
    result = defaultdict(set)

    for row in scored_pairs.itertuples(index=False):
        if float(row.score) >= threshold:
            result[str(row.source1_entity_id)].add(
                str(row.candidate_entity_id)
            )

    return dict(result)


def entity_f05(source1_ids, pred, truth):
    values = [
        f05(
            pred.get(str(sid), set()),
            truth.get(str(sid), set()),
        )
        for sid in source1_ids
    ]
    return float(np.mean(values)) if values else 0.0


def candidate_recall(candidate_pairs, truth):
    candidates = defaultdict(set)

    for row in candidate_pairs.itertuples(index=False):
        candidates[str(row.source1_entity_id)].add(
            str(row.candidate_entity_id)
        )

    total_true = 0
    covered_true = 0
    fully_covered_entities = 0
    entities_with_truth = 0

    for sid, truth_ids in truth.items():
        if not truth_ids:
            continue

        entities_with_truth += 1
        available = candidates.get(str(sid), set())
        total_true += len(truth_ids)
        covered_true += len(available & truth_ids)

        if truth_ids.issubset(available):
            fully_covered_entities += 1

    pair_recall = covered_true / total_true if total_true else 1.0
    entity_recall = (
        fully_covered_entities / entities_with_truth
        if entities_with_truth else 1.0
    )

    return pair_recall, entity_recall


def tune_threshold(scored_pairs, source1_ids, truth, minimum, maximum, count):
    thresholds = np.linspace(minimum, maximum, count)
    scores = []

    for threshold in thresholds:
        pred = predictions(scored_pairs, float(threshold))
        scores.append(
            entity_f05(source1_ids, pred, truth)
        )

    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])
