import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import matrix_for_pairs


def label_pairs(pairs, truth):
    labeled = pairs.copy()
    labeled["label"] = [
        1 if str(candidate_id) in truth.get(str(source1_id), set()) else 0
        for source1_id, candidate_id
        in zip(labeled["source1_entity_id"], labeled["candidate_entity_id"])
    ]
    return labeled


def add_ground_truth_pairs(pairs, source1, target, truth):
    existing = {
        (str(r.source1_entity_id), str(r.candidate_entity_id))
        for r in pairs.itertuples(index=False)
    }

    target_lookup = {
        str(row.entity_id): int(idx)
        for idx, row in target.iterrows()
    }

    extra = []

    for s1_idx, row in source1.iterrows():
        sid = str(row.entity_id)
        for target_id in truth.get(sid, set()):
            tid = str(target_id)
            if tid not in target_lookup:
                continue

            key = (sid, tid)
            if key in existing:
                continue

            t_idx = target_lookup[tid]
            t_row = target.iloc[t_idx]

            extra.append({
                "source1_index": int(s1_idx),
                "source1_entity_id": sid,
                "target_index": int(t_idx),
                "candidate_entity_id": tid,
                "target_source": str(t_row["source"]),
            })
            existing.add(key)

    if not extra:
        return pairs.copy()

    return pd.concat(
        [pairs, pd.DataFrame(extra)],
        ignore_index=True,
    )


def sample_training_pairs(pairs, truth, max_negative_per_entity, seed):
    labeled = label_pairs(pairs, truth)
    parts = []

    for sid, group in labeled.groupby("source1_entity_id", sort=False):
        pos = group[group.label == 1]
        neg = group[group.label == 0]

        if len(neg) > max_negative_per_entity:
            neg = neg.sample(
                n=max_negative_per_entity,
                random_state=seed,
            )

        parts.append(pd.concat([pos, neg], ignore_index=True))

    if not parts:
        return labeled.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)


def train(pairs, source1, target, truth, seed, max_negative_per_entity):
    pairs = add_ground_truth_pairs(pairs, source1, target, truth)
    train_pairs = sample_training_pairs(
        pairs,
        truth,
        max_negative_per_entity=max_negative_per_entity,
        seed=seed,
    )

    X = matrix_for_pairs(source1, target, train_pairs)
    y = train_pairs["label"].to_numpy(dtype=int)

    if len(X) == 0:
        raise RuntimeError("No training pairs were produced.")
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training pairs contain only one class. "
            "Increase candidate generation or check ground truth."
        )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=seed,
        )),
    ])

    model.fit(X, y)
    return model, train_pairs


def score(model, source1, target, pairs):
    X = matrix_for_pairs(source1, target, pairs)
    if len(X) == 0:
        return np.empty(0, dtype=float)
    return model.predict_proba(X)[:, 1]
