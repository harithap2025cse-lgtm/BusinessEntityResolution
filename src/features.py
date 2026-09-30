import numpy as np
from rapidfuzz import fuzz


FEATURE_NAMES = [
    "name_ratio",
    "name_token_set",
    "name_partial",
    "name_jaccard",
    "address_ratio",
    "address_token_set",
    "address_jaccard",
    "country_equal",
    "postal_equal",
    "exact_name",
    "name_prefix_equal",
    "shared_address_tokens",
    "name_length_ratio",
    "address_length_ratio",
]


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _length_ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def one_pair(left, right):
    name_a, name_b = left["norm_name"], right["norm_name"]
    addr_a, addr_b = left["norm_address"], right["norm_address"]

    return [
        fuzz.ratio(name_a, name_b) / 100.0,
        fuzz.token_set_ratio(name_a, name_b) / 100.0,
        fuzz.partial_ratio(name_a, name_b) / 100.0,
        _jaccard(left["name_tokens"], right["name_tokens"]),
        fuzz.ratio(addr_a, addr_b) / 100.0,
        fuzz.token_set_ratio(addr_a, addr_b) / 100.0,
        _jaccard(left["address_tokens"], right["address_tokens"]),
        float(
            bool(left["norm_country"])
            and left["norm_country"] == right["norm_country"]
        ),
        float(
            bool(left["postal"])
            and left["postal"] == right["postal"]
        ),
        float(
            bool(name_a)
            and name_a == name_b
        ),
        float(
            bool(left["name_prefix"])
            and left["name_prefix"] == right["name_prefix"]
        ),
        float(
            len(
                set(left["address_tokens"])
                & set(right["address_tokens"])
            )
        ),
        _length_ratio(name_a, name_b),
        _length_ratio(addr_a, addr_b),
    ]


def matrix_for_pairs(source1, target, pairs):
    if len(pairs) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)

    rows = []
    for pair in pairs.itertuples(index=False):
        left = source1.iloc[int(pair.source1_index)]
        right = target.iloc[int(pair.target_index)]
        rows.append(one_pair(left, right))

    return np.asarray(rows, dtype=float)
