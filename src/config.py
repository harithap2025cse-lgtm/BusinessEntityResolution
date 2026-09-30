from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("dataset")
    output_dir: Path = Path("output")
    limit: int = 10_000
    seed: int = 42

    top_k_tfidf: int = 25
    max_candidates_per_source1: int = 80
    block_key_limit: int = 40
    max_negative_pairs_per_entity: int = 35

    validation_fraction: float = 0.20
    threshold_min: float = 0.20
    threshold_max: float = 0.95
    threshold_count: int = 31
