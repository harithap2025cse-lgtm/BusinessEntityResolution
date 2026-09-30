from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .normalize import enrich


@dataclass
class Blocker:
    target: pd.DataFrame
    exact_name: dict
    name_prefix: dict
    postal: dict
    address_token: dict
    vectorizer: TfidfVectorizer
    nn: NearestNeighbors
    top_k_tfidf: int
    max_candidates: int
    block_key_limit: int

    @classmethod
    def fit(
        cls,
        target: pd.DataFrame,
        top_k_tfidf: int = 25,
        max_candidates: int = 80,
        block_key_limit: int = 40,
    ):
        target = enrich(target).reset_index(drop=True)

        exact_name = defaultdict(list)
        name_prefix = defaultdict(list)
        postal = defaultdict(list)
        address_token = defaultdict(list)

        for idx, row in target.iterrows():
            if row.norm_name:
                exact_name[row.norm_name].append(idx)
            if row.name_prefix:
                name_prefix[row.name_prefix].append(idx)
            if row.postal:
                postal[row.postal].append(idx)

            seen = set()
            for token in row.address_tokens:
                if len(token) >= 3 and token not in seen:
                    address_token[token].append(idx)
                    seen.add(token)

        text = (
            target["norm_name"].fillna("")
            + " "
            + target["norm_address"].fillna("")
        ).str.strip()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=100_000,
        )
        matrix = vectorizer.fit_transform(text)

        k = max(1, min(top_k_tfidf, len(target)))
        nn = NearestNeighbors(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
        )
        nn.fit(matrix)

        return cls(
            target=target,
            exact_name=dict(exact_name),
            name_prefix=dict(name_prefix),
            postal=dict(postal),
            address_token=dict(address_token),
            vectorizer=vectorizer,
            nn=nn,
            top_k_tfidf=top_k_tfidf,
            max_candidates=max_candidates,
            block_key_limit=block_key_limit,
        )

    def _add_limited(self, selected, seen, values):
        for idx in values[:self.block_key_limit]:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) >= self.max_candidates:
                break

    def generate(self, source1: pd.DataFrame) -> pd.DataFrame:
        query = enrich(source1).reset_index(drop=True)

        text = (
            query["norm_name"].fillna("")
            + " "
            + query["norm_address"].fillna("")
        ).str.strip()

        matrix = self.vectorizer.transform(text)
        _, nn_indices = self.nn.kneighbors(matrix)

        rows = []

        for q_idx, q in query.iterrows():
            selected = []
            seen = set()

            self._add_limited(
                selected, seen,
                self.exact_name.get(q.norm_name, []) if q.norm_name else []
            )

            self._add_limited(
                selected, seen,
                self.postal.get(q.postal, []) if q.postal else []
            )

            self._add_limited(
                selected, seen,
                self.name_prefix.get(q.name_prefix, []) if q.name_prefix else []
            )

            for token in q.address_tokens[:3]:
                if len(token) < 3:
                    continue
                self._add_limited(selected, seen, self.address_token.get(token, []))
                if len(selected) >= self.max_candidates:
                    break

            for idx in nn_indices[q_idx]:
                idx = int(idx)
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
                if len(selected) >= self.max_candidates:
                    break

            for target_idx in selected:
                target_row = self.target.iloc[target_idx]
                rows.append(
                    {
                        "source1_index": q_idx,
                        "source1_entity_id": str(q.entity_id),
                        "target_index": target_idx,
                        "candidate_entity_id": str(target_row.entity_id),
                        "target_source": str(target_row.source),
                    }
                )

        return pd.DataFrame(rows, columns=[
            "source1_index",
            "source1_entity_id",
            "target_index",
            "candidate_entity_id",
            "target_source",
        ])
