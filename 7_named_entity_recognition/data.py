from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset, Dataset


WHITESPACE_RE = re.compile(r"\s+")


def load_conll2003_test_split() -> Dataset:
    ds = load_dataset("conll2003", trust_remote_code=True)
    return ds["test"]


def get_label_mapping(dataset: Dataset) -> Tuple[List[str], Dict[str, int]]:
    names: List[str] = dataset.features["ner_tags"].feature.names  # type: ignore[attr-defined]
    name_to_id = {n: i for i, n in enumerate(names)}
    return names, name_to_id


def sanitize_tokens(tokens: List[str]) -> List[str]:
    sanitized: List[str] = []
    for t in tokens:
        t = t.replace("\u200b", "").strip()
        t = WHITESPACE_RE.sub(" ", t)
        if t:
            sanitized.append(t)
    return sanitized


def build_prompt(tokens: List[str], label_names: List[str]) -> str:
    joined_tokens = " ".join(tokens)
    labels_str = ", ".join(label_names)
    return (
        f"Tokens: {joined_tokens}\n"
        f"Label names (index aligned): [{labels_str}]\n"
        "Return a list of integers, one per token."
    )


def validate_predicted_ids(pred_ids: List[int], num_tokens: int, num_labels: int) -> bool:
    if len(pred_ids) != num_tokens:
        return False
    return all((isinstance(x, int) and 0 <= x < num_labels) for x in pred_ids)


