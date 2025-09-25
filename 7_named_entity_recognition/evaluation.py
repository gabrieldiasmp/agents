from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def evaluate_ner_predictions(
    df: pd.DataFrame,
    id2label: Dict[int, str],
    true_col: str = "ner_tags",
    pred_col: str = "pred_ner_tags",
    digits: int = 2,
) -> Dict[str, Any]:
    """
    Compute entity-level precision, recall, and F1 and a per-label report for NER.

    Args:
        df: DataFrame with columns true_col and pred_col, each a list[int] per row.
        id2label: mapping from tag id to BIO label string, e.g. {0:'O', 1:'B-PER', ...}.
        true_col: name of ground-truth column.
        pred_col: name of prediction column.
        digits: number of digits in the textual report.

    Returns:
        {
            'overall': {'f1': float, 'precision': float, 'recall': float},
            'per_label': pd.DataFrame,  # precision/recall/f1-score/support by label
            'report_text': str,
        }
    """
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{true_col}' and '{pred_col}' columns.")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError("id2label must be a non-empty dict mapping int -> BIO tag str.")

    def ids_to_tags(seqs: List[List[int]]) -> List[List[str]]:
        out: List[List[str]] = []
        for seq in seqs:
            out.append([id2label[int(t)] for t in seq])
        return out

    y_true_ids: List[List[int]] = df[true_col].tolist()
    y_pred_ids: List[List[int]] = df[pred_col].tolist()
    if len(y_true_ids) != len(y_pred_ids):
        raise ValueError("true and pred lists must have the same number of sequences.")

    y_true = ids_to_tags(y_true_ids)
    y_pred = ids_to_tags(y_pred_ids)

    from seqeval.metrics import (
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )

    overall_precision = precision_score(y_true, y_pred)
    overall_recall = recall_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred)

    report_text = classification_report(y_true, y_pred, digits=digits)

    # Try to get a structured report for programmatic access.
    try:
        report_dict = classification_report(y_true, y_pred, digits=digits, output_dict=True)
        # Exclude averages for the per-label DataFrame; keep entity labels only.
        label_keys = [k for k in report_dict.keys() if k not in ("micro avg", "macro avg", "weighted avg")] 
        per_label_df = pd.DataFrame(report_dict).T.loc[label_keys]
    except TypeError:
        # Fallback to parsing the text report minimally.
        rows: List[Tuple[str, float, float, float, int]] = []
        for line in report_text.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                label, p, r, f1, sup = parts
                try:
                    rows.append((label, float(p), float(r), float(f1), int(sup)))
                except ValueError:
                    pass
        per_label_df = pd.DataFrame(
            rows, columns=["label", "precision", "recall", "f1-score", "support"]
        ).set_index("label")

    return {
        "overall": {
            "f1": float(overall_f1),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
        },
        "per_label": per_label_df,
        "report_text": report_text,
    }


__all__ = ["evaluate_ner_predictions"]


