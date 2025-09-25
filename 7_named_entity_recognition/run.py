from __future__ import annotations

import asyncio
import os
import sys

from datasets import Dataset

# Support running as a script from project root ("agents")
# if __package__ is None or __package__ == "":
sys.path.append(os.path.dirname(__file__))
from data import load_conll2003_test_split  # type: ignore
from runner import run_dataset  # type: ignore

from config import settings  # type: ignore
# else:
#     from .data import load_conll2003_test_split
#     from .runner import run_dataset
#     from .config import settings


async def main() -> None:
    print("[NER] Environment loaded. Starting...")
    if not settings.api_key:
        raise ValueError("GOOGLE_API_KEY is required")
    print("[NER] Loading CoNLL2003 test split...")
    ds = load_conll2003_test_split()
    print(f"[NER] Dataset loaded with {len(ds)} rows. Limiting to 10.")
    ds_out: Dataset = await run_dataset(ds, limit=100)

    os.makedirs(settings.output_dir, exist_ok=True)
    parquet_path = os.path.join(settings.output_dir, "predictions.parquet")
    xlsx_path = os.path.join(settings.output_dir, "predictions.xlsx")
    print(f"[NER] Saving Parquet to {parquet_path}...")
    ds_out.to_parquet(parquet_path)
    print(f"[NER] Saving xlsx to {xlsx_path}...")
    df = ds_out.to_pandas()  # Convert to pandas DataFrame
    df.to_excel(xlsx_path, index=False)
    print(f"[NER] Saved predictions to {parquet_path} and {xlsx_path}")

    # Print first 3 outputs
    for i in range(min(10, len(ds_out))):
        row = ds_out[i]
        print("[NER] Sample", i, {"id": row.get("id"), "tokens": row.get("tokens"), "ner_tags": row.get("ner_tags"), "pred_ner_tags": row.get("pred_ner_tags")})


if __name__ == "__main__":
    asyncio.run(main())


