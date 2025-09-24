from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List

from datasets import Dataset
from agents import Runner, RunConfig

# Support running as a script from project root ("agents")
from config import settings  # type: ignore
from data import (  # type: ignore
    build_prompt,
    get_label_mapping,
    sanitize_tokens,
    validate_predicted_ids,
)
from model_provider import GEMINI_PROVIDER  # type: ignore
from ner_agent import build_ner_agent  # type: ignore


async def run_dataset(dataset: Dataset, limit: int = 10) -> Dataset:
    print("[NER] Preparing output directory...")
    os.makedirs(settings.output_dir, exist_ok=True)
    print("[NER] Extracting label mapping...")
    label_names, _ = get_label_mapping(dataset)
    num_labels = len(label_names)

    print("[NER] Building agent...")
    agent = build_ner_agent()

    n = min(limit, len(dataset))
    print(f"[NER] Beginning inference for {n} rows...")
    results: List[List[int]] = []
    for i in range(n):
        #tokens = sanitize_tokens(dataset[i]["tokens"])  # type: ignore[index]
        tokens = dataset[i]["tokens"]
        print(f"[NER] Row {i}: {len(tokens)} tokens")
        prompt = build_prompt(tokens, label_names)
        result = await Runner.run(agent, prompt, run_config=RunConfig(model_provider=GEMINI_PROVIDER))
        text = result.final_output if hasattr(result, "final_output") else str(result)
        arr = json.loads(text)
        preds = [int(x) for x in arr]
        #Basic validation; fallback to O if invalid length
        if not validate_predicted_ids(preds, len(tokens), num_labels):
            preds = [0] * len(tokens)
        results.append(preds)
        print(f"[NER] Row {i}: prediction length {len(preds)}")
    dataset = dataset.select(range(n))
    dataset = dataset.add_column("pred_ner_tags", results)
    print("[NER] Inference complete.")
    return dataset


