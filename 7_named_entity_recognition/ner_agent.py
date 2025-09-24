from __future__ import annotations

from typing import List
from agents import Agent


NER_INSTRUCTIONS = (
    "You are a Named Entity Recognition model following the CoNLL-2003 label set and BIO2 scheme.\n"
    "Return ONLY a raw JSON array of integers (no text, no keys, no code fences), one integer per input token, "
    "aligned exactly to the token order.\n"
    "Use this exact label-to-ID mapping: "
    "{ 'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8 }.\n"
    "Constraints:\n"
    "- Output length must equal the number of input tokens.\n"
    "- All values must be integers in [0,8].\n"
    "- Do NOT include any words, explanations, JSON objects, or markdown—only the array like [0,1,2].\n"
    "Examples:\n"
    "1. Tokens: 'John lives in Berlin' → [1,0,5,6]   (B-PER,O,B-LOC,I-LOC)\n"
    "2. Tokens: 'Microsoft released Windows 11' → [3,0,7,0]   (B-ORG,O,B-MISC,O)\n"
    "3. Tokens: 'Barack Obama met Angela Merkel' → [1,2,0,1,2]   (B-PER,I-PER,O,B-PER,I-PER)\n"
    "4. Tokens: 'I love pizza' → [0,0,0]   (no entities)\n"
    "5. Tokens: 'The United Nations is in New York' → [0,3,4,0,0,5,6]   (B-ORG,I-ORG,O,O,B-LOC,I-LOC)\n"
)


def build_ner_agent(model=None) -> Agent:
    return Agent(
        name="NERAgent",
        instructions=NER_INSTRUCTIONS,
        output_type=str,
        model=model,
    )


