Named Entity Recognition Agent Workflow (Gemini via OpenAI SDK)

Contents
- Architecture overview
- Node sequence and configurations
- Connection mappings
- Environment variables
- Deployment and testing checklist
- Scaling recommendations

Quick start
1) Set env vars: GOOGLE_API_KEY, (optional) TAVILY_API_KEY, WEBHOOK_URL
2) Install: uv pip install -r requirements.txt or pip install -r requirements.txt
3) Run: python -m ner_pipeline.run

Sample input/output
- Input dataset: load_dataset("conll2003", trust_remote_code=True)["test"], using fields: id, tokens, pos_tags, chunk_tags, ner_tags
- Output: adds column pred_ner_tags (List[int]) per example and saves results to outputs/predictions.parquet

Edge cases handled
- Empty tokens or whitespace-only tokens → skipped with warning, labeled O (0)
- Extremely long sequences → chunked to model_max_tokens, merged back to token-aligned labels
- Non-ASCII or control chars → normalized/sanitized before prompting
- Transient API failures → exponential backoff retries with jitter
- Provider timeouts/overload → rate-limited concurrent workers


