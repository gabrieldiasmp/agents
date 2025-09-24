import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model_name: str = os.getenv("NER_MODEL_NAME", "gemini-2.0-flash")
    webhook_url: str | None = os.getenv("WEBHOOK_URL")
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "8"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "5"))
    initial_backoff_s: float = float(os.getenv("INITIAL_BACKOFF_S", "0.5"))
    max_backoff_s: float = float(os.getenv("MAX_BACKOFF_S", "10.0"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
    output_dir: str = os.getenv(
        "OUTPUT_DIR",
        str((Path(__file__).resolve().parent / "output").as_posix()),
    )

settings = Settings()


