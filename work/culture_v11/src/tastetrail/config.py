"""Server-side credentials. No secrets or API payloads belong in a public repository."""
import json
import os
from pathlib import Path


def settings():
    path = Path(os.environ.get("TASTETRAIL_CREDENTIALS", "~/.config/tastetrail/credentials.json")).expanduser()
    values = json.loads(path.read_text()) if path.is_file() else {}
    return {**values, **os.environ}


def private_dir():
    path = Path(os.environ.get("TASTETRAIL_PRIVATE_DIR", "~/.local/share/tastetrail")).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path
