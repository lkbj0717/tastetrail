"""Private paths and atomic JSON storage for runtime asset construction."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIVATE = Path(os.environ.get('V11_PRIVATE', str(Path.home() / '.local/share/tastetrail/culture_v11')))
SEED = 20261025


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)
    if PRIVATE in path.parents:
        path.chmod(0o600)
