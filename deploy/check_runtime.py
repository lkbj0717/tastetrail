"""Offline startup check. Does not download, train, or invoke provider APIs."""
import os
from pathlib import Path
import sys

from bootstrap_runtime import check_runtime, read, V11


def main():
    private = Path(os.environ.get("V13_PRIVATE", str(Path.home() / ".local/share/tastetrail/culture_v13")))
    old = private.parent / "culture_v11"
    configured_old = Path(os.environ.get("V11_PRIVATE", str(old)))
    if configured_old.resolve() != old.resolve():
        print("V11_PRIVATE must be the culture_v11 directory beside V13_PRIVATE.", file=sys.stderr)
        return 1
    try:
        check = check_runtime(old, private, read(V11 / "catalog_movie_ids.json")["movie_ids"], False)
    except (OSError, ValueError, KeyError, TypeError):
        check = {"ready": False}
    if not check.get("ready"):
        print("Private runtime is incomplete. Follow docs/DEPLOYMENT.md before starting inference.", file=sys.stderr)
        return 1
    print("Private runtime passed the offline startup check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
