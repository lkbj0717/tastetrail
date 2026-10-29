"""Local v13 demonstration. Run on 127.0.0.1:8773; no API work on GET.

The recommendation implementation lives in service_v13.recommend. This module
owns only input validation, a small HTTP adapter, and the demonstration page.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parent
PRIVATE = Path(os.environ.get("V13_PRIVATE", str(Path.home() / ".local/share/tastetrail/culture_v13")))
LOCK = threading.Lock()
app = FastAPI(title="TasteTrail · Cultural preference comparison", version="0.13")


class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=120)
    year: int = Field(ge=1880, le=2030)


class Rated(Movie):
    rating: int = Field(ge=1, le=5)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    likes: list[Rated] = Field(min_length=5, max_length=5)
    dislikes: list[Rated] = Field(min_length=3, max_length=3)
    candidates: list[Movie] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_visible_ratings(self):
        if any(m.rating not in (4, 5) for m in self.likes):
            raise ValueError("Liked films must have a rating of 4 or 5.")
        if any(m.rating not in (1, 2) for m in self.dislikes):
            raise ValueError("Disliked films must have a rating of 1 or 2.")
        movies = self.likes + self.dislikes + self.candidates
        keys = [(re.sub(r"\s+", " ", m.title).casefold(), m.year) for m in movies]
        if len(set(keys)) != 10:
            raise ValueError("Choose ten distinct films: eight history films and two new candidates.")
        return self


def _read(path):
    return json.loads(Path(path).read_text())


def _metadata_path():
    local = PRIVATE / "metadata.json"
    return local if local.exists() else PRIVATE.parent / "culture_v11/metadata.json"


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "demo_v13.html").read_text()


@app.get("/health")
@app.get("/api/health", include_in_schema=False)
def health():
    # Module discovery does not import or invoke the paid service.
    return {"ok": True, "version": "0.13", "catalog_available": _metadata_path().exists(),
            "service_available": importlib.util.find_spec("service_v13") is not None}


@app.get("/api/catalog")
@app.get("/catalog", include_in_schema=False)
def catalog():
    try:
        metadata = _read(_metadata_path())
    except (OSError, ValueError):
        raise HTTPException(503, "The research film catalog is unavailable.") from None
    # Whitelist public fields. Do not expose internal IDs or private outcomes.
    return sorted([{k: row.get(k) for k in ("title", "year", "genres")}
                   for row in metadata.values()], key=lambda row: (row["title"], row["year"]))


@app.get("/api/example")
@app.get("/example", include_in_schema=False)
def example():
    local = ROOT / "example_request.json"
    path = local if local.exists() else ROOT.parent / "culture_v11/example_request.json"
    try:
        return Request.model_validate(_read(path)).model_dump()
    except (OSError, ValueError):
        raise HTTPException(503, "The fictional example is unavailable.") from None


@app.get("/api/research")
def research():
    """Show only a published aggregate summary, never evaluation case records."""
    path = ROOT / "research/confirmation_report.json"
    if not path.exists():
        return {"available": False, "message": "No confirmation results have been published for this version."}
    try:
        report = _read(path)
    except (OSError, ValueError):
        return {"available": False, "message": "The confirmation summary is currently unavailable."}
    summary = {key: report[key] for key in ("status", "selected_version", "sample_size", "evaluation_status")
               if key in report and isinstance(report[key], (str, int, float))}
    methods = report.get("methods", {})
    if isinstance(methods, dict):
        summary["methods"] = {
            name: {k: row[k] for k in ("correct", "total", "accuracy")
                   if k in row and isinstance(row[k], (int, float))}
            for name, row in methods.items() if name in ("glm", "agent", "agent_qloo", "final")
            and isinstance(row, dict)
        }
    return {"available": True, "summary": summary,
            "message": "A confirmation report is available. This film comparison does not validate city itinerary planning."}


def endpoint(request: Request, compare: bool):
    try:
        # Lazy import permits catalog/UI checks without loading model clients.
        from service_v13 import recommend
        with LOCK:
            result = recommend(request.model_dump(), compare=compare)
        if not isinstance(result, dict):
            raise TypeError("Recommendation service must return an object")
        return result
    except ImportError:
        raise HTTPException(503, "The recommendation service is not available yet.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except Exception:
        # Do not expose provider errors, file paths, credentials, or tracebacks.
        raise HTTPException(502, "The recommendation could not be completed. Please check the service log and retry.") from None


@app.post("/api/compare")
def compare(request: Request):
    return endpoint(request, True)


@app.post("/api/recommend")
def recommend(request: Request):
    return endpoint(request, False)
