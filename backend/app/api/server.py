"""FastAPI layer over the existing backend (Phase 5). Run: ``uvicorn app.api.server:app`` from backend/.

The operational endpoints serve the GFS + ECMWF blend verified against IMD (Phase 4, Experiment A).
The AI benchmark endpoint serves Experiment B (GraphCast vs HRES against ERA5) separately; it is
never merged with, or compared against, IMD results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_settings
from app.models.schema import DataPipelineError
from app.services import operational as op

STATIC = Path(__file__).parent / "static"
MANIFEST = op.PHASE4 / "manifests" / "phase4_manifest.json"

app = FastAPI(title="PS 26081 blended rainfall forecast", version="0.5.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)  # grids and boundaries compress ~10x
app.mount("/static", StaticFiles(directory=STATIC), name="static")  # dashboard assets (state boundaries)
settings = load_settings()


def _run_or_latest(run: str | None) -> str:
    runs = op.list_runs(settings)
    if run:
        if run not in runs:
            raise HTTPException(404, f"run {run} not found")
        return run
    if not runs:
        raise HTTPException(404, "no operational run yet; POST /api/forecast/run")
    return runs[0]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "runs": len(op.list_runs(settings)),
            "phase4_artifacts": op.REGIME_MODEL.is_file() and op.REGIME_WEIGHTS.is_file()}


@app.get("/api/forecast/runs")
def runs() -> list[str]:
    return op.list_runs(settings)


@app.post("/api/forecast/run")
def run_forecast(init: str | None = Query(None, description="YYYY-MM-DDT00; default latest 00 UTC")) -> dict:
    try:
        when = datetime.fromisoformat(init) if init else None
        return op.build_operational_forecast(settings, when)
    except DataPipelineError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/forecast")
def forecast(run: str | None = None) -> dict:
    return op.load_summary(settings, _run_or_latest(run))


@app.get("/api/forecast/field")
def field(product: str = "blend", lead_day: int = Query(1, ge=1, le=3), run: str | None = None,
          stride: int = Query(1, ge=1, le=8)) -> dict:
    run = _run_or_latest(run)
    try:
        return {"run": run, **op.load_field(settings, run, product, lead_day, stride)}
    except KeyError as exc:
        raise HTTPException(404, f"product {product!r} not available for lead day {lead_day} (model missing?)") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"run {run} has no forecast grids (no model was available)") from exc


@app.get("/api/weights/regional")
def regional_weights() -> list[dict]:
    t = pd.read_csv(op.REGIME_WEIGHTS)
    t = t[(t.feature_set == "rain") & (t.method == "regime")]
    return t[["regime", "regime_name", "lead_day", "region", "model", "weight", "training_error", "sample_count", "level"]].to_dict("records")


def _manifest() -> dict:
    if not MANIFEST.is_file():
        raise HTTPException(404, "Phase 4 results not found; run scripts/run_phase4.py")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@app.get("/api/validation")
def validation() -> dict:
    m = _manifest()
    return {"experiment": "A - India operational", "reference": "IMD gridded rainfall (03->03 UTC)",
            "test_season": m["experiment_a"]["test_seasons"], "scores": m["results"]["A"]}


@app.get("/api/benchmark/ai")
def ai_benchmark() -> dict:
    m = _manifest()
    return {"experiment": "B - AI/NWP benchmark", "reference": "ERA5 reanalysis (00->00 UTC, 1.5 deg)",
            "test_season": m["experiment_b"]["test"], "scores": m["results"].get("B", {}),
            "warning": "Not comparable with IMD results. ERA5 favours GraphCast (trained on/initialised from ERA5). "
                       "GraphCast is not part of the operational blend."}
