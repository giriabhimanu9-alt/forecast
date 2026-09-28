"""Plug-in point for AI weather-model forecasts (GraphCast, Pangu-Weather, FourCastNet, AIFS, ...).

Phase 1 ships NO AI model and NO AI forecast data. This module only defines
the interface and a registry so that a provider can be added later without
touching the pipeline, verification or blending code.

To add one:

    from app.services.data.ai_model import AIModelProvider, register_ai_provider

    @register_ai_provider("graphcast-wb2")
    class GraphCastWB2Provider(AIModelProvider):
        name = "GraphCast"
        model_family = "GraphCast (DeepMind), via WeatherBench 2 Zarr"
        def is_available(self, init_time, lead_hours): ...
        def fetch(self, init_time, lead_hours, variables): ...      # cache under data/raw/<name>/
        def load_fields(self, files, init_time, lead_hours): ...   # same field keys as NWP providers

then set ``AI_MODEL_PROVIDER=graphcast-wb2``. Because it subclasses
:class:`ForecastProvider`, ``get_forecast`` runs the same normalisation and
the output follows the same data contract as GFS/ECMWF.

Candidate sources: WeatherBench 2 (gs://weatherbench2, historical GraphCast /
Pangu / FuXi forecasts on 0.25/1.5° grids, needs gcsfs + zarr) and ECMWF AIFS
open data (same layout as ``ecmwf.ECMWFProvider`` with ``model_path =
"aifs-single"``).
"""

from __future__ import annotations

from typing import Callable

from app.config import Settings
from app.models.schema import NotConfiguredError
from app.services.data.base import ForecastProvider


class AIModelProvider(ForecastProvider):
    """Base class for AI-model forecast providers."""

    kind = "ai_forecast"
    model_family: str = ""


_REGISTRY: dict[str, type[AIModelProvider]] = {}


def register_ai_provider(key: str) -> Callable[[type[AIModelProvider]], type[AIModelProvider]]:
    def decorator(cls: type[AIModelProvider]) -> type[AIModelProvider]:
        if key in _REGISTRY:
            raise ValueError(f"AI provider {key!r} already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def registered_ai_providers() -> list[str]:
    return sorted(_REGISTRY)


def create_ai_provider(settings: Settings) -> AIModelProvider | None:
    """Return the configured AI provider, or ``None`` when running plug-in-only."""
    key = settings.ai_model_provider
    if not key:
        return None
    if key not in _REGISTRY:
        raise NotConfiguredError(
            f"AI_MODEL_PROVIDER={key!r} is not a registered provider "
            f"(registered: {registered_ai_providers() or 'none'})"
        )
    return _REGISTRY[key](settings)
