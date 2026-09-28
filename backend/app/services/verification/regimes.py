"""Interpretable weather-regime detection from FORECAST-TIME information.

A regime must be knowable when the forecast is issued, so features come only
from the models' forecast fields for the verification window (never from the
IMD observation of that window). Observations are used afterwards, on the
training period only, to describe and name the regimes.

Pipeline: features -> StandardScaler -> PCA (>= 80 % variance, <= 4 comps)
-> KMeans. Everything is fitted on training samples only. The fitted model
is persisted as JSON and re-applied with plain numpy, so validation and test
samples receive exactly the training transformation.

Limitation: only precipitation and 10 m wind are available (no MSLP,
geopotential or humidity are ingested), so these are rainfall/surface-wind
regimes, not full synoptic regimes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from app.services.verification.regions import REGIONS

HEAVY_MM = 64.5
LIGHT_MM = 2.5

# Ocean boxes for monsoon-flow proxies (ocean = outside the IMD land mask).
OCEAN_BOXES = {
    "wind_arabian_sea": {"lat": (8.0, 20.0), "lon": (68.0, 72.5)},
    "wind_bay_of_bengal": {"lat": (12.0, 20.0), "lon": (85.0, 92.0)},
}

FEATURE_NAMES = (
    [f"log_rain_{r.lower()}" for r in REGIONS]
    + ["frac_rain_ge_2.5mm", "frac_rain_ge_64.5mm", "model_disagreement"]
    + list(OCEAN_BOXES)
)


# --------------------------------------------------------------------------- features


def ocean_box_masks(latitudes: np.ndarray, longitudes: np.ndarray, land: xr.DataArray) -> dict[str, xr.DataArray]:
    lat = xr.DataArray(latitudes, dims="latitude", coords={"latitude": latitudes})
    lon = xr.DataArray(longitudes, dims="longitude", coords={"longitude": longitudes})
    out = {}
    for name, box in OCEAN_BOXES.items():
        inside = (lat >= box["lat"][0]) & (lat <= box["lat"][1]) & (lon >= box["lon"][0]) & (lon <= box["lon"][1])
        out[name] = inside & ~land
    return out


def regime_features(
    forecast: xr.DataArray,
    land: xr.DataArray,
    regions: xr.DataArray,
    wind: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One row per sample. ``forecast`` dims (model, sample, latitude, longitude), mm / 24 h.

    Rain features use the mean of the models available for each sample.
    ``wind`` maps each OCEAN_BOXES name to a per-sample array (m/s); missing
    wind features are left out rather than invented.
    """
    mean_fc = forecast.mean("model", skipna=True).where(land)
    rows: dict[str, np.ndarray] = {}
    for r in REGIONS:
        rows[f"log_rain_{r.lower()}"] = np.log1p(mean_fc.where(regions == r).mean(("latitude", "longitude")).values)
    n_land = float(land.sum())
    rows["frac_rain_ge_2.5mm"] = ((mean_fc >= LIGHT_MM).sum(("latitude", "longitude")) / n_land).values
    rows["frac_rain_ge_64.5mm"] = ((mean_fc >= HEAVY_MM).sum(("latitude", "longitude")) / n_land).values
    if forecast.sizes["model"] >= 2:
        spread = np.abs(forecast.isel(model=0) - forecast.isel(model=1)).where(land).mean(("latitude", "longitude"))
        rows["model_disagreement"] = (spread / (mean_fc.mean(("latitude", "longitude")) + 1.0)).values
    else:
        rows["model_disagreement"] = np.zeros(forecast.sizes["sample"])
    for name in OCEAN_BOXES:
        if wind is not None and name in wind:
            rows[name] = np.asarray(wind[name], dtype=float)
    df = pd.DataFrame(rows, index=pd.Index(forecast.sample.values, name="sample"))
    # A feature undefined for EVERY sample (region or ocean box outside the configured domain)
    # is a property of the domain: drop it and record it. Partial NaN means bad data: raise.
    absent = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=absent)
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"regime features contain NaN for some samples: {bad}")
    df.attrs["dropped_features"] = absent
    return df


# --------------------------------------------------------------------------- model


@dataclass
class RegimeModel:
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray
    centers: np.ndarray  # in PCA space, rows ordered by regime id
    k: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- fit (sklearn) ----------------------------------------------------
    @classmethod
    def fit(cls, features: pd.DataFrame, k: int, seed: int = 0, variance: float = 0.80, max_components: int = 4,
            order_by: str = "frac_rain_ge_2.5mm") -> "RegimeModel":
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = features.to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        full = PCA(random_state=seed).fit(Z)
        n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), variance) + 1)
        n = max(1, min(n, max_components, Z.shape[1]))
        pca = PCA(n_components=n, random_state=seed).fit(Z)
        P = pca.transform(Z)
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(P)
        # Deterministic regime ids: ascending mean of ``order_by`` (rain coverage) per cluster.
        col = features.columns.get_loc(order_by)
        means = [X[km.labels_ == c, col].mean() for c in range(k)]
        order = np.argsort(means)
        model = cls(
            feature_names=list(features.columns),
            scaler_mean=scaler.mean_.copy(),
            scaler_scale=scaler.scale_.copy(),
            pca_mean=pca.mean_.copy(),
            pca_components=pca.components_.copy(),
            explained_variance_ratio=pca.explained_variance_ratio_.copy(),
            centers=km.cluster_centers_[order].copy(),
            k=k,
            metadata={"seed": seed, "variance_target": variance, "order_by": order_by,
                      "n_fit_samples": int(X.shape[0]), "kmeans_inertia": float(km.inertia_)},
        )
        return model

    # ---- apply (numpy only) ---------------------------------------------
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"features missing for regime model: {missing}")
        X = features[self.feature_names].to_numpy(dtype=float)
        Z = (X - self.scaler_mean) / self.scaler_scale
        return (Z - self.pca_mean) @ self.pca_components.T

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        P = self.transform(features)
        d = ((P[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "pca_mean": self.pca_mean.tolist(),
            "pca_components": self.pca_components.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "centers": self.centers.tolist(),
            "k": self.k,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegimeModel":
        arr = {k: np.asarray(d[k], dtype=float) for k in
               ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "explained_variance_ratio", "centers")}
        return cls(feature_names=list(d["feature_names"]), k=int(d["k"]), metadata=dict(d.get("metadata", {})), **arr)

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.to_dict(), **(extra or {})}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RegimeModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- choosing k


def select_k(
    features: pd.DataFrame, ks: Iterable[int] = (2, 3, 4), seed: int = 0, min_cluster_size: int = 5,
    min_stability: float = 0.6, n_boot: int = 50,
) -> tuple[int, pd.DataFrame]:
    """Pick k on TRAINING features: max silhouette among k meeting size and stability constraints."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(seed)
    rows = []
    n_distinct = features.drop_duplicates().shape[0]
    for k in ks:
        if k >= n_distinct:  # cannot form k distinct clusters
            continue
        model = RegimeModel.fit(features, k, seed=seed)
        labels = model.predict(features)
        P = model.transform(features)
        sil = float(silhouette_score(P, labels)) if len(set(labels)) > 1 else float("nan")
        aris = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(features), len(features))
            boot = features.iloc[idx]
            if boot.drop_duplicates().shape[0] <= k:
                continue
            aris.append(adjusted_rand_score(labels, RegimeModel.fit(boot, k, seed=seed).predict(features)))
        sizes = np.bincount(labels, minlength=k)
        stability = float(np.mean(aris)) if aris else float("nan")
        rows.append({
            "k": k, "silhouette": sil, "stability_ari": stability, "min_cluster_size": int(sizes.min()),
            "cluster_sizes": sizes.tolist(), "n_components": int(model.pca_components.shape[0]),
            "eligible": bool(sizes.min() >= min_cluster_size and stability >= min_stability),
        })
    table = pd.DataFrame(rows)
    if table.empty:  # no feature variability: clustering is meaningless, use one regime
        return 1, pd.DataFrame([{"k": 1, "silhouette": float("nan"), "stability_ari": float("nan"),
                                 "min_cluster_size": len(features), "cluster_sizes": [len(features)],
                                 "n_components": 1, "eligible": False, "selected": True,
                                 "selection_note": f"only {n_distinct} distinct feature vector(s); single regime"}])
    eligible = table[table.eligible]
    pool = eligible if not eligible.empty else table
    best = pool.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    table["selected"] = table.k == best.k
    table["selection_note"] = "" if not eligible.empty else "no k met size/stability constraints; max silhouette used"
    return int(best.k), table


# --------------------------------------------------------------------------- description


def regime_statistics(
    samples: xr.Dataset, labels: np.ndarray, features: pd.DataFrame, regions: xr.DataArray, land: xr.DataArray,
) -> pd.DataFrame:
    """Per-regime statistics from the FIT-period samples (forecasts + IMD) and data-derived names."""
    obs = samples["observation"].where(land)
    mean_fc = samples["forecast"].mean("model", skipna=True).where(land)
    overall_heavy = float((obs >= HEAVY_MM).sum() / obs.notnull().sum())
    rows = []
    for r in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == r)
        o = obs.isel(sample=idx)
        f = mean_fc.isel(sample=idx)
        n_pts = int(o.notnull().sum())
        heavy = float((o >= HEAVY_MM).sum()) / n_pts if n_pts else float("nan")
        reg_means = {reg: float(o.where(regions == reg).mean()) for reg in REGIONS}
        row = {
            "regime": int(r),
            "n_samples": int(idx.size),
            "n_days": int(len(set(samples.valid_date.values[idx].tolist()))),
            "samples_lead_day_1": int((samples.lead_day.values[idx] == 1).sum()),
            "samples_lead_day_2": int((samples.lead_day.values[idx] == 2).sum()),
            "samples_lead_day_3": int((samples.lead_day.values[idx] == 3).sum()),
            "obs_mean_mm": float(o.mean()),
            "obs_variance_mm2": float(o.var()),
            "obs_heavy_frequency": heavy,
            "heavy_frequency_ratio": heavy / overall_heavy if overall_heavy > 0 else float("nan"),
            "fcst_mean_mm": float(f.mean()),
            "fcst_heavy_frequency": float((f >= HEAVY_MM).sum()) / max(int(f.notnull().sum()), 1),
            "wettest_region_obs": max(reg_means, key=reg_means.get),
            **{f"obs_mean_{reg.lower()}_mm": v for reg, v in reg_means.items()},
            **{f"feature_mean_{c}": float(features.iloc[idx][c].mean()) for c in features.columns},
        }
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats["overall_obs_heavy_frequency"] = overall_heavy
    stats["is_heavy_regime"] = stats.obs_heavy_frequency > overall_heavy
    stats["name"] = [regime_name(r, stats) for r in stats.itertuples()]
    return stats


def regime_name(r, stats: pd.DataFrame) -> str:
    """Name built only from statistics: intensity class from the heavy-rain frequency ratio,
    plus the wettest region."""
    ratio = r.heavy_frequency_ratio
    if not np.isfinite(ratio):
        cls = "UNDETERMINED"
    elif ratio > 1.5:
        cls = "HEAVY RAINFALL"
    elif ratio < 0.67:
        cls = "LOW RAINFALL"
    else:
        cls = "MODERATE RAINFALL"
    return f"REGIME_{r.regime}: {cls} (wettest: {r.wettest_region_obs})"
