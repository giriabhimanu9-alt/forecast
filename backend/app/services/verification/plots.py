"""Static PNG figures for a verification experiment (matplotlib, headless).

Colours follow the validated reference palette: categorical slots in fixed
order (blue, orange, aqua, yellow, magenta), a single-hue blue ramp for
magnitude, blue<->red with a grey midpoint for signed differences. Bars
carry direct value labels because three slots are below 3:1 contrast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
).with_extremes(bad=(1, 1, 1, 0))
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#6da7ec", "#f0efec", "#f08c8b", "#b8322f"]
).with_extremes(bad=(1, 1, 1, 0))

METHOD_LABELS = {
    "equal": "Equal blend",
    "inverse_error": "Inverse-error blend",
    "regional": "Regional blend",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def _fig(nrows: int, ncols: int, w: float, h: float):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, facecolor=SURFACE)
    return fig, axes


def _map(ax, da: xr.DataArray, cmap, norm=None, vmin=None, vmax=None):
    _style(ax)
    mesh = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                         shading="nearest")
    ax.set_aspect("equal")
    ax.set_xlim(float(da.longitude.min()), float(da.longitude.max()))
    ax.set_ylim(float(da.latitude.min()), float(da.latitude.max()))
    ax.grid(color=GRID, linewidth=0.5)
    return mesh


def label_of(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def skill_comparison(scores: pd.DataFrame, methods: list[str], path: Path, title: str) -> None:
    """Grouped bars of test MAE per lead day, one colour per method (fixed order)."""
    lead_days = sorted(d for d in scores.lead_day.unique() if d != "all")
    groups = [str(d) for d in lead_days] + (["all"] if "all" in scores.lead_day.astype(str).values else [])
    fig, axes = _fig(1, 1, 9, 4.6)
    ax = axes[0, 0]
    _style(ax)
    width = 0.8 / len(methods)
    x = np.arange(len(groups))
    for i, method in enumerate(methods):
        vals = []
        for g in groups:
            row = scores[(scores.method == method) & (scores.lead_day.astype(str) == g)]
            vals.append(float(row.mae.iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width * 0.92, color=CATEGORICAL[i % 5],
                      label=label_of(method), edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT_2)
    ax.set_xticks(x, [f"Day {g}" if g != "all" else "All lead days" for g in groups])
    ax.set_ylabel("Test MAE (mm / 24 h)", color=TEXT_2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncols=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def rainfall_vs_imd(fields: dict[str, xr.DataArray], path: Path, title: str) -> None:
    """Small multiples of mean daily rainfall on one shared sequential scale."""
    names = list(fields)
    ncols = 2 if len(names) == 4 else min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = _fig(nrows, ncols, 4.2 * ncols, 4.2 * nrows + 0.6)
    vmax = float(np.nanpercentile(np.concatenate([f.values.ravel() for f in fields.values()]), 99))
    mesh = None
    for ax, name in zip(axes.ravel(), names):
        mesh = _map(ax, fields[name], SEQ_BLUE, vmin=0, vmax=vmax)
        ax.set_title(name, color=TEXT, fontsize=10, loc="left")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7, extend="max")
    cbar.set_label("mm / day", color=TEXT_2, fontsize=9)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def weight_map(weights: dict[tuple[str, int], xr.DataArray], regions: xr.DataArray, path: Path, title: str) -> None:
    """Regional-blend weights on a diverging scale centred at equal weight (1/N), labelled per region.

    With two models only the first is drawn (the other is 1 - w at every point).
    """
    models = sorted({m for m, _ in weights})
    days = sorted({d for _, d in weights})
    shown = models[:1] if len(models) == 2 else models
    centre = 1.0 / len(models)
    dev = max(0.1, max(float(np.nanmax(np.abs(w.values - centre))) for w in weights.values()))
    # One model: its weight is 1 everywhere, so there is no "equal weight" midpoint to diverge from.
    norm = (TwoSlopeNorm(vcenter=centre, vmin=max(0.0, centre - dev), vmax=min(1.0, centre + dev))
            if len(models) > 1 else Normalize(vmin=0.0, vmax=1.0))
    fig, axes = _fig(len(shown), len(days), 3.8 * len(days), 3.9 * len(shown) + 0.8)
    mesh = None
    names = np.unique(regions.values)
    for i, m in enumerate(shown):
        for j, d in enumerate(days):
            ax = axes[i, j]
            w = weights[(m, d)]
            mesh = _map(ax, w, DIVERGING.reversed(), norm=norm)
            for name in names:
                sel = w.where(regions == name)
                if not bool(sel.notnull().any()):
                    continue
                lat = float(sel.latitude.where(sel.notnull().any("longitude")).mean())
                lon = float(sel.longitude.where(sel.notnull().any("latitude")).mean())
                ax.text(lon, lat, f"{float(sel.mean()):.2f}", ha="center", va="center", fontsize=7, color=TEXT,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8})
            ax.set_title(f"{m} weight, lead day {d}", color=TEXT, fontsize=9, loc="left")
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    other = f"; {models[1]} = 1 - w" if len(models) == 2 else ""
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label(f"weight (grey = equal weight {centre:.2f}{other})", color=TEXT_2, fontsize=9)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def error_map(mae_by_model: dict[str, xr.DataArray], path: Path, title: str) -> None:
    """Per-grid-point training MAE per model, plus the signed difference of the first two."""
    names = list(mae_by_model)
    panels = len(names) + (1 if len(names) >= 2 else 0)
    fig, axes = _fig(1, panels, 4.2 * panels, 4.8)
    vmax = float(np.nanpercentile(np.concatenate([f.values.ravel() for f in mae_by_model.values()]), 98))
    mesh = None
    for ax, name in zip(axes[0], names):
        mesh = _map(ax, mae_by_model[name], SEQ_BLUE, vmin=0, vmax=vmax)
        ax.set_title(f"{name} MAE", color=TEXT, fontsize=10, loc="left")
    fig.colorbar(mesh, ax=axes[0, : len(names)].tolist(), shrink=0.75, extend="max").set_label(
        "MAE, mm / 24 h", color=TEXT_2, fontsize=9)
    if len(names) >= 2:
        a, b = names[0], names[1]
        diff = mae_by_model[a] - mae_by_model[b]
        lim = float(np.nanpercentile(np.abs(diff.values), 98)) or 1.0
        ax = axes[0, -1]
        dm = _map(ax, diff, DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
        ax.set_title(f"{a} minus {b}", color=TEXT, fontsize=10, loc="left")
        fig.colorbar(dm, ax=ax, shrink=0.75, extend="both").set_label(
            f"red: {a} worse, blue: {b} worse", color=TEXT_2, fontsize=9)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== Phase 3

CATEGORICAL_6 = CATEGORICAL + ["#008300"]
METHOD_LABELS.update({
    "phase2_inverse_error": "Phase 2 inverse-error",
    "regime": "Phase 3 regime blend",
    "extreme_aware": "Phase 3 extreme-aware",
})
REGIME_MARKERS = ["o", "s", "^", "D"]


def _short(name: str) -> str:
    return name.split(":")[0]


def _bars(ax, groups, series, values, colors, fmt="{:.2f}", label_groups=None):
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [values.get((g, s), np.nan) for g in groups]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width * 0.92, color=colors[i % len(colors)],
                      label=label_of(s), edgecolor=SURFACE, linewidth=1)
        for g, b, v in zip(groups, bars, vals):
            if np.isfinite(v) and (label_groups is None or g in label_groups):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5,
                        color=TEXT_2, rotation=90 if len(series) > 4 else 0)
    ax.set_xticks(x, groups)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def phase3_regime_distribution(assign: pd.DataFrame, model, feats: pd.DataFrame, path: Path, names: dict) -> None:
    """Left: samples per regime and period. Right: PCA space of the regime model (fit vs test)."""
    fig, axes = _fig(1, 2, 12, 4.8)
    ax = axes[0, 0]
    _style(ax)
    periods = list(dict.fromkeys(assign.period))
    regimes = sorted(assign.regime.unique())
    width = 0.8 / len(regimes)
    x = np.arange(len(periods))
    for i, r in enumerate(regimes):
        v = [float(((assign.period == p) & (assign.regime == r)).sum()) for p in periods]
        bars = ax.bar(x + (i - (len(regimes) - 1) / 2) * width, v, width * 0.92, color=CATEGORICAL[i], edgecolor=SURFACE,
                      label=names.get(r, f"REGIME_{r}"))
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{int(val)}", ha="center", va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(x, periods)
    ax.set_ylabel("forecast cases (valid day x lead day)", color=TEXT_2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Regime frequency", color=TEXT, fontsize=10, loc="left")

    ax = axes[0, 1]
    _style(ax)
    P = model.transform(feats.loc[assign["sample"].values])
    if P.shape[1] == 1:
        P = np.column_stack([P[:, 0], np.zeros(len(P))])
    fit = (assign.period != "test").to_numpy()
    for i, r in enumerate(regimes):
        sel = assign.regime.to_numpy() == r
        ax.scatter(P[sel & fit, 0], P[sel & fit, 1], s=28, color=CATEGORICAL[i], marker=REGIME_MARKERS[i % 4],
                   edgecolor=SURFACE, linewidth=0.8, label=f"{_short(names.get(r, str(r)))} (fit)")
        ax.scatter(P[sel & ~fit, 0], P[sel & ~fit, 1], s=40, facecolor="none", edgecolor=CATEGORICAL[i],
                   marker=REGIME_MARKERS[i % 4], linewidth=1.4, label=f"{_short(names.get(r, str(r)))} (test)")
    C = model.centers if model.centers.shape[1] > 1 else np.column_stack([model.centers[:, 0], np.zeros(model.k)])
    ax.scatter(C[:, 0], C[:, 1], marker="x", s=70, color=TEXT, label="cluster centre")
    ev = model.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)", color=TEXT_2, fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})" if len(ev) > 1 else "(single component)", color=TEXT_2, fontsize=9)
    ax.grid(color=GRID, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Samples in regime-model space (filled: fit period, open: test)", color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def phase3_model_error_by_regime(scores: pd.DataFrame, models: list[str], names: dict, path: Path) -> None:
    """Fit-period MAE (top) and heavy-rain CSI (bottom) per regime, for each lead day and pooled."""
    sub = scores[scores.regime.astype(str) != "all"]
    leads = [d for d in ["1", "2", "3", "all"] if d in set(sub.lead_day.astype(str))]
    regimes = sorted(sub.regime.astype(int).unique())
    groups = [_short(names.get(r, f"REGIME_{r}")) for r in regimes]
    fig, axes = _fig(2, len(leads), 3.6 * len(leads), 7.2)
    for j, d in enumerate(leads):
        for i, (metric, ylab, fmt) in enumerate((("mae", "MAE (mm/24 h)", "{:.1f}"),
                                                 ("csi@64.5mm", "CSI >= 64.5 mm", "{:.3f}"))):
            ax = axes[i, j]
            _style(ax)
            vals = {}
            for r, g in zip(regimes, groups):
                for m in models:
                    row = sub[(sub.regime.astype(int) == r) & (sub.lead_day.astype(str) == d) & (sub.method == m)]
                    vals[(g, m)] = float(row[metric].iloc[0]) if len(row) else np.nan
            _bars(ax, groups, models, vals, CATEGORICAL, fmt)
            ax.set_title("Lead day " + d if d != "all" else "All lead days", color=TEXT, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(ylab, color=TEXT_2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model performance by regime, fit period (1-23 Jul 2025)", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def phase3_weights_by_regime(weights: pd.DataFrame, models: list[str], names: dict, path: Path) -> None:
    """Weight of the first model per (regime, region) for each lead day and method; annotated cells."""
    methods = [m for m in ("regime", "extreme_aware") if m in set(weights.method)]
    leads = sorted(weights.lead_day.unique())
    m0 = models[0]
    regions = list(dict.fromkeys(weights.region))
    regimes = sorted(weights.regime.unique())
    n = len(models)
    centre = 1.0 / n
    fig, axes = _fig(len(methods), len(leads), 4.3 * len(leads), 2.6 + 0.6 * len(regimes) * len(methods))
    fig.subplots_adjust(wspace=0.08, hspace=0.45)
    norm = TwoSlopeNorm(vcenter=centre, vmin=0.0, vmax=1.0) if n > 1 else Normalize(0, 1)
    im = None
    for i, method in enumerate(methods):
        for j, d in enumerate(leads):
            ax = axes[i, j]
            sub = weights[(weights.method == method) & (weights.lead_day == d) & (weights.model == m0)]
            cell = {(int(r.regime), r.region): r for r in sub.itertuples()}
            grid = np.array([[cell[(r, reg)].weight for reg in regions] for r in regimes], dtype=float)
            im = ax.imshow(grid, cmap=DIVERGING.reversed(), norm=norm, aspect="auto")
            for a, r in enumerate(regimes):
                for b, reg in enumerate(regions):
                    lvl = str(cell[(r, reg)].level)
                    mark = "*" if "static" in lvl else ("c" if "CSI" in lvl else "")
                    ax.text(b, a, f"{grid[a, b]:.2f}{mark}", ha="center", va="center", fontsize=7, color=TEXT)
            ax.set_xticks(range(len(regions)), regions, fontsize=7, rotation=30)
            ax.set_yticks(range(len(regimes)), [_short(names.get(r, str(r))) for r in regimes] if j == 0 else [],
                          fontsize=7)
            ax.set_title(f"{label_of(method)}, lead day {d}", color=TEXT, fontsize=9, loc="left")
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(f"{m0} weight by regime and region (grey = equal weight {centre:.2f}; other model = 1 - w; "
                 "* = static fallback, c = CSI-based)", color=TEXT, fontsize=10, x=0.01, ha="left")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8).set_label(f"{m0} weight", color=TEXT_2, fontsize=9)
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def phase3_method_comparison(final: pd.DataFrame, methods: list[str], path: Path, title: str) -> None:
    sub = final[final.regime.astype(str) == "all"]
    methods = [m for m in methods if m in set(sub.method)]
    leads = [d for d in ["1", "2", "3", "all"] if d in set(sub.lead_day.astype(str))]
    groups = [f"Day {d}" if d != "all" else "All lead days" for d in leads]
    fig, axes = _fig(2, 1, 11, 8)
    for i, (metric, ylab) in enumerate((("mae", "MAE (mm/24 h)"), ("rmse", "RMSE (mm/24 h)"))):
        ax = axes[i, 0]
        _style(ax)
        vals = {(g, m): float(sub[(sub.lead_day.astype(str) == d) & (sub.method == m)][metric].iloc[0])
                for g, d in zip(groups, leads) for m in methods}
        _bars(ax, groups, methods, vals, CATEGORICAL_6, "{:.3f}")
        ax.set_ylabel(ylab, color=TEXT_2, fontsize=9)
        ax.margins(y=0.15)
    axes[0, 0].set_title(title, color=TEXT, fontsize=11, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, frameon=False, fontsize=8, ncols=6, loc="lower center")
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def phase3_heavy_rain(final: pd.DataFrame, methods: list[str], path: Path, title: str) -> None:
    sub = final[final.regime.astype(str) == "all"]
    methods = [m for m in methods if m in set(sub.method)]
    leads = [d for d in ["1", "2", "3", "all"] if d in set(sub.lead_day.astype(str))]
    groups = [f"Day {d}" if d != "all" else "All" for d in leads]
    fig, axes = _fig(1, 3, 16, 4.8)
    for j, (metric, lab) in enumerate((("csi@64.5mm", "CSI (higher is better)"),
                                       ("pod@64.5mm", "POD (higher is better)"),
                                       ("far@64.5mm", "FAR (lower is better)"))):
        ax = axes[0, j]
        _style(ax)
        vals = {(g, m): float(sub[(sub.lead_day.astype(str) == d) & (sub.method == m)][metric].iloc[0])
                for g, d in zip(groups, leads) for m in methods}
        _bars(ax, groups, methods, vals, CATEGORICAL_6, "{:.3f}", label_groups={"All"})
        ax.set_title(lab, color=TEXT, fontsize=10, loc="left")
        ax.margins(y=0.15)
    fig.suptitle(title + "  (values labelled for all lead days pooled; full table in extreme_rain_scores.csv)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.legend(handles, labels, frameon=False, fontsize=8, ncols=6, loc="lower center")
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== Phase 4


def phase4_scores_by_season(scores: pd.DataFrame, methods: list[str], path: Path, prefix: str) -> None:
    """MAE (top) and heavy-rain CSI (bottom) per season for individual models and the equal blend (all lead days)."""
    sub = scores[(scores.lead_day.astype(str) == "all") & (scores.regime.astype(str) == "all")]
    seasons = [str(s) for s in sorted(sub.season.unique())]
    fig, axes = _fig(2, 1, 10, 7.5)
    for i, (metric, ylab, fmt) in enumerate((("mae", "MAE (mm/24 h)", "{:.2f}"), ("csi@64.5mm", "CSI >= 64.5 mm", "{:.3f}"))):
        ax = axes[i, 0]
        _style(ax)
        vals = {(s, m): float(sub[(sub.season.astype(str) == s) & (sub.method == m)][metric].iloc[0])
                for s in seasons for m in methods if len(sub[(sub.season.astype(str) == s) & (sub.method == m)])}
        _bars(ax, seasons, methods, vals, CATEGORICAL_6, fmt)
        ax.set_ylabel(ylab, color=TEXT_2, fontsize=9)
        ax.margins(y=0.15)
    axes[0, 0].set_title(f"{prefix}: skill by season (all lead days; equal blend needs no training)", color=TEXT,
                         fontsize=11, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, frameon=False, fontsize=8, ncols=len(methods), loc="lower center")
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
