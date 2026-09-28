# Phase 3 implementation plan — regime-conditioned adaptive blending

Status: written before implementation, 2026-09-28. It is based on inspecting the Phase 1/2 code, the local data cache and `results/july2025/aligned_samples.nc`.

## 0. What already exists and is reused, not rebuilt

| Need | Existing code, reused as-is |
|---|---|
| Download, cache and normalise forecasts | `app/services/data/{gfs,ecmwf,base,normalization,validation}.py` |
| IMD observations, with the **corrected** convention (label D = 24 h ending 08:30 IST on D) | `app/services/data/observations.py`, `PERIOD_END_OFFSET` (unchanged) |
| Cycle planning, fetching | `verification/experiment.py` → `plan_init_times`, `fetch_forecasts`, `fetch_observations` |
| 24 h accumulation, strict alignment | `verification/accumulation.py`, `verification/alignment.py` |
| Metrics | `verification/metrics.py` (extended with POD, FAR and correlation; existing keys unchanged) |
| Inverse-error weights, blend application with missing-model renormalisation | `verification/blending.py` (`inverse_error_weights`, `apply_blend`, `learn_weights`) |
| Regional blocks | `verification/regions.py` (6 lat/lon boxes, **not** IMD subdivisions) |
| Chronological split / leakage guard | `verification/split.py` (`assert_no_leakage`) |
| Day-block bootstrap | `experiment.daily_abs_error`, `experiment.bootstrap_mae_difference` |

Small, backward-compatible changes: `fetch_forecasts` gains a `variables` argument (default: precipitation only, as before), and providers gain a `model_name` property plus a standard forecast view (Step 17).

## 1. Available variables

* **Downloaded so far (historical):** 24 h precipitation, GFS and ECMWF, 33 00Z cycles.
* **Retrievable from the same real files with existing providers:** 10 m wind (u, v → speed) and 2 m temperature. Phase 3 downloads **10 m wind** at the window start and end leads (3/27/51/75 h), about 440 MB.
* **Not ingested (documented limitation):** MSLP, geopotential, humidity, 850 hPa winds. They exist in the source GRIB files, but no provider extracts them yet. The regimes are therefore **rainfall + surface-wind regimes, not full synoptic regimes**.

## 2. Spatial resolution
Common 0.25° grid, 5–38°N × 68–98°E, 133 × 121 points. Only IMD land points (4,964) are scored.

## 3. Time resolution
One sample = one 24 h window (03 UTC → 03 UTC) × one lead day (1, 2, 3) from a 00Z cycle. 31 IMD days × 3 lead days = 93 samples.

## 4. Regime features (forecast-time information only)

The regime must be knowable **when the forecast is issued**. The IMD observation of the target day is not, so it is never a regime feature. For each sample, features are computed from the **mean of the available models' forecast fields for that window**:

| # | Feature | Meaning |
|---|---|---|
| 1–6 | log(1 + regional mean forecast rain), one per region (North, West, Central, East, South, Northeast) | where it rains |
| 7 | fraction of land forecast ≥ 2.5 mm | rain coverage |
| 8 | fraction of land forecast ≥ 64.5 mm | heavy-rain extent |
| 9 | relative model disagreement, mean\|GFS−ECMWF\| / (mean rain + 1) | forecast uncertainty |
| 10 | mean 10 m wind speed, Arabian Sea box (8–20°N, 68–72.5°E, ocean) | monsoon westerly strength |
| 11 | mean 10 m wind speed, Bay of Bengal box (12–20°N, 85–92°E, ocean) | Bay branch and depression activity |

The wind for a window is the average of the wind at its start and end leads.

## 5. Regime algorithm
StandardScaler → PCA (keep components explaining ≥ 80 % of variance, at most 4) → KMeans (`n_init=50`, fixed seed). All three are fitted **on training samples only**. Clusters are relabelled deterministically in ascending order of forecast rain coverage (fraction of land >= 2.5 mm). The fitted scaler, PCA, centres and metadata are saved to `regime_model.json`. A pure-numpy `transform`/`predict` reproduces the sklearn labels from the JSON, and a test checks this.

## 6. Number of regimes
Candidates are k ∈ {2, 3, 4}. The choice is made on the **training** samples, using the following:
* **Objective:** maximum silhouette.
* **Constraint:** every cluster must have ≥ 5 samples, and bootstrap stability (mean adjusted Rand index over 50 resamples) must be ≥ 0.6.
* **Tie-break:** fewer regimes.

Validation MAE for each k is reported as a diagnostic, but it is **not** used to pick k, which avoids tuning on 6 days. Names come from training statistics only:
* regimes are ranked by mean IMD rain;
* qualifiers come from the heavy-rain frequency ratio versus the overall training rate: > 1.5 = HEAVY, < 0.67 = LOW, otherwise MODERATE;
* plus the region with the largest mean rain.

## 7. Train / validation / test design (chronological, no shuffling)

| Stage | Fit on | Evaluate on | Purpose |
|---|---|---|---|
| Development | TRAIN, IMD days 1–17 Jul (51 samples) | VALIDATION, 18–23 Jul (18) | choose k; check the extreme-aware variant before touching test |
| Final | TRAIN+VAL, 1–23 Jul (69 samples, the same training days as Phase 2) | **TEST, 24–31 Jul (24), evaluated once** | final comparison |

The final refit keeps the hyperparameters from the development stage. The test period is never used to fit, select or tune anything.

## 8. Adaptive weighting method
* **Adaptive weight:** `w(model | regime, lead_day, region) = (1/(MAE+ε)) / Σ(1/(MAE+ε))`, with ε = 0.1 mm. MAE is computed on the common training pairs of that cell of (regime, lead day, region).
* **Hierarchical fallback when data is thin:**
  * If (regime, lead_day, region) has < 3 training samples or < 500 scored points, use (regime, lead_day) over all of India.
  * If that is also too thin, use the static weight for the lead day.
  * The level used and the sample counts are written to `adaptive_weights.csv`.
* **Diagnostics:** `weight_difference_by_regime.csv` reports the spread of weights across regimes and the deviation from the static weights. If the regime-to-regime spread is < 0.02, the report says the weights are nearly identical.
* **Baselines**, retrained on 1–23 Jul, which should reproduce the Phase 2 numbers exactly: equal weights, and the Phase 2 global inverse-error weights.

## 9. Extreme-rainfall strategy (experimental, data-driven)
* **Heavy regimes** are those whose training heavy-rain frequency (IMD ≥ 64.5 mm) exceeds the overall training rate.
* **Extreme-aware weights:** in heavy regimes only, weights ∝ each model's **training CSI at 64.5 mm** for that regime and lead day. If CSI is undefined, or zero for every model, the MAE weights are used. Other regimes keep the MAE-based regime weights.
* No model is forced; if both models score equally, the weights stay equal.
* The 64.5 mm threshold is unchanged from Phase 2.
* **Heavy-rain evaluation:** CSI, POD, FAR at 64.5 mm, plus MAE and bias conditioned on observed ≥ 64.5 mm.
* **Known limitation:** any weighted *average* smooths peaks, so re-weighting alone may not recover ECMWF's heavy-rain CSI. That is reported as found.

## 10. Leakage prevention
* Regime features come from forecasts only, so the target day's observation never enters regime detection.
* Scaler, PCA and KMeans are fitted on training samples only; validation and test samples are only transformed.
* Weights use training observations only (TRAIN in development, TRAIN+VAL in the final fit). `assert_no_leakage` checks the period boundaries.
* **Automated tests** check that:
  * changing TEST observations changes nothing: scaler, PCA, centres, training labels, weights;
  * changing TEST forecasts changes nothing that was fitted;
  * changing VALIDATION observations does not change the development-stage clustering or weights, and affects only the final weights, which are explicitly fitted on 1–23 Jul.

## 11. Evaluation methodology
* **Methods:** GFS, ECMWF, equal blend, Phase 2 inverse-error blend, Phase 3 regime blend, Phase 3 extreme-aware blend.
* **Samples:** the same 24 test samples and the same common land points for every method.
* **Metrics:**
  * all rainfall: MAE, RMSE, bias, correlation, and CSI/POD/FAR at 2.5, 15.6 and 64.5 mm;
  * heavy rainfall: CSI, POD and FAR at 64.5 mm, plus MAE and bias on observed-heavy points.
  * All metrics are computed per lead day and pooled.
* **Uncertainty:** a paired day-block bootstrap (2,000 resamples of the 8 test days) of the MAE difference and of the heavy-rain CSI difference, for the Phase 3 blends versus the equal blend and versus the Phase 2 blend.
* **Reporting:**
  * "Improvement" is stated only when the 95 % CI excludes 0. Everything is phrased as "on the 24–31 July 2025 test period".
  * The report lists test dates, cases, scored points and heavy-event counts.
  * **Grid-cell weights are not attempted:** the median training cell has 0 heavy events.

## Deviations recorded during implementation

* **Regime ordering** uses forecast rain coverage instead of India-mean rain: coverage is bounded and less dominated by a few extreme cells. This is a cosmetic choice; it does not affect clustering.
* **Features outside the domain.** A feature that is undefined for every sample (a region or ocean box outside a user-configured domain) is dropped and recorded in the manifest. Partial NaN still raises an error. On the default India grid no feature is dropped.
* **Single-regime fallback.** If fewer distinct feature vectors exist than any candidate k, one regime is used and the choice is recorded. Only synthetic test fixtures trigger this.
* **Outcome of k selection on real data:** k = 4. Silhouette was 0.31, 0.30 and 0.33 for k = 2, 3, 4, and k = 3 failed the stability constraint. The cluster structure is weak (silhouette < 0.5); the report states this.
