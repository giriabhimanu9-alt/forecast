# SIH 2026 · PS 26081 — Hybrid AI–NWP Multi-Model Forecast Blending for India

## 1. Purpose

The goal is to build a forecast system that blends several numerical weather prediction (NWP) models (GFS, ECMWF IFS) with AI weather models. The blend weights each model by its measured skill against IMD observations over India. Rainfall is the primary variable.

**Status:** complete:
* Phase 1: ingestion;
* Phase 2: historical verification + first blend (§11);
* Phase 3: regime-conditioned adaptive blending (§12);
* Phase 4: multi-season validation + AI benchmark (§13). Phase 1 is a real, working pipeline that:

* ingests real forecast and observation data;
* validates it;
* normalises it onto one common India grid and one data contract.

There is no blending, no dashboard and no AI model yet.

## 2. Architecture

```
backend/
├── app/
│   ├── main.py                    CLI entry point + summary printer
│   ├── config.py                  Settings from env vars / .env, target grid
│   ├── models/schema.py           Data contract, statuses, error types
│   ├── services/
│   │   ├── http.py                Retries, byte ranges, atomic cached writes
│   │   ├── pipeline.py            Orchestration, isolation of failures, report
│   │   ├── data/
│   │   │   ├── base.py            ForecastProvider / IndexedGribProvider interfaces
│   │   │   ├── gfs.py             NOAA GFS 0.25° (AWS open data / NOMADS)
│   │   │   ├── ecmwf.py           ECMWF IFS 0.25° (ECMWF Open Data)
│   │   │   ├── observations.py    IMD 0.25° gridded daily rainfall (via imdlib)
│   │   │   ├── ai_model.py        AIModelProvider interface + registry (plug-in only)
│   │   │   ├── normalization.py   coords, units, time, missing values, regridding
│   │   │   └── validation.py      file integrity + data-quality checks
│   │   └── verification/          Phase 2: windows, alignment, metrics, split, blending, experiment, plots
│   └── api/                       (later phase)
├── data/raw/                      download cache (GRIB messages, IMD binaries + provenance .json)
├── data/processed/                normalised NetCDF + JSON reports
├── scripts/run_phase1.py
├── tests/
└── requirements.txt
```

**Data flow per provider:**

1. Check the cache.
2. Download only the GRIB messages that are needed. The published `.idx`/`.index` files give their byte ranges, so each message is about 1 MB instead of a 500 MB file.
3. Validate the GRIB framing, then write atomically into the cache.
4. Decode with cfgrib, and check that the init time and step match the request.
5. Standardise coordinates, then crop to the target region.
6. Convert units and put the data on the target grid.
7. Validate the result against the data contract.
8. Save it as NetCDF and add it to the report.

Providers are independent. If one fails, it is recorded with an explicit status and the others continue.

### Common data contract

| | Forecasts (GFS, ECMWF, future AI) | Observations (IMD) |
|---|---|---|
| dims | `model, time, lead_time, latitude, longitude` | `time, latitude, longitude` |
| `time` | initialisation time (UTC) | **end** of the 24 h accumulation window (UTC) |
| extra coords | `valid_time = time + lead_time` | `period_start`, `source_date` (IMD's own day label) |
| variables | `precipitation` (mm, accumulated since init), `temperature_2m` (degC), `wind_speed_10m` (m s-1) | `precipitation` (mm per 24 h) |
| grid | ascending lat, lon in [-180,180), target grid | same |

`app.models.schema.to_variable_array(ds)` stacks the variables into one DataArray with an explicit `variable` dimension. Forecasts from different models combine with `xr.concat(..., dim="model")`.

**Target grid (default):** lat 5–38°N, lon 68–98°E at 0.25°, which is 133 × 121 points. All three sources are native 0.25° grids aligned with this one, so values are **copied exactly, not interpolated** (`regrid_method = exact-subset`). Bilinear interpolation is used only when the configured grid does not align with the source grid. Points outside a source's domain become NaN; nothing is extrapolated.

### Status values

| Status | Meaning |
|---|---|
| `AVAILABLE` | Data was fetched, passed every validation check, and was saved |
| `NOT CONFIGURED` | The provider is disabled, or the requested cycle, lead time or date range is not supported |
| `DOWNLOAD FAILED` | The cycle or date is not published, or the network/server returned an error |
| `INVALID DATA` | Data was received but failed validation (corrupt, wrong units, NaN/Inf, impossible values…). Unexpected exceptions also get this status; they are never reported as success |
| `PLUG-IN ONLY` | The AI slot: the interface exists, but no model is configured |

**Exit codes:**
* `0`: every enabled provider is AVAILABLE
* `1`: partial (at least one enabled provider failed)
* `2`: nothing usable
* `3`: configuration error

## 3. Installation

Requires Python 3.11+ (developed and tested on 3.13, Windows 11).

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

The `eccodes` wheel bundles the ecCodes C library, so no separate system install is needed on Windows or Linux x86-64.

## 4. Configuration

Copy `.env.example` to `.env` and edit it. Real environment variables override `.env`. Main keys:

| Key | Default | Notes |
|---|---|---|
| `DATA_DIR` | `./data` | Relative paths resolve against `backend/` |
| `TARGET_LAT_MIN/MAX`, `TARGET_LON_MIN/MAX` | 5 / 38, 68 / 98 | Longitudes in −180..180 |
| `TARGET_GRID_RESOLUTION` | 0.25 | Must divide the extents exactly |
| `VARIABLES` | `precipitation,temperature_2m,wind_speed_10m` | `precipitation` is mandatory |
| `FORECAST_INIT_TIME` | `latest` | `latest` picks the newest 00/12 UTC cycle that every enabled provider has published. Or give a cycle, e.g. `2026-09-27T00` |
| `FORECAST_LEAD_HOURS` | `6-72/6` | Accepts a list (`24,48,72`) or a range (`start-end/step`). Lead 0 is rejected |
| `ENABLE_GFS`, `ENABLE_ECMWF`, `ENABLE_OBSERVATIONS` | true | |
| `GFS_BASE_URL` | NOAA AWS bucket | NOMADS has the same layout |
| `ECMWF_BASE_URL` | `https://data.ecmwf.int/forecasts` | For cycles older than ~4 days, use `https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com` |
| `IMD_SOURCE` | `auto` | `auto`: archive for past years, real-time for the current year |
| `IMD_START_DATE`, `IMD_END_DATE` | empty | If empty, the last `IMD_LOOKBACK_DAYS` (7) days ending 2 days ago are used |
| `AI_MODEL_PROVIDER` | empty | The name of a registered AI plug-in |
| `HTTP_TIMEOUT`, `HTTP_RETRIES`, `LOG_LEVEL` | 120, 3, INFO | |

No secrets are needed for Phase 1.

## 5. Running Phase 1

```bash
cd backend
python scripts/run_phase1.py          # or: python -m app.main
pytest                                # 173 offline tests (Phases 1-5)
PS81_RUN_NETWORK_TESTS=1 pytest -m network   # optional live-server smoke tests
```

**Outputs:**
* `data/processed/forecasts/<model>_<YYYYMMDDHH>.nc`
* `data/processed/observations/imd_rainfall_<start>_<end>.nc`
* `data/processed/phase1_report.json`: the latest run
* `data/processed/reports/phase1_report_<run_id>.json`: history of all runs

Repeated runs reuse `data/raw/`. A cached file that fails its integrity check is deleted and downloaded again.

## 6. Data sources

| Source | Access | Credentials |
|---|---|---|
| **GFS** 0.25° `pgrb2` (NOAA/NCEP) | `https://noaa-gfs-bdp-pds.s3.amazonaws.com` (archive back to 2021) | None |
| **ECMWF IFS** 0.25° `oper` (ECMWF Open Data, CC-BY-4.0) | `https://data.ecmwf.int/forecasts` (last ~4 days) or the AWS mirror (older cycles) | None. The full MARS archive and extra parameters need an ECMWF account and licence, and are **not** used |
| **IMD** gridded daily rainfall 0.25° (Pai et al. 2014) | imdpune.gov.in: real-time daily files, plus a quality-controlled yearly archive (1901 to the last full year) | None |
| **AI models** | Not ingested in Phase 1. See `app/services/data/ai_model.py` for the plug-in contract | Depends on the source |

**Fields used:**

| Variable | GFS | ECMWF |
|---|---|---|
| Precipitation | `APCP:surface:0-N hour acc` (kg m⁻² → mm) | `tp` (m → mm) |
| 2 m temperature | `TMP:2 m` (K → °C) | `2t` (K → °C) |
| 10 m wind | `UGRD`/`VGRD:10 m` → speed | `10u`/`10v` → speed |

## 7. Validation performed

**Checks, grouped by stage:**
* **Files:** missing or empty files; GRIB framing (`GRIB` … `7777`, declared length, trailing garbage), which catches truncated downloads and saved HTML error pages; exact byte size for IMD binaries.
* **Decoding:** GRIB init time and step must match the request.
* **Coordinates and time:** coordinate range, monotonicity and duplicates; duplicate or unsorted `time`/`lead_time`.
* **Content:** missing variables; unknown or mismatched units (explicit conversion table, anything else is rejected); NaN in forecasts; Inf anywhere.
* **Physical plausibility:**
  * Rainfall below 0 is rejected. Packing noise down to −0.05 mm is clipped to 0 and counted in the report.
  * Rainfall above 1900 mm per 24 h of accumulation is rejected.
  * Accumulated rainfall must not decrease with lead time (tolerance 0.1 mm).
  * Temperature must be within −90..65 °C. This also catches Kelvin values labelled as °C.
  * Wind speed must be within 0..120 m/s.
* **IMD daily coverage:** every day must have at least 90 % of the valid land points of the best day.

## 8. Known limitations

* **IMD accumulation window: corrected in Phase 2.** Both IMD routes label day D as the 24 h **ending** 08:30 IST (03 UTC) on D, i.e. 03 UTC D−1 → 03 UTC D.
  * Phase 1 originally assumed the archive used 03 UTC D → D+1 (a reading of Pai et al. 2014). That was wrong by one day.
  * The evidence (July 2025): the real-time file for D and the archive label D are the same field (r = 0.997, mean difference 0.18 mm). The real-time file for D is published before 08:30 IST on D+1. And GFS/ECMWF forecasts correlate best with the corrected label.
  * The rule lives in one place (`PERIOD_END_OFFSET` in `observations.py`), and every Phase 2 run re-checks it (`observation_convention_check`).
  * Observation files processed for archive dates before this correction must be regenerated. Re-running the pipeline does this from the raw cache.
* **Precipitation accumulation.** Forecast precipitation is stored accumulated since initialisation. Phase 2 differences it into 24 h windows ending at 03 UTC to match IMD (see §11).
* **The default Phase 1 run does not overlap in time.** It fetches the latest forecast (future valid times) and the last week of observations. Historical overlap is handled by Phase 2.
* **ECMWF rate limiting.** `data.ecmwf.int` sometimes answers HTTP 429. The client backs off and retries, but many parallel users could still exhaust the retries.
* **ECMWF cycles.** Only the 00/12 UTC `oper` runs are supported; 06/18 UTC belong to the short `scda` stream.
* **Regridding.** Only exact-subset and bilinear methods exist; there is no conservative remapping. This is fine for the aligned 0.25° grids, but should be revisited if a coarse AI model (e.g. 1.5°) is added for precipitation.
* **IMD coverage.** IMD data covers the Indian land mask only; ocean cells are NaN by design.
* **Archive download size.** The IMD archive is fetched a whole year at a time, about 25 MB per year.

## 9. Example output (real run, 28 Sep 2026)

```
================================================================
PHASE 1 PIPELINE SUMMARY
================================================================
GFS:       AVAILABLE
ECMWF:     AVAILABLE
OBSERVED:  AVAILABLE
AI:        PLUG-IN ONLY
           -> no AI model provider configured (AI_MODEL_PROVIDER is empty); interface only

Target grid: 0.25 deg (133 lat x 121 lon)
Region:      India (lat 5.0..38.0, lon 68.0..98.0)
Variables:   precipitation, temperature_2m, wind_speed_10m (primary: precipitation)
Init time:   2026-09-27T12:00:00  [latest cycle published by all enabled forecast providers]

[GFS] 48 raw GRIB messages (0 downloaded, 48 cached)
  dims:       {'model': 1, 'time': 1, 'lead_time': 12, 'latitude': 133, 'longitude': 121}
  lead times: 6..72 h (12 steps)
  valid:      2026-09-27T18:00:00 -> 2026-09-30T12:00:00
  precip mm:  min 0.0  mean 5.1335  max 305.8125
  saved:      ...\data\processed\forecasts\gfs_2026092712.nc

[ECMWF] 48 raw GRIB messages (0 downloaded, 48 cached)
  dims:       {'model': 1, 'time': 1, 'lead_time': 12, 'latitude': 133, 'longitude': 121}
  lead times: 6..72 h (12 steps)
  valid:      2026-09-27T18:00:00 -> 2026-09-30T12:00:00
  precip mm:  min 0.0  mean 4.8661  max 224.8993
  saved:      ...\data\processed\forecasts\ecmwf_2026092712.nc

[IMD] 7 days (0 files downloaded, 7 cached)
  dims:       {'latitude': 133, 'longitude': 121, 'time': 7}
  time range: 2026-09-19T03:00:00 -> 2026-09-25T03:00:00 (24 h windows ending 03 UTC)
  precip mm:  min 0.0  mean 6.2681  max 367.6031
  saved:      ...\data\processed\observations\imd_rainfall_20260919_20260925.nc

Files processed: 103 raw files
================================================================
```

A first (uncached) run downloads about 70 MB and takes roughly 80 s.

## 10. Next steps (after Phase 4)

Phase 4 (§13) added seasons, synoptic features and a real AI model. With three IMD seasons, regime conditioning still adds nothing measurable over equal weighting. The remaining bottlenecks are data and methods, not plumbing:

1. **AI against IMD.** Every accessible AI forecast is 6-hourly, so none can be verified exactly against IMD's 03 UTC day. The options, which need a project decision:
   * find a 3-hourly AI product;
   * or accept, and document, a disaggregation of 6 h accumulations to 3 h.
   Until then, AI skill is known only against ERA5 (Experiment B).
2. **Longer IMD record for NWP.** ECMWF open data starts in 2023. A longer training record needs licensed ECMWF archive access (MARS/TIGGE), or a GFS-only analysis back to 2021.
3. **Extremes.** Weighted averaging cannot recover peak intensity; ECMWF alone keeps a higher heavy-rain CSI than any blend (§11–§13). Test calibration for heavy rain rather than re-weighting, such as quantile mapping or probability-of-exceedance blending.
4. **Regimes.** Clustering remains weak (silhouette ≈ 0.33). Try physically defined monsoon indices (active/break, MISO phase) instead of unsupervised clusters.

## 11. Phase 2 — historical verification + first blend

### What it does

The experiment runs these steps:

1. **Plan cycles.** For every target IMD day, work out the forecast cycles whose lead-day windows coincide with that day's observation window.
2. **Download and normalise.** Fetch those cycles through the Phase 1 providers, which cache everything, retry failures and never download a file twice.
3. **Convert to 24 h totals** (`forecast_to_24h_accumulation`):
   * `rain_24h = acc(L0+24 h) − acc(L0)`, where `acc(0) = 0`.
   * The windows end at 03 UTC to match IMD. From a 00Z cycle, day 1 = 3–27 h, day 2 = 27–51 h, day 3 = 51–75 h. From a 12Z cycle, day 1 = 15–39 h.
   * A 48 or 72 h cumulative total can never be compared with a daily observation: every window must be exactly 24 h, or the function raises an error.
4. **Align** (`align_forecast_to_observation`):
   * A forecast is paired with an observation only when both ends of the period are identical and the forecast was initialised before the period starts.
   * There is no nearest-date matching, so a one-day offset raises `AlignmentError`.
   * Each sample records `init_time`, `lead_day`, `observation_period_start`, `observation_period_end` and `valid_date` (the IMD day label).
5. **Split chronologically by observation period.** The first 75 % of days are used for training and the rest for testing. The code checks that all training periods end before the first test period starts, and that no date is in both.
6. **Learn weights from the training data only** (`learn_weights`), with three methods:
   * equal weights;
   * global inverse-error weights, `w = 1/(MAE+ε)` normalised to sum to 1;
   * regional inverse-error weights, using 6 lat/lon blocks (North, West, Central, East, South, Northeast).

   Errors are computed on common pairs, i.e. only where every model and the observation are present. A missing model is removed and the remaining weights are renormalised.
7. **Blend the test period.** This step uses test forecasts and the learned weights only; it cannot see test observations, and a unit test proves it.
8. **Score** MAE, RMSE, bias, and precision / recall / F1 / CSI at 2.5, 15.6 and 64.5 mm/24 h.
9. **Test whether the blend is really better.** A paired bootstrap over test days (2000 resamples) compares each blend with the best individual model and with the equal blend. An improvement is claimed only when the 95 % confidence interval excludes 0 **and** the run has at least 20 days, 7 of them for testing. Smaller runs are labelled `ALIGNMENT/PIPELINE TEST`.

### Data routes (verified 2026-09-28)

| Source | Route | Status |
|---|---|---|
| GFS 0.25° | NOAA AWS bucket (archive back to 2021) | ✅ used |
| ECMWF IFS 0.25° | ECMWF Open Data **AWS archive mirror** `ecmwf-forecasts.s3.eu-central-1.amazonaws.com` (3-hourly `tp`, CC-BY-4.0) | ✅ used. The mirror sometimes answers `503 SlowDown`; the client paces requests and retries with backoff |
| ECMWF TIGGE | MARS Web API | ❌ not used, for two reasons (see `app/services/data/tigge.py`): **(1)** it needs an ECMWF account, TIGGE licence and API key (`403 Missing access token` without one); **(2)** TIGGE precipitation is 6-hourly, so 00/12Z cycles cannot bracket IMD's 03 UTC windows exactly |
| IMD 0.25° | yearly archive (quality-controlled) | ✅ used |

### Run

```bash
cd backend
pytest
python scripts/run_phase2.py --preset pilot      # IMD days 1-10 Jul 2025 (pipeline test)
python scripts/run_phase2.py --preset july2025   # IMD days 1-31 Jul 2025 (experiment)
# options: --start/--end YYYY-MM-DD, --cycles 0,12, --lead-days 1,2,3, --train-fraction 0.75,
#          --thresholds 2.5,15.6,64.5, --epsilon 0.1, --models GFS,ECMWF, --name <folder>
```

**Outputs** (`results/<name>/`):
* tables: `model_scores.csv`, `weights.csv`, `test_scores.csv`, `blend_comparison.csv`;
* figures: `skill_comparison.png`, `rainfall_forecast_vs_imd.png`, `weight_map.png`, `error_map.png`;
* records: `experiment_manifest.json`, `report.md`, `run.log`;
* `aligned_samples.nc`, which is git-ignored and can be regenerated.

A first run downloads about 250 MB of GRIB messages in roughly 4 minutes. Cached re-runs take about 1 minute.

### Results: July 2025 (real data)

**Setup:**
* 33 00Z cycles per model (28 Jun – 30 Jul), lead days 1–3, 133 × 121 grid.
* 93 aligned samples, with no missing forecasts or observations and no failed downloads.
* Training: IMD days 1–23 Jul. Test: 24–31 Jul (8 days, 24 samples, 119,136 scored land points).

**Test period, all lead days, in mm/24 h:**

| | MAE | RMSE | Bias | CSI ≥ 64.5 mm |
|---|---|---|---|---|
| GFS | 9.407 | 18.62 | −0.65 | 0.052 |
| ECMWF | 9.485 | 18.52 | +0.94 | 0.095 |
| Equal blend | 8.762 | 16.68 | +0.15 | 0.070 |
| Inverse-error blend | 8.761 | 16.68 | +0.15 | 0.070 |
| Regional blend | 8.761 | 16.68 | +0.15 | 0.070 |

**Findings:**
* **The blends beat the best individual model (GFS) on MAE by 6.9 %.** The MAE difference is −0.65 mm, 95 % CI [−0.84, −0.45] over the 8 test days. This holds at every lead day (+7.4 %, +7.1 % and +5.0 %).
* **Skill-based weighting adds nothing measurable over equal weighting.**
  * Inverse-error vs equal: −0.0004 mm, CI [−0.0018, +0.0011]. Regional vs equal: −0.0006 mm.
  * The two models have almost the same training MAE, so the learned weights are about 0.47–0.53.
  * The gain comes from averaging two models with **opposite biases**, not from adaptive weighting.
* **Heavy rain is worse with the blend than with ECMWF:** CSI at ≥64.5 mm is 0.070 vs 0.095, because averaging smooths extremes. At the 2.5 and 15.6 mm thresholds the blend is better.
* **Caveats.** This is one month of one monsoon season with 8 test days, and the bootstrap resamples whole days. It is evidence for this period, not a general result.

The pilot run (IMD days 1–10 Jul) is labelled `ALIGNMENT/PIPELINE TEST` and makes no claim.

## 12. Phase 3 — regime-conditioned adaptive blending

**Design:** see `PHASE3_IMPLEMENTATION_PLAN.md` (written before implementation).
**Code:** `app/services/verification/{regimes,adaptive,phase3}.py`.

### Method

**Regime features (11, forecast-time only).** They are computed from the mean of the GFS and ECMWF forecasts for each window:
* log mean rain in each of the 6 regions;
* rain coverage (fraction of land ≥ 2.5 mm) and heavy-rain area (fraction ≥ 64.5 mm);
* model disagreement;
* mean 10 m wind over the Arabian Sea and over the Bay of Bengal.

The IMD observation of the target day is never a feature, because it does not exist when the forecast is issued. No pressure, geopotential or humidity fields are ingested, so these are **rainfall and surface-wind regimes, not synoptic regimes**.

**Detector.** StandardScaler → PCA (≥ 80 % of variance) → KMeans, fitted only on the fit period and saved as `regime_model.json`. A numpy re-implementation reproduces the labels from the JSON. k ∈ {2, 3, 4} is chosen by training silhouette, with at least 5 samples per cluster and bootstrap ARI ≥ 0.6.

**Periods** (chronological):
* train 1–17 Jul, validation 18–23 Jul: used to choose k and check the methods before touching test;
* final fit on 1–23 Jul, the same training days as Phase 2;
* the Phase 2 test period, 24–31 Jul, left **untouched** until the single final evaluation.

**Weights.**
* **Regime blend:** `w(model | regime, lead_day, region)`, inverse-MAE on the fit period. If a cell of (regime, lead day, region) has too little data, it falls back to (regime, lead day) and then to static weights; the level used is recorded.
* **Extreme-aware blend:** identical, except that regimes whose heavy-rain frequency is above average use weights proportional to training CSI at 64.5 mm.

### Run

```bash
cd backend
pytest -q
python scripts/run_phase3.py        # results/phase3/ (~1 min from cache; ~15 min first run incl. 10 m wind download)
```

Outputs:
* `regimes/`: `regime_assignments.csv`, `regime_statistics.csv`, `regime_model.json`, `k_selection.csv`;
* `weights/`: `adaptive_weights.csv`, `weight_difference_by_regime.csv`, `static_weights_phase2_method.csv`;
* `verification/`: `model_scores_by_regime.csv`, `final_comparison.csv`, `extreme_rain_scores.csv`, `uncertainty.csv`, `validation_comparison.csv`;
* `figures/`: 5 PNGs;
* `experiment_manifest.json`, `report.md`, `run.log`.

### Results: test period 24–31 July 2025

The test period has 8 days, 24 forecast cases, 119,136 scored land points and 2,952 heavy-rain events. It is a proof of concept.

**Regimes (k = 4).**
* REGIME_0 LOW (13 fit samples, heavy ratio 0.65). REGIME_1 MODERATE, wettest in the East (12, 1.20). REGIME_2 MODERATE, Central (27, 1.18). REGIME_3 MODERATE, South (17, 0.84).
* Cluster structure is **weak**, with silhouette ≤ 0.33, and **no regime is a heavy-rain regime** (none reaches a 1.5× heavy-rain frequency).
* The regimes mostly group consecutive days. **23 of the 24 test cases are REGIME_2.**

| Method | MAE | RMSE | Bias | Corr | Heavy CSI | POD | FAR |
|---|---|---|---|---|---|---|---|
| GFS | 9.407 | 18.62 | −0.65 | 0.427 | 0.052 | 0.084 | 0.878 |
| ECMWF | 9.485 | 18.52 | +0.94 | 0.467 | 0.095 | 0.136 | 0.760 |
| Equal blend | 8.762 | 16.68 | +0.15 | 0.512 | 0.070 | 0.095 | 0.788 |
| Phase 2 inverse-error | 8.761 | 16.68 | +0.15 | 0.512 | 0.070 | 0.095 | 0.787 |
| Phase 3 regime blend | 8.767 | 16.68 | +0.16 | 0.512 | 0.070 | 0.095 | 0.788 |
| Phase 3 extreme-aware | 8.890 | 17.16 | +0.24 | 0.493 | 0.068 | 0.093 | 0.796 |

**Findings.** Confidence intervals come from a paired bootstrap over the 8 test days.
* **Regime conditioning barely changes the weights.** The spread across regimes is at most 0.078 (median 0.038), and weights stay within 0.43–0.56.
* **The regime blend is statistically worse than the equal blend, but only trivially:** MAE +0.006 mm, 95 % CI [+0.003, +0.008], i.e. about 0.07 %. Heavy-rain CSI is indistinguishable.
* **The extreme-aware blend is worse on MAE** (+0.13 mm, CI [+0.06, +0.20]) and does not improve heavy-rain CSI (−0.002, CI [−0.012, +0.008]).
  * It stays below ECMWF on heavy rain (CSI −0.027, CI [−0.056, −0.002]).
  * Its CSI-based weights are unstable, because training CSI rests on few events. For example, REGIME_2's GFS weight is 0.62 on day 2 but 0.21 on day 3.
* **Validation (18–23 Jul) showed the same picture:** regime 7.702 vs equal 7.706 MAE, and the extreme-aware blend was worse.
* **Grid-cell-level adaptive weights were not statistically supportable with the current sample size.** The median land cell has 0 heavy-rain events in the fit period.

**Conclusion.** On the 24–31 July 2025 test period, regime-conditioned weighting did **not** improve on equal weighting, and the extreme-aware variant made MAE worse without helping heavy rain. The pipeline is correct and leakage-tested, but one month of data cannot demonstrate regime-dependent skill. The next steps are listed in §10.

## 13. Phase 4 — multi-season validation + AI forecast benchmark

**Documents:**
* `PHASE4_DATA_AUDIT.md`: every source tested before download;
* `PHASE4_RESULTS.md`: generated automatically, with all numbers and confidence intervals.

**Code:** `app/services/verification/{phase4,phase4_report,production}.py` and `app/services/data/wb2.py`.

### Two experiments that are never combined

| | **A — India operational** | **B — AI/NWP benchmark** |
|---|---|---|
| Truth | IMD 0.25° gauge-based rainfall | ERA5 reanalysis (WeatherBench 2) |
| Models | GFS, ECMWF IFS (NWP) | **GraphCast (AI)**, IFS HRES (NWP) |
| Seasons (Jun–Sep) | 2023 train · 2024 validation · **2025 test** | 2018 train · 2020 validation · **2022 test** |
| Window / grid | 24 h ending 03 UTC · 0.25° | 24 h ending 00 UTC · 1.5° |

**Why AI is only in B.**
* Every accessible AI forecast is **6-hourly**: AIFS in ECMWF open data, GraphCast/Pangu/FourCastNet in the NOAA MLWP archive, NOAA GraphCastGFS, and WeatherBench 2.
* With 6-hourly output, no forecast cycle can produce a 24 h total ending at 03 UTC, so exact alignment with IMD is impossible.
* Pangu and NeuralGCM have no precipitation output at all.
* GraphCast is integrated as a real registered `AIModelProvider` (`graphcast-wb2`), using the same interface, cache and normalisation as GFS and ECMWF.

### What was added
* **More data:**
  * multi-season datasets and manifests (100 % cycle coverage in every season);
  * the ECMWF product path by date (0.4° beta product in 2023, 0.25° afterwards);
  * the GFS 1° product for MSLP, 850 hPa wind and precipitable water, used as atmospheric regime features.
* **Evaluation:**
  * a rainfall-only vs atmospheric + rainfall regime comparison;
  * a missing-model fallback with explanation metadata (`production.blend_available`, plus `*_blend_explanations.jsonl` with one record per test forecast);
  * a **practical-significance rule**: an effect is "meaningful" only if the day-bootstrap CI excludes 0 *and* |ΔMAE| ≥ 1 % (or |ΔCSI| ≥ 0.005).
* **Data-quality outputs:** coverage, missing data, per-season alignment checks (the IMD convention passed in every season), grid information and variable metadata.

### Headline results (test seasons; details and CIs in `PHASE4_RESULTS.md`)

**Experiment A (IMD, June–September 2025, 122 days, 27,669 heavy-rain events).**
* **The equal GFS + ECMWF blend beats both models on MAE by about 6.3 %** (7.17 vs 7.65 / 7.66 mm), and in all three seasons.
* On heavy rain, the blend is better than GFS but **worse than ECMWF** (CSI 0.078 vs 0.088).
* Skill-based and regime-conditioned weights add nothing meaningful over equal weights. Regimes are weak (silhouette ≈ 0.33).
* Atmospheric features did not change this.

**Experiment B (ERA5, June–September 2022).**
* **GraphCast beats HRES by a wide margin** (MAE 2.52 vs 3.71; heavy-rain CSI 0.41 vs 0.25).
* Skill-based weights beat equal weights here (by about 5 %), because the two models differ in skill.
* Regime conditioning adds no meaningful MAE gain over static weights (−0.15 %). It does add a small heavy-rain CSI gain (+0.0095, CI excludes 0), but this is one of many comparisons, so it is tentative.
* **But no blend beats GraphCast alone.**
* ERA5 favours GraphCast by construction, so B says nothing about skill against IMD.

### Run

```bash
cd backend
pytest -q
python scripts/audit_phase4_sources.py                  # re-run the access probes
python scripts/fetch_phase4_data.py --source GFS        # optional parallel pre-download
python scripts/fetch_phase4_data.py --source ECMWF      #   (also GFS_ATMOS, IMD)
python scripts/run_phase4.py                            # ~35 min from cache; first run downloads ~3 GB
```

## 14. Phase 5 — operational dashboard (MVP)

```bash
cd backend
uvicorn app.api.server:app --port 8000     # then open http://localhost:8000
```

**What it does.** The dashboard's **Run latest 00 UTC cycle** button (`POST /api/forecast/run`) runs one operational cycle:

1. Pick the newest 00 UTC cycle published by **both** GFS and ECMWF. A single-model cycle is used only as a fallback.
2. Build IMD-aligned 24 h windows for lead days 1–3.
3. Assign the regime with the Phase 4 regime model.
4. Blend with the Phase 4 regional weights, fitted on 2023–24 and validated on 2025.
5. Flag heavy rain ≥ 64.5 mm per model and for the blend.

Results are saved in `data/processed/operational/`. The code is `app/services/operational.py` and `app/api/server.py`; the page is `app/api/static/index.html`, with no build step.

**What the page shows.**
* A map of India for the blend, GFS or ECMWF, by lead day.
* Heavy-rain cells, and regional model weights with their training errors.
* A model availability and fallback indicator.
* The validated IMD skill (Experiment A).
* A separate, dashed panel for the ERA5 AI benchmark (Experiment B). GraphCast is **not** part of the operational blend and is never compared with IMD.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | service status, number of runs, Phase 4 artefacts present |
| `POST /api/forecast/run?init=YYYY-MM-DDT00` | runs one operational cycle (default: latest) |
| `GET /api/forecast[?run=]` | run summary: model status, regime, weights, fallback, alerts |
| `GET /api/forecast/runs` | available runs |
| `GET /api/forecast/field?product=blend\|GFS\|ECMWF&lead_day=1..3[&stride=]` | grid for maps (India land only) |
| `GET /api/weights/regional` | Phase 4 regime × lead day × region weights |
| `GET /api/validation` | Experiment A test scores vs IMD |
| `GET /api/benchmark/ai` | Experiment B scores vs ERA5, with the not-comparable warning |

The summary repeats one validated caveat: the blend has the lowest MAE, but ECMWF alone detects heavy rain best. So ECMWF exceedances are shown next to the blend's.
