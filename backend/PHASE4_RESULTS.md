# Phase 4 results — multi-season validation + AI forecast benchmark

Generated automatically by `scripts/run_phase4.py` on 2026-09-28 05:09 UTC. Every number below is read from the result tables in `results/phase4/`.

> **Experiments A and B are separate and must not be compared with each other.** A verifies against IMD gauge-based rainfall (03→03 UTC); B verifies against ERA5 reanalysis (00→00 UTC, 1.5°). No AI model could be aligned exactly with IMD (6-hourly output only), so **no AI result here is evidence of skill against IMD**.

## 1–6. Datasets, dates, models, variables, resolutions

| | Experiment A — India operational | Experiment B — AI/NWP benchmark |
|---|---|---|
| Truth | IMD 0.25° gridded rainfall (archive), India land | ERA5 `total_precipitation_24hr` (WeatherBench 2), India land cells (≥50 % IMD land) |
| Seasons (JJAS) | 2023, 2024, 2025 | 2018, 2020, 2022 |
| Models | GFS 0.25° (NWP), ECMWF IFS (NWP; 0.4° in 2023, 0.25° 2024–25) | **GraphCast (AI)**, IFS HRES (NWP) |
| Forecast cycles | 00 UTC daily | 00 UTC daily |
| Lead days | 1–3 (3–27, 27–51, 51–75 h) | 1–3 (0–24, 24–48, 48–72 h) |
| Verification window | 24 h ending 03 UTC (08:30 IST), IMD label D | 24 h ending 00 UTC |
| Grid | 0.25°, 5–38°N × 68–98°E (133×121); ECMWF 2023 bilinear from 0.4° | 1.5°, 6–37.5°N × 67.5–97.5°E (22×21), native |
| Variable | 24 h precipitation (mm) | 24 h precipitation (mm) |
| Regime features | forecast rainfall (both models) + GFS 1.0° MSLP, u850, PWAT | forecast rainfall |

## 7. Split — Experiment A

* train [2023] (366 forecast cases), validation [2024] (366), test **[2025]** (366 cases, 122 days, 1,816,824 scored grid points, 27,669 heavy-rain events ≥64.5 mm).
* Development: fit on train, choose k and regime feature set on validation. Final: refit on train+validation, evaluate **once** on test. No shuffling; `assert_no_leakage` checked both boundaries.

## 8–11. Test scores — Experiment A (2025, all lead days)

| Method | MAE | RMSE | Bias | Corr | CSI ≥64.5 | POD ≥64.5 | FAR ≥64.5 | CSI ≥15.6 |
|---|---|---|---|---|---|---|---|---|
| GFS | 7.650 | 16.04 | -0.52 | 0.396 | 0.066 | 0.095 | 0.820 | 0.251 |
| ECMWF | 7.658 | 15.15 | +0.84 | 0.470 | 0.088 | 0.117 | 0.737 | 0.296 |
| Equal blend | 7.168 | 14.19 | +0.16 | 0.501 | 0.078 | 0.095 | 0.694 | 0.301 |
| Phase 2 inverse-error | 7.168 | 14.18 | +0.16 | 0.501 | 0.078 | 0.095 | 0.693 | 0.301 |
| Phase 3 regime blend | 7.167 | 14.18 | +0.16 | 0.502 | 0.078 | 0.095 | 0.693 | 0.301 |
| Phase 3 extreme-aware | 7.178 | 14.16 | +0.24 | 0.504 | 0.080 | 0.097 | 0.685 | 0.304 |
| regime[atmos+rain] | 7.167 | 14.18 | +0.16 | 0.501 | 0.078 | 0.095 | 0.692 | 0.301 |

By lead day:

| Method | Day 1 MAE | Day 2 MAE | Day 3 MAE | Day 1 CSI≥64.5 | Day 2 CSI≥64.5 | Day 3 CSI≥64.5 |
|---|---|---|---|---|---|---|
| GFS | 7.134 | 7.662 | 8.155 | 0.092 | 0.063 | 0.045 |
| ECMWF | 7.157 | 7.716 | 8.100 | 0.118 | 0.080 | 0.069 |
| Equal blend | 6.693 | 7.199 | 7.612 | 0.108 | 0.074 | 0.053 |
| Phase 2 inverse-error | 6.694 | 7.199 | 7.612 | 0.108 | 0.074 | 0.053 |
| Phase 3 regime blend | 6.694 | 7.199 | 7.607 | 0.108 | 0.074 | 0.053 |
| Phase 3 extreme-aware | 6.696 | 7.208 | 7.631 | 0.110 | 0.076 | 0.055 |
| regime[atmos+rain] | 6.695 | 7.199 | 7.607 | 0.108 | 0.074 | 0.053 |

Best individual model by MAE: **GFS**; by heavy-rain CSI: **ECMWF**.

Uncertainty (paired bootstrap over test **days**, all lead days of a day resampled together, 2000 resamples). An effect is called *meaningful* only if the CI excludes 0 **and** |ΔMAE| ≥ 1 % (or |ΔCSI| ≥ 0.005); otherwise it is *negligible* even when statistically detectable (millions of correlated grid points make tiny effects 'significant'):

* Equal blend vs GFS: MAE -0.482 mm (-6.30 %) [-0.540, -0.426] → **meaningful improvement**; heavy CSI +0.0121 [+0.0033, +0.0212] → **meaningful improvement** (122 test days)
* Equal blend vs ECMWF: MAE -0.490 mm (-6.39 %) [-0.540, -0.439] → **meaningful improvement**; heavy CSI -0.0100 [-0.0168, -0.0031] → **meaningful degradation** (122 test days)
* Equal blend vs Phase 2 inverse-error: MAE -0.000 mm (-0.00 %) [-0.001, +0.000] → **not distinguishable**; heavy CSI -0.0000 [-0.0003, +0.0002] → **not distinguishable** (122 test days)
* Phase 2 inverse-error vs GFS: MAE -0.482 mm (-6.30 %) [-0.540, -0.425] → **meaningful improvement**; heavy CSI +0.0121 [+0.0032, +0.0213] → **meaningful improvement** (122 test days)
* Phase 2 inverse-error vs ECMWF: MAE -0.489 mm (-6.39 %) [-0.540, -0.439] → **meaningful improvement**; heavy CSI -0.0100 [-0.0167, -0.0031] → **meaningful degradation** (122 test days)
* Phase 2 inverse-error vs Equal blend: MAE +0.000 mm (+0.00 %) [-0.000, +0.001] → **not distinguishable**; heavy CSI +0.0000 [-0.0002, +0.0003] → **not distinguishable** (122 test days)
* Phase 3 regime blend vs GFS: MAE -0.484 mm (-6.32 %) [-0.542, -0.427] → **meaningful improvement**; heavy CSI +0.0120 [+0.0031, +0.0213] → **meaningful improvement** (122 test days)
* Phase 3 regime blend vs ECMWF: MAE -0.491 mm (-6.42 %) [-0.542, -0.440] → **meaningful improvement**; heavy CSI -0.0101 [-0.0167, -0.0032] → **meaningful degradation** (122 test days)
* Phase 3 regime blend vs Equal blend: MAE -0.002 mm (-0.02 %) [-0.003, -0.000] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI -0.0001 [-0.0006, +0.0004] → **not distinguishable** (122 test days)
* Phase 3 regime blend vs Phase 2 inverse-error: MAE -0.002 mm (-0.03 %) [-0.003, -0.000] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI -0.0001 [-0.0005, +0.0003] → **not distinguishable** (122 test days)
* Phase 3 extreme-aware vs GFS: MAE -0.472 mm (-6.17 %) [-0.534, -0.412] → **meaningful improvement**; heavy CSI +0.0136 [+0.0038, +0.0235] → **meaningful improvement** (122 test days)
* Phase 3 extreme-aware vs ECMWF: MAE -0.480 mm (-6.26 %) [-0.525, -0.434] → **meaningful improvement**; heavy CSI -0.0085 [-0.0144, -0.0022] → **meaningful degradation** (122 test days)
* Phase 3 extreme-aware vs Equal blend: MAE +0.010 mm (+0.14 %) [+0.003, +0.016] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI +0.0015 [+0.0001, +0.0030] → **negligible (CI excludes 0 but |effect| < 0.005)** (122 test days)
* Phase 3 extreme-aware vs Phase 2 inverse-error: MAE +0.010 mm (+0.14 %) [+0.003, +0.016] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI +0.0015 [+0.0001, +0.0029] → **negligible (CI excludes 0 but |effect| < 0.005)** (122 test days)
* regime[atmos+rain] vs GFS: MAE -0.483 mm (-6.32 %) [-0.541, -0.427] → **meaningful improvement**; heavy CSI +0.0120 [+0.0032, +0.0212] → **meaningful improvement** (122 test days)
* regime[atmos+rain] vs ECMWF: MAE -0.491 mm (-6.41 %) [-0.542, -0.440] → **meaningful improvement**; heavy CSI -0.0101 [-0.0167, -0.0031] → **meaningful degradation** (122 test days)
* regime[atmos+rain] vs Equal blend: MAE -0.001 mm (-0.02 %) [-0.003, +0.000] → **not distinguishable**; heavy CSI -0.0000 [-0.0005, +0.0005] → **not distinguishable** (122 test days)
* regime[atmos+rain] vs Phase 2 inverse-error: MAE -0.001 mm (-0.02 %) [-0.003, +0.000] → **not distinguishable**; heavy CSI -0.0001 [-0.0005, +0.0004] → **not distinguishable** (122 test days)

### By season — Experiment A (individual models + equal blend, no training needed)

| Season | Method | MAE | RMSE | Bias | CSI ≥64.5 | POD | FAR | heavy events |
|---|---|---|---|---|---|---|---|---|
| 2023 | GFS | 6.919 | 15.68 | +0.00 | 0.111 | 0.162 | 0.739 | 27393 |
| 2023 | ECMWF | 6.708 | 14.44 | +0.91 | 0.129 | 0.176 | 0.675 | 27393 |
| 2023 | Equal blend | 6.375 | 13.61 | +0.45 | 0.132 | 0.165 | 0.604 | 27393 |
| 2024 | GFS | 8.171 | 17.43 | +0.10 | 0.106 | 0.156 | 0.750 | 33600 |
| 2024 | ECMWF | 8.257 | 16.96 | +1.34 | 0.139 | 0.198 | 0.682 | 33600 |
| 2024 | Equal blend | 7.693 | 15.56 | +0.72 | 0.131 | 0.169 | 0.632 | 33600 |
| 2025 | GFS | 7.650 | 16.04 | -0.52 | 0.066 | 0.095 | 0.820 | 27669 |
| 2025 | ECMWF | 7.658 | 15.15 | +0.84 | 0.088 | 0.117 | 0.737 | 27669 |
| 2025 | Equal blend | 7.168 | 14.19 | +0.16 | 0.078 | 0.095 | 0.694 | 27669 |

## 12. Regimes — Experiment A

* feature set `rain`: k = 2, training silhouette 0.331, bootstrap ARI 0.810, validation MAE of regime blend 7.686
* feature set `atmos+rain`: k = 4, training silhouette 0.341, bootstrap ARI 0.827, validation MAE of regime blend 7.686
* chosen (lowest validation MAE, pre-registered rule): **`rain`**
  * REGIME_0: LOW RAINFALL (wettest: Northeast): 250 fit samples, mean IMD/ref rain 3.74 mm, heavy-rain frequency ratio 0.45, test samples 86
  * REGIME_1: MODERATE RAINFALL (wettest: Northeast): 482 fit samples, mean IMD/ref rain 8.98 mm, heavy-rain frequency ratio 1.29, test samples 280
* cluster structure: best silhouette 0.34 → **weak** (< 0.5)
* regime weights: spread across regimes max 0.056, median 0.020

Weights (lead-day static vs regime range across regimes and regions):

| Lead day | static GFS | static ECMWF | regime GFS (min–max) | regime ECMWF (min–max) |
|---|---|---|---|---|
| 1 | 0.491 | 0.509 | 0.46–0.52 | 0.48–0.54 |
| 2 | 0.500 | 0.500 | 0.46–0.54 | 0.46–0.54 |
| 3 | 0.502 | 0.498 | 0.45–0.54 | 0.46–0.55 |

## 7. Split — Experiment B

* train [2018] (366 forecast cases), validation [2020] (366), test **[2022]** (366 cases, 122 days, 50,874 scored grid points, 321 heavy-rain events ≥64.5 mm).
* Development: fit on train, choose k and regime feature set on validation. Final: refit on train+validation, evaluate **once** on test. No shuffling; `assert_no_leakage` checked both boundaries.

## 8–11. Test scores — Experiment B (2022, all lead days)

| Method | MAE | RMSE | Bias | Corr | CSI ≥64.5 | POD ≥64.5 | FAR ≥64.5 | CSI ≥15.6 |
|---|---|---|---|---|---|---|---|---|
| GraphCast | 2.519 | 4.82 | +0.37 | 0.903 | 0.408 | 0.477 | 0.261 | 0.624 |
| HRES | 3.706 | 7.24 | +0.71 | 0.798 | 0.247 | 0.389 | 0.597 | 0.502 |
| Equal blend | 2.790 | 5.33 | +0.54 | 0.882 | 0.325 | 0.411 | 0.392 | 0.594 |
| Phase 2 inverse-error | 2.646 | 5.05 | +0.49 | 0.894 | 0.355 | 0.433 | 0.335 | 0.611 |
| Phase 3 regime blend | 2.642 | 5.04 | +0.48 | 0.894 | 0.365 | 0.442 | 0.324 | 0.612 |
| Phase 3 extreme-aware | 2.665 | 5.08 | +0.50 | 0.892 | 0.356 | 0.436 | 0.340 | 0.609 |

By lead day:

| Method | Day 1 MAE | Day 2 MAE | Day 3 MAE | Day 1 CSI≥64.5 | Day 2 CSI≥64.5 | Day 3 CSI≥64.5 |
|---|---|---|---|---|---|---|
| GraphCast | 1.605 | 2.772 | 3.181 | 0.626 | 0.336 | 0.268 |
| HRES | 3.237 | 3.711 | 4.171 | 0.317 | 0.255 | 0.178 |
| Equal blend | 2.130 | 2.916 | 3.325 | 0.459 | 0.304 | 0.217 |
| Phase 2 inverse-error | 1.868 | 2.832 | 3.237 | 0.508 | 0.336 | 0.227 |
| Phase 3 regime blend | 1.865 | 2.828 | 3.231 | 0.531 | 0.338 | 0.229 |
| Phase 3 extreme-aware | 1.933 | 2.805 | 3.257 | 0.508 | 0.338 | 0.230 |

Best individual model by MAE: **GraphCast**; by heavy-rain CSI: **GraphCast**.

Uncertainty (paired bootstrap over test **days**, all lead days of a day resampled together, 2000 resamples). An effect is called *meaningful* only if the CI excludes 0 **and** |ΔMAE| ≥ 1 % (or |ΔCSI| ≥ 0.005); otherwise it is *negligible* even when statistically detectable (millions of correlated grid points make tiny effects 'significant'):

* Equal blend vs GraphCast: MAE +0.271 mm (+10.75 %) [+0.214, +0.325] → **meaningful degradation**; heavy CSI -0.0829 [-0.1289, -0.0329] → **meaningful degradation** (122 test days)
* Equal blend vs HRES: MAE -0.916 mm (-24.71 %) [-1.000, -0.831] → **meaningful improvement**; heavy CSI +0.0781 [+0.0392, +0.1155] → **meaningful improvement** (122 test days)
* Equal blend vs Phase 2 inverse-error: MAE +0.145 mm (+5.47 %) [+0.129, +0.161] → **meaningful degradation**; heavy CSI -0.0304 [-0.0489, -0.0133] → **meaningful degradation** (122 test days)
* Phase 2 inverse-error vs GraphCast: MAE +0.126 mm (+5.01 %) [+0.083, +0.166] → **meaningful degradation**; heavy CSI -0.0525 [-0.0968, -0.0058] → **meaningful degradation** (122 test days)
* Phase 2 inverse-error vs HRES: MAE -1.060 mm (-28.61 %) [-1.160, -0.960] → **meaningful improvement**; heavy CSI +0.1085 [+0.0686, +0.1479] → **meaningful improvement** (122 test days)
* Phase 2 inverse-error vs Equal blend: MAE -0.145 mm (-5.18 %) [-0.161, -0.129] → **meaningful improvement**; heavy CSI +0.0304 [+0.0133, +0.0489] → **meaningful improvement** (122 test days)
* Phase 3 regime blend vs GraphCast: MAE +0.122 mm (+4.85 %) [+0.079, +0.162] → **meaningful degradation**; heavy CSI -0.0430 [-0.0877, +0.0023] → **not distinguishable** (122 test days)
* Phase 3 regime blend vs HRES: MAE -1.064 mm (-28.72 %) [-1.165, -0.964] → **meaningful improvement**; heavy CSI +0.1180 [+0.0775, +0.1590] → **meaningful improvement** (122 test days)
* Phase 3 regime blend vs Equal blend: MAE -0.149 mm (-5.33 %) [-0.166, -0.133] → **meaningful improvement**; heavy CSI +0.0399 [+0.0217, +0.0595] → **meaningful improvement** (122 test days)
* Phase 3 regime blend vs Phase 2 inverse-error: MAE -0.004 mm (-0.15 %) [-0.006, -0.002] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI +0.0095 [+0.0018, +0.0198] → **meaningful improvement** (122 test days)
* Phase 3 extreme-aware vs GraphCast: MAE +0.146 mm (+5.79 %) [+0.102, +0.188] → **meaningful degradation**; heavy CSI -0.0518 [-0.0980, -0.0043] → **meaningful degradation** (122 test days)
* Phase 3 extreme-aware vs HRES: MAE -1.041 mm (-28.09 %) [-1.139, -0.943] → **meaningful improvement**; heavy CSI +0.1092 [+0.0689, +0.1491] → **meaningful improvement** (122 test days)
* Phase 3 extreme-aware vs Equal blend: MAE -0.125 mm (-4.48 %) [-0.139, -0.112] → **meaningful improvement**; heavy CSI +0.0311 [+0.0138, +0.0495] → **meaningful improvement** (122 test days)
* Phase 3 extreme-aware vs Phase 2 inverse-error: MAE +0.019 mm (+0.74 %) [+0.016, +0.023] → **negligible (CI excludes 0 but |effect| < 1%)**; heavy CSI +0.0007 [-0.0093, +0.0119] → **not distinguishable** (122 test days)

### By season — Experiment B (individual models + equal blend, no training needed)

| Season | Method | MAE | RMSE | Bias | CSI ≥64.5 | POD | FAR | heavy events |
|---|---|---|---|---|---|---|---|---|
| 2018 | GraphCast | 2.316 | 4.64 | +0.14 | 0.306 | 0.364 | 0.343 | 195 |
| 2018 | HRES | 3.787 | 8.43 | +0.51 | 0.146 | 0.421 | 0.817 | 195 |
| 2018 | Equal blend | 2.702 | 5.66 | +0.32 | 0.248 | 0.400 | 0.606 | 195 |
| 2020 | GraphCast | 2.705 | 5.15 | +0.33 | 0.371 | 0.475 | 0.371 | 318 |
| 2020 | HRES | 4.229 | 8.75 | +0.79 | 0.253 | 0.544 | 0.679 | 318 |
| 2020 | Equal blend | 3.121 | 6.08 | +0.56 | 0.364 | 0.538 | 0.471 | 318 |
| 2022 | GraphCast | 2.519 | 4.82 | +0.37 | 0.408 | 0.477 | 0.261 | 321 |
| 2022 | HRES | 3.706 | 7.24 | +0.71 | 0.247 | 0.389 | 0.597 | 321 |
| 2022 | Equal blend | 2.790 | 5.33 | +0.54 | 0.325 | 0.411 | 0.392 | 321 |

## 12. Regimes — Experiment B

* feature set `rain`: k = 2, training silhouette 0.354, bootstrap ARI 0.964, validation MAE of regime blend 2.929
* chosen (lowest validation MAE, pre-registered rule): **`rain`**
  * REGIME_0: LOW RAINFALL (wettest: Northeast): 174 fit samples, mean IMD/ref rain 4.01 mm, heavy-rain frequency ratio 0.33, test samples 94
  * REGIME_1: MODERATE RAINFALL (wettest: Northeast): 558 fit samples, mean IMD/ref rain 8.90 mm, heavy-rain frequency ratio 1.21, test samples 272
* cluster structure: best silhouette 0.35 → **weak** (< 0.5)
* regime weights: spread across regimes max 0.081, median 0.029

Weights (lead-day static vs regime range across regimes and regions):

| Lead day | static GraphCast | static HRES | regime GraphCast (min–max) | regime HRES (min–max) |
|---|---|---|---|---|
| 1 | 0.659 | 0.341 | 0.60–0.72 | 0.28–0.40 |
| 2 | 0.593 | 0.407 | 0.53–0.66 | 0.34–0.47 |
| 3 | 0.597 | 0.403 | 0.52–0.65 | 0.35–0.48 |

## Conclusions (13–14 of the brief: did AI / adaptive weighting help?)

**Experiment A (IMD)**
* Equal blend vs best individual (GFS) MAE: meaningful improvement; heavy CSI vs best heavy-rain model (ECMWF): meaningful degradation.
* Skill-based static weights (Phase 2 inverse-error) vs equal blend: MAE not distinguishable; heavy CSI not distinguishable.
* Adaptive regime blend vs equal blend: MAE negligible (CI excludes 0 but |effect| < 1%); heavy CSI not distinguishable.
* **What the regime adds** — regime blend vs static inverse-error weights: MAE negligible (CI excludes 0 but |effect| < 1%); heavy CSI not distinguishable.
* Extreme-aware vs equal: MAE negligible (CI excludes 0 but |effect| < 1%); heavy CSI negligible (CI excludes 0 but |effect| < 0.005).

**Experiment B (ERA5 benchmark)**
* Equal blend vs best individual (GraphCast) MAE: meaningful degradation; heavy CSI vs best heavy-rain model (GraphCast): meaningful degradation.
* Skill-based static weights (Phase 2 inverse-error) vs equal blend: MAE meaningful improvement; heavy CSI meaningful improvement.
* Adaptive regime blend vs equal blend: MAE meaningful improvement; heavy CSI meaningful improvement.
* **What the regime adds** — regime blend vs static inverse-error weights: MAE negligible (CI excludes 0 but |effect| < 1%); heavy CSI meaningful improvement.
* Extreme-aware vs equal: MAE meaningful improvement; heavy CSI meaningful improvement.

**Did the AI model improve the blend?** Measurable only in Experiment B, i.e. agreement with ERA5 — not IMD:
* Adding GraphCast to the NWP model: inverse-error GraphCast+HRES blend vs HRES alone: MAE meaningful improvement (2.646 vs 3.706); heavy CSI meaningful improvement.
* Adding HRES to GraphCast: the same blend vs GraphCast alone: MAE meaningful degradation (2.646 vs 2.519); heavy CSI meaningful degradation. Against ERA5, GraphCast alone beats every blend; ERA5 favours GraphCast by construction (see limitations).

## 13. Limitations

* **No AI model in the IMD experiment.** Every accessible AI forecast (AIFS, GraphCast, GraphCastGFS, NOAA MLWP) outputs 6-hourly steps, which cannot bracket IMD's 03 UTC day; exact alignment would need temporal interpolation (not done).
* **Experiment B favours GraphCast by construction:** GraphCast is trained on and initialised from ERA5, which is also the reference; HRES starts from its own analysis. B is a benchmark of agreement with ERA5, not with observed rain.
* **ERA5 is a reanalysis**, not gauge data; its 1.5° rainfall is smoother than IMD; heavy-rain events at 1.5° are rare.
* **Experiment A has three seasons only** (ECMWF open data starts 2023-01-18): one training season, one validation, one test.
* ECMWF 2023 is 0.4° interpolated to 0.25°; its skill may differ from the native 0.25° seasons.
* The 2025 test season contains **July 2025, which was used to fit Phases 2–3** and whose results informed this design; no Phase 4 parameter is fitted on 2025, but the test is not researcher-blind.
* Regional blocks are coarse lat/lon boxes, not IMD subdivisions; grid-cell weights were not attempted.
* Confidence intervals resample days within one test season; they do not capture season-to-season variability.

## 14. Data-access issues

* ECMWF open-data AWS mirror throttles (`503 SlowDown`); handled by paced requests + exponential backoff (see `data_quality/missing_data.csv`).
* TIGGE needs an ECMWF account/API key and is 6-hourly; NOAA MLWP files are ~9.3 GB per forecast; Pangu and NeuralGCM (WB2) have no precipitation variable; FuXi covers 2020 only. Details: `PHASE4_DATA_AUDIT.md`.

## 15. Reproduce

```bash
cd backend
pytest -q
python scripts/audit_phase4_sources.py      # re-runs the data-access probes -> results/phase4/data_audit/
python scripts/run_phase4.py                # both experiments; first run downloads ~3 GB (cached afterwards)
```
