# Phase 4 data availability audit

**Tested:** 2026-09-28, before any bulk download. Every "✅" below comes from an actual request (S3 listings, index files, Zarr metadata, file downloads), not from documentation. The probe outputs are summarised here.

## Summary

| # | Source | Rainfall variable | Grid | Steps | Dates (tested) | Inits | Access | Exact IMD 03 UTC window? | Used in |
|---|---|---|---|---|---|---|---|---|---|
| A | NOAA **GFS** 0.25° (`noaa-gfs-bdp-pds`, AWS) | `APCP 0-N h acc` (kg m⁻²) | 0.25° | 1 h to 120 h | **2021-01-01 → today** (JJAS spot checks 2021–25: no gaps) | 00/06/12/18 | open, public domain | ✅ (3 h steps) | **Exp. A** |
| A′ | NOAA **GFS 1.0°** (same bucket, `pgrb2.1p00`) | also `PRMSL`, `UGRD/VGRD 850 mb`, `PWAT` | 1.0° | 3 h | 2021 → today | 00/06/12/18 | open | n/a (regime features only) | **Exp. A regimes** |
| B | **ECMWF IFS** open data, AWS mirror (`ecmwf-forecasts`) | `tp` (m, since init) | **0.4°** (`0p4-beta`) to 2024-01-31, then **0.25°** | 3 h to 144 h | **2023-01-18 → today**; nothing for 2021–2022 | 00/12 (oper) | open, CC-BY-4.0 | ✅ | **Exp. A** (2023–2025) |
| C | **ECMWF AIFS** (AI) in the same mirror (`aifs/` 2024, `aifs-single/` 2025) | `tp` | 0.25° | **6 h only** | 2024-03 → today | 00/06/12/18 | open, CC-BY-4.0 | ❌ windows end 00/06/12/18 UTC | not used (blocker) |
| D | **WeatherBench 2** GraphCast (`graphcast_v2`, AI) | `total_precipitation_6hr/24hr` (m) | 1.5° (240×121, conservative) | 6 h to 240 h | 2018, 2020, 2022 datasets (inits every 6 h) | 00/06/12/18 | open (GCS HTTPS); GraphCast outputs are non-commercial (CC BY-NC-SA, model licence) | ❌; verified against ERA5 | **Exp. B** |
| E | **WB2 IFS HRES** | `total_precipitation` (m, since init) | 1.5° | 6 h to 240 h | 2016–2022 | 00/12 | open | ❌ (6 h) | **Exp. B** |
| F | **WB2 ERA5** (reference) | `total_precipitation_24hr` (m) | 1.5° | 6-hourly | 1959 → 2023-01-10 | – | open (Copernicus licence) | – | **Exp. B truth** |
| G | WB2 Pangu-Weather | **no precipitation variable** | 1.5° | 6 h | 2018–2022 | 00/12 | open | – | ❌ cannot forecast rain |
| H | WB2 NeuralGCM (deterministic) | only `P_minus_E_cumulative` (precipitation minus evaporation) | 1.5° | 24 h | 2020 | 00/12 | open | – | ❌ not precipitation |
| I | WB2 FuXi | `total_precipitation_6hr` | 1.5° | 6 h | **2020 only** | 00/12 | open | ❌ | ❌ single season, cannot be split train/test |
| J | NOAA **MLWP** archive (`noaa-oar-mlwp-data`): GraphCast / Pangu / FourCastNet / Aurora | GraphCast only | 0.25° | 6 h | 2020-09 → today | 00/12 | open | ❌ (6 h) | ❌ **~9.3 GB per forecast file** (one NetCDF per init, no index) |
| K | NOAA **GraphCastGFS** operational (`noaa-nws-graphcastgfs-pds`) | yes | 0.25° | 6 h | 2024-02-05 → today | 00/06/12/18 | open | ❌ (6 h) | ❌ (IMD alignment blocker) |
| L | **ECMWF TIGGE** (Phase 2 audit) | `tp` | 0.5° | 6 h | long record | 00/12 | account + licence + API key (`403 Missing access token`) | ❌ | ❌ |
| M | **IMD** gridded rainfall, yearly archive | 24 h rain (mm) | 0.25° | daily, 08:30 IST → 08:30 IST | archives **2023, 2024, 2025 downloaded and size-validated today** (1901 → 2025 offered) | – | open (cite Pai et al. 2014) | defines the window | **Exp. A truth** |

### Checks behind the summary

* **Download size** (JJAS season, 00Z cycles, leads 3/27/51/75 h, measured message sizes):
  * GFS 0.25° APCP ≈ 0.3 GB;
  * GFS 1.0° atmospheric fields ≈ 0.1 GB;
  * ECMWF 0.4° `tp` ≈ 0.2 GB, ECMWF 0.25° ≈ 0.4 GB;
  * IMD 25 MB per year;
  * WB2: two chunks of about 1 MB (uncompressed) per init, model and season, so about 0.3 GB per model-season, plus about 40 MB of ERA5.
* **WB2 semantics, verified on real data** (2020-07-10 00Z, India box):
  * GraphCast `24hr@24h` equals the sum of its four 6 h buckets (max difference 7e-9).
  * GraphCast values are in metres: the mean ratio to ERA5 (which is in m) is 1.02.
  * GraphCast 6 h buckets can be slightly **negative** (−9e-5 m); these are clipped to 0 and counted.
  * HRES `total_precipitation` is accumulated since initialisation: it is 0 at step 0, and `tp@48 − tp@24` equals `24hr@48` (difference 8e-7).
* **ERA5 favours GraphCast:** GraphCast is trained on ERA5 and initialised from it. On the probe day, its day-1 pattern correlation with ERA5 was 0.986, against 0.914 for HRES.

## Consequences for Phase 4

1. **Every AI source outputs precipitation only at 6-hourly steps.**
   * From any cycle (00/06/12/18 UTC), the step differences never bracket IMD's 03 UTC → 03 UTC day: the window would need steps 3/27, 21/45, 15/39 or 9/33.
   * An exact AI-vs-IMD verification is therefore **impossible without temporal interpolation**, which this project does not do.
   * **No AI model enters the IMD experiment.**
2. **Two strictly separate experiments.**
   * **Experiment A (India operational, IMD truth):** GFS 0.25° + ECMWF IFS (0.4° in 2023, 0.25° in 2024–2025), JJAS **2023, 2024, 2025**. 2021–2022 are excluded because ECMWF open data does not exist for them.
   * **Experiment B (AI/NWP benchmark, ERA5 truth):** GraphCast (AI) + IFS HRES (NWP), 1.5°, JJAS **2018, 2020, 2022**, 24 h windows ending 00 UTC.
   * Scores from A and B are never put in one table.
3. **ECMWF resolution changes inside Experiment A.** The 2023 season is 0.4° and must be interpolated bilinearly to the 0.25° grid; 2024–2025 are native 0.25°. This is recorded per season.
4. **Regime features for Experiment A.** GFS 1.0° MSLP, 850 hPa zonal wind and precipitable water, all available 2021 → today, allow the requested comparison between **rainfall-only** and **atmospheric + rainfall** regimes.
5. **July 2025 caveat.** It was used to fit Phases 2–3, and its results informed the Phase 4 design. Phase 4 never fits on 2025, but the 2025 test season is therefore not *researcher-blind*. This is stated in the results.
