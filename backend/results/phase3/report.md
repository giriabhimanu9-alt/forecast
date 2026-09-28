# phase3_july2025

```
Phase 3 experiment: phase3_july2025  (PROOF OF CONCEPT: one monsoon month, 8 test days)
Train 2025-07-01..2025-07-17 | validation 2025-07-18..2025-07-23 | test 2025-07-24..2025-07-31 (24 forecast cases, 119136 scored land points, 2952 heavy-rain events)
Regimes: k = 4 (from forecast-time features); REGIME_0: LOW RAINFALL (wettest: Central) [test samples: 0]; REGIME_1: MODERATE RAINFALL (wettest: East) [test samples: 1]; REGIME_2: MODERATE RAINFALL (wettest: Central) [test samples: 23]; REGIME_3: MODERATE RAINFALL (wettest: South) [test samples: 0]

TEST PERIOD, all lead days (mm/24 h)  MAE / RMSE / bias / corr | heavy>=64.5: CSI / POD / FAR
  GFS                          9.407 / 18.62 / -0.65 / 0.427 | 0.052 / 0.084 / 0.878
  ECMWF                        9.485 / 18.52 / +0.94 / 0.467 | 0.095 / 0.136 / 0.760
  Equal blend                  8.762 / 16.68 / +0.15 / 0.512 | 0.070 / 0.095 / 0.788
  Phase 2 inverse-error        8.761 / 16.68 / +0.15 / 0.512 | 0.070 / 0.095 / 0.787
  Phase 3 regime blend         8.767 / 16.68 / +0.16 / 0.512 | 0.070 / 0.095 / 0.788
  Phase 3 extreme-aware        8.890 / 17.16 / +0.24 / 0.493 | 0.068 / 0.093 / 0.796
NOTE: 23 of 24 test cases fall in one regime (REGIME_2); on this test period regime conditioning can act only like a single fixed weight set.
Heavy regimes for the extreme-aware blend (fit-period heavy-rain frequency above average): REGIME_1 (ratio 1.20), REGIME_2 (ratio 1.18); none reaches the 1.5 ratio required for a HEAVY name
Cluster structure is weak (best training silhouette 0.33 < 0.5); regimes are loosely separated.

Regime weights: spread across regimes max 0.078, median 0.038 (max |regime - static| 0.060)
Phase 3 regime blend vs Equal blend: MAE +0.0057 mm [+0.0033, +0.0079] -> worse (95% CI > 0); heavy CSI +0.0001 [-0.0010, +0.0016] -> not distinguishable
Phase 3 regime blend vs Phase 2 inverse-error: MAE +0.0061 mm [+0.0028, +0.0089] -> worse (95% CI > 0); heavy CSI -0.0001 [-0.0006, +0.0004] -> not distinguishable
Phase 3 extreme-aware vs Equal blend: MAE +0.1285 mm [+0.0583, +0.2025] -> worse (95% CI > 0); heavy CSI -0.0017 [-0.0123, +0.0083] -> not distinguishable
Phase 3 extreme-aware vs Phase 2 inverse-error: MAE +0.1289 mm [+0.0578, +0.2033] -> worse (95% CI > 0); heavy CSI -0.0020 [-0.0132, +0.0086] -> not distinguishable
Extreme-aware vs ECMWF heavy CSI: -0.0266 [-0.0563, -0.0020] -> worse (95% CI < 0)
Validation (18-23 Jul, dev fit on 1-17 Jul): MAE equal 7.706, regime 7.702, extreme-aware 7.747; heavy CSI equal 0.089, regime 0.087, extreme-aware 0.087
Grid-cell-level adaptive weights were not statistically supportable with the current sample size. (median heavy events per land cell in the fit period: 0)
All statements refer to the 24-31 July 2025 test period only; no generalisation is claimed.
```
