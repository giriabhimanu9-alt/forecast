# july2025

```
Experiment: july2025   [HISTORICAL EXPERIMENT]
Models available: GFS, ECMWF
Aligned samples: 93 (lead days 1, 2, 3), train days 23, test days 8
Test period (IMD labels): 2025-07-24 .. 2025-07-31

TEST PERIOD, all lead days pooled (mm / 24 h, common grid points only):
Best individual model by MAE: GFS (9.407)
Other individual model ECMWF MAE: 9.485
Equal blend MAE: 8.762
Inverse-error blend MAE: 8.761
Regional adaptive blend MAE: 8.761
Blend improvement versus best individual: best blend = Regional blend, +6.87% MAE change (MAE difference -0.646 mm, 95% CI [-0.837, -0.451], 8 test days) -> better (95% CI < 0)
The blend beat the best individual model on the independent test period.
Inverse-error blend vs equal blend: MAE difference -0.0004 mm (95% CI [-0.0018, +0.0011])
Regional blend vs equal blend: MAE difference -0.0006 mm (95% CI [-0.0037, +0.0036])
-> Skill-based weights add no measurable benefit over equal weighting here (|diff| < 0.01 mm); the blend gain comes from averaging the two models.
Mean bias (forecast - IMD): GFS -0.646, ECMWF +0.939, Equal blend +0.147, Inverse-error blend +0.148, Regional blend +0.145
Heavy rain (>= 64.5 mm/24 h) CSI: best individual ECMWF 0.095, best blend Regional blend 0.070 -> blend is LOWER (averaging smooths extremes)

IMD period check (lead-day-1 pattern correlation by label offset -1/0/+1 day): GFS: -1:0.285, 0:0.442, 1:0.297; ECMWF: -1:0.308, 0:0.531, 1:0.315
Convention consistent with data: YES
```
