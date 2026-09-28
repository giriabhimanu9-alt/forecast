# july2025_pilot

```
Experiment: july2025_pilot   [ALIGNMENT/PIPELINE TEST - NOT A SCIENTIFIC SKILL CLAIM]
Models available: GFS, ECMWF
Aligned samples: 30 (lead days 1, 2, 3), train days 8, test days 2
Test period (IMD labels): 2025-07-09 .. 2025-07-10

TEST PERIOD, all lead days pooled (mm / 24 h, common grid points only):
Best individual model by MAE: GFS (9.249)
Other individual model ECMWF MAE: 9.502
Equal blend MAE: 8.796
Inverse-error blend MAE: 8.797
Regional adaptive blend MAE: 8.801
Blend improvement versus best individual: best blend = Equal blend, +4.89% MAE change (MAE difference -0.453 mm, 95% CI [-0.507, -0.398], 2 test days) -> not assessed (only 2 test days, need 7)
No improvement claim: this run is an alignment/pipeline test (too few days).
Inverse-error blend vs equal blend: MAE difference +0.0007 mm (95% CI [+0.0002, +0.0012])
Regional blend vs equal blend: MAE difference +0.0044 mm (95% CI [-0.0045, +0.0133])
-> Skill-based weights add no measurable benefit over equal weighting here (|diff| < 0.01 mm); the blend gain comes from averaging the two models.
Mean bias (forecast - IMD): GFS -0.989, ECMWF +0.694, Equal blend -0.147, Inverse-error blend -0.141, Regional blend -0.130
Heavy rain (>= 64.5 mm/24 h) CSI: best individual GFS 0.088, best blend Regional blend 0.086 -> blend is LOWER (averaging smooths extremes)

IMD period check (lead-day-1 pattern correlation by label offset -1/0/+1 day): GFS: -1:0.292, 0:0.418, 1:0.319; ECMWF: -1:0.331, 0:0.490, 1:0.354
Convention consistent with data: YES
NOTE: sample too small for a skill claim; these numbers only demonstrate that the pipeline works.
```
