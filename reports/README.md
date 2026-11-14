# Single-seed 25 ps validation

6/8 cases completed and analyzed. Each uses seed 0 and the main-text replica count.

FES RMSE is against the archived **same-length 25 ps** calculation, in kcal/mol. Both surfaces are minimum-shifted; the mask is the original joint or projected FES below 12 kcal/mol. Tetra-alanine has a three-dimensional joint FES; its main projection is phi2/phi3. Other cases use the full two-dimensional FES.

| Case | Replicas | Joint FES RMSE | Main 2D RMSE | New delta G | Original 25 ps delta G | Independent reference delta G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [dialanine_vacuum_sfpt](dialanine_vacuum_sfpt_comparison.png) | 256 | 0.288 | 0.288 | 2.383 | 2.131 | 2.091 |
| [dialanine_vacuum_metad](dialanine_vacuum_metad_comparison.png) | 256 | 0.326 | 0.326 | 2.373 | 2.198 | 2.091 |
| [dialanine_water_sfpt](dialanine_water_sfpt_comparison.png) | 320 | 0.359 | 0.359 | 4.897 | 5.019 | 4.384 |
| [tetra_alanine_sfpt](tetra_alanine_sfpt_comparison.png) | 2048 | 0.232 | 0.204 | 1.436 | 1.484 | 0.301 |
| [tetra_alanine_metad](tetra_alanine_metad_comparison.png) | 64 | 0.670 | 0.480 | -0.301 | 0.091 | 0.301 |
| [rearrangement_metad](rearrangement_metad_comparison.png) | 64 | 1.995 | 1.995 | -8.191 | -11.661 | -6.310 |

Basin definitions and physical-density reconstruction are shared with the archived comparison. Independent references are umbrella sampling for dialanine and rearrangement, and REMD for tetra-alanine. Short-run agreement with the original 25 ps output does not establish convergence to the long-time paper reference. In particular, the original tetra-alanine sFPT 25 ps delta G is already 1.484 kcal/mol versus REMD 0.301 kcal/mol.

Reaction WT-MetaD has substantial archived run-to-run variation. The three original 25 ps delta G values are -11.661, -7.989 and -7.980 kcal/mol, versus the new -8.191. New FES RMSE against the three runs is 1.995, 0.727 and 0.632 kcal/mol. The main table retains original seed 0; all three comparisons are recorded in `rearrangement_metad_variation.json`. This is descriptive variation, not a confidence interval.

`*_metrics.json` records all marginal errors and the separate paper-reference comparisons. `*_fes.npz` holds reconstructed numerical surfaces; `benchmark_summary.csv` is the machine-readable table. Full trajectory NPZ files remain under local `results/`, with their checksums and shape checks in `archive_integrity.json`.
