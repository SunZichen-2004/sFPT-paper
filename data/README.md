# Paper FES data

This directory supplies the numerical FES versus time for all four paper systems and both methods, at every stored time point and seed used by the paper. It also includes the current numerical source caches for SI replica-number, lambda and metadynamics hill-height comparisons.

| Main series | Seeds | Stored time coverage / ps | Representation |
| --- | --- | --- | --- |
| Dialanine vacuum sFPT | 0, 1, 2 | 5–50, every 5 | joint 2D FES and phi profile |
| Dialanine vacuum WT-MetaD | 0, 1, 2 | 1–50, every 1 | cumulative 2D FES and phi profile |
| Dialanine water sFPT | 0, 2 | 10–200, every 10 | joint 2D FES and phi profile |
| Dialanine water WT-MetaD | 0, 1, 2 | 1–200, every 1 | cumulative 2D FES and phi profile |
| Tetra-alanine sFPT | 0, 1, 2 | 20–500, every 20 | all three 1D and pairwise 2D marginals of the physical 3D distribution |
| Tetra-alanine WT-MetaD | 0, 1, 2 | 0–500, every 20 | all three 1D and pairwise 2D marginals; separate original final 10 ns projections |
| Reaction sFPT | 10, 0, 100 | 4–80, every 4; final 83 | joint 2D FES and d2-d1 profile; final 79–83 ps window is marked separately |
| Reaction WT-MetaD | 0, 1, 2 | 1–400, every 1 | original paper FFT/grid reconstruction of cumulative 2D FES |

## Read a time series

Files are under `fes_time/CASE/seed_N.npz`. All NPZ files are readable with `allow_pickle=False`.

```python
import numpy as np
with np.load('data/fes_time/dialanine_water_sfpt/seed_0.npz', allow_pickle=False) as data:
    time_ps = data['time_ps']
    phi, psi = data['axis_0'], data['axis_1']
    fes = data['fes_2d_0_1']  # [time, phi, psi], kcal/mol
    delta_g = data['delta_g_kcal_mol']
```

`axis_i` supplies each 2D grid axis. `fes_2d_i_j` has shape `[time, len(axis_i), len(axis_j)]`. One-dimensional paper projections can use a separate bin grid, supplied as `axis_1d_i`. The reaction's difference-coordinate profile is `fes_1d_reaction` on `axis_1d_0`. `metadata_json` records reconstruction settings. `support_mask_2d_i_j` marks the finite support of each surface; masked/zero-probability regions from the original FES remain masked, without filling or interpolation. For display with `pcolormesh(x,y,...)`, transpose the selected two-dimensional frame.

```bash
python fes_time.py plot --case rearrangement_sfpt --seed 0 --times 4 40 83
python fes_time.py verify
```

## Numerical source caches and SI

`sources/` preserves the paper and SI numerical caches under their original experiment-relative hierarchy. These include FES at all stored times and parameter values, static surfaces, references, per-seed basin values, and plotted convergence metrics. The source field names and numeric precision are retained. Superseded uncorrected dipeptide FES caches are replaced by their existing `alpha_corrected` versions; raw MD coordinate trajectories are not duplicated here.

`catalog.json` lists every payload's SHA-256, source and size. Historical 3 ps water input/reference caches are marked in the catalog because some sFPT fields use the old `1/lambda` scaling; the canonical series and corrected 5/10 ps caches use the paper's physical `(1+lambda)/lambda` reconstruction. Larger source NPZ files are split by **array key** into `*.partNNN.npz`; the manifest records each part's keys and the original file checksum. This keeps individual Git files small without dropping time points, lowering precision, or changing any array. Load all parts into one dictionary to recover the original key set.

The reaction WT-MetaD time series uses the original weighted histogram, FFT convolution and interpolation on its 140×140, 1–7 Å grid. Its basin curves match the archived paper curves. The dipeptide WT-MetaD surfaces also match their archived cache points. Checks are recorded in `validation.json`.

Static tetra-alanine WT-MetaD 10 ns projections retain their separate native grid in `sources/tetra_alanine_sFPT/replica_scan_20261031/lambda0p075_500ps_3seeds/analysis_joint3d_20ps_latestmeta/meta_final10ns_all_projections.npz`. The paper time-series snapshots stop at 500 ps; no surfaces between 500 ps and the final 10 ns snapshot have been fabricated. The reaction's final 79–83 ps window is included with `is_static_final=True` and its explicit start time; it is not a further nonoverlapping 4 ps block.
