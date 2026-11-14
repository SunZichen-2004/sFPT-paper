# Experimental protocols and inputs

All dynamics use 300 K, a 1 fs timestep and Langevin friction 10/ps. The native Amber/SPONGE backend converts the supplied ps timestep to its force-model units; the TorchSim backend uses eV, Å and ps. The complete resolved settings are in `protocols/paper.json`, and `run_case.py CASE --dry-run` prints the executable command.

## Force fields, structures and collective variables

| Case | Supplied files, relative to `experiments/` |
| --- | --- |
| Dialanine vacuum, both methods | `dialanine/dialanine_*` parameter tables, `dialanine.pdb`, `dialanine.xyz`, `backbone_cv.txt`, `init_coords_256.npy` |
| Dialanine water, both methods | `dialanine_water/mm_box/box.xyz`, `box.pdb`, `cell.txt`, `topology.npz`, `backbone_cv.txt`; `dialanine_water/init_coords_1024.npy` |
| Tetra-alanine vacuum, both methods | `tetra_alanine_sFPT/tetra_alanine_*` parameter tables, structure files, `backbone_phi.txt`; method-specific initial pools below |
| Reaction, both methods | `rearrangement_water_fpt/mlmm_box/` topology, coordinates, cell and solute force-field files; `cv_pairs.txt`; `rearrangement_water_metadynamics/init_coords.npy` for WT-MetaD |

The supplied tabulated parameters are the exact exported molecular mechanics inputs used by the sampling code. The alanine force-field loader retains the historical `GaffPotential` name. Explicit solvent uses TIP3P: the dipeptide box contains 192 waters in an 18 Å box, and the reaction box contains 334 waters in a 22 Å box. The reaction solute has 20 ML atoms and uses MACE-MH-0 with the `omol` head, mechanical-embedding ML/MM and the included classical solute–solvent interactions. Model weights are obtained using `download_model.py`; the exact checkpoint hash is recorded in `model_checksums.json`.

`input_checksums.json` covers every force-field, topology, CV-definition, structure and initial-ensemble file. Run `python check_inputs.py` before sampling. No original cluster directories or symlinks are required.

## sFPT

| System | R | Fixed production lambda | Healing lambdas | Steps at each healing lambda | Additional preparation | Seeds |
| --- | ---: | ---: | --- | ---: | --- | --- |
| Dialanine vacuum | 256 | 0.1 | 0.5, 0.2 | 500 | none | 0, 1, 2 |
| Dialanine water | 320 | 0.0625 | 0.5, 0.1 | 100 | supplied initial pool | 0, 2 |
| Tetra-alanine vacuum | 2048 | 0.075 | none | — | supplied pool already includes the additional 1 ps unbiased relaxation | 0, 1, 2 |
| Reaction water | 512 | 0.025 | 0.1, 0.05 | 500 | 100 unbiased steps from the box coordinates | 10, 0, 100 |

The joint replica gradient and per-replica raw-row-sum bandwidth update are used. Bandwidth updates occur every 20 steps, with an adaptive median band of 0.8–1.2 and absolute row-mass bounds 2–4 for alanines or 2–5 for the reaction. Bandwidth limits are 0.001–8 for alanines and 0.001–4 for the reaction. The reaction uses distance walls at 8 Å with force constant 10, plus the supplied soft-min wall (`a=10`, threshold 2.4 Å, force constant 10). These settings are executable fields in the protocol file rather than assumptions about parser defaults.

Explicit-solvent electrostatics uses PME: dipeptide cutoff 8 Å, reaction cutoff 9 Å, reciprocal mesh spacing 1.5 Å. Production time excludes all preparation and healing. The archived legacy runs can contain tiny ESS-driven lambda increments; the cleaned sampler uses the stated constant production lambda. Archived FES data retain the original reconstruction settings and are not replaced by newly generated examples.

## WT-MetaD

| System | Walkers | Initial hill height / eV | Bias factor | Widths | Unbiased preparation | Seeds |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Dialanine vacuum | 256 | 0.005 | 15 | 0.35, 0.35 rad | 5 ps | 0, 1, 2 |
| Dialanine water | 320 | 0.004 | 15 | 0.35, 0.35 rad | 10 ps | 0, 1, 2 |
| Tetra-alanine vacuum | 64 | 0.02 | 15 | 0.35, 0.35, 0.35 rad | supplied initial pool | 0, 1, 2 |
| Reaction water | 64 | 0.02 | 30 | 0.2, 0.2 Å | supplied initial pool | 0, 1, 2 |

Hills are deposited every 50 MD steps. The interpolation grid uses 81 points per dimension. Angular CVs are periodic; the reaction bias grid spans 0–10 Å per distance. The initial nominal sum of walker hill heights is 1.28 eV for the main comparisons.

## Initial ensembles

The vacuum dipeptide supplies separate pools for R=64, 128, 256, 512, 1024 and 2048. Changing `--replicas` selects the matching pool. Water cases subsample the included 1024-member water ensemble.

The main tetra-alanine sFPT case uses `tetra_alanine_sFPT/init_coords_2048.npy`, after the documented additional 1 ps unbiased relaxation. Replica scans use `tetra_alanine_sFPT/initial_ensemble/init_coords_R.npy` for R=1024, 2048, 4096 or 8192. **Tetra-alanine WT-MetaD uses the separate original `tetra_alanine_sFPT/init_coords_1024.npy` pool**, before selecting 64 walkers. The two 1024-member pools are distinct and are both retained.

The reaction sFPT case replicates the supplied box and performs its stated preparation. WT-MetaD uses its archived 64-member prepared ensemble. The original reaction replica/height scan scripts reuse this same pool for R=256 and512; the retained runner selects or repeats initial configurations with their original linspace index rule. For custom pools use `--init-npy PATH --output PATH`.

## FES reconstruction and time windows

| System | sFPT time-series windows | Static sFPT window | WT-MetaD time series | Static WT-MetaD surface |
| --- | --- | --- | --- | --- |
| Dialanine vacuum | nonoverlapping 5 ps | 25–50 ps | cumulative bias through 50 ps, 1 ps spacing | mean bias over 25–50 ps |
| Dialanine water | nonoverlapping 10 ps | 100–200 ps | cumulative bias through 200 ps, 1 ps spacing | mean bias over 100–200 ps |
| Tetra-alanine | nonoverlapping 20 ps | 450–500 ps | stored snapshots through 500 ps, 20 ps spacing, including time zero | final 10 ns projections |
| Reaction | nonoverlapping 4 ps through 80 ps | 79–83 ps | cumulative bias through 400 ps, 1 ps spacing | final cumulative surface |

The reaction data also include the separate final 79–83 ps surface. Its time-series entry is marked `is_static_final`; it overlaps the preceding 76–80 ps window. The tetra-alanine final 10 ns projections are supplied separately at their original grid resolution. No intermediate 500–10000 ps surfaces are interpolated or invented.

For sFPT, average the joint KDE in the selected time window, raise it to `(1+lambda)/lambda`, normalize, and only then marginalize. WT-MetaD uses `F=-gamma*V/(gamma-1)`. FES values are in kcal/mol, with each surface shifted to a minimum of zero. The paper's basin convention is `delta G = kB*T*ln(P1/P2)`: P1 has phi<0 for dialanine, all three phi angles<0 for tetra-alanine, and d2-d1<0 for the reaction.

The bundled tetra-alanine 1D/2D time-dependent projections were computed from full three-dimensional physical probabilities. A 2D slice of the three-dimensional bias is not used as a marginal FES. The REMD estimate supplied with the paper remains a preliminary independent population reference.

## Parameter scans

`protocols/scans.json` lists the R/lambda/hill-height variants represented in the SI data. For example:

```bash
python run_case.py dialanine_vacuum_sfpt --replicas 512 --production-lambda 0.2 --seed 0
python run_case.py tetra_alanine_sfpt --replicas 4096 --seed 0
python run_case.py tetra_alanine_sfpt --replicas 2048 --production-lambda 0.03 --seed 3 --steps 100000
```

When `--replicas` changes a WT-MetaD case and no explicit `--hill-height` is supplied, the runner sets the height to `1.28/R` eV, preserving the paper's matched total deposition budget.

Use one process per available GPU or allocate the GPUs needed by the standalone reaction runner's `--devices` option. Run seeds independently; the protocol's seed lists define the paper series. The cleaned runners seed both the global PyTorch RNG and TorchSim's per-state RNG. Exact trajectories can change across software/hardware versions; the force fields, sampling parameters and FES definitions are provided explicitly.
