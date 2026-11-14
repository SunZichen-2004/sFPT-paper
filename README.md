# sFPT paper experiments

Standalone inputs, single-seed reproduction runners, original comparison data and FES analysis for the subspace Fokker–Planck Transformer paper. The sampling library is maintained separately in [fp_transformer_torch](https://github.com/SunZichen-2004/fp_transformer_torch).

Each case requests 25 ps of **production per replica**, with seed 0, a 1 fs timestep, temperature 300 K and friction 10/ps. Preparation is additional and is recorded separately.

| System | sFPT replicas | Production lambda | Healing | WT-MetaD walkers | Initial hill height / eV | Bias factor |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Dialanine, vacuum | 256 | 0.1 | 0.5 then 0.2; 500 steps each | 256 | 0.005 | 15 |
| Dialanine, water | 320 | 0.0625 | 0.5 then 0.1; 100 steps each | 320 | 0.004 | 15 |
| Tetra-alanine, vacuum | 2048 | 0.075 | none | 64 | 0.02 | 15 |
| Claisen rearrangement, water | 512 | 0.025 | 0.1 then 0.05; 500 steps each | 64 | 0.02 | 30 |

Metadynamics deposits every 50 steps. Alanine widths are 0.35 rad per CV; reaction widths are 0.2 Å. The matched dipeptide settings and reaction walker scan keep the nominal initial sum of hill heights at `R*h = 1.28 eV`.

The supplied SPONGE-format alanine parameter tables encode the Amber force field used by the paper; the loader retains its historical `GaffPotential` name. Water boxes, MM topologies and initial ensembles are included. The reaction loads MACE-MH-0 from the model cache; its checkpoint is not redistributed here. `input_checksums.json` records every molecular input.

## Installation

Install the cleaned library branch alongside this repository:

```bash
git clone https://github.com/SunZichen-2004/fp_transformer_torch.git
cd fp_transformer_torch
git checkout cleanup/public-core-20261114
pip install -e ".[simulation,mlip,analysis,dev]"
cd ../sFPT-paper
pip install -r requirements.txt
```

The observed package versions are recorded in `environment-tested.json`. CUDA production runs require a suitable GPU. `--device cpu` in an individual runner is intended for small diagnostics.

## Run

```bash
python run_case.py dialanine_vacuum_sfpt
python run_case.py dialanine_vacuum_metad
python run_case.py tetra_alanine_sfpt
```

The eight case names and resolved parameter sets are in `cases.json`. Paths are resolved relative to this repository. `--dry-run` prints the complete command; `--steps 100 --output smoke/example` provides a short dynamics check. Every standalone `experiments/*/sample.py` also has `--help`.

Vacuum and water WT-MetaD retain the selected original scripts' 5 ps and 10 ps unbiased preparation, respectively. Reaction sFPT retains 0.1 ps of unbiased preparation before healing. Tetra-alanine uses the paper's initial ensemble after its additional 1 ps unbiased relaxation. These times are excluded from the 25 ps production time.

Run on an existing Slurm allocation:

```bash
srun --jobid=YOUR_JOB_ID -N1 -n1 -c12 python run_case.py rearrangement_sfpt
```

Alternatively, submit the supplied portable template:

```bash
sbatch --export=ALL,CASE=dialanine_water_sfpt cluster/submit.slurm
```

## Output and continuation

The completed result is `results/CASE/trajectory.npz`, with explicit frame/replica/CV axes, full coordinates by default, bandwidths, phase, time, lambda, metadata and completeness status. Metadynamics also saves every deposited hill and its final B-spline bias grid. No per-stage NPY fragments are produced.

A local `checkpoint.pt` is written once per ps during a running calculation. Repeating the command restores coordinates, momenta, bandwidth, accumulated output and RNG states. The temporary checkpoint is removed after the complete NPZ is written. Checkpoints are trusted local Python state and should only be resumed from your own run directory. A failed run's NPZ is explicitly marked incomplete.

## Compare with original results

```bash
python analyze.py --references-only --device cuda
python analyze.py --device cuda
```

The analysis reconstructs the full physical joint distribution before marginalization. It reports one- and two-dimensional FES RMSE in kcal/mol, the basin free-energy difference, and its difference from both the original 25 ps calculation and the independent paper reference. Figures compare new 25 ps, original 25 ps, and long-time paper/REMD surfaces. All metrics and scientific plots are written to `reports/`.

sFPT final-window lengths are 5 ps (vacuum dipeptide), 10 ps (water), 20 ps (tetrapeptide) and 4 ps (reaction). The original and new 25 ps results use identical reconstruction windows. Metadynamics uses the cumulative bias at 25 ps. RMSE masks and minimum-shift conventions are recorded with the metrics. The paper's basin-error guides are reported separately; they are not confidence intervals or proof of full FES convergence.

`references/` contains compact original 25 ps CV/bandwidth or HILLS archives and the longer paper surfaces. `references/provenance.json` records their exact sources, time calibration and checksums. The archived reaction sFPT seed is 10; the requested new benchmark uses seed 0. All other selected original 25 ps references use seed 0.

A 25 ps, single-seed benchmark has a different statistical scope from the paper's multi-seed static surfaces: vacuum dipeptide 25–50 ps, water 100–200 ps, tetra-alanine 450–500 ps (and WT-MetaD 10 ns), and reaction sFPT 79–83 ps. Agreement is evaluated quantitatively; short-run completion alone does not establish agreement with the final paper FES.
