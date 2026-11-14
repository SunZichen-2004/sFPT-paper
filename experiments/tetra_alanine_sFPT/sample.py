"""Tetra-alanine (vacuum GAFF/SPONGE) + 3D φ CV + torus kernel + BAOAB-FP.

Sequence ACE–ALA–ALA–ALA–NME. CVs: φ1, φ2, φ3 (no ψ).
Fixed production lambda with optional prescribed healing.
Save the complete CV/bandwidth/coordinate trajectory into one NPZ archive.

Kernel / rescale recipe matches vacuum dialanine sFPT used in
``dialanine_main_picture`` (unnormalized row-sum band [2, 4], γ=10 /ps).
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "checkpoint.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from checkpoint import Restart
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))
from io_utils import load_torsion_indices
from fp_transformer_torch import (
    BaoabFokkerPlanckIntegrator,
    EnergySystem,
    GaffPotential,
    GaussianKernel,
    PerReplicaRescaleConfig,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates
from fp_transformer_torch.units import (
    KB_KCAL_PER_MOL_K,
    friction_per_picosecond_to_akma,
    timestep_picosecond_to_akma,
)

DEFAULT_SPONGE = EXPERIMENT_DIR / "tetra_alanine"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "backbone_phi.txt"
DEFAULT_INIT_NPY = EXPERIMENT_DIR / "init_coords_2048.npy"
DEFAULT_REMD = EXPERIMENT_DIR / "remd_280_400_n64_90605"


def initial_bandwidth_from_collective_variable(
    collective_variable: torch.Tensor,
) -> float:
    distance = torus_pairwise_distance(collective_variable)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    if not np.isfinite(median) or median <= 0.0:
        return 0.5
    return median


def select_phi_torsions(torsion_indices: np.ndarray) -> np.ndarray:
    """Keep 3 φ rows. A 6-row φψ file is subsampled as rows 0,2,4."""
    n = int(torsion_indices.shape[0])
    if n == 3:
        return torsion_indices
    if n == 6:
        return torsion_indices[0:6:2]
    raise ValueError(f"expected 3 φ or 6 φψ torsions, got {n}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=2048)
    p.add_argument(
        "--dump-every",
        type=int,
        default=200,
        help="MD steps between dumps of CV / bandwidth",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001, help="BAOAB dt in ps")
    p.add_argument("--baoab-dt", type=float, default=None, help="Alias of --timestep")
    p.add_argument(
        "--langevin-friction", type=float, default=10.0, help="Langevin γ in 1/ps"
    )
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_KCAL_PER_MOL_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=0.0)
    p.add_argument(
        "--init-npy",
        type=str,
        default=str(DEFAULT_INIT_NPY) if DEFAULT_INIT_NPY.is_file() else None,
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cv-grad-clip-norm", type=float, default=0.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=0.001)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--production-steps", type=int, default=25000)
    p.add_argument(
        "--archive-coordinates", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--production-lambda", dest="production_lambda", type=float, default=0.075
    )
    p.add_argument("--heal-lambdas", default="")
    p.add_argument("--heal-steps", type=int, default=100)
    return p


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    if args.production_lambda <= 0 or args.timestep <= 0:
        raise ValueError("lambda, timestep and FES intervals must be positive")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    dtype = torch.float64
    prefix = Path(args.sponge_prefix)
    lam = float(args.production_lambda)
    dt_ps = float(args.timestep)
    dump_every = max(1, int(args.dump_every))
    n_total = int(args.production_steps)
    if n_total < dump_every:
        raise ValueError(
            f"total steps {n_total} < dump-every {dump_every} (total-time={args.total_time} ps, dt={dt_ps} ps)"
        )
    n_stages = n_total // dump_every
    n_run = n_stages * dump_every
    dt_akma = timestep_picosecond_to_akma(dt_ps)
    gamma_ps = float(args.langevin_friction)
    gamma_akma = friction_per_picosecond_to_akma(gamma_ps, dt_ps, dt_akma)
    print(f"device={device}  beta0={args.beta0:.6g}  sponge={prefix}")
    print(
        f"[baoab time] dt={dt_ps} ps → dt_akma={dt_akma:.12g}; γ={gamma_ps} /ps → γ_akma={gamma_akma:.12g}; γ·dt={gamma_ps * dt_ps:.6g}"
    )
    print(
        f"[run] fixed λ={lam:g}  total={n_run} steps ({n_run * dt_ps:g} ps)  dump_every={dump_every}  n_stages={n_stages}  R={args.replicas}",
        flush=True,
    )
    potential = GaffPotential(str(prefix))
    print(f"GAFF num_atoms={potential.num_atoms}")
    if args.compile and device.type == "cuda":
        try:
            potential.energy_calculator = torch.compile(
                potential.energy_calculator, fullgraph=False, dynamic=False
            )
            print("[init] torch.compile enabled for GAFF energy")
        except Exception as exc:
            print(f"[warn] torch.compile failed: {exc}")
    torsion_indices = load_torsion_indices(Path(args.cv_file))
    torsion_indices = torsion_indices[torsion_indices.max(axis=1) < potential.num_atoms]
    torsion_indices = select_phi_torsions(torsion_indices)
    print(f"[CV] 3φ (0-based)={torsion_indices.tolist()}", flush=True)
    if args.init_npy:
        path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {path}: {loaded.shape}")
        if loaded.shape[1] != potential.num_atoms:
            raise ValueError(
                f"init npy n_atoms={loaded.shape[1]} != GAFF n_atoms={potential.num_atoms}"
            )
        if loaded.shape[0] < int(args.replicas):
            raise ValueError(
                f"init npy B={loaded.shape[0]} < --replicas={args.replicas}"
            )
        if loaded.shape[0] > int(args.replicas):
            idx = np.linspace(
                0, loaded.shape[0] - 1, int(args.replicas), dtype=np.int64
            )
            loaded = loaded[idx]
            print(f"[init] subsampled {path} → B={loaded.shape[0]}", flush=True)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        args.replicas = int(coordinate.shape[0])
    else:
        base = load_sponge_coordinates(prefix, num_atoms=potential.num_atoms)
        coordinate = (
            torch.as_tensor(base, dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(args.replicas, 1, 1)
        )
        if args.noise_scale > 0:
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            noise = torch.randn(coordinate.shape, dtype=dtype, generator=generator).to(
                device
            )
            coordinate = coordinate + args.noise_scale * noise
    energy_system = EnergySystem(potential, coordinate)
    collective_variable = Torsion(torsion_indices)
    collective0 = collective_variable(coordinate)
    bandwidth0 = initial_bandwidth_from_collective_variable(collective0)
    print(
        f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}  cv_shape={tuple(collective0.shape)}",
        flush=True,
    )
    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=args.rescale_adaptive_ref_quantile,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
        global_pull=0.02,
        fixed_row_sum_min=float(args.rescale_fixed_row_sum_min),
        fixed_row_sum_max=float(args.rescale_fixed_row_sum_max),
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
    )
    print(
        f"[init] rescale metric=unnormalized_K_row_sum  mode=adaptive_quantile  band=[{args.rescale_adaptive_band_low},{args.rescale_adaptive_band_high}]*median clipped to [{rescale.fixed_row_sum_min},{rescale.fixed_row_sum_max}]  bw_clip=[{rescale.bandwidth_min},{rescale.bandwidth_max}]"
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bandwidth0,
        distance_function=torus_pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )
    mass_path = Path(str(prefix) + "_mass.txt")
    if mass_path.is_file():
        mass_vals = np.loadtxt(mass_path, dtype=np.float64).reshape(-1)
        if (
            int(mass_vals[0]) == potential.num_atoms
            and mass_vals.size == potential.num_atoms + 1
        ):
            mass_vals = mass_vals[1:]
        mass = torch.as_tensor(
            mass_vals[: potential.num_atoms], dtype=dtype, device=device
        ).view(1, -1, 1)
    else:
        mass = 1.0
    cv_clip = (
        None
        if args.cv_grad_clip_norm is None or args.cv_grad_clip_norm <= 0.0
        else float(args.cv_grad_clip_norm)
    )
    integrator = BaoabFokkerPlanckIntegrator(
        energy_system,
        collective_variable,
        kernel,
        timestep=dt_akma,
        langevin_friction=gamma_akma,
        beta0=args.beta0,
        production_lambda=lam,
        mass=mass,
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        collective_variable_grad_clip=cv_clip,
        device=device,
        dtype=dtype,
    )
    integrator.lambda_current = lam
    integrator.sync_lambda_to_kernel_grad_beta()
    print(f"[init] cv_grad_clip_norm={cv_clip}  fixed λ={lam:g} (fixed production)")
    out = Path(args.output or EXPERIMENT_DIR / "results")
    out.mkdir(parents=True, exist_ok=True)
    from fp_transformer_torch.output import TrajectoryArchive

    recorder = TrajectoryArchive(
        out / "trajectory.npz",
        metadata=vars(args),
        include_coordinates=args.archive_coordinates,
    )
    restart = Restart(out, recorder, kernel)
    state_resume = restart.load(integrator.init_state(seed=args.seed))

    def on_frame(phase, step, lam, state):
        if phase == "production":
            if restart.resumed and step == 0:
                return
            step += restart.offset
        coords = state.coordinate
        recorder.append(
            phase,
            step,
            float(args.timestep),
            lam,
            coords,
            collective_variable(coords),
            kernel.bandwidth,
            metrics=integrator._last_step_metrics,
        )
        if phase == "production":
            restart.save(step, state)

    final = None
    try:
        final = integrator.run(
            args.production_steps - restart.offset,
            initial_state=state_resume,
            dump_every=args.dump_every,
            heal_lambdas=[]
            if restart.resumed
            else [float(x) for x in args.heal_lambdas.split(",") if x.strip()],
            heal_steps=args.heal_steps,
            on_frame=on_frame,
            verbose=args.verbose,
            verbose_pace=args.verbose_pace,
            show_progress=False,
        )
    finally:
        recorder.save(complete=final is not None)
        if final is not None:
            restart.finish()


if __name__ == "__main__":
    main()
