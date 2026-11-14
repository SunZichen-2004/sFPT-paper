"""Dialanine (GAFF/SPONGE) + torsion CV + torus kernel + BAOAB-FP production."""

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

DEFAULT_SPONGE = EXPERIMENT_DIR / "dialanine"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "backbone_cv.txt"
DEFAULT_INIT_NPY = EXPERIMENT_DIR / "init_coords_256.npy"


def initial_bandwidth_from_collective_variable(
    collective_variable: torch.Tensor,
) -> float:
    """Median pairwise torus distance (same recipe as JAX ``init_bandwidth``)."""
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


def load_torsion_indices(path: Path) -> np.ndarray:
    """Load torsion quadruples; file is 1-based (same as JAX ``sample_xponge``)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        chunk: list[int] = []
        for part in parts:
            try:
                chunk.append(int(part) - 1)
            except ValueError:
                continue
            if len(chunk) == 4:
                rows.append(chunk)
                chunk = []
    if not rows:
        raise ValueError(f"no torsion indices in {path}")
    return np.asarray(rows, dtype=np.int64)


def plot_phi_psi(
    collective_variable: np.ndarray,
    save_path: Path,
    *,
    title: str,
    x_name: str = "phi",
    y_name: str = "psi",
) -> None:
    """Scatter φ–ψ (radians) for dialanine backbone torsions."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip {save_path}: {exc}", flush=True)
        return
    cv = np.asarray(collective_variable, dtype=np.float64)
    if cv.ndim == 3:
        cv = cv.reshape(cv.shape[0], cv.shape[1])
    if cv.shape[1] < 2:
        print(f"[warn] need ≥2 CVs for phi-psi, got shape {cv.shape}", flush=True)
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(cv[:, 0], cv[:, 1], c="C3", s=2.0, alpha=0.6, linewidths=0)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(f"{x_name} (rad)")
    plt.ylabel(f"{y_name} (rad)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"[plot] wrote {save_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=256)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--timestep",
        type=float,
        default=0.001,
        help="BAOAB dt in picoseconds (converted to AKMA internally)",
    )
    p.add_argument(
        "--baoab-dt", type=float, default=None, help="Alias of --timestep (ps)"
    )
    p.add_argument(
        "--langevin-friction",
        type=float,
        default=1.0,
        help="Langevin γ in 1/ps (converted to AKMA internally)",
    )
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_KCAL_PER_MOL_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument(
        "--production-lambda", type=float, default=0.1, dest="production_lambda"
    )
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="Coordinate jitter (Å). Keep 0 when using a diverse --init-npy.",
    )
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
    p.add_argument(
        "--cv-grad-clip-norm",
        type=float,
        default=0.05,
        help="Clip CV-space kernel gradient by global norm (JAX-compatible). Needed for GAFF stability at small λ; set <=0 to disable.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile GAFF energy (CUDA; first steps pay compile cost).",
    )
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-global-pull", type=float, default=0.02)
    p.add_argument("--rescale-max-iterations", type=int, default=128)
    p.add_argument("--rescale-log-error-clip", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=0.001)
    p.add_argument("--rescale-bw-max", type=float, default=1.0)
    p.add_argument(
        "--rescale-fixed-row-sum-min",
        type=float,
        default=2.0,
        help="Absolute clip on adaptive band for unnormalized row_sum metric",
    )
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=0,
        help="Unbiased Langevin steps before FP (kernel force off). 0 disables.",
    )
    p.add_argument(
        "--heal-lambdas",
        type=str,
        default=None,
        help="Comma-separated λ heal schedule before production, e.g. 0.1,0.05",
    )
    p.add_argument(
        "--heal-steps",
        type=int,
        default=500,
        help="Steps per heal λ (no ESS reweight between heal values)",
    )
    p.add_argument(
        "--no-stage-coords",
        action="store_true",
        help="Do not write per-stage *_coords.npy (keep dihedral+bandwidth).",
    )
    p.add_argument(
        "--no-stage-plots",
        action="store_true",
        help="Do not write per-stage phi_psi PNGs.",
    )
    p.add_argument("--production-steps", type=int, default=25000)
    p.add_argument(
        "--archive-coordinates", action=argparse.BooleanOptionalAction, default=True
    )
    return p


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    temp = args.temperature
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    heal: list[float] = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    device = torch.device(args.device)
    dtype = torch.float64
    prefix = Path(args.sponge_prefix)
    dt_ps = float(args.timestep)
    gamma_ps = float(args.langevin_friction)
    dt_akma = timestep_picosecond_to_akma(dt_ps)
    gamma_akma = friction_per_picosecond_to_akma(gamma_ps, dt_ps, dt_akma)
    print(f"device={device}  beta0={args.beta0:.6g}  sponge={prefix}")
    print(
        f"[baoab time] dt={dt_ps} ps → dt_akma={dt_akma:.12g}; γ={gamma_ps} /ps → γ_akma={gamma_akma:.12g}; γ·dt={gamma_ps * dt_ps:.6g}"
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
    if torsion_indices.size == 0:
        raise ValueError("no torsion indices valid for this GAFF system")
    if args.init_npy:
        path = Path(os.path.abspath(os.path.expanduser(args.init_npy)))
        loaded = np.load(path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {path}: {loaded.shape}")
        if loaded.shape[1] != potential.num_atoms:
            raise ValueError(
                f"init npy n_atoms={loaded.shape[1]} != GAFF n_atoms={potential.num_atoms}"
            )
        if loaded.shape[0] != args.replicas:
            print(
                f"[init] --replicas={args.replicas} overridden by loaded B={loaded.shape[0]}"
            )
            args.replicas = int(loaded.shape[0])
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    else:
        base = load_sponge_coordinates(prefix, num_atoms=potential.num_atoms)
        coordinate = (
            torch.as_tensor(base, dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(args.replicas, 1, 1)
        )
        if args.noise_scale > 0:
            print(
                f"[warn] applying coordinate noise_scale={args.noise_scale} Å (bonded systems should use 0)"
            )
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
    print(f"[init] bw_init={bandwidth0:.6g}  replicas={args.replicas}")
    rescale = PerReplicaRescaleConfig(
        enabled=bool(args.rescale_per_replica),
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=args.rescale_adaptive_ref_quantile,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
        global_pull=args.rescale_global_pull,
        max_rescale_iterations=args.rescale_max_iterations,
        log_error_clip=args.rescale_log_error_clip,
        fixed_row_sum_min=float(args.rescale_fixed_row_sum_min),
        fixed_row_sum_max=float(args.rescale_fixed_row_sum_max),
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
    )
    if args.bandwidth_method == "rescale":
        print(
            f"[init] rescale metric=unnormalized_K_row_sum  mode=adaptive_quantile  band=[{args.rescale_adaptive_band_low},{args.rescale_adaptive_band_high}]*median clipped to [{rescale.fixed_row_sum_min},{rescale.fixed_row_sum_max}]  bw_clip=[{rescale.bandwidth_min},{rescale.bandwidth_max}]  global_pull={rescale.global_pull}"
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
        production_lambda=args.production_lambda,
        mass=mass,
        boltzmann_constant=float(args.kB),
        update_pace=args.update_pace,
        collective_variable_grad_clip=cv_clip,
        device=device,
        dtype=dtype,
    )
    print(f"[init] cv_grad_clip_norm={cv_clip}")
    out = args.output or EXPERIMENT_DIR / "results" / f"dialanine_{args.replicas}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "phi_psi"
    plots_dir.mkdir(parents=True, exist_ok=True)
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
            dump_every=args.stage_steps,
            heal_lambdas=[] if restart.resumed else args.heal_lambda_list,
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
