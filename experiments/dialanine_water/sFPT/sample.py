"""Solvated dialanine (GAFF + TIP3P) + φ/ψ torsion CV + torch-sim sFPT.

Full classical MM via ``MechanicalEmbeddingMLMM(enable_mace=False)``.
Box: ``../mm_box`` (run ``../prepare_box.py`` first). Metal units (eV / Å / ps).
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

ROOT = Path(__file__).resolve().parents[3]
WATER_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WATER_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from io_utils import load_xyz
from fp_transformer_torch import (
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    TorchSimFokkerPlanckIntegrator,
    Torsion,
)
from fp_transformer_torch.distance import torus_pairwise_distance
from fp_transformer_torch.potential.mace import (
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices

DEFAULT_BOX = WATER_DIR / "mm_box"
DEFAULT_CV = DEFAULT_BOX / "backbone_cv.txt"
KB_EV_PER_K = 8.617330337217213e-05
N_SOLUTE = 22


def load_torsion_indices(path: Path) -> np.ndarray:
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


def initial_bandwidth(cv: torch.Tensor) -> float:
    distance = torus_pairwise_distance(cv)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    return median if np.isfinite(median) and median > 0 else 0.5


def _attach_charges(state, charges: torch.Tensor, n_replicas: int):
    import torch_sim as ts

    extras = dict(state.system_extras)
    atom_extras = dict(state.atom_extras)
    atom_extras["partial_charges"] = charges.repeat(n_replicas)
    return ts.SimState(
        positions=state.positions,
        masses=state.masses,
        cell=state.row_vector_cell,
        pbc=True,
        atomic_numbers=state.atomic_numbers,
        system_idx=state.system_idx,
        _system_extras=extras,
        _atom_extras=atom_extras,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument(
        "--production-lambda", type=float, default=0.1, dest="production_lambda"
    )
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--pre-md-steps", type=int, default=100)
    p.add_argument("--heal-lambdas", type=str, default="0.1,0.05")
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument(
        "--electrostatics",
        type=str,
        default=None,
        choices=["pme", "dsf", "torchpme"],
        help="MM electrostatics. Default: pme on 1 GPU, torchpme when --n-gpus>1.",
    )
    p.add_argument(
        "--torchpme-method",
        type=str,
        default="p3m",
        choices=["p3m", "pme"],
        help="Calculator inside electrostatics=torchpme (default P3M).",
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated MM GPUs, e.g. 0,1,2,3 (overrides --n-gpus)",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Shard classical MM across this many GPUs (replica-parallel)",
    )
    p.add_argument(
        "--integrator",
        type=str,
        default="auto",
        choices=["auto", "torchsim", "sharded"],
        help="auto: torch-sim on 1 GPU, sharded BAOAB when --n-gpus>1. sharded: always in-repo BAOAB (use this to debug on 1 GPU).",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--no-stage-plots",
        action="store_true",
        help="Skip per-stage scatter plots; dynamics and saved arrays are unchanged.",
    )
    p.add_argument(
        "--bandwidth-method",
        choices=["rescale", "local_mass", "knn", "log_mass"],
        default="rescale",
    )
    p.add_argument(
        "--freeze-kernel-centers",
        action="store_true",
        help="Opt into the fixed-center KDE score; default uses the original joint replica gradient.",
    )
    p.add_argument("--bandwidth-target-occupancy", type=float, default=3.0)
    p.add_argument("--bandwidth-solver-iterations", type=int, default=24)
    p.add_argument("--bandwidth-neighbor-count", type=int, default=3)
    p.add_argument("--bandwidth-neighbor-scale", type=float, default=2 ** (-0.5))
    p.add_argument("--bandwidth-relaxation", type=float, default=1.0)
    p.add_argument("--rescale-global-pull", type=float, default=0.02)
    p.add_argument("--rescale-max-iterations", type=int, default=128)
    p.add_argument("--rescale-log-error-clip", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=0.001)
    p.add_argument("--rescale-bw-max", type=float, default=8.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=4.0)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--production-steps", type=int, default=25000)
    p.add_argument(
        "--archive-coordinates", action=argparse.BooleanOptionalAction, default=True
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    from torch_sim.models.interface import SumModel
    import torch_sim as ts

    primary = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.devices:
        mm_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mm_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mm_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.electrostatics is None:
        args.electrostatics = "torchpme" if len(mm_devices) > 1 else "pme"
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    box = Path(args.box_dir)
    xyz_path = box / "box.xyz"
    topo_path = box / "topology.npz"
    cell_path = box / "cell.txt"
    if not topo_path.is_file():
        raise FileNotFoundError(
            f"missing {topo_path}; run dialanine_water/prepare_box.py"
        )
    symbols, xyz_coords = load_xyz(xyz_path)
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell = torch.as_tensor(
        np.loadtxt(cell_path).reshape(3, 3), dtype=dtype, device=device
    )
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz N={xyz_coords.shape[0]} != topology N={num_atoms}")
    tors = load_torsion_indices(Path(args.cv_file))[:2]
    cv_fn = Torsion(tors)
    solute_symbols = symbols[:N_SOLUTE]
    mm_kw = dict(
        mace_model=None,
        enable_mace=False,
        enable_ml_coulomb_correction=False,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=0.0001,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        torchpme_method=str(args.torchpme_method),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )
    if len(mm_devices) > 1:
        mm_kw["mm_devices"] = mm_devices
    mm = MechanicalEmbeddingMLMM(topo, **mm_kw)
    n_mm = len(mm._mm_workers) if mm._mm_workers else 1
    print(
        f"Dialanine-water sFPT  device={device}  N={num_atoms}  R={args.replicas}  λ0={args.production_lambda}  elec={args.electrostatics}"
        + (f"/{args.torchpme_method}" if args.electrostatics == "torchpme" else "")
        + f"  cutoff={args.pme_cutoff}  box={float(cell[0, 0]):.1f} Å  mm_gpus={n_mm}  devices={[str(d) for d in mm_devices]}",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}", flush=True)
    out = Path(
        args.output
        or EXPERIMENT_DIR
        / "results"
        / f"dialanine_water_sfpt_r{args.replicas}_s{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "phi_psi"
    plots.mkdir(parents=True, exist_ok=True)
    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        coordinate = torch.as_tensor(loaded[idx], dtype=dtype, device=device)
        print(f"[init] from {args.init_npy}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        if args.pre_md_steps > 0:
            print(f"[pre-md] {args.pre_md_steps} unbiased NVT steps", flush=True)
            state_md = batched_coordinates_to_sim_state(
                coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
            )
            state_md = _attach_charges(
                state_md, topo.charges.to(device=device, dtype=dtype), args.replicas
            )
            state_md.rng = int(args.seed)
            state_md = ts.integrate(
                state_md,
                mm,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=int(args.pre_md_steps),
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                pbar=False,
            )
            coordinate = sim_state_to_batched_coordinates(state_md).to(dtype=dtype)
    cv0 = cv_fn(coordinate)
    bw0 = initial_bandwidth(cv0)
    print(f"[init] bw={bw0:.4g}  R={args.replicas}", flush=True)
    beta0 = 1.0 / (float(args.kB) * float(args.temperature))
    rescale = PerReplicaRescaleConfig(
        enabled=True,
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=0.5,
        adaptive_band_low=args.rescale_adaptive_band_low,
        adaptive_band_high=args.rescale_adaptive_band_high,
        bandwidth_min=args.rescale_bw_min,
        bandwidth_max=args.rescale_bw_max,
        global_pull=args.rescale_global_pull,
        max_rescale_iterations=args.rescale_max_iterations,
        log_error_clip=args.rescale_log_error_clip,
        target_occupancy=args.bandwidth_target_occupancy,
        solver_iterations=args.bandwidth_solver_iterations,
        neighbor_count=args.bandwidth_neighbor_count,
        neighbor_scale=args.bandwidth_neighbor_scale,
        relaxation=args.bandwidth_relaxation,
        fixed_row_sum_min=float(args.rescale_fixed_row_sum_min),
        fixed_row_sum_max=float(args.rescale_fixed_row_sum_max),
        use_normalized_kernel=False,
        normalize_row_sum_by_replicas=False,
    )
    print(
        f"[init] bandwidth_method={args.bandwidth_method} occupancy={args.bandwidth_target_occupancy} neighbors={args.bandwidth_neighbor_count} relaxation={args.bandwidth_relaxation}",
        flush=True,
    )
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bw0,
        distance_function=torus_pairwise_distance,
        freeze_centers=args.freeze_kernel_centers,
        update_bandwidth_method=args.bandwidth_method,
        rescale_per_replica=True,
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )
    print(f"[init] freeze_kernel_centers={kernel.freeze_centers}", flush=True)
    driver_kw = dict(
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        beta0=beta0,
        production_lambda=float(args.production_lambda),
        update_pace=int(args.update_pace),
        device=device,
        dtype=dtype,
    )
    fpt_bias = FPTBias(
        cv_fn,
        kernel,
        kernel_grad_beta=float(args.production_lambda) * beta0,
        device=device,
        dtype=dtype,
        use_kernel_gradient=True,
        update_pace=args.update_pace,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mm, fpt_bias)
    driver = TorchSimFokkerPlanckIntegrator(model, fpt_bias, cv_fn, kernel, **driver_kw)
    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
    )
    state0 = _attach_charges(
        state0, topo.charges.to(device=device, dtype=dtype), args.replicas
    )
    state0.rng = int(args.seed)
    print(f"[rng] TorchSim state seed={state0.rng.initial_seed()}", flush=True)
    from fp_transformer_torch.output import TrajectoryArchive

    recorder = TrajectoryArchive(
        out / "trajectory.npz",
        metadata=vars(args),
        include_coordinates=args.archive_coordinates,
    )
    restart = Restart(out, recorder, kernel)
    state_resume = restart.load(state0)

    def on_frame(phase, step, lam, state):
        if phase == "production":
            if restart.resumed and step == 0:
                return
            step += restart.offset
        coords = sim_state_to_batched_coordinates(state)
        recorder.append(
            phase,
            step,
            float(args.timestep),
            lam,
            coords,
            cv_fn(coords),
            kernel.bandwidth,
            metrics=driver._last_step_metrics,
        )
        if phase == "production":
            restart.save(step, state)

    final = None
    try:
        final = driver.run(
            state_resume,
            args.production_steps - restart.offset,
            dump_every=args.stage_steps,
            heal_lambdas=[] if restart.resumed else heal,
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
