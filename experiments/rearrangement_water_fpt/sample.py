"""Solvated Claisen + mechanical embedding ML/MM + FPT.

Same box / topology as ``rearrangement_water_metadynamics`` (MACE solute +
TIP3P/PME water). CVs: d1=d(O7,C8), d2=d(C1,C10). Default ``production lambda=0.1``.
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
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from io_utils import load_xyz
from fp_transformer_torch import (
    FPTBias,
    GaussianKernel,
    PerReplicaRescaleConfig,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    MultiParticlePairDistance,
    TorchSimFokkerPlanckIntegrator,
    load_sharded_mace_torch_sim_model,
)
from fp_transformer_torch.distance import pairwise_distance
from fp_transformer_torch.potential.mace import (
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)
from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices

DEFAULT_BOX = EXPERIMENT_DIR / "mlmm_box"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
KB_EV_PER_K = 8.617330337217213e-05


def initial_bandwidth_from_collective_variable(cv: torch.Tensor) -> float:
    distance = pairwise_distance(cv)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    values = distance[upper]
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.5
    median = float(torch.median(positive).item())
    return median if np.isfinite(median) and median > 0 else 0.5


def load_pair_indices(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ints = []
        for part in line.replace(",", " ").split():
            try:
                ints.append(int(part) - 1)
            except ValueError:
                continue
        if len(ints) >= 2:
            rows.append((ints[0], ints[1]))
    if len(rows) != 2:
        raise ValueError(f"need exactly 2 pairs in {path}, got {len(rows)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--xyz", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--cell-file", type=str, default=None)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-enable-cueq", action="store_true")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=256)
    p.add_argument("--stage-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--beta0", type=float, default=None)
    p.add_argument(
        "--production-lambda", type=float, default=0.1, dest="production_lambda"
    )
    p.add_argument("--update-pace", type=int, default=20)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--pre-md-steps", type=int, default=100)
    p.add_argument("--pre-md-dump-every", type=int, default=50)
    p.add_argument("--max-xyz-frames", type=int, default=32)
    p.add_argument("--heal-lambdas", type=str, default="0.1,0.05,0.02")
    p.add_argument("--heal-steps", type=int, default=100)
    p.add_argument("--pme-cutoff", type=float, default=9.0)
    p.add_argument(
        "--pme-mesh-spacing",
        type=float,
        default=1.5,
        help="PME reciprocal mesh spacing (Å); larger = faster, less accurate",
    )
    p.add_argument(
        "--electrostatics",
        type=str,
        default="pme",
        choices=["pme", "dsf"],
        help="MM long-range electrostatics: pme (accurate) or dsf (faster)",
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2, help="DSF damping α (1/Å)")
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--wall-d-max", type=float, default=8.0)
    p.add_argument("--wall-k", type=float, default=10.0)
    p.add_argument("--softmin-a", type=float, default=10.0)
    p.add_argument("--softmin-d-wall", type=float, default=2.7)
    p.add_argument("--softmin-k", type=float, default=10.0)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=20)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cv-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--rescale-per-replica", action="store_true", default=True)
    p.add_argument("--no-rescale-per-replica", action="store_true")
    p.add_argument("--rescale-adaptive-ref-quantile", type=float, default=0.5)
    p.add_argument("--rescale-adaptive-band-low", type=float, default=0.8)
    p.add_argument("--rescale-adaptive-band-high", type=float, default=1.2)
    p.add_argument("--rescale-bw-min", type=float, default=0.001)
    p.add_argument("--rescale-bw-max", type=float, default=4.0)
    p.add_argument("--rescale-fixed-row-sum-min", type=float, default=2.0)
    p.add_argument("--rescale-fixed-row-sum-max", type=float, default=5.0)
    p.add_argument("--production-steps", type=int, default=25000)
    p.add_argument(
        "--archive-coordinates", action=argparse.BooleanOptionalAction, default=True
    )
    return p


def _resolve(args):
    if args.traj_full:
        args.traj_solute_only = False
    if args.no_rescale_per_replica:
        args.rescale_per_replica = False
    temp = args.temperature
    if args.beta0 is None:
        args.beta0 = 1.0 / (float(args.kB) * float(temp))
    heal = []
    if args.heal_lambdas:
        for part in str(args.heal_lambdas).replace(";", ",").split(","):
            part = part.strip()
            if part:
                heal.append(float(part))
    args.heal_lambda_list = heal
    box = Path(args.box_dir)
    args.xyz = Path(args.xyz or box / "box.xyz")
    args.topology = Path(args.topology or box / "topology.npz")
    args.cell_file = Path(args.cell_file or box / "cell.txt")
    return args


def main() -> None:
    args = _resolve(build_parser().parse_args())
    primary = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.devices:
        mace_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mace_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mace_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    import torch_sim as ts
    from torch_sim.models.interface import SumModel

    if not Path(args.topology).is_file():
        raise FileNotFoundError(
            f"missing topology {args.topology}; run prepare_mlmm_box.py first"
        )
    symbols, xyz_coords = load_xyz(Path(args.xyz))
    topo = MLMMTopology.load_npz(args.topology).to(device, dtype)
    cell_np = np.loadtxt(args.cell_file, dtype=np.float64).reshape(3, 3)
    cell = torch.as_tensor(cell_np, dtype=dtype, device=device)
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz atoms {xyz_coords.shape[0]} != topology N={num_atoms}")
    pairs = load_pair_indices(Path(args.cv_file))
    n_solute = int(topo.ml_mask.sum().item())
    solute_symbols = symbols[:n_solute]

    def _art_coords_symbols(coords: np.ndarray):
        if args.traj_solute_only:
            return (coords[..., :n_solute, :], solute_symbols)
        return (coords, symbols)

    mace_model = load_sharded_mace_torch_sim_model(
        args.mace_model,
        devices=mace_devices,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
        enable_cueq=bool(args.mace_enable_cueq),
    )
    mlmm = MechanicalEmbeddingMLMM(
        topo,
        mace_model=mace_model,
        cutoff=float(args.pme_cutoff),
        electrostatics=str(args.electrostatics),
        dsf_alpha=float(args.dsf_alpha),
        pme_accuracy=0.0001,
        pme_mesh_spacing=float(args.pme_mesh_spacing),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        mm_devices=None,
    )
    print(
        f"Claisen-water FPT  device={device}  mace={args.mace_model}  N={num_atoms}  ML={n_solute}  R={args.replicas}  production lambda={args.production_lambda}  elec={args.electrostatics}  mace_gpus={len(mace_devices)}  mm_gpus={len(mlmm._mm_workers) or 1}",
        flush=True,
    )
    cv_fn = MultiParticlePairDistance(pairs)
    wall = QuadraticDistanceWall(
        pairs,
        d_max=float(args.wall_d_max),
        k=float(args.wall_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    softmin_wall = SoftminDistanceUpperWall(
        pairs[0],
        pairs[1],
        a=float(args.softmin_a),
        d_wall=float(args.softmin_d_wall),
        k=float(args.softmin_k),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    out = Path(
        args.output
        or EXPERIMENT_DIR
        / "results"
        / f"claisen_water_fpt_{args.mace_model}_{args.replicas}"
    )
    out.mkdir(parents=True, exist_ok=True)
    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    if args.init_npy:
        loaded = np.load(Path(args.init_npy))[: args.replicas]
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
        print(f"[pre-md] steps={args.pre_md_steps}", flush=True)
        md_model = SumModel(mlmm, wall, softmin_wall)
        state_md = batched_coordinates_to_sim_state(
            coordinate,
            atomic_numbers=z_t,
            masses=mass_t,
            cell=cell,
            pbc=True,
            charge=float(args.charge),
            spin=float(args.spin),
        )
        extras = dict(state_md.system_extras)
        atom_extras = dict(state_md.atom_extras)
        atom_extras["partial_charges"] = topo.charges.repeat(args.replicas)
        state_md = ts.SimState(
            positions=state_md.positions,
            masses=state_md.masses,
            cell=state_md.row_vector_cell,
            pbc=True,
            atomic_numbers=state_md.atomic_numbers,
            system_idx=state_md.system_idx,
            _system_extras=extras,
            _atom_extras=atom_extras,
        )
        state_md.rng = int(args.seed)
        done = 0
        dump_every = max(1, int(args.pre_md_dump_every))
        md_steps = max(0, int(args.pre_md_steps))
        dumps = []
        while done < md_steps:
            n_this = min(dump_every, md_steps - done)
            state_md = ts.integrate(
                state_md,
                md_model,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=n_this,
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                autobatcher=False,
                pbar=False,
            )
            done += n_this
            dumps.append(
                sim_state_to_batched_coordinates(state_md).detach().cpu().numpy()
            )
            print(f"[pre-md] step={done}", flush=True)
        if dumps:
            coordinate = torch.as_tensor(dumps[-1], dtype=dtype, device=device)
    collective0 = cv_fn(coordinate)
    bw0 = initial_bandwidth_from_collective_variable(collective0)
    cv0 = collective0.detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    print(
        f"[init] bw={bw0:.4g}  d1 med={np.median(cv0[:, 0]):.3f}  d2 med={np.median(cv0[:, 1]):.3f}",
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
    kernel = GaussianKernel(
        args.replicas,
        bandwidth=bw0,
        distance_function=pairwise_distance,
        rescale_per_replica=bool(args.rescale_per_replica),
        rescale_per_replica_config=rescale,
        device=device,
        dtype=dtype,
    )
    cv_clip = None if args.cv_grad_clip_norm <= 0 else float(args.cv_grad_clip_norm)
    fpt_bias = FPTBias(
        cv_fn,
        kernel,
        kernel_grad_beta=float(args.production_lambda) * float(args.beta0),
        device=device,
        dtype=dtype,
        collective_variable_grad_clip=cv_clip,
        update_pace=args.update_pace,
        n_atoms_per_system=num_atoms,
    )
    model = SumModel(mlmm, fpt_bias, wall, softmin_wall)
    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=True,
        charge=float(args.charge),
        spin=float(args.spin),
    )
    extras = dict(state0.system_extras)
    atom_extras = dict(state0.atom_extras)
    atom_extras["partial_charges"] = topo.charges.repeat(int(coordinate.shape[0]))
    state0 = ts.SimState(
        positions=state0.positions,
        masses=state0.masses,
        cell=state0.row_vector_cell,
        pbc=True,
        atomic_numbers=state0.atomic_numbers,
        system_idx=state0.system_idx,
        _system_extras=extras,
        _atom_extras=atom_extras,
    )
    integrator = TorchSimFokkerPlanckIntegrator(
        model,
        fpt_bias,
        cv_fn,
        kernel,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        beta0=float(args.beta0),
        production_lambda=float(args.production_lambda),
        update_pace=int(args.update_pace),
        device=device,
        dtype=dtype,
    )
    state0.rng = int(args.seed)
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
            metrics=integrator._last_step_metrics,
        )
        if phase == "production":
            restart.save(step, state)

    final = None
    try:
        final = integrator.run(
            state_resume,
            args.production_steps - restart.offset,
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
