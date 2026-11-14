"""Solvated Claisen (allyl phenyl ether) + ML/MM + 2D WT-MetaD.

ML region = solute (MACE); solvent = flexible TIP3P with PME electrostatics
and heterogeneous LJ (mechanical embedding).

CVs (Å): d1=d(O7,C8), d2=d(C1,C10). Multi-walker shared bias.
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
MACE_EXP = EXPERIMENT_DIR.parent / "rearrangement_mace"
sys.path.insert(0, str(MACE_EXP))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from io_utils import load_xyz
from fp_transformer_torch import (
    MultiParticlePairDistance,
    QuadraticDistanceWall,
    SoftminDistanceUpperWall,
    TorchSimWelltemperMetadynamic,
    WellTemperedMetadynamicsBias,
    load_mace_torch_sim_model,
)
from fp_transformer_torch.potential.mace import (
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)

DEFAULT_BOX = EXPERIMENT_DIR / "mlmm_box"
DEFAULT_CV_FILE = EXPERIMENT_DIR / "cv_pairs.txt"
KB_EV_PER_K = 8.617330337217213e-05


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
        raise ValueError(f"need 2 pairs in {path}, got {len(rows)}")
    return rows


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--xyz", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--cell-file", type=str, default=None)
    p.add_argument("--mace-model", type=str, default="mh-0")
    p.add_argument("--mace-cache", type=str, default=None)
    p.add_argument(
        "--mace-compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for MACE (needs --mace-enable-cueq for MH)",
    )
    p.add_argument(
        "--mace-enable-cueq",
        action="store_true",
        help="convert MACE to cuEquivariance kernels (recommended with compile)",
    )
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--max-xyz-frames", type=int, default=None)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true", help="write full box XYZ")
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
    p.add_argument("--metad-sigma", type=str, default="0.2,0.2")
    p.add_argument("--metad-height", type=float, default=0.02)
    p.add_argument("--metad-bias-factor", type=float, default=30.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument(
        "--metad-max-hills",
        type=int,
        default=200000,
        help="hill buffer cap; <=0 means unlimited (grow on demand)",
    )
    p.add_argument(
        "--metad-grid-n",
        type=int,
        default=0,
        help="cubic B-spline grid bins per CV (0=exact sum of Gaussians)",
    )
    p.add_argument(
        "--metad-grid-min",
        type=str,
        default="0.0,0.0",
        help="non-periodic grid lower edge per CV (Å); used when grid_n>0",
    )
    p.add_argument(
        "--metad-grid-max",
        type=str,
        default="10.0,10.0",
        help="non-periodic grid upper edge per CV (Å); used when grid_n>0",
    )
    p.add_argument(
        "--resume-hills",
        type=str,
        default=None,
        help="load prior hills.npz and continue depositing",
    )
    p.add_argument(
        "--live-dir",
        type=str,
        default="live",
        help="subdir under --output for fes/hills/plots (e.g. live_rerun)",
    )
    p.add_argument("--fes-d1-min", type=float, default=1.0)
    p.add_argument("--fes-d1-max", type=float, default=4.0)
    p.add_argument("--fes-d2-min", type=float, default=1.0)
    p.add_argument("--fes-d2-max", type=float, default=6.0)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated MM GPUs, e.g. 0,1 (overrides --n-gpus)",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Shard classical MM across this many GPUs (replica-parallel)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-pace", type=int, default=100)
    p.add_argument("--no-progress", action="store_true")
    return p


def _load_last_xyz_frames(path: Path, n_frames: int) -> np.ndarray:
    """Return ``coords[n_frames,N,3]`` from the end of a multi-frame XYZ."""
    text = Path(path).read_text().splitlines()
    frames: list[np.ndarray] = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        try:
            nat = int(line)
        except ValueError:
            i += 1
            continue
        i += 2
        xyz = np.zeros((nat, 3), dtype=np.float64)
        for a in range(nat):
            parts = text[i + a].split()
            xyz[a] = [float(parts[1]), float(parts[2]), float(parts[3])]
        frames.append(xyz)
        i += nat
    if not frames:
        raise ValueError(f"no frames in {path}")
    take = frames[-int(n_frames) :]
    while len(take) < n_frames:
        take.append(take[-1].copy())
    return np.stack(take[-n_frames:], axis=0)


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


def main() -> None:
    args = build_parser().parse_args()
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.traj_full:
        args.traj_solute_only = False
    box = Path(args.box_dir)
    xyz_path = Path(args.xyz or box / "box.xyz")
    topo_path = Path(args.topology or box / "topology.npz")
    cell_path = Path(args.cell_file or box / "cell.txt")
    if not topo_path.is_file():
        raise FileNotFoundError(
            f"missing {topo_path}; run rearrangement_mace/prepare_mlmm_box.py first"
        )
    from torch_sim.models.interface import SumModel
    from fp_transformer_torch.potential.sharded_mace import parse_cuda_devices

    primary = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.devices:
        mm_devices = parse_cuda_devices(args.devices, fallback=primary)
    else:
        mm_devices = parse_cuda_devices(int(args.n_gpus), fallback=primary)
    device = mm_devices[0]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    symbols, xyz_coords = load_xyz(xyz_path)
    topo = MLMMTopology.load_npz(topo_path).to(device, dtype)
    cell_np = np.loadtxt(cell_path, dtype=np.float64).reshape(3, 3)
    cell = torch.as_tensor(cell_np, dtype=dtype, device=device)
    num_atoms = topo.n_atoms
    if xyz_coords.shape[0] != num_atoms:
        raise ValueError(f"xyz N={xyz_coords.shape[0]} != topology N={num_atoms}")
    n_solute = int(topo.ml_mask.sum().item())
    solute_symbols = symbols[:n_solute]
    pairs = load_pair_indices(Path(args.cv_file))
    print(
        f"Claisen-water MetaD  device={device}  mace={args.mace_model}  N={num_atoms} ML={n_solute}  R={args.replicas}  steps={args.steps}  T={args.temperature} K  mm_gpus={len(mm_devices)}  devices={[str(d) for d in mm_devices]}",
        flush=True,
    )
    print(f"[CV] d1/d2 pairs(0-based)={pairs}", flush=True)
    if args.init_npy and Path(args.init_npy).is_file():
        loaded = np.load(Path(args.init_npy))
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(f"[init] from npy {args.init_npy}", flush=True)
    elif args.resume_hills:
        hills_p = Path(args.resume_hills)
        cand_traj = hills_p.parent.parent / "trajectory.xyz"
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        if cand_traj.is_file():
            frames = _load_last_xyz_frames(cand_traj, int(args.replicas))
            if frames.shape[-2] == num_atoms:
                loaded = frames
                print(
                    f"[init] resume: full-box last frames from {cand_traj}", flush=True
                )
            else:
                print(
                    f"[init] resume: traj is solute-only (N={frames.shape[-2]}); using box xyz × {args.replicas} + loaded hills (avoid grafting onto stale solvent)",
                    flush=True,
                )
        else:
            print(
                f"[init] resume hills but no traj; box xyz × {args.replicas}",
                flush=True,
            )
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        print(f"[init] box xyz × {args.replicas}", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    mace_kw: dict = dict(
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
        cache_dir=args.mace_cache,
        enable_cueq=bool(args.mace_enable_cueq),
    )
    if args.mace_compile_mode:
        mace_kw["compile_mode"] = str(args.mace_compile_mode)
    mace_model = load_mace_torch_sim_model(args.mace_model, **mace_kw)
    print(
        f"[init] MACE={args.mace_model}  cueq={bool(args.mace_enable_cueq)}  compile_mode={args.mace_compile_mode or 'off'}",
        flush=True,
    )
    mlmm_kw = dict(
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
    )
    if len(mm_devices) > 1:
        mlmm_kw["mm_devices"] = mm_devices
    mlmm = MechanicalEmbeddingMLMM(topo, **mlmm_kw)
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
    print(
        f"[restrain] softmin  a={args.softmin_a}/Å  d_wall={args.softmin_d_wall} Å  k={args.softmin_k} eV/Å²",
        flush=True,
    )
    sigma = _parse_floats(args.metad_sigma)
    grid_n = int(args.metad_grid_n)
    grid_min = _parse_floats(args.metad_grid_min)
    grid_max = _parse_floats(args.metad_grid_max)
    bias_kw: dict = dict(
        n_cv=2,
        sigma=sigma,
        height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor),
        temperature=float(args.temperature),
        kB=float(args.kB),
        max_hills=int(args.metad_max_hills),
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    if grid_n > 0:
        bias_kw["grid_n"] = grid_n
        bias_kw["grid_min"] = grid_min
        bias_kw["grid_max"] = grid_max
    bias = WellTemperedMetadynamicsBias(cv_fn, **bias_kw)
    if args.resume_hills:
        n_loaded = bias.load_hills(args.resume_hills)
        print(
            f"[resume] loaded {n_loaded} hills from {args.resume_hills}  max_hills={('unlimited' if int(args.metad_max_hills) <= 0 else args.metad_max_hills)}",
            flush=True,
        )
    model = SumModel(mlmm, bias, wall, softmin_wall)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    if grid_n > 0:
        dx = bias._grid_dx.detach().cpu().numpy()
        grid_msg = f"grid_n={grid_n}  bounds=[{grid_min},{grid_max}]  dx={dx} Å"
    else:
        grid_msg = "grid_n=off(exact)"
    print(
        f"[init] WT-MetaD σ={sigma} Å  w0={args.metad_height} γ={args.metad_bias_factor}  PME cutoff={args.pme_cutoff} Å  {grid_msg}  n_hills0={bias.n_hills}",
        flush=True,
    )
    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=True,
        charge=float(args.charge),
        spin=float(args.spin),
    )
    state0 = _attach_charges(state0, topo.charges, int(args.replicas))
    out = Path(
        args.output
        or EXPERIMENT_DIR
        / "results"
        / f"claisen_water_metad_{args.mace_model}_r{args.replicas}"
    )
    out.mkdir(parents=True, exist_ok=True)
    live = out / str(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)
    traj_name = "trajectory_rerun.xyz" if args.resume_hills else "trajectory.xyz"
    traj = out / traj_name
    if traj.is_file():
        traj.unlink()

    def _traj_coords(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if args.traj_solute_only:
            return (coords[..., :n_solute, :], solute_symbols)
        return (coords, symbols)

    cv0 = cv_fn(coordinate).detach().cpu().numpy()
    if cv0.ndim == 3:
        cv0 = cv0.reshape(cv0.shape[0], -1)
    from fp_transformer_torch.output import TrajectoryArchive

    recorder = TrajectoryArchive(out / "trajectory.npz", metadata=vars(args))
    restart = Restart(out, recorder)
    state0 = restart.load(state0, bias=bias)

    def on_dump(done, state, metrics):
        if restart.resumed and done == 0:
            return
        done += restart.offset
        coords = sim_state_to_batched_coordinates(state)
        recorder.append(
            "production",
            done,
            float(args.timestep),
            float("nan"),
            coords,
            cv_fn(coords),
            metrics=metrics,
            hills_count=bias.n_hills,
        )
        print(f"[dump] {done} steps; {bias.n_hills} hills", flush=True)
        restart.save(done, state, bias=bias)

    if not restart.resumed:
        state0.rng = int(args.seed)
    on_dump(0, state0, {})
    final = None
    try:
        final = driver.run(
            state0,
            int(args.steps) - restart.offset,
            verbose=args.verbose,
            verbose_pace=args.verbose_pace,
            show_progress=False,
            on_dump=on_dump,
            dump_every=int(args.dump_every),
        )
    finally:
        recorder.save(
            complete=final is not None,
            extra={
                "hills_centers": bias._centers[: bias.n_hills].detach().cpu().numpy(),
                "hills_heights": bias._heights[: bias.n_hills].detach().cpu().numpy(),
                **(
                    {
                        "bias_grid": bias._grid.detach().cpu().numpy(),
                        "bias_coefficients": bias._coeffs.detach().cpu().numpy(),
                        "bias_grid_origin": bias._grid_origin.detach().cpu().numpy(),
                        "bias_grid_period": bias._grid_period.detach().cpu().numpy(),
                    }
                    if bias.use_grid
                    else {}
                ),
                "hills_sigma": bias.sigma.detach().cpu().numpy(),
                "bias_factor": np.asarray(bias.bias_factor),
            },
        )
        if final is not None:
            restart.finish()


if __name__ == "__main__":
    main()
