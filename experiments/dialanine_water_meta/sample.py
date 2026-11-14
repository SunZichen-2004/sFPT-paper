"""Solvated dialanine (GAFF + TIP3P) + φ/ψ WT-MetaD (multi-walker, shared bias).

Full classical MM via ``MechanicalEmbeddingMLMM(enable_mace=False)``.
Box: ``../dialanine_water/mm_box`` (run ``../dialanine_water/prepare_box.py``).
φ/ψ are periodic with period 2π. Metal units (eV / Å / ps).
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
WATER_DIR = EXPERIMENT_DIR.parent / "dialanine_water"
sys.path.insert(0, str(WATER_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from io_utils import load_xyz
from fp_transformer_torch import (
    TorchSimWelltemperMetadynamic,
    Torsion,
    WellTemperedMetadynamicsBias,
)
from fp_transformer_torch.potential.mace import (
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.mechanical_embedding_MLMM import (
    MechanicalEmbeddingMLMM,
    MLMMTopology,
)

DEFAULT_BOX = WATER_DIR / "mm_box"
DEFAULT_CV = DEFAULT_BOX / "backbone_cv.txt"
KB_EV_PER_K = 8.617330337217213e-05
TWO_PI = 2.0 * np.pi
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


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


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


def _load_last_xyz_frames(path: Path, n_frames: int) -> np.ndarray:
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box-dir", type=Path, default=DEFAULT_BOX)
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument(
        "--pre-md-steps",
        type=int,
        default=0,
        help="unbiased NVT Langevin steps (no MetaD hills) before WT-MetaD",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0)
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=None)
    p.add_argument("--traj-solute-only", action="store_true", default=True)
    p.add_argument("--traj-full", action="store_true")
    p.add_argument("--pme-cutoff", type=float, default=8.0)
    p.add_argument("--pme-mesh-spacing", type=float, default=1.5)
    p.add_argument(
        "--electrostatics", type=str, default="pme", choices=["pme", "dsf", "torchpme"]
    )
    p.add_argument("--dsf-alpha", type=float, default=0.2)
    p.add_argument(
        "--torchpme-method",
        type=str,
        default="p3m",
        choices=["p3m", "pme"],
        help="Calculator inside electrostatics=torchpme (default P3M).",
    )
    p.add_argument("--metad-sigma", type=str, default="0.35,0.35", help="rad")
    p.add_argument("--metad-height", type=float, default=0.02, help="eV")
    p.add_argument("--metad-bias-factor", type=float, default=15.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument(
        "--metad-max-hills", type=int, default=0, help="<=0 means unlimited hill buffer"
    )
    p.add_argument(
        "--metad-grid-n",
        type=int,
        default=0,
        help="periodic cubic B-spline grid bins per CV (0=exact sum of Gaussians)",
    )
    p.add_argument("--resume-hills", type=str, default=None)
    p.add_argument("--live-dir", type=str, default="live")
    p.add_argument("--fes-phi-min", type=float, default=-np.pi)
    p.add_argument("--fes-phi-max", type=float, default=np.pi)
    p.add_argument("--fes-psi-min", type=float, default=-np.pi)
    p.add_argument("--fes-psi-max", type=float, default=np.pi)
    p.add_argument("--dump-every", type=int, default=500)
    p.add_argument("--max-xyz-frames", type=int, default=None)
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


def main() -> None:
    args = build_parser().parse_args()
    if args.baoab_dt is not None:
        args.timestep = args.baoab_dt
    if args.baoab_langevin_gamma is not None:
        args.langevin_friction = args.baoab_langevin_gamma
    if args.traj_full:
        args.traj_solute_only = False
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
    print(
        f"Dialanine-water MetaD  device={device}  N={num_atoms}  R={args.replicas}  steps={args.steps}  elec={args.electrostatics}  cutoff={args.pme_cutoff}  box={float(cell[0, 0]):.1f} Å  mm_gpus={len(mm_devices)}  devices={[str(d) for d in mm_devices]}",
        flush=True,
    )
    print(f"[CV] φ/ψ torsions(0-based)={tors.tolist()}", flush=True)
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
                print(f"[init] resume: full-box frames from {cand_traj}", flush=True)
            else:
                print(
                    f"[init] resume: traj solute-only (N={frames.shape[-2]}); using box xyz × {args.replicas}",
                    flush=True,
                )
        else:
            print(f"[init] resume hills; box xyz × {args.replicas}", flush=True)
    else:
        loaded = np.repeat(xyz_coords[None, ...], args.replicas, axis=0)
        print(f"[init] box xyz × {args.replicas}", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
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
    sigma = _parse_floats(args.metad_sigma)
    grid_n = int(args.metad_grid_n)
    bias = WellTemperedMetadynamicsBias(
        cv_fn,
        n_cv=2,
        sigma=sigma,
        height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor),
        temperature=float(args.temperature),
        kB=float(args.kB),
        max_hills=int(args.metad_max_hills),
        period=TWO_PI,
        grid_n=grid_n if grid_n > 0 else None,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    if args.resume_hills:
        n_loaded = bias.load_hills(args.resume_hills)
        print(f"[resume] loaded {n_loaded} hills from {args.resume_hills}", flush=True)
    model = SumModel(mm, bias)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    grid_msg = f"grid_n={grid_n}" if grid_n > 0 else "grid_n=off(exact)"
    print(
        f"[init] WT-MetaD σ={sigma} rad  w0={args.metad_height} eV  γ={args.metad_bias_factor}  period=2π  {grid_msg}  n_hills0={bias.n_hills}",
        flush=True,
    )
    mass_t = topo.masses.to(device=device, dtype=dtype)
    z_t = topo.atomic_numbers.to(device=device)
    state0 = batched_coordinates_to_sim_state(
        coordinate, atomic_numbers=z_t, masses=mass_t, cell=cell, pbc=True
    )
    state0 = _attach_charges(
        state0, topo.charges.to(device=device, dtype=dtype), int(args.replicas)
    )
    out = Path(
        args.output
        or EXPERIMENT_DIR
        / "results"
        / f"dialanine_water_metad_r{args.replicas}_s{args.seed}"
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
            return (coords[..., :N_SOLUTE, :], solute_symbols)
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
    if not restart.resumed and args.pre_md_steps > 0:
        import torch_sim as ts

        remaining_pre = int(args.pre_md_steps)
        while remaining_pre:
            chunk_pre = min(500, remaining_pre)
            state0 = ts.integrate(
                state0,
                mm,
                integrator=ts.Integrator.nvt_langevin,
                n_steps=chunk_pre,
                temperature=float(args.temperature),
                timestep=float(args.timestep),
                gamma=float(args.langevin_friction),
                pbar=False,
            )
            remaining_pre -= chunk_pre
            print(
                f"[pre-md] {args.pre_md_steps - remaining_pre}/{args.pre_md_steps}",
                flush=True,
            )
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
