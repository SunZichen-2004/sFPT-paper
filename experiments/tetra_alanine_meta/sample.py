"""Tetra-alanine (vacuum GAFF) + 3φ WT-MetaD (multi-walker, shared bias).

CVs: φ1, φ2, φ3 (period 2π). Live dumps Boltzmann-project the 3D MetaD FES
onto the three 2D pairs (φi, φj).
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
SFP_DIR = EXPERIMENT_DIR.parent / "tetra_alanine_sFPT"
sys.path.insert(0, str(SFP_DIR))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from io_utils import load_torsion_indices
from fp_transformer_torch import (
    GaffTorchSimModel,
    TorchSimWelltemperMetadynamic,
    Torsion,
    WellTemperedMetadynamicsBias,
)
from fp_transformer_torch.potential.gaff_torch_sim import KCAL_PER_MOL_TO_EV
from fp_transformer_torch.potential.mace import (
    atomic_numbers_from_pdb_organic,
    batched_coordinates_to_sim_state,
    sim_state_to_batched_coordinates,
)
from fp_transformer_torch.potential.sponge_io import load_sponge_coordinates

DEFAULT_SPONGE = SFP_DIR / "tetra_alanine"
DEFAULT_CV_FILE = SFP_DIR / "backbone_phi.txt"
DEFAULT_PDB = SFP_DIR / "tetra_alanine.pdb"
DEFAULT_MASS = SFP_DIR / "tetra_alanine_mass.txt"
DEFAULT_INIT_NPY = SFP_DIR / "init_coords_1024.npy"
KB_EV_PER_K = 8.617330337217213e-05
EV_TO_KCAL = 1.0 / KCAL_PER_MOL_TO_EV
TWO_PI = 2.0 * np.pi
PHI_PAIRS = ((0, 1, 1, 2), (0, 2, 1, 3), (1, 2, 2, 3))
PI_TICKS = (-np.pi, 0.0, np.pi)
PI_LABELS = ("$-\\pi$", "$0$", "$\\pi$")
FE_LABEL = "free energy / kcal mol$^{-1}$"
FES_CMAP = "RdBu_r"


def select_phi_torsions(torsion_indices: np.ndarray) -> np.ndarray:
    n = int(torsion_indices.shape[0])
    if n == 3:
        return torsion_indices
    if n == 6:
        return torsion_indices[0:6:2]
    raise ValueError(f"expected 3 φ or 6 φψ torsions, got {n}")


def load_masses(path: Path, num_atoms: int) -> np.ndarray:
    mass_vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if int(mass_vals[0]) == num_atoms and mass_vals.size == num_atoms + 1:
        mass_vals = mass_vals[1:]
    if mass_vals.size < num_atoms:
        raise ValueError(f"mass file too short: {path}")
    return mass_vals[:num_atoms].astype(np.float64)


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def _symbols_from_z(z: np.ndarray) -> list[str]:
    table = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S"}
    return [table.get(int(x), f"Z{int(x)}") for x in z]


def write_xyz(
    path: Path,
    coords: np.ndarray,
    symbols: list[str],
    *,
    comment_prefix: str = "",
    append: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        frames = arr[None, ...]
    elif arr.ndim == 3:
        frames = arr
    else:
        raise ValueError(f"coords must be (N,3) or (B,N,3), got {arr.shape}")
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for i, frame in enumerate(frames):
            f.write(f"{frame.shape[0]}\n{comment_prefix} frame={i}\n")
            for a in range(frame.shape[0]):
                s = symbols[a] if a < len(symbols) else "X"
                x, y, zc = frame[a]
                f.write(f"{s} {x:.8f} {y:.8f} {zc:.8f}\n")


def relative_fes(f: np.ndarray) -> np.ndarray:
    return np.asarray(f, dtype=np.float64) - np.nanmin(f)


def fes_cmap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(FES_CMAP).copy()
    cmap.set_over("white")
    cmap.set_bad("white")
    return cmap


def metad_fes_3d(bias: WellTemperedMetadynamicsBias, ax: np.ndarray) -> np.ndarray:
    """F(φ1,φ2,φ3) in eV on an ``indexing='ij'`` grid, shape (n,n,n)."""
    n = int(ax.shape[0])
    scale = bias.bias_factor / (bias.bias_factor - 1.0)
    if (
        getattr(bias, "use_grid", False)
        and bias.n_cv == 3
        and (tuple((int(x) for x in bias._grid_shape)) == (n, n, n))
    ):
        axes = [a.detach().cpu().numpy() for a in bias.grid_axes()]
        if all((np.allclose(a, ax, atol=1e-12, rtol=0.0) for a in axes)):
            V = bias._grid.detach().cpu().numpy()
            return (-scale * V).astype(np.float64)
    g1, g2, g3 = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)
    with torch.no_grad():
        s_t = torch.as_tensor(pts, device=bias._device, dtype=bias._dtype)
        V = bias.bias_potential_cv(s_t).detach().cpu().numpy()
    return (-scale * V).reshape(n, n, n)


def project_pairs_kcal(
    F3_eV: np.ndarray, *, kT_eV: float
) -> dict[tuple[int, int], np.ndarray]:
    """Boltzmann-project 3D F onto (φi, φj); return contourf-ready (n, n) in kcal/mol."""
    beta = 1.0 / float(kT_eV)
    w = np.exp(-np.clip(np.asarray(F3_eV, dtype=np.float64), None, 80.0) * beta)
    out: dict[tuple[int, int], np.ndarray] = {}
    p01 = np.clip(w.sum(axis=2), 1e-300, None)
    p02 = np.clip(w.sum(axis=1), 1e-300, None)
    p12 = np.clip(w.sum(axis=0), 1e-300, None)
    kT_kcal = float(kT_eV) * EV_TO_KCAL
    out[0, 1] = relative_fes(-kT_kcal * np.log(p01)).T
    out[0, 2] = relative_fes(-kT_kcal * np.log(p02)).T
    out[1, 2] = relative_fes(-kT_kcal * np.log(p12)).T
    return out


def plot_fes_pairs(
    out_png: Path,
    *,
    ax: np.ndarray,
    f_pairs: dict[tuple[int, int], np.ndarray],
    vmax: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    levels = np.linspace(0.0, float(vmax), 25)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    cmap = fes_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.05), squeeze=True)
    im = None
    for a, (i, j, ii, jj) in zip(axes, PHI_PAIRS):
        z = np.minimum(relative_fes(f_pairs[i, j]), float(vmax))
        im = a.contourf(ax, ax, z, levels=levels, cmap=cmap, norm=norm, extend="max")
        a.contour(
            ax,
            ax,
            np.clip(z, 0.0, float(vmax)),
            levels=levels[::3],
            colors="0.45",
            linewidths=0.35,
            alpha=0.45,
        )
        a.set_xlim(-np.pi, np.pi)
        a.set_ylim(-np.pi, np.pi)
        a.set_xticks(PI_TICKS)
        a.set_xticklabels(PI_LABELS)
        a.set_yticks(PI_TICKS)
        a.set_yticklabels(PI_LABELS)
        a.set_xlabel(f"$\\phi_{ii}$ / rad")
        a.set_ylabel(f"$\\phi_{jj}$ / rad")
        a.set_title(f"$(\\phi_{ii},\\phi_{jj})$", fontsize=11)
        a.set_aspect("equal")
        a.tick_params(which="both", top=False, right=False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, extend="max")
    cbar.set_label(FE_LABEL)
    fig.suptitle(title, fontsize=11)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fes] wrote {out_png}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sponge-prefix", type=str, default=str(DEFAULT_SPONGE))
    p.add_argument("--cv-file", type=str, default=str(DEFAULT_CV_FILE))
    p.add_argument("--pdb", type=str, default=str(DEFAULT_PDB))
    p.add_argument("--mass-file", type=str, default=str(DEFAULT_MASS))
    p.add_argument("--replicas", type=int, default=64)
    p.add_argument("--total-time", type=float, default=1000.0, help="ps")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.001, help="ps")
    p.add_argument("--baoab-dt", type=float, default=None)
    p.add_argument("--langevin-friction", type=float, default=10.0, help="1/ps")
    p.add_argument("--baoab-langevin-gamma", type=float, default=None)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--kB", type=float, default=KB_EV_PER_K)
    p.add_argument("--init-npy", type=str, default=str(DEFAULT_INIT_NPY))
    p.add_argument("--cell-length", type=float, default=80.0)
    p.add_argument("--metad-sigma", type=str, default="0.35,0.35,0.35", help="rad")
    p.add_argument("--metad-height", type=float, default=0.02, help="eV")
    p.add_argument("--metad-bias-factor", type=float, default=15.0)
    p.add_argument("--metad-deposit-pace", type=int, default=50)
    p.add_argument("--metad-max-hills", type=int, default=0)
    p.add_argument(
        "--metad-grid-n",
        type=int,
        default=81,
        help="periodic cubic B-spline grid bins per φ (0=sum Gaussians)",
    )
    p.add_argument("--resume-hills", type=str, default=None)
    p.add_argument("--live-dir", type=str, default="live")
    p.add_argument(
        "--dump-every", type=int, default=2000, help="MD steps between live FES"
    )
    p.add_argument("--grid-n", type=int, default=81)
    p.add_argument("--fes-vmax", type=float, default=16.0)
    p.add_argument(
        "--dtype", type=str, default="float64", choices=["float32", "float64"]
    )
    p.add_argument("--device", type=str, default="cuda")
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
    dt_ps = float(args.timestep)
    n_steps = (
        int(args.steps)
        if args.steps is not None
        else int(round(float(args.total_time) / dt_ps))
    )
    from torch_sim.models.interface import SumModel

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    prefix = Path(args.sponge_prefix)
    gaff_model = GaffTorchSimModel(str(prefix), device=device, dtype=dtype)
    num_atoms = int(gaff_model.gaff.num_atoms)
    torsion_indices = select_phi_torsions(load_torsion_indices(Path(args.cv_file)))
    torsion_indices = torsion_indices[torsion_indices.max(axis=1) < num_atoms]
    if torsion_indices.shape[0] != 3:
        raise ValueError(f"need 3 φ torsions, got {torsion_indices.shape[0]}")
    cv_fn = Torsion(torsion_indices)
    z_t = atomic_numbers_from_pdb_organic(args.pdb).to(device=device)
    if int(z_t.numel()) != num_atoms:
        raise ValueError(f"pdb Z={z_t.numel()} != GAFF N={num_atoms}")
    mass_np = load_masses(Path(args.mass_file), num_atoms)
    mass_t = torch.as_tensor(mass_np, dtype=dtype, device=device)
    symbols = _symbols_from_z(z_t.detach().cpu().numpy())
    print(
        f"tetra-alanine 3φ WT-MetaD  device={device}  dtype={dtype}  N={num_atoms}  R={args.replicas}  steps={n_steps} ({n_steps * dt_ps:g} ps)  T={args.temperature} K",
        flush=True,
    )
    print(f"[CV] 3φ (0-based)={torsion_indices.tolist()}", flush=True)
    init_path = Path(args.init_npy) if args.init_npy else None
    if init_path is not None and init_path.is_file():
        loaded = np.load(init_path)
        if loaded.ndim != 3 or loaded.shape[-1] != 3:
            raise ValueError(f"bad init npy {init_path}: {loaded.shape}")
        if loaded.shape[1] != num_atoms:
            raise ValueError(
                f"init n_atoms={loaded.shape[1]} != GAFF n_atoms={num_atoms}"
            )
        idx = np.linspace(0, loaded.shape[0] - 1, args.replicas).astype(np.int64)
        loaded = loaded[idx]
        print(
            f"[init] subsampled {args.replicas} from {init_path} (pool={np.load(init_path).shape[0]})",
            flush=True,
        )
    else:
        base = load_sponge_coordinates(prefix, num_atoms=num_atoms)
        loaded = np.repeat(base[None, ...], args.replicas, axis=0)
        print(f"[init] SPONGE coords × {args.replicas}", flush=True)
    coordinate = torch.as_tensor(loaded, dtype=dtype, device=device)
    sigma = _parse_floats(args.metad_sigma)
    bias = WellTemperedMetadynamicsBias(
        cv_fn,
        n_cv=3,
        sigma=sigma,
        height=float(args.metad_height),
        bias_factor=float(args.metad_bias_factor),
        temperature=float(args.temperature),
        kB=float(args.kB),
        max_hills=int(args.metad_max_hills),
        grid_n=int(args.metad_grid_n),
        period=TWO_PI,
        device=device,
        dtype=dtype,
        n_atoms_per_system=num_atoms,
    )
    if args.resume_hills:
        n_loaded = bias.load_hills(args.resume_hills)
        print(f"[resume] loaded {n_loaded} hills from {args.resume_hills}", flush=True)
    model = SumModel(gaff_model, bias)
    driver = TorchSimWelltemperMetadynamic(
        model,
        bias,
        timestep=float(args.timestep),
        langevin_friction=float(args.langevin_friction),
        temperature=float(args.temperature),
        deposit_pace=int(args.metad_deposit_pace),
    )
    grid_msg = "sum-of-Gaussians"
    if bias.use_grid:
        dx = bias._grid_dx.detach().cpu().numpy()
        sig_np = bias.sigma.detach().cpu().numpy()
        grid_msg = f"B-spline n={bias._grid_shape}  dx={dx} rad  σ/dx={sig_np / dx}"
    print(
        f"[init] WT-MetaD σ={sigma} rad  w0={args.metad_height} eV  γ={args.metad_bias_factor}  period=2π  n_hills0={bias.n_hills}  {grid_msg}",
        flush=True,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * float(args.cell_length)
    state0 = batched_coordinates_to_sim_state(
        coordinate,
        atomic_numbers=z_t,
        masses=mass_t,
        cell=cell,
        pbc=False,
        charge=0.0,
        spin=1.0,
    )
    state0.rng = int(args.seed)
    out = Path(
        args.output
        or EXPERIMENT_DIR
        / "results"
        / f"tetra_alanine_metad_r{args.replicas}_s{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    live = out / str(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)
    traj = out / "trajectory.xyz"
    if traj.is_file():
        traj.unlink()
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
            n_steps - restart.offset,
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
