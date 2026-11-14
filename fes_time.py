"""Plot bundled paper FES time series or reconstruct a new sampling archive."""

from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent


def verify_data():
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    for entry in catalog["files"]:
        path = ROOT / "data" / entry["path"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed data file: {path}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    if z[key].dtype.kind == "O":
                        raise ValueError(f"Pickled object array: {path}:{key}")
    for entry in catalog["series"]:
        with np.load(ROOT / "data" / entry["path"], allow_pickle=False) as z:
            times = z["time_ps"]
            if len(times) != entry["frames"] or not np.all(np.diff(times) > 0):
                raise ValueError(f"Incomplete time axis: {entry['path']}")
            if times[-1] != entry["time_max_ps"]:
                raise ValueError(f"Wrong final time: {entry['path']}")
            for key in z.files:
                if key.startswith("fes_2d_"):
                    i, j = map(int, key.split("_")[-2:])
                    if z[key].shape != (
                        len(times),
                        len(z[f"axis_{i}"]),
                        len(z[f"axis_{j}"]),
                    ):
                        raise ValueError(f"Wrong time/grid axes: {entry['path']}:{key}")
                    support = np.isfinite(z[key])
                    mask_key = key.replace("fes_", "support_mask_", 1)
                    if mask_key not in z or not np.array_equal(z[mask_key], support):
                        raise ValueError(
                            f"Missing or inconsistent FES support mask: {entry['path']}:{key}"
                        )
                    if not support.reshape(len(times), -1).any(axis=1).all():
                        raise ValueError(f"Empty FES frame: {entry['path']}:{key}")
                    if entry["case"].startswith("tetra"):
                        from scipy.special import logsumexp

                        kt = 8.617330337217213e-5 * 300 * 23.06054783061903
                        marginal = -kt * logsumexp(-z[key] / kt, axis=2)
                        marginal -= marginal.min(axis=1, keepdims=True)
                        if not np.allclose(
                            marginal, z[f"fes_1d_{i}"], atol=2e-5, rtol=1e-6
                        ):
                            raise ValueError(
                                f"Inconsistent marginal axes: {entry['path']}:{key}"
                            )
    print(
        f"Verified {len(catalog['series'])} full time series and {len(catalog['files'])} data files."
    )


def plot(path, output, pair=None, times=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as z:
        case = json.loads(str(z["metadata_json"])).get("case", path.parent.name)
        pair = pair or ((1, 2) if case.startswith("tetra") else (0, 1))
        time = z["time_ps"]
        if times:
            chosen = []
            for t in times:
                indexes = np.flatnonzero(np.isclose(time, t))
                if not len(indexes):
                    raise ValueError(
                        f"No saved FES at {t} ps; available: {time.tolist()}"
                    )
                chosen.append(int(indexes[0]))
        else:
            chosen = sorted(set([0, len(time) // 2, len(time) - 1]))
        fig = plt.figure(figsize=(4.0 * len(chosen), 6.8), layout="constrained")
        grid = fig.add_gridspec(2, len(chosen), height_ratios=[1.2, 0.8])
        labels = (
            [r"$d_1$ / Å", r"$d_2$ / Å"]
            if case.startswith("rearrangement")
            else [rf"$\phi_{{{i + 1}}}$ / rad" for i in pair]
            if case.startswith("tetra")
            else [r"$\phi$ / rad", r"$\psi$ / rad"]
        )
        axes = []
        for col, index in enumerate(chosen):
            ax = fig.add_subplot(grid[0, col])
            axes.append(ax)
            m = ax.pcolormesh(
                z[f"axis_{pair[0]}"],
                z[f"axis_{pair[1]}"],
                z[f"fes_2d_{pair[0]}_{pair[1]}"][index].T,
                shading="auto",
                vmin=0,
                vmax=12,
            )
            ax.set(title=f"{time[index]:g} ps", xlabel=labels[0], ylabel=labels[1])
        fig.colorbar(m, ax=axes, label="F / kcal mol$^{-1}$", shrink=0.85)
        ax = fig.add_subplot(grid[1, :])
        ax.plot(time, z["delta_g_kcal_mol"], marker=".", ms=3, lw=1)
        ax.set(xlabel="Production time / ps", ylabel=r"$\Delta G$ / kcal mol$^{-1}$")
        ax.grid(alpha=0.2)
        fig.suptitle(case.replace("_", " "))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
    print(output)


def reconstruct(case, trajectory, output, device, interval=None, end=None):
    from fp_transformer_torch.reconstruct_FES.archive import (
        reconstruct_sfpt,
        reconstruct_metad,
    )

    config = json.loads((ROOT / "protocols/paper.json").read_text())[case]
    with np.load(trajectory, allow_pickle=False) as z:
        if not bool(z["complete"]):
            raise ValueError("Trajectory is incomplete")
        maximum = float(z["time_ps"][z["phase"] == "production"].max())
    end = maximum if end is None else end
    if end <= 0 or end > maximum:
        raise ValueError("End must be within the completed production time")
    interval = config["fes_interval_ps"] if interval is None else interval
    if interval <= 0:
        raise ValueError("Interval must be positive")
    ends = np.arange(interval, end + 1e-8, interval)
    if len(ends) == 0 or not np.isclose(ends[-1], end):
        ends = np.r_[ends, end]
    axes = (
        [np.linspace(1, 7, 140)] * 2
        if case.startswith("rearrangement")
        else [np.linspace(-np.pi, np.pi, 72, endpoint=False)]
        * (3 if case.startswith("tetra") else 2)
    )
    rows = []
    starts = []
    for t in ends:
        start = (
            max(0.0, float(t) - config["sfpt_window_ps"])
            if case.endswith("sfpt")
            else 0.0
        )
        result = (
            reconstruct_sfpt(
                trajectory,
                axes,
                start_ps=start,
                end_ps=float(t),
                periodic=not case.startswith("rearrangement"),
                device=device,
            )
            if case.endswith("sfpt")
            else reconstruct_metad(
                trajectory,
                axes,
                end_ps=float(t),
                periodic=not case.startswith("rearrangement"),
                device=device,
            )
        )
        rows.append(result)
        starts.append(start)
        print(f"[FES] {case}: {start:g}–{t:g} ps", flush=True)
    values = {
        "time_ps": ends,
        "window_start_ps": starts,
        "metadata_json": np.asarray(
            json.dumps(
                dict(
                    case=case,
                    units="kcal/mol",
                    axis_order="ij; time first",
                    trajectory=str(trajectory),
                )
            )
        ),
    }
    for key in rows[0]:
        if key.startswith("axis_"):
            values[key] = rows[0][key]
        elif key.startswith("fes_") or key == "delta_g_kcal_mol":
            values[key] = np.stack([r[key] for r in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="action", required=True)
    s.add_parser("verify")
    a = s.add_parser("plot")
    a.add_argument("--case")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--input", type=Path)
    a.add_argument("--output", type=Path)
    a.add_argument("--times", type=float, nargs="+")
    a.add_argument("--pair", type=int, nargs=2)
    a = s.add_parser("reconstruct")
    a.add_argument("--case", required=True)
    a.add_argument("--trajectory", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--interval", type=float)
    a.add_argument("--end", type=float)
    a = p.parse_args()
    if a.action == "verify":
        verify_data()
    elif a.action == "plot":
        if a.input is None and a.case is None:
            p.error("plot requires --case or --input")
        path = a.input or ROOT / f"data/fes_time/{a.case}/seed_{a.seed}.npz"
        output = (
            a.output
            or ROOT / f"reports/time_series/{path.parent.name}_seed{a.seed}.png"
        )
        plot(path, output, tuple(a.pair) if a.pair else None, a.times)
    else:
        reconstruct(a.case, a.trajectory, a.output, a.device, a.interval, a.end)


if __name__ == "__main__":
    main()
