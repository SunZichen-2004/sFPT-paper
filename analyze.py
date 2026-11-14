"""Reconstruct FES, compare the original 25 ps runs, and plot paper references."""

from pathlib import Path
import argparse, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from fp_transformer_torch.reconstruct_FES.archive import (
    reconstruct_sfpt,
    reconstruct_metad,
)

ROOT = Path(__file__).resolve().parent
GUIDE = {
    "dialanine_vacuum": 0.5,
    "dialanine_water": 1.0,
    "tetra_alanine": 0.5,
    "rearrangement": 1.0,
}
DG_REFERENCE = {
    "dialanine_vacuum": 2.0909,
    "dialanine_water": 4.384338917757023,
    "tetra_alanine": 0.3014,
    "rearrangement": -6.31,
}
WINDOW = {
    "dialanine_vacuum": 5.0,
    "dialanine_water": 10.0,
    "tetra_alanine": 20.0,
    "rearrangement": 4.0,
}


def validate_initial_source(case, config, metadata):
    """Identify the selected input by its repo path, including after relocation."""
    expected = config.get("init_npy")
    actual = metadata.get("init_npy")
    if expected is None:
        if actual is not None:
            raise ValueError(f"{case}: expected initialization from the water box")
        expected = "experiments/rearrangement_water_fpt/mlmm_box/box.xyz"
        actual = metadata.get("xyz")
    if actual is None:
        raise ValueError(f"{case}: missing initial source metadata")
    parts = Path(expected).parts
    if Path(actual).parts[-len(parts) :] != parts:
        raise ValueError(f"{case}: archive used a different initial ensemble")


def system_of(case):
    return case.rsplit("_", 1)[0]


def axes_for(case):
    if case.startswith("rearrangement"):
        with np.load(ROOT / "references/rearrangement_paper_fes.npz") as z:
            return [z["d1"].copy(), z["d2"].copy()]
    axis = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    return [axis] * (3 if case.startswith("tetra") else 2)


def reconstruct(case, path, device):
    axes = axes_for(case)
    periodic = not case.startswith("rearrangement")
    if case.endswith("_sfpt"):
        return reconstruct_sfpt(
            path,
            axes,
            start_ps=25.0 - WINDOW[system_of(case)],
            end_ps=25.0,
            periodic=periodic,
            device=device,
        )
    return reconstruct_metad(path, axes, periodic=periodic, device=device)


def compare(actual, reference):
    details = {}
    for key in ["joint_fes"] + [
        k for k in actual if k.startswith("fes_1d_") or k.startswith("fes_2d_")
    ]:
        ref = np.asarray(reference[key])
        value = np.asarray(actual[key])
        mask = (
            np.isfinite(ref)
            & np.isfinite(value)
            & (ref < (8 if key.startswith("fes_1d") else 12))
        )
        difference = value[mask] - ref[mask]
        details[key] = dict(
            rmse_kcal_mol=float(np.sqrt(np.mean(difference**2))),
            offset_aligned_rmse_kcal_mol=float(np.std(difference)),
            points=int(mask.sum()),
            reference_cutoff_kcal_mol=8 if key.startswith("fes_1d") else 12,
        )
    details["delta_g_actual"] = float(actual["delta_g_kcal_mol"])
    details["delta_g_original_25ps"] = float(reference["delta_g_kcal_mol"])
    details["delta_g_difference"] = (
        details["delta_g_actual"] - details["delta_g_original_25ps"]
    )
    return details


def paper_surface(case):
    system = system_of(case)
    method = "sfpt" if case.endswith("sfpt") else "meta"
    if system in ["dialanine_vacuum", "dialanine_water"]:
        with np.load(ROOT / "references/dialanine_paper_fes.npz") as z:
            prefix = "vac" if system == "dialanine_vacuum" else "wat"
            return [z["phi"], z["psi"]], z[f"{prefix}_F_{method}"].T, (0, 1)
    if system == "tetra_alanine":
        with np.load(ROOT / "references/tetra_paper_fes.npz") as z:
            return [z["remd_axis"]] * 2, z["remd_F23"].T, (1, 2)
    with np.load(ROOT / "references/rearrangement_paper_fes.npz") as z:
        return (
            [z["d1"], z["d2"]],
            z["F_sfpt" if method == "sfpt" else "F_metad"].T,
            (0, 1),
        )


def compare_paper_surface(case, actual):
    axes, surface, (i, j) = paper_surface(case)
    axes = [np.asarray(a) for a in axes]
    surface = np.asarray(surface, dtype=float)
    if not case.startswith("rearrangement"):
        for dim in (0, 1):
            if np.isclose(axes[dim][-1] - axes[dim][0], 2 * np.pi):
                axes[dim] = axes[dim][:-1]
                surface = np.take(surface, np.arange(len(axes[dim])), axis=dim)
        axes = [np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi]) for a in axes]
        surface = np.tile(surface, (3, 3))
    target_axes = [actual[f"axis_{i}"], actual[f"axis_{j}"]]
    points = np.stack(np.meshgrid(*target_axes, indexing="ij"), -1)
    reference = RegularGridInterpolator(
        axes, surface, bounds_error=False, fill_value=np.nan
    )(points)
    reference -= np.nanmin(reference)
    value = actual[f"fes_2d_{i}_{j}"]
    mask = np.isfinite(reference) & np.isfinite(value) & (reference < 12)
    error = value[mask] - reference[mask]
    return dict(
        rmse_kcal_mol=float(np.sqrt(np.mean(error**2))),
        offset_aligned_rmse_kcal_mol=float(np.std(error)),
        reference_cutoff_kcal_mol=12,
        points=int(mask.sum()),
        projection_axes=[i, j],
    )


def make_figure(case, actual, original, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    system = system_of(case)
    i, j = (1, 2) if system == "tetra_alanine" else (0, 1)
    key = f"fes_2d_{i}_{j}"
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    axes = [actual[f"axis_{i}"], actual[f"axis_{j}"]]
    pa, paper, pair = paper_surface(case)
    for ax, data, grid, title in zip(
        axs,
        [actual[key], original[key], paper],
        [axes, axes, pa],
        [
            "New, 25 ps",
            "Original, 25 ps",
            "Paper long-time reference"
            if system != "tetra_alanine"
            else "Independent REMD reference",
        ],
    ):
        m = ax.pcolormesh(
            grid[0],
            grid[1],
            np.ma.masked_invalid(data.T),
            vmin=0,
            vmax=12,
            cmap="viridis",
            shading="auto",
        )
        labels = (
            [r"$d_1$ / Å", r"$d_2$ / Å"]
            if system == "rearrangement"
            else [r"$\phi_2$ / rad", r"$\phi_3$ / rad"]
            if system == "tetra_alanine"
            else [r"$\phi$ / rad", r"$\psi$ / rad"]
        )
        ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
    fig.colorbar(m, ax=axs, label="F / kcal mol$^{-1}$", shrink=0.85)
    fig.suptitle(case.replace("_", " "))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--references-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cases = (
        list(json.loads((ROOT / "cases.json").read_text()))
        if args.case == "all"
        else [args.case]
    )
    derived = ROOT / "references/derived"
    derived.mkdir(exist_ok=True)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    summary = (
        json.loads((out / "summary.json").read_text())
        if (out / "summary.json").exists()
        else {}
    )
    for case in cases:
        reference_path = derived / f"{case}_25ps_fes.npz"
        if args.force or not reference_path.exists():
            print(f"[original FES] {case}", flush=True)
            ref = reconstruct(
                case, ROOT / f"references/{case}_original_25ps.npz", args.device
            )
            np.savez_compressed(reference_path, **ref)
        if args.references_only:
            continue
        path = ROOT / f"results/{case}/trajectory.npz"
        if not path.exists():
            summary[case] = {"status": "pending"}
            continue
        with np.load(path, allow_pickle=False) as z:
            if not bool(z["complete"]):
                summary[case] = {
                    "status": "incomplete",
                    "last_production_ps": float(
                        z["time_ps"][z["phase"] == "production"].max()
                    ),
                }
                continue
            config = json.loads((ROOT / "cases.json").read_text())[case]
            metadata = json.loads(str(z["metadata_json"]))
            for key in (
                "seed",
                "replicas",
                "timestep",
                "temperature",
                "langevin_friction",
            ):
                if not np.isclose(float(metadata[key]), float(config[key])):
                    raise ValueError(
                        f"{case}: archive parameter {key} does not match cases.json"
                    )
            validate_initial_source(case, config, metadata)
            if z["cv"].shape[1] != config["replicas"] or "coordinates" not in z:
                raise ValueError(f"{case}: incomplete ensemble/coordinate output")
            production = z["phase"] == "production"
            times = z["time_ps"][production]
            expected_times = (
                np.arange(0, 25001, config["dump_every"]) * config["timestep"]
            )
            if not np.allclose(times, expected_times):
                raise ValueError(f"{case}: missing or duplicated production frames")
            if case.endswith("sfpt"):
                if not np.allclose(
                    z["lambda_value"][production], config["production_lambda"]
                ):
                    raise ValueError(
                        f"{case}: production lambda is not the configured constant"
                    )
            else:
                expected_hills = config["replicas"] * (
                    25000 // config["metad_deposit_pace"]
                )
                if len(z["hills_heights"]) != expected_hills:
                    raise ValueError(
                        f"{case}: hill count does not cover the full production run"
                    )
                if not np.isclose(float(z["bias_factor"]), config["metad_bias_factor"]):
                    raise ValueError(f"{case}: wrong metadynamics bias factor")
                if not np.allclose(
                    z["hills_sigma"], np.fromstring(config["metad_sigma"], sep=",")
                ):
                    raise ValueError(f"{case}: wrong metadynamics widths")
            elapsed = float(z["time_ps"][z["phase"] == "production"].max())
        if not np.isclose(elapsed, 25):
            raise ValueError(f"{case}: production length {elapsed}, expected 25 ps")
        print(f"[new FES] {case}", flush=True)
        result = reconstruct(case, path, args.device)
        np.savez_compressed(out / f"{case}_fes.npz", **result)
        with np.load(reference_path) as z:
            reference = {k: z[k] for k in z.files}
        metrics = compare(result, reference)
        metrics["paper_reference_2d_fes"] = compare_paper_surface(case, result)
        system = system_of(case)
        metrics.update(
            status="analyzed",
            production_ps=elapsed,
            seed=0,
            reference_basin_delta_g=DG_REFERENCE[system],
            absolute_basin_error_kcal_mol=abs(
                metrics["delta_g_actual"] - DG_REFERENCE[system]
            ),
            paper_basin_guide_kcal_mol=GUIDE[system],
            within_paper_basin_guide=abs(
                metrics["delta_g_actual"] - DG_REFERENCE[system]
            )
            <= GUIDE[system],
            fes_window_ps=[25 - WINDOW[system], 25]
            if case.endswith("sfpt")
            else "cumulative bias at 25 ps",
        )
        summary[case] = metrics
        (out / f"{case}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        make_figure(case, result, reference, out / f"{case}_comparison.png")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.references_only:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
