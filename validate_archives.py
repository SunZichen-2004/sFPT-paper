"""Check bundled input hashes and complete trajectory arrays, without pickle."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from fp_transformer_torch.collective_variable import Torsion

ROOT = Path(__file__).resolve().parent


def final_cv_error(case, final, cv):
    """Confirm recorded CVs describe the full coordinates saved beside them."""
    if case.startswith("rearrangement"):
        pairs = (
            np.loadtxt(
                ROOT / "experiments/rearrangement_water_fpt/cv_pairs.txt", dtype=int
            )
            - 1
        )
        actual = np.linalg.norm(final[:, pairs[:, 0]] - final[:, pairs[:, 1]], axis=-1)
        error = actual - cv
    else:
        source = (
            "tetra_alanine_sFPT/backbone_phi.txt"
            if case.startswith("tetra")
            else "dialanine_water/mm_box/backbone_cv.txt"
            if case.startswith("dialanine_water")
            else "dialanine/backbone_cv.txt"
        )
        indices = np.loadtxt(ROOT / "experiments" / source, dtype=int) - 1
        actual = (
            Torsion(indices)(torch.as_tensor(final, dtype=torch.float64))
            .numpy()
            .squeeze(-1)
        )
        error = np.angle(np.exp(1j * (actual - cv)))
    maximum = float(np.abs(error).max())
    if maximum > 2e-5:
        raise ValueError(
            f"{case}: stored CVs disagree with final coordinates ({maximum})"
        )
    return maximum


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    inputs = json.loads((ROOT / "input_checksums.json").read_text())
    for name, expected in inputs.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Bundled molecular input changed: {name}")
    cases = json.loads((ROOT / "cases.json").read_text())
    report = {"verified_input_files": len(inputs), "cases": {}}
    for case, config in cases.items():
        path = ROOT / f"results/{case}/trajectory.npz"
        if not path.exists():
            report["cases"][case] = {"status": "pending"}
            continue
        with np.load(path, allow_pickle=False) as z:
            if not bool(z["complete"]):
                report["cases"][case] = {"status": "incomplete"}
                continue
            cv = z["cv"]
            coords = z["coordinates"]
            final = z["final_coordinates"]
            expected_dim = 3 if case.startswith("tetra") else 2
            if cv.shape != (len(z["time_ps"]), config["replicas"], expected_dim):
                raise ValueError(f"{case}: invalid CV axes")
            if coords.shape[:2] != cv.shape[:2] or coords.shape[-1] != 3:
                raise ValueError(f"{case}: invalid full-coordinate axes")
            if final.shape != coords.shape[1:]:
                raise ValueError(f"{case}: invalid final coordinates")
            for name, array in [
                ("cv", cv),
                ("coordinates", coords),
                ("final_coordinates", final),
            ]:
                if not np.isfinite(array).all():
                    raise ValueError(f"{case}: nonfinite {name}")
            if not np.array_equal(coords[-1], final.astype(coords.dtype)):
                raise ValueError(f"{case}: final coordinates differ from last frame")
            cv_error = final_cv_error(case, final, cv[-1])
            production = z["phase"] == "production"
            if case.endswith("sfpt"):
                bw = z["bandwidth"][production]
                if not np.isfinite(bw).all() or not (bw > 0).all():
                    raise ValueError(f"{case}: invalid production bandwidths")
            else:
                for name in ["hills_centers", "hills_heights", "bias_grid"]:
                    if not np.isfinite(z[name]).all():
                        raise ValueError(f"{case}: nonfinite {name}")
            entry = dict(
                status="verified",
                archive=str(path.relative_to(ROOT)),
                bytes=path.stat().st_size,
                sha256=sha256(path),
                cv_shape=list(cv.shape),
                coordinates_shape=list(coords.shape),
                coordinate_dtype=str(coords.dtype),
                final_coordinate_dtype=str(final.dtype),
                complete=True,
                finite_coordinates_and_cvs=True,
                final_coordinates_match_last_frame=True,
                final_cv_from_coordinates_max_error=cv_error,
            )
        report["cases"][case] = entry
        print(f"[verified archive] {case}", flush=True)
    report["verified_cases"] = sum(
        v["status"] == "verified" for v in report["cases"].values()
    )
    (ROOT / "reports/archive_integrity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Verified {report['verified_cases']}/{len(cases)} complete archives and {len(inputs)} input files."
    )


if __name__ == "__main__":
    main()
