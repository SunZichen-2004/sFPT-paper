"""Check bundled input hashes and complete trajectory arrays, without pickle."""

from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


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
