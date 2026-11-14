"""Write a compact table of completed, validated 25 ps FES comparisons."""

from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parent


def main():
    cases = json.loads((ROOT / "cases.json").read_text())
    rows = []
    for case, config in cases.items():
        path = ROOT / f"reports/{case}_metrics.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r.get("status") != "analyzed":
            continue
        pair = "fes_2d_1_2" if case.startswith("tetra") else "fes_2d_0_1"
        rows.append(
            dict(
                case=case,
                seed=config["seed"],
                replicas=config["replicas"],
                production_ps=r["production_ps"],
                joint_dimension=3 if case.startswith("tetra") else 2,
                joint_fes_rmse_vs_original_kcal_mol=r["joint_fes"]["rmse_kcal_mol"],
                main_projection_rmse_vs_original_kcal_mol=r[pair]["rmse_kcal_mol"],
                delta_g_kcal_mol=r["delta_g_actual"],
                original_25ps_delta_g_kcal_mol=r["delta_g_original_25ps"],
                independent_reference_delta_g_kcal_mol=r["reference_basin_delta_g"],
                within_paper_basin_error_guide=r["within_paper_basin_guide"],
            )
        )
    out = ROOT / "reports/benchmark_summary.csv"
    if rows:
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Validated {len(rows)}/{len(cases)} cases. Summary: {out}")


if __name__ == "__main__":
    main()
