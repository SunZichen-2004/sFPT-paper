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
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Single-seed 25 ps validation",
        "",
        f"{len(rows)}/{len(cases)} cases completed and analyzed. Each uses seed 0 and the main-text replica count.",
        "",
        "FES RMSE is against the archived **same-length 25 ps** calculation, in kcal/mol. Both surfaces are minimum-shifted; the mask is the original joint or projected FES below 12 kcal/mol. Tetra-alanine has a three-dimensional joint FES; its main projection is phi2/phi3. Other cases use the full two-dimensional FES.",
        "",
        "| Case | Replicas | Joint FES RMSE | Main 2D RMSE | New delta G | Original 25 ps delta G | Independent reference delta G |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        name = row["case"]
        lines.append(
            f"| [{name}]({name}_comparison.png) | {row['replicas']} | "
            f"{row['joint_fes_rmse_vs_original_kcal_mol']:.3f} | "
            f"{row['main_projection_rmse_vs_original_kcal_mol']:.3f} | "
            f"{row['delta_g_kcal_mol']:.3f} | "
            f"{row['original_25ps_delta_g_kcal_mol']:.3f} | "
            f"{row['independent_reference_delta_g_kcal_mol']:.3f} |"
        )
    lines += [
        "",
        "The sign convention is `delta G = kB*T*ln(P1/P2)`. Basin 1 has phi < 0 for dialanine, all three phi angles < 0 for tetra-alanine, and d2-d1 < 0 (product) for rearrangement. Physical-density reconstruction is shared with the archived comparison. Independent references are umbrella sampling for dialanine and rearrangement, and the paper's preliminary REMD estimate for tetra-alanine. Short-run agreement with the original 25 ps output does not establish convergence to the long-time paper reference. In particular, the original tetra-alanine sFPT 25 ps delta G is already 1.484 kcal/mol versus REMD 0.301 kcal/mol.",
        "",
        "Reaction WT-MetaD has substantial archived run-to-run variation. The three original 25 ps delta G values are -11.661, -7.989 and -7.980 kcal/mol, versus the new -8.191. New FES RMSE against the three runs is 1.995, 0.727 and 0.632 kcal/mol. The main table retains original seed 0; all three comparisons are recorded in `rearrangement_metad_variation.json`. This is descriptive variation, not a confidence interval.",
        "",
        "`*_metrics.json` records all marginal errors and the separate paper-reference comparisons. `*_fes.npz` holds reconstructed numerical surfaces; `benchmark_summary.csv` is the machine-readable table. Full trajectory NPZ files remain under local `results/`, with their checksums and shape checks in `archive_integrity.json`.",
        "",
    ]
    out.with_name("README.md").write_text("\n".join(lines))
    print(f"Validated {len(rows)}/{len(cases)} cases. Summary: {out}")


if __name__ == "__main__":
    main()
