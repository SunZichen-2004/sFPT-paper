"""Compare the new seed-0 reaction WT-MetaD FES with original run-to-run variation."""

from pathlib import Path
import argparse, itertools, json
import numpy as np
from fp_transformer_torch.reconstruct_FES.archive import reconstruct_metad

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    with np.load(ROOT / "references/derived/rearrangement_metad_25ps_fes.npz") as z:
        original = [{k: z[k] for k in z.files}]
    axes = [original[0]["axis_0"], original[0]["axis_1"]]
    for seed in (1, 2):
        path = ROOT / f"references/variation/rearrangement_metad_seed{seed}_25ps.npz"
        result = reconstruct_metad(path, axes, periodic=False, device=args.device)
        np.savez_compressed(
            ROOT / f"references/variation/rearrangement_metad_seed{seed}_25ps_fes.npz",
            **result,
        )
        original.append(result)
    with np.load(ROOT / "reports/rearrangement_metad_fes.npz") as z:
        new = {k: z[k] for k in z.files}
    mask = original[0]["joint_fes"] < 12

    def rmse(a, b):
        return float(
            np.sqrt(np.mean((a["joint_fes"][mask] - b["joint_fes"][mask]) ** 2))
        )

    delta_g = np.asarray([float(r["delta_g_kcal_mol"]) for r in original])
    pairwise = [
        dict(seeds=[i, j], rmse_kcal_mol=rmse(original[i], original[j]))
        for i, j in itertools.combinations(range(3), 2)
    ]
    report = dict(
        time_ps=25,
        walkers=64,
        new_seed=0,
        new_delta_g=float(new["delta_g_kcal_mol"]),
        original_seed_delta_g=delta_g.tolist(),
        original_mean_delta_g=float(delta_g.mean()),
        original_sample_sd_delta_g=float(delta_g.std(ddof=1)),
        new_within_original_observed_delta_g_range=bool(
            delta_g.min() <= float(new["delta_g_kcal_mol"]) <= delta_g.max()
        ),
        original_pairwise_fes_rmse=pairwise,
        new_vs_original_fes_rmse=[rmse(new, r) for r in original],
        mask="original seed-0 FES below 12 kcal/mol",
        interpretation="Descriptive variation across three archived runs; not a fitted tolerance or confidence interval.",
    )
    (ROOT / "reports/rearrangement_metad_variation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
