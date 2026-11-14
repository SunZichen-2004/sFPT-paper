"""Rebuild the paper's reaction sFPT surface from bundled 79–83 ps raw data."""

from pathlib import Path
import argparse
import numpy as np
from fp_transformer_torch.reconstruct_FES.archive import joint_kde, KB_KCAL

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    target = ROOT / "references/rearrangement_paper_fes.npz"
    with np.load(target, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    with np.load(
        ROOT / "references/rearrangement_long_raw.npz", allow_pickle=False
    ) as z:
        lam = float(z["production_lambda"])
        probabilities = []
        for cv, bw in zip(z["cv"], z["bandwidth"]):
            rho = joint_kde(
                cv, bw, [values["d1"], values["d2"]], periodic=False, device=args.device
            )
            logp = np.log(np.maximum(rho, np.finfo(float).tiny)) * (1 + lam) / lam
            logp -= logp.max()
            prob = np.exp(logp)
            prob /= prob.sum()
            probabilities.append(prob)
        values["sfpt_seeds"] = z["seeds"]
    pooled = np.mean(probabilities, axis=0)
    fes = -KB_KCAL * 300 * np.log(np.maximum(pooled, np.finfo(float).tiny))
    fes -= fes.min()
    values["F_sfpt"] = fes.T
    values["sfpt_window_ps"] = np.asarray([79.0, 83.0])
    np.savez_compressed(target, **values)
    print(
        "Rebuilt 79–83 ps sFPT surface with equal normalized probability weights.",
        flush=True,
    )


if __name__ == "__main__":
    main()
