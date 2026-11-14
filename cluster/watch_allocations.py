"""Dispatch remaining cases only to explicitly authorized, already-running jobs."""

from pathlib import Path
import argparse, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", nargs="+", required=True, type=int)
    p.add_argument("--core", type=Path, required=True)
    args = p.parse_args()
    if 863142 in args.jobs:
        raise SystemExit("Job 863142 is excluded by the user.")
    dispatched = {}
    while True:
        import json
        import numpy as np

        all_complete = True
        for case in json.loads((ROOT / "cases.json").read_text()):
            path = ROOT / "results" / case / "trajectory.npz"
            if not path.exists():
                all_complete = False
                break
            with np.load(path, allow_pickle=False) as z:
                if not bool(z["complete"]):
                    all_complete = False
                    break
        if all_complete:
            return
        for job in args.jobs:
            if job in dispatched and dispatched[job].poll() is None:
                continue
            status = subprocess.run(
                ["squeue", "--noheader", "-j", str(job), "-o", "%T"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if status != "RUNNING":
                continue
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.core.resolve())
            env["OMP_NUM_THREADS"] = "4"
            env["MPLBACKEND"] = "Agg"
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            log = (ROOT / "logs" / f"worker_job{job}.log").open("a")
            cmd = [
                "srun",
                f"--jobid={job}",
                "--overlap",
                "-N1",
                "-n1",
                "-c12",
                sys.executable,
                "-u",
                str(ROOT / "cluster/worker.py"),
            ]
            dispatched[job] = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            print(f"Dispatched remaining cases to authorized job {job}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
