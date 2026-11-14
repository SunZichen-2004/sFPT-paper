"""Run remaining cases on one allocated GPU; skip active or completed cases."""

from pathlib import Path
import argparse, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]


def active(case):
    record = ROOT / "results" / case / "run.json"
    if not record.exists():
        return False
    try:
        state = json.loads(record.read_text())
    except (ValueError, OSError):
        return False
    if state.get("status") != "running" or not state.get("job_id"):
        return False
    result = subprocess.run(
        ["squeue", "--noheader", "-j", str(state["job_id"]), "-o", "%T"],
        capture_output=True,
        text=True,
    )
    return "RUNNING" in result.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    cases = list(json.loads((ROOT / "cases.json").read_text()))
    (ROOT / "logs").mkdir(exist_ok=True)
    while True:
        pending = False
        for case in cases:
            path = ROOT / "results" / case / "trajectory.npz"
            if path.exists():
                import numpy as np

                with np.load(path, allow_pickle=False) as z:
                    if bool(z["complete"]):
                        continue
            pending = True
            if active(case):
                continue
            with (
                ROOT
                / "logs"
                / f"{case}_job{os.environ.get('SLURM_JOB_ID', 'local')}.log"
            ).open("a") as log:
                result = subprocess.run(
                    [sys.executable, "-u", str(ROOT / "run_case.py"), case],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
            print(f"{case}: exit {result.returncode}", flush=True)
        if args.once or not pending:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
