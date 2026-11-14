"""Run one paper case, using paths relative to this repository."""

from pathlib import Path
import argparse, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=list(json.loads((ROOT / "cases.json").read_text())))
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override production length for smoke tests",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config = json.loads((ROOT / "cases.json").read_text())[args.case]
    script = config.pop("script")
    method = config.pop("method")
    steps = config.pop("production_steps")
    if args.steps is not None:
        steps = args.steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    dump = config.pop("dump_every")
    if method == "sfpt":
        config["production_steps"] = steps
        config["dump_every" if args.case == "tetra_alanine_sfpt" else "stage_steps"] = (
            min(steps, dump)
        )
    else:
        config["steps"] = steps
        config["dump_every"] = min(steps, dump)
    config["output"] = str((args.output or ROOT / "results" / args.case).resolve())
    cmd = [sys.executable, "-u", str(ROOT / "experiments" / script)]
    for key, value in config.items():
        if key == "init_npy":
            value = str(ROOT / value)
        if value is False:
            continue
        cmd.append("--" + key.replace("_", "-"))
        if value is not True:
            cmd.append(str(value))
    print(json.dumps(dict(case=args.case, command=cmd), indent=2), flush=True)
    if args.dry_run:
        return
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock_handle = (out / ".run.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("This case is already running in another worker.", flush=True)
        raise SystemExit(75)
    # Atomic directory claims also protect runs across cluster nodes, where
    # a filesystem may provide only node-local flock semantics.
    import atexit
    import shutil
    import socket

    claim = out / ".running"

    def owner_alive():
        try:
            owner = json.loads((claim / "owner.json").read_text())
        except (OSError, ValueError):
            return time.time() - claim.stat().st_mtime < 60 if claim.exists() else False
        job = owner.get("job_id")
        if job:
            status = subprocess.run(
                ["squeue", "--noheader", "-j", str(job), "-o", "%T"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            return status == "RUNNING"
        if owner.get("host") == socket.gethostname():
            try:
                os.kill(int(owner["pid"]), 0)
                return True
            except ProcessLookupError:
                return False
        return True

    if claim.exists() and not owner_alive():
        reaper = out / ".claim-reaper"
        try:
            reaper.mkdir()
        except FileExistsError:
            print("Another worker is reclaiming the expired run.", flush=True)
            raise SystemExit(75)
        try:
            if claim.exists() and not owner_alive():
                shutil.rmtree(claim)
        finally:
            reaper.rmdir()
    try:
        claim.mkdir()
    except FileExistsError:
        print("This case is active on another node.", flush=True)
        raise SystemExit(75)
    (claim / "owner.json").write_text(
        json.dumps(
            dict(
                job_id=os.environ.get("SLURM_JOB_ID"),
                host=socket.gethostname(),
                pid=os.getpid(),
            )
        )
        + "\n"
    )
    atexit.register(shutil.rmtree, claim, ignore_errors=True)
    archive = out / "trajectory.npz"
    if archive.is_file():
        import numpy as np

        with np.load(archive, allow_pickle=False) as z:
            if (
                bool(z["complete"])
                and "time_ps" in z
                and float(z["time_ps"][z["phase"] == "production"].max())
                >= steps * float(config["timestep"])
            ):
                print("Requested production is already complete.", flush=True)
                return
    record = dict(
        case=args.case,
        command=cmd,
        job_id=os.environ.get("SLURM_JOB_ID"),
        start_unix=time.time(),
        status="running",
    )
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(cmd, cwd=ROOT)
    record.update(
        exit_code=result.returncode,
        stop_unix=time.time(),
        status="complete"
        if result.returncode == 0
        else "checkpointed"
        if (out / "checkpoint.pt").exists()
        else "failed",
    )
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
