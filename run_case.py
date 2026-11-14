"""Run one paper case, using paths relative to this repository."""

from pathlib import Path
import argparse, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parent


def resolve_case(
    case,
    *,
    protocol="paper",
    seed=None,
    steps=None,
    replicas=None,
    production_lambda=None,
    hill_height=None,
    init_npy=None,
    device=None,
    output=None,
):
    if protocol == "paper":
        config = json.loads((ROOT / "protocols/paper.json").read_text())[case][
            "parameters"
        ]
    elif protocol == "example":
        config = json.loads((ROOT / "cases.json").read_text())[case]
    else:
        raise ValueError("protocol must be paper or example")
    if seed is not None:
        if seed < 0 or seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        config["seed"] = seed
    if steps is not None:
        config["production_steps"] = steps
    if config["production_steps"] <= 0:
        raise ValueError("steps must be positive")
    if replicas is not None:
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        config["replicas"] = replicas
        if config["method"] == "metad" and hill_height is None:
            config["metad_height"] = 1.28 / replicas
        if case.startswith("dialanine_vacuum"):
            candidate = ROOT / f"experiments/dialanine/init_coords_{replicas}.npy"
            if not candidate.exists() and init_npy is None:
                raise ValueError("Supply --init-npy for this replica count")
            if candidate.exists():
                config["init_npy"] = str(candidate)
        if case == "tetra_alanine_sfpt":
            candidate = (
                ROOT
                / f"experiments/tetra_alanine_sFPT/initial_ensemble/init_coords_{replicas}.npy"
            )
            if not candidate.exists() and init_npy is None:
                raise ValueError("Supply --init-npy for this replica count")
            if candidate.exists():
                config["init_npy"] = str(candidate)
    for value, key, method in [
        (production_lambda, "production_lambda", "sfpt"),
        (hill_height, "metad_height", "metad"),
    ]:
        if value is not None:
            if config["method"] != method or not 0 < value < float("inf"):
                raise ValueError(f"{key} requires a positive value and {method} method")
            config[key] = value
    if init_npy is not None:
        config["init_npy"] = str(Path(init_npy).resolve())
    if device is not None:
        config["device"] = device
    if output is None:
        if (
            protocol == "example"
            and seed in (None, 0)
            and all(
                v is None
                for v in (steps, replicas, production_lambda, hill_height, init_npy)
            )
        ):
            output = ROOT / "results" / case
        else:
            suffix = "".join(
                f"_{k}{v}"
                for k, v in [
                    ("r", replicas),
                    ("lambda", production_lambda),
                    ("h", hill_height),
                    ("steps", steps),
                ]
                if v is not None
            )
            if init_npy is not None:
                raise ValueError("Supply --output when using a custom --init-npy")
            output = (
                ROOT / "results" / protocol / (case + suffix) / f"seed_{config['seed']}"
            )
    config["output"] = str(Path(output).resolve())
    return config


def build_command(config):
    config = config.copy()
    script = config.pop("script")
    method = config.pop("method")
    steps = config.pop("production_steps")
    dump = config.pop("dump_every")
    if method == "sfpt":
        config["production_steps"] = steps
        config[
            "dump_every" if script.startswith("tetra_alanine_sFPT") else "stage_steps"
        ] = min(steps, dump)
    else:
        config["steps"] = steps
        config["dump_every"] = min(steps, dump)
    cmd = [sys.executable, "-u", str(ROOT / "experiments" / script)]
    for key, value in config.items():
        if key == "init_npy":
            value = str(ROOT / value)
        if value is False:
            continue
        cmd.append("--" + key.replace("_", "-"))
        if value is not True:
            cmd.append(str(value))
    return cmd


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
    p.add_argument("--protocol", choices=["paper", "example"], default="paper")
    p.add_argument("--seed", type=int)
    p.add_argument("--replicas", type=int)
    p.add_argument("--production-lambda", type=float)
    p.add_argument("--hill-height", type=float)
    p.add_argument("--init-npy", type=Path)
    p.add_argument("--device")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config = resolve_case(
        args.case,
        protocol=args.protocol,
        seed=args.seed,
        steps=args.steps,
        replicas=args.replicas,
        production_lambda=args.production_lambda,
        hill_height=args.hill_height,
        init_npy=args.init_npy,
        device=args.device,
        output=args.output,
    )
    steps = config["production_steps"]
    cmd = build_command(config)
    print(
        json.dumps(
            dict(
                case=args.case,
                protocol=args.protocol,
                production_ps=steps * config["timestep"],
                command=cmd,
            ),
            indent=2,
        ),
        flush=True,
    )
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
        protocol=args.protocol,
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
