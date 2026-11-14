"""Stage the exact MACE-MH-0 checkpoint used by the paper reproduction."""

from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path.home() / ".cache/mace/macemh0model"
    )
    args = parser.parse_args()
    expected = json.loads((ROOT / "model_checksums.json").read_text())[
        "checkpoint_sha256"
    ]
    output = args.output.expanduser()
    if output.exists():
        if checksum(output) != expected:
            raise ValueError(
                "Existing checkpoint has a different SHA-256; select a different --output path"
            )
        print(f"Verified existing checkpoint: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        with (
            urllib.request.urlopen(MODEL_URL, timeout=60) as response,
            temporary.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
        if checksum(temporary) != expected:
            raise ValueError(
                "Downloaded checkpoint does not match the tested model SHA-256"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {output}")


if __name__ == "__main__":
    main()
