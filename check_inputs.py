"""Verify every bundled molecular input before launching dynamics."""

from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parent


def main():
    checks = json.loads((ROOT / "input_checksums.json").read_text())
    for name, digest in checks.items():
        path = ROOT / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Missing or changed molecular input: {name}")
    print(
        f"Verified {len(checks)} force-field, topology, structure and initial-ensemble files."
    )


if __name__ == "__main__":
    main()
