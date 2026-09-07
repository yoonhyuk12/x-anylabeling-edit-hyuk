"""Download pinned SafetyVision v2 weights and verify their SHA-256 hashes."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen


REVISION = "56a71758b55f0e9f2b4b2d6b51a779a1f882da10"
BASE_URL = (
    "https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/"
    f"{REVISION}/v2"
)
FILES = {
    "best_896.onnx": (
        44926046,
        "b250353639e01800f9cbe79c6002b8b041bdae7560328b8e18ad4a42dc3844e1",
    ),
    "best.pt": (
        22547434,
        "7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a",
    ),
}
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "weights" / "safetyvision-v2"


def download(name):
    size, expected_hash = FILES[name]
    target = OUTPUT / name
    if target.exists():
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size == size and digest == expected_hash:
            print(f"Verified existing file: {name}", flush=True)
            return
        raise RuntimeError(
            f"Existing file failed verification: {target}. "
            "Move it aside before retrying."
        )

    temporary_path = None
    try:
        print(f"Downloading {name} ({size:,} bytes)...", flush=True)
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=OUTPUT, delete=False) as stream:
            temporary_path = Path(stream.name)
            with urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
        if (
            temporary_path.stat().st_size != size
            or digest.hexdigest() != expected_hash
        ):
            raise RuntimeError(f"Download failed verification: {name}")
        temporary_path.replace(target)
        print(f"Downloaded and verified: {name}", flush=True)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-pt", action="store_true",
        help="Also download the original PyTorch checkpoint.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    download("best_896.onnx")
    if args.include_pt:
        download("best.pt")
    print("Ready. Load examples/detection/no_harness/no_harness.yaml in X-AnyLabeling.")


if __name__ == "__main__":
    main()
