#!/usr/bin/env python3
"""Download a Kaggle dataset to a local directory.

Examples:
    python download_kaggle_dataset.py
    python download_kaggle_dataset.py --dataset carlosgdcj/genius-song-lyrics-with-language-information --dest ./data
    python download_kaggle_dataset.py --dest ./data --no-unzip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"


def ensure_kaggle_credentials() -> None:
    """Ensure Kaggle API credentials exist in a standard file or via env vars."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    access_token = kaggle_dir / "access_token"

    if kaggle_json.exists():
        return

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        os.chmod(kaggle_json, 0o600)
        return

    if access_token.exists():
        token = access_token.read_text().strip()
        if username and token:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_json.write_text(json.dumps({"username": username, "key": token}))
            os.chmod(kaggle_json, 0o600)
            return
        raise FileNotFoundError(
            "Found ~/.kaggle/access_token, but Kaggle also needs KAGGLE_USERNAME.\n"
            "Set KAGGLE_USERNAME and re-run, or create ~/.kaggle/kaggle.json with both fields."
        )

    raise FileNotFoundError(
        "Kaggle credentials not found.\n"
        "1) Install the Kaggle API: pip install kaggle\n"
        "2) Download your Kaggle API token from https://www.kaggle.com/settings\n"
        "3) Save it as ~/.kaggle/kaggle.json\n"
        "   or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
        "   A raw ~/.kaggle/access_token file is also supported if KAGGLE_USERNAME is set."
    )


def install_kaggle_cli() -> None:
    """Install the Kaggle CLI if it is not already available."""
    if shutil.which("kaggle"):
        return

    user_bin = Path.home() / ".local" / "bin"
    kaggle_user_bin = user_bin / "kaggle"
    if kaggle_user_bin.exists():
        os.environ["PATH"] = f"{user_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        return

    print("Kaggle CLI not found. Installing via pip in the user site...", file=sys.stderr)
    install_cmds = [
        [sys.executable, "-m", "pip", "install", "--user", "kaggle"],
        [sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "kaggle"],
    ]

    last_error = None
    for cmd in install_cmds:
        try:
            subprocess.run(cmd, check=True)
            if kaggle_user_bin.exists() or shutil.which("kaggle"):
                os.environ["PATH"] = f"{user_bin}{os.pathsep}{os.environ.get('PATH', '')}"
                return
        except subprocess.CalledProcessError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Kaggle CLI installation did not succeed.")


def download_dataset(dataset: str, dest: str, unzip: bool) -> Path:
    ensure_kaggle_credentials()
    install_kaggle_cli()

    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(destination),
        "--force",
    ]
    if unzip:
        cmd.append("--unzip")

    print(f"Downloading dataset '{dataset}' to '{destination}'...")
    subprocess.run(cmd, check=True)

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset using the Kaggle API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset identifier in the form 'owner/dataset-name'.",
    )
    parser.add_argument(
        "--dest",
        default="./data",
        help="Directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Keep the archive as a .zip file instead of extracting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_dataset(args.dataset, args.dest, unzip=not args.no_unzip)
    except Exception as exc:  # pragma: no cover - simple CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
