#!/usr/bin/env python3
"""Download the exact FineWeb-Edu 10BT and UltraChat-200k snapshots used by configs.

The files are stored under data/raw so the *_local.yaml configs can train without
redownloading them. FineWeb-Edu 10BT is large; ensure adequate disk capacity.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def download(repo_id: str, local_dir: Path, allow_patterns: list[str]):
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    print(f"downloaded {repo_id} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--fineweb", action="store_true")
    ap.add_argument("--ultrachat", action="store_true")
    args = ap.parse_args()

    if not args.fineweb and not args.ultrachat:
        args.fineweb = args.ultrachat = True

    root = Path(args.root)
    if args.fineweb:
        download(
            "HuggingFaceFW/fineweb-edu",
            root / "fineweb-edu-10bt",
            ["sample/10BT/*.parquet", "README.md"],
        )
    if args.ultrachat:
        download(
            "HuggingFaceH4/ultrachat_200k",
            root / "ultrachat_200k",
            ["data/*.parquet", "README.md"],
        )


if __name__ == "__main__":
    main()
