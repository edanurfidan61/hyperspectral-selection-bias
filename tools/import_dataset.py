"""Helper to link/copy/move an external Dataset into the project `data/raw`.

Usage examples:
  python scripts/import_dataset.py "C:\bitirme\bitirme_projesi\Dataset" --mode junction
  python scripts/import_dataset.py "C:\path\to\Dataset" --mode copy --target data/raw

Modes:
  - junction (default on Windows): create a directory junction via mklink /J
  - symlink: try os.symlink (may require admin/dev mode on Windows)
  - copy: copy files with shutil.copytree
  - move: move files (destructive)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import os
from pathlib import Path
import sys

from src.core.paths import ROOT


def has_results_hdr(src: Path) -> bool:
    return any(src.glob("**/results/*.hdr"))


def create_junction(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Target exists: {dest}")
    cmd = ["cmd", "/c", "mklink", "/J", str(dest), str(src)]
    subprocess.check_call(cmd)


def create_symlink(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Target exists: {dest}")
    os.symlink(str(src), str(dest), target_is_directory=True)


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Target exists: {dest}")
    shutil.copytree(src, dest)


def move_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Target exists: {dest}")
    shutil.move(str(src), str(dest))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Source Dataset path (e.g. C:\\bitirme\\bitirme_projesi\\Dataset)")
    p.add_argument("--target", default=str((ROOT / "data" / "raw")), help="Target path inside project (default: data/raw)")
    p.add_argument("--mode", choices=("junction", "symlink", "copy", "move"), default="junction")
    args = p.parse_args(argv)

    src = Path(args.source).resolve()
    dest = Path(args.target)
    # if dest is relative, resolve inside project
    if not dest.is_absolute():
        dest = (ROOT / dest).resolve()

    if not src.exists():
        print(f"Source not found: {src}")
        return 2
    if not has_results_hdr(src):
        print(f"No results/*.hdr found under {src}. Aborting.")
        return 3

    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "junction":
            create_junction(src, dest)
        elif args.mode == "symlink":
            create_symlink(src, dest)
        elif args.mode == "copy":
            copy_tree(src, dest)
        elif args.mode == "move":
            move_tree(src, dest)
    except subprocess.CalledProcessError as e:
        print("System command failed:", e)
        return 4
    except FileExistsError as e:
        print(e)
        return 5
    except OSError as e:
        print("OS error:", e)
        return 6

    print(f"Done: mode={args.mode} src={src} -> dest={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
