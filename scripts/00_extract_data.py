"""Selectively extract DLRSD from the project archive.

The only complete copy of DLRSD on this machine lives inside a 19 GB archive
that holds an entire unrelated research project. We pull out just the three
directories we need (~250 MB) rather than unpacking 21 GB.

Every other unpacked DLRSD copy on this machine is a partial subset with an
inconsistent train/test split, and one of them (dlrsd_new/val_masks_remapped)
holds RGB->luminance garbage instead of class ids. Use this script's output.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DEFAULT_ARCHIVE = Path("/home/cse-sdpl/Downloads/dlrsd.zip")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# archive directory -> local directory
WANTED = {
    "dlrsd/full_images/": "data/dlrsd/images",     # RGB, 256x256
    "dlrsd/full_1cmasks/": "data/dlrsd/labels",    # mode L, values 1..17
    "dlrsd/full_masks/": "data/dlrsd/colour",      # mode P, DLRSD palette
}
EXPECTED_COUNT = 2100


def extract(archive: Path, dest_root: Path, force: bool = False) -> int:
    total = 0
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for prefix, rel_dest in WANTED.items():
            dest = dest_root / rel_dest
            dest.mkdir(parents=True, exist_ok=True)
            wanted = [
                m for m in members
                if m.filename.startswith(prefix) and m.filename.lower().endswith(".png")
            ]
            if len(wanted) != EXPECTED_COUNT:
                print(f"  WARNING {prefix}: found {len(wanted)} PNGs, expected {EXPECTED_COUNT}")

            written = 0
            for m in wanted:
                out = dest / Path(m.filename).name
                if out.exists() and not force and out.stat().st_size == m.file_size:
                    continue
                out.write_bytes(zf.read(m.filename))
                written += 1
            size_mb = sum(m.file_size for m in wanted) / 1e6
            print(f"  {prefix:26s} -> {rel_dest:22s} "
                  f"{len(wanted)} files, {size_mb:7.1f} MB ({written} newly written)")
            total += written
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--dest", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="rewrite files that already exist at the right size")
    args = ap.parse_args()

    if not args.archive.is_file():
        print(f"archive not found: {args.archive}", file=sys.stderr)
        return 1

    print(f"archive: {args.archive} ({args.archive.stat().st_size / 1e9:.1f} GB)")
    written = extract(args.archive, args.dest, force=args.force)
    print(f"done, {written} files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
