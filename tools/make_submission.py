"""Build the Canvas submission zip from the code committed at a git ref.

Layout of the zip (instruction.pdf §4):

    report.pdf          the report, renamed from --report
    code/algorithm/...  every file tracked by git under code/algorithm/ at --ref
    code/data/          an EMPTY directory (the markers add the datasets)

Only files tracked by git at --ref are read (through ``git archive``), so
untracked or generated files (datasets, results, figures, caches) can never
end up in the zip. The name is the group ID and the student IDs of all
members separated by underscores (instruction.pdf §4), for example
GroupID07_123456789_234567890_345678901_456789012.zip:

    python tools/make_submission.py --report report.pdf --group-id 07 \\
        --sids 123456789 234567890 345678901 456789012

Standard library only. The script refuses to write the zip if it would
contain datasets, arrays, images, caches or any file over 5 MB.
"""

import argparse
import io
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = "submission-code-v4"
MAX_FILE_BYTES = 5 * 1024 * 1024
# Directories whose content never belongs in the zip, even if tracked.
EXCLUDED_DIRS = ("code/results/", "code/figures/", "code/data/")
EXCLUDED_PARTS = {"__pycache__", ".DS_Store", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
# Suffixes that must never be shipped (datasets, arrays, images).
FORBIDDEN_SUFFIXES = {".pgm", ".npz", ".npy", ".bmp", ".png", ".jpg", ".jpeg",
                      ".gif", ".tif", ".tiff", ".pdf"}
ALLOWED_PDF = {"report.pdf"}  # the only PDF allowed in the zip
# Unix permissions stored in the zip: without them unzip creates files as
# 0600 and the empty data directory without the execute bit, so the markers
# could not copy the datasets into code/data/.
FILE_MODE = 0o100644
DIR_MODE = 0o040755
MS_DOS_DIRECTORY = 0x10


def parse_args(argv=None):
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True,
                        help="report PDF; stored in the zip as report.pdf")
    parser.add_argument("--group-id", required=True, help="group number, e.g. 07")
    parser.add_argument("--sids", nargs="+", required=True,
                        help="student IDs of all group members (3 or 4)")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help=f"git ref whose committed code/ is packed (default {DEFAULT_REF})")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(),
                        help="folder for the zip (default: current directory)")
    args = parser.parse_args(argv)
    if not 3 <= len(args.sids) <= 4:
        parser.error("--sids needs the IDs of all 3 or 4 group members")
    for value in [args.group_id, *args.sids]:
        if not re.fullmatch(r"[0-9A-Za-z]+", value):
            parser.error(f"invalid group or student ID: {value!r}")
    if args.report.suffix.lower() != ".pdf" or not args.report.is_file():
        parser.error(f"--report must be an existing PDF file: {args.report}")
    return args


def zip_name(group_id, sids):
    """Return the archive name required by instruction.pdf §4."""
    return "_".join([f"GroupID{group_id}", *sids]) + ".zip"


def committed_code(ref):
    """Return {path: bytes} for every regular file under code/ at ``ref``."""
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", ref, "--", "code"],
        capture_output=True, check=True,
    ).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
    return files


def keep(path):
    """Return True if a committed path belongs in the zip."""
    posix = PurePosixPath(path)
    if any(path.startswith(prefix) for prefix in EXCLUDED_DIRS):
        return False
    if EXCLUDED_PARTS & set(posix.parts) or posix.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.startswith("code/algorithm/")


def check(entries):
    """Raise SystemExit listing every entry that breaks a safety rule."""
    problems = []
    for name, data in entries.items():
        suffix = PurePosixPath(name).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES and name not in ALLOWED_PDF:
            problems.append(f"forbidden file type: {name}")
        if len(data) > MAX_FILE_BYTES:
            problems.append(f"larger than 5 MB: {name} ({len(data)} bytes)")
        if name.startswith("code/data/"):
            problems.append(f"file inside code/data/: {name}")
    if problems:
        sys.exit("Refusing to build the zip:\n  " + "\n  ".join(problems))


def _entry(name, is_dir=False):
    """Return a ZipInfo with the current time and explicit Unix permissions."""
    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.external_attr = (DIR_MODE << 16) | MS_DOS_DIRECTORY if is_dir else FILE_MODE << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build(args):
    """Write the zip and return its path."""
    entries = {"report.pdf": args.report.read_bytes()}
    entries.update({path: data for path, data in committed_code(args.ref).items()
                    if keep(path)})
    check(entries)
    if not any(name.endswith("code/algorithm/main.py") for name in entries):
        sys.exit(f"code/algorithm/main.py not found at {args.ref}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / zip_name(args.group_id, args.sids)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_entry("report.pdf"), entries.pop("report.pdf"))
        for name in sorted(entries):
            archive.writestr(_entry(name), entries[name])
        archive.writestr(_entry("code/data/", is_dir=True), b"")  # empty directory
    return path


def main(argv=None):
    """Build the zip, then print its contents and size."""
    args = parse_args(argv)
    path = build(args)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        for info in infos:
            print(f"{info.file_size:>10}  {info.filename}")
        total = sum(info.file_size for info in infos)
    print(f"{len(infos)} entries, {total / 1024:.1f} KiB uncompressed, "
          f"{path.stat().st_size / 1024:.1f} KiB zip (code from {args.ref})")
    print(path)


if __name__ == "__main__":
    main()
