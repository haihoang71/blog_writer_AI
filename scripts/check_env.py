"""
scripts/check_env.py
─────────────────────
Ensure a `.env` file exists before `make up` proceeds; copy it from
`.env.example` (and stop) if it's missing.

This used to be a shell `if [ ! -f .env ]; then ...; fi` block written
directly in the Makefile. That works fine on macOS/Linux, but GNU Make on
Windows runs recipes through `cmd.exe` by default (not bash), and cmd.exe
chokes on POSIX conditional syntax — hence the
`! was unexpected at this time.` error. Python already runs identically
on Windows/macOS/Linux (and is a hard requirement of this whole project
anyway), so doing this check here sidesteps shell portability entirely.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
env_path = ROOT / ".env"
example_path = ROOT / ".env.example"


def main() -> int:
    if env_path.exists():
        return 0

    if not example_path.exists():
        print("ERROR: neither .env nor .env.example found in the project root.")
        return 1

    shutil.copy(example_path, env_path)
    print("No .env found — copied .env.example to .env.")
    print("Fill in your real API keys in .env, then run 'make up' again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
