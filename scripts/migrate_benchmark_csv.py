"""One-off repair to migrate benchmark CSV headers to match ROW_COLUMNS.

Copies the original to .llm_failures/<name>.bak (gitignored), then rewrites
the file with the full ROW_COLUMNS header. Rows with 25 fields get retry_of_failed=False,
and rows with 26 keep their value. Confirms row count and column count.

Usage:
    python scripts/migrate_benchmark_csv.py <path>
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oncall.evaluate import ROW_COLUMNS, migrate_csv


def migrate_file(path: Path, backup_dir: Path = Path(".llm_failures")) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Count rows before
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_before = list(reader)

    rows_before = len(raw_before)
    if rows_before == 0:
        print(f"{path} is empty, nothing to migrate.")
        return

    # Backup original to .llm_failures/<name>.bak
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.bak"
    shutil.copy2(path, backup_path)
    print(f"Backed up {path} to {backup_path}")

    # Migrate the file
    migrate_csv(path, force=True)

    # Count rows after
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_after = list(reader)

    rows_after = len(raw_after)

    # Print before and after row counts (must be equal)
    print(f"Row count before: {rows_before}")
    print(f"Row count after:  {rows_after}")
    if rows_before != rows_after:
        raise ValueError(
            f"Row count mismatch: before={rows_before}, after={rows_after}"
        )

    # Confirm every row has the full column count
    for idx, row in enumerate(raw_after):
        if len(row) != len(ROW_COLUMNS):
            raise ValueError(
                f"Row {idx} has {len(row)} columns, expected {len(ROW_COLUMNS)}"
            )

    print(f"Confirmed every row has the full column count ({len(ROW_COLUMNS)}).")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python scripts/migrate_benchmark_csv.py <path>", file=sys.stderr)
        return 1

    target = Path(args[0])
    try:
        migrate_file(target)
    except Exception as exc:
        print(f"Migration error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
