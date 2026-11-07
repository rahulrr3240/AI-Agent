"""Load the 28 resolved history incidents into a Hindsight bank.

Idempotent by construction: each incident uses document_id=incident-INC-014, so
re-running replaces rather than duplicates. Still, seeding a bank that already
holds data requires --force, to avoid quietly mixing runs.

  python scripts/seed_memory.py            # seed (or verify existing)
  python scripts/seed_memory.py --force    # re-seed over existing data
  python scripts/seed_memory.py --reset    # delete the bank, then seed
  python scripts/seed_memory.py --bank oncall-demo-2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oncall.config import load_settings  # noqa: E402
from oncall.memory import MemoryUnavailable, build_memory  # noqa: E402
from oncall.schemas import Incident  # noqa: E402
from oncall.tools import SimulatedWorld  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-seed even if the bank has data")
    parser.add_argument("--reset", action="store_true", help="delete the bank first")
    parser.add_argument("--bank", default=None, help="override HINDSIGHT_BANK_ID")
    parser.add_argument("--no-verify", action="store_true", help="skip the recall spot-check")
    args = parser.parse_args()

    settings = load_settings()
    bank_id = args.bank or settings.hindsight_bank_id
    try:
        memory, warning = build_memory(settings, bank_id=bank_id)
    except MemoryUnavailable as exc:
        print(f"FAIL  {exc}")
        print("      Set HINDSIGHT_BASE_URL / HINDSIGHT_API_KEY in .env, then retry.")
        return 1

    print(f"Bank:    {bank_id}")
    print(f"Backend: {memory.name}")
    if warning:
        print(f"WARNING: {warning}")

    if args.reset:
        if not args.force:
            print("Refusing to reset without --force (--reset deletes the bank's memories).")
            return 1
        memory.delete_bank()
        print("Deleted the existing bank.")

    try:
        existing = memory.stats()
    except MemoryUnavailable as exc:
        print(f"FAIL  cannot read bank stats: {exc}")
        return 1
    if existing.get("incidents", 0) > 0 and not args.force:
        print(f"Bank already holds {existing['incidents']} incident(s) / "
              f"{existing.get('memories', 0)} memories. Use --force to re-seed.")
        return 1

    world = SimulatedWorld(settings.data_dir)
    incidents = world.history()
    print(f"Seeding {len(incidents)} resolved incidents...")
    failures = 0
    for index, raw in enumerate(incidents, start=1):
        incident = Incident.model_validate(raw)
        try:
            memory.retain_incident(incident)
        except MemoryUnavailable as exc:
            failures += 1
            print(f"  {incident.id}: FAILED — {exc}")
        if index % 7 == 0 or index == len(incidents):
            print(f"  {index}/{len(incidents)} retained")

    if failures:
        print(f"FAIL  {failures} incident(s) were not retained.")
        return 1

    stats = memory.stats()
    print(f"Stored: {stats['incidents']} incidents, {stats['memories']} memories.")

    if not args.no_verify:
        print("Recall spot-check (does the matching past incident come back?):")
        checks = 0
        passed = 0
        for raw in [incidents[3], incidents[12], incidents[17]]:  # redis, TLS, slow query
            incident = Incident.model_validate(raw)
            hits = memory.recall_similar(
                alert=incident.alert,
                service=incident.service,
                symptoms=incident.symptoms,
                logs_excerpt=world.log_excerpt(incident.id),
                k=5,
                as_of=incident.opened_at,
            )
            ids = [(hit.metadata or {}).get("incident_id", "") for hit in hits]
            ok = incident.id in ids
            checks += 1
            passed += int(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {incident.id} -> {ids or 'no hits'}")
        if not checks or passed < checks:
            print("WARNING: recall did not return every known incident. This is honest signal — "
                  "inspect the recall query or seeding before the demo.")
    memory.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
