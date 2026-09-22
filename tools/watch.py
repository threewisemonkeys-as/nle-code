#!/usr/bin/env python3
"""Rebuild the replay page while the run is still playing.

    .env-venv/bin/python tools/watch.py ~/agent-runs/20260911-054617 \
        --out ~/bai/cc_nle/replay.html --every 900

A 30,000-key run is about twenty hours of wall clock, and a page built at the end of
it is a page nobody can look at until then. This rebuilds it on a timer instead, so
there is always a current one on disk, and stops of its own accord when the run has
spent its budget.

Three things make that safe to leave running beside a live game.

**It only reads.** The launcher owns the workspace and the record; this opens them and
nothing else. `replay.load_record` retries a half-written `result.json` rather than
failing on it — the file is rewritten after every action, so catching it mid-write is
ordinary rather than exceptional.

**It replaces the page atomically.** `replay.render` writes beside the target and
renames, so a reader with the last build open never sees half of the next one.

**It does not die on one bad build.** A rebuild that throws is reported and the timer
carries on; the run outlasts any single transient, and a watcher that exits on the
first one is a watcher that was not there for the other nineteen hours.

The cost of a rebuild is one replay of the whole history — 3 seconds at 3,000 keys and
about 35 at 30,000 — which is why the default interval is a quarter of an hour rather
than a minute.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import replay  # noqa: E402


def once(where: Path, out: Path) -> tuple[bool, str]:
    """One rebuild. Returns (the run still has budget left, a line about it)."""
    bundle = replay.build(where)
    if not bundle["runs"]:
        return True, "nothing has been played yet"
    size = replay.render(bundle, out)
    said = " · ".join(
        f"{one['label']} {one['used']}/{one['budget']} keys, {one['turns']} turns, "
        f"best life {one['best']}, {one['lives']} lives, dlvl {one['depth']}"
        for one in bundle["runs"]
    )
    return bundle["live"], f"{said} -> {size / 1e6:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(prog="watch", description=__doc__.splitlines()[0])
    parser.add_argument("where", type=Path,
                        help="a launch directory, or one workspace inside one")
    parser.add_argument("--out", type=Path, default=Path("replay.html"))
    parser.add_argument("--every", type=int, default=900,
                        help="seconds between rebuilds (default 900)")
    parser.add_argument("--forever", action="store_true",
                        help="keep rebuilding after the run has spent its budget")
    args = parser.parse_args()

    where = args.where.expanduser().resolve()
    out = args.out.expanduser().resolve()
    print(f"watch: {where} -> {out}, every {args.every / 60:.0f} min", flush=True)

    builds = failures = 0
    while True:
        started = time.time()
        try:
            live, said = once(where, out)
            builds += 1
            print(f"[{time.strftime('%H:%M:%S')}] {said} "
                  f"({time.time() - started:.0f}s)", flush=True)
        except Exception:  # noqa: BLE001 — the point is to survive whatever it is
            failures += 1
            live = True
            print(f"[{time.strftime('%H:%M:%S')}] build failed "
                  f"({failures} so far) — carrying on:", flush=True)
            traceback.print_exc()
        if not live and not args.forever:
            print(f"watch: the run has spent its budget — {builds} builds, "
                  f"{failures} failed. The page is final.", flush=True)
            return 0
        time.sleep(max(30, args.every - (time.time() - started)))


if __name__ == "__main__":
    sys.exit(main())
