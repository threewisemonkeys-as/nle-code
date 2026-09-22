#!/usr/bin/env python3
"""Real human games, measured — nethack.alt.org's own record of what people did.

    .env-venv/bin/python tools/nao.py           # what the archive says
    .env-venv/bin/python tools/nao.py --rebuild # re-distil the cache

**Why this file exists.** `humans.py` carries the NLD-NAO median as a *quoted* triple
(836 points, 3,766 turns, 1,724 keys) because that dataset is ~100 GB and is not on
this disk. That single point is all the page had behind it on the human side: a line,
not a cloud. nethack.alt.org publishes its raw xlogfiles — every game the server ever
recorded, 8.68 million of them, in under 100 MB compressed — and one of those files
holds 2,083,672 games of **NetHack 3.6.6**, which is the exact version this harness
plays. That is a bigger, version-matched population than the quoted one, and it is
measured here rather than cited.

**The published median is not reproduced here, and should not be.** NLD-NAO is a
different population: 1,511,228 games collected around 2020, overwhelmingly 3.4.3 and
early 3.6, with a filtering rule its authors did not publish. This file measures
3.6.6 only. The two numbers sit beside each other on the page, each labelled as what
it is — one quoted from a paper, one measured from an archive — and the gap between
them is a fact about two populations, not an error in either.

**The filter, and why it is the least arbitrary one available.** 1,691,863 of those
2,083,672 games — **81%** — have `turns=1`: someone connected, the character was
generated, and they quit without taking an action. `death=quit` accounts for
1,558,053 of them. Those are not games, and leaving them in puts the median human
score at **0** and the median human game at **one turn**, which would be a
nonsense line to draw under a run. So `turns > 1` — the character did something —
is the cut. It is mechanical rather than tuned, and the sensitivity is not hidden:

        raw          2,083,672 games   median score     0   median turns     1
        turns > 1      391,809 games   median score   369   median turns 1,080
        turns > 100    302,484 games   median score   657   median turns 1,661
        turns > 1000   202,817 games   median score 1,375   median turns 2,603

Every threshold above the first tells a similar story; the first tells a different
one, because below it the population is not playing.

**What cannot be measured here.** The xlogfile has no experience level — the field set
is NetHack's own 26 and `xplevel` is not among them, exactly as in the AutoAscend
taster. So a human game's *progression* can only be read off its dungeon rung, which
makes it a **lower bound**: progression is the higher of the two rungs and we can see
one. The page says so wherever the cloud is drawn. It is still worth drawing — the
bound is tight for the deep games, which are the ones a run is measured against.

**People do win, and the metric cannot say so.** 3,279 of these games — 0.84% —
ended `death=ascended`, which is the ladder's own definition of 100%. Not one of them
scores 100%, and none ever could: BALROG's NetHack progress system only ever looks up
`Dlvl:n` and `Xp:n`, so the ladder's `Astral Plane` and `You ascend t` entries are
never consulted and **80.68% is the ceiling** (see `progression.py`). Through the
depth rung an ascension scores between 78.59% and 80.68%, median 80.18%.

Two consequences the page has to carry rather than hide. The thin band at the top of
the human cloud is mostly winners. And because you finish NetHack from the Sanctum
rather than from the bottom, **1,963 of the 3,279 ascensions never reached dungeon
level 50** — so a depth-only reading ranks the 249 people who died on level 50 above
them. That is why the ascended games are drawn as their own mark at 100%: the flag
comes from `death=`, which is a fact in the record, not from a rung.

**Provenance.** `altorg-xlogfile-nh363+-partial-20260805.xz`, 29,457,720 bytes, from
https://archive.alt.org/archive/ — md5 of the uncompressed content
`ea41909b0004a4cebc82d6768af424ec`, which `--check` verifies. The archive's own
manifest carries three traps this file is built around: filenames name a *server era*
and not a version (this one holds 3.6.3, 3.6.4, 3.6.6 and 3.6.7, so `version=` is the
only way to select), the `*dev*` files are excluded from NAO's published statistics
and are indistinguishable by content, and the oldest archive (`nh343`) is
colon-separated rather than tab-separated. Only the one file is read here, filtered on
`version=`, and it is not a dev file.
"""

import argparse
import collections
import hashlib
import json
import lzma
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from humans import DATA, quantiles, xlog  # noqa: E402
from progression import CEILING, rungs  # noqa: E402

# The one archive file read here, and the version selected out of it. The file holds
# four versions; 3.6.6 is the one `nle==1.3.0` plays, so it is the one that can stand
# behind this run without an asterisk about rule changes.
SOURCE = "altorg-xlogfile-nh363+.xz"
VERSION = "3.6.6"
SOURCE_MD5 = "ea41909b0004a4cebc82d6768af424ec"  # of the *uncompressed* content
# The compressed size the manifest publishes. Checking it is free and catches the
# failure that actually happens — a partial download, or the wrong file — where the
# md5 above costs 12 s because it has to expand 1.16 GB to see anything.
SOURCE_BYTES = 29_457_720

# A game is a game if the character acted. See the docstring: this is the whole of the
# filtering policy, and 81% of the raw file fails it.
MIN_TURNS = 2

# How many games the page draws. The full 391,809 would be ~12 MB of JSON on top of a
# page that is already 13.7 MB, and a scatter that dense is a filled rectangle rather
# than a distribution. The quantiles below are computed over *every* game, so the
# sample decides only what the eye sees, never what the page claims.
SAMPLE = 4000
SAMPLE_SEED = 0

CACHE = "nao-366.json"
# Bumped whenever the shape of the cache changes. Without it a cache written by an
# older build loads cleanly and feeds the page rows of the wrong width — the archive
# has not changed, so the size key alone would never notice.
SCHEMA = 2


def distil(source: Path, sample: int = SAMPLE) -> dict:
    """Read the archive once and boil it down to what the page needs.

    Streams the xz rather than decompressing it: the file is 29 MB on disk and 1.16 GB
    open, and there is no reason for the expanded form to exist.
    """
    raw = kept = 0
    scores: list[int] = []
    turns: list[int] = []
    depths: list[int] = []
    ascended = 0
    # Reservoir sampling, seeded, so the same archive always yields the same cloud and
    # a rebuilt page does not reshuffle its dots for no reason.
    rng = random.Random(SAMPLE_SEED)
    reservoir: list[list[int]] = []

    with lzma.open(source, "rt", errors="replace") as handle:
        for line in handle:
            got = xlog(line)
            if got.get("version") != VERSION:
                continue
            raw += 1
            try:
                turn = int(got["turns"])
                score = int(got["points"])
                depth = int(got.get("maxlvl", 1))
            except (KeyError, ValueError):
                continue
            if turn < MIN_TURNS:
                continue
            kept += 1
            scores.append(score)
            turns.append(turn)
            depths.append(depth)
            # `death=ascended...` is NetHack's own record of a won game, and it is the
            # only way to know: the rung cannot say it (see `progression.py` — the
            # ladder's `You ascend t` key is never looked up by the NetHackChallenge
            # progress system, so 80.68% is the ceiling any level can reach).
            won = got.get("death", "").startswith("ascended")
            if won:
                ascended += 1
            row = [turn, score, depth, 1 if won else 0]
            if len(reservoir) < sample:
                reservoir.append(row)
            else:
                spot = rng.randrange(kept)
                if spot < sample:
                    reservoir[spot] = row

    if not kept:
        return {}
    # The dungeon rung of every game, which is the *lower bound* on its progression:
    # the experience rung is unknowable from an xlogfile.
    floor = [rungs(d, 1)[0] for d in depths]
    return {
        "source": source.name,
        "source_bytes": source.stat().st_size,
        "schema": SCHEMA,
        "version": VERSION,
        "raw_games": raw,
        "games": kept,
        "dropped": raw - kept,
        "min_turns": MIN_TURNS,
        "ascended": ascended,
        "ascended_pct": round(100 * ascended / kept, 2),
        "score": quantiles(scores),
        "turns": quantiles(turns),
        "depth": quantiles(depths),
        # Exact, over every game: what share never got past each dungeon level.
        "depth_cdf": depth_cdf(depths),
        "prog": {
            "median": round(statistics.median(floor), 2),
            "mean": round(statistics.mean(floor), 2),
            "max": round(max(floor), 2),
            "at_zero_pct": round(100 * sum(1 for v in floor if v == 0) / kept, 2),
        },
        # Sorted so the JSON diffs sanely between rebuilds of the same archive.
        "sample": sorted(reservoir),
    }


def depth_cdf(depths: list[int]) -> list[float]:
    """Share of games that got no deeper than each dungeon level, 0..53.

    Over every game, not the drawn sample — so the page can say what share of real
    human games a run is past without that claim resting on 4,000 dots. Cheap to ship:
    54 floats.

    **Keyed on the dungeon level and never on the rung.** The rung for level 25 is
    46.63763…, which every panel prints as 46.64, and counting games whose rung beats
    "46.64" drops all 589 games that ended on exactly level 25 — a tenth of the answer,
    lost to a comparison against a rounded display value. Integers do not have that
    failure mode.
    """
    total = len(depths)
    counts = collections.Counter(depths)
    out, running = [], 0
    for level in range(54):
        running += counts.get(level, 0)
        out.append(round(100 * running / total, 3))
    return out


def load(root: Path | None = None, rebuild: bool = False) -> dict:
    """The distilled archive, rebuilt when the cache is missing or the source moved.

    Distilling costs ~50 s, and the watcher rebuilds the page every 15 minutes, so a
    cache is not an optimisation here — without one the archive would be re-read a
    hundred times a day to produce a file that cannot have changed.
    """
    where = root or DATA
    source = where / SOURCE
    if not source.exists():
        return {}
    cache = where / CACHE
    if not rebuild and cache.exists():
        try:
            got = json.loads(cache.read_text())
        except json.JSONDecodeError:
            got = {}
        # Keyed on the source's size: the archive is published as an immutable
        # snapshot, so a different size is a different file and not an edit. And on the
        # schema, because a cache of the right archive in the wrong shape is worse than
        # no cache — it loads without complaint.
        if (got.get("source_bytes") == source.stat().st_size
                and got.get("schema") == SCHEMA):
            return got
    got = distil(source)
    if got:
        cache.write_text(json.dumps(got, separators=(",", ":")))
    return got


def check(root: Path | None = None) -> tuple[bool, str]:
    """Verify the archive against the md5 the manifest publishes."""
    source = (root or DATA) / SOURCE
    if not source.exists():
        return False, f"{source} is not here"
    digest = hashlib.md5()  # noqa: S324 — integrity against the manifest, not security
    with lzma.open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    ok = got == SOURCE_MD5
    return ok, (f"{got} — matches the manifest" if ok
                else f"{got} — the manifest says {SOURCE_MD5}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="nao", description=__doc__.splitlines()[0])
    parser.add_argument("where", nargs="?", type=Path, default=DATA,
                        help=f"a directory holding {SOURCE} (default {DATA})")
    parser.add_argument("--rebuild", action="store_true", help="re-distil the cache")
    parser.add_argument("--check", action="store_true",
                        help="md5 the archive against the published manifest")
    args = parser.parse_args()
    where = args.where.expanduser()

    if args.check:
        ok, said = check(where)
        print(said)
        return 0 if ok else 1

    got = load(where, rebuild=args.rebuild)
    if not got:
        print(f"nao: no {SOURCE} under {where} — the page keeps the published median "
              f"and draws no human cloud")
        return 1

    print(f"{got['games']:,} human games of NetHack {got['version']}, from "
          f"{got['source']}")
    print(f"  {got['raw_games']:,} in the file; {got['dropped']:,} dropped for "
          f"turns < {got['min_turns']} (a character that never acted)")
    for name in ("score", "turns", "depth"):
        dist = got[name]
        print(f"  {name:>6}  p10 {dist['p10']:>7,}  median {dist['median']:>7,}  "
              f"p90 {dist['p90']:>7,}  max {dist['max']:>12,}")
    prog = got["prog"]
    print(f"  progression (dungeon rung only, a lower bound): median "
          f"{prog['median']}%, mean {prog['mean']}%, max {prog['max']}%")
    print(f"    {prog['at_zero_pct']}% never left dungeon level 1")
    won = [row for row in got["sample"] if row[3]]
    print(f"  {got['ascended']:,} ascensions ({got['ascended_pct']}%) — the ladder's "
          f"own 100%, which the metric scores at most {CEILING}%")
    if won:
        rung = [rungs(row[2], 1)[0] for row in won]
        print(f"    of the drawn sample, {len(won)} ascended, scoring "
              f"{min(rung):.2f}%–{max(rung):.2f}% on the depth rung")
    print(f"  {len(got['sample']):,} games sampled for the page's scatter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
