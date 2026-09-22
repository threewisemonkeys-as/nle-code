#!/usr/bin/env python3
"""The company this run is in: what else has played this game, and for how much.

    .env-venv/bin/python tools/humans.py            # what is on this disk

Craftax's sibling of this file decodes six human trajectories, because six is all
that exists for that environment. NetHack is thirty-eight years old and the record is
the other way round — there is far more of it than any page can hold — so this reads
the two things that are actually comparable and is honest about which is which.

**What is on this disk is the bot, not the people.** `nle_data/nld-aa-taster/` is the
public taster of NLD-AA (Hambro et al., NeurIPS 2022 D&B): 1,934 complete AutoAscend
games, each with NetHack's own `xlogfile` line — final score, game turns, deepest
level, role, and what killed it. Those are real trajectories, run by the symbolic bot
that won the NeurIPS 2021 NetHack Challenge, and they give the page a cloud of real
endpoints to put a run against rather than a single quoted number.

**The human set is published only.** NLD-NAO is 1,511,228 games by ~48,454 people on
nethack.alt.org, and it is not on this disk and is ~100 GB if it were. What this
carries instead is the paper's own summary of it (F10), as three numbers: the median
game scores **836**, runs **3,766 turns**, and costs **1,724 keypresses**. That is the
line to beat, and it is a better calibration than any six trajectories would be —
half a million people is not a sample, it is the population.

**Why turns and not keystrokes.** The xlogfile records game turns and not keypresses,
so game turns is the one axis on which the bot's 1,934 games and this run's trajectory
are the same measurement. Keystrokes is the axis the *budget* is denominated in, and
the page offers both — but in keystrokes only the published medians can stand beside
a run, because the ttyrec frame count is redraws and not keys (measured: 49,459 frames
for a 9,717-turn game, where the published median is 1.38 keys a turn).
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# Where `nle_data/` sits. The taster is a checkout inside the parent repository and
# this harness is its own, so the link between them is a path and not an import.
DATA = Path(os.environ.get("NLE_DATA") or Path.home() / "bai" / "nle_data")

# F10, from the dataset paper. Keys, turns and score for the median game of each
# population. Quoted and not measured here: NLD-NAO is not on this disk.
PUBLISHED = [
    {
        "name": "median human game",
        "who": "NLD-NAO — 1,511,228 games by ~48,454 people on nethack.alt.org",
        "score": 836, "turns": 3766, "keys": 1724, "games": 1511228,
        "human": True,
    },
    {
        "name": "median AutoAscend game",
        "who": "NLD-AA — the symbolic bot that won the NeurIPS 2021 Challenge",
        "score": 5422, "turns": 20414, "keys": 28181, "games": 109545,
        "human": False,
    },
]

# F13, measured by `tools/baselines.py` over 5 seeds at this harness's own rules. The
# floor a session's number is read against is `wander`, not `random`: uniform-random
# over the real keyboard is a *worse* floor than random movement, because most of the
# keys open something modal and a random policy never closes it.
FLOORS = [
    {"name": "noop (`.` only)", "score": 0, "turns": 956, "keys": 3000},
    {"name": "random (all 118 keys)", "score": 0, "turns": 37, "keys": 3000},
    {"name": "blind (8 compass keys)", "score": 4, "turns": 172, "keys": 3000},
    {"name": "wander (compass + cr/esc/y + travel)", "score": 158, "turns": 5732,
     "keys": 3000},
]


def xlog(line: str) -> dict:
    """One `xlogfile` line: tab-separated `key=value`, NetHack's own format."""
    out = {}
    for field in line.strip().split("\t"):
        key, _, value = field.partition("=")
        out[key] = value
    return out


def games(root: Path | None = None) -> list[dict]:
    """Every complete AutoAscend game on this disk, from the xlogfiles.

    Cheap enough to do on every build — 1,934 lines of text — so there is no cache to
    go stale. A game that never wrote a score is dropped rather than counted as zero:
    the tombstone frame has no status line (F12) and a truncated record is not a
    finished game.
    """
    where = (root or DATA) / "nld-aa-taster" / "nle_data"
    if not where.is_dir():
        return []
    out = []
    for path in sorted(where.glob("*/*.xlogfile")):
        for line in path.read_text(errors="replace").splitlines():
            got = xlog(line)
            if not got.get("points") or not got.get("turns"):
                continue
            out.append({
                "score": int(got["points"]),
                "turns": int(got["turns"]),
                "depth": int(got.get("maxlvl", 1)),
                "role": got.get("role", ""),
                "death": got.get("death", ""),
            })
    return out


def quantiles(values: list[int]) -> dict:
    """Median and the outer deciles. NetHack scores are Zipfian — the NLD-NAO mean is
    127,218 against a median of 836 — so a mean here would say nothing about a typical
    game, and the page quotes neither without the spread beside it.
    """
    if not values:
        return {}
    ordered = sorted(values)
    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "n": len(ordered),
        "p10": at(0.10), "p25": at(0.25), "median": int(statistics.median(ordered)),
        "p75": at(0.75), "p90": at(0.90), "max": ordered[-1],
    }


def company(root: Path | None = None) -> dict:
    """Everything the page draws behind a run, in one bundle."""
    from nao import load as nao_load  # noqa: PLC0415
    from progression import (  # noqa: PLC0415
        BOARD, BOARD_MEDIAN, BOARD_N, CEILING, episodes, rungs,
    )

    played = games(root)
    # The scatter is every game, because 1,934 points is 25 KB of JSON and thinning it
    # would be hiding the shape of the cloud to save nothing.
    return {
        "published": PUBLISHED,
        "floors": FLOORS,
        # The other agents, on the one axis that was built to compare agents. Kept
        # apart from the populations above because it is a different kind of claim:
        # those are games of NetHack, and these are runs of a language-wrapped NLE.
        "board": [{"name": name, "prog": value} for name, value in BOARD],
        "board_n": BOARD_N,
        "board_median": BOARD_MEDIAN,
        # The highest score the metric can actually produce. Not 100: the ladder's
        # ascension entries are message strings the NetHackChallenge progress system
        # never looks up, so a won game scores about 80%.
        "ceiling": CEILING,
        # The board's individual episodes, where a submission shipped them. These are
        # the only other trajectories that carry the progression metric natively, and
        # they carry nothing else — no score, no turns — so the page draws them on the
        # progression axis alone.
        "agents": episodes(root),
        # The dungeon rung by dungeon level, so the page can place a cloud of games
        # that carry a depth and nothing else. 54 numbers, against a fourth column on
        # every one of 5,934 rows.
        "dlvl_rung": [round(rungs(level, 1)[0], 2) for level in range(54)],
        "autoascend": {
            # Fourth column is "did this game ascend", to match the human rows so the
            # page can draw both clouds with one code path. It is 0 on every one of
            # these 1,934: the bot that won the NeurIPS challenge never won a game
            # here, and its deepest is dungeon level 16.
            "games": [[g["turns"], g["score"], g["depth"],
                       1 if g["death"].startswith("ascended") else 0] for g in played],
            "ascended": sum(1 for g in played if g["death"].startswith("ascended")),
            "score": quantiles([g["score"] for g in played]),
            "turns": quantiles([g["turns"] for g in played]),
            "depth": quantiles([g["depth"] for g in played]),
        },
        # Real human games, measured rather than quoted — see `nao.py`. Empty where
        # the archive has not been fetched, which is the state the page was in before
        # and still renders: the published median stays either way.
        "nao": nao_load(root),
        "where": str((root or DATA) / "nld-aa-taster"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="humans", description=__doc__.splitlines()[0])
    parser.add_argument("where", nargs="?", type=Path, default=DATA,
                        help=f"a directory holding nld-aa-taster/ (default {DATA})")
    parser.add_argument("--json", type=Path, help="write the bundle")
    args = parser.parse_args()
    # Deferred: `nao` imports this module, so a top-level import here would be a cycle.
    import nao  # noqa: PLC0415

    got = company(args.where.expanduser())
    bot = got["autoascend"]
    if not bot["score"]:
        print(f"humans: no xlogfiles under {got['where']} — the page will have only "
              f"the published medians behind it")
    else:
        print(f"{bot['score']['n']} AutoAscend games under {got['where']}")
        for name, dist in (("score", bot["score"]), ("turns", bot["turns"]),
                           ("dlvl", bot["depth"])):
            print(f"  {name:>6}  p10 {dist['p10']:>7}  median {dist['median']:>7}  "
                  f"p90 {dist['p90']:>7}  max {dist['max']:>8}")
    people = got["nao"]
    if not people:
        print(f"\nno {nao.SOURCE} under {args.where} — no measured human cloud; "
              f"run `tools/nao.py` after fetching it")
    else:
        print(f"\n{people['games']:,} human games of NetHack {people['version']}, "
              f"measured (see tools/nao.py):")
        for name in ("score", "turns", "depth"):
            dist = people[name]
            print(f"  {name:>6}  p10 {dist['p10']:>7,}  median {dist['median']:>7,}  "
                  f"p90 {dist['p90']:>7,}  max {dist['max']:>12,}")
        print(f"  progression {people['prog']['median']}% median, "
              f"{people['prog']['max']}% max — dungeon rung only, so a lower bound")

    board = got["agents"]
    if board:
        best = max(board, key=lambda one: one["prog"])
        print(f"\n{len(board)} BALROG episodes carry progression natively; the best "
              f"single one is {best['prog']}% ({best['who']})")

    print("\npublished medians (F10), quoted and not measured here:")
    for row in got["published"]:
        print(f"  {row['name']:<24} {row['score']:>6} points  "
              f"{row['turns']:>7} turns  {row['keys']:>7} keys  "
              f"over {row['games']:,} games")
    if args.json:
        args.json.write_text(json.dumps(got, indent=2))
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
