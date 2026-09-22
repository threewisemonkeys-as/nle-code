#!/usr/bin/env python3
"""How far through NetHack a run got, on an axis calibrated against human games.

    .env-venv/bin/python tools/progression.py --depth 25 --xp 5

**Why score is the wrong instrument, in the words of the people who built this
metric.** NetHack's own score rewards kills, gold, items and food, and the BALROG
authors' objection to using it as progress is the one this run ran into: *"players can
win the game with scores ranging from a few hundred thousand to several million
points"* — so a score says how a game went, not how far through it got. Our best life
scores 7,622 at dungeon level 25 while the bot's best games on this disk score 250,914
without passing level 16. Both numbers are true and they order the two runs opposite
ways.

**What this measures instead.** BALROG built a ladder from a dataset of human-played
NetHack games where, in their words, *"each data point represents the probability of a
human player winning the game after reaching a specific dungeon level or experience
level."* Dungeon level 1 and experience level 1 are 0%; ascension is 100%. A run's
progression is the **higher of its two rungs** — the dungeon level it reached and the
experience level it reached — so a character who dug deep without levelling and one who
levelled without descending are both credited for what they did do.

Keeping both rungs visible is the point of `rungs()`. The max alone flatters this run
badly: it reached dungeon level 25, which is 46.6% of the way to an ascension by this
measure, at experience level 5, which is 2.9%. The gap is the finding, and a single
number hides it.

**Provenance, and why the ladder is vendored.** `tools/progression.json` is copied
verbatim from `balrog/environments/nle/achievements.json` (BALROG at 38bb52f), and
`BOARD` is a snapshot of the published NLE column of its leaderboard. Both are copied
rather than imported: this harness is its own checkout and has to score a run on a box
where the parent repository is not present. The cost of copying is that the snapshot
can fall behind, so `--board` re-reads the live results directory when it is there and
says whether the two still agree.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = json.loads((HERE / "progression.json").read_text())

# The published NLE column of the BALROG leaderboard: the mean progression each
# submission reached, over 5-40 independent episodes. Snapshotted from
# `nle_data/balrog_results/submissions/*/*/summary.json`, 44 submissions.
#
# These are not a like-for-like target and the page says so. BALROG plays a
# *language-wrapped* NLE — the screen is rendered to a textual description and the
# agent issues high-level commands like `wait`, `apply`, `north`, with menus and
# `--More--` handled by the wrapper — where an action here is one keystroke. Its
# episodes are capped at 500 of those steps, and each is independent: a fresh
# character in a fresh dungeon with no memory of the last. A run here is one dungeon,
# one character, and `notes.md` carried between lives. The comparison worth making is
# to a *first* life, and then to what the run climbed to.
BOARD = [
    ("gemini-3-pro", 6.77),
    ("gemini-3-flash", 3.96),
    ("gemini-3.1-pro", 3.03),
    ("gemini-3.1-pro-thinking", 2.60),
    ("claude-opus-4.5-thinking", 2.39),
    ("claude-opus-4.5", 2.03),
    ("grok-4", 1.76),
    ("Gemini-2.5-Pro", 1.69),
    ("DeepSeek-R1 (robust CoT)", 1.38),
    ("gpt-5-minimal", 1.32),
    ("Claude-3.5-Sonnet (VLM)", 1.16),
]
BOARD_N = 44
BOARD_MEDIAN = 0.37
BOARD_WHERE = "BALROG leaderboard, NLE column"

# Where the live results sit, for `--board` to check the snapshot against.
RESULTS = Path(os.environ.get("NLE_DATA") or Path.home() / "bai" / "nle_data")


# What no run can score. `LADDER` holds 87 entries and only 80 of them are levels: the
# other seven are message strings — `Home 1`..`Home 5`, `Astral Plane` (87.46%) and
# `You ascend t` (100%). BALROG's own `Progress.update()` for NetHackChallenge builds
# its keys as `f"Dlvl:{n}"` and `f"Xp:{n}"` and looks up nothing else, so those seven
# are never consulted by the metric as it is actually computed — by them or by us.
#
# So this is the real top of the axis, for every agent on the board and for the 3,279
# human ascensions in `nao.py`'s archive alike: a won game scores about 80%, and 100%
# is unreachable rather than merely hard. The page draws the axis to 100 and says so.
CEILING = 80.68


def rungs(depth: int, xp: int) -> tuple[float, float]:
    """The two rungs, as percentages: (dungeon level, experience level).

    A level past the end of the ladder takes the deepest rung there is rather than
    falling to zero — the ladder stops at dungeon level 50 and experience level 30,
    and a run that gets past either of those has not gone backwards.

    Neither rung can reach 100: see `CEILING` above.
    """
    def rung(prefix: str, value: int, top: int) -> float:
        for candidate in range(min(value, top), 0, -1):
            if f"{prefix}:{candidate}" in LADDER:
                return 100 * LADDER[f"{prefix}:{candidate}"]
        return 0.0

    return rung("Dlvl", depth, 50), rung("Xp", xp, 30)


def progression(depth: int, xp: int) -> float:
    """One number: the higher of the two rungs, which is BALROG's own definition."""
    return max(rungs(depth, xp))


def of_lives(lives: list[dict]) -> dict:
    """A run's progression, three ways, because one way would mislead.

    * **best** — the furthest any single life got. The headline, and the number that
      is most flattered by a run whose lives share a dungeon and a notes file.
    * **mean** — over every life, which is the shape of the statistic BALROG reports
      (a mean over episodes) even though its episodes are independent and these are
      not.
    * **first** — the life played before the run had learned anything, which is the
      only one of the three that is measured under the same conditions as a fresh
      episode, and so the only one that can be set beside the board directly.

    Plus **recent**, the best of the last five lives, which is the one that answers
    whether the run is still climbing. It is a maximum and not a mean because the last
    life of a live run is one being played right now and is worth almost nothing yet —
    a mean would read a healthy run as falling off every time it started again.
    """
    if not lives:
        return {}
    scored = [progression(one["depth"], one["xp"]) for one in lives]
    # Ties broken by the *other* rung, because they are common and the tie-break is
    # what the panel is for: nine of this run's lives reached dungeon level 25 and
    # score an identical 46.64%, and the one to quote is the one that also got
    # furthest on the axis the metric is not crediting.
    best = max(range(len(scored)),
               key=lambda i: (scored[i], min(rungs(lives[i]["depth"], lives[i]["xp"]))))
    dlvl, xp = rungs(lives[best]["depth"], lives[best]["xp"])
    return {
        "best": round(max(scored), 2),
        "mean": round(sum(scored) / len(scored), 2),
        "first": round(scored[0], 2),
        "recent": round(max(scored[-5:]), 2),
        "per_life": [round(v, 2) for v in scored],
        # The two rungs of the best life, kept apart: the max alone reads a run that
        # dug to level 25 at experience level 5 as half way to an ascension.
        "depth_rung": round(dlvl, 2),
        "xp_rung": round(xp, 2),
        "depth": lives[best]["depth"],
        "xp": lives[best]["xp"],
    }


def episodes(root: Path | None = None) -> list[dict]:
    """Every individual NLE episode the board's submissions recorded, not just means.

    `live_board` reads each submission's mean; this reads the episodes behind it. 24
    of the 44 submissions shipped their per-episode files — 135 episodes — and they
    are the only other trajectories anywhere that carry *this* metric natively.

    **Three of the scalar fields in those files are dead** and one is not. `score`,
    `time`, `depth` and `experience_level` are zero in all 135, because they are read
    off the status line after the episode has ended and NetHack has torn the character
    down. What survives is `progression`, `num_steps`, and the two ladder lists — so
    that is what this returns, and there is deliberately no score here to plot.

    **`progression` in those files is a fraction, not a percentage.** The summary that
    feeds the leaderboard multiplies by 100 (0.020257699 -> 2.02576990), so this does
    too, and everything on the page is in the same units. Recomputing the rungs from
    `dlvl_list`/`xplvl_list` through `progression()` above reproduces their number on
    every one of the 135, which is the check that the vendored ladder is the live one.
    """
    where = (root or RESULTS) / "balrog_results" / "submissions"
    if not where.is_dir():
        return []
    out = []
    for path in sorted(where.glob("*/*/nle/*/*_run_*.json")):
        try:
            got = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if got.get("progression") is None:
            continue
        def top(key: str) -> int:
            return max((int(entry.split(":")[1]) for entry in got.get(key) or []),
                       default=1)
        out.append({
            "who": path.parts[-4],
            "prog": round(100 * float(got["progression"]), 2),
            "steps": int(got.get("num_steps") or 0),
            "depth": top("dlvl_list"),
            "xp": top("xplvl_list"),
        })
    return out


def live_board(root: Path | None = None) -> list[tuple[str, float]]:
    """The board as the results directory has it now, if that directory is here."""
    where = (root or RESULTS) / "balrog_results" / "submissions"
    if not where.is_dir():
        return []
    out = []
    for path in sorted(where.glob("*/*/summary.json")):
        try:
            got = json.loads(path.read_text()).get("environments", {}).get("nle")
        except json.JSONDecodeError:
            continue
        if not got or got.get("progression_percentage") is None:
            continue
        out.append((path.parent.name, float(got["progression_percentage"])))
    return sorted(out, key=lambda row: -row[1])


def main() -> int:
    parser = argparse.ArgumentParser(prog="progression",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=1, help="deepest dungeon level")
    parser.add_argument("--xp", type=int, default=1, help="highest experience level")
    parser.add_argument("--board", action="store_true",
                        help="check the snapshot against the live results directory")
    args = parser.parse_args()

    dlvl, xp = rungs(args.depth, args.xp)
    print(f"dungeon level {args.depth}: {dlvl:.2f}%")
    print(f"experience level {args.xp}: {xp:.2f}%")
    print(f"progression (the higher of the two): {max(dlvl, xp):.2f}%")

    if args.board:
        live = live_board()
        print(f"\n{BOARD_WHERE} — snapshot of {BOARD_N}, median {BOARD_MEDIAN}%")
        for name, value in BOARD[:6]:
            print(f"  {value:>5.2f}%  {name}")
        if not live:
            print(f"\nno live results under {RESULTS / 'balrog_results'} — "
                  f"the snapshot is all there is to go on")
        else:
            print(f"\nlive: {len(live)} submissions, best {live[0][1]:.2f}% "
                  f"({live[0][0]})")
            drift = abs(live[0][1] - BOARD[0][1])
            print(f"the snapshot's best is {BOARD[0][1]:.2f}% — "
                  + ("they agree" if drift < 0.01
                     else f"they differ by {drift:.2f} points, so the snapshot is stale"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
