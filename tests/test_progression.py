"""The human-calibrated axis: how far through the game a run got.

Score answers a different question from the one the run is being asked. NetHack pays
for kills, gold and items, so a game can score 250,914 without passing dungeon level
16 and another can score 7,622 at level 25 — both true, and they order the two runs
opposite ways. BALROG's ladder is a probability instead: given that a human reached
this dungeon level or this experience level, how often did they go on to ascend.

What these pin is the arithmetic of reading a run off that ladder, and the two places
it can quietly mislead — a single number that hides which rung is carrying it, and a
vendored copy of the ladder that has drifted from the one it was taken from.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import progression  # noqa: E402


def test_the_start_of_the_game_is_zero_and_ascension_is_one():
    """The two ends of the ladder, which are what make the middle a percentage."""
    assert progression.progression(1, 1) == 0.0
    assert progression.LADDER["Dlvl:1"] == 0.0
    assert progression.LADDER["Xp:1"] == 0.0
    ascend = [v for k, v in progression.LADDER.items() if "ascend" in k.lower()]
    assert ascend == [1.0], "ascension is what 100% means on this axis"


def test_the_ladder_only_goes_up():
    """A deeper level is never worth less than a shallower one, on either axis.

    Not a tautology about the data: the rungs are estimated from a human dataset and a
    thin bucket could have come out below the one above it, in which case a run would
    be punished for descending and `rungs`'s walk-down would return the wrong number.
    """
    for prefix, top in (("Dlvl", 50), ("Xp", 30)):
        seen = [progression.LADDER[f"{prefix}:{n}"] for n in range(1, top + 1)
                if f"{prefix}:{n}" in progression.LADDER]
        assert seen == sorted(seen), f"{prefix} goes down somewhere"


def test_a_run_is_credited_with_the_higher_of_its_two_rungs():
    """BALROG's own definition: the max, so that a character who dug deep without
    levelling and one who levelled without descending are both credited."""
    deep, shallow_xp = progression.rungs(25, 5)
    assert deep > shallow_xp
    assert progression.progression(25, 5) == deep
    # And the other way round, which is the shape of the bot's games on this disk:
    # eleven thousand turns on dungeon level 1 with a developed character.
    low, high_xp = progression.rungs(1, 14)
    assert high_xp > low
    assert progression.progression(1, 14) == high_xp


def test_no_run_can_ever_score_a_hundred_on_this_metric():
    """The ceiling, and why it is not 100.

    `LADDER` holds 87 entries and only 80 are levels. The other seven are message
    strings — `Home 1`..`Home 5`, `Astral Plane`, `You ascend t` — and BALROG's own
    `Progress.update()` for NetHackChallenge builds its lookup keys as `f"Dlvl:{n}"`
    and `f"Xp:{n}"`, so it never consults them. A game that literally ascends is
    scored about 80%, by them and by us.

    This is worth a test and not just a comment because the ladder *contains* a 100%
    entry, so anyone reading the JSON would reasonably assume it is reachable.
    """
    assert progression.LADDER["You ascend t"] == 1.0, "100% is in the file"
    best = max(progression.progression(depth, xp)
               for depth in range(1, 61) for xp in range(1, 41))
    assert best < 100, "no pair of levels can reach the top of the ladder"
    assert round(best, 2) == progression.CEILING
    # And the ceiling is the dungeon axis, not the experience one.
    assert progression.rungs(50, 1)[0] > progression.rungs(1, 30)[1]


def test_the_unreachable_entries_are_the_ones_that_name_a_message():
    """Which seven are stranded, so a future ladder that wires them up fails here
    loudly rather than silently changing what every number on the page means."""
    stranded = sorted(key for key in progression.LADDER
                      if not key.startswith(("Dlvl:", "Xp:")))
    assert stranded == ["Astral Plane", "Home 1", "Home 2", "Home 3", "Home 4",
                        "Home 5", "You ascend t"]
    assert all(progression.LADDER[key] > progression.CEILING / 100
               for key in ("Astral Plane", "You ascend t"))


def test_past_the_end_of_the_ladder_a_run_keeps_the_last_rung():
    """The ladder stops at dungeon level 50 and experience level 30. A run that gets
    past either has not gone backwards, and a naive lookup would return zero."""
    assert progression.progression(60, 1) == progression.progression(50, 1)
    assert progression.progression(1, 40) == progression.progression(1, 30)
    assert progression.progression(60, 40) > 0


def test_a_run_is_reported_three_ways_because_one_would_mislead():
    lives = [
        {"depth": 6, "xp": 2},     # first, before it had learned anything
        {"depth": 25, "xp": 4},
        {"depth": 25, "xp": 5},    # ties the one above on the max rung
        {"depth": 1, "xp": 1},     # died immediately
    ]
    got = progression.of_lives(lives)
    assert got["first"] == progression.of_lives(lives[:1])["best"]
    assert got["best"] == round(progression.progression(25, 5), 2)
    assert got["mean"] < got["best"], "a mean over lives includes the aborted ones"
    # `recent` is a maximum over the tail, not a mean: the last life of a live run is
    # one being played right now, and a mean would read a healthy run as collapsing
    # every time it started again.
    assert got["recent"] == got["best"]
    assert len(got["per_life"]) == 4


def test_the_better_of_two_tied_lives_is_the_one_quoted():
    """Ties are the common case, not the corner: nine of this run's lives reached
    dungeon level 25 and score an identical 46.64%. The one worth quoting is the one
    that also got furthest on the axis the metric is *not* crediting, or the panel
    reports a character weaker than the run actually had."""
    got = progression.of_lives([{"depth": 25, "xp": 4}, {"depth": 25, "xp": 5}])
    assert got["xp"] == 5
    assert got["xp_rung"] == round(progression.rungs(25, 5)[1], 2)
    assert got["depth_rung"] == got["best"]


def test_both_rungs_are_carried_and_not_just_the_winner():
    """The whole reason the panel shows two numbers. The max alone reads a character
    who dug past everything as half way to an ascension."""
    got = progression.of_lives([{"depth": 25, "xp": 5}])
    assert got["depth_rung"] > 40 and got["xp_rung"] < 5
    assert got["best"] == got["depth_rung"]


def test_no_lives_is_not_a_zero():
    """A run that has played nothing has no rung, which is different from the floor."""
    assert progression.of_lives([]) == {}


def test_the_vendored_ladder_is_the_one_it_was_copied_from():
    """The ladder lives here so the harness can score a run on a box without the
    parent repository. The cost of copying is drift, so this is the check — and it
    skips rather than fails where the original is not present."""
    original = (ROOT.parents[1] / "BALROG" / "balrog" / "environments" / "nle"
                / "achievements.json")
    if not original.exists():
        pytest.skip(f"{original} is not on this box")
    assert json.loads(original.read_text()) == progression.LADDER


def test_our_ladder_reproduces_balrogs_own_arithmetic_episode_by_episode():
    """The strongest check there is that the vendored copy is the live one.

    Each submission's per-episode file carries BALROG's *own* progression for that
    episode, alongside the `dlvl_list` and `xplvl_list` it was computed from. Feeding
    those two lists through `progression()` here has to land on their number, on every
    episode, or this harness is scoring runs on a ladder the board is not using.

    Their per-episode field is a **fraction** and the leaderboard column is its mean
    times 100 (0.020257699 -> 2.02576990), which is the trap this pins: everything on
    the page is in percent, and a factor of a hundred would be invisible in a single
    number and obvious only in a comparison.
    """
    eps = progression.episodes()
    if not eps:
        pytest.skip("no BALROG per-episode results on this box")
    original = (ROOT.parents[1] / "BALROG" / "balrog" / "environments" / "nle"
                / "achievements.json")
    if not original.exists():
        pytest.skip(f"{original} is not on this box")
    for one in eps:
        assert one["prog"] == round(progression.progression(one["depth"], one["xp"]), 2)
    assert any(one["prog"] > 0 for one in eps), "every episode scoring zero is a bug"


def test_a_board_episode_carries_no_score_because_the_file_has_none():
    """Four of the scalar fields in those files — score, time, depth, experience_level
    — are zero on all 135, read off the status line after NetHack has torn the
    character down. Only the two ladder lists survive, so an episode gets a rung and
    deliberately no score: there is nothing to put on the score axis."""
    eps = progression.episodes()
    if not eps:
        pytest.skip("no BALROG per-episode results on this box")
    assert not any("score" in one or "turns" in one for one in eps)
    assert all(one["depth"] >= 1 and one["xp"] >= 1 for one in eps)


def test_the_snapshot_of_the_board_is_the_one_on_this_disk():
    """`BOARD` is quoted, and quoted numbers go stale. Where the results directory is
    here, the snapshot's top entry has to still be the top entry."""
    live = progression.live_board()
    if not live:
        pytest.skip("no BALROG results on this box")
    assert len(live) == progression.BOARD_N, (
        f"the snapshot says {progression.BOARD_N} submissions and the directory "
        f"holds {len(live)}"
    )
    assert round(live[0][1], 2) == progression.BOARD[0][1]
    assert progression.BOARD[0][0] in live[0][0]
    middle = sorted(value for _, value in live)[len(live) // 2]
    assert round(middle, 2) == progression.BOARD_MEDIAN
