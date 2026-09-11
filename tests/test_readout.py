"""The readout: a run recovered from its own history, and the three questions.

Nothing on disk holds the curve — `result.json` is rewritten in place — so this
replays the keystroke history and measures what no file kept. The tests are about
the two ways that can go wrong: a replay that is not the run it claims to be, and a
measurement that says a session handled the interface when it did not.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("nle")

import act  # noqa: E402
import readout  # noqa: E402


@pytest.fixture
def played(tmp_path):
    """A short run in a workspace, driven key by key."""

    def build(keys, seed=3, budget=200):
        ws, env_dir = tmp_path / "ws", tmp_path / "env"
        ws.mkdir(exist_ok=True)
        env_dir.mkdir(exist_ok=True)
        state = act.RunState(seed=seed, obs=["tty"], budget=budget, env_dir=str(env_dir))
        session = act.Session.create(ws, state)
        for key in keys(session) if callable(keys) else keys:
            session.play(key)
        return ws, env_dir

    return build


def test_a_replayed_run_is_the_run(played):
    ws, env_dir = played(["l", "l", "j", "cr", "k", "h"])
    one = readout.read(ws, env_dir)
    assert one["keys"] == ["l", "l", "j", "cr", "k", "h"]
    assert one["interface"]["keys"] == 6
    assert one["seed"] == 3 and one["blind"] is True
    assert one["curve"] == [], "no milestone falls inside a six-key run"


def test_a_history_that_is_not_this_run_is_refused(played):
    ws, env_dir = played(["l", "l"])
    record = json.loads((env_dir / "result.json").read_text())
    record["history"] = ["l"]
    (env_dir / "result.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="not a history of that run"):
        readout.read(ws, env_dir)


def test_a_replay_that_scores_differently_is_refused(played):
    ws, env_dir = played(["l", "l"])
    record = json.loads((env_dir / "result.json").read_text())
    record["episode_scores"] = [999]
    (env_dir / "result.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="not the run"):
        readout.read(ws, env_dir)


def test_keys_thrown_into_a_prompt_are_counted(played):
    """The measurement the pilot turns on (F2). A key played at a `--More--` that the
    prompt cannot accept costs an action, changes nothing, and says nothing about
    having changed nothing — so the run has to be able to say how many there were."""

    def drive(session):
        # Walk in circles until the game has something to say. Seed 4 swaps places
        # with the pony at key 22 and puts a `--More--` up to say so.
        cycle, played = ("l", "j", "h", "k", "u", "n", "b", "y"), 0
        while "--More--" not in session.game.screen() and played < 100:
            yield cycle[played % len(cycle)]
            played += 1
        assert "--More--" in session.game.screen(), "no message in 100 keys"
        # Four keys the prompt cannot take, then the one it can.
        yield from ("l", "l", "l", "l", "cr")

    ws, env_dir = played(drive, seed=4, budget=400)
    face = readout.read(ws, env_dir)["interface"]
    assert face["wasted"] == 4
    assert face["longest_stuck"] == 4
    assert 0 < face["wasted_pct"] < 100
    assert face["at_a_prompt_pct"] > 0


def test_the_readout_sums_cost_over_sessions_too(tmp_path):
    """The same defect on the other side of the fence: `tools/readout.py` reads the
    stream itself, and a chained run has one `result` per session."""
    stream = tmp_path / "agent_stream.jsonl"
    stream.write_text("\n".join(
        json.dumps(event) for event in (
            {"type": "system", "subtype": "init"},
            {"type": "result", "total_cost_usd": 31.77,
             "usage": {"cache_read_input_tokens": 100, "output_tokens": 10}},
            {"type": "system", "subtype": "init"},
            {"type": "result", "total_cost_usd": 12.10,
             "usage": {"cache_read_input_tokens": 40, "output_tokens": 5}},
        )
    ))
    said = readout.agent(stream)
    assert said["cost_usd"] == pytest.approx(43.87)
    assert said["sessions"] == 2
    assert said["cache_read_tokens"] == 140 and said["output_tokens"] == 15


def test_turns_are_summed_over_lives_and_not_carried_across(played):
    """The clock resets when a life does, so a run's turns are the sum over lives —
    and its best life is a maximum, never a total."""
    ws, env_dir = played(["l", "l", "#quit", "y", "l", "l"])
    one = readout.read(ws, env_dir)
    assert len(one["episode_scores"]) == 2
    assert one["endings"] == ["quit"]
    assert one["interface"]["turns"] >= 1
    assert one["best_episode"] == max(one["episode_scores"])


def test_the_curve_reports_the_best_life_and_not_a_total(played):
    rows = [
        {"life": 0, "score": 10, "turns": 5, "depth": 1, "xplevel": 1},
        {"life": 0, "score": 40, "turns": 9, "depth": 2, "xplevel": 1},
        {"life": 1, "score": 25, "turns": 3, "depth": 1, "xplevel": 2},
    ]
    rows = [dict(row, action=i, key="l", reward=0, moved=True, prompt="", wasted=False,
                 ended="") for i, row in enumerate(rows, start=1)]
    monkeyed = readout.MILESTONES
    readout.MILESTONES = (3,)
    try:
        mark = readout.curve(rows)[0]
    finally:
        readout.MILESTONES = monkeyed
    assert mark["best_life"] == 40, "the curve summed the lives"
    assert mark["turns"] == 12, "the clock was carried across a life"
    assert mark["lives"] == 2 and mark["depth"] == 2 and mark["xplevel"] == 2
