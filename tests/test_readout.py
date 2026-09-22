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


def test_the_curve_carries_the_rung_beside_the_score(played):
    """The two answer different questions, which is why both are in the table: a
    score says how a game went and a rung says how far through it got."""
    rows = [
        {"life": 0, "score": 10, "turns": 5, "depth": 1, "xplevel": 1},
        {"life": 0, "score": 40, "turns": 9, "depth": 12, "xplevel": 1},
    ]
    rows = [dict(row, action=i, key="l", reward=0, moved=True, prompt="", wasted=False,
                 ended="") for i, row in enumerate(rows, start=1)]
    monkeyed = readout.MILESTONES
    readout.MILESTONES = (1, 2)
    try:
        marks = readout.curve(rows)
    finally:
        readout.MILESTONES = monkeyed
    from progression import progression

    assert marks[0]["progression"] == 0.0, "dungeon level 1 is the start, not progress"
    assert marks[1]["progression"] == round(progression(12, 1), 2)
    assert marks[1]["progression"] > marks[0]["progression"]


def test_the_pair_quoted_is_a_pair_one_life_actually_had(played):
    """The max over lives cannot differ from the max over the run — a maximum
    distributes — so the number is safe either way. The *pair* is not: a run whose
    first life dug to level 29 and whose second levelled to 20 never had a character
    that was both, and reporting `dlvl 29, xp 20` would invent one."""
    from progression import of_lives, rungs

    got = of_lives([{"depth": 29, "xp": 1}, {"depth": 1, "xp": 20}])
    assert (got["depth"], got["xp"]) == (1, 20), "it stitched two lives into one"
    assert got["best"] == round(max(rungs(1, 20)), 2)
    assert got["xp_rung"] > got["depth_rung"], "experience is carrying this one"


def test_a_run_carries_its_lives_and_its_rungs(played):
    one = readout.read(*played(["l", "l", "j", "#quit", "y", "l"]))
    assert [life["life"] for life in one["lives"]] == [0, 1]
    got = one["progression"]
    assert got["first"] == got["per_life"][0]
    assert got["best"] == max(got["per_life"])
    # Both rungs, always: the higher one alone is what hides an unbalanced run.
    assert got["best"] == max(got["depth_rung"], got["xp_rung"])


def codex_stream(where: Path, events: list[dict]) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    stream = where / "agent_stream.jsonl"
    stream.write_text("\n".join(json.dumps(event) for event in events))
    return stream


def test_a_codex_stream_is_read_as_codex_and_priced_from_its_model(tmp_path):
    """Neither CLI's stream reads as the other's. Read as Claude's, a Codex stream has
    no sessions, no tool calls and no cost — a readout of nothing that looks like a
    readout of a session that did nothing. The model is taken from what the CLI kept
    for the session, because a run that is still playing has no report to say."""
    ws = tmp_path / "launch" / "ZJLJX"
    stream = codex_stream(ws, [
        {"type": "thread.started", "thread_id": "a"},
        {"type": "item.completed", "item": {"type": "command_execution",
                                            "command": "./act do s", "exit_code": 0}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1_000_000,
                                             "cached_input_tokens": 900_000,
                                             "output_tokens": 10_000}},
        {"type": "thread.started", "thread_id": "b"},
        {"type": "harvested.item", "at": 1, "item": {"type": "imageView", "path": "/a.png"}},
        {"type": "turn.completed", "usage": {"input_tokens": 500_000,
                                             "cached_input_tokens": 400_000,
                                             "output_tokens": 5_000}},
    ])
    kept = tmp_path / "launch" / ".sessions" / "ZJLJX" / ".codex" / "sessions" / "2026" / "09" / "22"
    kept.mkdir(parents=True)
    (kept / "rollout-x.jsonl").write_text(json.dumps(
        {"type": "turn_context", "payload": {"model": "gpt-6-astra"}}, separators=(",", ":")))

    said = readout.agent(stream)
    assert said["cli"] == "codex" and said["model"] == "gpt-6-astra"
    assert said["sessions"] == 2 and said["turns"] == 1 and said["tool_calls"] == 2
    price = readout.priced("gpt-6-astra", input_tokens=1_500_000, output_tokens=15_000,
                           cache_read_tokens=1_300_000)
    assert said["cost_usd"] == pytest.approx(round(price, 2)) and price > 0
    assert said["cache_read_tokens"] == 1_300_000 and said["output_tokens"] == 15_000


def test_the_report_names_the_model_once_there_is_one(tmp_path):
    ws = tmp_path / "launch" / "ZJLJX"
    codex_stream(ws, [{"type": "thread.started"}])
    reports = tmp_path / "launch" / ".rig" / "reports"
    reports.mkdir(parents=True)
    (reports / "ZJLJX.json").write_text(json.dumps({"model": "gpt-5.6-sol"}))
    assert readout.model_of(ws, "codex") == "gpt-5.6-sol"
    assert readout.model_of(tmp_path / "launch" / "NOPE", "codex") == "gpt-5.6-sol", \
        "with nothing to read, the adapter's default"
