"""The launcher: what a workspace holds, and what it is allowed to be called."""

import asyncio
import json
import re
import sqlite3
import subprocess
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

from agents import AGENTS, HARVESTED, PRICES  # noqa: E402

import run  # noqa: E402
from nle_game import VARIANTS  # noqa: E402


def test_a_workspace_holds_the_prompt_and_two_shims(tmp_path):
    ws = run.make_workspace(tmp_path, "B4XPT", ["tty"], "blind")
    assert sorted(p.name for p in ws.iterdir()) == ["CLAUDE.md", "act", "python"]
    brief = (ws / "CLAUDE.md").read_text()
    assert brief.startswith("You are playing a game you have never seen before.")
    assert "notes.md" in brief, "the playing doctrine did not travel"


def test_the_shims_do_not_name_what_is_being_played(tmp_path):
    """The first thing a curious session does with a command it has been handed is
    read it. Written directly, `act` names the harness — and the harness is named
    after the game (F18)."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["tty"], "blind")
    for name in ("act", "python"):
        said = (ws / name).read_text()
        assert "nle" not in said.lower(), f"{name} says {said!r}"
        assert str(tmp_path / run.BIN) in said
    # And they still work.
    out = subprocess.run([str(ws / "python"), "-c", "import numpy; print('ok')"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_the_default_brief_withholds_the_game(tmp_path):
    """Every sibling harness withholds it, and here it takes more than keeping quiet
    (F18): the game names itself on the opening screen, so `--opening blind` has to
    reach the actuator too."""
    assert run.OPENING and "blind" in run.OPENING
    parsed = run.build_parser().parse_args([]) if hasattr(run, "build_parser") else None
    ws = run.make_workspace(tmp_path, "B4XPT", ["tty"], "blind")
    assert "NetHack" not in (ws / "CLAUDE.md").read_text()
    assert parsed is None or parsed.opening == "blind"


def test_naming_the_game_reaches_the_actuator_and_not_only_the_brief(tmp_path):
    """A session told nothing, whose first screen says `welcome to NetHack!`, has
    been told. The two have to agree."""
    import inspect

    said = inspect.getsource(run.play)
    assert '--named' in said and 'args.opening == "named"' in said


def test_the_two_openings_differ_in_one_paragraph_and_nothing_else(tmp_path):
    """`named` is the ablation, so it has to be the same run with one sentence
    changed — not a second brief that drifted."""
    named = (run.make_workspace(tmp_path / "a", "B4XPT", ["tty"], "named") / "CLAUDE.md").read_text()
    blind = (run.make_workspace(tmp_path / "b", "K7M3Q", ["tty"], "blind") / "CLAUDE.md").read_text()
    assert "NetHack" in named and "NetHack" not in blind
    assert blind.startswith("You are playing a game you have never seen before.")
    tail = "Every action you play is **one keystroke**"
    assert named[named.index(tail):] == blind[blind.index(tail):]


def test_the_brief_says_which_channels_this_run_gives(tmp_path):
    """Written by hand it would drift from what the log carries the first time
    somebody ran with a different --obs."""
    screens = (run.make_workspace(tmp_path / "a", "B4XPT", ["tty"], "named") / "CLAUDE.md").read_text()
    assert "`screens/`" in screens and "`frames/`" not in screens

    both = (run.make_workspace(tmp_path / "b", "K7M3Q", ["tty", "pixels"], "named") / "CLAUDE.md").read_text()
    assert "`screens/`" in both and "`frames/`" in both
    assert "more than one form" in both

    arrays = (run.make_workspace(tmp_path / "c", "ZZZZ9", ["tty", "symbolic"], "named") / "CLAUDE.md").read_text()
    assert "`symbolic/`" in arrays and "`frames/`" not in arrays


def test_the_brief_explains_no_mechanic(tmp_path):
    """Naming the game is the whole of what the named brief gives away. Everything
    a player would have to work out — or recall — stays out of it, in both
    conditions."""
    for opening in ("named", "blind"):
        brief = (
            run.make_workspace(tmp_path / opening, "B4XPT", ["tty"], opening) / "CLAUDE.md"
        ).read_text().lower()
        # Whole words: the doctrine tells a session its files "survive" a
        # compaction, and a substring match would read that as an explanation.
        for word in ("elbereth", "staircase", "descend", "altar", "pray", "monster",
                     "inventory", "hunger", "wield", "armour", "armor", "spoiler",
                     "valkyrie", "tourist", "dungeon"):
            assert not re.search(rf"\b{word}\b", brief), f"the brief says {word!r}"


def test_the_brief_says_what_the_run_is_judged_on(tmp_path):
    """The first pilot was told what it could do and not what any of it was for, so it
    maximised the only number that grew: reward summed over the run, which counts one
    achievement once per life. Naming the unit the benchmark itself reports — return
    on a single episode — gives away nothing about the world."""
    brief = (run.make_workspace(tmp_path, "B4XPT", ["tty"], "named") / "CLAUDE.md").read_text()
    # Wrapped prose: the sentence is there but the line breaks are not where a naive
    # substring search would put them.
    said = " ".join(brief.split())
    assert "judged on is the best single life in it" in said
    assert "a total across lives is not the thing to grow" in said


def test_the_agents_interpreter_reads_observations_and_not_the_package(tmp_path):
    """The run is unplayable without numpy and Pillow, and compromised with nle."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["tty"], "blind")
    ok = subprocess.run([str(ws / "python"), "-c", "import numpy, PIL; print('ok')"],
                        capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0 and "ok" in ok.stdout, ok.stderr
    for module in ("nle", "gymnasium"):
        blocked = subprocess.run([str(ws / "python"), "-c", f"import {module}"],
                                 capture_output=True, text=True, timeout=120)
        assert blocked.returncode != 0, f"the agent's interpreter can import {module}"


def test_a_chained_runs_cost_is_the_sum_of_its_sessions():
    """One session emits one `result`; a stinted run is several sessions appending
    to one stream. Taking the last one reported the final session's bill as the
    run's — the void 3000-key pilot printed $15.57 for three sessions whose first
    alone cost $31.77."""
    import json as _json

    report = run.Report(label="B4XPT", workspace="x")
    agent = run.AGENTS["claude"]
    for cost, tokens in ((31.77, 1000), (28.40, 900), (12.10, 700)):
        run.absorb(agent, report, _json.dumps({
            "type": "result",
            "total_cost_usd": cost,
            "usage": {"input_tokens": tokens, "output_tokens": 10,
                      "cache_read_input_tokens": 5, "cache_creation_input_tokens": 2},
        }))
    assert report.cost_usd == pytest.approx(72.27)
    assert report.input_tokens == 2600
    assert report.output_tokens == 30
    assert report.cache_read_tokens == 15
    assert report.cache_creation_tokens == 6


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_labels_are_opaque_and_recorded_outside_every_workspace(tmp_path):
    plan = [("nethack", 0), ("nethack", 1), ("score", 3)]
    known = run.assign_labels(tmp_path, plan)
    assert sorted(known.values()) == sorted(plan)
    for label in known:
        assert len(label) == run.LABEL_LEN
        assert set(label) <= set(run.LABEL_CHARS)
        assert not any(v in label.lower() for v in VARIANTS)
    # The mapping lives under .rig/, not the launch root: the root is one `..` from
    # every workspace.
    assert (tmp_path / run.RIG / run.LABELS).is_file()
    assert not (tmp_path / run.LABELS).exists()


def test_labels_are_stable_across_calls_and_extend(tmp_path):
    first = run.assign_labels(tmp_path, [("nethack", 0)])
    again = run.assign_labels(tmp_path, [("nethack", 0), ("score", 0)])
    assert first.items() <= again.items(), "a second call renamed an existing game"
    assert len(again) == 2


def test_labels_are_not_derived_from_the_spec(tmp_path):
    """A name a session could invert would let it recognise a world it had played,
    and a launch running one world twice would have two workspaces announcing it."""
    seen = set()
    for i in range(6):
        root = tmp_path / str(i)
        seen |= set(run.assign_labels(root, [("nethack", 0)]))
    assert len(seen) > 1, "the label is a function of the game"


# --------------------------------------------------------------------------- #
# What to play
# --------------------------------------------------------------------------- #


def test_specs():
    def parsed(games, seed=0):
        return run.specs(type("A", (), {"games": games, "seed": seed})())

    assert parsed([]) == [("nethack", 0)]
    assert parsed([], seed=4) == [("nethack", 4)]
    assert parsed(["nethack", "nethack:1", "score:3"]) == [
        ("nethack", 0), ("nethack", 1), ("score", 3)
    ]
    assert parsed(["nethack", "nethack"]) == [("nethack", 0)], "a repeat is one game"
    with pytest.raises(SystemExit, match="not one of"):
        parsed(["minihack"])


def test_unfinished_reads_the_launch_record_not_the_workspace(tmp_path):
    run.assign_labels(tmp_path, [("nethack", 0), ("score", 2)])
    known = json.loads((tmp_path / run.RIG / run.LABELS).read_text())
    done, going = list(known)
    (tmp_path / done).mkdir()
    (tmp_path / done / "state.json").write_text(json.dumps({"terminal": True}))
    (tmp_path / going).mkdir()
    (tmp_path / going / "state.json").write_text(json.dumps({"terminal": False}))

    left = run.unfinished(tmp_path)
    assert [label for label, _ in left] == [going]
    assert left[0][1] == (known[going]["variant"], known[going]["seed"])

    with pytest.raises(SystemExit, match="not a launch directory"):
        run.unfinished(tmp_path / "nowhere")


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_the_verdict_is_kept_out_of_the_workspace(tmp_path):
    """It carries the run's league table across lives, and a replay is told to read
    what its predecessor left behind."""
    report = run.Report(label="B4XPT", variant="nethack", seed=0,
                        workspace=str(tmp_path / "B4XPT"),
                        episode_scores=[112, 8], best_episode=112, score=112)
    path = run.write_report(tmp_path, report)
    assert path == tmp_path / run.RIG / "reports" / "B4XPT.json"
    assert not (tmp_path / "B4XPT" / "report.json").exists()
    assert "112" in path.read_text()


def test_rotation_leaves_no_verdict_behind(tmp_path):
    """A replay must not overwrite the result of the session it is replaying."""
    first = run.Report(label="B4XPT", workspace="x", score=2)
    run.write_report(tmp_path, first)
    run.write_report(tmp_path, run.Report(label="B4XPT", workspace="x", score=9))
    out = tmp_path / run.RIG / "reports"
    assert json.loads((out / "B4XPT-attempt1.json").read_text())["score"] == 2
    assert json.loads((out / "B4XPT.json").read_text())["score"] == 9


# --------------------------------------------------------------------------- #
# Carrying a run on
# --------------------------------------------------------------------------- #


def test_continuable_is_the_opposite_question_to_unfinished(tmp_path):
    """`--replay` asks which games never reached an ending, and starts those over.
    `--continue` asks which have played anything at all, because a run that spent
    its budget is exactly the one worth extending."""
    run.assign_labels(tmp_path, [("nethack", 0), ("nethack", 1), ("score", 2)])
    known = json.loads((tmp_path / run.RIG / run.LABELS).read_text())
    spent, going, untouched = list(known)
    for label, state in (
        (spent, {"terminal": True, "outcome": "budget", "actions_used": 3000}),
        (going, {"terminal": False, "actions_used": 412}),
        (untouched, {"terminal": False, "actions_used": 0}),
    ):
        (tmp_path / label).mkdir()
        (tmp_path / label / "state.json").write_text(json.dumps(state))

    assert [label for label, _ in run.unfinished(tmp_path)] == [going, untouched]
    assert sorted(label for label, _ in run.continuable(tmp_path)) == sorted([spent, going])
    assert run.continuable(tmp_path)[0][1] == (
        known[run.continuable(tmp_path)[0][0]]["variant"],
        known[run.continuable(tmp_path)[0][0]]["seed"],
    )
    with pytest.raises(SystemExit, match="not a launch directory"):
        run.continuable(tmp_path / "nowhere")


def test_played_reads_the_open_record(tmp_path):
    assert run.played(tmp_path / "nothing here") == 0
    (tmp_path / "W").mkdir()
    (tmp_path / "W" / "state.json").write_text(json.dumps({"actions_used": 7}))
    assert run.played(tmp_path / "W") == 7


def test_the_continuation_task_hands_over_a_run_and_not_a_world():
    """REPLAY_TASK's game went back to the beginning and only the notes carried
    over. This one carries the run itself, so what it must not say is that anything
    has been reset — and what it must say is that the log is too big to read and
    that this session has a successor of its own."""
    said = " ".join(run.CONTINUE_TASK.split())
    assert "already part-played" in said
    assert "exactly where the last session left it" in said
    assert "too big to read" in said
    assert "leave notes.md fit for whoever comes next" in said
    assert "beginning" not in said and "started again" not in said
    assert "reports this session is over" in said, "it would play past its stint"


def test_a_kept_workspace_gets_the_current_brief(tmp_path):
    """`--replay` and `--continue` both keep the workspace a previous launch built,
    and with it whatever CLAUDE.md the harness wrote then. A brief that contradicts
    the log is worse than a stale one."""
    ws = run.make_workspace(tmp_path, "B4XPT", ["tty"], "named")
    (ws / "state.json").write_text(json.dumps({"obs": ["tty", "symbolic"]}))
    (ws / "notes.md").write_text("what I worked out")
    (ws / "CLAUDE.md").write_text("a brief from an older harness")

    run.refresh_brief(ws, "named")
    now = (ws / "CLAUDE.md").read_text()
    # From the record, not from how this workspace happened to be built.
    assert "`symbolic/`" in now and "`frames/`" not in now
    assert "stint" in now, "the prompt that travelled did not carry the handover"
    assert (ws / "notes.md").read_text() == "what I worked out"


def test_the_prompt_says_a_stint_is_not_the_whole_budget():
    """Craftax's pilot ground `reset` because the interface offered a lever the game
    did not. A prompt that told a stinted session the budget was all its own would be
    the same mistake in the other direction — it would spend a successor's actions
    on the assumption they were about to be lost."""
    said = " ".join((run.REPO / "PROMPT.md").read_text().split())
    assert "The budget belongs to the run" in said
    assert "another continues the same run from exactly where you stopped" in said
    assert "a finding you did not write down did not happen" in said
    assert "the budget is the whole of what you have" not in said


def test_a_run_without_a_stint_is_one_session(tmp_path, monkeypatch):
    """Chaining is what `--stint` turns on. Without one, a session that stops with
    budget left has decided it is finished, and the launcher does not overrule it —
    which is how every run behaved before stints existed."""
    starts, ws = [], tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["tty"], "actions_used": 0, "terminal": False}))

    async def fake_act(where, *argv):
        if argv[0] == "init":
            starts.append(argv)
            played = json.loads((ws / "state.json").read_text())
            played["actions_used"] += 10
            (ws / "state.json").write_text(json.dumps(played))
        return ""

    async def fake_session(argv, where, env, report, agent):
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w, o: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "best_episode", "episodes_completed",
                        "mean_episode", "score", "lives", "deaths", "quits", "turns",
                        "unique_cells", "max_depth", "max_xplevel")}
        | {"episode_scores": [], "endings": [], "roles": []}))

    def go(stint, max_sessions=3, retries=2):
        starts.clear()
        args = type("A", (), {
            "carry_on": str(tmp_path), "replay": None, "obs": ["tty"], "budget": 500,
            "stint": stint, "max_sessions": max_sessions, "fresh_world": False,
            "agent": "claude", "model": "m", "dry_run": False, "retries": retries,
            "opening": "named", "idle_wait": 0, "no_fence": False,
        })()
        report = asyncio.run(run.play("W", ("nethack", 0), tmp_path, args, {}, ["W"],
                                      asyncio.Semaphore(1)))
        return report, list(starts)

    report, opened = go(stint=0)
    assert report.sessions == 1 and len(opened) == 1

    report, opened = go(stint=10)
    assert report.sessions == 3, "a stinted run stopped chaining"
    assert all("--resume" in one for one in opened), "a chained session restarted the world"
    assert all("--stint" in one for one in opened)


# --------------------------------------------------------------------------- #
# Surviving the night
# --------------------------------------------------------------------------- #


def credential(path: Path, hours: float, days: float = 30.0) -> Path:
    """A stored login whose access token has `hours` left and refresh token `days`."""
    now = time.time() * 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": f"token-{hours}", "refreshToken": "r",
        "expiresAt": int(now + hours * 3_600_000),
        "refreshTokenExpiresAt": int(now + days * 86_400_000),
    }}))
    return path


def test_a_refreshed_credential_is_not_thrown_away(tmp_path, monkeypatch):
    """The failure this exists to prevent, and it only appears on a run long enough
    to need several sessions. An access token lasts about eight hours and nothing
    refreshes the operator's file while a chain is running, so from the second
    session on the session's own refreshed copy is the newer of the two."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    theirs = credential(home / ".claude" / ".credentials.json", hours=8)

    # Session one: nothing local, so it links to the operator's.
    config = run.session_config(tmp_path / "launch", "B4XPT")
    link = config / ".credentials.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()

    # Session one refreshes: the CLI replaces the link with a copy of a newer token.
    link.unlink()
    credential(link, hours=8)
    # ...and time passes, so the operator's is now the stale one.
    credential(theirs, hours=-0.5)

    run.session_config(tmp_path / "launch", "B4XPT")
    assert not link.is_symlink(), "the refreshed token was replaced by an expired one"
    assert run.credential_life(link)[0] > time.time()


def test_a_stale_local_credential_is_replaced(tmp_path, monkeypatch):
    """The other direction, which is the bug that put the re-seeding here: a copy
    left over from a previous launch expires where the operator's does not."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    theirs = credential(home / ".claude" / ".credentials.json", hours=8)
    config = tmp_path / "launch" / ".sessions" / "B4XPT" / ".claude"
    credential(config / ".credentials.json", hours=-20)

    run.session_config(tmp_path / "launch", "B4XPT")
    link = config / ".credentials.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()


def test_an_unreadable_credential_is_replaced(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    credential(home / ".claude" / ".credentials.json", hours=8)
    config = tmp_path / "launch" / ".sessions" / "B4XPT" / ".claude"
    config.mkdir(parents=True)
    (config / ".credentials.json").write_text("{ this is not json")

    run.session_config(tmp_path / "launch", "B4XPT")
    assert (config / ".credentials.json").is_symlink()
    assert run.credential_life(tmp_path / "nothing here") == (0.0, 0.0)


def test_a_session_that_plays_nothing_is_retried_and_then_gives_up(tmp_path, monkeypatch):
    """A run meant to go unattended for hours should not die on one API error at
    startup — a session that plays nothing costs almost nothing, so a retry is cheap.
    A chain that cannot make progress must still stop rather than spend the night
    failing, which is what an expired credential looks like from here."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["tty"], "actions_used": 0, "terminal": False}))
    plays = iter([10, 0, 0, 10, 0, 0, 0])  # two idle sessions, then progress, then three

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += next(plays, 0)
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", lambda *a: _nothing())
    monkeypatch.setattr(run, "refresh_brief", lambda w, o: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "best_episode", "episodes_completed",
                        "mean_episode", "score", "lives", "deaths", "quits", "turns",
                        "unique_cells", "max_depth", "max_xplevel")}
        | {"episode_scores": [], "endings": [], "roles": []}))

    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["tty"], "budget": 500,
        "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
        "model": "m", "dry_run": False, "retries": 2, "opening": "named",
        "idle_wait": 0, "no_fence": False,
    })()
    report = asyncio.run(run.play("W", ("nethack", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    # 1 plays, 2 and 3 idle, 4 plays (the counter resets), 5-7 idle and it gives up.
    assert report.sessions == 7
    assert run.played(ws) == 20


async def _nothing():
    return "", 0


def test_the_wait_between_idle_sessions_doubles_and_is_capped(tmp_path, monkeypatch):
    """The retries were written for a failure that has passed by the time the next
    session starts. A rate limit has not: it is a window, and three retries in three
    seconds are three ways of asking the same question inside it. So the wait doubles
    — and stops doubling at an hour, because the thing being waited out is that long
    and checking less often than that buys nothing."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["tty"], "actions_used": 0, "terminal": False}))
    waits: list[float] = []

    async def no_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(run, "act", lambda where, *argv: _nothing_at_all())
    monkeypatch.setattr(run, "one_session", lambda *a: _nothing())
    monkeypatch.setattr(run, "refresh_brief", lambda w, o: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    monkeypatch.setattr(run.asyncio, "sleep", no_wait)
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "best_episode", "episodes_completed",
                        "mean_episode", "score", "lives", "deaths", "quits", "turns",
                        "unique_cells", "max_depth", "max_xplevel")}
        | {"episode_scores": [], "endings": [], "roles": []}))

    def go(retries, idle_wait):
        waits.clear()
        args = type("A", (), {
            "carry_on": str(tmp_path), "replay": None, "obs": ["tty"], "budget": 500,
            "stint": 10, "max_sessions": 20, "fresh_world": False, "agent": "claude",
            "model": "m", "dry_run": False, "retries": retries, "opening": "named",
            "idle_wait": idle_wait, "no_fence": False,
        })()
        report = asyncio.run(run.play("W", ("nethack", 0), tmp_path, args, {}, ["W"],
                                      asyncio.Semaphore(1)))
        return report, list(waits)

    report, waited = go(retries=4, idle_wait=60)
    assert report.sessions == 5, "the chain stopped somewhere other than the retries"
    assert waited == [60, 120, 240, 480], "the wait did not double"
    # And nothing is waited after the last one: the chain is over, so an hour spent
    # there is an hour of a finished run.
    assert len(waited) == report.sessions - 1

    _, waited = go(retries=3, idle_wait=1800)
    assert waited == [1800, run.MAX_IDLE_WAIT, run.MAX_IDLE_WAIT]


async def _nothing_at_all():
    return ""


# --------------------------------------------------------------------------- #
# The other CLI
# --------------------------------------------------------------------------- #


def test_codex_is_handed_the_same_workspace_and_a_narrower_machine():
    """Two things at once, because they are the same requirement. The brief must be
    the file both arms read, so that a workspace differs between them in nothing; and
    what the session can reach beyond it must be narrowed by name, since this CLI has
    no denylist to pass and ships a browser and agents of its own switched on. Here the
    web matters more than anywhere: NetHack's wiki answers every question a run asks."""
    codex = AGENTS["codex"]
    argv = codex.argv("PLAY", codex.model, Path("/ws"))
    flat = " ".join(argv)

    assert 'project_doc_fallback_filenames=["CLAUDE.md"]' in argv, "reads AGENTS.md only"
    assert "--ignore-user-config" in argv, "the operator's config would decide the model"
    assert "tools.web_search=false" in argv
    assert "tools.view_image=true" in argv, "the Claude arm can open a PNG"
    for feature in ("browser_use", "computer_use", "multi_agent", "plugins"):
        assert f"--disable {feature}" in flat
    # Its own sandbox cannot start on this machine, and a session whose every command
    # fails answers the prompt anyway rather than stopping. See the class docstring.
    assert "danger-full-access" in argv and "workspace-write" not in argv
    assert argv[-1] == "PLAY", "anything after the prompt is read as part of it"


def test_a_session_gets_the_config_directory_its_own_cli_reads(tmp_path, monkeypatch):
    """The same walk, guarded the same way, for a CLI that spells all three of the
    directory, the credential and the variable differently."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    codex = AGENTS["codex"]
    theirs = home / ".codex" / "auth.json"
    theirs.parent.mkdir(parents=True)
    theirs.write_text(json.dumps({"last_refresh": "2026-09-21T19:43:44.604670599Z"}))

    config = run.session_config(tmp_path / "launch", "B4XPT", codex)
    assert config == tmp_path / "launch" / ".sessions" / "B4XPT" / ".codex"
    link = config / "auth.json"
    assert link.is_symlink() and link.resolve() == theirs.resolve()
    # And the same keep-the-fresher rule, on a stamp that is an age rather than an
    # expiry: a session that refreshed its own copy does not get it taken away.
    link.unlink()
    link.write_text(json.dumps({"last_refresh": "2026-09-22T19:43:44.000000Z"}))
    run.session_config(tmp_path / "launch", "B4XPT", codex)
    assert not link.is_symlink(), "the refreshed token was replaced by an older one"


def test_a_chained_codex_runs_cost_is_the_sum_of_its_sessions():
    """The bug `Claude.absorb` already paid for here, on the other CLI. `codex exec`
    plays one prompt, so each session reports its usage once, at its end — and a
    stinted run is many sessions appending to one stream. Assigned rather than summed,
    a thirty-session run would report its last session's bill as the run's."""
    report = run.Report(label="B4XPT", workspace="x", agent="codex", model="gpt-5.6-sol")
    codex = AGENTS["codex"]
    for fresh, cached, out in ((1_000_000, 9_000_000, 100_000), (500_000, 4_500_000, 50_000)):
        run.absorb(codex, report, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": fresh + cached, "cached_input_tokens": cached,
            "cache_write_input_tokens": 0, "output_tokens": out,
            "reasoning_output_tokens": out // 2}}))
    assert report.input_tokens == 15_000_000
    assert report.cache_read_tokens == 13_500_000
    price = PRICES["gpt-5.6-sol"]
    expect = (1_500_000 * price["input"] + 13_500_000 * price["cached"]
              + 150_000 * price["output"]) / 1_000_000
    assert report.cost_usd == pytest.approx(expect)


def test_reasoning_is_billed_once():
    """Reasoning tokens are a part of `output_tokens`, not a third term beside it:
    the CLI's own rollout reports `total_tokens == input_tokens + output_tokens`.
    The adapter this was ported from added them again, which bills them twice."""
    report = run.Report(label="B4XPT", workspace="x", agent="codex", model="gpt-6-astra")
    run.absorb(AGENTS["codex"], report, json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
        "output_tokens": 1_000_000, "reasoning_output_tokens": 600_000}}))
    assert report.output_tokens == 1_000_000
    assert report.cost_usd == pytest.approx(PRICES["gpt-6-astra"]["output"])


def test_an_unpriced_model_costs_nothing_rather_than_a_guess():
    report = run.Report(label="B4XPT", workspace="x", agent="codex", model="gpt-unknown")
    run.absorb(AGENTS["codex"], report, json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 10, "output_tokens": 10}}))
    assert report.cost_usd == 0.0 and report.input_tokens == 10


def history(home: Path, rows: list[tuple[int, dict]]) -> Path:
    """A CLI thread history holding `rows`, as the real one is shaped."""
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / AGENTS["codex"].HISTORY)
    db.execute("create table thread_items (created_at_ms int, item_json text)")
    db.executemany("insert into thread_items values (?, ?)",
                   [(at, json.dumps(item)) for at, item in rows])
    db.commit()
    db.close()
    return home


def test_a_picture_opened_by_eye_is_counted_and_audited(tmp_path):
    """The failure this exists to prevent is silent. This CLI's stream carries no
    event for looking at an image, so before harvesting a session could open any file
    as a picture and the stream would show none of it."""
    ws = tmp_path / "W"
    ws.mkdir()
    home = history(tmp_path / "home", [
        (1000, {"type": "imageView", "path": "/ws/frames/000001.png"}),
        (1001, {"type": "agentMessage", "text": "not this"}),
    ])
    codex, report = AGENTS["codex"], run.Report(label="W", workspace=str(ws))

    at = run.harvest(codex, home, ws, report, since=0.0)
    assert at == 1000
    assert report.tool_calls == 1, "looking at a picture is a tool call on both arms"

    events = [json.loads(line)
              for line in (ws / "agent_stream.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == [HARVESTED], "the message was harvested too"
    assert sum(codex.images(e) for e in events) == 1
    assert codex.ran(events[0]) == ["/ws/frames/000001.png"], "the audit cannot see it"


def test_a_picture_is_not_harvested_twice(tmp_path):
    """A run's sessions share one history and a run outlives the launch that started
    it, so the watermark has to be recoverable from the stream itself."""
    ws = tmp_path / "W"
    ws.mkdir()
    home = history(tmp_path / "home", [(1000, {"type": "imageView", "path": "/a.png"})])
    codex, report = AGENTS["codex"], run.Report(label="W", workspace=str(ws))

    run.harvest(codex, home, ws, report, since=0.0)
    assert run.harvested_since(ws) == 1000, "a later launch cannot find the watermark"
    run.harvest(codex, home, ws, report, since=run.harvested_since(ws))

    assert report.tool_calls == 1
    assert len((ws / "agent_stream.jsonl").read_text().splitlines()) == 1


def test_a_shell_that_could_not_start_is_told_apart_from_giving_up():
    """Not the agent's doing, and it does not get better with a retry: when the shell
    cannot start this CLI answers the prompt out of what it expected the commands to
    produce, which leaves a session that spoke and ran nothing."""
    codex = AGENTS["codex"]
    assert codex.no_commands(turns=3, tool_calls=0)
    assert not codex.no_commands(turns=3, tool_calls=9), "it ran things and stopped"
    assert not codex.no_commands(turns=0, tool_calls=0), "it never started"
    assert not AGENTS["claude"].no_commands(turns=3, tool_calls=0)


def test_a_codex_chain_harvests_every_session_and_sets_each_one_a_home(tmp_path, monkeypatch):
    """The loop, end to end, with the CLI faked. Each session is handed its own CLI's
    config directory under that CLI's own variable, is fenced, and has its history
    harvested before the next one starts — so every picture is counted once."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["tty"], "actions_used": 0, "terminal": False}))
    home = tmp_path / ".sessions" / "W" / ".codex"
    seen: list[tuple[list[str], dict]] = []

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += 10
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    async def fake_session(argv, where, env, report, agent):
        n = len(seen)
        seen.append((argv, env))
        history_rows = [(1000 + i, {"type": "imageView", "path": f"/p{i}.png"})
                        for i in range(n + 1)]
        (home / AGENTS["codex"].HISTORY).unlink(missing_ok=True)
        history(home, history_rows)
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w, o: None)
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    monkeypatch.setattr(run, "fenced_argv", lambda agent, argv, w, h: ["FENCE", *argv])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nobody"))
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "best_episode", "episodes_completed",
                        "mean_episode", "score", "lives", "deaths", "quits", "turns",
                        "unique_cells", "max_depth", "max_xplevel")}
        | {"episode_scores": [], "endings": [], "roles": []}))
    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["tty"], "budget": 500,
        "stint": 10, "max_sessions": 3, "fresh_world": False, "agent": "codex",
        "model": "gpt-5.6-sol", "dry_run": False, "retries": 2, "opening": "blind",
        "idle_wait": 0, "no_fence": False,
    })()
    report = asyncio.run(run.play("W", ("nethack", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))

    assert report.fenced and report.sessions == 3
    assert all(argv[0] == "FENCE" and argv[1] == "codex" for argv, _ in seen)
    assert all(env["CODEX_HOME"] == str(home) for _, env in seen)
    assert all("CLAUDE_CONFIG_DIR" not in env for _, env in seen)
    # Session n's history holds n pictures, one per session so far: three in all,
    # each harvested once.
    assert report.tool_calls == 3
    lines = (ws / "agent_stream.jsonl").read_text().splitlines()
    assert sorted(json.loads(line)["item"]["path"] for line in lines) == [
        "/p0.png", "/p1.png", "/p2.png"]


def test_no_fence_is_recorded_as_no_fence(tmp_path, monkeypatch):
    """`--no-fence` exists to find out what a session reaches for, and a report from
    one must not claim a fence it was not given."""
    assert AGENTS["codex"].FENCED, "the arm that reached on cc_craftax is unfenced here"
    assert not AGENTS["claude"].FENCED, "fencing this arm rewrites M6"
    assert run.Report(label="W", workspace="/w").fenced is False


def test_a_claude_run_can_be_fenced_and_says_so(tmp_path, monkeypatch):
    """M6 and the published Craftax pass stay unfenced, and nothing here changes
    that. But a *new* Claude model has no record of staying out of the package, and
    `--fence` lets one be asked for. The CLI is wrapped, and the report says which
    side of the asymmetry the run was on — an unfenced number and a fenced one are
    not the same measurement."""
    ws = tmp_path / "W"
    ws.mkdir()
    (ws / "state.json").write_text(json.dumps(
        {"obs": ["tty"], "actions_used": 0, "terminal": False}))
    seen: list[list[str]] = []

    async def fake_act(where, *argv):
        if argv[0] == "init":
            state = json.loads((ws / "state.json").read_text())
            state["actions_used"] += 10
            (ws / "state.json").write_text(json.dumps(state))
        return ""

    async def fake_session(argv, where, env, report, agent):
        seen.append(argv)
        return "", 0

    monkeypatch.setattr(run, "act", fake_act)
    monkeypatch.setattr(run, "one_session", fake_session)
    monkeypatch.setattr(run, "refresh_brief", lambda w, o: None)
    monkeypatch.setattr(run, "session_config", lambda r, label, agent=None: tmp_path / "cfg")
    monkeypatch.setattr(run, "audit_session", lambda *a, **k: run.Audit())
    monkeypatch.setattr(run, "fenced_argv", lambda agent, argv, w, h: ["FENCE", *argv])
    (tmp_path / ".envs" / "W").mkdir(parents=True)
    (tmp_path / ".envs" / "W" / "result.json").write_text(json.dumps(
        {f: 0 for f in ("actions_used", "best_episode", "episodes_completed",
                        "mean_episode", "score", "lives", "deaths", "quits", "turns",
                        "unique_cells", "max_depth", "max_xplevel")}
        | {"episode_scores": [], "endings": [], "roles": []}))
    args = type("A", (), {
        "carry_on": str(tmp_path), "replay": None, "obs": ["tty"], "budget": 500,
        "stint": 0, "max_sessions": 3, "fresh_world": False, "agent": "claude",
        "model": "claude-opus-5-5", "dry_run": False, "retries": 2, "idle_wait": 0,
        "opening": "named", "fence": True, "no_fence": False,
    })()
    report = asyncio.run(run.play("W", ("nethack", 0), tmp_path, args, {}, ["W"],
                                  asyncio.Semaphore(1)))
    assert report.fenced and seen and all(argv[0] == "FENCE" for argv in seen)
    assert seen[0][1] == "claude", "the fence wrapped something other than the CLI"


def test_fence_and_no_fence_cannot_both_be_asked_for():
    """They mean opposite things, so asking for both is a mistake worth a failure
    rather than a silent precedence rule."""
    done = subprocess.run([sys.executable, str(ROOT / "run.py"), "--fence", "--no-fence",
                           "--dry-run"], capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "not allowed with" in done.stderr


# --------------------------------------------------------------------------- #
# The fence
# --------------------------------------------------------------------------- #


def fenced(argv: list[str], ws: Path) -> subprocess.CompletedProcess:
    """Run `argv` under exactly the fence a codex session of this launch would get."""
    import fence

    if not fence.supported():
        pytest.skip("this kernel has no Landlock, so there is no fence to test")
    if not run.shutil.which("codex"):
        pytest.skip("codex is not installed, so there is no binary to fence")
    full = run.fenced_argv(AGENTS["codex"], ["codex"], ws, ws)
    wrapped = full[: full.index("--")] + ["--", *argv]
    return subprocess.run(wrapped, cwd=ws, capture_output=True, text=True, timeout=120)


def reads(path: Path, ws: Path) -> bool:
    """Whether a fenced session can read this file."""
    return fenced(["cat", str(path)], ws).returncode == 0


def test_a_fenced_session_cannot_reach_the_game_it_is_playing(tmp_path):
    """The reach this exists to close, named file by file: the package's second copy
    of the game and its data, this harness's floors and populations, the launcher."""
    ws = run.make_workspace(tmp_path, "W", ["tty"], "blind")
    package = run.site_packages(ROOT / ".env-venv") / "nle"
    assert package.exists(), "the package moved, so this test proves nothing"

    assert not reads(package / "__init__.py", ws), "the package is readable"
    assert not reads(ROOT / "tools" / "baselines.py", ws), "the floors are readable"
    assert not reads(ROOT / "run.py", ws), "the launcher is readable"
    # Its names are listable, as everything under `site-packages` is: a list-only
    # grant covers the hierarchy, which is what lets Python find `pydantic`. What is
    # shut is every file's contents, the game's data among them.
    assert not reads(package / "nethackdir" / "nhdat", ws), "the game's data is readable"
    assert fenced(["ls", str(Path.home())], ws).returncode != 0, "the home is walkable"
    blocked = fenced([str(run.bin_dir(tmp_path) / "env-python"), "-c", "import nle"], ws)
    assert blocked.returncode != 0, "the actuator's interpreter imports the game"


def test_the_fence_keeps_everything_a_session_plays_with(tmp_path):
    """Fenced too tight is a run that cannot play, and the failure would arrive as a
    session that scored nothing rather than as an error. `./act` is the whole of what
    a session does, so it is run for real: through the workspace's own shim, the
    launch's neutral names and the symlinked actuator, against a live game."""
    ws = run.make_workspace(tmp_path, "W", ["tty"], "blind")
    asyncio.run(run.act(ws, "init", "--variant", "nethack", "--seed", "0",
                        "--budget", "5", "--obs", "tty",
                        "--env-dir", str(tmp_path / ".envs" / "W")))
    try:
        played = fenced(["./act", "do", "s", "--plan", "look"], ws)
        assert played.returncode == 0, played.stdout + played.stderr
        status = fenced(["./act", "status"], ws)
        assert status.returncode == 0 and "1/5" in status.stdout, status.stdout
        done = fenced(["sh", "-c", "echo kept > notes.md && cat notes.md"], ws)
        assert done.stdout.strip() == "kept", "a session cannot keep notes"
        pictures = fenced(["./python", "-c", "import numpy, PIL.Image; print('ok')"], ws)
        assert pictures.stdout.strip() == "ok", pictures.stderr
    finally:
        asyncio.run(run.act(ws, "stop"))
