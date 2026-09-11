"""What the actuator promises: one key, one observation, and nothing else said.

`cc_humanrl`'s end-to-end check was the 199-action route that wins every game.
NetHack has no route and no win short of ascension, so what is checked here is the
protocol instead: a batch stops when the state it was planned against is gone, the
budget is enforced before anything reaches the game, a life ending is not the run
ending, and nothing the agent can read carries the run's own view of itself.

The reward is shown, as it was in the Craftax arm and for a stronger reason: it is
the change in a number NetHack itself draws on the status line, so withholding it
would hide something the screen already says.
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("nle")

import act  # noqa: E402
from act import RunState, Session, parse_tokens  # noqa: E402

# Two keys, and the game is over: `#quit` asks, `y` answers. It is the only ending
# a test can reach in constant time, and it is a key of the game rather than an
# addition to it (F14).
QUIT = ["#quit", "y"]


@pytest.fixture
def rig(tmp_path):
    """A workspace and, somewhere else entirely, the environment's directory."""

    def build(variant="nethack", budget=0, obs=("tty",), seed=3, fresh_world=False,
              blind=True):
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        env_dir = tmp_path / "env"
        env_dir.mkdir(exist_ok=True)
        state = RunState(
            variant=variant,
            seed=seed,
            obs=list(obs),
            fresh_world=fresh_world,
            blind=blind,
            budget=budget or act.DEFAULT_BUDGET[variant],
            env_dir=str(env_dir),
        )
        return Session.create(ws, state), ws, env_dir

    return build


def run(session, tokens, plan=None):
    args = type("A", (), {"actions": list(tokens), "plan": plan})()
    return act.cmd_do(args, session)


def blocks(ws):
    return (ws / act.LOG).read_text().split(act.SEP)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def test_a_run_is_played_in_batches(rig):
    session, ws, env_dir = rig(budget=40)
    out = run(session, ["l", "l", "j"], plan="walk into whatever is that way")
    assert "ran 3/3: l l j" in out
    assert session.state.actions_used == 3

    record = json.loads((env_dir / act.RESULT).read_text())
    assert record["history"] == ["l", "l", "j"]
    assert record["episodes"][0]["role"], "the character was not recorded"

    log = blocks(ws)
    assert len(log) == 5  # preamble+start, then one per key
    assert "plan: walk into whatever is that way" in log[2]
    assert "action 3 | budget 3/40" in log[-1]
    assert "[tty] screens/000003.txt" in log[-1]


@pytest.mark.parametrize(
    "obs", [("tty",), ("tty", "ansi"), ("tty", "pixels"), ("tty", "symbolic")]
)
def test_one_observation_per_action_per_channel(rig, obs):
    session, ws, _ = rig(budget=40, obs=obs)
    run(session, ["."] * 3)
    for channel in obs:
        folder, suffix = {
            "tty": ("screens", "txt"), "ansi": ("ansi", "txt"),
            "pixels": ("frames", "png"), "symbolic": ("symbolic", "npz"),
        }[channel]
        written = sorted(p.name for p in (ws / folder).iterdir())
        assert written == [f"{i:06d}.{suffix}" for i in range(4)]


def test_a_life_ending_stops_the_batch_and_the_next_life_begins(rig):
    session, ws, env_dir = rig(budget=40)
    out = run(session, [*QUIT, "l", "l"])
    assert "ran 2/4" in out
    assert "life over: quit" in out and "2 action(s) dropped" in out
    assert session.state.lives == 2 and not session.state.terminal

    block = blocks(ws)[-1]
    assert "[event] life over: quit" in block
    # Two observations: how the life ended, and what the next one began in.
    assert "[tty] screens/000002.txt" in block
    assert "[tty] screens/000002-restart.txt" in block
    assert (ws / "screens" / "000002-restart.txt").is_file()

    record = json.loads((env_dir / act.RESULT).read_text())
    assert record["episodes"][0]["ended"] == "quit"
    assert record["quits"] == 1 and record["deaths"] == 0


def test_a_life_ending_costs_only_what_was_spent(rig):
    session, _, _ = rig(budget=40)
    run(session, QUIT)
    assert session.state.actions_left == 38


def test_budget_is_refused_before_the_game_is_touched(rig):
    session, _, _ = rig(budget=10)
    with pytest.raises(SystemExit, match="only 10 left in the budget"):
        run(session, ["."] * 11)
    assert session.state.actions_used == 0


def test_budget_ends_the_run(rig):
    session, ws, _ = rig(budget=4)
    out = run(session, ["."] * 4)
    assert "budget spent" in out
    assert session.state.terminal and session.state.outcome == "budget"
    assert "[event] run over: budget" in blocks(ws)[-1]
    with pytest.raises(SystemExit, match="this run is over"):
        run(session, ["."])


def test_the_same_character_is_dealt_again_after_a_life_ends(rig):
    """The standing answer to the re-roll (F14).

    NetHack rolls a role, a race, an alignment and a gender, and they are not
    equally survivable — so a run judged on its best life, dealt a new character
    each time, would reward quitting until a good one turned up. Every life of a run
    is the same character in the same dungeon, so two keys of quitting buy nothing.
    """
    session, ws, env_dir = rig(budget=40)
    run(session, QUIT)
    record = json.loads((env_dir / act.RESULT).read_text())
    assert len(record["roles"]) == 2
    assert record["roles"][0] == record["roles"][1]
    opening = (ws / "screens" / "000000.txt").read_text()
    assert (ws / "screens" / "000002-restart.txt").read_text() == opening


def test_fresh_world_rolls_a_new_character(rig):
    session, ws, _ = rig(budget=40, fresh_world=True)
    run(session, QUIT)
    opening = (ws / "screens" / "000000.txt").read_text()
    assert (ws / "screens" / "000002-restart.txt").read_text() != opening


def test_the_curve_across_lives_is_kept(rig):
    """Score per life in order, which is the thing no RL baseline has an analogue
    for: the agent carries notes across deaths where a policy carries weights."""
    session, _, env_dir = rig(budget=40)
    run(session, ["l", "l", *QUIT])
    run(session, ["l"])
    record = json.loads((env_dir / act.RESULT).read_text())
    assert len(record["episode_scores"]) == 2
    assert record["episodes_completed"] == 1, "a truncated life is not an episode"
    assert record["endings"] == ["quit"]


# --------------------------------------------------------------------------- #
# What the workspace may and may not carry
# --------------------------------------------------------------------------- #


def test_state_json_carries_no_league_table_and_does_not_name_the_game(rig):
    """`variant` is the string "nethack", in a file that sits beside the log the
    session is told to read. Withholding the name from the brief and from the screen
    and leaving it here would be withholding nothing (F18)."""
    session, ws, _ = rig(budget=40)
    run(session, ["l", "l", *QUIT])
    saved = json.loads((ws / act.STATE).read_text())
    for leak in ("episodes", "score", "max_depth", "max_xplevel", "deaths",
                 "quits", "unique_cells", "turns", "env_dir", "variant", "blind"):
        assert leak not in saved, f"state.json carries {leak}"
    assert "nethack" not in (ws / act.STATE).read_text().lower()
    # The life's own return stays, because `status` shows it and the status line
    # shows the score anyway. The run total goes, because it is the number a session
    # would grow instead of playing.
    assert "life_reward" in saved and "reward" not in saved


def test_nothing_in_the_workspace_names_the_game(rig):
    """The whole point of the blind condition, checked over the workspace rather
    than over any one file that was supposed to hold the line."""
    session, ws, _ = rig(budget=40)
    run(session, ["l", "v", "cr", "l"])  # `v` is the version line, which names it
    for path in ws.rglob("*"):
        if path.is_file():
            assert "nethack" not in path.read_text(errors="replace").lower(), path


def test_a_named_run_lets_the_game_name_itself(rig):
    session, ws, _ = rig(budget=40, blind=False, seed=5)
    assert "NetHack" in (ws / "screens" / "000000.txt").read_text()


def test_the_log_says_nothing_about_the_observation(rig):
    session, ws, _ = rig(budget=40, obs=("tty", "pixels"))
    run(session, ["l", "l", *QUIT])
    log = (ws / act.LOG).read_text()
    for line in log.splitlines():
        if line.startswith("["):
            assert line.startswith(("[tty] screens/", "[pixels] frames/", "[event] "))
    # Nothing computed from a screen, and nothing named off it.
    for leak in ("dlvl", "hp:", "role", "hitpoints", "altar", "kobold"):
        assert leak not in log.lower()


def test_the_preamble_names_the_keys_and_explains_nothing(rig):
    session, ws, _ = rig(budget=40)
    said = (ws / act.LOG).read_text()
    for key in ("k", "h", ">", "cr", "esc", "#pray"):
        assert f" {key} " in said or f" {key}\n" in said
    # `#pray` and `#sit` are the game's own names for those keys, so they are in
    # the list. What must not be there is anything explaining what a key *does*.
    for explanation in ("north", "west", "descend", "staircase", "attack", "dungeon"):
        assert explanation not in said.lower()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_repeat_syntax():
    playable = ("h", "H", "cr", "#pray", "*")
    assert parse_tokens(["h", "cr*3", "#pray"], playable) == [
        "h", "cr", "cr", "cr", "#pray"
    ]
    assert parse_tokens(["*"], playable) == ["*"], "a key that is also the repeat mark"
    with pytest.raises(SystemExit, match="not a key of this game"):
        parse_tokens(["left"], playable)
    with pytest.raises(SystemExit, match="want 1 to 500"):
        parse_tokens(["h*0"], playable)
    with pytest.raises(SystemExit, match="no keys given"):
        parse_tokens([], playable)


def test_case_is_not_folded():
    """`H` runs west and `h` steps west. Every earlier port of this lowercased."""
    playable = ("h", "H", "s", "S")
    assert parse_tokens(["H*2"], playable) == ["H", "H"]
    with pytest.raises(SystemExit, match="did you mean h"):
        parse_tokens(["H"], ("h", "s"))  # a game whose keys do not include `H`


# --------------------------------------------------------------------------- #
# The daemon, end to end
# --------------------------------------------------------------------------- #


def act_cli(ws: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), *args],
        cwd=ws, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_daemon_round_trip(tmp_path):
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--seed", "3", "--env-dir", str(env_dir), "--budget", "20")
        assert (ws / "screens" / "000000.txt").is_file()
        out = act_cli(ws, "do", "l*3", "--plan", "walk")
        assert "ran 3/3: l l l" in out
        assert "budget 3/20" in act_cli(ws, "status")
        assert "screens/000003.txt" in act_cli(ws, "board")
        record = json.loads((env_dir / act.RESULT).read_text())
        assert record["variant"] == "nethack" and record["seed"] == 3
        # NetHack's own recording of the life, beside the environment and not in
        # the workspace: the xlogfile names how a life ended and what it scored.
        assert list((env_dir / "ttyrec").glob("*.ttyrec3.bz2"))
        assert not (ws / act.RESULT).exists()
        assert not list(ws.glob("*.ttyrec*"))

        # A client that hangs up before its reply. Craftax's first pilot was killed
        # mid-batch, `sendall` raised BrokenPipeError out of the accept loop, the
        # daemon exited, `act stop` then failed, and run.py discarded the report of
        # a 2757-action run because its shutdown was untidy.
        rude = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        rude.connect(str(ws / act.SOCKET))
        rude.sendall(json.dumps({"command": "status", "args": {}}).encode() + b"\n")
        rude.close()
        assert "budget 3/20" in act_cli(ws, "status"), "a rude client killed the daemon"
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_again_rotates_the_observations_with_the_log(tmp_path):
    """Two runs interleaved would parse as neither — screens/000012.txt must not
    mean two different states."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--env-dir", str(env_dir), "--budget", "20")
        act_cli(ws, "do", ".*2")
        (ws / "notes.md").write_text("what I worked out")
        act_cli(ws, "init", "--env-dir", str(env_dir), "--budget", "20", "--again")
        assert (ws / "logs-attempt1.txt").is_file()
        assert (ws / "screens-attempt1" / "000002.txt").is_file()
        assert sorted(p.name for p in (ws / "screens").iterdir()) == ["000000.txt"]
        assert (ws / "notes.md").read_text() == "what I worked out", "memory was cleared"
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


# --------------------------------------------------------------------------- #
# One run, several sessions
# --------------------------------------------------------------------------- #
# A run of 30,000 keys is longer than any one agent session, so it is played by
# several. Two things make that a continuation rather than a series of restarts: a
# stint, which ends a session without ending the run, and a rebuild, which recovers
# the game from the record because the game is never saved anywhere.


def stinted(tmp_path, size, budget=20, seed=3):
    """A session whose stint is set before the log is written, as `serve` does it."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(variant="nethack", seed=seed, obs=["tty"], budget=budget,
                     env_dir=str(env_dir))
    state.begin_stint(size)
    return Session.create(ws, state), ws, env_dir


def test_a_stint_ends_the_session_and_not_the_run(tmp_path):
    session, ws, _ = stinted(tmp_path, 5)
    out = run(session, [".*5"])
    assert "stint spent" in out
    assert session.state.stint_over and not session.state.terminal
    assert session.state.actions_left == 15, "a spent stint took the budget with it"
    with pytest.raises(SystemExit, match="your stint is spent"):
        run(session, ["."])
    assert "[event] session over" in blocks(ws)[-1]
    assert "this session is over" in act.cmd_status(None, session)


def test_a_batch_larger_than_the_stint_is_refused_before_the_game_is_touched(tmp_path):
    session, _, _ = stinted(tmp_path, 5)
    with pytest.raises(SystemExit, match="only 5 left in your stint"):
        run(session, [".*6"])
    assert session.state.actions_used == 0


def test_a_stint_that_runs_out_with_the_budget_is_the_run_ending(tmp_path):
    session, _, _ = stinted(tmp_path, 5, budget=5)
    out = run(session, [".*5"])
    assert session.state.terminal and session.state.outcome == "budget"
    assert not session.state.stint_over
    assert "budget spent" in out


def test_the_preamble_explains_a_stint_only_when_there_is_one(tmp_path):
    _, stint_ws, _ = stinted(tmp_path / "a", 5)
    _, plain_ws, _ = stinted(tmp_path / "b", 0)
    said = (stint_ws / act.LOG).read_text()
    assert "# stint: 5 of those actions are yours" in said
    assert "notes.md" in said, "the session was not told how to talk to its successor"
    assert "# stint:" not in (plain_ws / act.LOG).read_text()


def test_a_rebuilt_run_is_the_same_run(tmp_path):
    """The game is never saved and does not have to be.

    NetHack is a pure function of its two seeds and the keys pressed once the
    anti-TAS reseeding is off, so replaying the record into a fresh game must land
    on the same state. The screen is where that is checked rather than the counters:
    it is drawn from the whole of the game — the map, the inventory, the position,
    the hunger clock — so two that match cannot have come from two different states.
    """
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    common = ("--seed", "3", "--env-dir", str(env_dir), "--budget", "20", "--stint", "6")
    try:
        act_cli(ws, "init", *common)
        act_cli(ws, "do", "l*2", "j*2", "cr", "k")
        assert "this session is over" in act_cli(ws, "status")
        was = json.loads((env_dir / act.RESULT).read_text())

        act_cli(ws, "init", *common, "--resume")
        now = json.loads((env_dir / act.RESULT).read_text())
        assert (ws / "screens" / "000006.txt").read_text() == (
            ws / "screens" / "000006-resume.txt"
        ).read_text(), "the rebuilt game is not the game the record was made in"
        for field in ("history", "actions_used", "lives", "score", "episode_scores",
                      "roles", "unique_cells", "max_depth", "deaths", "best_episode"):
            assert now[field] == was[field], field
        assert now["sessions"] == 2, "the rebuild did not count as a new session"
        assert not now["terminal"] and now["stint_end"] == 12

        act_cli(ws, "do", "h*2")
        assert json.loads((env_dir / act.RESULT).read_text())["history"] == (
            was["history"] + ["h", "h"]
        )
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_a_rebuilt_run_replays_its_deaths_too(tmp_path):
    """A run with several lives is several games, and the seeds for life *n* are a
    function of *n* — so the rebuild has to land in the right life, not just the
    right number of keys."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    common = ("--seed", "3", "--env-dir", str(env_dir), "--budget", "20")
    try:
        act_cli(ws, "init", *common)
        act_cli(ws, "do", "l", "#quit", "y")
        act_cli(ws, "do", "l", "l")
        was = json.loads((env_dir / act.RESULT).read_text())
        assert was["lives"] == 2 and was["endings"] == ["quit"]

        act_cli(ws, "init", *common, "--resume")
        now = json.loads((env_dir / act.RESULT).read_text())
        assert now["lives"] == 2 and now["episodes"] == was["episodes"]
        assert (ws / "screens" / "000005.txt").read_text() == (
            ws / "screens" / "000005-resume.txt"
        ).read_text()
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_a_rebuilt_run_keeps_what_it_has_already_written(tmp_path):
    """`--again` rotates the log and the observations because the action numbering
    starts over. `--resume` must not: the numbering continues, and rotating would
    hide the run's own first half from the session continuing it."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    common = ("--seed", "3", "--env-dir", str(env_dir), "--budget", "20")
    try:
        act_cli(ws, "init", *common)
        act_cli(ws, "do", ".*2")
        before = (ws / act.LOG).read_text()
        act_cli(ws, "init", *common, "--resume")
        after = (ws / act.LOG).read_text()
        assert after.startswith(before), "the log was rewritten rather than continued"
        assert "[event] resumed:" in after
        assert not list(ws.glob("logs-attempt*.txt")) and not list(ws.glob("screens-*"))
        assert (ws / "screens" / "000002.txt").is_file()
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_a_rebuilt_run_takes_its_game_from_the_record_not_the_arguments(tmp_path):
    """Variant, seed, channels, fresh_world and blindness define the game the history
    is a history of. A launcher that disagreed about any of them would rebuild a
    different game and call it the same run — and the record beside the environment
    is where they live, because two of the five are what the workspace withholds."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--seed", "3", "--obs", "tty", "--env-dir", str(env_dir),
                "--budget", "20")
        act_cli(ws, "do", ".")
        act_cli(ws, "init", "--variant", "score", "--seed", "7", "--obs", "symbolic",
                "--named", "--env-dir", str(env_dir), "--budget", "20", "--resume")
        now = json.loads((env_dir / act.RESULT).read_text())
        assert now["variant"] == "nethack" and now["seed"] == 3
        assert now["obs"] == ["tty"] and now["blind"] is True
        assert "NetHack" not in (ws / "screens" / "000001-resume.txt").read_text()
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_resuming_without_the_record_is_refused(tmp_path):
    """The record beside the environment is what says which game this is a run of,
    now that `state.json` does not."""
    ws, env_dir = tmp_path / "ws", tmp_path / "env"
    ws.mkdir()
    try:
        act_cli(ws, "init", "--seed", "3", "--env-dir", str(env_dir), "--budget", "20")
        act_cli(ws, "do", ".")
        act_cli(ws, "stop")
        lost = subprocess.run(
            [sys.executable, str(ROOT / "act.py"), "init", "--resume",
             "--env-dir", str(tmp_path / "elsewhere"), "--budget", "20"],
            cwd=ws, capture_output=True, text=True, timeout=300,
        )
        said = lost.stdout + lost.stderr
        assert lost.returncode != 0 and "needs the --env-dir" in said
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_resuming_is_refused_when_there_is_nothing_to_resume_or_no_room_to(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    empty = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), "init", "--resume"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    assert empty.returncode != 0 and "no run to resume" in empty.stdout + empty.stderr

    env_dir = tmp_path / "env"
    try:
        act_cli(ws, "init", "--env-dir", str(env_dir), "--budget", "20")
        act_cli(ws, "do", ".*4")
        act_cli(ws, "stop")
        small = subprocess.run(
            [sys.executable, str(ROOT / "act.py"), "init", "--resume",
             "--env-dir", str(env_dir), "--budget", "3"],
            cwd=ws, capture_output=True, text=True, timeout=300,
        )
        said = small.stdout + small.stderr
        assert small.returncode != 0 and "at least as large as what is already spent" in said
    finally:
        subprocess.run([sys.executable, str(ROOT / "act.py"), "stop"], cwd=ws, timeout=60)


def test_again_and_resume_are_opposites(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    clash = subprocess.run(
        [sys.executable, str(ROOT / "act.py"), "init", "--again", "--resume"],
        cwd=ws, capture_output=True, text=True, timeout=120,
    )
    assert clash.returncode != 0
    assert "starts the world over" in clash.stdout + clash.stderr


def test_the_record_is_replaced_and_never_written_in_place(rig, monkeypatch):
    """The launcher builds its whole report out of result.json, and a 30,000-key run
    rewrites it once per action — thirty thousand chances to be killed mid-write.
    A half-written one would lose the run it was recording."""
    session, _, env_dir = rig(budget=10)
    target = env_dir / act.RESULT
    seen, real = [], Path.write_text

    def spy(self, *a, **kw):
        seen.append(Path(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", spy)
    run(session, [".*2"])
    assert seen, "nothing was written at all"
    assert target not in seen, "result.json was written in place"
    assert json.loads(target.read_text())["actions_used"] == 2
    assert not list(env_dir.glob("*.tmp")), "a temporary file was left behind"
