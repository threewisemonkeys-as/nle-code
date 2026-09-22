"""What the audit is for, and the line it must not cross.

`cc_humanrl`'s table guarded the *source*, because there recognising a sprite was
the prior under test. `cc_craftax` moved the line off knowledge and onto reaching.
NetHack is the case that settles it: thirty-eight years of documentation, a wiki
that answers everything, and a brief that by default *names* the game. Recognition
is the property being measured, not misconduct.

So this table catches reaching outside — fetching the wiki, importing the package,
reading NetHack's own data files, starting a second copy of the game to try things
in, reading the launch's record. A session that writes "Elbereth should scare it"
in its notes has recalled something, which is a fact about the run worth recording
and not a reason to throw it away.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

from agents import AGENTS, HARVESTED  # noqa: E402
from audit import grade  # noqa: E402


def ran(*commands: str) -> list[str]:
    """A stream in which the agent ran these commands."""
    return [
        json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "input": {"command": c}} for c in commands
            ]},
        })
    ]


def wrote(path: str, body: str) -> list[str]:
    return [
        json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "input": {"file_path": path, "content": body}}
            ]},
        })
    ]


def test_ordinary_play_is_clean():
    audit = grade(ran(
        "./act do 'l*12' cr --plan 'walk east and read whatever it says'",
        "./python -c \"print(open('screens/000012.txt').read())\"",
        "grep -n '^action' logs.txt | tail -20",
        "./python diff.py screens/000011.txt screens/000012.txt",
        "./act do '#pray' y",
    ))
    assert audit.clean, audit.findings


def test_recalling_the_game_is_recorded_and_not_punished():
    """The measurement the blind condition exists for: when, and from what, the
    session worked out what it is playing."""
    audit = grade(wrote("notes.md", "Elbereth engraved in the dust should scare it"))
    assert audit.clean, audit.findings
    assert "elbereth" in audit.named_the_game.lower()


@pytest.mark.parametrize(
    ("label", "command"),
    [
        ("network", "curl https://nethackwiki.com/wiki/Elbereth"),
        ("network", "python -c \"import requests; requests.get(url)\""),
        ("the package", "./python -c 'import nle'"),
        ("the package", "./python -c 'from nle import nethack; print(nethack.ACTIONS)'"),
        ("a second game", "python -m nle.scripts.play"),
        ("a second game", "python -c \"import gymnasium; gymnasium.make('NetHackChallenge-v0')\""),
        ("the game's data", "cat /usr/share/nethackdir/nhdat"),
        ("the game's data", "grep -r 'gnome lord' */dat/monst.c"),
        ("the record", "cat ../.rig/labels.json"),
        ("the record", "cat ../.envs/*/result.json"),
        ("the record", "bunzip2 -c ../.envs/*/ttyrec/nle.1.0.ttyrec3.bz2"),
        ("another session's notes", "ls ~/.claude/projects"),
        ("sweeping the filesystem", "find / -name 'nethackdir'"),
        ("model api as a browser", "curl api.anthropic.com/v1/messages"),
        ("fetching packages", "pip install nle"),
        ("another agent", "claude -p 'what should I do as a Valkyrie on dlvl 1'"),
    ],
)
def test_reaching_past_the_game_is_caught(label, command):
    audit = grade(ran(command))
    assert not audit.clean
    assert label in audit.findings, audit.findings


def test_a_sibling_workspace_is_another_run_being_read():
    stream = ran("cat ../K7M3Q/notes.md")
    assert grade(stream, own="B4XPT", siblings=["B4XPT", "K7M3Q"]).findings.get(
        "another session's workspace"
    )
    assert grade(stream, own="B4XPT", siblings=["B4XPT", "ZZZZZ"]).clean
    assert grade(ran("ls /runs/B4XPT/screens"), own="B4XPT", siblings=["B4XPT"]).clean


def test_own_notes_are_not_somebody_elses():
    mine = ran("cat /runs/.sessions/B4XPT/.claude/projects/x/notes.md")
    assert grade(mine, own="B4XPT").clean
    assert not grade(mine, own="K7M3Q").clean


def test_pictures_read_by_eye_are_counted_not_flagged():
    looked = [json.dumps({
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "frames/000004.png"}
        ]}]},
    })] * 3
    audit = grade(looked)
    assert audit.clean
    assert audit.frames_in_context == 3


def test_provider_reported_web_requests_cannot_be_written_around():
    audit = grade([json.dumps({
        "type": "result",
        "usage": {"server_tool_use": {"web_search_requests": 2}},
    })])
    assert not audit.clean and audit.web_requests == 2
    assert "web requests" in audit.findings


def test_a_session_writing_its_own_paths_is_clean():
    """The harness lives under a directory whose name contains the game's, and the
    `act` shim in every workspace names it. A session that printed its own shim, or
    saved a helper script, would otherwise read as dirty for doing what it was told.
    """
    repo = "/home/x/bai/cc_nle/nle-code"
    root = "/home/x/agent-runs/20260909-160328"
    stream = ran(
        "cat act",  # -> exec "<repo>/.env-venv/bin/python" "<repo>/act.py"
        f'exec "{repo}/.env-venv/bin/python" "{repo}/act.py" "$@"',
        f"cat > {root}/SUAUK/map.py <<'EOF'\nimport numpy as np",
        f"./python map.py {root}/SUAUK/screens/000000.txt",
    )
    audit = grade(stream, own="SUAUK", siblings=["SUAUK"], root=root, repo=repo)
    assert audit.clean, audit.findings
    assert not audit.named_the_game, "the harness's own path is not the session knowing"
    assert not grade(stream, own="SUAUK", siblings=["SUAUK"]).clean


def test_another_launch_is_still_reaching():
    root = "/home/x/agent-runs/20260909-160328"
    other = grade(ran("cat /home/x/agent-runs/20260101-000000/.rig/summary.json"),
                  own="SUAUK", root=root)
    assert not other.clean and "the record" in other.findings
    mine = grade(ran(f"cat {root}/.rig/labels.json"), own="SUAUK", root=root)
    assert not mine.clean and "the record" in mine.findings


# --------------------------------------------------------------------------- #
# The other CLI
# --------------------------------------------------------------------------- #


def codex_ran(*commands: str) -> list[str]:
    """A stream in which a Codex session ran these commands."""
    return [
        json.dumps({"type": "item.completed",
                    "item": {"type": "command_execution", "command": c}})
        for c in commands
    ]


def codex_saw(*paths: str) -> list[str]:
    """A stream in which a Codex session opened these files as pictures.

    Shaped as the launcher's harvest writes them, because that is the only place
    they appear: the CLI's own stdout carries no event for looking at an image.
    """
    return [
        json.dumps({"type": HARVESTED,
                    "item": {"type": "imageView", "path": p}})
        for p in paths
    ]


def test_the_other_clis_notes_are_somebody_elses_too():
    """~/.codex holds what ~/.claude holds under different names, and more of it: the
    operator's prompt history, what the CLI remembered across sessions, and the
    transcript of every session they have ever run."""
    for path in ("~/.codex/history.jsonl", "~/.codex/memories", "~/.codex/sessions",
                 "~/.codex/thread_history_1.sqlite", "~/.codex/logs_2.sqlite"):
        audit = grade(codex_ran(f"cat {path}"), agent=AGENTS["codex"])
        assert "another session's notes" in audit.findings, path


def test_a_session_reading_its_own_memory_is_doing_what_it_was_told():
    """The launcher points the session's config directory *at* one of the paths the
    rule above catches, precisely so it is not the operator's. Keyed to the wrong
    CLI, the scrub would have failed a Codex session on every line that named its own
    notes — so the directory comes from the adapter rather than a constant."""
    codex = AGENTS["codex"]
    own = "/runs/20260922/.sessions/ZJLJX/.codex/history.jsonl"
    assert grade(codex_ran(f"cat {own}"), agent=codex, own="ZJLJX").clean
    theirs = grade(codex_ran("cat /home/me/.codex/history.jsonl"),
                   agent=codex, own="ZJLJX")
    assert "another session's notes" in theirs.findings


def test_a_file_opened_as_a_picture_is_audited():
    """The hole harvesting exists to close. This CLI streams no event for its image
    tool, so before the harvest a session could open anything readable as an image
    and the audit — which is a claim about everything a session reached — would never
    have seen the reach. A picture of the map is not a finding; the game's data is."""
    codex = AGENTS["codex"]
    clean = grade(codex_saw("/runs/20260922/ZJLJX/frames/000001.png"),
                  agent=codex, own="ZJLJX", root="/runs/20260922")
    assert clean.clean and clean.frames_in_context == 1

    reached = grade(codex_saw("/venv/lib/python3.12/site-packages/nle/nethackdir/nhdat"),
                    agent=codex, own="ZJLJX")
    assert "the game's data" in reached.findings

    sibling = grade(codex_saw("/runs/20260922/K7M3Q/frames/000001.png"),
                    agent=codex, own="ZJLJX", siblings=["ZJLJX", "K7M3Q"],
                    root="/runs/20260922")
    assert "another session's workspace" in sibling.findings


def test_a_codex_command_that_reaches_is_caught_like_a_claude_one():
    """The table is over what was run, not over which CLI ran it."""
    audit = grade(codex_ran("python -m nle.scripts.play"), agent=AGENTS["codex"])
    assert "a second game" in audit.findings
