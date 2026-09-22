"""The watcher: a page that is current while the run is still playing.

Two things are worth pinning about a process meant to be left alone for twenty hours.
It has to know when to stop — a watcher that never exits is a watcher someone has to
remember to kill — and it has to survive a build that throws, because the run outlasts
any one transient and a watcher that exits on the first is a watcher that was not
there for the other nineteen hours.
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
import watch  # noqa: E402


@pytest.fixture
def played(tmp_path):
    """A launch with a few keys in it, under a budget the caller chooses."""

    def build(keys, budget):
        root = tmp_path / "launch"
        ws, env_dir = root / "AAAAA", root / ".envs" / "AAAAA"
        ws.mkdir(parents=True)
        env_dir.mkdir(parents=True)
        (root / ".rig").mkdir()
        (root / ".rig" / "labels.json").write_text(
            json.dumps({"AAAAA": {"variant": "nethack", "seed": 3}})
        )
        state = act.RunState(seed=3, obs=["tty"], budget=budget, env_dir=str(env_dir))
        session = act.Session.create(ws, state)
        args = type("A", (), {"actions": list(keys), "plan": "walk"})()
        act.cmd_do(args, session)
        return root

    return build


def test_a_run_with_budget_left_is_still_live(played, tmp_path):
    root = played(["l", "l", "j"], budget=200)
    live, said = watch.once(root, tmp_path / "replay.html")
    assert live, "3 of 200 keys spent, and the watcher thinks the run is over"
    assert "3/200 keys" in said
    assert (tmp_path / "replay.html").exists()


def test_a_run_that_has_spent_its_budget_is_not(played, tmp_path):
    """This is the whole stopping condition, and it is the record's own number rather
    than the socket: the launcher stops the daemon between stints, so a live run has
    no socket for the seconds it takes one session to hand over to the next."""
    root = played(["l", "l", "j"], budget=3)
    live, said = watch.once(root, tmp_path / "replay.html")
    assert not live, said


def test_a_build_that_throws_does_not_end_the_watch(played, tmp_path, monkeypatch,
                                                    capsys):
    """The run outlasts any single transient — a half-written record, a file being
    rotated. The watcher reports it and comes back."""
    # Exactly the budget, so the second build is also the last one: a watcher whose
    # stopping condition never fires is a watcher that runs until the box does, and
    # this test would hang rather than fail.
    root = played(["l", "l"], budget=2)
    out = tmp_path / "replay.html"
    calls = {"n": 0}
    real = watch.replay.build

    def flaky(where):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("the record was being rewritten")
        if calls["n"] > 4:
            raise SystemExit("the watcher did not stop when the run did")
        return real(where)

    monkeypatch.setattr(watch.replay, "build", flaky)
    monkeypatch.setattr(watch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv",
                        ["watch", str(root), "--out", str(out), "--every", "0"])
    assert watch.main() == 0
    said = capsys.readouterr().out
    assert "build failed" in said and "carrying on" in said
    assert calls["n"] == 2, "it stopped on the failure instead of trying again"
    assert out.exists(), "the second build never wrote the page"


def test_the_page_is_replaced_whole_or_not_at_all(played, tmp_path):
    """A reader has the last build open while the next one is written. The rename is
    what makes that safe, and the scratch file must not survive it."""
    root = played(["l", "l"], budget=200)
    out = tmp_path / "pages" / "replay.html"
    watch.once(root, out)
    first = out.read_text()
    watch.once(root, out)
    assert out.read_text().count("<script>") == 1
    assert len(first) > 1000
    assert not list(out.parent.glob("*.part"))
