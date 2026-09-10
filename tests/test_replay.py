"""The replay page: what it claims to show has to be what the run saw.

The page is built from the keystroke history rather than from the screens on disk,
which is what lets it carry colour and score for a run played with `--obs tty` — and
which means the one thing worth pinning is that the two agree. So the fold the
browser does is done here in Python and compared against the files the actuator
wrote at the time.
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
import replay  # noqa: E402


@pytest.fixture
def launch(tmp_path):
    """One workspace inside a launch, played key by key with real batches."""

    def build(batches, seed=3, budget=200):
        root = tmp_path / "launch"
        ws, env_dir = root / "AAAAA", root / ".envs" / "AAAAA"
        ws.mkdir(parents=True)
        env_dir.mkdir(parents=True)
        (root / ".rig").mkdir()
        (root / ".rig" / "labels.json").write_text(
            json.dumps({"AAAAA": {"variant": "nethack", "seed": seed}})
        )
        state = act.RunState(seed=seed, obs=["tty"], budget=budget, env_dir=str(env_dir))
        session = act.Session.create(ws, state)
        for keys, plan in batches:
            args = type("A", (), {"actions": list(keys), "plan": plan})()
            act.cmd_do(args, session)
        return root, ws, env_dir

    return build


def fold(built: dict) -> list[list[str]]:
    """The browser's own reconstruction, in Python: apply each delta in turn."""
    text = [" " * replay.COLS] * replay.ROWS
    out = []
    for delta in built["screens"]:
        text = list(text)
        for row, line, _hue in delta:
            text[row] = line
        out.append(text)
    return out


def test_the_page_shows_the_screens_the_run_actually_saw(launch):
    """The fold has to land on the files the actuator wrote at the time — otherwise
    the page is a plausible reconstruction of a different game."""
    root, ws, env_dir = launch([(["l", "l", "j"], "east then south"), (["cr", "k"], None)])
    built = replay.one_run(ws, env_dir)
    screens = fold(built)
    assert len(screens) == 6, "one screen before anything was played, then one each"
    for index, rows in enumerate(screens):
        written = (ws / "screens" / f"{index:06d}.txt").read_text().splitlines()
        assert [line.rstrip() for line in rows] == [line.rstrip() for line in written]


def test_a_delta_carries_only_what_changed(launch):
    """The whole reason a NetHack page inlines where the Craftax one could not."""
    root, ws, env_dir = launch([(["l"] * 12, "walk east")])
    built = replay.one_run(ws, env_dir)
    later = built["screens"][2:]
    assert all(len(delta) < replay.ROWS for delta in later)
    average = sum(len(delta) for delta in later) / len(later)
    assert average < 6, f"{average} rows change per key — that is not a delta"


def test_batches_are_found_by_the_step_counter_and_not_by_the_plan_text(launch):
    """`--plan` is sticky: the actuator writes the standing plan into every block
    until a new one is given. Splitting on the text would read a 12-key batch as
    twelve batches and scatter the work between them."""
    root, ws, env_dir = launch([
        (["l", "l", "l"], "three east"),
        (["j", "j"], None),          # no plan of its own — the last one still stands
        (["k"], "back north"),
    ])
    said = replay.plans(ws / "logs.txt")
    assert sorted(said) == [1, 4, 6], said
    assert said[1] == "three east" and said[6] == "back north"
    assert said[4] == "three east", "a batch with no plan of its own keeps the standing one"

    built = replay.one_run(ws, env_dir)
    assert len(built["plans"]) == 3
    # Every step belongs to the batch it was played in, the opening screen to none.
    assert [step["plan"] for step in built["steps"]] == [-1, 0, 0, 0, 1, 1, 2]


def test_a_life_ending_gets_the_screen_it_ended_on_and_the_one_after(launch):
    root, ws, env_dir = launch([(["#quit", "y"], "end it"), (["l"], "carry on")])
    built = replay.one_run(ws, env_dir)
    keys = [step["key"] for step in built["steps"]]
    assert keys == ["start", "#quit", "y", "(new life)", "l"]
    assert built["steps"][2]["ended"] == "quit"
    assert built["lives"] == 2


def test_keys_thrown_into_a_prompt_are_marked(launch):
    def drive():
        cycle = ("l", "j", "h", "k", "u", "n", "b", "y")
        return [cycle[i % len(cycle)] for i in range(22)]

    root, ws, env_dir = launch([(drive(), "circle"), (["l", "cr"], "into the prompt")],
                               seed=4, budget=100)
    built = replay.one_run(ws, env_dir)
    assert built["wasted"] >= 1
    assert any(step["wasted"] for step in built["steps"])


def test_the_page_is_one_self_contained_file(launch, tmp_path):
    root, ws, env_dir = launch([(["l", "l"], "east")])
    out = tmp_path / "replay.html"
    sys.argv = ["replay", str(root), "--out", str(out)]
    assert replay.main() == 0
    page = out.read_text()
    assert "__DATA__" not in page and "__CSS__" not in page
    assert page.count("<script>") == 1
    assert "AAAAA" in page
    # No file:// dependency of any kind: everything but the webfont is inline.
    assert "src=" not in page
