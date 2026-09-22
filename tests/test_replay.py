"""The replay page: what it claims to show has to be what the run saw.

The page is built from the keystroke history rather than from the screens on disk,
which is what lets it carry colour and score for a run played with `--obs tty` — and
which means the one thing worth pinning is that the two agree. So the fold the
browser does is done here in Python and compared against the files the actuator
wrote at the time.

Two encodings sit between the run and the page — deltas against the screen before,
and a run-length coding of the colours — and each of them is a way for a page to be
a plausible picture of a different game, so each is checked against the files rather
than against itself.
"""

import json
import re
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


def unrle(code: str) -> str:
    """The browser's colour decoder, in Python."""
    return "".join(letter * int(count) for letter, count in re.findall(r"([A-P])(\d+)", code))


def fold(built: dict) -> tuple[list[list[str]], list[list[str]]]:
    """The browser's own reconstruction, in Python: apply each delta in turn."""
    text = [""] * replay.ROWS
    hue = ["H" * replay.COLS] * replay.ROWS
    screens, colours = [], []
    for delta in built["screens"]:
        text, hue = list(text), list(hue)
        for row, line, code in delta:
            text[row] = line
            hue[row] = unrle(code)
        screens.append(text)
        colours.append(hue)
    return screens, colours


def test_the_page_shows_the_screens_the_run_actually_saw(launch):
    """The fold has to land on the files the actuator wrote at the time — otherwise
    the page is a plausible reconstruction of a different game."""
    root, ws, env_dir = launch([(["l", "l", "j"], "east then south"), (["cr", "k"], None)])
    built = replay.one_run(ws, env_dir)
    screens, _ = fold(built)
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


def test_every_colour_survives_the_run_length_coding(launch):
    """The coding is only safe because a colour is a letter and a count is digits.

    Hex digits would have collided — `7` then `40` then `1` is not decodable — and the
    failure mode is silent: the page would draw a different colour rather than throw.
    """
    for value in range(256):
        assert replay.colour_char(value) in "ABCDEFGHIJKLMNOP"
    assert unrle(replay.rle("A" * 80)) == "A" * 80
    assert replay.rle("A" * 40 + "B" * 2 + "A" * 38) == "A40B2A38"

    root, ws, env_dir = launch([(["l", "j", "l"], "about")])
    built = replay.one_run(ws, env_dir)
    _, colours = fold(built)
    assert all(len(row) == replay.COLS for screen in colours for row in screen)
    # A NetHack screen is not monochrome, and a coding that lost the colours would
    # still decode to the right *length*.
    assert len({cell for row in colours[-1] for cell in row}) > 1


def test_the_status_line_becomes_the_meters(launch):
    """The condition panel is read off `blstats` at each step and not off the
    episode's running maxima: a maximum of hit points draws a character who never
    bled, which is the one thing the panel exists to show."""
    root, ws, env_dir = launch([(["l"] * 6, "east")])
    built = replay.one_run(ws, env_dir)
    steps = built["steps"]
    assert steps["hpmax"][1] > 0 and steps["hp"][1] <= steps["hpmax"][1]
    assert len(steps["hp"]) == len(steps["n"]) == len(steps["score"])
    screen = (ws / "screens" / "000001.txt").read_text()
    assert f"HP:{steps['hp'][1]}({steps['hpmax'][1]})" in screen
    assert f"AC:{steps['ac'][1]}" in screen


def test_batches_are_found_by_the_step_counter_and_not_by_the_plan_text(launch):
    """`--plan` is sticky: the actuator writes the standing plan into every block
    until a new one is given. Splitting on the text would read a 12-key batch as
    twelve batches and scatter the work between them."""
    root, ws, env_dir = launch([
        (["l", "l", "l"], "three east"),
        (["j", "j"], None),          # no plan of its own — the last one still stands
        (["k"], "back north"),
    ])
    said, hands = replay.read_log(ws / "logs.txt")
    assert sorted(said) == [1, 4, 6], said
    assert said[1] == "three east" and said[6] == "back north"
    assert said[4] == "three east", "a batch with no plan of its own keeps the standing one"
    assert hands == [], "nothing handed over in a single-session run"

    built = replay.one_run(ws, env_dir)
    assert len(built["plans"]) == 3
    # Every step belongs to the batch it was played in, the opening screen to none.
    assert built["steps"]["batch"] == [-1, 0, 0, 0, 1, 1, 2]


def test_a_life_ending_gets_the_screen_it_ended_on_and_the_one_after(launch):
    root, ws, env_dir = launch([(["#quit", "y"], "end it"), (["l"], "carry on")])
    built = replay.one_run(ws, env_dir)
    assert built["steps"]["key"] == ["start", "#quit", "y", "(new life)", "l"]
    assert "2" in built["steps"]["ended"], "the step the life ended on is marked"
    assert built["lives"] == 2


def test_a_new_life_does_not_relist_where_every_life_starts(launch):
    """Dungeon level 1 and experience level 1 are where a life begins, not something
    that happened in it — and a run that dies a dozen times would otherwise carry two
    dozen landmarks saying so."""
    root, ws, env_dir = launch([(["#quit", "y"], "end it"), (["l", "l"], "carry on")])
    built = replay.one_run(ws, env_dir)
    said = [m["text"] for m in built["milestones"]]
    assert not any("dungeon level 1" in t for t in said), said
    assert not any(t == "experience level 1" for t in said), said
    assert any(m["kind"] == "death" for m in built["milestones"])


def test_keys_thrown_into_a_prompt_are_marked(launch):
    def drive():
        cycle = ("l", "j", "h", "k", "u", "n", "b", "y")
        return [cycle[i % len(cycle)] for i in range(22)]

    root, ws, env_dir = launch([(drive(), "circle"), (["l", "cr"], "into the prompt")],
                               seed=4, budget=100)
    built = replay.one_run(ws, env_dir)
    assert built["wasted"] >= 1
    assert built["steps"]["wasted"], "the steps that wasted a key are named"
    for i in built["steps"]["wasted"]:
        assert built["steps"]["key"][i] not in {"cr", "esc", "space"}


# --------------------------------------------------------------------------- #
# Attributing the work to the batch it was done for
# --------------------------------------------------------------------------- #
def tool(title: str, out: str) -> dict:
    return {"kind": "tool", "name": "Bash", "title": title, "why": "", "body": "",
            "out": out}


def reply(used: int) -> str:
    """What the actuator writes back after a batch."""
    return f"budget {used}/3000 life 1 reward +0 this life over False"


def test_one_call_that_plays_several_batches_is_recorded_against_all_of_them():
    """A session that drives through a script of its own plays many batches per call.
    The call is what played each of them, so each gets it; the work only happened
    once, so only the first gets that."""
    work, played = replay.assign([1, 4, 7], [
        tool("cat notes.md", "..."),
        tool("./python dive.py", reply(9)),
    ])
    assert [len(w) for w in work] == [1, 0, 0]
    assert work[0][0]["title"] == "cat notes.md"
    assert [p["title"] for p in played] == ["./python dive.py"] * 3


def test_a_session_reading_its_own_log_does_not_move_the_cursor():
    """The hazard this file exists to pin. A batch is known to have played because
    the budget the actuator reported back went up — but a session that `cat`s
    `logs.txt` gets hundreds of `| budget 2999/3000 |` lines back, and a looser match
    would read that as having played the whole run and file every later batch's
    working-out against the wrong batch."""
    dump = "action 2999 | budget 2999/3000 | l | step 1/1\n" * 40
    work, played = replay.assign([1, 4], [
        tool("cat logs.txt", dump),
        tool("./act do l l l", reply(3)),
        tool("./act do j j j", reply(6)),
    ])
    assert work[0][0]["title"] == "cat logs.txt", "the log dump is work, not a play"
    assert [p["title"] for p in played] == ["./act do l l l", "./act do j j j"]


def test_the_record_is_read_through_a_half_written_file(tmp_path):
    """`result.json` is rewritten after every action, so a page built during a live
    run catches it mid-write. That is ordinary, and a retry, not a failure."""
    path = tmp_path / "result.json"
    path.write_text('{"a": 1')       # truncated, as a concurrent writer leaves it
    with pytest.raises(json.JSONDecodeError):
        replay.load_record(path, tries=2)
    path.write_text('{"a": 1}')
    assert replay.load_record(path) == {"a": 1}


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
STUB = r"""
import fs from 'fs';
const page = fs.readFileSync(process.argv[2], 'utf8');
const open = '<script type="application/json" id="bundle">';
const island = page.slice(page.indexOf(open) + open.length,
                          page.indexOf('<' + '/script>', page.indexOf(open)));
const codeAt = page.indexOf('<script>', page.indexOf(open));
const js = page.slice(codeAt + 8, page.lastIndexOf('<' + '/script>'));
const els = {};
// The canvas records where it was asked to draw, so a test can ask *where* a dot
// landed and not just whether the legend mentions it. Everything else is swallowed.
let arcs = [];
const ctx = new Proxy({ arc(x, y, r) { arcs.push({ x, y, r }); } }, {
  get: (t, k) => (k in t ? t[k] : () => {}), set: () => true,
});
const stub = (id) => ({
  id, textContent: id === 'bundle' ? island : '', innerHTML: '', style: {},
  value: 0, max: 0, hidden: false, clientWidth: 900, width: 900, height: 340,
  dataset: {}, classList: { toggle(){}, add(){}, remove(){}, contains: () => false },
  on: {}, addEventListener(k, f){ (this.on[k] = this.on[k] || []).push(f); },
  fire(k, e){ for (const f of (this.on[k] || [])) f(Object.assign({target: this,
              preventDefault(){}}, e)); },
  querySelectorAll: () => [], querySelector: () => null,
  scrollIntoView(){}, getContext: () => ctx, closest: () => null,
});
globalThis.document = {
  body: {}, getElementById: (id) => els[id] || (els[id] = stub(id)),
  querySelectorAll: () => [], addEventListener(){},
};
globalThis.window = { devicePixelRatio: 1 };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#123456' });
globalThis.requestAnimationFrame = (f) => f();
globalThis.addEventListener = () => {};
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
new Function(js)();

const drawn = () => els.screen.innerHTML.replace(/<[^>]+>/g, '').split('\n');
const rows = drawn();
if (rows.length !== 25) throw new Error('drew ' + rows.length + ' rows');
if (/undefined/.test(els.screen.innerHTML)) throw new Error('a row drew in no colour');
if (!/Dlvl:1/.test(rows.join('\n'))) throw new Error('no status line');
if (!/AAAAA/.test(els['run-title'].textContent)) throw new Error('no run picked');
if (!/median human game/.test(els['baselines'].innerHTML))
  throw new Error('nothing to compare against');
if (!/hp/.test(els['meters'].innerHTML)) throw new Error('no condition meters');
if (!/progression/.test(els['facts'].innerHTML))
  throw new Error('no rung on the current step');
if (!/first life/.test(els['board'].innerHTML))
  throw new Error('the ladder panel has no comparable row');
if (!/gemini/.test(els['board'].innerHTML))
  throw new Error('nobody else is on the ladder');

// The progression axis: a second y mode, and it must not take the page down with it.
els['y-prog'].fire('click', {});
if (!/probability/.test(els['chart-note'].innerHTML))
  throw new Error('the progression axis did not explain itself');
els['ax-keys'].fire('click', {});
if (!/keystrokes/.test(els['chart-note'].innerHTML))
  throw new Error('the keystroke axis was lost in progression mode');
els['y-score'].fire('click', {});
els['ax-turns'].fire('click', {});
if (!/Zipfian/.test(els['chart-note'].innerHTML))
  throw new Error('could not get back to the score axis');

// The population toggles. Each hides a cloud that is already in the bundle, so the
// test is: the legend names it, a click stops naming it, a second click brings it
// back — and nothing throws with every population turned off at once. A box without
// the archive or the board's episodes has nothing to toggle, which is a skip and not
// a failure, so each one is guarded on its own data being there.
const bundle = JSON.parse(island);
const has = {
  nao: ((bundle.compare.nao || {}).sample || []).length > 0,
  bot: (bundle.compare.autoascend.games || []).length > 0,
  agents: (bundle.compare.agents || []).length > 0,
};
els['y-score'].fire('click', {});
els['ax-turns'].fire('click', {});
for (const [key, id, naming] of [['nao', 'cl-nao', /human games/],
                                 ['bot', 'cl-bot', /AutoAscend games/]]) {
  if (!has[key]) continue;
  if (!naming.test(els['chart-legend'].innerHTML))
    throw new Error(id + ': the legend does not name a cloud that is in the bundle');
  els[id].fire('click', {});
  if (naming.test(els['chart-legend'].innerHTML))
    throw new Error(id + ': turning the cloud off left it in the legend');
  els[id].fire('click', {});
  if (!naming.test(els['chart-legend'].innerHTML))
    throw new Error(id + ': turning the cloud back on did not restore it');
}
// Everything off at once: the chart still has to draw, with this run alone on it.
for (const id of ['cl-nao', 'cl-bot', 'cl-agents']) els[id].fire('click', {});
if (!/this run/.test(els['chart-legend'].innerHTML))
  throw new Error('with every population hidden the run left the legend too');
for (const id of ['cl-nao', 'cl-bot', 'cl-agents']) els[id].fire('click', {});

// On the progression axis a game read off an xlogfile is a LOWER BOUND, because
// that file has no experience level. If the page ever draws those dots without
// saying so, it is claiming a measurement it does not have.
els['y-prog'].fire('click', {});
if (has.nao || has.bot) {
  if (!/lower bound/.test(els['chart-note'].innerHTML))
    throw new Error('the progression clouds are drawn without the bound being stated');
}
if (has.agents && !/individual BALROG/.test(els['chart-note'].innerHTML))
  throw new Error('the board episodes are on the chart but not in the note');

// The axis runs to 100 and no rung on the ladder reaches it. A chart that draws the
// headroom without saying it is unreachable is implying a target that does not exist.
if (bundle.compare.ceiling) {
  if (bundle.compare.ceiling >= 100)
    throw new Error('the ceiling is meant to be below the top of the axis');
  if (!/Nothing can score 100/.test(els['chart-note'].innerHTML))
    throw new Error('the unreachable top of the axis goes unmentioned');
}
// A won game is drawn at 100 and not at the ~80% rung the ladder can give it, so the
// dots sitting on the very top of the plot have to be exactly the ascensions in the
// sample — counted off the canvas, because the claim is about where they are.
if (has.nao && (bundle.compare.nao.ascended || 0) > 0) {
  const won = bundle.compare.nao.sample.filter((r) => r[3]).length;
  if (!won) throw new Error('an archive with ascensions drew none into the sample');
  if (!/ascensions/.test(els['chart-legend'].innerHTML))
    throw new Error('the winners are on the chart but not in the legend');
  arcs = [];
  els['y-prog'].fire('click', {});   // already in prog mode; redraws and records
  const top = arcs.filter((a) => Math.abs(a.y - 14) < 0.01);   // ly(100) = pad.t
  if (top.length !== won)
    throw new Error(`${top.length} dots at 100% but ${won} games in the sample won`);
  // And on the score axis a winner keeps its own score: lifting it there would be
  // inventing a number, since the y is points and not a rung.
  arcs = [];
  els['y-score'].fire('click', {});
  const lifted = arcs.filter((a) => Math.abs(a.y - 14) < 0.01).length;
  if (lifted >= won)
    throw new Error('winners were pinned to the top of the score axis too');
  els['y-prog'].fire('click', {});
}
els['y-score'].fire('click', {});

// Scrub: the page has to survive being moved, and has to move.
els['slider'].value = 3;
els['slider'].fire('input', {});
if (els['played'].textContent === 'start') throw new Error('scrubbing did nothing');
if (drawn().join('\n') === rows.join('\n')) throw new Error('the screen did not change');
if (!/Dlvl:/.test(drawn().join('\n'))) throw new Error('scrubbed off the status line');
console.log('ok');
"""


def test_the_pages_own_javascript_runs(launch, tmp_path):
    """The fold above is the Python twin of what the browser does; this is the
    browser's own code, against a stub DOM. Without it the data could be perfect and
    the page still blank — which is the way a viewer usually breaks."""
    import shutil
    import subprocess

    js = shutil.which("bun") or shutil.which("node")
    if not js:
        pytest.skip("no javascript runtime on this box")
    root, ws, env_dir = launch([(["l", "l", "j"], "east then south"),
                                (["k", "h"], "back")])
    out = tmp_path / "replay.html"
    sys.argv = ["replay", str(root), "--out", str(out)]
    assert replay.main() == 0

    check = tmp_path / "check.mjs"
    check.write_text(STUB)
    ran = subprocess.run([js, str(check), str(out)], capture_output=True, text=True,
                         timeout=180)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "ok" in ran.stdout


def test_the_page_carries_the_rung_at_every_step(launch):
    """The metric is per step on the page because the panel moves with the scrubber,
    and per *life* in the arithmetic because that is what a rung scores."""
    from progression import progression as rung

    root, ws, env_dir = launch([([">", "esc", "l", "l"], "down if we can")])
    built = replay.one_run(ws, env_dir)
    steps = built["steps"]
    assert len(steps["prog"]) == len(steps["n"])
    assert steps["prog"][0] == 0.0, "the opening screen is the start, not progress"
    for i, value in enumerate(steps["prog"]):
        assert value == round(rung(steps["depth"][i], steps["xp"][i]), 2)
    got = built["progression"]
    assert got["per_life"] == [
        round(rung(one["depth"], one["xp"]), 2) for one in built["lives_played"]
    ]


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
    # And nothing left beside it: the atomic rename leaves no scratch file behind.
    assert not list(out.parent.glob("*.part"))
