#!/usr/bin/env python3
"""What each session saw, what it played, and what it did between the two.

    .env-venv/bin/python tools/replay.py ~/agent-runs/20260911-054617 --out replay.html

Adapted from cc_craftax's `tools/replay.py`, and this is the port where the page gets
*smaller*. Craftax's observation is an 832x704 frame and a step scrolls the whole view,
so 292,365 of 585,728 pixels change per action and a 3,000-action run is 255 MB of PNG
that nothing recovers — it needed two modes and a frame budget. NetHack's observation
is 24 rows of 80 characters and a step usually changes two of them, so every run
inlines, whole, in one file, and there is nothing to choose and nothing to thin.

**The screens are replayed, not read.** The page is built from the keystroke history
rather than from the files a run happened to write: `(seeds, keys)` is the whole game
(F7), replaying is a fraction of a millisecond an action, and it gives the colours, the
score and the whole status line at every step whatever `--obs` the run was played
under. A run that wrote only `tty` still gets a coloured page with meters on it.

**The colours are the game's.** `tty_colors` is a curses attribute per cell — three
bits of colour and a bright bit — and NetHack uses it to tell a yellow `d` from a red
one. The page keeps all sixteen, and `replay.css` defines them twice so they stay
legible on both grounds.

**Two encodings, because a 30,000-key run is a large page.** A screen is stored as the
rows that changed since the one before it; each row's text is right-stripped and each
row's colour is run-length coded (`A40B2A38`, one letter per curses colour), which the
browser expands on the way in. Together they take about two thirds off the payload,
and nothing about the trajectory is thinned to get it — every key is still on the
track. The per-step numbers are columnar for the same reason: thirty thousand objects
with sixteen named fields each is four megabytes of repeated key names.

**Where this run sits.** `tools/humans.py` reads the 1,934 complete AutoAscend games in
`nle_data/nld-aa-taster/` and the published NLD-NAO human medians, and the page draws a
run against both. Game turns is the axis they share — the xlogfile records turns and
not keypresses — so the cloud is drawn against turns and the budget axis carries only
what can honestly stand on it.

**Reasoning.** The CLI returns thinking blocks with their content encrypted, so the
page cannot show what a session thought and does not pretend to. What it can show is
what the session *wrote down*: the `--plan` it had to state to move at all, the shell
and the scripts it ran, and what came back.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from act import SEP  # noqa: E402
from humans import company  # noqa: E402
from nle_game import NleGame  # noqa: E402
from progression import of_lives, progression  # noqa: E402

ROWS, COLS = 24, 80

# How much of a command and its output the page carries. A session's scripts are worth
# reading and its `cat logs.txt` is not; the whole of both is in `agent_stream.jsonl`
# either way.
COMMAND_CAP = 800
OUTPUT_CAP = 2000
NOTES_CAP = 60000

# `action N | budget used/total | key | step i/k`, and the handover the launcher writes
# where one session's stint ended and the next rebuilt the game. A handover is not an
# action: it repeats the number the run stopped at.
HEAD = re.compile(r"^action (\d+) \| budget (\d+)/(\d+)(.*)$")
STEP = re.compile(r"\| step (\d+)/(\d+)")
HANDOVER = re.compile(r"\| session (\d+)\s*$")

# The actuator's own reply, which is how a tool call is known to have played rather
# than merely read. Deliberately tighter than `budget (\d+)/(\d+)`: a session that
# `cat`s its own log gets hundreds of `| budget 999/3000 |` lines back, and matching
# those would jump the cursor to the end of the run and put every batch's working-out
# against the wrong batch. Only the actuator writes `budget N/M life`.
REPLY = re.compile(r"budget (\d+)/(\d+) life ")

# The three keys a `--More--` accepts. Anything else played at one is thrown away (F2).
DISMISS = {"cr", "esc", "space"}

# NetHack's own `u.uhs` and encumbrance ladders, for the meters.
HUNGER = ("satiated", "not hungry", "hungry", "weak", "fainting", "fainted", "starved")
CARRY = ("unencumbered", "burdened", "stressed", "strained", "overtaxed", "overloaded")

# A new run-best worth marking on the track. Every point would be a solid bar on a
# 30,000-key run; fifty is roughly a kill worth noticing.
SCORE_STEP = 50


# --------------------------------------------------------------------------- #
# The screens
# --------------------------------------------------------------------------- #
def colour_char(value: int) -> str:
    """One `tty_colors` cell as a letter: the low three bits, plus the bright bit.

    NLE gives this as a signed byte and the top bit is reverse video, which the page
    does not draw — the cursor is not part of what the session saw in a file. Letters
    rather than hex digits so that the run-length coding below is unambiguous: a run
    is one letter and then its length in digits, and a hex colour would collide with
    the digits of the count.
    """
    value = int(value) & 0x7F
    return "ABCDEFGHIJKLMNOP"[(value & 0x07) | (0x08 if value & 0x08 else 0)]


def rle(row: str) -> str:
    """A row of colour letters as runs: `A40B2A38`.

    The message line and the status line are one colour end to end and the map is
    mostly two, so this is where most of a NetHack screen's redundancy is.
    """
    out, last, run = [], row[0], 0
    for cell in row:
        if cell == last:
            run += 1
            continue
        out.append(f"{last}{run}")
        last, run = cell, 1
    out.append(f"{last}{run}")
    return "".join(out)


def screens(record: dict) -> tuple[list, dict, dict, list]:
    """Replay the run: the screens, the numbers under them, and what stands out.

    Returns `(deltas, steps, facts, milestones)`. A delta is the rows that changed
    since the step before, each `[row, text, colours]`; the first is a delta against a
    blank screen, so the page folds from nothing and never needs a base frame. `steps`
    is columnar — one array per field, indexed by step — because thirty thousand
    objects of named fields is megabytes of repeated names.
    """
    game = NleGame(
        record["variant"],
        seed=record["seed"],
        obs=("tty",),
        fresh_world=record["fresh_world"],
        blind=record.get("blind", True),
    )
    was_text = [" " * COLS] * ROWS
    was_hue = ["H" * COLS] * ROWS
    deltas: list = []
    cols: dict[str, list] = {
        name: [] for name in
        ("n", "key", "reward", "score", "turns", "depth", "xp", "life", "prog",
         "hp", "hpmax", "pw", "pwmax", "ac", "hunger", "carry", "gold", "prompt")
    }
    ended: dict[int, str] = {}
    wasted: list[int] = []
    milestones: list[dict] = []
    # Every life starts on dungeon level 1 at experience level 1, so those are where
    # the marks are counted from rather than things that happened.
    deepest = best_xp = 1
    best_score = 0

    def snapshot(n: int, key: str, reward: float, prompt: str) -> None:
        nonlocal was_text, was_hue, deepest, best_xp, best_score
        obs = game.state()
        text = [row.decode("latin-1") for row in game._rows()]  # noqa: SLF001
        hue = ["".join(colour_char(c) for c in row) for row in obs["tty_colors"]]
        deltas.append([
            [i, text[i].rstrip(), rle(hue[i])]
            for i in range(ROWS)
            if text[i] != was_text[i] or hue[i] != was_hue[i]
        ])
        was_text, was_hue = text, hue
        episode = game.episodes[-1]
        at = len(deltas) - 1
        stat = game._stat  # noqa: SLF001
        for name, value in (
            ("n", n), ("key", key), ("reward", round(reward, 3) if reward else 0),
            ("score", episode.score), ("turns", episode.turns),
            ("depth", episode.depth), ("xp", episode.xplevel),
            ("life", episode.index), ("prompt", prompt),
            # The rung this life stood on at this step, on BALROG's human-calibrated
            # ladder. Per life and not per run: the metric scores an episode, and a
            # run's deepest level is not somewhere any one of its lives reached.
            ("prog", round(progression(episode.depth, episode.xplevel), 2)),
            # Straight off the status line rather than off the episode's running
            # maxima: the meters are the condition the session was *in*, and a
            # running maximum of hit points would draw a character who never bled.
            ("hp", stat("HP")), ("hpmax", stat("HPMAX")),
            ("pw", stat("ENE")), ("pwmax", stat("ENEMAX")),
            ("ac", stat("AC")), ("hunger", stat("HUNGER")),
            ("carry", stat("CAP")), ("gold", stat("GOLD")),
        ):
            cols[name].append(value)

        if episode.depth > deepest:
            deepest = episode.depth
            milestones.append({"at": at, "kind": "depth",
                               "text": f"reached dungeon level {deepest}"})
        if episode.xplevel > best_xp:
            best_xp = episode.xplevel
            milestones.append({"at": at, "kind": "xp",
                               "text": f"experience level {best_xp}"})
        if episode.score >= best_score + SCORE_STEP:
            best_score = episode.score
            milestones.append({"at": at, "kind": "score",
                               "text": f"{best_score} points"})

    snapshot(0, "start", 0.0, "")
    for index, key in enumerate(record["history"], start=1):
        screen = game.screen()
        first = screen.splitlines()[0] if screen else ""
        prompt = (
            "more" if "--More--" in screen
            else "menu" if "(end)" in screen or "(1 of " in screen
            else "yn" if "[yn" in first
            else ""
        )
        reward, alive = game.play(key)
        snapshot(index, key, reward, prompt)
        if prompt == "more" and key not in DISMISS:
            wasted.append(len(deltas) - 1)
        if not alive:
            # The message line of the tombstone is the only place the cause of death
            # is written down: the record keeps `died` and nothing more.
            said = " ".join(game.screen().splitlines()[0].split())
            at = len(deltas) - 1
            ended[at] = said or game.episodes[-1].ended
            milestones.append({"at": at, "kind": "death",
                               "text": said or "this life ended"})
            game.restart()
            # The state the next life began in is a screen no action produced, and the
            # log carries it for the same reason: it is the only view of it.
            deepest = best_xp = 1
            snapshot(index, "(new life)", 0.0, "")

    # Each life's high-water marks. Everything compared across lives comes through
    # here, because the score, the clock, the depth and the experience level all reset
    # when a life does and the tombstone frame reports zeroes for all of them (F12).
    lives: dict[int, dict] = {}
    for i, life in enumerate(cols["life"]):
        one = lives.setdefault(life, {"life": life, "depth": 1, "xp": 1, "score": 0,
                                      "turns": 0, "keys": 0})
        for field, column in (("depth", "depth"), ("xp", "xp"),
                              ("score", "score"), ("turns", "turns")):
            one[field] = max(one[field], cols[column][i])
        one["keys"] += 1
    per_life = [lives[life] for life in sorted(lives)]
    turns = {life: one["turns"] for life, one in lives.items()}
    scores = game.episode_scores()
    facts = {
        "lives_played": per_life,
        "progression": of_lives(per_life),
        "turns": sum(turns.values()),
        "best": max(scores, default=0),
        "scores": scores,
        "depth": game.max_depth,
        "xplevel": game.max_xplevel,
        "lives": len(game.episodes),
        "wasted": len(wasted),
        "role": game.episodes[0].role if game.episodes else "",
        "endings": sorted({e.ended for e in game.episodes if e.ended}),
        "cells": game.unique_cells,
    }
    game.close()
    cols["ended"] = {str(k): v for k, v in ended.items()}
    cols["wasted"] = wasted
    return deltas, cols, facts, milestones


# --------------------------------------------------------------------------- #
# The log: where the batches were, and where a session handed over
# --------------------------------------------------------------------------- #
def read_log(path: Path) -> tuple[dict[int, str], list[dict]]:
    """The plan each batch was given, and the session boundaries.

    The batch boundary is `step 1/k` in the header and **not** a change of plan text:
    `--plan` is sticky, so the actuator writes the standing plan into every block of
    every batch until a new one is given, and attributing a batch to each block would
    make a 65-key run read as 65 batches with the work between them in the wrong place.
    """
    if not path.exists():
        return {}, []
    plans: dict[int, str] = {}
    hands: list[dict] = []
    for block in path.read_text(errors="replace").split(SEP):
        lines = block.splitlines()
        head = next((line for line in lines if HEAD.match(line)), "")
        if not head:
            continue
        found = HEAD.match(head)
        n, rest = int(found[1]), found[4]
        hand = HANDOVER.search(rest)
        if hand:
            # The event runs over several lines and only the first is tagged, so this
            # takes the block from the tag onwards rather than the tagged lines.
            body = lines[lines.index(next(
                (line for line in lines if line.startswith("[event]")), lines[-1]
            )):]
            note = " ".join(" ".join(
                line for line in body if not line.startswith("[tty]")
            ).split())
            hands.append({"n": n, "session": int(hand[1]),
                          "note": note.replace("[event] ", "")[:400]})
            continue
        step = STEP.search(rest)
        if not step or step[1] != "1":
            continue
        said = next((line for line in lines if line.startswith("plan: ")), "")
        plans[n] = said[len("plan: "):].strip()
    return plans, hands


# --------------------------------------------------------------------------- #
# The stream: what the session did between batches
# --------------------------------------------------------------------------- #
def moments(stream: Path, ws: Path) -> list[dict]:
    """The session's turns, flattened into what it said, ran and was told back."""
    if not stream.exists():
        return []
    out: list[dict] = []
    pending: dict[str, dict] = {}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                shape = block.get("type")
                if shape == "text" and block.get("text", "").strip():
                    out.append({"kind": "say", "text": block["text"][:OUTPUT_CAP]})
                elif shape == "thinking":
                    # Content arrives encrypted — an empty string beside a signature —
                    # so this records that the session reasoned here and claims nothing
                    # about what it reasoned.
                    out.append({"kind": "think"})
                elif shape == "tool_use":
                    got = block.get("input", {}) or {}
                    moment = {
                        "kind": "tool",
                        "name": block.get("name", "?"),
                        "title": str(got.get("command") or got.get("file_path")
                                     or got.get("pattern") or got.get("path")
                                     or "")[:COMMAND_CAP],
                        "why": str(got.get("description", ""))[:200],
                        "body": str(got.get("content", "")
                                    or got.get("new_string", ""))[:OUTPUT_CAP],
                        "out": "",
                    }
                    out.append(moment)
                    if block.get("id"):
                        pending[block["id"]] = moment
        elif kind == "user":
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                body = block.get("content", "")
                if isinstance(body, list):
                    body = "\n".join(str(p.get("text", "")) for p in body
                                     if isinstance(p, dict))
                target = pending.get(block.get("tool_use_id", ""))
                if target is None:
                    continue
                # Kept whole for the cursor below, cut for the page afterwards.
                target["out"] = str(body)
                target["error"] = bool(block.get("is_error"))
    here = str(ws) + "/"
    for moment in out:
        if moment["kind"] == "tool":
            moment["title"] = moment["title"].replace(here, "./")
    return out


def assign(starts: list[int], said: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """Hand each batch the call that played it and the work done on the way to it.

    A call played if the budget the actuator reported back went up — true of
    `./act do` and equally of any script that ran it, and this session drove through
    five scripts of its own, so keying on the text of the command would have missed
    most of its batches. Everything between two such calls is work done in the
    workspace for free, and belongs to the batch it was preparing.

    One call often plays several batches: the *call* is recorded against each of them,
    because that is what played it, and the *work* only against the first, because
    that is when it was done.
    """
    buckets: list[list[dict]] = [[] for _ in starts]
    played: list[dict] = [{} for _ in starts]
    gap: list[dict] = []
    cursor, at = 0, 0
    for moment in said:
        if moment["kind"] != "tool":
            gap.append(moment)
            continue
        seen = [int(n) for n, _ in REPLY.findall(moment.get("out", ""))]
        end = max(seen, default=0)
        if end <= cursor:
            gap.append(moment)
            continue
        first = True
        while at < len(starts) and starts[at] <= end:
            played[at] = {"title": moment["title"], "why": moment.get("why", ""),
                          "out": str(moment.get("out", ""))[:OUTPUT_CAP]}
            if first:
                buckets[at], first = gap, False
            at += 1
        gap, cursor = [], end
    if gap and buckets:
        buckets[min(at, len(buckets) - 1)].extend(gap)
    for bucket in buckets:
        for moment in bucket:
            moment["out"] = str(moment.get("out", ""))[:OUTPUT_CAP]
            moment["body"] = str(moment.get("body", ""))[:OUTPUT_CAP]
    return buckets, played


def agent_stats(stream: Path) -> dict:
    """What the session cost and how hard it batched, from its own event stream.

    Read here rather than from `.rig/reports/`: a report is written when the launcher
    finishes, and this page is built while the run is still playing.
    """
    from readout import agent  # noqa: PLC0415

    return agent(stream)


# --------------------------------------------------------------------------- #
# Building the page
# --------------------------------------------------------------------------- #
def thin(pairs: list[list[int]], points: int = 600) -> list[list[int]]:
    """Log-spaced samples of a curve, keeping every point where it moves.

    The moves are the whole shape of the line — they are the kills and the descents —
    and a 30,000-point path drawn into a 600-pixel chart is 29,400 wasted bytes.
    """
    if not pairs:
        return []
    n = len(pairs)
    wanted = {0, n - 1}
    wanted |= {i for i in range(1, n) if pairs[i][1] != pairs[i - 1][1]}
    wanted |= {min(n - 1, round((i / points) ** 2 * (n - 1))) for i in range(points)}
    return [pairs[i] for i in sorted(wanted)]


def curves(cols: dict) -> dict:
    """The best single life's score so far, against keys and against game turns.

    Best-life and not a running total, for the reason `act.py` keeps the run total to
    itself: a sum across lives is not a score anyone quotes, and it cannot be compared
    with a population of games that each ended once.
    """
    by_key: list[list[int]] = []
    by_turn: list[list[int]] = []
    best = spent = 0
    # The clock restarts when a life does, so turns spent by a run is the sum over
    # lives of each one's high-water mark and never the last reading.
    lives: dict[int, int] = {}
    for i, life in enumerate(cols["life"]):
        was = lives.get(life, 0)
        if cols["turns"][i] > was:
            spent += cols["turns"][i] - was
            lives[life] = cols["turns"][i]
        best = max(best, cols["score"][i])
        by_key.append([cols["n"][i], best])
        by_turn.append([spent, best])
    return {"keys": thin(by_key), "turns": thin(by_turn)}


def load_record(path: Path, tries: int = 12) -> dict:
    """The record, read while the run may be rewriting it.

    `result.json` is written after every action, so a page built during a live run can
    catch it half-written. A truncated read is a retry and not a failure: the next one
    is a few milliseconds later and the file is whole again.
    """
    for attempt in range(tries):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == tries - 1:
                raise
            time.sleep(0.25)
    raise AssertionError("unreachable")


def one_run(ws: Path, env_dir: Path) -> dict | None:
    record_file = env_dir / "result.json"
    if not record_file.exists():
        return None
    record = load_record(record_file)
    if not record["history"]:
        return None
    deltas, cols, facts, milestones = screens(record)
    plans, hands = read_log(ws / "logs.txt")

    starts = sorted(plans)
    # Which batch each step belongs to: the last one that began at or before it.
    where, current = [], -1
    at = 0
    for n in cols["n"]:
        while at < len(starts) and starts[at] <= n:
            current = at
            at += 1
        where.append(current)
    cols["batch"] = where
    for hand in hands:
        hand["at"] = next((i for i, n in enumerate(cols["n"]) if n > hand["n"]),
                          len(cols["n"]) - 1)
        milestones.append({"at": hand["at"], "kind": "hand",
                           "text": f"session {hand['session']} takes over"})
    milestones.sort(key=lambda m: m["at"])
    work, played = assign(starts, moments(ws / "agent_stream.jsonl", ws))

    return {
        "label": ws.name,
        "seed": record["seed"],
        "blind": record.get("blind", True),
        "budget": record["budget"],
        "used": record["actions_used"],
        "terminal": record.get("terminal", False),
        "outcome": record.get("outcome", ""),
        "screens": deltas,
        "steps": cols,
        "plans": [plans[n] for n in starts],
        "starts": starts,
        "work": work,
        "played": played,
        "handovers": hands,
        "milestones": milestones,
        "curves": curves(cols),
        "agent": agent_stats(ws / "agent_stream.jsonl"),
        "audit": (audit_of(ws) or {}),
        "notes": (ws / "notes.md").read_text(errors="replace")[:NOTES_CAP]
                 if (ws / "notes.md").exists() else "",
        "files": sorted(p.name for p in ws.glob("*.py")),
        **facts,
    }


def audit_of(ws: Path) -> dict | None:
    """The audit, if a launcher has finished and written one for this workspace."""
    report = ws.parent / ".rig" / "reports" / f"{ws.name}.json"
    if not report.exists():
        return None
    try:
        return json.loads(report.read_text()).get("audit")
    except json.JSONDecodeError:
        return None


def pairs_under(where: Path) -> list[tuple[Path, Path]]:
    if (where / "state.json").exists():
        return [(where, where.parent / ".envs" / where.name)]
    labels = where / ".rig" / "labels.json"
    if not labels.exists():
        raise SystemExit(f"replay: {where} is neither a workspace nor a launch")
    return [(where / label, where / ".envs" / label)
            for label in json.loads(labels.read_text())]


def build(where: Path) -> dict:
    """The whole bundle the page is rendered from."""
    runs = []
    for ws, env_dir in pairs_under(where):
        built = one_run(ws, env_dir)
        if built is None:
            print(f"{ws.name}: nothing was played", flush=True)
            continue
        runs.append(built)
        print(f"{ws.name}: {built['used']}/{built['budget']} keys, "
              f"{len(built['plans'])} batches, best life {built['best']}", flush=True)
    return {
        "launch": where.name,
        "built": time.strftime("%Y-%m-%d %H:%M"),
        # Whether there is more of this run to come. Not the socket: the launcher
        # stops the daemon between stints, so a live run has no socket for the
        # seconds it takes to hand over.
        "live": any(one["used"] < one["budget"] for one in runs),
        "compare": company(),
        "runs": runs,
    }


def render(bundle: dict, out: Path) -> int:
    """Write the page, whole, over whatever was there.

    Written beside the target and moved into place, because a watcher rebuilds this
    every few minutes and a reader may have the last one open: a rename is atomic and
    a half-written 20 MB file in a browser tab is not.
    """
    here = Path(__file__).resolve().parent
    page = (here / "replay_body.html").read_text()
    for marker, text in (("/*__CSS__*/", (here / "replay.css").read_text()),
                         ("/*__DATA__*/", json.dumps(bundle, separators=(",", ":")))):
        if marker not in page:
            raise SystemExit(f"replay: replay_body.html has no {marker} to substitute")
        # `</` inside the JSON island or the stylesheet would close its element.
        page = page.replace(marker, text.replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.with_suffix(out.suffix + ".part")
    scratch.write_text(page)
    scratch.replace(out)
    return len(page)


def main() -> int:
    parser = argparse.ArgumentParser(prog="replay", description=__doc__.splitlines()[0])
    parser.add_argument("where", type=Path,
                        help="a launch directory, or one workspace inside one")
    parser.add_argument("--out", type=Path, default=Path("replay.html"))
    args = parser.parse_args()

    bundle = build(args.where.expanduser().resolve())
    if not bundle["runs"]:
        raise SystemExit("replay: nothing to show")
    size = render(bundle, args.out.expanduser().resolve())
    bot = bundle["compare"]["autoascend"]["score"]
    print(f"\n{len(bundle['runs'])} run(s) -> {args.out} ({size / 1e6:.1f} MB)"
          + (f", against {bot['n']} AutoAscend games" if bot else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
