#!/usr/bin/env python3
"""What each session saw, what it played, and what it did between the two.

    .env-venv/bin/python tools/replay.py ~/agent-runs/20260910-153914 --out replay.html

Adapted from cc_craftax's tools/replay.py, and this is the port where the page gets
*smaller*. Craftax's observation is an 832x704 frame and a step scrolls the whole
view, so 292,365 of 585,728 pixels change per action and a 3000-action run is 255 MB
of PNG that nothing recovers — it needed two modes and a frame budget. NetHack's
observation is 24 rows of 80 characters, and a step usually changes two of them. So
every run inlines, whole, in one publishable file, and there is no sampling and
nothing to choose.

**The screens are replayed, not read.** The page is built from the keystroke history
rather than from the files a run happened to write: `(seeds, keys)` is the whole game
(F7), replaying is 0.11 ms an action, and it gives the colours and the score at every
step whatever `--obs` the run was played under. A run that wrote only `tty` still gets
a coloured page; a run whose screens were rotated away by `--again` still replays.

**The colours are the game's.** `tty_colors` is a curses attribute per cell — three
bits of colour and a bright bit — and NetHack uses it to tell a yellow `d` from a red
one. The page keeps all sixteen, and `replay.css` defines them twice so they stay
legible on both grounds.

**Reasoning.** The CLI returns thinking blocks with their content encrypted, so the
page cannot show what a session thought and does not pretend to. What it can show is
what the session *wrote down*: the `--plan` it had to state to move at all, the shell
it ran, and what came back.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act import SEP  # noqa: E402
from nle_game import NleGame  # noqa: E402

ROWS, COLS = 24, 80
# How much of a command and its output the page carries. A session's scripts are
# worth reading and its `cat logs.txt` is not: the cap is what keeps a launch of six
# runs inlinable, and the whole of both is in `agent_stream.jsonl` either way.
COMMAND_CAP = 600
OUTPUT_CAP = 1200


def curses_colour(value: int) -> str:
    """One `tty_colors` cell as a hex digit: the low three bits, plus the bright bit.

    NLE gives this as a signed byte and the top bit is reverse video, which the page
    does not draw — the cursor is not part of what the session saw in a file.
    """
    value = int(value) & 0x7F
    return "0123456789abcdef"[(value & 0x07) | (0x08 if value & 0x08 else 0)]


def screens(record: dict) -> tuple[list, list]:
    """Replay the run and return (per-step deltas, per-step facts).

    A delta is the rows that changed since the step before, each as
    `[row, characters, colours]`. The first step is a delta against a blank screen,
    so the page can fold from nothing and never needs a base frame.
    """
    game = NleGame(
        record["variant"],
        seed=record["seed"],
        obs=("tty",),
        fresh_world=record["fresh_world"],
        blind=record.get("blind", True),
    )
    was_text = [" " * COLS] * ROWS
    was_hue = ["7" * COLS] * ROWS
    deltas, steps = [], []

    def snapshot(key: str, reward: float, wasted: bool) -> None:
        nonlocal was_text, was_hue
        obs = game.state()
        text = [row.decode("latin-1") for row in game._rows()]  # noqa: SLF001
        hue = ["".join(curses_colour(c) for c in row) for row in obs["tty_colors"]]
        delta = [
            [i, text[i], hue[i]]
            for i in range(ROWS)
            if text[i] != was_text[i] or hue[i] != was_hue[i]
        ]
        was_text, was_hue = text, hue
        episode = game.episodes[-1]
        deltas.append(delta)
        steps.append({
            "key": key,
            "reward": round(reward, 3) if reward else 0,
            "life": episode.index,
            "score": episode.score,
            "turns": episode.turns,
            "depth": episode.depth,
            "xplevel": episode.xplevel,
            "ended": episode.ended,
            "wasted": wasted,
        })

    snapshot("start", 0.0, False)
    for key in record["history"]:
        prompt = "--More--" in game.screen()
        reward, alive = game.play(key)
        snapshot(key, reward, prompt and key not in {"cr", "esc", "space"})
        if not alive:
            game.restart()
            # The state the next life began in is a screen no action produced, and
            # the log carries it for the same reason: it is the only view of it.
            snapshot("(new life)", 0.0, False)
    turns = {}
    for one in steps:
        turns[one["life"]] = max(turns.get(one["life"], 0), one["turns"])
    facts = {
        "turns": sum(turns.values()),
        "best": max(game.episode_scores(), default=0),
        "depth": game.max_depth,
        "xplevel": game.max_xplevel,
        "lives": len(game.episodes),
        "wasted": sum(one["wasted"] for one in steps),
    }
    game.close()
    return deltas, steps, facts


def plans(log: Path) -> dict[int, str]:
    """Where each batch started, and the plan it was given.

    Read from `logs.txt` rather than from the record, because a plan belongs to a
    batch and only the log knows where the batches were. The batch boundary is
    `step 1/k` in the header and **not** a change of plan text: `--plan` is sticky,
    so the actuator writes the standing plan into every block of every batch until a
    new one is given — attributing a batch to each block would make a 65-key run
    read as 65 batches, and the work between them would land in the wrong place.
    """
    if not log.exists():
        return {}
    out = {}
    for block in log.read_text(errors="replace").split(SEP):
        head = [line for line in block.splitlines() if line.startswith("action ")]
        if not head or " | step 1/" not in head[0]:
            continue
        said = [line for line in block.splitlines() if line.startswith("plan: ")]
        out[int(head[0].split()[1])] = said[0][len("plan: "):].strip() if said else ""
    return out


def work(stream: Path) -> list[tuple[str, str]]:
    """Every command the session ran and what came back, in order.

    Pairs are matched by tool-use id, because a session runs several commands per
    turn and the results arrive in their own events afterwards.
    """
    if not stream.exists():
        return []
    ran: dict[str, str] = {}
    order: list[str] = []
    back: dict[str, str] = {}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                got = block.get("input", {})
                said = got.get("command") or got.get("file_path") or ""
                if got.get("content") or got.get("new_string"):
                    said = f"write {said}"
                if said:
                    ran[block["id"]] = str(said)[:COMMAND_CAP]
                    order.append(block["id"])
            elif block.get("type") == "tool_result":
                body = block.get("content", "")
                if isinstance(body, list):
                    body = " ".join(str(part.get("text", "")) for part in body
                                    if isinstance(part, dict))
                back[block.get("tool_use_id", "")] = str(body)[:OUTPUT_CAP]
    return [(ran[i], back.get(i, "")) for i in order]


def split_work(commands: list[tuple[str, str]], said: dict[int, str],
               steps: list[dict]) -> tuple[list[str], list[int], list[list]]:
    """Attach the off-board work to the batch it came before.

    A session thinks, runs things, and then plays a batch; what the page wants
    beside action N is what it ran on the way there. `./act do` is itself a command,
    so the run of commands between one `do` and the next is that batch's working-out.
    """
    order = sorted(said)
    text = [said[n] for n in order]
    at = {n: i for i, n in enumerate(order)}
    # Which plan each step belongs to: the last batch that started at or before it.
    which, current = [], -1
    for index in range(len(steps)):
        if index in at:
            current = at[index]
        which.append(current)

    buckets: list[list] = [[] for _ in text]
    batch, held = 0, []
    for command, out in commands:
        if "./act do" in command or command.strip().startswith("act do"):
            if batch < len(buckets):
                buckets[batch] = held
            batch += 1
            held = []
        else:
            held.append([command, out])
    if batch < len(buckets):
        buckets[batch] = held
    return text, which, buckets


def one_run(ws: Path, env_dir: Path) -> dict | None:
    record_file = env_dir / "result.json"
    if not record_file.exists():
        return None
    record = json.loads(record_file.read_text())
    if not record["history"]:
        return None
    deltas, steps, facts = screens(record)
    said = plans(ws / "logs.txt")
    text, which, buckets = split_work(work(ws / "agent_stream.jsonl"), said, steps)
    for step, plan in zip(steps, which, strict=True):
        step["plan"] = plan
    return {
        "label": ws.name,
        "seed": record["seed"],
        "blind": record.get("blind", True),
        "budget": record["budget"],
        "role": record["roles"][0] if record["roles"] else "",
        "screens": deltas,
        "steps": steps,
        "plans": text,
        "work": buckets,
        **facts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="replay", description=__doc__.splitlines()[0])
    parser.add_argument("where", type=Path,
                        help="a launch directory, or one workspace inside one")
    parser.add_argument("--out", type=Path, default=Path("replay.html"))
    args = parser.parse_args()

    where = args.where.expanduser().resolve()
    if (where / "state.json").exists():
        pairs = [(where, where.parent / ".envs" / where.name)]
    else:
        labels = where / ".rig" / "labels.json"
        if not labels.exists():
            raise SystemExit(f"replay: {where} is neither a workspace nor a launch")
        pairs = [(where / label, where / ".envs" / label)
                 for label in json.loads(labels.read_text())]

    runs = []
    for ws, env_dir in pairs:
        built = one_run(ws, env_dir)
        if built is None:
            print(f"{ws.name}: nothing was played", flush=True)
            continue
        runs.append(built)
        print(f"{ws.name}: {len(built['steps']) - 1} keys, {len(built['plans'])} batches",
              flush=True)
    if not runs:
        raise SystemExit("replay: nothing to show")

    here = Path(__file__).resolve().parent
    page = (here / "replay_body.html").read_text()
    page = page.replace("/*__CSS__*/", (here / "replay.css").read_text())
    page = page.replace("/*__DATA__*/", json.dumps(runs, separators=(",", ":")))
    args.out.write_text(page)
    print(f"\n{len(runs)} run(s) -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
