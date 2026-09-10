#!/usr/bin/env python3
"""What a run did, at every action — recovered by replaying it.

Nothing on disk holds this. `result.json` is rewritten in place after each action,
so it holds the last state and not the curve; the screens hold the curve but only as
pictures of it. The keystroke history plus the seeds *is* the run (F7), and replaying
it costs 0.11 ms an action, so the whole thing is recoverable in under a second and
checked against the record at the end.

Three questions this exists to answer, in the order the pilot asks them:

* **Did the session handle the interface at all?** `--More--` swallows every key that
  is not `cr`, `esc` or `space`, silently (F2), and it is the failure that makes a run
  meaningless without looking like anything. `wasted` counts keys played into a
  prompt that could not accept them.
* **What did it buy with the budget?** Humans get 2.2 game turns per keypress (F10)
  by using count prefixes, travel and run. `turns/key` is that number for this run.
* **How far did it get?** Score, depth and experience level against the keys spent —
  the curve, at every milestone, for the best life and for the run.

Plus the agent side, from the stream: actions per tool call, context per turn, and
what it cost.

    .env-venv/bin/python tools/readout.py ~/agent-runs/20260910-153914
    .env-venv/bin/python tools/readout.py ~/agent-runs/20260910-153914/LPL4S --json out.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nle_game import NleGame  # noqa: E402

# Where the curve is read off. A 400-key pilot and a 30,000-key run both want the
# same shape of table, so the milestones that fall past the end are dropped.
MILESTONES = (100, 250, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000)
# The three keys a `--More--` accepts. Anything else played at one is thrown away.
DISMISS = {"cr", "esc", "space"}


def replay(record: dict) -> list[dict]:
    """Play the recorded history back and note what every key did.

    Every action goes through `NleGame.play` exactly as it did the first time, so the
    life boundaries and the score come out the way they went in — which the caller
    checks. What is added is the per-action detail no file kept: whether the clock
    moved, and whether the screen was waiting for something else.
    """
    game = NleGame(
        record["variant"],
        seed=record["seed"],
        obs=("tty",),
        fresh_world=record["fresh_world"],
        blind=record.get("blind", True),
    )
    rows = []
    for index, key in enumerate(record["history"], start=1):
        screen = game.screen()
        before = game.episodes[-1].turns
        prompt = (
            "more" if "--More--" in screen
            else "menu" if "(end)" in screen or "(1 of " in screen
            else "yn" if "[yn" in screen.splitlines()[0]
            else ""
        )
        reward, alive = game.play(key)
        episode = game.episodes[-1]
        rows.append({
            "action": index,
            "key": key,
            "reward": reward,
            "life": episode.index,
            "score": episode.score,
            "turns": episode.turns,
            "moved": episode.turns > before,
            "depth": episode.depth,
            "xplevel": episode.xplevel,
            "prompt": prompt,
            # A key played into a `--More--` that the prompt cannot accept: it cost an
            # action, changed nothing, and said nothing about having changed nothing.
            "wasted": bool(prompt == "more" and key not in DISMISS),
            "ended": episode.ended,
        })
        if not alive:
            game.restart()
    scores = game.episode_scores()
    game.close()
    return rows, scores


def per_life(rows: list[dict], field: str) -> dict[int, int]:
    """The highest `field` each life reached. Both the score and the clock reset when
    a life does, so anything summed or compared across lives has to go through here.
    """
    best: dict[int, int] = {}
    for row in rows:
        best[row["life"]] = max(best.get(row["life"], 0), row[field])
    return best


def curve(rows: list[dict]) -> list[dict]:
    """The run at each milestone: what it had reached by then.

    `best_life` and not a running total, for the reason `act.py` keeps the run total
    to itself: a sum across lives is not a score anyone quotes.
    """
    out = []
    for mark in MILESTONES:
        if mark > len(rows):
            break
        seen = rows[:mark]
        out.append({
            "actions": mark,
            "best_life": max(per_life(seen, "score").values(), default=0),
            "turns": sum(per_life(seen, "turns").values()),
            "depth": max(row["depth"] for row in seen),
            "xplevel": max(row["xplevel"] for row in seen),
            "lives": 1 + max(row["life"] for row in seen),
        })
    return out


def interface(rows: list[dict]) -> dict:
    """How much of the budget the terminal took, and how much the game got."""
    played = len(rows) or 1
    # Game turns are per life and reset with it, so the run's total is the sum over
    # lives rather than the last reading.
    turns = sum(per_life(rows, "turns").values())
    return {
        "keys": len(rows),
        "turns": turns,
        "turns_per_key": round(turns / played, 2),
        "moved_pct": round(100 * sum(r["moved"] for r in rows) / played, 1),
        "at_a_prompt_pct": round(100 * sum(bool(r["prompt"]) for r in rows) / played, 1),
        "wasted": sum(r["wasted"] for r in rows),
        "wasted_pct": round(100 * sum(r["wasted"] for r in rows) / played, 1),
        "longest_stuck": longest_stuck(rows),
        "keys_used": len({r["key"] for r in rows}),
    }


def longest_stuck(rows: list[dict]) -> int:
    """The longest run of keys thrown into a prompt that could not take them.

    The number that says whether a session noticed. One or two is a session
    checking; forty is a session that stopped playing and did not find out.
    """
    worst = run = 0
    for row in rows:
        run = run + 1 if row["wasted"] else 0
        worst = max(worst, run)
    return worst


def agent(stream: Path) -> dict:
    """What the session cost and how hard it batched, from its own event stream."""
    if not stream.exists():
        return {}
    turns = calls = compactions = 0
    cost = 0.0
    usage = {}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            turns += 1
            said = event.get("message", {}).get("content", [])
            calls += sum(1 for block in said if block.get("type") == "tool_use")
        elif kind == "system" and event.get("subtype") == "compact_boundary":
            compactions += 1
        elif kind == "result":
            cost = event.get("total_cost_usd", cost)
            usage = event.get("usage", {}) or usage
    return {
        "turns": turns,
        "tool_calls": calls,
        "compactions": compactions,
        "cost_usd": round(cost, 2),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def read(ws: Path, env_dir: Path) -> dict:
    """One workspace, replayed and measured."""
    record = json.loads((env_dir / "result.json").read_text())
    rows, scores = replay(record)
    if len(rows) != record["actions_used"]:
        raise SystemExit(
            f"{ws}: the record holds {record['actions_used']} actions and replaying "
            f"its history played {len(rows)} — this is not a history of that run"
        )
    if scores != record["episode_scores"]:
        raise SystemExit(
            f"{ws}: replaying the history scored {scores} where the record says "
            f"{record['episode_scores']} — the replay is not the run"
        )
    return {
        "workspace": ws.name,
        "variant": record["variant"],
        "seed": record["seed"],
        "blind": record.get("blind", True),
        "budget": record["budget"],
        "best_episode": record["best_episode"],
        "mean_episode": record["mean_episode"],
        "episodes_completed": record["episodes_completed"],
        "episode_scores": scores,
        "roles": record["roles"],
        "endings": record["endings"],
        "max_depth": record["max_depth"],
        "max_xplevel": record["max_xplevel"],
        "interface": interface(rows),
        "curve": curve(rows),
        "agent": agent(ws / "agent_stream.jsonl"),
        "keys": [row["key"] for row in rows],
    }


def show(one: dict) -> None:
    face = one["interface"]
    said = one["agent"]
    print(f"\n{one['workspace']}  {one['variant']}:{one['seed']}  "
          f"{'blind' if one['blind'] else 'named'}  "
          f"{one['roles'][0] if one['roles'] else '?'}")
    print(f"  best life {one['best_episode']}, mean {one['mean_episode']} over "
          f"{one['episodes_completed']} finished, lives {one['episode_scores']}, "
          f"ended {', '.join(one['endings']) or '—'}")
    print(f"  dlvl {one['max_depth']}, xp {one['max_xplevel']}, "
          f"{face['keys']}/{one['budget']} keys, {face['turns']} game turns "
          f"({face['turns_per_key']} per key; a person gets 2.2)")
    print(f"  {face['moved_pct']}% of keys moved the clock, "
          f"{face['at_a_prompt_pct']}% were played at a prompt, "
          f"{face['wasted']} were thrown away at a --More-- "
          f"({face['wasted_pct']}%, longest run {face['longest_stuck']}), "
          f"{face['keys_used']} distinct keys used")
    if said:
        per = round(face["keys"] / said["tool_calls"], 1) if said["tool_calls"] else 0
        print(f"  {said['turns']} turns, {said['tool_calls']} tool calls "
              f"({per} keys each), {said['compactions']} compactions, "
              f"${said['cost_usd']}")
    if one["curve"]:
        print(f"  {'keys':>6}{'best life':>11}{'turns':>8}{'dlvl':>6}{'xp':>4}{'lives':>7}")
        for mark in one["curve"]:
            print(f"  {mark['actions']:>6}{mark['best_life']:>11}{mark['turns']:>8}"
                  f"{mark['depth']:>6}{mark['xplevel']:>4}{mark['lives']:>7}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="readout", description=__doc__.splitlines()[0])
    parser.add_argument("where", type=Path,
                        help="a launch directory, or one workspace inside one")
    parser.add_argument("--json", type=Path, help="write every row as json")
    args = parser.parse_args()

    where = args.where.expanduser().resolve()
    if (where / "state.json").exists():  # one workspace
        pairs = [(where, where.parent / ".envs" / where.name)]
    else:
        labels = where / ".rig" / "labels.json"
        if not labels.exists():
            raise SystemExit(f"readout: {where} is neither a workspace nor a launch")
        pairs = [(where / label, where / ".envs" / label)
                 for label in json.loads(labels.read_text())]
    out = []
    for ws, env_dir in pairs:
        if not (env_dir / "result.json").exists():
            print(f"{ws.name}: no record — nothing was played", flush=True)
            continue
        one = read(ws, env_dir)
        out.append(one)
        show(one)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\n{len(out)} run(s) -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
