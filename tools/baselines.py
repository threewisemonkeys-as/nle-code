#!/usr/bin/env python3
"""The floor, so that a session's number can be read.

Four policies, all playing by exactly the rules a session plays by — `act.ending` is
shared, so a baseline cannot drift into measuring a different game:

* **noop** — the null policy: `.` and nothing else. It should still die, because
  hunger does not care whether you move, and that is the cheapest check that the
  game is actually running.
* **random** — uniform over all 118 keys. In the Challenge configuration this is a
  much lower floor than it sounds, because most of the keyboard opens something
  modal and a random policy rarely closes it (F2).
* **blind** — random over the eight compass keys. Movement only, and no way out of
  a message: the policy that shows what `--More--` costs.
* **wander** — the same, but it dismisses `--More--` with `cr`, escapes menus with
  `esc`, answers `[yn]` prompts, and travels to a down staircase whenever one is on
  the screen. Roughly the least a program can do and still be playing NetHack, and
  the floor a session's number should be read against.

None of them reads anything a session could not: `wander` looks at the screen and
the character grid, which are channels the run gives out, and never at `blstats`
except to record the score afterwards. A baseline that read the true state would be
measuring a different game.

Observations are not written. A baseline needs the score, not the screen.

    .env-venv/bin/python tools/baselines.py                          # all four
    .env-venv/bin/python tools/baselines.py --policy wander --trials 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act import DEFAULT_BUDGET, ending  # noqa: E402
from nle_game import NleGame  # noqa: E402

POLICIES = ("noop", "random", "blind", "wander")
COMPASS = ("h", "j", "k", "l", "y", "u", "b", "n")
RUN = ("H", "J", "K", "L", "Y", "U", "B", "N")


def next_key(policy: str, game: NleGame, rng: random.Random, state: dict) -> str:
    """One key, chosen from what a session could see and nothing else."""
    if policy == "noop":
        return "."
    if policy == "random":
        return rng.choice(game.tokens)
    if policy == "blind":
        return rng.choice(COMPASS)

    screen = game.screen()
    if "--More--" in screen:
        return "cr"
    if "(end)" in screen or "(1 of " in screen:
        return "esc"
    if "[yn" in screen.splitlines()[0]:
        return "y"
    # A staircase on the screen is worth walking to: `_` asks where, `>` picks the
    # stairs, `.` confirms, and then `>` again goes down. Four keys, and if the
    # game did not take them the next screen says so.
    if state["travel"]:
        return state["travel"].pop(0)
    if ">" in screen and rng.random() < 0.5:
        state["travel"] = ["_", ">", ".", ">"]
        return state["travel"].pop(0)
    return rng.choice(RUN)


def rollout(variant: str, seed: int, policy: str, budget: int) -> dict:
    """One run: a shared budget across lives, ending when the budget does."""
    game = NleGame(variant, seed=seed)
    rng = random.Random(f"{policy}-{variant}-{seed}")
    state = {"travel": []}
    used = 0
    while used < budget:
        _, alive = game.play(next_key(policy, game, rng, state))
        used += 1
        # The same rules act.Session plays by: a life ending deals the game again
        # and costs only what was already spent.
        if ending(alive, budget - used) == "restart":
            game.restart()
            state["travel"] = []

    scores = game.episode_scores()
    finished = [e.score for e in game.episodes if e.ended]
    row = {
        "variant": variant,
        "seed": seed,
        "policy": policy,
        "actions_used": used,
        "best_episode": max(scores, default=0),
        "episodes_completed": len(finished),
        "mean_episode": round(sum(finished) / len(finished), 1) if finished else 0.0,
        "episode_scores": scores,
        "turns": sum(e.turns for e in game.episodes),
        "lives": len(game.episodes),
        "deaths": game.deaths,
        "endings": sorted({e.ended for e in game.episodes if e.ended}),
        "roles": [e.role for e in game.episodes],
        "max_depth": game.max_depth,
        "max_xplevel": game.max_xplevel,
        "unique_cells": game.unique_cells,
    }
    game.close()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(prog="baselines", description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="nethack", choices=("nethack", "score"))
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5, help="seeds per policy")
    parser.add_argument("--out", type=Path, help="write every row as json")
    args = parser.parse_args()

    budget = args.budget or DEFAULT_BUDGET[args.variant]
    rows = []
    print(f"{'policy':8}{'seed':>5}{'keys':>7}{'turns':>7}{'best':>7}{'mean':>7}"
          f"{'n':>4}{'lives':>7}{'died':>6}{'dlvl':>6}{'xp':>4}{'cells':>7}")
    for policy in args.policy or list(POLICIES):
        for seed in range(args.trials):
            row = rollout(args.variant, seed, policy, budget)
            rows.append(row)
            print(f"{policy:8}{seed:>5}{row['actions_used']:>7}{row['turns']:>7}"
                  f"{row['best_episode']:>7}{row['mean_episode']:>7.0f}"
                  f"{row['episodes_completed']:>4}{row['lives']:>7}{row['deaths']:>6}"
                  f"{row['max_depth']:>6}{row['max_xplevel']:>4}"
                  f"{row['unique_cells']:>7}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
