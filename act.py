#!/usr/bin/env python3
"""Deterministic actuator for NetHack.

Adapted from cc_craftax's act.py, which took it from cc_humanrl, cc_autumn and
arc-code: same contract — every action goes through here, every action is appended
verbatim to ``logs.txt`` next to the observation it produced, and nothing degrades
gracefully. If something is wrong, it raises.

Three things differ from the Craftax port it is cut down from:

* **An action is a keystroke.** Craftax had 43 named actions and each was a move.
  NetHack has one action per key, a letter is a command in the move loop and the
  answer to a question at a prompt, and a key the game is not waiting for is
  swallowed without the clock moving (F1, F2). So a batch is a sequence of keys,
  ``do`` refuses one that is not a key of this game, and nothing here knows or
  says what any of them means.
* **A life can be ended by the player, and that is the game.** Craftax removed the
  ``reset`` this harness had added, because none of its 43 actions abandoned a
  life. NetHack ships ``#quit`` and ``S``, so both stay: they are keys of the
  game as much as ``h`` is. What stops them being farmable is that every life of a
  run is dealt the same character in the same dungeon (F6, F14) — quitting to
  re-roll gets the same roll back.
* **There is no denominator.** Craftax quoted everything as a percentage of 226.
  NetHack's score has no maximum, so a run is quoted in points, beside the median
  human game and the Challenge's own bots (F10, F11).

The rule the whole harness rests on is unchanged: **the log tells the agent about
the protocol, never about the world.**
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from nle_game import CHANNELS, DEFAULT_CHANNELS, VARIANTS, NleGame

LOG = "logs.txt"
STATE = "state.json"
SOCKET = ".act.sock"
# Both live outside the workspace: result.json carries the run's aggregate across
# lives, which is the one number a session must not be able to grow.
DAEMON_LOG = "daemon.log"
RESULT = "result.json"
SEP = "=" * 80

# The median game on nethack.alt.org is 1,724 keypresses long and scores 836
# points (F10); AutoAscend's median game is 28,181 keypresses for 5,422. 3000 is
# above the median human game and a tenth of the bot's — enough to be past the
# opening and into the dungeon, not enough to grind a character up.
DEFAULT_BUDGET = {"nethack": 3000, "score": 3000}
# A batch may repeat a key: `l*12`. NetHack's own count prefix does this too —
# `2` `0` `s` is three keys and twenty turns of searching — but that is the
# game's and this is the command line's, and locomotion is most of what a run
# spends. The token may be punctuation, so anything up to the last `*` is it.
REPEAT = re.compile(r"^(.+)\*(\d+)$")
MAX_REPEAT = 500


class ActError(SystemExit):
    """Loud, non-zero exit. The message is the agent's feedback channel."""

    def __init__(self, message: str) -> None:
        super().__init__(f"act: {message}")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class RunState(BaseModel):
    """The record of a run. Not the game — the game is the daemon's NleGame."""

    variant: str = "nethack"
    seed: int = 0
    obs: list[str] = list(DEFAULT_CHANNELS)
    fresh_world: bool = False
    # Whether the game is allowed to name itself in the observations. Off by
    # default: the opening screen says `welcome to NetHack!` and so does `v`, and a
    # session that was not told what it is playing should not be told by the first
    # frame either (F18). It is `PRIVATE` for the same reason `variant` is.
    blind: bool = True
    budget: int = 0
    env_dir: str = ""

    actions_used: int = 0
    lives: int = 1
    # Two totals, because they are not the same question. `life_reward` is this life's
    # score, which is the unit the Challenge reports and the one `status` shows. The
    # run total is kept for analysis and not shown: a run is a sequence of lives and
    # a total across them is not a score anyone quotes, but it is a number a session
    # would grow — and in a game with a two-key suicide, growing it is not playing
    # (F14).
    life_reward: float = 0.0
    reward: float = 0.0
    terminal: bool = False
    outcome: str = ""  # "" | "budget"
    plan: str | None = None
    history: list[str] = []

    # -- the stint ----------------------------------------------------------- #
    # A run longer than one session is a run played by several of them, and `stint`
    # is how many actions each may play before it is over and the next takes over
    # from exactly where it stopped. 0 means the whole budget, which is what a
    # single-session run has and what every run had before 30k ones existed.
    #
    # This is protocol and not world: it says nothing about what is being played,
    # and the preamble explains it in the same breath as the budget. It also makes
    # notes.md load-bearing rather than merely advised — a successor session has the
    # workspace and nothing else.
    stint: int = 0
    stint_end: int = 0
    sessions: int = 1

    # -- the score ----------------------------------------------------------- #
    # Read off the game after every action and never written into the workspace.
    # Not because any of it is secret — the score, the depth and the experience
    # level are all on NetHack's own status line — but because the *run's* view of
    # them is: how many lives there have been, which was the best, and how they
    # compare. A session that could read its own league table would play it.
    episodes: list[dict] = []
    score: int = 0
    max_depth: int = 1
    max_xplevel: int = 1
    unique_cells: int = 0
    turns: int = 0
    deaths: int = 0
    quits: int = 0

    # `variant` is here for a reason that took a moment to see: it is the string
    # "nethack", and it is written into a file that sits in the workspace next to
    # the log the session is told to read. Withholding the name from the brief and
    # from the screen and then leaving it in `state.json` would be withholding
    # nothing. What the run *is* lives in the record beside the environment, which
    # is also where `pick_up` reads it back from.
    PRIVATE: ClassVar[set[str]] = {
        "env_dir", "variant", "blind", "episodes", "score", "max_depth",
        "max_xplevel", "unique_cells", "turns", "deaths", "quits", "reward",
    }

    def save(self, ws: Path) -> None:
        tmp = ws / f"{STATE}.tmp"
        tmp.write_text(self.model_dump_json(indent=2, exclude=self.PRIVATE))
        tmp.replace(ws / STATE)

    def record(self) -> None:
        """The full record, including the score, beside the environment.

        Written after every action rather than at the end: a session that is killed
        still leaves a readable result, and the launcher reads this rather than
        state.json for anything the agent must not see.

        Through a temporary file, like `save`, and for a sharper reason: this is
        rewritten once per action, so a 30,000-action run offers 30,000 chances to be
        killed mid-write, and the launcher builds its whole report out of what is
        here. A half-written one would lose the run it was recording.
        """
        if not self.env_dir:
            return
        best = max((e["score"] for e in self.episodes), default=0)
        # The Challenge reports the score of an *episode*, and only a life the game
        # ended is an episode: the last one is usually cut off by the budget, and
        # averaging a truncated life in with finished ones reports a number the
        # comparison cannot carry. How many there were is recorded beside the mean,
        # because on a 3000-key run it is often one or two.
        done = [e["score"] for e in self.episodes if e["ended"]]
        mean = sum(done) / len(done) if done else 0.0
        body = self.model_dump() | {
            "episode_scores": [e["score"] for e in self.episodes],
            "best_episode": best,
            "episodes_completed": len(done),
            "mean_episode": round(mean, 2),
            # NetHack's own vocabulary for how each life ended, and which character
            # each was dealt: the roles differ enormously in difficulty, so a score
            # that does not say who was playing says less than it looks like.
            "endings": sorted({e["ended"] for e in self.episodes if e["ended"]}),
            "roles": [e["role"] for e in self.episodes],
        }
        tmp = Path(self.env_dir, f"{RESULT}.tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(Path(self.env_dir, RESULT))

    @property
    def actions_left(self) -> int:
        return self.budget - self.actions_used

    @property
    def stint_left(self) -> int:
        """Actions left in this session's share, which is the budget without one."""
        if not self.stint:
            return self.actions_left
        return min(self.actions_left, self.stint_end - self.actions_used)

    @property
    def stint_over(self) -> bool:
        """This session is done and the run is not. Distinct from `terminal`.

        A stint that runs out at the same moment the budget does is the run ending,
        not a handover, so a terminal run is never reported this way.
        """
        return bool(self.stint) and not self.terminal and self.stint_left <= 0

    def begin_stint(self, size: int) -> None:
        """Hand the next `size` actions to a new session."""
        self.stint = size
        self.stint_end = self.actions_used + size if size else 0


# --------------------------------------------------------------------------- #
# The log format every parser downstream depends on
# --------------------------------------------------------------------------- #


def wrap(items: tuple[str, ...], width: int = 68, lead: str = "#   ") -> str:
    out, line = [], lead
    for item in items:
        if len(line) + len(item) + 1 > width and line != lead:
            out.append(line.rstrip())
            line = lead
        line += item + " "
    out.append(line.rstrip())
    return "\n".join(out)


def preamble(state: RunState, tokens: tuple[str, ...], channels: dict[str, str]) -> str:
    seen = "\n".join(
        f"#   [{channel}] {Path(name).parent}/NNNNNN{Path(name).suffix} — one per action"
        for channel in channels
        for name in [channels[channel]]
    )
    return (
        f"# logs.txt — the complete record of this run, written by the actuator.\n"
        f"# One block per action: a ruler of = signs, then\n"
        f"#   action N | budget used/total | reward R | <what was played> | step i/k\n"
        f"# then the plan that batch was given, then the observation that action\n"
        f"# produced — one line per channel, each naming a file.\n"
        f"#\n"
        f"# observations:\n{seen}\n"
        f"#   Action 000000 is the state before anything was played. Nothing else\n"
        f"#   about an observation is written here: what is in it is for you to work\n"
        f"#   out.\n"
        f"#\n"
        f"# keys: every action is one keystroke, and these are the ones this game\n"
        f"#   accepts. A key the game is not waiting for costs an action and does\n"
        f"#   nothing else.\n{wrap(tokens)}\n"
        f"#   `./act do h h l cr` plays them in order; any key may be repeated with\n"
        f"#   *N, as in `h*12`. Quote the ones your shell would eat. `esc`, `cr`\n"
        f"#   and `space` are those keys under names a command line can carry, and\n"
        f"#   a `#word` is one key too — the extended command of that name.\n"
        f"#   A batch stops early if this life ends, and the rest of it is dropped:\n"
        f"#   the state you planned against is gone.\n"
        f"# reward: what the action was worth, when it was worth anything. It is\n"
        f"#   the change in a number the game itself shows you. `status` totals it\n"
        f"#   for the life you are in, and a new life starts that total again.\n"
        f"# budget: {state.budget} actions for the whole run, across every life.\n"
        f"#   Every action counts, including the ones that turn out to do nothing.\n"
        f"#   There is one phase and nothing is held back for later.\n"
        f"{stint_note(state)}"
        f"# [event] lines are things the actuator did that you did not ask for:\n"
        f"#   life over — this life ended and the next one began. The actions\n"
        f"#     already spent stay spent. That block holds two observations: first\n"
        f"#     the one the action produced, which is how the life ended, and then\n"
        f"#     a `-restart` one, what the next life began in.\n"
        f"#   run over — the budget is spent and nothing further can be played.\n"
        f"#   session over — your stint is spent. See `stint` above.\n"
        f"#   resumed — the actuator was restarted and rebuilt this run from its\n"
        f"#     record. The game is exactly as the last action left it.\n"
    )


def stint_note(state: RunState) -> str:
    """What a session is told about being one of several. Nothing, when it is not."""
    if not state.stint:
        return ""
    return (
        f"# stint: {state.stint} of those actions are yours. The budget belongs to\n"
        f"#   the run, not to you: when your stint is spent this session is over and\n"
        f"#   another one continues from exactly where you stopped, in the same\n"
        f"#   game, with this workspace. It will have your files and nothing else —\n"
        f"#   not your reasoning, not what you were about to try. `notes.md` is how\n"
        f"#   you talk to it, and the only way anything you worked out survives.\n"
    )


def header(state: RunState, reward: float, played: str, index: int, total: int) -> str:
    bits = [
        f"action {state.actions_used}",
        f"budget {state.actions_used}/{state.budget}",
    ]
    if reward:
        bits.append(f"reward {reward:+g}")
    bits.append(played)
    if total:
        bits.append(f"step {index}/{total}")
    return " | ".join(bits)


def ending(alive: bool, actions_left: int) -> str | None:
    """How this action ended the life, or the run, or neither.

    The rules in one place, because the baselines have to play by exactly the ones
    the agent plays by or the floor they measure is a floor for a different game.
    There is no reward that means victory and no move that means defeat: a life
    ends because NetHack ended it — death, starvation, a quit the player asked for,
    or the abort a game that has stopped moving eventually gets — and the run ends
    only when the budget does.
    """
    if not alive:
        return "restart"
    return "budget" if actions_left <= 0 else None


# --------------------------------------------------------------------------- #
# Parsing actions
# --------------------------------------------------------------------------- #


def parse_tokens(tokens: list[str], playable: tuple[str, ...]) -> list[str]:
    """``h`` ``l*12`` -> a flat list of keys. Anything else raises.

    Validation happens before a single key reaches the game, so that a typo
    halfway through a batch of sixty costs nothing and says why.

    Case is *not* folded, unlike every earlier port of this function. `H` and `h`
    are different keys — one runs west and the other steps west — and so are `S`
    and `s`, `D` and `d`. Lowercasing a batch would quietly turn twelve runs into
    twelve steps.
    """
    out: list[str] = []
    for raw in tokens:
        token, times = raw.strip(), 1
        match = REPEAT.match(token)
        if match:
            token, times = match.group(1), int(match.group(2))
            if not 1 <= times <= MAX_REPEAT:
                raise ActError(f"{raw!r} repeats {times} times — want 1 to {MAX_REPEAT}")
        if token not in playable:
            near = " ".join(k for k in playable if k.lower() == token.lower())
            hint = f" — did you mean {near}?" if near else ""
            raise ActError(f"{token!r} is not a key of this game{hint}")
        out.extend([token] * times)
    if not out:
        raise ActError("no keys given")
    return out


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class Session:
    """The world, the log and the record — everything the daemon owns."""

    def __init__(self, ws: Path, state: RunState) -> None:
        self.ws = ws
        self.state = state
        self.game = NleGame(
            state.variant,
            seed=state.seed,
            obs=tuple(state.obs),
            fresh_world=state.fresh_world,
            blind=state.blind,
            # NetHack's own recording of every life, in the format its `ttyplay`
            # reads, plus the xlogfile it writes when a game ends (F16). Beside the
            # environment and never in the workspace: the xlogfile names how each
            # life ended and what it scored.
            savedir=str(Path(state.env_dir, "ttyrec")) if state.env_dir else None,
        )
        self.last = {}
        self.stopped = False
        # Set only while `resume` is rebuilding a run from its history. It suppresses
        # the three things `play` writes — the observation, the log block, the record
        # — because all three are already on disk from when the action was first
        # played. Everything else about the control flow is shared, which is what
        # makes a rebuilt run the same run rather than a similar one.
        self.replaying = False

    @classmethod
    def create(cls, ws: Path, state: RunState) -> "Session":
        session = cls(ws, state)
        session.last = session.game.observe(ws, 0)
        session.measure()
        block = (
            f"{preamble(state, session.game.tokens, session.last)}{SEP}\n"
            f"{header(state, 0.0, 'start', 0, 0)}\n\n"
            f"{session.seen(session.last)}\n"
        )
        (ws / LOG).write_text(block)
        state.save(ws)
        state.record()
        return session

    @classmethod
    def resume(cls, ws: Path, state: RunState, history: list[str]) -> "Session":
        """Rebuild a run that is already part-played and carry on from where it is.

        The game is not saved anywhere and does not have to be. NetHack is a pure
        function of its two seeds and the keys pressed, once its anti-TAS reseeding
        is off (F7) — so playing the recorded history back into a fresh game
        reproduces the state the last session left: the map, the inventory, the
        monsters, the hunger clock, the character's own history. A run that had
        several lives replays each of them, because the seeds for life *n* are a
        function of *n* and nothing else (F6).

        What makes this a restoration rather than a re-enactment is that every
        action goes through `play` like any other, so the life boundaries, the
        counters and the score come out the way they went in. The check at the end
        is the proof: a history that does not replay to its own length is not a
        history of this game.

        About 0.3ms an action, so a run resumed at 27,000 actions costs about ten
        seconds of boot — the one place NetHack is cheaper to hold than Craftax
        was, since there is no compile to pay for. `logs.txt` and the observations
        are left exactly as they are, because they already hold these actions and
        this is the same run.
        """
        if state.budget < len(history):
            raise ActError(
                f"{len(history)} actions have already been played and the budget "
                f"given is {state.budget} — resuming needs a budget at least as "
                f"large as what is already spent"
            )
        state.actions_used, state.lives = 0, 1
        state.reward = state.life_reward = 0.0
        state.terminal, state.outcome = False, ""
        state.history, state.stint, state.stint_end = [], 0, 0

        session = cls(ws, state)
        session.replaying = True
        try:
            for token in history:
                session.play(token)
        finally:
            session.replaying = False
        if state.actions_used != len(history):
            raise ActError(
                f"the record holds {len(history)} actions but replaying them played "
                f"{state.actions_used} — this history is not a history of this game"
            )

        # Written beside the screen the last action produced rather than over it: the
        # run has not moved, and the observation the previous session was looking at
        # when it stopped is part of the record.
        session.last = session.game.observe(ws, state.actions_used, tag="-resume")
        state.save(ws)
        state.record()
        return session

    def announce(self, note: str, tag: str) -> None:
        """Append a block that no action produced — a handover, not a move."""
        state = self.state
        block = [
            f"\n{SEP}\n",
            (f"action {state.actions_used} | "
             f"budget {state.actions_used}/{state.budget} | "
             f"session {state.sessions}\n\n"),
            textwrap.fill(f"[event] {tag}: {note}", width=79,
                          subsequent_indent="        ") + "\n",
            self.seen(self.last),
            "\n",
        ]
        with (self.ws / LOG).open("a") as log:
            log.write("".join(block))

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def seen(written: dict[str, str]) -> str:
        return "".join(f"[{channel}] {name}\n" for channel, name in written.items())

    @property
    def playable(self) -> tuple[str, ...]:
        """The game's own keys, and nothing else."""
        return self.game.tokens

    def summary(self) -> str:
        state = self.state
        line = (
            f"budget {state.actions_used}/{state.budget} "
            f"life {state.lives} reward {state.life_reward:+g} this life "
            f"over {state.terminal}"
        )
        if state.terminal:
            line += f" outcome {state.outcome}"
        elif state.stint:
            line += (
                f" | session {state.sessions}, {state.stint_left} of your "
                f"{state.stint} left"
            )
        return line

    def check_budget(self, wanted: int) -> None:
        state = self.state
        if state.terminal:
            raise ActError("this run is over — nothing left to play")
        if state.stint_over:
            raise ActError(
                "your stint is spent — this session is over, and the run is not. "
                "Another session continues from here with this workspace and "
                "nothing else, so anything you have not written down is lost."
            )
        if wanted > state.stint_left:
            whose = ("left in the budget" if state.stint_left == state.actions_left
                     else "left in your stint")
            raise ActError(f"{wanted} actions requested but only {state.stint_left} {whose}")

    def measure(self) -> None:
        """Read the run's own view of itself off the game.

        None of this reaches the workspace. The per-life numbers are all on the
        status line the session can already see; what it must not see is the table
        of them.
        """
        state, game = self.state, self.game
        state.episodes = [
            {
                "index": e.index,
                "actions": e.actions,
                "score": e.score,
                "reward": round(e.reward, 3),
                "turns": e.turns,
                "depth": e.depth,
                "xplevel": e.xplevel,
                "gold": e.gold,
                "role": e.role,
                "ended": e.ended,
            }
            for e in game.episodes
        ]
        state.score = max((e.score for e in game.episodes), default=0)
        state.turns = sum(e.turns for e in game.episodes)
        state.unique_cells = game.unique_cells
        state.max_depth = game.max_depth
        state.max_xplevel = game.max_xplevel
        state.deaths = game.deaths
        state.quits = game.quits

    # -- playing ------------------------------------------------------------- #

    def play(self, token: str, index: int = 0, total: int = 0) -> str | None:
        """Send one action, log it, record it. The only path to the environment.

        Returns a reason the batch should stop, or None.
        """
        state = self.state
        reward, alive = self.game.play(token)

        state.actions_used += 1
        state.reward += reward
        state.life_reward += reward
        state.history.append(token)
        self.measure()

        # Written before anything else happens to the game: this is the screen the
        # action produced, and for the action that ends a life it is the only view
        # of how that life ended — NetHack's tombstone, and the only place it is
        # ever drawn.
        produced = self.look()
        body = [self.seen(produced)]
        self.last = produced

        stop = None
        match ending(alive, state.actions_left):
            case "restart":
                # The life ended and the run carries on into the next one. Two
                # observations, in the order they happened: how it ended, then what
                # it began again in.
                how = self.game.episodes[-1].ended
                self.game.restart()
                state.lives += 1
                state.life_reward = 0.0
                self.measure()
                restarted = self.look(tag="-restart")
                body += [f"[event] life over: {how}\n", self.seen(restarted)]
                self.last = restarted
                stop = f"life over: {how}"
            case "budget":
                state.terminal, state.outcome = True, "budget"
                stop = "budget spent"

        # After the life and the run, because it is the weakest of the three: a
        # stint that ends on the same action the budget does is the run ending.
        if state.stint_over:
            body.append(
                "[event] session over: your stint is spent. The run is not over — "
                "another session continues from here.\n"
            )
            stop = f"{stop}, and your stint is spent" if stop else "stint spent"

        block = [f"\n{SEP}\n{header(state, reward, token, index, total)}\n\n"]
        if state.plan:
            block.append(f"plan: {state.plan}\n\n")
        block.extend(body)
        if state.terminal:
            block.append(f"[event] run over: {state.outcome}\n")
        block.append("\n")
        if not self.replaying:
            with (self.ws / LOG).open("a") as log:
                log.write("".join(block))
            state.save(self.ws)
            state.record()
        if stop and total - index:
            stop += f", {total - index} action(s) dropped"
        return stop

    def look(self, tag: str = "") -> dict[str, str]:
        """The observation this action produced — unless the action is being replayed.

        A rebuilt run re-derives the game, not the record. The screens are already
        on disk from when these actions were first played, and writing them again
        would cost a file apiece to produce files that are already correct.
        """
        if self.replaying:
            return {}
        return self.game.observe(self.ws, self.state.actions_used, tag=tag)


# --------------------------------------------------------------------------- #
# Commands, as the daemon runs them
# --------------------------------------------------------------------------- #


def cmd_do(args: argparse.Namespace, s: Session) -> str:
    steps = parse_tokens(args.actions, s.playable)
    s.check_budget(len(steps))
    if args.plan:
        s.state.plan = args.plan

    done, stopped = [], None
    for index, token in enumerate(steps, start=1):
        stopped = s.play(token, index, len(steps))
        done.append(token)
        if stopped:
            break
    out = f"ran {len(done)}/{len(steps)}: {' '.join(done)}\n"
    if stopped:
        out += f"stopped early: {stopped}\n"
    return out + s.summary() + "\n"


def cmd_status(args: argparse.Namespace, s: Session) -> str:
    out = s.summary() + "\n"
    if s.state.stint_over:
        out += (
            "this session is over — your stint is spent. The run is not over: "
            "another session continues from here with this workspace.\n"
        )
    elif not s.state.terminal:
        out += f"./act do {' '.join(s.playable)}\n"
    return out


def cmd_board(args: argparse.Namespace, s: Session) -> str:
    """Where the latest observation is. Not what is in it."""
    return f"action {s.state.actions_used}\n{s.seen(s.last)}"


def cmd_stop(args: argparse.Namespace, s: Session) -> str:
    # Unlinked before replying, not on the way out: otherwise the caller returns
    # while the socket is still there and the next command connects to a daemon
    # that is already leaving.
    (s.ws / SOCKET).unlink(missing_ok=True)
    s.stopped = True
    return "stopped\n"


COMMANDS = {
    "do": cmd_do,
    "status": cmd_status,
    "board": cmd_board,
    "stop": cmd_stop,
}


# --------------------------------------------------------------------------- #
# Daemon and client
# --------------------------------------------------------------------------- #


def serve(args: argparse.Namespace) -> int:
    """Own the world for the life of the run and answer commands."""
    ws = Path.cwd()
    # Bound and connected to by its bare name, because a unix socket address is
    # limited to about a hundred bytes and a workspace can be nested deeper than
    # that. Every command runs from the workspace, so the name is enough.
    path = ws / SOCKET
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET)
    server.listen(1)

    env_dir = Path(args.env_dir) if args.env_dir else ws / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Building the game starts a NetHack process and, on a resume, replays the
        # whole recorded history into it. It happens here, before the first command
        # is answered, so that a session's first `./act do` is fast rather than
        # looking hung.
        session = pick_up(ws, args, env_dir) if args.resume else start(ws, args, env_dir)
        print(f"serving a run on {path}", flush=True)
        while not session.stopped:
            conn, _ = server.accept()
            with conn:
                try:
                    payload = json.loads(read_line(conn))
                    command = COMMANDS[payload["command"]]
                    wanted = argparse.Namespace(**payload["args"])
                    reply = {"stdout": command(wanted, session), "rc": 0}
                except SystemExit as exc:
                    reply = {"stdout": f"{exc}\n", "rc": 1}
                except Exception as exc:  # a bug here must not silently strand the agent
                    reply = {"stdout": f"act: {type(exc).__name__}: {exc}\n", "rc": 1}
                try:
                    conn.sendall(json.dumps(reply).encode() + b"\n")
                except OSError as exc:
                    # The client went away before its reply: the agent's shell was
                    # killed, or a long batch outlived whatever was waiting on it.
                    # The actions it asked for are already played and recorded, and
                    # there is nobody left to tell. Losing a reply is not losing the
                    # run, so the loop carries on — before this, one hung-up client
                    # took the daemon down with a BrokenPipeError, the launcher's
                    # shutdown then failed, and the failure discarded the report.
                    print(f"a client hung up before its reply: {exc}", flush=True)
    finally:
        # Whatever happened, leave no socket behind for the next command to trust.
        server.close()
        path.unlink(missing_ok=True)
    return 0


def start(ws: Path, args: argparse.Namespace, env_dir: Path) -> Session:
    """A run from nothing: a new world, an empty log, action zero."""
    state = RunState(
        variant=args.variant,
        seed=args.seed,
        obs=list(args.obs),
        fresh_world=args.fresh_world,
        blind=not args.named,
        budget=args.budget or DEFAULT_BUDGET[args.variant],
        env_dir=str(env_dir),
    )
    # Before `create`, because the preamble it writes is where a session is told it
    # has a stint at all.
    state.begin_stint(args.stint)
    return Session.create(ws, state)


def pick_up(ws: Path, args: argparse.Namespace, env_dir: Path) -> Session:
    """A run that is already part-played, rebuilt from its own record.

    What the run *is* — the variant, the seed, the channels, whether each life gets a
    fresh game, whether the game may name itself — comes from **the record beside the
    environment** and never from the arguments. Those five define the game the
    recorded history is a history of, and a launcher that disagreed with them would
    rebuild a different game and call it the same run.

    From the record and not from `state.json`, because `state.json` sits in the
    workspace and no longer carries the variant or the blind flag: they are the two
    fields that would tell a session what it is playing (F18). The budget and the
    stint are the launcher's to set, and are the only reason this exists: a run is
    extended by resuming it with a larger budget.
    """
    if not (ws / STATE).exists():
        raise ActError(f"{ws} holds no run to resume — there is no {STATE}")
    record = Path(env_dir, RESULT)
    if not record.exists():
        raise ActError(
            f"{ws} holds a run but {record} does not — the record beside the "
            f"environment is what says which game this is a run of, so a resume "
            f"needs the --env-dir the run was played under"
        )
    was = json.loads(record.read_text())
    state = RunState(
        variant=was["variant"],
        seed=was["seed"],
        obs=list(was["obs"]),
        fresh_world=was["fresh_world"],
        blind=bool(was.get("blind", True)),
        budget=args.budget or was["budget"],
        env_dir=str(env_dir),
        sessions=int(was.get("sessions", 1)) + 1,
    )
    session = Session.resume(ws, state, list(was["history"]))
    state.begin_stint(args.stint)
    grew = state.budget - was["budget"]
    session.announce(
        f"this run was rebuilt from its record at action {state.actions_used} and "
        f"continues. The world is exactly as that action left it."
        + (f" The budget is now {state.budget}, {grew} more than the blocks above "
           f"were played under." if grew > 0 else "")
        + (f" {state.stint} of the {state.actions_left} remaining actions are this "
           f"session's; the rest belong to the sessions after it." if state.stint else ""),
        tag="resumed",
    )
    # Both, not just state.json: the stint is decided after the rebuild, and the
    # launcher reads the record rather than the workspace for anything it has to be
    # sure of.
    state.save(ws)
    state.record()
    return session


def read_line(conn: socket.socket) -> bytes:
    chunks = []
    while not (chunks and chunks[-1].endswith(b"\n")):
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def call(ws: Path, command: str, args: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    if not (ws / SOCKET).exists():
        raise ActError(f"no game running in {ws} — run `act init` first")
    # See serve(): addressed by its bare name, from the workspace.
    target = SOCKET if ws == Path.cwd() else str(ws / SOCKET)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(target)
        conn.sendall(json.dumps({"command": command, "args": args}).encode() + b"\n")
        reply = read_line(conn)
    except OSError as exc:
        raise ActError(f"the environment is not answering: {exc}") from exc
    finally:
        conn.close()
    if not reply:
        raise ActError("the environment closed the connection without answering")
    return json.loads(reply)


# What one attempt leaves behind, and what it is called once another begins. The
# log and the observations are act.py's and the stream is run.py's, but they are
# one attempt and have to be numbered together: rotating only the log let a replay
# overwrite the verdict of the session it was replaying.
#
# `report.json` is deliberately not here, because it is deliberately not in the
# workspace: it carries the run's league table across lives, and a replay is *told*
# to read what its predecessor left behind. The launcher keeps it under `.rig/`.
ATTEMPT = {
    LOG: "logs-attempt{n}.txt",
    "agent_stream.jsonl": "agent_stream-attempt{n}.jsonl",
    "screens": "screens-attempt{n}",
    "ansi": "ansi-attempt{n}",
    "frames": "frames-attempt{n}",
    "symbolic": "symbolic-attempt{n}",
}


def hang_up(ws: Path) -> None:
    """Shut down whatever daemon is running here, and leave no socket to trust."""
    if (ws / SOCKET).exists():
        try:
            call(ws, "stop", {})
        except SystemExit:
            pass
        (ws / SOCKET).unlink(missing_ok=True)


def restart(ws: Path) -> None:
    """Clear the workspace's *game* while leaving its memory alone.

    Notes and helper scripts stay put; the previous attempt is kept beside the new
    one rather than appended to or replaced, because action numbering restarts and
    two runs interleaved in one file would parse as neither. The observations go
    with it for the same reason: screens/000012.txt would otherwise mean two states.
    """
    hang_up(ws)
    taken = (n for n in count(1)
             if not any((ws / name.format(n=n)).exists() for name in ATTEMPT.values()))
    number = next(taken)
    for name, pattern in ATTEMPT.items():
        found = ws / name
        if found.exists():
            found.rename(ws / pattern.format(n=number))
    (ws / STATE).unlink(missing_ok=True)


def cmd_init(args: argparse.Namespace) -> int:
    ws = Path.cwd()
    if args.again and args.resume:
        raise ActError("--again starts the world over and --resume continues it")
    if (ws / STATE).exists():
        if args.resume:
            # Nothing is rotated and nothing is cleared. The log, the observations
            # and the record are this run's and the daemon is about to rebuild the
            # world that produced them; all that has to go is the old socket.
            hang_up(ws)
        elif args.again:
            restart(ws)
        else:
            raise ActError(
                f"{ws} already holds a game — use a fresh directory, --again to "
                f"start it over, or --resume to carry it on"
            )
    elif args.resume:
        raise ActError(f"{ws} holds no run to resume — there is no {STATE}")

    argv = [
        sys.executable,
        os.path.abspath(__file__),
        "serve",
        "--variant", args.variant,
        "--seed", str(args.seed),
        "--budget", str(args.budget or DEFAULT_BUDGET[args.variant]),
        "--stint", str(args.stint),
        "--obs", *args.obs,
    ]
    if args.fresh_world:
        argv.append("--fresh-world")
    if args.named:
        argv.append("--named")
    if args.resume:
        argv.append("--resume")
    env_dir = Path(args.env_dir or tempfile.mkdtemp(prefix="act-env-"))
    # mkdtemp makes its own; a directory named by the launcher may not exist yet,
    # and the daemon log below is opened before the daemon runs.
    env_dir.mkdir(parents=True, exist_ok=True)
    argv += ["--env-dir", str(env_dir)]
    with (env_dir / DAEMON_LOG).open("w") as daemon_log:
        proc = subprocess.Popen(
            argv,
            cwd=ws,
            stdin=subprocess.DEVNULL,
            stdout=daemon_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def died(problem: str) -> ActError:
        return ActError(f"{problem}:\n{(env_dir / DAEMON_LOG).read_text()[-2000:]}")

    deadline = time.monotonic() + args.start_timeout
    while not (ws / SOCKET).exists():
        if proc.poll() is not None:
            raise died("the environment failed to start")
        if time.monotonic() > deadline:
            proc.kill()
            raise ActError(f"the environment did not start within {args.start_timeout}s")
        time.sleep(0.05)

    # The socket is bound before the world is built, so this first command is what
    # waits for it — and what surfaces a failure while building it.
    try:
        reply = call(ws, "status", {})
    except SystemExit:
        raise died("the environment failed to start") from None
    sys.stdout.write("ready\n" + reply["stdout"])
    reply = call(ws, "board", {})
    sys.stdout.write(reply["stdout"])
    return int(reply["rc"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="act", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--variant", choices=VARIANTS, default="nethack")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget", type=int, default=0,
                       help=f"default {DEFAULT_BUDGET} by variant")
        p.add_argument("--obs", nargs="+", choices=CHANNELS, default=list(DEFAULT_CHANNELS),
                       help="which of the package's own observations to write")
        p.add_argument("--fresh-world", action="store_true",
                       help="roll a new character and dungeon on each life instead "
                            "of dealing this one again")
        p.add_argument("--named", action="store_true",
                       help="let the game name itself in the observations. The "
                            "default redacts that one word, because a session that "
                            "was not told what it is playing should not be told by "
                            "the first screen either")
        p.add_argument("--stint", type=int, default=0,
                       help="actions one session may play before another takes over "
                            "(0: the whole budget, in one session)")
        p.add_argument("--resume", action="store_true",
                       help="rebuild the run already recorded here and carry it on")
        p.add_argument(
            "--env-dir", default=None,
            help="where the environment's log and the result record go, outside the "
                 "workspace",
        )

    p_init = sub.add_parser("init", help="start a run in this directory")
    add_env_flags(p_init)
    p_init.add_argument("--again", action="store_true",
                        help="start the world over, keeping the notes")
    p_init.add_argument("--start-timeout", type=float, default=300.0)

    p_serve = sub.add_parser("serve", help=argparse.SUPPRESS)
    add_env_flags(p_serve)

    p_do = sub.add_parser("do", help="play keys, in order, until this life ends")
    p_do.add_argument("actions", nargs="+", metavar="KEY", help="e.g. h l*12 cr")
    p_do.add_argument("--plan", help="briefing recorded in the log beside these actions")

    sub.add_parser("status", help="one-line summary and what is playable now")
    sub.add_parser("board", help="where the latest observation is")
    sub.add_parser("stop", help="shut the environment down")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args)
    if args.command == "init":
        return cmd_init(args)
    values = {k: v for k, v in vars(args).items() if k != "command"}
    reply = call(Path.cwd(), args.command, values)
    sys.stdout.write(reply["stdout"])
    return int(reply["rc"])


if __name__ == "__main__":
    sys.exit(main())
