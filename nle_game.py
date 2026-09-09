#!/usr/bin/env python3
"""One NetHack game, held in this process.

NetHack is a 1987 roguelike with no goal short of ascension: descend, survive,
and score. This wraps one instance of the NetHack Learning Environment so the
actuator can treat it as "play a key, get a screen" — and holds the four things
the wrapper exists to own.

**The action space is the keyboard.** NLE's Challenge configuration hands over all
121 actions of `nethack.ACTIONS`, which are 118 distinct bytes (F1). A letter is
both a command and the answer to a prompt: `d` is "drop" in the move loop and the
inventory letter `d` at "What do you want to eat?". So an action here is a
keystroke and nothing else, and this file names them by the key rather than by the
command — naming `d` "drop" would be a lie at every prompt. The non-printable keys
get words (`esc`, `cr`, `^d`) and the meta keys get NetHack's own extended-command
names (`#pray` is M-p), because that is what the game calls them.

**The seeds.** `NetHackChallenge-v0` refuses to be seeded — it overwrites the seed
setters on the underlying game object as an anti-TAS measure for the competition
(F5) — so a run of it can never be replayed, and a run that cannot be replayed
cannot be resumed. The same configuration built on its parent `NetHackScore` is
seedable, and the seed fixes the dungeon *and* the character. Seeding is per-reset
and not sticky (F6): the second life of a seeded game is a different character in a
different dungeon unless the harness seeds again, which is why `_begin` seeds every
life. With `reseed=False` the same keys replay bit-identically (F7), which is what
makes a death cost only the keys already spent and a stint hand over to the next
session.

**The score.** `env.step`'s reward is the in-game score delta, plus a time penalty
this run sets to zero (F3). The score is already on the status line — NetHack draws
`S:` there — so showing the reward gives nothing away that the screen does not.
What the harness keeps to itself is the aggregate across lives, for the reason
`cc_craftax` learned the hard way: a session shown a total will optimise the total.

**The observation channels.** NLE renders the terminal (`tty_chars`, `tty_colors`),
a symbolic view (`glyphs`, `chars`, `blstats`, the inventory) and — since 1.0 — a
tile-drawn frame from NetHack's own tileset (F9). Which of them a run gives out is
configuration and lives here, so `act.py` only ever asks for what the run was
configured to give. The tile frame draws **only the map**: no message line, no
status line, no menus. A run given pixels alone would be a run whose agent cannot
read "You die..." — so that combination is refused at construction rather than
producing an unplayable workspace.

Two upstream facts a caller must not take on trust:

* `reset()` presses SPACE up to a thousand times to get past NetHack's opening
  menus. Those keys are NLE's, not the run's, and the first observation is already
  in the move loop.
* `--More--` and menus are *not* skipped in this configuration, and a key the
  prompt does not accept is swallowed without advancing the game (F2). A policy
  that never sends `cr` stops playing NetHack at the first message and does not
  notice.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VARIANTS = ("nethack", "score")
CHANNELS = ("tty", "ansi", "pixels", "symbolic")
DEFAULT_CHANNELS = ("tty",)

# The screen NetHack draws, and the shape every channel is derived from.
ROWS, COLS = 24, 80


class GameError(RuntimeError):
    """Something about the environment is wrong. Never degrade, always raise."""


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
# The token vocabulary is the key table and nothing else (F1). Printable keys are
# themselves; the seven control keys and the meta keys get names, because a shell
# cannot pass a raw escape byte and because NetHack itself calls M-p "#pray".

CONTROL = {13: "cr", 27: "esc"}  # the two with a spelling of their own
SPACE = "space"


def key_token(byte: int, name: str) -> str:
    """What one action of the game is called on the command line."""
    if byte in CONTROL:
        return CONTROL[byte]
    if byte == 32:
        return SPACE
    if byte < 32:
        return f"^{chr(byte + 64).lower()}"
    if byte < 127:
        return chr(byte)
    # 128..255 are NetHack's meta keys, which are its extended commands.
    return f"#{name.lower()}"


@dataclass
class Spec:
    """Everything that differs between the two shipped configurations."""

    actions: tuple[Any, ...]  # the nethack.ACTIONS members, in index order
    tokens: tuple[str, ...]  # the distinct keys, in that order
    index: dict[str, int]  # key -> the action index that sends it
    kwargs: dict
    skips_menus: bool


@functools.lru_cache(maxsize=None)
def _spec(variant: str) -> Spec:
    """Build the per-variant table. Imported late: nle costs a second to load."""
    from nle import nethack  # noqa: PLC0415
    from nle.env import tasks  # noqa: PLC0415

    common = dict(
        # Every channel the run might be asked for is read off one step, so the
        # keys are the same whatever `obs` says. What differs is what gets written.
        observation_keys=(
            "tty_chars", "tty_colors", "tty_cursor", "blstats", "message",
            "glyphs", "chars", "colors", "specials", "misc",
            "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
        ),
        # The score task's default is -0.01 per frozen step, which would make the
        # reward a comment on the interface rather than on the game: reading the
        # inventory would cost points. The Challenge sets both to zero and so do we.
        penalty_mode="constant",
        penalty_step=0.0,
        penalty_time=0.0,
        max_episode_steps=int(1e6),
    )
    if variant == "nethack":
        # NetHackChallenge's own configuration, built on the parent class that will
        # accept a seed (F5). Everything here is copied from `tasks.NetHackChallenge`:
        # the full keyboard, a rolled character, and menus left alone.
        actions = tuple(nethack.ACTIONS)
        kwargs = common | dict(
            actions=actions,
            character="@",
            allow_all_yn_questions=True,
            allow_all_modes=True,
        )
        skips_menus = False
    else:
        # The NLE paper's score task as registered: 23 actions, and NLE steps past
        # menus and --More-- on the agent's behalf. This is the configuration the
        # published RL numbers are on, and it is a different game to play.
        actions = tuple(tasks.TASK_ACTIONS)
        kwargs = common | dict(actions=actions, character="@")
        skips_menus = True
    # 121 actions over 118 distinct bytes: `+`, `"` and `$` each appear twice,
    # once as a command and once as a text character (F1). They are the same
    # keystroke either way, so the vocabulary is the distinct keys and the first
    # index that sends each one.
    index: dict[str, int] = {}
    for i, action in enumerate(actions):
        index.setdefault(key_token(int(action), action.name), i)
    return Spec(actions=actions, tokens=tuple(index), index=index,
                kwargs=kwargs, skips_menus=skips_menus)


@dataclass
class Episode:
    """One life, and what it reached before it ended."""

    index: int
    actions: int = 0
    score: int = 0
    reward: float = 0.0
    turns: int = 0
    depth: int = 1
    xplevel: int = 1
    gold: int = 0
    role: str = ""
    ended: str = ""  # "" | "death" | "quit" | "escaped" | "ascended" | "aborted"

    @property
    def alive(self) -> bool:
        return not self.ended


class NleGame:
    """One game of NetHack, one budget's worth of screens."""

    def __init__(
        self,
        variant: str = "nethack",
        seed: int = 0,
        obs: tuple[str, ...] = DEFAULT_CHANNELS,
        fresh_world: bool = False,
        savedir: str | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise GameError(f"{variant!r} is not one of {VARIANTS}")
        channels = tuple(dict.fromkeys(obs))
        if not channels:
            raise GameError("a run with no observation channel is not playable")
        for channel in channels:
            if channel not in CHANNELS:
                raise GameError(f"{channel!r} is not one of {CHANNELS}")
        if channels == ("pixels",):
            raise GameError(
                "the tile renderer draws the map and nothing else — no message "
                "line, no status line, no menus (F9). A run given pixels alone "
                "cannot read what happened to it; pair it with tty"
            )

        from nle.env import tasks  # noqa: PLC0415

        self.variant = variant
        self.seed = seed
        self.channels = channels
        self.fresh_world = fresh_world
        self.spec = _spec(variant)

        kwargs = dict(self.spec.kwargs)
        if savedir:
            # NetHack's own recording of the game, in the format `ttyplay` reads,
            # plus the xlogfile it writes when a game ends (F16). Harness-side: it
            # goes beside the environment's log, never into the workspace.
            kwargs |= dict(savedir=savedir, save_ttyrec_every=1)
        self._env = tasks.NetHackScore(**kwargs)
        # The Challenge aborts a life that has not advanced the clock in 10,000
        # keystrokes. Kept, because it is the shipped behaviour and because a
        # session stuck at a prompt should lose the life rather than the run.
        self._no_progress_limit = 10_000
        self._frozen = 0
        self._turn = 0

        self.episodes: list[Episode] = []
        self.reward = 0.0  # the raw env reward, summed over the whole run
        self._cells: set[tuple[int, int, int]] = set()
        self._max_depth = 1
        self._max_xplevel = 1
        self._obs: dict[str, Any] = {}
        self._begin()

    # -- the environment ----------------------------------------------------- #

    def _seeds(self, life: int) -> tuple[int, int]:
        """The seeds for one life.

        By default every life of a run is dealt the *same* game: the same dungeon,
        the same character, the same first three levels. What the session learned
        about this map still applies, and a life it has already played replays the
        same way. `fresh_world=True` rolls a new one per life, which is what the
        Challenge does and is the generalisation setting rather than the default.

        Two seeds, not one, because NetHack has two RNGs and NLE takes both: core
        for the game and disp for the anti-TAS display stream. They are derived
        apart so that the game a seed draws and the noise it plays out cannot be
        confused for one another.
        """
        base = self.seed + (life if self.fresh_world else 0) * 1_000_003
        return base, base + 7_919

    def _begin(self) -> None:
        """Seed and start the next life."""
        life = len(self.episodes)
        core, disp = self._seeds(life)
        # reseed=False turns off NetHack's periodic reseeding with true
        # randomness, which is the whole of what makes a keystroke history a
        # complete record of a game (F7).
        self._env.seed(core, disp, False)
        self._obs, _ = self._env.reset()
        self._frozen = 0
        self._turn = self._stat("TIME")
        episode = Episode(index=life, role=self._role())
        self.episodes.append(episode)
        self._note(episode)

    def play(self, token: str) -> tuple[float, bool]:
        """Play one key. Returns (reward, alive).

        The reward is what `env.step` returns: the in-game score delta, with the
        time penalty zeroed (F3). A key the game does not accept still costs an
        action and still returns here — being swallowed by a prompt is a fact about
        NetHack's interface and not an error (F2).

        A dead game answers nothing: `act.py` stops a batch on the death that
        produced it, so reaching here means a caller ignored that.
        """
        if token not in self.tokens:
            raise GameError(f"{token!r} is not a key of this game")
        episode = self.episodes[-1]
        if not episode.alive:
            return 0.0, False

        obs, reward, terminated, truncated, info = self._env.step(
            self.spec.index[token]
        )
        self._obs = obs
        reward = float(reward)
        episode.actions += 1
        episode.reward += reward
        self.reward += reward
        self._note(episode)

        turn = self._stat("TIME")
        self._frozen = 0 if turn != self._turn else self._frozen + 1
        self._turn = turn
        if terminated or truncated:
            episode.ended = self._ending(info)
        elif self._frozen >= self._no_progress_limit:
            episode.ended = "aborted"
        return reward, episode.alive

    def restart(self) -> None:
        """End this life and begin the next, in the same game unless told otherwise."""
        self.episodes[-1].ended = self.episodes[-1].ended or "restart"
        self._begin()

    def _ending(self, info: dict) -> str:
        """How a life ended, in NetHack's own vocabulary.

        `how_done()` is the game's end-of-game type — the thing the xlogfile
        records — and it is the only place a quit is told apart from a death, and
        starving apart from being killed. NLE's `end_status` says only that
        something ended. There are sixteen of them and eleven are ways to die, so
        the name is kept as it comes and `died` does the grouping.
        """
        if int(info.get("end_status", 0)) < 0:
            return "aborted"
        return str(self._env.nethack.how_done()).split(".")[-1].lower()

    @property
    def alive(self) -> bool:
        return self.episodes[-1].alive

    @property
    def tokens(self) -> tuple[str, ...]:
        """The keys, under the names this harness gives them (F1)."""
        return self.spec.tokens

    # -- reading the game ---------------------------------------------------- #

    def _stat(self, name: str) -> int:
        from nle import nethack  # noqa: PLC0415

        return int(self._obs["blstats"][getattr(nethack, f"NLE_BL_{name}")])

    def _role(self) -> str:
        """The character this life was dealt, off the top line of the screen.

        A seed rolls a role, a race, an alignment and a gender (F5), and which one
        it rolled is most of what a life is worth in NetHack. It is read from the
        welcome message rather than from a table because that is where NetHack
        says it.
        """
        first = self.screen().splitlines()[0]
        found = re.search(r"You are a[n]? (.+?)\.", first)
        return found.group(1) if found else ""

    def _note(self, episode: Episode) -> None:
        """Read the score and the position off the status line.

        Every one of these is a running maximum rather than the latest value, and
        that is not tidiness. **The frame a life ends on has no status line**: NLE
        zeroes `blstats` for the tombstone, so the last thing a dead life reports
        about itself is 0 points on turn 0 at depth 0. Taking the maximum keeps
        what the life actually reached, which is the number the run is scored on.
        """
        episode.score = max(episode.score, self._stat("SCORE"))
        episode.turns = max(episode.turns, self._stat("TIME"))
        episode.depth = max(episode.depth, self._stat("DEPTH"))
        episode.xplevel = max(episode.xplevel, self._stat("XP"))
        episode.gold = max(episode.gold, self._stat("GOLD"))
        self._max_depth = max(self._max_depth, episode.depth)
        self._max_xplevel = max(self._max_xplevel, episode.xplevel)
        if self._stat("TIME"):  # not the zeroed frame of a life that has ended
            self._cells.add((self._stat("DNUM"), self._stat("DLEVEL"),
                             self._stat("X") * 100 + self._stat("Y")))

    # -- what the agent is given --------------------------------------------- #

    def screen(self) -> str:
        """The terminal as NetHack drew it: 24 lines of 80 columns."""
        return "\n".join(
            "".join(chr(c) for c in row).rstrip() for row in self._obs["tty_chars"]
        )

    def ansi(self) -> str:
        """The same screen with the colours the terminal would have shown.

        NetHack's colours carry information a person playing it uses — a yellow `d`
        is not a red one — so a run that gives out only the characters has given
        out less than the game draws.
        """
        out = []
        for chars, colors in zip(self._obs["tty_chars"], self._obs["tty_colors"], strict=True):
            line, last = [], None
            for char, color in zip(chars, colors, strict=True):
                color = int(color)
                if color != last:
                    line.append(f"\033[0m" if color == 0 else f"\033[{_sgr(color)}m")
                    last = color
                line.append(chr(char))
            out.append("".join(line).rstrip() + "\033[0m")
        return "\n".join(out)

    def frame(self):
        """The map, drawn with NetHack's own tileset. 336x1264, and no HUD (F9)."""
        import numpy as np  # noqa: PLC0415
        from nle import nethack  # noqa: PLC0415

        if not hasattr(self, "_tiles"):
            if not self._env.nethack.setup_tiles():
                raise GameError("the tileset would not load")
            self._tiles = np.zeros(nethack.TILE_RENDER_SHAPE, dtype=np.uint8)
        self._env.nethack.draw_frame(self._tiles)
        return self._tiles.copy()

    def write_frame(self, path: Path) -> Path:
        from PIL import Image  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.frame()).save(path, "PNG")
        return path

    def symbolic(self) -> dict:
        """The arrays NLE hands a network: the glyph grid, the stats, the inventory."""
        import numpy as np  # noqa: PLC0415

        return {
            key: np.asarray(self._obs[key])
            for key in ("glyphs", "chars", "colors", "specials", "blstats",
                        "message", "inv_glyphs", "inv_strs", "inv_letters",
                        "inv_oclasses", "tty_cursor")
        }

    def observe(self, into: Path, index: int, tag: str = "") -> dict[str, str]:
        """Write every enabled channel, and say where each one went.

        Uniform across channels on purpose: a screen is under 2KB, and inlining it
        would put several megabytes of game into the one file `PROMPT.md` tells the
        session to read back. Paths keep `logs.txt` scannable.

        `tag` names a second observation of the same action — the screen a run
        restarted into, written beside the screen the action produced rather than
        over it.

        Returns channel -> path, relative to `into`, in a fixed order so a log
        block reads the same way every time. This is the only place that knows
        which channels a run has, which is what keeps `act.py` from growing a
        branch per modality.
        """
        import numpy as np  # noqa: PLC0415

        written: dict[str, str] = {}
        for channel, folder, suffix in (
            ("tty", "screens", "txt"),
            ("ansi", "ansi", "txt"),
            ("pixels", "frames", "png"),
            ("symbolic", "symbolic", "npz"),
        ):
            if channel not in self.channels:
                continue
            name = f"{folder}/{index:06d}{tag}.{suffix}"
            (into / name).parent.mkdir(parents=True, exist_ok=True)
            if channel == "tty":
                (into / name).write_text(self.screen() + "\n")
            elif channel == "ansi":
                (into / name).write_text(self.ansi() + "\n")
            elif channel == "pixels":
                self.write_frame(into / name)
            else:
                np.savez_compressed(into / name, **self.symbolic())
            written[channel] = name
        return written

    def close(self) -> None:
        self._env.close()

    # -- what the harness keeps to itself ------------------------------------ #
    # The score is on the status line and so is not a secret. What stays out of
    # the workspace is the aggregate across lives: a session shown a running total
    # will grow the total, which in a game with a two-key suicide is not the same
    # thing as playing well (F14).

    def state(self) -> dict:
        """The raw observation. For the scripted floors and for tests."""
        return self._obs

    def episode_scores(self) -> list[int]:
        """Score per life, in order. The learning curve."""
        return [e.score for e in self.episodes]

    # Every ending NetHack knows that is not one of these is a way of dying —
    # starving, choking, drowning, being turned to slime. Eleven of the sixteen.
    NOT_DEATH = frozenset(
        {"quit", "escaped", "ascended", "panicked", "tricked", "aborted", "restart"}
    )

    @property
    def deaths(self) -> int:
        return sum(1 for e in self.episodes if e.ended and e.ended not in self.NOT_DEATH)

    @property
    def quits(self) -> int:
        return sum(1 for e in self.episodes if e.ended == "quit")

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def max_xplevel(self) -> int:
        return self._max_xplevel

    @property
    def unique_cells(self) -> int:
        return len(self._cells)

    @property
    def max_score(self) -> int:
        """NetHack has no maximum score, and saying it has one would be a fiction.

        Craftax's 226 made every number a percentage; here the unit is the score
        itself, which is what the Challenge reports and what the human dataset is
        quoted in (F10, F11).
        """
        return 0


def _sgr(color: int) -> str:
    """One of NLE's terminal colour codes as an SGR parameter.

    tty_colors carries curses attribute bits: the low three are the colour, 8 is
    bold/bright, and 128 is reverse video for the cursor's own cell.
    """
    parts = []
    if color & 0x80:
        parts.append("7")
    if color & 0x08:
        parts.append("1")
    parts.append(str(30 + (color & 0x07)))
    return ";".join(parts)
