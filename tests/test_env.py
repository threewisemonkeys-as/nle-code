"""What this harness assumes about the package it is built on.

`cc_humanrl` vendored the game and pinned a commit; `cc_craftax` pinned a version
and tested the facts it rested on. Same here: `nle==1.3.0`, and every number in
notes/nle-harness-plan.md was measured against it. These tests fail rather than
letting a silent upgrade move the ground — the seeding lockdown in particular is a
class attribute of upstream's, and the whole resume mechanism rests on being able to
get around it (F5).
"""

import numpy as np
import pytest

pytest.importorskip("nle")


def test_version_is_pinned():
    import nle

    assert nle.__version__ == "1.3.0"


def test_the_challenge_refuses_to_be_seeded():
    """The fact the whole design is arranged around.

    NetHackChallenge overwrites the seed setters on the underlying game object, so
    a run of it as registered can never be replayed and therefore never resumed.
    If upstream ever relaxes this, `nle_game` should be built on the registered
    environment instead of reconstructing it.
    """
    import gymnasium as gym

    env = gym.make("NetHackChallenge-v0")
    with pytest.raises(RuntimeError, match="doesn't allow seed changes"):
        env.unwrapped.seed(0, 0, False)
    env.close()


def test_the_parent_class_accepts_a_seed_and_the_challenge_kwargs():
    from nle import nethack
    from nle.env import tasks

    env = tasks.NetHackScore(
        actions=nethack.ACTIONS, character="@",
        allow_all_yn_questions=True, allow_all_modes=True,
    )
    assert env.seed(11, 12, False) == (11, 12, False, None)
    # get_seeds() reads the *running* game, so it needs a reset first — the seeds
    # set before one are the seeds the next game will be dealt from.
    env.reset()
    assert env.get_seeds()[:3] == (11, 12, False)
    env.close()


def test_the_action_space_is_the_keyboard():
    """121 actions over 118 distinct bytes, and `?` is not one of them (F1)."""
    from nle import nethack

    assert len(nethack.ACTIONS) == 121
    bytes_ = {int(a) for a in nethack.ACTIONS}
    assert len(bytes_) == 118
    assert ord("k") in bytes_ and ord(">") in bytes_ and 27 in bytes_
    assert ord("?") not in bytes_


def test_the_screen_is_24_by_80():
    import gymnasium as gym

    env = gym.make("NetHackChallenge-v0")
    obs, _ = env.reset()
    assert obs["tty_chars"].shape == (24, 80)
    assert obs["glyphs"].shape == (21, 79)
    assert obs["blstats"].shape == (27,)
    env.close()


def test_the_tile_renderer_draws_the_map_and_nothing_else():
    """336x1264 is 21 rows by 79 columns of 16px tiles — the map, not the screen (F9)."""
    from nle import nethack

    assert nethack.TILE_SHAPE == (16, 16, 3)
    assert nethack.TILE_RENDER_SHAPE == (21 * 16, 79 * 16, 3)


def test_a_more_prompt_swallows_every_other_key():
    """The interface fact that dominates every floor (F2).

    A policy that never sends `cr` stops playing at the first message and does not
    notice: the clock does not advance and nothing says why.
    """
    import gymnasium as gym

    env = gym.make("NetHackChallenge-v0")
    names = [f"{type(a).__name__}.{a.name}" for a in env.unwrapped.actions]
    more, east = names.index("MiscAction.MORE"), names.index("CompassDirection.E")
    obs, _ = env.reset()
    from nle import nethack

    def turn(o):
        return int(o["blstats"][nethack.NLE_BL_TIME])

    # Walk until a --More-- turns up, then keep walking and watch the clock stop.
    for _ in range(3000):
        obs, *_ = env.step(east)
        if "--More--" in bytes(obs["tty_chars"].reshape(-1)).decode("latin-1"):
            break
    stuck = turn(obs)
    for _ in range(50):
        obs, *_ = env.step(east)
    assert turn(obs) == stuck
    obs, *_ = env.step(more)
    assert "--More--" not in bytes(obs["tty_chars"].reshape(-1)).decode("latin-1")
    env.close()


def test_a_seeded_game_replays_bit_exactly():
    """(seeds, keys) is the whole record of a game, with reseeding off (F7)."""
    from nle import nethack
    from nle.env import tasks

    keys = [int(k) for k in np.random.RandomState(0).randint(0, 121, size=200)]

    def rollout(seed):
        env = tasks.NetHackScore(
            actions=nethack.ACTIONS, character="@",
            allow_all_yn_questions=True, allow_all_modes=True,
        )
        env.seed(seed, seed, False)
        obs, _ = env.reset()
        out = [obs["tty_chars"].copy()]
        for key in keys:
            obs, _, term, trunc, _ = env.step(key)
            out.append(obs["tty_chars"].copy())
            if term or trunc:
                break
        env.close()
        return out

    once, again, other = rollout(4), rollout(4), rollout(5)
    assert len(once) == len(again)
    assert all((a == b).all() for a, b in zip(once, again, strict=True))
    assert any(
        a.shape != b.shape or not (a == b).all()
        for a, b in zip(once, other, strict=False)
    )


def test_the_seed_rolls_the_character_too():
    """A seed is a role, a race, an alignment and a gender as well as a dungeon (F5)."""
    from nle import nethack
    from nle.env import tasks

    def welcome(seed):
        env = tasks.NetHackScore(
            actions=nethack.ACTIONS, character="@",
            allow_all_yn_questions=True, allow_all_modes=True,
        )
        env.seed(seed, seed, False)
        obs, _ = env.reset()
        env.close()
        return "".join(chr(c) for c in obs["tty_chars"][0])

    assert welcome(1) == welcome(1)
    assert welcome(1) != welcome(2)


def test_seeding_is_not_sticky_across_resets():
    """Why `_begin` seeds every life and not just the first (F6).

    The `.copy()` is not politeness. NLE hands out **views into buffers it reuses**,
    so `first` and `second` are the same array and comparing them without copying
    says they are equal whatever the game did (F17). Anything that keeps an
    observation past the next step has to copy it.
    """
    from nle import nethack
    from nle.env import tasks

    env = tasks.NetHackScore(
        actions=nethack.ACTIONS, character="@",
        allow_all_yn_questions=True, allow_all_modes=True,
    )
    env.seed(17, 17, False)
    first = env.reset()[0]["tty_chars"].copy()
    second = env.reset()[0]["tty_chars"].copy()
    env.close()
    assert not (first == second).all()


def test_the_observation_buffers_are_reused(): 
    """The gotcha the test above nearly hid, pinned on its own (F17)."""
    from nle import nethack
    from nle.env import tasks

    env = tasks.NetHackScore(actions=nethack.ACTIONS)
    env.seed(2, 2, False)
    first, _ = env.reset()
    kept = first["tty_chars"]
    before = kept.copy()
    for _ in range(20):
        env.step(1)
    env.close()
    assert not (kept == before).all(), "the buffer was not reused after all"


def test_the_agent_interpreter_cannot_reach_the_game():
    """The one fence that is not a pattern in the audit."""
    import subprocess
    from pathlib import Path

    python = Path(__file__).resolve().parents[1] / ".agent-venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("agent interpreter not built — tools/make_agent_venv.sh")
    ok = subprocess.run([python, "-c", "import numpy, PIL"], capture_output=True)
    assert ok.returncode == 0
    for module in ("nle", "gymnasium"):
        blocked = subprocess.run(
            [python, "-c", f"import {module}"], capture_output=True
        )
        assert blocked.returncode != 0, f"the agent's interpreter can import {module}"
