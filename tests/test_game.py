"""`NleGame`: one game, its channels, its lives and its record."""

import numpy as np
import pytest

pytest.importorskip("nle")

from nle_game import CHANNELS, GameError, NleGame, key_token  # noqa: E402


@pytest.fixture
def game():
    made = NleGame(seed=3)
    yield made
    made.close()


def walk(game, keys, limit=4000):
    """Play `keys` in a loop, dismissing messages, until the life ends."""
    played = 0
    while game.alive and played < limit:
        screen = game.screen()
        if "--More--" in screen:
            key = "cr"
        elif "[yn" in screen.splitlines()[0]:
            key = "y"
        else:
            key = keys[played % len(keys)]
        game.play(key)
        played += 1
    return played


# -- the keys ---------------------------------------------------------------- #


def test_the_vocabulary_is_the_distinct_keys(game):
    assert len(game.tokens) == 118
    for key in ("k", "h", ">", "<", ".", "cr", "esc", "space", "#pray", "^d", "_"):
        assert key in game.tokens
    # `?` is not an action of this game, so it is not a token either.
    assert "?" not in game.tokens


def test_case_is_a_different_key(game):
    """`H` runs and `h` steps: a harness that folded case would silently swap them."""
    assert "h" in game.tokens and "H" in game.tokens
    assert game.spec.index["h"] != game.spec.index["H"]


def test_an_unknown_key_raises(game):
    with pytest.raises(GameError, match="not a key"):
        game.play("left")


def test_key_token_names_every_kind_of_key():
    assert key_token(ord("k"), "N") == "k"
    assert key_token(13, "MORE") == "cr"
    assert key_token(27, "ESC") == "esc"
    assert key_token(32, "SPACE") == "space"
    assert key_token(4, "KICK") == "^d"
    assert key_token(240, "PRAY") == "#pray"


# -- the channels ------------------------------------------------------------ #


def test_every_channel_writes_what_it_claims(tmp_path):
    made = NleGame(seed=3, obs=CHANNELS)
    written = made.observe(tmp_path, 7)
    assert set(written) == set(CHANNELS)
    for name in written.values():
        assert (tmp_path / name).exists()
    screen = (tmp_path / written["tty"]).read_text()
    assert len(screen.splitlines()) == 24
    assert "Dlvl:1" in screen
    frame = np.array(__import__("PIL.Image", fromlist=["Image"]).open(tmp_path / written["pixels"]))
    assert frame.shape == (336, 1264, 3)
    arrays = np.load(tmp_path / written["symbolic"])
    assert arrays["glyphs"].shape == (21, 79)
    made.close()


def test_a_channel_does_not_change_how_the_game_steps():
    """The channels are renderings of one state; adding one must not move it."""
    keys = ["l", "j", "cr", "h", "k", "s"]
    seen = []
    for obs in (("tty",), ("tty", "ansi"), ("tty", "pixels", "symbolic")):
        made = NleGame(seed=8, obs=obs)
        for key in keys * 6:
            made.play(key)
        seen.append((made.screen(), made.episodes[-1].score, made.episodes[-1].turns))
        made.close()
    assert seen[0] == seen[1] == seen[2]


def test_pixels_alone_is_refused():
    with pytest.raises(GameError, match="map and nothing else"):
        NleGame(obs=("pixels",))


def test_no_channel_at_all_is_refused():
    with pytest.raises(GameError, match="not playable"):
        NleGame(obs=())


def test_the_ansi_screen_carries_the_same_characters(game):
    import re

    plain = game.screen()
    stripped = re.sub(r"\033\[[0-9;]*m", "", game.ansi())
    assert [line.rstrip() for line in stripped.splitlines()] == plain.splitlines()


# -- the game ---------------------------------------------------------------- #


def test_a_seed_is_a_character_as_well_as_a_dungeon():
    one, two = NleGame(seed=1), NleGame(seed=2)
    again = NleGame(seed=1)
    assert one.episodes[0].role == again.episodes[0].role
    assert one.screen() == again.screen()
    assert one.screen() != two.screen()
    for made in (one, two, again):
        made.close()


def test_a_prefix_replays_bit_exactly():
    keys = ["l", "l", "j", "cr", "h", "k", "s", "."] * 12

    def rollout():
        made = NleGame(seed=13)
        screens = []
        for key in keys:
            made.play(key)
            screens.append(made.screen())
        made.close()
        return screens

    assert rollout() == rollout()


def test_the_same_game_is_dealt_again_on_restart():
    made = NleGame(seed=21)
    opening = made.screen()
    made.restart()
    assert made.screen() == opening
    assert made.episodes[0].role == made.episodes[1].role
    assert len(made.episodes) == 2
    made.close()


def test_fresh_world_deals_a_new_one():
    made = NleGame(seed=21, fresh_world=True)
    opening = made.screen()
    made.restart()
    assert made.screen() != opening
    made.close()


def test_a_life_that_ends_is_named_in_nethacks_own_vocabulary():
    """Wandering into the dungeon eventually kills you, and how it did is recorded."""
    made = NleGame(seed=7)
    played = walk(made, ["l", "j", "h", "k", "u", "b"], limit=6000)
    assert not made.alive, f"still alive after {played} keys"
    episode = made.episodes[-1]
    assert episode.ended and episode.ended not in {"", "restart"}
    assert made.deaths + made.quits >= 1
    made.close()


def test_the_stats_survive_the_frame_that_has_none():
    """NLE zeroes blstats for the tombstone, so a life's numbers are maxima."""
    made = NleGame(seed=7)
    walk(made, ["l", "j", "h", "k", "u", "b"], limit=6000)
    episode = made.episodes[-1]
    assert episode.turns > 0
    assert episode.depth >= 1
    assert all(int(v) == 0 for v in made.state()["blstats"]), "expected a zeroed frame"
    made.close()


def test_quitting_is_two_keys_and_the_game_ends_it():
    """`#quit` then `y`. It is a key of the game, so it stays playable (F14)."""
    made = NleGame(seed=4)
    made.play("#quit")
    assert "quit" in made.screen().splitlines()[0].lower()
    made.play("y")
    assert not made.alive
    assert made.episodes[-1].ended == "quit"
    assert made.quits == 1 and made.deaths == 0
    made.close()


def test_a_dead_game_answers_nothing(game):
    made = NleGame(seed=4)
    made.play("#quit")
    made.play("y")
    assert made.play("l") == (0.0, False)
    made.close()


def test_reward_is_the_score_moving():
    """Picking up gold is the cheapest thing that pays, and it pays what it says."""
    made = NleGame(seed=3)
    total = 0.0
    for _ in range(600):
        if not made.alive:
            break
        screen = made.screen()
        key = "cr" if "--More--" in screen else "l"
        reward, _ = made.play(key)
        total += reward
    assert total == pytest.approx(made.episodes[-1].score, abs=1e-6)
    made.close()


def test_the_score_task_is_a_different_game():
    """23 actions, and NLE steps past the menus itself."""
    made = NleGame(variant="score", seed=3)
    assert len(made.tokens) == 23
    assert made.spec.skips_menus
    made.close()


def test_an_unknown_variant_raises():
    with pytest.raises(GameError, match="not one of"):
        NleGame(variant="minihack")
