"""The human population, and the ways reading it off an archive can mislead.

The page had one human number on it — a median quoted from a paper — and now it has a
cloud of real games behind that. Everything here guards the distance between those two
things: a filter that decides what counts as a game at all, a sample that must never be
what a claim rests on, and a rung that can only be half-read from this kind of record.
"""

import json
import lzma
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import nao  # noqa: E402
import progression  # noqa: E402

FIELDS = "version={v}\tpoints={p}\tturns={t}\tmaxlvl={d}\tdeath={x}"


def archive(tmp_path: Path, games: list[tuple], name: str = nao.SOURCE) -> Path:
    """An xlogfile in the archive's own shape: xz, tab-separated `key=value`."""
    lines = [FIELDS.format(v=v, p=p, t=t, d=d, x=x) for v, p, t, d, x in games]
    out = tmp_path / name
    with lzma.open(out, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    return out


def test_a_character_that_never_acted_is_not_a_game():
    """The whole filtering policy, and the reason it exists: 81% of the real archive
    is someone connecting and quitting at turn 1. Counting those puts the median human
    game at one turn and zero points, which is a nonsense line to draw under a run."""
    assert nao.MIN_TURNS == 2


def test_the_filter_drops_the_non_games_and_keeps_the_rest(tmp_path):
    source = archive(tmp_path, [
        ("3.6.6", 0, 1, 1, "quit"),        # never acted
        ("3.6.6", 0, 1, 1, "quit"),        # never acted
        ("3.6.6", 500, 900, 5, "died"),
        ("3.6.6", 1500, 4000, 9, "died"),
    ])
    got = nao.distil(source)
    assert got["raw_games"] == 4
    assert got["games"] == 2, "only the games where the character did something"
    assert got["dropped"] == 2
    assert got["score"]["median"] == 1000  # the median of 500 and 1500


def test_only_the_version_this_harness_plays_is_read(tmp_path):
    """The archive is named for a server era and holds four NetHack versions. The
    filename cannot be trusted for this — the manifest is explicit that a binary gets
    upgraded under a running server — so the `version=` field is the only selector."""
    source = archive(tmp_path, [
        ("3.6.6", 900, 500, 6, "died"),
        ("3.6.7", 900, 500, 6, "died"),
        ("3.6.4", 900, 500, 6, "died"),
        ("3.6.3", 900, 500, 6, "died"),
    ])
    got = nao.distil(source)
    assert got["games"] == 1
    assert got["version"] == "3.6.6"


def test_the_quantiles_are_over_every_game_and_the_sample_is_only_drawn(tmp_path):
    """The dots are thinned because 391,809 of them would be a filled rectangle on a
    13 MB page. The numbers beside the chart must not be thinned with them, or the page
    would be quoting a sample while looking like it quotes a population."""
    source = archive(tmp_path, [("3.6.6", n, 100 + n, 3, "died") for n in range(1, 201)])
    got = nao.distil(source, sample=10)
    assert got["games"] == 200
    assert got["score"]["n"] == 200, "the quantiles saw every game"
    assert len(got["sample"]) == 10, "the scatter saw ten of them"
    assert got["score"]["median"] == 100


def test_the_sample_is_the_same_every_time_the_same_archive_is_read(tmp_path):
    """A rebuilt page should not reshuffle its dots. The watcher rebuilds every
    fifteen minutes and a cloud that moves each time reads as data changing."""
    source = archive(tmp_path, [("3.6.6", n, 100 + n, 3, "died") for n in range(1, 201)])
    assert nao.distil(source, sample=12)["sample"] == \
           nao.distil(source, sample=12)["sample"]


def test_a_human_game_can_only_be_placed_on_one_of_the_two_rungs(tmp_path):
    """The finding this file is built around. An xlogfile records `maxlvl` and has no
    experience level at all, so the progression it yields is the *dungeon* rung — at or
    below the real one, never above it. A page that drew these as exact would be
    claiming a measurement the record cannot make."""
    source = archive(tmp_path, [("3.6.6", 4000, 9000, 25, "died")])
    got = nao.distil(source)
    depth_rung, xp_rung = progression.rungs(25, 1)
    assert got["prog"]["max"] == round(depth_rung, 2)
    # The same character at experience level 14 would score far higher, and nothing in
    # the file could tell us that it was.
    assert progression.progression(25, 14) >= depth_rung
    assert xp_rung < depth_rung


def test_the_depth_curve_only_climbs_and_ends_at_everything(tmp_path):
    source = archive(tmp_path, [("3.6.6", 10, 50, d, "died")
                                for d in (1, 1, 3, 5, 5, 12)])
    curve = nao.distil(source)["depth_cdf"]
    assert curve == sorted(curve), "a cumulative share cannot go down"
    assert curve[-1] == 100.0
    assert curve[0] == 0.0, "nothing is shallower than dungeon level 1"
    # Two of six games never left level 1.
    assert curve[1] == pytest.approx(100 * 2 / 6, abs=0.01)


def test_the_share_of_games_a_run_is_past_is_counted_on_levels_not_on_rungs(tmp_path):
    """The bug this shape avoids, which bit once while building the page.

    The rung for dungeon level 25 is 46.63763…, and every panel prints it as 46.64.
    Counting the games a run is past by comparing rungs against the *printed* number
    silently drops every game that ended on exactly that level — 589 of them in the
    real archive, a tenth of the answer. The curve is indexed by integer level, so the
    comparison never touches a float.
    """
    source = archive(tmp_path, [("3.6.6", 10, 50, d, "died")
                                for d in [1] * 3 + [25] * 2 + [30]])
    curve = nao.distil(source)["depth_cdf"]
    past24 = 100 - curve[24]
    assert past24 == pytest.approx(100 * 3 / 6, abs=0.01), (
        "the two games that ended on level 25 have to be counted as having reached it"
    )
    # The trap itself: the rung and its printed form are not the same number.
    assert progression.rungs(25, 1)[0] != round(progression.rungs(25, 1)[0], 2)
    assert progression.rungs(25, 1)[0] < 46.64


def test_a_won_game_is_flagged_from_the_record_because_the_rung_cannot_say_it(tmp_path):
    """People do win NetHack — 3,279 of the real games — and the metric cannot report
    it: the ladder's ascension entries are message strings the progress system never
    looks up, so the ceiling is 80.68%. `death=ascended` is the only evidence there is,
    and it is a fact in the record rather than an inference from a level."""
    source = archive(tmp_path, [
        ("3.6.6", 2_000_000, 60000, 49, "ascended"),
        ("3.6.6", 900, 5000, 49, "died in The Dungeons of Doom"),
    ])
    got = nao.distil(source)
    assert got["ascended"] == 1
    assert got["ascended_pct"] == 50.0
    won = [row for row in got["sample"] if row[3]]
    assert len(won) == 1
    # Both games reached the same dungeon level, so the rung cannot tell them apart.
    assert len({row[2] for row in got["sample"]}) == 1
    assert progression.rungs(49, 1)[0] < progression.CEILING + 0.01


def test_an_ascension_scores_below_the_ceiling_on_the_rung_it_can_be_given(tmp_path):
    """And the sharpest version of it: you finish NetHack from the Sanctum, not from
    the bottom, so a winner can have a *shallower* maxlvl than someone who died deep.
    A depth-only reading then ranks the corpse above the victory."""
    source = archive(tmp_path, [
        ("3.6.6", 3_000_000, 70000, 45, "ascended"),
        ("3.6.6", 800, 9000, 50, "died in The Sanctum"),
    ])
    rows = {row[2]: row for row in nao.distil(source)["sample"]}
    winner, corpse = rows[45], rows[50]
    assert winner[3] == 1 and corpse[3] == 0
    assert progression.rungs(corpse[2], 1)[0] > progression.rungs(winner[2], 1)[0], (
        "the metric really does rank the deeper death above the ascension — which is "
        "why the page draws a won game at 100 and not at its rung"
    )


def test_a_cache_in_the_old_shape_is_rebuilt_and_not_served(tmp_path):
    """The archive is immutable, so the size key alone never notices that the *cache*
    is stale. Without the schema key a page built today would be handed rows of the
    wrong width by a cache written last week, and read the ascension flag off the end.
    """
    archive(tmp_path, [("3.6.6", 100, 500, 4, "died")] * 2)
    fresh = nao.load(tmp_path)
    assert len(fresh["sample"][0]) == 4

    stale = json.loads((tmp_path / nao.CACHE).read_text())
    stale["schema"] = nao.SCHEMA - 1
    stale["sample"] = [row[:3] for row in stale["sample"]]
    (tmp_path / nao.CACHE).write_text(json.dumps(stale))

    again = nao.load(tmp_path)
    assert again["schema"] == nao.SCHEMA
    assert len(again["sample"][0]) == 4, "the old-shaped cache was served anyway"


def test_the_cache_is_rebuilt_when_the_archive_is_a_different_file(tmp_path):
    """Keyed on size, because the archive is published as an immutable snapshot — a
    different size is a different snapshot and not an edit. Without this the page would
    keep serving last month's population after a re-fetch."""
    archive(tmp_path, [("3.6.6", 100, 500, 4, "died")] * 3)
    first = nao.load(tmp_path)
    assert first["games"] == 3
    assert (tmp_path / nao.CACHE).exists()

    archive(tmp_path, [("3.6.6", 100, 500, 4, "died")] * 9)
    assert nao.load(tmp_path)["games"] == 9


def test_no_archive_is_an_empty_answer_and_not_a_crash(tmp_path):
    """The state every box that has not fetched the archive is in, including the one
    this page was built on yesterday. The published median has to survive it."""
    assert nao.load(tmp_path) == {}
    ok, said = nao.check(tmp_path)
    assert not ok and "not here" in said


def test_a_corrupt_cache_is_rebuilt_rather_than_fatal(tmp_path):
    archive(tmp_path, [("3.6.6", 100, 500, 4, "died")] * 2)
    (tmp_path / nao.CACHE).write_text("{ this is not json")
    assert nao.load(tmp_path)["games"] == 2


def test_the_archive_on_this_disk_is_the_one_the_manifest_describes():
    """Provenance, cheaply. The numbers on the page are only as good as the file they
    came from, and the failure that actually happens to a 29 MB download is that it is
    truncated or is the wrong era's file — both of which the published size catches.

    The manifest's md5 is of the *uncompressed* content, so verifying it means
    expanding 1.16 GB and costs ~12 s. That is `tools/nao.py --check`, run when the
    archive is fetched, rather than on every edit of every file in this repository.
    `NLE_CHECK_MD5=1` runs it here too.
    """
    source = nao.DATA / nao.SOURCE
    if not source.exists():
        pytest.skip(f"{nao.SOURCE} is not on this box")
    assert source.stat().st_size == nao.SOURCE_BYTES, (
        "the archive is not the size the manifest publishes — a partial download, or "
        "a different server era's file"
    )
    if not os.environ.get("NLE_CHECK_MD5"):
        pytest.skip("set NLE_CHECK_MD5=1 to verify the manifest's md5 (~12 s)")
    ok, said = nao.check()
    assert ok, said


def test_the_published_median_is_not_quietly_replaced_by_the_measured_one():
    """Two populations, not two readings of one: NLD-NAO is 1,511,228 games from around
    2020 across older versions, and this is 3.6.6 as the server has it now. Both belong
    on the page, and the quoted one must not drift into being described as measured."""
    import humans

    quoted = [row for row in humans.PUBLISHED if row.get("human")]
    assert len(quoted) == 1
    assert quoted[0]["score"] == 836 and quoted[0]["games"] == 1511228
