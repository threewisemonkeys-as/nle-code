"""The company a run is put in: what is measured off this disk, and what is quoted.

The page draws a run against 1,934 complete AutoAscend games and against two published
medians, and the two are not the same kind of claim. What these pin is that the
measured side really is read off the files — an xlogfile is NetHack's own format and
not ours — and that the quoted side survives the files being absent, because a page
built on a box without the dataset should lose its cloud and nothing else.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import humans  # noqa: E402

# One real line, from nld-aa-taster/nle_data/20220518-053802_x1ygqon1.
REAL = (
    "version=3.6.6\tpoints=1781\tdeathdnum=0\tdeathlev=1\tmaxlvl=1\thp=0\tmaxhp=56\t"
    "deaths=1\tdeathdate=20220518\tbirthdate=20220518\tuid=1185200751\trole=Sam\t"
    "race=Hum\tgender=Mal\talign=Law\tname=Agent\tdeath=killed by a hobbit\t"
    "while=frozen by a monster's gaze\tconduct=0xfc0\tturns=9717\tachieve=0x0\t"
    "realtime=53\tstarttime=1652877482\tendtime=1652877535\tgender0=Mal\t"
    "align0=Law\tflags=0x4\tttyrecname=nle.169020.0.ttyrec3.bz2"
)


def test_an_xlogfile_line_is_read_in_nethacks_own_format():
    got = humans.xlog(REAL)
    assert got["points"] == "1781"
    assert got["turns"] == "9717"
    assert got["role"] == "Sam"
    # `death` carries spaces and `while` is a second field about the same death:
    # splitting on anything but the tab would merge or truncate them.
    assert got["death"] == "killed by a hobbit"
    assert got["while"] == "frozen by a monster's gaze"


def test_games_are_read_from_the_directory_layout_the_dataset_ships(tmp_path):
    where = tmp_path / "nld-aa-taster" / "nle_data" / "20220518-053802_x1ygqon1"
    where.mkdir(parents=True)
    (where / "nle.169020.xlogfile").write_text(
        REAL + "\n"
        + REAL.replace("points=1781", "points=10000").replace("turns=9717", "turns=26577")
        + "\n"
        # A game that never wrote a score is not a finished game, and counting it as
        # zero would drag the median of a Zipfian population down for nothing.
        + REAL.replace("points=1781", "points=") + "\n"
    )
    got = humans.games(tmp_path)
    assert [g["score"] for g in got] == [1781, 10000]
    assert [g["turns"] for g in got] == [9717, 26577]


def test_a_box_without_the_dataset_keeps_the_published_medians(tmp_path):
    """The page should lose its cloud and nothing else."""
    got = humans.company(tmp_path / "nothing-here")
    assert got["autoascend"]["games"] == []
    assert got["autoascend"]["score"] == {}
    assert [row["name"] for row in got["published"]] == [
        "median human game", "median AutoAscend game"]
    assert any(row["human"] for row in got["published"])
    assert got["floors"], "the measured floors do not come off this disk"


def test_the_spread_is_carried_and_not_just_the_middle():
    """NetHack scores are Zipfian — the NLD-NAO mean is 127,218 against a median of
    836 — so a single number for a population would be a number about nothing."""
    dist = humans.quantiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    assert dist["n"] == 10
    assert dist["median"] == 5.5 or dist["median"] == 5
    assert dist["p10"] == 2 and dist["p90"] == 1000
    assert dist["max"] == 1000


def test_the_taster_on_this_disk_matches_the_published_set():
    """Worth asserting rather than assuming: the cloud is only worth drawing behind a
    run if it is a sample of the population the paper measured. If the dataset is not
    on this box there is nothing to check and nothing to draw."""
    got = humans.company()
    bot = got["autoascend"]["score"]
    if not bot:
        import pytest  # noqa: PLC0415

        pytest.skip(f"no dataset under {got['where']}")
    published = next(r for r in got["published"] if not r["human"])
    assert 0.7 < bot["median"] / published["score"] < 1.4, (
        f"the taster's median game scores {bot['median']} where the paper's set "
        f"medians {published['score']} — this is not a sample of that population"
    )
