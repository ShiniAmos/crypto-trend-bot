"""The book backups must be consistent, verified, and must fail loudly.

WHY. Until 5 Nov 2026 the whole forward-test record existed only as three sqlite files on one
1 GB VM with no copy anywhere. The research is reproducible from the code; the forward test is not
reproducible from anything, so losing those files loses the only evidence about how this strategy
behaves on live data.

Two properties are worth tests rather than trust:

  1. the snapshot uses sqlite3's online backup API, not a file copy. The bot writes to these
     databases continuously and `cp` on a live sqlite can capture a page mid-write - the copy
     opens, reports no error, and is silently short of the last transaction.
  2. a failure is loud. A backup that fails quietly is worse than no backup at all, because it is
     mistaken for one.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import backup_books                                              # noqa: E402


def _book(path, fills=3):
    c = sqlite3.connect(path)
    c.execute("create table fills (id integer primary key, symbol text)")
    c.execute("create table equity (time text, equity real, cash real)")
    c.executemany("insert into fills (symbol) values (?)", [("BTCUSDT",)] * fills)
    c.execute("insert into equity values ('2026-11-06 08:00:00+00:00', 4603.98, 4603.98)")
    c.commit()
    c.close()
    return path


def test_a_snapshot_is_complete_and_verified(tmp_path):
    src = _book(tmp_path / "state_demo.sqlite", fills=212)
    info = backup_books.backup_one(src, tmp_path / "out" / "snap.sqlite")
    assert info["fills"] == 212
    assert info["equity"] == pytest.approx(4603.98)
    assert sqlite3.connect(tmp_path / "out" / "snap.sqlite").execute(
        "select count(*) from fills").fetchone()[0] == 212


def test_it_keeps_writing_while_the_book_is_open_for_writes(tmp_path):
    """The live case: the bot holds the database open and is mid-transaction."""
    src = _book(tmp_path / "state_paper.sqlite", fills=5)
    writer = sqlite3.connect(src)
    writer.execute("insert into fills (symbol) values ('ETHUSDT')")      # uncommitted
    try:
        info = backup_books.backup_one(src, tmp_path / "snap.sqlite")
        # an uncommitted row must NOT appear: the snapshot is of committed state
        assert info["fills"] == 5, "the snapshot captured an uncommitted write"
    finally:
        writer.rollback()
        writer.close()


def test_a_corrupt_source_fails_loudly_and_leaves_no_snapshot(tmp_path, capsys):
    """A file that is not a database must produce a non-zero exit, not a silent success."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "state_paper.sqlite").write_bytes(b"this is not a database" * 50)
    _book(data / "state_paper100.sqlite")
    _book(data / "state_demo.sqlite")

    rc = backup_books.main(["--data", str(data), "--out", str(tmp_path / "out"), "--keep", "5"])
    out = capsys.readouterr()
    assert rc == 1, "a corrupt book was reported as a successful backup"
    assert "FAILED" in out.out + out.err
    assert not list((tmp_path / "out").glob("state_paper.*.sqlite")), (
        "a broken snapshot was left behind, where it would later be mistaken for a good one")
    # the other two books must still have been backed up
    assert len(list((tmp_path / "out").glob("state_demo.*.sqlite"))) == 1
    assert len(list((tmp_path / "out").glob("state_paper100.*.sqlite"))) == 1


def test_a_missing_book_is_skipped_not_failed(tmp_path):
    """Not every deployment runs all three books."""
    data = tmp_path / "data"
    data.mkdir()
    _book(data / "state_paper.sqlite")
    assert backup_books.main(["--data", str(data), "--out", str(tmp_path / "out")]) == 0


def test_pruning_keeps_the_newest_and_leaves_other_books_alone(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for d in range(1, 6):
        for book in ("paper", "paper100"):
            _book(out / f"state_{book}.202611{d:02d}T000000Z.sqlite")

    assert backup_books.prune(out, "paper", 2) == 3
    kept = sorted(p.name for p in out.glob("state_paper.*.sqlite"))
    assert kept == ["state_paper.20261104T000000Z.sqlite", "state_paper.20261105T000000Z.sqlite"]
    assert len(list(out.glob("state_paper100.*.sqlite"))) == 5, (
        "pruning one book deleted another's snapshots - 'state_paper.' must not match 'paper100'")
    assert backup_books.prune(out, "paper100", 0) == 0, "keep=0 must delete nothing"
