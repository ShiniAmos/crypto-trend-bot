"""Back up the three sqlite books, verify each copy, prune old ones.

    venv/bin/python deploy/backup_books.py            # to data/backups
    venv/bin/python deploy/backup_books.py --keep 30

WHY THIS EXISTS. Until 5 Nov 2026 the entire forward-test record - every fill, every equity point,
every decision these books have ever made - existed only as three files on one 1 GB VM, with no
copy anywhere. The research is reproducible from the code; the forward test is not reproducible
from anything. Losing it loses the only evidence about how this strategy behaves on live data.

It uses sqlite3's online backup API rather than `cp`. The bot writes to these files continuously,
and copying a live sqlite database with `cp` can capture a page mid-write: the copy opens, reports
no error, and is silently short of the last transaction or corrupt in a way only a later read
finds. `.backup` takes a consistent snapshot of a database being written to, which is the whole
reason the API exists.

Every copy is then opened and `PRAGMA integrity_check`ed, and the fill count is compared with the
source. A backup nobody has read is a guess, and this is the one file that has to be right.
"""

import argparse
import datetime
import pathlib
import sqlite3
import sys

BOOKS = ("paper", "paper100", "demo")


def backup_one(src: pathlib.Path, dest: pathlib.Path) -> dict:
    """Snapshot `src` to `dest`, then verify the copy. Raises if the copy is not sound."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()

    # verify: the copy has to be readable, intact, and hold the same number of fills
    want = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    got = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        integrity = got.execute("pragma integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"{dest.name}: integrity_check said {integrity!r}")
        n_src = want.execute("select count(*) from fills").fetchone()[0]
        n_dst = got.execute("select count(*) from fills").fetchone()[0]
        if n_dst < n_src:
            raise RuntimeError(f"{dest.name}: {n_dst} fills copied but the source has {n_src}")
        equity = got.execute("select equity, time from equity order by rowid desc limit 1").fetchone()
        return {"fills": n_dst, "equity": equity[0] if equity else None,
                "as_of": equity[1] if equity else None, "bytes": dest.stat().st_size}
    finally:
        want.close()
        got.close()


def prune(root: pathlib.Path, book: str, keep: int) -> int:
    """Keep the newest `keep` snapshots of one book. Returns how many were removed."""
    old = sorted(root.glob(f"state_{book}.*.sqlite"))[:-keep] if keep > 0 else []
    for p in old:
        p.unlink()
    return len(old)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/opt/bot/data", help="where the live books are")
    ap.add_argument("--out", default="", help="where to write snapshots (default <data>/backups)")
    ap.add_argument("--keep", type=int, default=90, help="snapshots to keep per book")
    args = ap.parse_args(argv)

    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out) if args.out else data / "backups"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    failures = []
    for book in BOOKS:
        src = data / f"state_{book}.sqlite"
        if not src.exists():
            print(f"{now} backup {book}: SKIPPED, {src} does not exist")
            continue
        dest = out / f"state_{book}.{stamp}.sqlite"
        try:
            info = backup_one(src, dest)
        except Exception as e:                                    # noqa: BLE001
            failures.append(f"{book}: {e}")
            print(f"{now} backup {book}: FAILED - {e}")
            dest.unlink(missing_ok=True)
            continue
        removed = prune(out, book, args.keep)
        eq = f"{info['equity']:,.2f}" if info["equity"] is not None else "n/a"
        print(f"{now} backup {book}: ok, {info['fills']} fills, equity {eq} "
              f"at {info['as_of']}, {info['bytes']:,} bytes, pruned {removed}")

    if failures:
        # Exit non-zero so cron mails it and a monitor can see it. A backup that fails quietly is
        # worse than no backup, because it is mistaken for one.
        print(f"{now} BACKUP FAILED for: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
