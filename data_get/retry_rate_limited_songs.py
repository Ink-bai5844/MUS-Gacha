#!/usr/bin/env python
"""
Find song JSON snapshots that contain a rate-limit message and re-crawl them.

Default marker:
    操作频繁，请稍候再试

The script delegates actual crawling to qcloud_song_store.py so the retry path
uses the same SQLite schema, JSON snapshot format, cookie handling, and
QCloudMusicApi loading logic as the main crawler.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MARKER = "操作频繁"
DEFAULT_RESOLVE_HOSTS = [
    "interface.music.163.com=117.135.207.67",
    "music.163.com=112.29.230.13",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-crawl songs whose JSON snapshots contain a rate-limit marker."
    )
    parser.add_argument("--json-dir", default="data/ink_bai_liked_json")
    parser.add_argument("--db", default="data/ink_bai_liked.sqlite3")
    parser.add_argument("--library", default="QCloudMusicApi/build/QCloudMusicApi/QCloudMusicApi.dll")
    parser.add_argument("--cookie-file", default="cookie.txt")
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--endpoints", default="default")
    parser.add_argument("--levels", default="standard,exhigh,lossless,hires")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument(
        "--resolve-host",
        action="append",
        default=None,
        help="HOST=IP mapping passed to qcloud_song_store.py. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-resolve-host",
        action="store_true",
        help="Do not pass the default Netease host mappings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later batches if one batch fails.",
    )
    return parser.parse_args()


def find_marked_song_ids(json_dir: Path, marker: str) -> list[int]:
    song_ids: list[int] = []
    seen: set[int] = set()
    for path in sorted(json_dir.glob("*.json")):
        try:
            song_id = int(path.stem)
        except ValueError:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        if marker not in text or song_id in seen:
            continue
        seen.add(song_id)
        song_ids.append(song_id)
    return song_ids


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_command(args: argparse.Namespace, song_ids: list[int]) -> list[str]:
    command = [
        sys.executable,
        "qcloud_song_store.py",
        *[str(song_id) for song_id in song_ids],
        "--db",
        args.db,
        "--json-dir",
        args.json_dir,
        "--library",
        args.library,
        "--cookie-file",
        args.cookie_file,
        "--endpoints",
        args.endpoints,
        "--levels",
        args.levels,
        "--workers",
        str(args.workers),
        "--sleep",
        str(args.sleep),
    ]

    resolve_hosts: list[str] = []
    if not args.no_default_resolve_host:
        resolve_hosts.extend(DEFAULT_RESOLVE_HOSTS)
    if args.resolve_host:
        resolve_hosts.extend(args.resolve_host)
    for item in resolve_hosts:
        command.extend(["--resolve-host", item])

    return command


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise SystemExit(f"JSON directory not found: {json_dir}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    song_ids = find_marked_song_ids(json_dir, args.marker)
    if args.max_songs is not None:
        song_ids = song_ids[: args.max_songs]

    print(f"Found {len(song_ids)} songs containing marker: {args.marker}")
    if song_ids:
        print("Song ids:", ", ".join(str(song_id) for song_id in song_ids[:50]))
        if len(song_ids) > 50:
            print(f"... and {len(song_ids) - 50} more")

    if args.dry_run or not song_ids:
        return 0

    for index, batch in enumerate(chunks(song_ids, args.batch_size), start=1):
        print(f"Retrying batch {index}: {len(batch)} songs")
        result = subprocess.run(build_command(args, batch), cwd=Path(__file__).resolve().parent)
        if result.returncode != 0:
            print(f"Batch {index} failed with exit code {result.returncode}", file=sys.stderr)
            if not args.keep_going:
                return result.returncode

    remaining = find_marked_song_ids(json_dir, args.marker)
    print(f"Remaining songs containing marker after retry: {len(remaining)}")
    if remaining:
        print("Remaining ids:", ", ".join(str(song_id) for song_id in remaining[:50]))
        if len(remaining) > 50:
            print(f"... and {len(remaining) - 50} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
