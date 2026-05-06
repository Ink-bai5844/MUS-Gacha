#!/usr/bin/env python
"""
Export per-song QCloudMusicApi JSON snapshots into one flat CSV table.

The default input and output paths match the current project layout:

    python export_liked_json_to_csv.py

Songs marked by the API as "no copyright available" are skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/source/ink_bai_liked_json")
DEFAULT_OUTPUT = Path("data/source/ink_bai_liked_songs.csv")
DEFAULT_LYRICS_DIR = Path("data/source/lyrics")
COPYRIGHT_UNAVAILABLE_MARKER = "\u6682\u65e0\u7248\u6743"
URL_LEVELS = ("standard", "exhigh", "lossless", "hires")
AUDIO_QUALITIES = ("l", "m", "h", "sq", "hr")

BASE_COLUMNS = [
    "source_file",
    "song_id",
    "name",
    "aliases",
    "translations",
    "artist_names",
    "artist_ids",
    "album_id",
    "album_name",
    "album_pic_url",
    "duration_ms",
    "duration_seconds",
    "duration_text",
    "publish_time_ms",
    "publish_date",
    "disc",
    "track_no",
    "popularity",
    "mv_id",
    "fee",
    "copyright",
    "status",
    "version",
    "collected_at",
    "check_success",
    "check_message",
    "playable",
    "max_br_level",
    "max_bitrate",
    "play_max_br_level",
    "play_max_bitrate",
    "download_max_br_level",
    "download_max_bitrate",
    "free_level",
    "paid",
    "privilege_fee",
    "privilege_status",
    "comment_total",
    "hot_comment_count",
    "first_hot_comment_user",
    "first_hot_comment_likes",
    "first_hot_comment",
    "first_comment_user",
    "first_comment_likes",
    "first_comment",
    "has_lyric",
    "lyric_line_count",
    "lyric_excerpt",
    "has_translation",
    "translation_excerpt",
    "has_romaji",
    "romaji_excerpt",
    "similar_song_ids",
    "similar_song_names",
    "similar_artist_names",
    "wiki_summary_excerpt",
]

QUALITY_COLUMNS = [
    f"{quality}_{field}"
    for quality in AUDIO_QUALITIES
    for field in ("br", "size", "sr")
]

URL_COLUMNS = [
    f"{level}_{field}"
    for level in URL_LEVELS
    for field in ("url", "br", "size", "type", "code", "md5")
]

LYRIC_COLUMNS = ["lyric_text", "translation_text", "romaji_text"]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def joined(values: list[Any], sep: str = " | ") -> str:
    texts = [as_text(value).strip() for value in values]
    return sep.join(text for text in texts if text)


def excerpt(value: Any, limit: int = 300) -> str:
    text = as_text(value).replace("\r", " ").replace("\n", " / ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def duration_text(duration_ms: Any) -> str:
    if not isinstance(duration_ms, (int, float)):
        return ""
    total_seconds = int(round(duration_ms / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def timestamp_ms_to_date(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value / 1000, timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def endpoint_rows(snapshot: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
    rows = snapshot.get("api_results", {}).get(endpoint, [])
    return rows if isinstance(rows, list) else []


def first_response(snapshot: dict[str, Any], endpoint: str) -> dict[str, Any]:
    for row in endpoint_rows(snapshot, endpoint):
        if row.get("error"):
            continue
        response = row.get("response")
        if isinstance(response, dict):
            return response
    return {}


def detail_song(snapshot: dict[str, Any]) -> dict[str, Any]:
    raw_detail = snapshot.get("raw_detail")
    if not isinstance(raw_detail, dict):
        raw_detail = first_response(snapshot, "song_detail")

    songs = raw_detail.get("songs") if isinstance(raw_detail, dict) else None
    if not isinstance(songs, list):
        return {}

    target_id = snapshot.get("song_id")
    for song in songs:
        if isinstance(song, dict) and song.get("id") == target_id:
            return song
    return songs[0] if songs and isinstance(songs[0], dict) else {}


def privilege_for_song(snapshot: dict[str, Any], song: dict[str, Any]) -> dict[str, Any]:
    raw_detail = snapshot.get("raw_detail")
    if isinstance(raw_detail, dict):
        privileges = raw_detail.get("privileges")
        if isinstance(privileges, list):
            target_id = snapshot.get("song_id") or song.get("id")
            for privilege in privileges:
                if isinstance(privilege, dict) and privilege.get("id") == target_id:
                    return privilege
            if privileges and isinstance(privileges[0], dict):
                return privileges[0]

    privilege = song.get("privilege")
    return privilege if isinstance(privilege, dict) else {}


def artist_list(snapshot: dict[str, Any], song: dict[str, Any]) -> list[dict[str, Any]]:
    artists = snapshot.get("artists")
    if isinstance(artists, list) and artists:
        return [artist for artist in artists if isinstance(artist, dict)]

    artists = song.get("ar")
    if isinstance(artists, list):
        return [artist for artist in artists if isinstance(artist, dict)]
    return []


def album_info(snapshot: dict[str, Any], song: dict[str, Any]) -> dict[str, Any]:
    album = song.get("al")
    if not isinstance(album, dict):
        album = {}
    return {
        "id": snapshot.get("album_id") or album.get("id"),
        "name": snapshot.get("album_name") or album.get("name"),
        "picUrl": album.get("picUrl"),
    }


def audio_value(song: dict[str, Any], quality: str, field: str) -> Any:
    item = song.get(quality)
    if isinstance(item, dict):
        return item.get(field)
    return ""


def url_items_by_level(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in endpoint_rows(snapshot, "song_url_v1"):
        response = row.get("response")
        if not isinstance(response, dict):
            continue
        data = response.get("data")
        if not isinstance(data, list) or not data:
            continue
        item = data[0]
        if not isinstance(item, dict):
            continue
        level = row.get("params", {}).get("level") or item.get("level")
        if level in URL_LEVELS:
            result[str(level)] = item
    return result


def normalize_api_lyric_text(lyric_text: str) -> str:
    lyric_text = lyric_text.strip()
    if not lyric_text:
        return ""

    normalized_lines: list[str] = []
    saw_json_line = False
    for line in lyric_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            normalized_lines.append(line)
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            saw_json_line = True
            continue
        parts = item.get("c") if isinstance(item, dict) else None
        if not isinstance(parts, list):
            saw_json_line = True
            continue
        text = "".join(
            as_text(part.get("tx"))
            for part in parts
            if isinstance(part, dict) and part.get("tx") is not None
        ).strip()
        if text:
            normalized_lines.append(text)
        saw_json_line = True

    if saw_json_line:
        return "\n".join(normalized_lines)
    return lyric_text


def lyric_parts(snapshot: dict[str, Any]) -> dict[str, str]:
    result = {
        "lyric_text": "",
        "translation_text": "",
        "romaji_text": "",
    }

    responses: list[dict[str, Any]] = []
    for endpoint in ("lyric_new", "lyric"):
        response = first_response(snapshot, endpoint)
        if response:
            responses.append(response)

    for source_key, target_key in (
        ("lrc", "lyric_text"),
        ("tlyric", "translation_text"),
        ("romalrc", "romaji_text"),
    ):
        for response in responses:
            source = response.get(source_key)
            if not isinstance(source, dict):
                continue
            lyric_text = normalize_api_lyric_text(as_text(source.get("lyric")))
            if lyric_text:
                result[target_key] = lyric_text
                break
    return result


def lyric_text_for_file(lyric_text: str) -> str:
    lines: list[str] = []
    for line in lyric_text.splitlines():
        if line.lstrip().startswith("{"):
            continue
        clean = re.sub(r"\[[^\]]*\]", "", line).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def write_lyric_file(lyrics_dir: Path, song_id: Any, lyric_text: str) -> bool:
    clean = lyric_text_for_file(lyric_text)
    if not clean:
        return False

    lyrics_dir.mkdir(parents=True, exist_ok=True)
    path = lyrics_dir / f"{song_id}.txt"
    path.write_text(clean + "\n", encoding="utf-8")
    return True


def comment_info(snapshot: dict[str, Any]) -> dict[str, Any]:
    response = first_response(snapshot, "comment_music")
    comments = response.get("comments") if isinstance(response.get("comments"), list) else []
    hot_comments = (
        response.get("hotComments") if isinstance(response.get("hotComments"), list) else []
    )

    def first_comment(rows: list[Any]) -> dict[str, Any]:
        if not rows or not isinstance(rows[0], dict):
            return {}
        comment = rows[0]
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        return {
            "user": user.get("nickname"),
            "likes": comment.get("likedCount"),
            "content": comment.get("content"),
        }

    hot = first_comment(hot_comments)
    regular = first_comment(comments)
    return {
        "comment_total": response.get("total"),
        "hot_comment_count": len(hot_comments),
        "first_hot_comment_user": hot.get("user"),
        "first_hot_comment_likes": hot.get("likes"),
        "first_hot_comment": excerpt(hot.get("content"), 500),
        "first_comment_user": regular.get("user"),
        "first_comment_likes": regular.get("likes"),
        "first_comment": excerpt(regular.get("content"), 500),
    }


def similar_info(snapshot: dict[str, Any]) -> dict[str, str]:
    response = first_response(snapshot, "simi_song")
    songs = response.get("songs") if isinstance(response.get("songs"), list) else []
    ids: list[Any] = []
    names: list[Any] = []
    artists: list[str] = []
    for song in songs:
        if not isinstance(song, dict):
            continue
        ids.append(song.get("id"))
        names.append(song.get("name"))
        song_artists = song.get("artists") or song.get("ar")
        if isinstance(song_artists, list):
            artists.append(
                joined(
                    [
                        artist.get("name")
                        for artist in song_artists
                        if isinstance(artist, dict)
                    ],
                    sep="/",
                )
            )
    return {
        "similar_song_ids": joined(ids),
        "similar_song_names": joined(names),
        "similar_artist_names": joined(artists),
    }


def collect_interesting_texts(value: Any) -> list[str]:
    texts: list[str] = []
    interesting_keys = {"summary", "content", "text", "desc", "description", "title", "name"}

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif key in interesting_keys and item:
            text = as_text(item).strip()
            if text:
                texts.append(text)

    visit(value)
    return texts


def wiki_excerpt(snapshot: dict[str, Any]) -> str:
    response = first_response(snapshot, "song_wiki_summary")
    data = response.get("data")
    return excerpt(joined(collect_interesting_texts(data)), 800)


def build_row(path: Path, snapshot: dict[str, Any], include_lyrics: bool) -> dict[str, Any]:
    song = detail_song(snapshot)
    privilege = privilege_for_song(snapshot, song)
    artists = artist_list(snapshot, song)
    album = album_info(snapshot, song)
    check = first_response(snapshot, "check_music")
    urls = url_items_by_level(snapshot)
    lyrics = lyric_parts(snapshot)

    duration_ms = snapshot.get("duration_ms") or song.get("dt")
    publish_time = song.get("publishTime")
    aliases = song.get("alia") if isinstance(song.get("alia"), list) else []
    translations = song.get("tns") if isinstance(song.get("tns"), list) else []
    lyric_text = lyrics["lyric_text"]
    translation_text = lyrics["translation_text"]
    romaji_text = lyrics["romaji_text"]

    row: dict[str, Any] = {
        "source_file": path.name,
        "song_id": snapshot.get("song_id") or song.get("id") or path.stem,
        "name": snapshot.get("name") or song.get("name"),
        "aliases": joined(aliases),
        "translations": joined(translations),
        "artist_names": joined([artist.get("name") for artist in artists]),
        "artist_ids": joined([artist.get("id") for artist in artists]),
        "album_id": album["id"],
        "album_name": album["name"],
        "album_pic_url": album["picUrl"],
        "duration_ms": duration_ms,
        "duration_seconds": round(duration_ms / 1000, 3) if isinstance(duration_ms, (int, float)) else "",
        "duration_text": duration_text(duration_ms),
        "publish_time_ms": publish_time,
        "publish_date": timestamp_ms_to_date(publish_time),
        "disc": song.get("cd"),
        "track_no": song.get("no"),
        "popularity": song.get("pop"),
        "mv_id": song.get("mv"),
        "fee": song.get("fee"),
        "copyright": song.get("copyright"),
        "status": song.get("st"),
        "version": song.get("version") or song.get("v"),
        "collected_at": snapshot.get("collected_at"),
        "check_success": check.get("success"),
        "check_message": check.get("message") or check.get("msg"),
        "playable": bool(privilege.get("pl")) if privilege else "",
        "max_br_level": privilege.get("maxBrLevel"),
        "max_bitrate": privilege.get("maxbr"),
        "play_max_br_level": privilege.get("playMaxBrLevel"),
        "play_max_bitrate": privilege.get("playMaxbr"),
        "download_max_br_level": privilege.get("downloadMaxBrLevel"),
        "download_max_bitrate": privilege.get("downloadMaxbr"),
        "free_level": privilege.get("flLevel"),
        "paid": privilege.get("payed"),
        "privilege_fee": privilege.get("fee"),
        "privilege_status": privilege.get("st"),
        "has_lyric": bool(lyric_text),
        "lyric_line_count": len(lyric_text.splitlines()) if lyric_text else 0,
        "lyric_excerpt": excerpt(lyric_text, 500),
        "has_translation": bool(translation_text),
        "translation_excerpt": excerpt(translation_text, 500),
        "has_romaji": bool(romaji_text),
        "romaji_excerpt": excerpt(romaji_text, 500),
        "wiki_summary_excerpt": wiki_excerpt(snapshot),
    }

    for quality in AUDIO_QUALITIES:
        for field in ("br", "size", "sr"):
            row[f"{quality}_{field}"] = audio_value(song, quality, field)

    for level in URL_LEVELS:
        item = urls.get(level, {})
        for field in ("url", "br", "size", "type", "code", "md5"):
            row[f"{level}_{field}"] = item.get(field) if isinstance(item, dict) else ""

    row.update(comment_info(snapshot))
    row.update(similar_info(snapshot))

    if include_lyrics:
        row.update(lyrics)

    return row


def load_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_copyright_unavailable_marker(value: Any, key: str = "") -> bool:
    if isinstance(value, dict):
        return any(
            has_copyright_unavailable_marker(child, str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, list):
        return any(has_copyright_unavailable_marker(child, key) for child in value)
    if key not in {"message", "msg", "error"}:
        return False
    return COPYRIGHT_UNAVAILABLE_MARKER in as_text(value)


def is_no_copyright_snapshot(snapshot: dict[str, Any]) -> bool:
    api_results = snapshot.get("api_results")
    if not isinstance(api_results, dict):
        return False

    for rows in api_results.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and has_copyright_unavailable_marker(row):
                return True
    return False


def export_csv(
    input_dir: Path,
    output_path: Path,
    include_lyrics: bool,
    skip_no_copyright: bool,
    extract_lyrics_files: bool,
    lyrics_dir: Path,
) -> tuple[int, int, int, list[str]]:
    json_paths = sorted(input_dir.glob("*.json"), key=lambda item: item.name)
    columns = BASE_COLUMNS + QUALITY_COLUMNS + URL_COLUMNS
    if include_lyrics:
        columns += LYRIC_COLUMNS

    errors: list[str] = []
    skipped_no_copyright = 0
    lyric_files_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        written = 0
        for path in json_paths:
            try:
                snapshot = load_snapshot(path)
                no_copyright = is_no_copyright_snapshot(snapshot)
                if skip_no_copyright and no_copyright:
                    skipped_no_copyright += 1
                    continue
                row = build_row(path, snapshot, include_lyrics)
                if extract_lyrics_files and not no_copyright:
                    lyric_text = lyric_parts(snapshot)["lyric_text"]
                    if write_lyric_file(lyrics_dir, row["song_id"], lyric_text):
                        lyric_files_written += 1
            except Exception as exc:  # Keep one bad snapshot from stopping the export.
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            writer.writerow({column: as_text(row.get(column)) for column in columns})
            written += 1
    return written, skipped_no_copyright, lyric_files_written, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one CSV table from QCloudMusicApi per-song JSON snapshots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing per-song JSON files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--include-lyrics",
        action="store_true",
        help="Include full lyric/translation/romaji columns in addition to excerpts.",
    )
    parser.add_argument(
        "--keep-no-copyright",
        action="store_true",
        help='Keep songs whose API status says "no copyright available".',
    )
    parser.add_argument(
        "--lyrics-dir",
        type=Path,
        default=DEFAULT_LYRICS_DIR,
        help=f"Directory for extracted lyric txt files. Default: {DEFAULT_LYRICS_DIR}",
    )
    parser.add_argument(
        "--no-lyrics-files",
        action="store_true",
        help="Do not write separate lyric txt files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    written, skipped_no_copyright, lyric_files_written, errors = export_csv(
        args.input_dir,
        args.output,
        args.include_lyrics,
        skip_no_copyright=not args.keep_no_copyright,
        extract_lyrics_files=not args.no_lyrics_files,
        lyrics_dir=args.lyrics_dir,
    )
    print(f"Wrote {written} rows to {args.output}")
    if skipped_no_copyright:
        print(f"Skipped {skipped_no_copyright} no-copyright row(s)")
    if not args.no_lyrics_files:
        print(f"Wrote {lyric_files_written} lyric file(s) to {args.lyrics_dir}")
    if errors:
        print(f"Skipped {len(errors)} file(s):", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
