#!/usr/bin/env python
"""
Collect and store complete song-oriented metadata with QCloudMusicApi.

This script uses the C ABI exposed by the QCloudMusicApi shared library
(`invoke`, `set_cookie`, `set_proxy`, etc.). It stores raw endpoint responses
in SQLite and can also write a per-song JSON snapshot for easy inspection.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import select
import socket
import socketserver
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


DEFAULT_LEVELS = ("standard", "exhigh", "lossless", "hires")
DEFAULT_ENDPOINTS = (
    "check_music",
    "lyric_new",
    "lyric",
    "song_music_detail",
    "song_wiki_summary",
    "song_dynamic_cover",
    "song_chorus",
    "comment_music",
    "simi_song",
    "simi_playlist",
)


class QCloudMusicApiError(RuntimeError):
    """Raised when the QCloudMusicApi shared library cannot be used."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def system_library_suffix() -> str:
    suffix = {
        "Windows": "dll",
        "Linux": "so",
        "Darwin": "dylib",
    }.get(platform.system())
    if not suffix:
        raise QCloudMusicApiError(f"Unsupported platform: {platform.system()}")
    return suffix


def candidate_library_paths(repo_root: Path) -> list[Path]:
    name = f"QCloudMusicApi.{system_library_suffix()}"
    return [
        repo_root / name,
        repo_root / "QCloudMusicApi" / name,
        repo_root / "QCloudMusicApi" / "build" / "QCloudMusicApi" / name,
        repo_root / "QCloudMusicApi" / "build" / "bin" / name,
        repo_root / "QCloudMusicApi" / "build" / "Release" / name,
        repo_root / "QCloudMusicApi" / "build" / "Debug" / name,
        repo_root / "QCloudMusicApi" / "build" / "QCloudMusicApi" / "Release" / name,
        repo_root / "QCloudMusicApi" / "build" / "QCloudMusicApi" / "Debug" / name,
    ]


def resolve_library_path(repo_root: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise QCloudMusicApiError(f"QCloudMusicApi library not found: {path}")
        return path

    for path in candidate_library_paths(repo_root):
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"  - {path}" for path in candidate_library_paths(repo_root))
    raise QCloudMusicApiError(
        "QCloudMusicApi shared library was not found.\n"
        "Build it first or pass --library explicitly.\n"
        "Typical build command:\n"
        "  cmake -S QCloudMusicApi -B QCloudMusicApi/build "
        "-DQCLOUDMUSICAPI_BUILD_TEST=OFF -DQCLOUDMUSICAPI_BUILD_SHARED=ON\n"
        "  cmake --build QCloudMusicApi/build --config Release\n"
        "Searched:\n"
        f"{searched}"
    )


def windows_runtime_dll_dirs(library_path: Path) -> list[str]:
    if platform.system() != "Windows":
        return []

    dirs: list[Path] = [library_path.parent]
    dirs.extend(Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)

    for qt_root in (Path("D:/Qt"), Path("C:/Qt")):
        if not qt_root.exists():
            continue
        dirs.extend(qt_root.glob("*/mingw_64/bin"))
        dirs.extend(qt_root.glob("Tools/mingw*_64/bin"))
        dirs.extend(qt_root.glob("Tools/CMake_64/bin"))
        dirs.extend(qt_root.glob("Tools/Ninja"))

    result: list[str] = []
    seen: set[str] = set()
    for dll_dir in dirs:
        try:
            normalized = os.path.normcase(str(dll_dir.resolve()))
        except OSError:
            continue
        if normalized in seen or not dll_dir.is_dir():
            continue
        seen.add(normalized)
        result.append(str(dll_dir))
    return result


class QCloudMusicApi:
    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path
        self._dll_dir_handles: list[Any] = []
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            for dll_dir in windows_runtime_dll_dirs(library_path):
                self._dll_dir_handles.append(os.add_dll_directory(dll_dir))
        try:
            self.lib = ctypes.CDLL(str(library_path))
        except FileNotFoundError as exc:
            searched = "\n".join(f"  - {item}" for item in windows_runtime_dll_dirs(library_path))
            raise QCloudMusicApiError(
                "Could not load QCloudMusicApi or one of its runtime dependencies.\n"
                "On Windows this usually means Qt/MinGW runtime DLLs were not found.\n"
                "Searched DLL directories:\n"
                f"{searched}"
            ) from exc
        self._bind()

    def _bind(self) -> None:
        self.lib.invoke.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.invoke.restype = ctypes.c_char_p

        self.lib.invokeUrl.argtypes = [ctypes.c_char_p]
        self.lib.invokeUrl.restype = ctypes.c_char_p

        self.lib.freeApi.argtypes = []
        self.lib.freeApi.restype = None

        self.lib.set_cookie.argtypes = [ctypes.c_char_p]
        self.lib.set_cookie.restype = None

        self.lib.set_proxy.argtypes = [ctypes.c_char_p]
        self.lib.set_proxy.restype = None

        self.lib.setFilterRules.argtypes = [ctypes.c_char_p]
        self.lib.setFilterRules.restype = None

        if hasattr(self.lib, "set_realIP"):
            self.lib.set_realIP.argtypes = [ctypes.c_char_p]
            self.lib.set_realIP.restype = None

    def close(self) -> None:
        self.lib.freeApi()

    def set_cookie(self, cookie: str) -> None:
        self.lib.set_cookie(cookie.encode("utf-8"))

    def set_proxy(self, proxy: str) -> None:
        self.lib.set_proxy(proxy.encode("utf-8"))

    def set_real_ip(self, real_ip: str) -> None:
        if not hasattr(self.lib, "set_realIP"):
            raise QCloudMusicApiError("This QCloudMusicApi build does not expose set_realIP")
        self.lib.set_realIP(real_ip.encode("utf-8"))

    def set_filter_rules(self, rules: str) -> None:
        self.lib.setFilterRules(rules.encode("utf-8"))

    def invoke(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        ptr = self.lib.invoke(endpoint.encode("utf-8"), payload.encode("utf-8"))
        if not ptr:
            raise QCloudMusicApiError(f"{endpoint} returned a null pointer")
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise QCloudMusicApiError(f"{endpoint} returned invalid JSON: {text[:500]}") from exc


class HostOverrideProxy:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = {host.lower(): ip for host, ip in mapping.items()}
        proxy = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                proxy.handle_client(self.request)

        class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def handle_client(self, client: socket.socket) -> None:
        client.settimeout(20)
        header = self._read_header(client)
        if not header:
            return
        first_line = header.split(b"\r\n", 1)[0].decode("iso-8859-1", errors="replace")
        parts = first_line.split()
        if len(parts) < 3:
            return

        method, target, version = parts[0].upper(), parts[1], parts[2]
        if method == "CONNECT":
            host, port = self._split_host_port(target, 443)
            self._tunnel(client, host, port)
            return

        parsed = urlsplit(target)
        if not parsed.hostname:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            return
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        rewritten = header.replace(
            first_line.encode("iso-8859-1"),
            f"{parts[0]} {path} {version}".encode("iso-8859-1"),
            1,
        )
        self._forward_plain(client, host, port, rewritten)

    @staticmethod
    def _read_header(sock: socket.socket) -> bytes:
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data

    @staticmethod
    def _split_host_port(target: str, default_port: int) -> tuple[str, int]:
        if target.startswith("["):
            host, _, rest = target[1:].partition("]")
            port = int(rest[1:]) if rest.startswith(":") else default_port
            return host, port
        host, sep, port_text = target.rpartition(":")
        if sep and port_text.isdigit():
            return host, int(port_text)
        return target, default_port

    def _mapped_host(self, host: str) -> str:
        return self.mapping.get(host.lower(), host)

    def _tunnel(self, client: socket.socket, host: str, port: int) -> None:
        upstream = socket.create_connection((self._mapped_host(host), port), timeout=20)
        try:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._pipe(client, upstream)
        finally:
            upstream.close()

    def _forward_plain(self, client: socket.socket, host: str, port: int, first_packet: bytes) -> None:
        upstream = socket.create_connection((self._mapped_host(host), port), timeout=20)
        try:
            upstream.sendall(first_packet)
            self._pipe(client, upstream)
        finally:
            upstream.close()

    @staticmethod
    def _pipe(left: socket.socket, right: socket.socket) -> None:
        sockets = [left, right]
        for sock in sockets:
            sock.setblocking(False)
        while sockets:
            readable, _, exceptional = select.select(sockets, [], sockets, 30)
            if exceptional:
                break
            if not readable:
                break
            for sock in readable:
                other = right if sock is left else left
                try:
                    data = sock.recv(65536)
                    if not data:
                        return
                    other.sendall(data)
                except OSError:
                    return


@dataclass(frozen=True)
class ApiCall:
    endpoint: str
    params: dict[str, Any]


class SongStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.create_schema()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def create_schema(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    song_id INTEGER PRIMARY KEY,
                    name TEXT,
                    album_id INTEGER,
                    album_name TEXT,
                    duration_ms INTEGER,
                    artists_json TEXT NOT NULL DEFAULT '[]',
                    raw_detail_json TEXT,
                    collected_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    status INTEGER,
                    code INTEGER,
                    response_json TEXT,
                    error TEXT,
                    collected_at TEXT NOT NULL,
                    UNIQUE(song_id, endpoint, params_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_api_results_song_endpoint
                ON api_results(song_id, endpoint);
                """
            )
            self.conn.commit()

    def upsert_song_detail(self, song_id: int, detail_response: dict[str, Any]) -> None:
        body = detail_response.get("body", {})
        songs = body.get("songs") or detail_response.get("songs") or []
        song = songs[0] if songs else {}
        album = song.get("al") or song.get("album") or {}
        artists = song.get("ar") or song.get("artists") or []

        with self.lock:
            self.conn.execute(
                """
                INSERT INTO songs (
                    song_id, name, album_id, album_name, duration_ms,
                    artists_json, raw_detail_json, collected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    name = excluded.name,
                    album_id = excluded.album_id,
                    album_name = excluded.album_name,
                    duration_ms = excluded.duration_ms,
                    artists_json = excluded.artists_json,
                    raw_detail_json = excluded.raw_detail_json,
                    collected_at = excluded.collected_at
                """,
                (
                    song_id,
                    song.get("name"),
                    album.get("id"),
                    album.get("name"),
                    song.get("dt") or song.get("duration"),
                    json.dumps(artists, ensure_ascii=False),
                    json.dumps(detail_response, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
            self.conn.commit()

    def save_result(
        self,
        song_id: int,
        endpoint: str,
        params: dict[str, Any],
        response: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
        params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
        body = response.get("body", {}) if response else {}
        status = response.get("status") if response else None
        code = body.get("code") if isinstance(body, dict) else None
        response_json = (
            json.dumps(response, ensure_ascii=False, sort_keys=True) if response is not None else None
        )
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO api_results (
                    song_id, endpoint, params_json, params_hash, status, code,
                    response_json, error, collected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id, endpoint, params_hash) DO UPDATE SET
                    status = excluded.status,
                    code = excluded.code,
                    response_json = excluded.response_json,
                    error = excluded.error,
                    collected_at = excluded.collected_at
                """,
                (
                    song_id,
                    endpoint,
                    params_json,
                    params_hash,
                    status,
                    code,
                    response_json,
                    error,
                    utc_now(),
                ),
            )
            self.conn.commit()

    def song_snapshot(self, song_id: int) -> dict[str, Any]:
        with self.lock:
            song = self.conn.execute(
                "SELECT * FROM songs WHERE song_id = ?",
                (song_id,),
            ).fetchone()
            song_columns = [col[1] for col in self.conn.execute("PRAGMA table_info(songs)")]
            result_columns = [col[1] for col in self.conn.execute("PRAGMA table_info(api_results)")]
            api_rows = self.conn.execute(
                """
                SELECT * FROM api_results
                WHERE song_id = ?
                ORDER BY endpoint, id
                """,
                (song_id,),
            ).fetchall()

        song_data = dict(zip(song_columns, song)) if song else {"song_id": song_id}
        for key in ("artists_json", "raw_detail_json"):
            if song_data.get(key):
                song_data[key.removesuffix("_json")] = json.loads(song_data.pop(key))

        endpoints: dict[str, list[dict[str, Any]]] = {}
        for row in api_rows:
            item = dict(zip(result_columns, row))
            item["params"] = json.loads(item.pop("params_json"))
            item.pop("params_hash", None)
            if item.get("response_json"):
                item["response"] = json.loads(item.pop("response_json"))
            endpoints.setdefault(item["endpoint"], []).append(item)
        song_data["api_results"] = endpoints
        return song_data


def parse_song_ids(values: Iterable[str], ids_file: str | None) -> list[int]:
    raw_ids: list[str] = []
    for value in values:
        raw_ids.extend(part.strip() for part in value.split(","))
    if ids_file:
        text = Path(ids_file).read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            clean = line.split("#", 1)[0].strip()
            if clean:
                raw_ids.extend(part.strip() for part in clean.split(","))

    song_ids: list[int] = []
    seen: set[int] = set()
    for raw in raw_ids:
        if not raw:
            continue
        try:
            song_id = int(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid song id: {raw}") from exc
        if song_id not in seen:
            seen.add(song_id)
            song_ids.append(song_id)
    return song_ids


def unique_ints(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def body_songs(response: dict[str, Any]) -> list[dict[str, Any]]:
    body = response_body(response)
    songs = body.get("songs") if isinstance(body, dict) else None
    return songs if isinstance(songs, list) else []


def extract_album_id(detail_response: dict[str, Any]) -> int | None:
    songs = body_songs(detail_response)
    if not songs:
        return None
    album = songs[0].get("al") or songs[0].get("album") or {}
    album_id = album.get("id")
    return int(album_id) if album_id else None


def response_body(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body", response)
    return body if isinstance(body, dict) else {}


def extract_account_user_id(response: dict[str, Any]) -> int:
    body = response_body(response)
    candidates = [
        body.get("account", {}).get("id") if isinstance(body.get("account"), dict) else None,
        body.get("profile", {}).get("userId") if isinstance(body.get("profile"), dict) else None,
        body.get("profile", {}).get("userId") if isinstance(body.get("profile"), dict) else None,
    ]
    for candidate in candidates:
        if candidate:
            return int(candidate)
    raise QCloudMusicApiError("Could not read user id from user_account response")


def extract_playlists(response: dict[str, Any]) -> list[dict[str, Any]]:
    body = response_body(response)
    playlists = body.get("playlist", [])
    return playlists if isinstance(playlists, list) else []


def find_playlist_id(playlists: list[dict[str, Any]], name: str) -> int:
    normalized = name.strip().casefold()
    for playlist in playlists:
        if str(playlist.get("name", "")).strip().casefold() == normalized:
            return int(playlist["id"])
    for playlist in playlists:
        if normalized in str(playlist.get("name", "")).strip().casefold():
            return int(playlist["id"])
    names = ", ".join(str(item.get("name", "")) for item in playlists[:20])
    raise QCloudMusicApiError(f"Playlist not found: {name}. First playlists: {names}")


def extract_playlist_song_ids(response: dict[str, Any]) -> list[int]:
    songs = body_songs(response)
    song_ids = [int(song["id"]) for song in songs if song.get("id")]
    if song_ids:
        return unique_ints(song_ids)

    body = response_body(response)
    playlist = body.get("playlist", {})
    if isinstance(playlist, dict):
        track_ids = playlist.get("trackIds", [])
        if isinstance(track_ids, list):
            return unique_ints(
                int(item["id"]) for item in track_ids if isinstance(item, dict) and item.get("id")
            )
    return []


def build_calls(
    song_id: int,
    detail_response: dict[str, Any],
    levels: list[str],
    include_endpoints: set[str],
    comment_limit: int,
    simi_limit: int,
    common_params: dict[str, Any],
) -> list[ApiCall]:
    calls: list[ApiCall] = []
    for endpoint in DEFAULT_ENDPOINTS:
        if endpoint not in include_endpoints:
            continue
        params: dict[str, Any] = {"id": song_id}
        params.update(common_params)
        if endpoint == "comment_music":
            params.update({"limit": comment_limit, "offset": 0})
        elif endpoint in {"simi_song", "simi_playlist"}:
            params.update({"limit": simi_limit, "offset": 0})
        calls.append(ApiCall(endpoint, params))

    if "song_url_v1" in include_endpoints:
        for level in levels:
            calls.append(ApiCall("song_url_v1", {"id": song_id, "level": level, **common_params}))

    if "song_download_url_v1" in include_endpoints:
        for level in levels:
            calls.append(ApiCall("song_download_url_v1", {"id": song_id, "level": level, **common_params}))

    album_id = extract_album_id(detail_response)
    if album_id:
        if "album" in include_endpoints:
            calls.append(ApiCall("album", {"id": album_id, **common_params}))
        if "album_detail_dynamic" in include_endpoints:
            calls.append(ApiCall("album_detail_dynamic", {"id": album_id, **common_params}))
        if "album_privilege" in include_endpoints:
            calls.append(ApiCall("album_privilege", {"id": album_id, **common_params}))

    return calls


def endpoint_set(raw: str) -> set[str]:
    if raw == "default":
        return set(DEFAULT_ENDPOINTS) | {
            "song_url_v1",
            "album",
            "album_detail_dynamic",
        }
    if raw == "all":
        return set(DEFAULT_ENDPOINTS) | {
            "song_url_v1",
            "song_download_url_v1",
            "album",
            "album_detail_dynamic",
            "album_privilege",
        }
    return {item.strip() for item in raw.split(",") if item.strip()}


def write_snapshot(json_dir: Path, store: SongStore, song_id: int) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)
    snapshot = store.song_snapshot(song_id)
    path = json_dir / f"{song_id}.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def collect_song(
    api: QCloudMusicApi,
    store: SongStore,
    song_id: int,
    endpoints: set[str],
    levels: list[str],
    comment_limit: int,
    simi_limit: int,
    common_params: dict[str, Any],
    json_dir: Path,
    sleep_seconds: float,
) -> None:
    detail_params = {"ids": str(song_id), **common_params}
    try:
        detail = api.invoke("song_detail", detail_params)
        store.save_result(song_id, "song_detail", detail_params, detail)
        store.upsert_song_detail(song_id, detail)
    except Exception as exc:
        store.save_result(song_id, "song_detail", detail_params, None, str(exc))
        print(f"song_detail failed for {song_id}: {exc}", file=sys.stderr)
        return

    for call in build_calls(
        song_id,
        detail,
        levels,
        endpoints,
        comment_limit,
        simi_limit,
        common_params,
    ):
        try:
            response = api.invoke(call.endpoint, call.params)
            store.save_result(song_id, call.endpoint, call.params, response)
        except Exception as exc:
            store.save_result(song_id, call.endpoint, call.params, None, str(exc))
            print(f"{call.endpoint} failed for {song_id}: {exc}", file=sys.stderr)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_snapshot(json_dir, store, song_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Store full song metadata by calling QCloudMusicApi endpoints."
    )
    parser.add_argument("song_ids", nargs="*", help="Song ids, separated by spaces or commas.")
    parser.add_argument("--ids-file", help="Text file containing song ids, one id or CSV per line.")
    parser.add_argument("--playlist-id", type=int, help="Collect songs from a playlist id.")
    parser.add_argument("--user-playlist-uid", type=int, help="List this user's playlists and collect one by name.")
    parser.add_argument(
        "--my-playlist-name",
        help="Use cookie login to find one of your playlists by name, e.g. Ink_bai喜欢的音乐.",
    )
    parser.add_argument("--playlist-name", help="Playlist name used with --user-playlist-uid.")
    parser.add_argument("--playlist-limit", type=int, default=1000, help="Maximum playlists to scan.")
    parser.add_argument("--max-songs", type=int, default=None, help="Maximum songs to collect from sources.")
    parser.add_argument("--db", default="data/source/qcloud_songs.sqlite3", help="SQLite output path.")
    parser.add_argument("--json-dir", default="data/source/qcloud_song_json", help="Per-song JSON output dir.")
    parser.add_argument("--library", help="Path to QCloudMusicApi.dll/.so/.dylib.")
    parser.add_argument("--cookie", help="Netease Cloud Music cookie string for login-only data.")
    parser.add_argument("--cookie-file", help="UTF-8 text file containing a cookie string.")
    parser.add_argument("--proxy", help="Proxy URL passed to QCloudMusicApi.")
    parser.add_argument("--real-ip", help="realIP value passed to QCloudMusicApi, if supported.")
    parser.add_argument("--domain", help="Override QCloudMusicApi request domain, e.g. https://music.163.com.")
    parser.add_argument(
        "--resolve-host",
        action="append",
        default=[],
        help="Resolve a host through a local CONNECT proxy, e.g. interface.music.163.com=117.135.207.67. Can be repeated.",
    )
    parser.add_argument(
        "--levels",
        default=",".join(DEFAULT_LEVELS),
        help="Quality levels for song_url_v1 and song_download_url_v1.",
    )
    parser.add_argument(
        "--endpoints",
        default="default",
        help="default, all, or comma-separated endpoint names.",
    )
    parser.add_argument("--comment-limit", type=int, default=20)
    parser.add_argument("--simi-limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between API calls.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of songs to collect in parallel. Default: 1.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    song_ids = parse_song_ids(args.song_ids, args.ids_file)

    library_path = resolve_library_path(repo_root, args.library)
    api = QCloudMusicApi(library_path)
    store = SongStore(Path(args.db))
    override_proxy: HostOverrideProxy | None = None

    try:
        api.set_filter_rules("QCloudMusicApi.debug=false")
        cookie = args.cookie
        if args.cookie_file:
            cookie = Path(args.cookie_file).read_text(encoding="utf-8-sig").strip()
        if cookie:
            api.set_cookie(cookie)
        if args.real_ip:
            api.set_real_ip(args.real_ip)
        if args.resolve_host:
            host_map: dict[str, str] = {}
            for item in args.resolve_host:
                if "=" not in item:
                    raise SystemExit("--resolve-host must use HOST=IP format")
                host, ip = (part.strip() for part in item.split("=", 1))
                if not host or not ip:
                    raise SystemExit("--resolve-host must use HOST=IP format")
                host_map[host] = ip
            override_proxy = HostOverrideProxy(host_map)
            override_proxy.start()
            api.set_proxy(override_proxy.url)
            if not args.quiet:
                print(f"Using local host-override proxy: {override_proxy.url}")
        elif args.proxy:
            api.set_proxy(args.proxy)

        endpoints = endpoint_set(args.endpoints)
        levels = [item.strip() for item in args.levels.split(",") if item.strip()]
        common_params: dict[str, Any] = {}
        if args.domain:
            common_params["domain"] = args.domain

        playlist_ids: list[int] = []
        if args.playlist_id:
            playlist_ids.append(args.playlist_id)

        if args.my_playlist_name:
            if not cookie:
                raise SystemExit("--my-playlist-name requires --cookie or --cookie-file")
            account = api.invoke("user_account", common_params)
            uid = extract_account_user_id(account)
            playlists_response = api.invoke(
                "user_playlist",
                {"uid": uid, "limit": args.playlist_limit, "offset": 0, **common_params},
            )
            playlists = extract_playlists(playlists_response)
            playlist_ids.append(find_playlist_id(playlists, args.my_playlist_name))

        if args.user_playlist_uid:
            if not args.playlist_name:
                raise SystemExit("--user-playlist-uid requires --playlist-name")
            playlists_response = api.invoke(
                "user_playlist",
                {
                    "uid": args.user_playlist_uid,
                    "limit": args.playlist_limit,
                    "offset": 0,
                    **common_params,
                },
            )
            playlists = extract_playlists(playlists_response)
            playlist_ids.append(find_playlist_id(playlists, args.playlist_name))

        source_song_ids: list[int] = []
        for playlist_id in unique_ints(playlist_ids):
            if not args.quiet:
                print(f"Reading playlist {playlist_id}")
            params: dict[str, Any] = {"id": playlist_id, **common_params}
            playlist_response = api.invoke("playlist_detail", params)
            next_ids = extract_playlist_song_ids(playlist_response)
            if not next_ids:
                body = response_body(playlist_response)
                detail = body.get("msg") or body.get("message") or list(body.keys())[:10]
                raise QCloudMusicApiError(
                    f"No songs found in playlist {playlist_id}; playlist_detail returned: {detail}"
                )
            source_song_ids.extend(next_ids)

        song_ids = unique_ints([*song_ids, *source_song_ids])
        if args.max_songs is not None:
            song_ids = song_ids[: args.max_songs]
        if not song_ids:
            raise SystemExit(
                "Provide song ids or a source, e.g. --my-playlist-name Ink_bai喜欢的音乐"
            )

        requested_workers = max(1, args.workers)
        workers = min(requested_workers, len(song_ids), 1000)
        if requested_workers != workers and not args.quiet:
            print(
                f"Using {workers} workers instead of requested {requested_workers}; "
                "the crawler caps concurrency at 1000."
            )
        json_dir = Path(args.json_dir)
        if workers == 1:
            for index, song_id in enumerate(song_ids, start=1):
                if not args.quiet:
                    print(f"[{index}/{len(song_ids)}] collecting song {song_id}")
                collect_song(
                    api,
                    store,
                    song_id,
                    endpoints,
                    levels,
                    args.comment_limit,
                    args.simi_limit,
                    common_params,
                    json_dir,
                    args.sleep,
                )
        else:
            if not args.quiet:
                print(f"Collecting {len(song_ids)} songs with {workers} workers")
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        collect_song,
                        api,
                        store,
                        song_id,
                        endpoints,
                        levels,
                        args.comment_limit,
                        args.simi_limit,
                        common_params,
                        json_dir,
                        args.sleep,
                    ): song_id
                    for song_id in song_ids
                }
                for future in as_completed(futures):
                    song_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"song {song_id} failed: {exc}", file=sys.stderr)
                    completed += 1
                    if not args.quiet:
                        print(f"[{completed}/{len(song_ids)}] finished song {song_id}")

        if not args.quiet:
            print(f"SQLite saved to: {Path(args.db).resolve()}")
            print(f"JSON snapshots saved to: {Path(args.json_dir).resolve()}")
    finally:
        store.close()
        api.close()
        if override_proxy:
            override_proxy.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
