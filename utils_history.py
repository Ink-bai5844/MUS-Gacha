import json
import math
import hashlib
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from config import (
    HISTORY_CACHE_FILE,
    HISTORY_LINK_TRACKING_HOST,
    HISTORY_LINK_TRACKING_PORT,
    HISTORY_RECOMMENDATION_CACHE_SIZE,
    HISTORY_SETTINGS_FILE,
)
from utils_text import safe_text

_TRACKED_LINK_ITEMS = {}
_TRACKED_LINK_LOCK = threading.Lock()
_TRACKING_SERVER = None


DEFAULT_HISTORY_SETTINGS = {
    "selection_writes_history": True,
}

HISTORY_FEATURES = {
    "all_tags": "generated_tag_list",
    "language": "language_tags",
    "style": "style_tag_list",
    "emotion": "emotion_tag_list",
    "theme": "theme_tag_list",
    "scene": "scene_tag_list",
    "audio": "audio_tag_list",
    "lyric_terms": "lyric_terms",
    "lyric_semantic": "lyric_semantic_tags",
    "comment_semantic": "comment_semantic_tags",
    "artists": "artist_list",
    "title_terms": "title_terms",
}


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [safe_text(item) for item in value if safe_text(item)]
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    return []


def _unique_items(items):
    seen_items = set()
    unique_items = []
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        unique_items.append(item)
    return unique_items


def _trim_entries(entries, max_entries=HISTORY_RECOMMENDATION_CACHE_SIZE):
    valid_entries = [entry for entry in entries if isinstance(entry, dict)]
    if max_entries <= 0:
        return []
    return valid_entries[-max_entries:]


def load_history_entries():
    if not HISTORY_CACHE_FILE.exists():
        return []
    try:
        entries = json.loads(HISTORY_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(entries, list):
        return []
    return _trim_entries(entries)


def save_history_entries(entries):
    HISTORY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    trimmed_entries = _trim_entries(entries)
    temp_file = HISTORY_CACHE_FILE.with_suffix(f"{HISTORY_CACHE_FILE.suffix}.tmp")
    temp_file.write_text(json.dumps(trimmed_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(HISTORY_CACHE_FILE)
    return trimmed_entries


def clear_history_entries():
    return save_history_entries([])


def load_history_settings():
    settings = DEFAULT_HISTORY_SETTINGS.copy()
    if not HISTORY_SETTINGS_FILE.exists():
        return settings
    try:
        saved_settings = json.loads(HISTORY_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    if not isinstance(saved_settings, dict):
        return settings

    if "selection_writes_history" in saved_settings:
        settings["selection_writes_history"] = bool(saved_settings["selection_writes_history"])
    return settings


def save_history_settings(settings):
    merged_settings = DEFAULT_HISTORY_SETTINGS.copy()
    if isinstance(settings, dict):
        merged_settings.update(
            {
                "selection_writes_history": bool(
                    settings.get(
                        "selection_writes_history",
                        DEFAULT_HISTORY_SETTINGS["selection_writes_history"],
                    )
                )
            }
        )

    HISTORY_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = HISTORY_SETTINGS_FILE.with_suffix(f"{HISTORY_SETTINGS_FILE.suffix}.tmp")
    temp_file.write_text(json.dumps(merged_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(HISTORY_SETTINGS_FILE)
    return merged_settings


def save_selection_writes_history(enabled):
    settings = load_history_settings()
    settings["selection_writes_history"] = bool(enabled)
    return save_history_settings(settings)


def build_history_entry(row_data, action):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()
    row_data = row_data or {}

    entry = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "action": safe_text(action),
        "song_id": safe_text(row_data.get("song_id")),
        "name": safe_text(row_data.get("name")),
        "artist_names": safe_text(row_data.get("artist_names")),
        "album_name": safe_text(row_data.get("album_name")),
    }
    for history_key, row_key in HISTORY_FEATURES.items():
        entry[history_key] = _unique_items(_coerce_list(row_data.get(row_key)))
    return entry


def record_recommendation_history(row_data, action="select"):
    entry = build_history_entry(row_data, action)
    if not entry["song_id"] and not entry["name"]:
        return load_history_entries()

    entries = load_history_entries()
    entries.append(entry)
    return save_history_entries(entries)


def _is_valid_web_link(link):
    normalized_link = safe_text(link).lower()
    return normalized_link.startswith(("http://", "https://"))


def _build_tracking_token(row_data):
    song_id = safe_text(row_data.get("song_id"))
    if song_id:
        return song_id

    raw_token = json.dumps(
        {
            "name": safe_text(row_data.get("name")),
            "artist_names": safe_text(row_data.get("artist_names")),
            "link": safe_text(row_data.get("netease_url")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw_token.encode("utf-8")).hexdigest()


def register_tracked_link_item(row_data):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()

    row_data = row_data or {}
    token = _build_tracking_token(row_data)
    with _TRACKED_LINK_LOCK:
        _TRACKED_LINK_ITEMS[token] = dict(row_data)
    return token


def build_tracked_link(row_data):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()

    row_data = row_data or {}
    target_link = safe_text(row_data.get("netease_url"))
    if not _is_valid_web_link(target_link):
        return target_link

    token = register_tracked_link_item(row_data)
    return (
        f"http://{HISTORY_LINK_TRACKING_HOST}:{HISTORY_LINK_TRACKING_PORT}/open"
        f"?token={quote(token)}&target={quote(target_link, safe='')}"
    )


class _HistoryTrackingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        if parsed_url.path != "/open":
            self.send_error(404)
            return

        query = parse_qs(parsed_url.query)
        token = query.get("token", [""])[0]
        target_link = query.get("target", [""])[0]

        with _TRACKED_LINK_LOCK:
            item_payload = _TRACKED_LINK_ITEMS.get(token)

        if item_payload:
            record_recommendation_history(item_payload, "netease_link")

        if not _is_valid_web_link(target_link):
            target_link = safe_text((item_payload or {}).get("netease_url"))

        if not _is_valid_web_link(target_link):
            self.send_error(400, "Invalid target link")
            return

        self.send_response(302)
        self.send_header("Location", target_link)
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_link_tracking_server():
    global _TRACKING_SERVER

    if _TRACKING_SERVER is not None:
        return _TRACKING_SERVER

    try:
        server = ThreadingHTTPServer(
            (HISTORY_LINK_TRACKING_HOST, HISTORY_LINK_TRACKING_PORT),
            _HistoryTrackingHandler,
        )
    except OSError:
        return None

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _TRACKING_SERVER = server
    return server


def _count_history_features(history_entries, field_name):
    counter = Counter()
    for entry in history_entries:
        if isinstance(entry, dict):
            counter.update(_unique_items(_coerce_list(entry.get(field_name))))
    return counter


def _build_rarity_bonus_map(history_counter, database_counter, bonus_scale):
    if not history_counter or bonus_scale <= 0:
        return {}

    total_database_occurrences = max(sum(database_counter.values()), 1)
    bonus_map = {}
    for feature_name, history_count in history_counter.items():
        database_count = max(int(database_counter.get(feature_name, 0)), 0)
        rarity_factor = math.log1p((total_database_occurrences + 1) / (database_count + 1))
        bonus_map[feature_name] = float(history_count) * rarity_factor * float(bonus_scale)
    return bonus_map


def build_history_preference_maps(history_entries, scoring_resources, dimension_weights):
    if not history_entries:
        return {}

    scale_by_key = {
        "all_tags": float(dimension_weights.get("综合标签", 1.0)),
        "language": float(dimension_weights.get("语种", 1.0)),
        "style": float(dimension_weights.get("风格", 1.0)),
        "emotion": float(dimension_weights.get("情绪", 1.0)),
        "theme": float(dimension_weights.get("主题", 1.0)),
        "scene": float(dimension_weights.get("场景", 1.0)),
        "audio": float(dimension_weights.get("音频标签", 1.0)),
        "artists": float(dimension_weights.get("歌手", 1.0)),
        "title_terms": float(dimension_weights.get("歌名关键词", 1.0)),
        "lyric_terms": float(dimension_weights.get("歌词关键词", 1.0)),
        "lyric_semantic": float(dimension_weights.get("歌词语义", 1.0)),
        "comment_semantic": float(dimension_weights.get("评论语义", 1.0)),
    }

    preference_maps = {}
    for feature_key in HISTORY_FEATURES:
        history_counter = _count_history_features(history_entries, feature_key)
        preference_maps[feature_key] = _build_rarity_bonus_map(
            history_counter,
            scoring_resources.get(feature_key, Counter()),
            scale_by_key.get(feature_key, 1.0),
        )
    return preference_maps
