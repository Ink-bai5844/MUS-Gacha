"""
Build explainable song tags from the liked-song CSV, local lyrics, and optional
local audio files.

Examples:
    python data_processing/build_song_tags.py
    python data_processing/build_song_tags.py --audio-dir H:\音乐
    python data_processing/build_song_tags.py --analyze-audio --audio-feature-seconds 45
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
TAG_DATA_DIR = DATA_DIR / "tags"
MATCH_DATA_DIR = DATA_DIR / "matches"
AUDIO_FEATURE_DATA_DIR = DATA_DIR / "features" / "audio"
MERT_DATA_DIR = DATA_DIR / "features" / "mert"
DEFAULT_LYRICS_DIR = SOURCE_DATA_DIR / "lyrics"
DEFAULT_MERT_MODEL_DIR = BASE_DIR / "models" / "MERT-v1-330M"
DEFAULT_MERT_EMBEDDINGS_DIR = MERT_DATA_DIR / "embeddings"
DEFAULT_AUDIO_DIR = Path(r"H:\音乐")
DEFAULT_SOURCE_SEPARATION_CHECKPOINT_DIR = BASE_DIR / "models"
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"}
SOURCE_SEPARATION_MODELS = {
    "hdemucs_high_musdb_plus": {
        "bundle": "HDEMUCS_HIGH_MUSDB_PLUS",
        "checkpoint": DEFAULT_SOURCE_SEPARATION_CHECKPOINT_DIR / "hdemucs_high_trained.pt",
    },
    "hdemucs_high_musdb": {
        "bundle": "HDEMUCS_HIGH_MUSDB",
        "checkpoint": DEFAULT_SOURCE_SEPARATION_CHECKPOINT_DIR / "hdemucs_high_musdbhq_only.pt",
    },
}
SOURCE_SEPARATION_SOURCES = ["drums", "bass", "other", "vocals"]


LANGUAGE_TAGS = ["国语", "粤语", "英语", "日语", "韩语", "纯音乐", "华语", "欧美"]

STYLE_RULES = [
    ("华语流行", ["流行-华语流行", "华语流行"]),
    ("粤语流行", ["流行-粤语流行", "粤语流行"]),
    ("欧美流行", ["流行-欧美流行", "欧美流行"]),
    ("日语流行", ["流行-日语流行", "日语流行", "J-Pop", "JPOP"]),
    ("韩语流行", ["流行-韩语流行", "韩语流行", "K-Pop", "KPOP"]),
    ("轻音乐", ["轻音乐", "New Age", "新世纪"]),
    ("纯音乐", ["纯音乐，请欣赏", "纯音乐"]),
    ("钢琴", ["钢琴", "Piano", "piano"]),
    ("管弦", ["管弦", "交响", "Orchestra", "orchestra", "Philharmonic"]),
    ("电子", ["电子", "EDM", "Future Bass", "House", "Trance"]),
    ("摇滚", ["摇滚", "Rock", "rock"]),
    ("民谣", ["民谣", "Folk", "folk"]),
    ("古风", ["古风", "国风"]),
    ("说唱", ["说唱", "Hip-Hop", "Rap", "rap"]),
    ("R&B", ["R&B", "节奏布鲁斯"]),
    ("爵士", ["爵士", "Jazz", "jazz"]),
    ("古典", ["古典", "Classical", "classical"]),
    ("游戏音乐", ["游戏原声", "游戏音乐", "HOYO-MiX", "原神", "崩坏", "鸣潮"]),
    ("影视原声", ["影视原声", "电视剧原声", "电影原声", "主题曲", "插曲", "片尾曲", "片头曲", "OST"]),
]

EMOTION_RULES = [
    ("治愈", ["治愈", "温暖", "暖暖", "安心", "疗愈"]),
    ("欢快", ["欢快", "快乐", "开心", "明亮", "阳光", "活力"]),
    ("悲伤", ["悲伤", "忧伤", "难过", "心酸", "眼泪", "流泪", "哭", "遗憾"]),
    ("怀旧", ["怀旧", "回忆", "青春", "童年", "光阴", "旧时光", "经典老歌"]),
    ("热血", ["热血", "励志", "燃", "加油", "坚持", "不应舍弃", "命运"]),
    ("宁静", ["宁静", "安静", "平静", "舒缓", "放松", "夜的钢琴曲"]),
    ("孤独", ["孤独", "孤单", "寂寞", "一个人"]),
    ("浪漫", ["浪漫", "爱", "陪伴", "想你", "温柔"]),
    ("史诗", ["史诗", "宏大", "壮阔", "苍茫", "英雄", "战争"]),
    ("神秘", ["神秘", "梦境", "幻想", "奇幻"]),
]

WIKI_EXPLICIT_EMOTIONS = {
    "治愈": "治愈",
    "悲伤": "悲伤",
    "欢快": "欢快",
    "活力": "欢快",
    "热血励志": "热血",
    "自信": "热血",
    "宁静": "宁静",
}

WIKI_EXPLICIT_THEMES = {
    "思念": "思念",
}

THEME_RULES = [
    ("爱情", ["爱情", "爱你", "爱意", "恋人", "喜欢你", "陪伴你", "想你"]),
    ("离别", ["离别", "分别", "再见", "离开", "失去", "告别"]),
    ("成长", ["成长", "青春", "少年", "人生", "长大"]),
    ("时光", ["时光", "光阴", "岁月", "昨天", "从前", "回忆"]),
    ("江湖", ["江湖", "恩义", "射雕", "天边", "风沙", "侠"]),
    ("旅途", ["旅途", "旅行", "远方", "路程", "流浪", "风景"]),
    ("梦想", ["梦想", "梦", "希望", "未来", "坚持"]),
    ("思念", ["思念", "想念", "想你", "挂念"]),
    ("自然", ["春天", "秋天", "冬天", "夏日", "大海", "海边", "星空", "森林", "花海", "雨中"]),
]

SCENE_RULES = [
    ("学习", ["学习", "自习", "专注"]),
    ("睡前", ["睡前", "入眠", "晚安", "摇篮曲"]),
    ("工作", ["工作", "办公", "专注"]),
    ("通勤", ["通勤", "开车", "车上"]),
    ("运动", ["运动", "跑步", "健身"]),
    ("KTV", ["KTV", "卡拉OK"]),
    ("回忆杀", ["回忆杀", "童年", "青春", "经典老歌"]),
    ("放松", ["放松", "舒缓", "治愈", "轻音乐"]),
    ("燃向", ["热血", "励志", "燃", "史诗"]),
]

MERT_EMOTION_TAGS = {
    "calm": "宁静",
    "happy": "欢快",
    "sad": "悲伤",
    "angry": "激烈",
    "romantic": "浪漫",
    "tense": "紧张",
    "energetic": "热血",
    "melancholic": "忧郁",
}


def strip_dataset_suffix(name: str) -> str:
    clean = name.strip().replace(" ", "_")
    for suffix in ("_song_tags", "_song_features", "_song_matches", "_songs", "_json"):
        if clean.endswith(suffix):
            return clean[: -len(suffix)] or "songs"
    return clean or "songs"


def dataset_name_from_path(path: Path) -> str:
    return strip_dataset_suffix(path.stem)


def first_source_csv_path() -> Path:
    if SOURCE_DATA_DIR.exists():
        paths = sorted(path for path in SOURCE_DATA_DIR.glob("*.csv") if path.is_file())
        if paths:
            return paths[0]
    return SOURCE_DATA_DIR / "songs.csv"


def apply_derived_output_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.input is None:
        args.input = first_source_csv_path()

    dataset = dataset_name_from_path(args.input)
    if args.output is None:
        args.output = TAG_DATA_DIR / f"{dataset}_song_tags.csv"
    if args.jsonl_output is None:
        args.jsonl_output = TAG_DATA_DIR / f"{dataset}_song_tags.jsonl"
    if args.matches_output is None:
        args.matches_output = MATCH_DATA_DIR / f"{dataset}_song_matches.csv"
    if args.audio_features_csv is None:
        args.audio_features_csv = AUDIO_FEATURE_DATA_DIR / f"{dataset}_song_features.csv"
    if args.audio_features_parquet is None:
        args.audio_features_parquet = AUDIO_FEATURE_DATA_DIR / f"{dataset}_song_features.parquet"
    if args.mert_index is None:
        args.mert_index = MERT_DATA_DIR / f"{dataset}_mert_index.csv"
    if args.mert_clusters_output is None:
        args.mert_clusters_output = MERT_DATA_DIR / f"{dataset}_mert_clusters.csv"
    return args


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def progress_iter(iterable: Any, total: int | None = None, description: str = "", enabled: bool = True):
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm  # type: ignore

        yield from tqdm(iterable, total=total, desc=description, unit="项", ncols=100)
        return
    except Exception:
        pass

    prefix = f"{description}: " if description else ""
    update_every = max(1, (total or 100) // 100)
    for index, item in enumerate(iterable, start=1):
        if total:
            if index == 1 or index == total or index % update_every == 0:
                ratio_value = index / max(total, 1)
                filled = int(ratio_value * 28)
                bar = "#" * filled + "-" * (28 - filled)
                sys.stderr.write(f"\r{prefix}[{bar}] {index}/{total} {ratio_value:>6.1%}")
                sys.stderr.flush()
        elif index == 1 or index % 50 == 0:
            sys.stderr.write(f"\r{prefix}{index} 项")
            sys.stderr.flush()
        yield item
    if total is not None or description:
        sys.stderr.write("\n")
        sys.stderr.flush()


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def compact_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）", " ", value)
    value = re.sub(r"[`~!@#$%^&*_=+\[\]{}\\|;:'\",.<>/?，。、《》？；：‘’“”【】！￥…—·]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", "", compact_text(value))


def contains_any(text: str, needles: list[str]) -> bool:
    text_lower = text.lower()
    return any(needle.lower() in text_lower for needle in needles)


def parse_bpm(text: str) -> int | None:
    patterns = [
        r"BPM\s*\|\s*(\d{2,3})",
        r"BPM[:：\s]+(\d{2,3})",
        r"bpm\s*\|\s*(\d{2,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def read_lyric(lyrics_dir: Path, song_id: str) -> str:
    path = lyrics_dir / f"{song_id}.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").strip()


def add_tag(
    categories: dict[str, dict[str, float]],
    sources: dict[str, list[str]],
    category: str,
    tag: str,
    score: float,
    source: str,
) -> None:
    if not tag:
        return
    categories[category][tag] = max(categories[category].get(tag, 0.0), score)
    if source not in sources[tag]:
        sources[tag].append(source)


def add_rules(
    categories: dict[str, dict[str, float]],
    sources: dict[str, list[str]],
    category: str,
    text: str,
    rules: list[tuple[str, list[str]]],
    score: float,
    source: str,
) -> None:
    for tag, needles in rules:
        if contains_any(text, needles):
            add_tag(categories, sources, category, tag, score, source)


def add_explicit_wiki_tags(
    categories: dict[str, dict[str, float]],
    sources: dict[str, list[str]],
    wiki: str,
) -> None:
    tokens = [token.strip() for token in wiki.split("|") if token.strip()]
    for token in tokens:
        if token in WIKI_EXPLICIT_EMOTIONS:
            add_tag(categories, sources, "emotion", WIKI_EXPLICIT_EMOTIONS[token], 0.86, "csv.wiki_explicit_tag")
        if token in WIKI_EXPLICIT_THEMES:
            add_tag(categories, sources, "theme", WIKI_EXPLICIT_THEMES[token], 0.82, "csv.wiki_explicit_tag")


def decade_tag(publish_date: str) -> str:
    match = re.match(r"(\d{4})", publish_date or "")
    if not match:
        return ""
    year = int(match.group(1))
    if year < 1970:
        return "年代未知"
    decade = year // 10 * 10
    return f"{decade}年代"


def build_tags_for_row(row: pd.Series, lyric: str) -> dict[str, Any]:
    categories: dict[str, dict[str, float]] = {
        "language": {},
        "style": {},
        "emotion": {},
        "theme": {},
        "scene": {},
        "audio": {},
    }
    sources: dict[str, list[str]] = defaultdict(list)

    wiki = safe_text(row.get("wiki_summary_excerpt"))
    metadata_text = " ".join(
        [
            safe_text(row.get("name")),
            safe_text(row.get("aliases")),
            safe_text(row.get("translations")),
            safe_text(row.get("artist_names")),
            safe_text(row.get("album_name")),
            safe_text(row.get("lyric_excerpt")),
            safe_text(row.get("first_hot_comment")),
            safe_text(row.get("first_comment")),
        ]
    )
    style_text = " ".join(
        [
            metadata_text,
            safe_text(row.get("similar_song_names")),
            safe_text(row.get("similar_artist_names")),
            wiki,
        ]
    )
    lyric_blob = lyric or safe_text(row.get("lyric_excerpt"))

    for language in LANGUAGE_TAGS:
        if language in wiki:
            add_tag(categories, sources, "language", language, 0.95, "csv.wiki_summary_excerpt")
        elif language in metadata_text:
            add_tag(categories, sources, "language", language, 0.65, "csv.metadata")

    add_rules(categories, sources, "style", style_text, STYLE_RULES, 0.78, "csv.text")
    add_rules(categories, sources, "style", wiki, STYLE_RULES, 0.92, "csv.wiki_summary_excerpt")
    add_rules(categories, sources, "emotion", metadata_text, EMOTION_RULES, 0.62, "csv.metadata")
    add_explicit_wiki_tags(categories, sources, wiki)
    add_rules(categories, sources, "theme", metadata_text, THEME_RULES, 0.56, "csv.metadata")
    add_rules(categories, sources, "scene", metadata_text, SCENE_RULES, 0.62, "csv.metadata")

    if lyric_blob:
        add_rules(categories, sources, "emotion", lyric_blob, EMOTION_RULES, 0.76, "lyrics")
        add_rules(categories, sources, "theme", lyric_blob, THEME_RULES, 0.78, "lyrics")
        add_rules(categories, sources, "scene", lyric_blob, SCENE_RULES, 0.62, "lyrics")

    if "纯音乐，请欣赏" in lyric_blob or ("纯音乐" in wiki and safe_text(row.get("has_lyric")).lower() != "true"):
        add_tag(categories, sources, "language", "纯音乐", 0.98, "lyrics")
        add_tag(categories, sources, "style", "纯音乐", 0.96, "lyrics")
        add_tag(categories, sources, "scene", "放松", 0.55, "lyrics")

    bpm = parse_bpm(wiki)
    if bpm:
        if bpm >= 125:
            add_tag(categories, sources, "audio", "快节奏", 0.82, "csv.wiki_bpm")
        elif bpm <= 80:
            add_tag(categories, sources, "audio", "慢节奏", 0.82, "csv.wiki_bpm")
        else:
            add_tag(categories, sources, "audio", "中速", 0.62, "csv.wiki_bpm")

    decade = decade_tag(safe_text(row.get("publish_date")))
    if decade and decade != "年代未知":
        add_tag(categories, sources, "scene", decade, 0.52, "csv.publish_date")

    category_strings = {
        f"{category}_tags": " | ".join(sorted(tags, key=lambda item: (-categories[category][item], item)))
        for category, tags in categories.items()
    }
    all_scores = {}
    for tags in categories.values():
        all_scores.update(tags)
    sorted_tags = sorted(all_scores, key=lambda item: (-all_scores[item], item))

    return {
        **category_strings,
        "all_tags": " | ".join(sorted_tags),
        "tag_confidence": round(max(all_scores.values()) if all_scores else 0.0, 3),
        "tag_sources": json.dumps(sources, ensure_ascii=False, sort_keys=True),
        "bpm": bpm or "",
    }


def parse_audio_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    for sep in [" - ", " – ", " — ", "-"]:
        if sep in stem:
            artist, title = stem.split(sep, 1)
            return artist.strip(), title.strip()
    return "", stem.strip()


def decode_text_frame(payload: bytes) -> str:
    if not payload:
        return ""
    encoding = payload[0]
    data = payload[1:]
    encodings = {
        0: "latin-1",
        1: "utf-16",
        2: "utf-16-be",
        3: "utf-8",
    }
    try:
        return data.decode(encodings.get(encoding, "utf-8"), errors="replace").strip("\x00").strip()
    except Exception:
        return ""


def synchsafe_to_int(data: bytes) -> int:
    value = 0
    for byte in data:
        value = (value << 7) | (byte & 0x7F)
    return value


def read_mp3_metadata(path: Path) -> dict[str, str]:
    frame_map = {"TIT2": "title", "TPE1": "artist", "TALB": "album"}
    metadata = {"title": "", "artist": "", "album": ""}
    try:
        with path.open("rb") as file:
            header = file.read(10)
            if len(header) != 10 or header[:3] != b"ID3":
                return metadata
            major = header[3]
            tag_size = synchsafe_to_int(header[6:10])
            tag_data = file.read(tag_size)
    except OSError:
        return metadata

    offset = 0
    while offset + 10 <= len(tag_data):
        frame_id = tag_data[offset : offset + 4].decode("latin-1", errors="ignore")
        if not frame_id.strip("\x00"):
            break
        size_bytes = tag_data[offset + 4 : offset + 8]
        frame_size = synchsafe_to_int(size_bytes) if major == 4 else int.from_bytes(size_bytes, "big")
        payload_start = offset + 10
        payload_end = payload_start + frame_size
        if frame_size <= 0 or payload_end > len(tag_data):
            break
        if frame_id in frame_map:
            metadata[frame_map[frame_id]] = decode_text_frame(tag_data[payload_start:payload_end])
        offset = payload_end
    return metadata


def read_flac_metadata(path: Path) -> dict[str, str]:
    metadata = {"title": "", "artist": "", "album": ""}
    try:
        with path.open("rb") as file:
            if file.read(4) != b"fLaC":
                return metadata
            is_last = False
            while not is_last:
                header = file.read(4)
                if len(header) != 4:
                    break
                is_last = bool(header[0] & 0x80)
                block_type = header[0] & 0x7F
                block_size = int.from_bytes(header[1:4], "big")
                payload = file.read(block_size)
                if block_type != 4:
                    continue

                cursor = 0
                if cursor + 4 > len(payload):
                    break
                vendor_len = int.from_bytes(payload[cursor : cursor + 4], "little")
                cursor += 4 + vendor_len
                if cursor + 4 > len(payload):
                    break
                comment_count = int.from_bytes(payload[cursor : cursor + 4], "little")
                cursor += 4
                for _ in range(comment_count):
                    if cursor + 4 > len(payload):
                        break
                    comment_len = int.from_bytes(payload[cursor : cursor + 4], "little")
                    cursor += 4
                    comment = payload[cursor : cursor + comment_len].decode("utf-8", errors="replace")
                    cursor += comment_len
                    key, _, value = comment.partition("=")
                    key = key.lower()
                    if key in metadata and value and not metadata[key]:
                        metadata[key] = value.strip()
    except OSError:
        return metadata
    return metadata


def read_audio_metadata(path: Path) -> dict[str, str]:
    try:
        from mutagen import File as MutagenFile  # type: ignore

        audio = MutagenFile(str(path), easy=True)
        if audio:
            return {
                "title": safe_text((audio.get("title") or [""])[0]),
                "artist": safe_text((audio.get("artist") or [""])[0]),
                "album": safe_text((audio.get("album") or [""])[0]),
            }
    except Exception:
        pass

    if path.suffix.lower() == ".flac":
        return read_flac_metadata(path)
    if path.suffix.lower() == ".mp3":
        return read_mp3_metadata(path)
    return {"title": "", "artist": "", "album": ""}


def get_audio_duration(path: Path) -> tuple[float | None, str]:
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.num_frames / info.sample_rate, ""
        return None, "empty duration metadata"
    except Exception as exc:
        return None, str(exc)


def estimate_tempo_from_onsets(onset_envelope: Any, sample_rate: int, hop_length: int) -> float | None:
    try:
        import torch
    except Exception:
        return None

    if onset_envelope.numel() < 8:
        return None

    envelope = onset_envelope.float()
    envelope = envelope - envelope.mean()
    if float(envelope.abs().max()) <= 1e-8:
        return None

    min_bpm, max_bpm = 55, 190
    min_lag = max(1, round(60 * sample_rate / (max_bpm * hop_length)))
    max_lag = max(min_lag + 1, round(60 * sample_rate / (min_bpm * hop_length)))
    max_lag = min(max_lag, envelope.numel() - 1)
    if max_lag <= min_lag:
        return None

    scores = []
    for lag in range(min_lag, max_lag + 1):
        scores.append(torch.sum(envelope[:-lag] * envelope[lag:]))
    corr = torch.stack(scores)
    best_lag = min_lag + int(torch.argmax(corr).item())
    tempo = 60 * sample_rate / (best_lag * hop_length)

    while tempo < 70:
        tempo *= 2
    while tempo > 180:
        tempo /= 2
    return float(tempo)


def choose_torch_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_source_separator_model(bundle: Any, checkpoint_path: Path | None) -> Any:
    import torch

    if checkpoint_path and checkpoint_path.exists():
        model = bundle._model_factory_func()
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return bundle.get_model()


def build_source_separator(model_name: str, device_name: str, checkpoint_path: Path | None = None) -> dict[str, Any]:
    import torchaudio

    model_spec = SOURCE_SEPARATION_MODELS.get(model_name)
    if not model_spec:
        available = ", ".join(sorted(SOURCE_SEPARATION_MODELS))
        raise ValueError(f"unknown source separation model {model_name!r}; choose one of: {available}")

    bundle = getattr(torchaudio.pipelines, model_spec["bundle"])
    resolved_checkpoint = checkpoint_path or model_spec["checkpoint"]
    device = choose_torch_device(device_name)
    model = _load_source_separator_model(bundle, resolved_checkpoint).to(device)
    model.eval()
    return {
        "name": model_name,
        "bundle": bundle,
        "checkpoint": str(resolved_checkpoint) if resolved_checkpoint and resolved_checkpoint.exists() else "",
        "model": model,
        "device": device,
        "sample_rate": int(bundle.sample_rate),
    }


def _fit_waveform_for_separator(waveform: Any, sample_rate: int, separator: dict[str, Any]) -> tuple[Any, int]:
    import torch
    import torchaudio

    waveform = waveform.float()
    target_sample_rate = separator["sample_rate"]
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        sample_rate = target_sample_rate
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    peak = waveform.abs().max()
    if bool(torch.isfinite(peak)) and float(peak) > 1.0:
        waveform = waveform / peak
    return waveform, sample_rate


def estimate_sources_with_separator(
    path: Path,
    seconds: float,
    separator: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torchaudio

    try:
        info = torchaudio.info(str(path))
        source_sample_rate = info.sample_rate
        frame_count = min(info.num_frames, int(source_sample_rate * seconds)) if seconds > 0 else info.num_frames
        waveform, sample_rate = torchaudio.load(str(path), num_frames=frame_count)
        if waveform.numel() == 0:
            return {"source_separation_error": "empty audio"}

        waveform, sample_rate = _fit_waveform_for_separator(waveform, sample_rate, separator)
        mixture = waveform.unsqueeze(0).to(separator["device"])
        with torch.inference_mode():
            estimates = separator["model"](mixture)

        if estimates.ndim != 4 or estimates.shape[1] < len(SOURCE_SEPARATION_SOURCES):
            return {"source_separation_error": f"unexpected model output shape: {tuple(estimates.shape)}"}

        source_index = {name: index for index, name in enumerate(SOURCE_SEPARATION_SOURCES)}
        source_energies = {
            name: float(estimates[:, source_index[name]].square().mean().detach().cpu())
            for name in SOURCE_SEPARATION_SOURCES
        }
        total_energy = sum(source_energies.values()) + 1e-12
        vocal_ratio = source_energies["vocals"] / total_energy
        instrumental_ratio = (
            source_energies["drums"] + source_energies["bass"] + source_energies["other"]
        ) / total_energy
        return {
            "source_separation_model": separator["name"],
            "source_separation_error": "",
            "source_drums_energy_ratio": round(source_energies["drums"] / total_energy, 4),
            "source_bass_energy_ratio": round(source_energies["bass"] / total_energy, 4),
            "source_other_energy_ratio": round(source_energies["other"] / total_energy, 4),
            "source_vocals_energy_ratio": round(vocal_ratio, 4),
            "source_vocal_energy_ratio": round(vocal_ratio, 4),
            "source_instrumental_energy_ratio": round(instrumental_ratio, 4),
        }
    except Exception as exc:  # pragma: no cover - depends on local model/cache/codecs.
        return {
            "source_separation_model": safe_text(separator.get("name")),
            "source_separation_error": str(exc),
        }


def infer_vocal_instrumental_tags(row: pd.Series) -> dict[str, Any]:
    source_vocal_ratio = pd.to_numeric(row.get("source_vocal_energy_ratio"), errors="coerce")
    source_instrumental_ratio = pd.to_numeric(row.get("source_instrumental_energy_ratio"), errors="coerce")
    source_error = safe_text(row.get("source_separation_error"))
    if pd.notna(source_vocal_ratio) and pd.notna(source_instrumental_ratio) and not source_error:
        vocal_score = max(0.0, min(1.0, float(source_vocal_ratio)))
        instrumental_score = max(0.0, min(1.0, float(source_instrumental_ratio)))
        if vocal_score >= 0.34:
            label = "人声强"
        elif instrumental_score >= 0.78:
            label = "器乐强"
        else:
            label = "人声/器乐均衡"
        return {
            "vocal_presence_score": round(vocal_score, 3),
            "instrumental_presence_score": round(instrumental_score, 3),
            "vocal_instrumental_tags": label,
        }

    text = " | ".join(
        [
            safe_text(row.get("language_tags")),
            safe_text(row.get("style_tags")),
            safe_text(row.get("all_tags")),
        ]
    )
    pure_instrumental = any(tag in text for tag in ["纯音乐", "轻音乐", "钢琴", "管弦", "古典"])
    has_lyric = safe_text(row.get("has_lyric")).lower() in {"true", "1", "yes", "y"}
    band_ratio = pd.to_numeric(row.get("audio_vocal_band_ratio"), errors="coerce")
    band_ratio = float(band_ratio) if pd.notna(band_ratio) else 0.0

    vocal_score = 0.2
    if has_lyric:
        vocal_score += 0.38
    if band_ratio >= 0.44:
        vocal_score += 0.22
    elif band_ratio >= 0.34:
        vocal_score += 0.12
    if pure_instrumental:
        vocal_score -= 0.35
    vocal_score = max(0.0, min(1.0, vocal_score))

    instrumental_score = 1.0 - vocal_score
    if pure_instrumental:
        instrumental_score = max(instrumental_score, 0.82)
    if not has_lyric:
        instrumental_score = max(instrumental_score, 0.68)
    instrumental_score = max(0.0, min(1.0, instrumental_score))

    labels = []
    if vocal_score >= 0.62:
        labels.append("人声强")
    elif instrumental_score >= 0.62:
        labels.append("器乐强")
    else:
        labels.append("人声/器乐均衡")

    return {
        "vocal_presence_score": round(vocal_score, 3),
        "instrumental_presence_score": round(instrumental_score, 3),
        "vocal_instrumental_tags": " | ".join(labels),
    }


def build_match_candidates(df: pd.DataFrame) -> list[dict[str, str]]:
    candidates = []
    for _, row in df.iterrows():
        song_id = safe_text(row.get("song_id"))
        name = safe_text(row.get("name"))
        artist_names = safe_text(row.get("artist_names"))
        artists = split_pipe(artist_names) or [artist_names]
        aliases = split_pipe(row.get("aliases"))
        titles = [name] + aliases

        for title in titles:
            if not title:
                continue
            title_norm = normalize_match_text(title)
            for artist in artists:
                artist_norm = normalize_match_text(artist)
                candidates.append(
                    {
                        "song_id": song_id,
                        "name": name,
                        "artist_names": artist_names,
                        "duration_seconds": safe_text(row.get("duration_seconds")),
                        "title_norm": title_norm,
                        "artist_norm": artist_norm,
                        "combo_norm": normalize_match_text(f"{artist} {title}"),
                    }
                )
    return candidates


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    if shorter >= 5 and (left in right or right in left):
        return 0.92
    if shorter / longer < 0.42:
        return 0.0
    if left[0] != right[0] and left[-1] != right[-1]:
        left_bigrams = {left[i : i + 2] for i in range(max(0, len(left) - 1))}
        right_bigrams = {right[i : i + 2] for i in range(max(0, len(right) - 1))}
        if left_bigrams and right_bigrams and not left_bigrams.intersection(right_bigrams):
            return 0.0
    return SequenceMatcher(None, left, right).ratio()


def duration_match_score(local_duration: float | None, catalog_duration: str) -> tuple[float, float | None]:
    if local_duration is None:
        return 0.0, None
    catalog = pd.to_numeric(catalog_duration, errors="coerce")
    if pd.isna(catalog) or catalog <= 0:
        return 0.0, None
    diff = abs(float(local_duration) - float(catalog))
    if diff <= 3:
        return 1.0, diff
    if diff <= 8:
        return 0.86, diff
    if diff <= 20:
        return 0.55, diff
    return 0.0, diff


def build_audio_identity_variants(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    metadata = read_audio_metadata(path)
    file_artist, file_title = parse_audio_filename(path)
    variants = []
    seen = set()
    if metadata.get("title"):
        identity = {
            "source": "metadata",
            "artist": metadata.get("artist", ""),
            "title": metadata.get("title", ""),
        }
        seen.add((normalize_match_text(identity["artist"]), normalize_match_text(identity["title"])))
        variants.append(identity)
    filename_key = (normalize_match_text(file_artist), normalize_match_text(file_title))
    if filename_key not in seen:
        variants.append({"source": "filename", "artist": file_artist, "title": file_title})
    return variants, metadata


def match_audio_files(df: pd.DataFrame, audio_dir: Path, threshold: float, show_progress: bool = True) -> pd.DataFrame:
    if not audio_dir.exists():
        return pd.DataFrame(
            columns=[
                "file_path",
                "song_id",
                "name",
                "artist_names",
                "match_score",
                "match_reason",
                "match_source",
                "audio_title",
                "audio_artist",
                "audio_album",
                "local_duration_seconds",
                "duration_diff_seconds",
                "duration_error",
            ]
        )

    candidates = build_match_candidates(df)
    audio_paths = [
        path for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    rows = []
    for path in progress_iter(audio_paths, total=len(audio_paths), description="匹配本地音频", enabled=show_progress):
        identity_variants, metadata = build_audio_identity_variants(path)

        best_score = 0.0
        best_candidate: dict[str, str] | None = None
        best_reason = ""
        best_source = ""
        best_audio_title = ""
        best_audio_artist = ""
        for candidate in candidates:
            for identity in identity_variants:
                title_norm = normalize_match_text(identity["title"])
                artist_norm = normalize_match_text(identity["artist"])
                combo_norm = normalize_match_text(f"{identity['artist']} {identity['title']}")
                title_score = ratio(title_norm, candidate["title_norm"])
                artist_score = ratio(artist_norm, candidate["artist_norm"]) if artist_norm else 0.0
                combo_score = ratio(combo_norm, candidate["combo_norm"])
                score = max(combo_score * 0.95, title_score * 0.68 + artist_score * 0.32)
                if artist_norm and artist_score < 0.45 and combo_score < 0.86:
                    score *= 0.82
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_source = identity["source"]
                    best_audio_title = identity["title"]
                    best_audio_artist = identity["artist"]
                    best_reason = f"title={title_score:.2f};artist={artist_score:.2f};combo={combo_score:.2f}"

        local_duration: float | None = None
        duration_error = ""
        best_duration_diff: float | None = None
        min_text_score_for_duration = max(0.80, (threshold - 0.18) / 0.82)
        if best_candidate and best_score >= min_text_score_for_duration:
            local_duration, duration_error = get_audio_duration(path)
            duration_score, best_duration_diff = duration_match_score(local_duration, best_candidate["duration_seconds"])
            if duration_score:
                best_score = best_score * 0.82 + duration_score * 0.18
            elif best_duration_diff is not None and best_duration_diff > 20 and best_score < 0.96:
                best_score *= 0.88
            best_reason = f"{best_reason};duration={duration_score:.2f}"

        if best_candidate and best_score >= threshold:
            rows.append(
                {
                    "file_path": str(path),
                    "song_id": best_candidate["song_id"],
                    "name": best_candidate["name"],
                    "artist_names": best_candidate["artist_names"],
                    "match_score": round(best_score, 4),
                    "match_reason": best_reason,
                    "match_source": best_source,
                    "audio_title": best_audio_title,
                    "audio_artist": best_audio_artist,
                    "audio_album": metadata.get("album", ""),
                    "local_duration_seconds": round(local_duration, 3) if local_duration is not None else "",
                    "duration_diff_seconds": round(best_duration_diff, 3) if best_duration_diff is not None else "",
                    "duration_error": duration_error,
                }
            )

    matches = pd.DataFrame(rows)
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "file_path",
                "song_id",
                "name",
                "artist_names",
                "match_score",
                "match_reason",
                "match_source",
                "audio_title",
                "audio_artist",
                "audio_album",
                "local_duration_seconds",
                "duration_diff_seconds",
                "duration_error",
            ]
        )
    return matches.sort_values(["song_id", "match_score"], ascending=[True, False]).drop_duplicates("song_id")


def analyze_audio_file(path: Path, seconds: float, source_separator: dict[str, Any] | None = None, source_seconds: float = 30.0) -> dict[str, Any]:
    try:
        import torch
        import torchaudio
    except Exception as exc:  # pragma: no cover - depends on local install.
        return {"audio_error": f"torchaudio unavailable: {exc}"}

    try:
        info = torchaudio.info(str(path))
        sample_rate = info.sample_rate
        frame_count = min(info.num_frames, int(sample_rate * seconds)) if seconds > 0 else info.num_frames
        waveform, sample_rate = torchaudio.load(str(path), num_frames=frame_count)
        if waveform.numel() == 0:
            return {"audio_error": "empty audio"}
        mono = waveform.float().mean(dim=0)
        if mono.numel() < 2048:
            mono = torch.nn.functional.pad(mono, (0, 2048 - mono.numel()))

        rms = float(torch.sqrt(torch.mean(mono.square()) + 1e-9))
        zcr = float(torch.mean((mono[1:] * mono[:-1] < 0).float())) if mono.numel() > 1 else 0.0
        hop_length = 512
        window = torch.hann_window(2048, device=mono.device)
        spectrum = torch.stft(mono, n_fft=2048, hop_length=hop_length, window=window, return_complex=True).abs()
        freqs = torch.linspace(0, sample_rate / 2, spectrum.size(0))
        centroid = float((spectrum * freqs[:, None]).sum() / (spectrum.sum() + 1e-9))
        vocal_band = (freqs >= 300) & (freqs <= 3400)
        vocal_band_ratio = float(spectrum[vocal_band, :].sum() / (spectrum.sum() + 1e-9))
        peak = float(mono.abs().max())
        crest = float(peak / (rms + 1e-9))
        flux = torch.relu(spectrum[:, 1:] - spectrum[:, :-1]).sum(dim=0)
        onset_strength = float(flux.mean() / (spectrum.mean() + 1e-9)) if flux.numel() else 0.0
        tempo = estimate_tempo_from_onsets(flux, sample_rate, hop_length)

        labels = []
        if tempo is not None:
            if tempo >= 125:
                labels.append("快节奏")
            elif tempo <= 80:
                labels.append("慢节奏")
            else:
                labels.append("中速")
        if rms >= 0.11:
            labels.append("高能量")
        elif rms <= 0.035:
            labels.append("低能量")
        else:
            labels.append("中等能量")
        if centroid >= 3100:
            labels.append("明亮")
        elif centroid <= 1500:
            labels.append("柔和")
        if zcr >= 0.12:
            labels.append("节奏颗粒感")
        if crest >= 8:
            labels.append("动态大")
        if onset_strength >= 220:
            labels.append("律动明显")
        elif onset_strength and onset_strength <= 170:
            labels.append("律动弱")

        result = {
            "audio_duration_seconds": round(info.num_frames / info.sample_rate, 3) if info.sample_rate > 0 else "",
            "audio_sample_rate": sample_rate,
            "audio_rms": round(rms, 6),
            "audio_zcr": round(zcr, 6),
            "audio_centroid_hz": round(centroid, 2),
            "audio_vocal_band_ratio": round(vocal_band_ratio, 4),
            "audio_crest": round(crest, 3),
            "audio_tempo_bpm": round(tempo, 2) if tempo is not None else "",
            "audio_onset_strength": round(onset_strength, 4),
            "audio_feature_tags": " | ".join(labels),
            "audio_error": "",
        }
        if source_separator is not None:
            result.update(estimate_sources_with_separator(path, source_seconds, source_separator))
        return result
    except Exception as exc:  # pragma: no cover - depends on codecs/files.
        return {"audio_error": str(exc)}


def normalize_output_song_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "song_id" not in df.columns:
        return df.copy()
    result = df.copy()
    result["song_id"] = result["song_id"].astype("string").str.strip()
    return result[result["song_id"].fillna("").ne("")]


def merge_existing_output(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = normalize_output_song_ids(df)
    if not path.exists() or path.stat().st_size == 0:
        return result, False

    existing = pd.read_csv(path, dtype={"song_id": "string"})
    existing.columns = [col.strip() for col in existing.columns]
    existing = normalize_output_song_ids(existing)
    columns = list(dict.fromkeys([*existing.columns, *result.columns]))
    merged = pd.concat(
        [existing.reindex(columns=columns), result.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    return merge_latest_nonempty_rows(merged), True


def read_jsonl_frame(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                rows.append(record)
    return pd.DataFrame(rows)


def merge_existing_jsonl_output(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = normalize_output_song_ids(df)
    if not path.exists() or path.stat().st_size == 0:
        return result, False

    existing = read_jsonl_frame(path)
    existing = normalize_output_song_ids(existing)
    columns = list(dict.fromkeys([*existing.columns, *result.columns]))
    merged = pd.concat(
        [existing.reindex(columns=columns), result.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    return merge_latest_nonempty_rows(merged), True


def merge_latest_nonempty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "song_id" not in df.columns:
        return df

    merged = normalize_output_song_ids(df)
    merged["_output_position"] = range(len(merged))
    song_order = (
        merged[["song_id", "_output_position"]]
        .drop_duplicates("song_id", keep="last")
        .sort_values("_output_position", kind="stable")
    )
    result = song_order[["song_id"]].reset_index(drop=True)

    for col in [item for item in merged.columns if item not in {"song_id", "_output_position"}]:
        values = merged[["song_id", col]].copy()
        values[col] = values[col].fillna("").astype(str).str.strip()
        values = values[values[col].ne("")]
        if values.empty:
            continue
        latest_values = values.drop_duplicates("song_id", keep="last")
        result = result.merge(latest_values, on="song_id", how="left")

    return result.fillna("")


def print_append_notice(label: str, path: Path, existing: bool, rows: int) -> None:
    if existing:
        print(f"{label} 已存在，将追加合并而不是覆盖：{path}（合并后 {rows} 行）")


def write_csv_output(path: Path, df: pd.DataFrame, label: str) -> pd.DataFrame:
    output_df, existing = merge_existing_output(path, df)
    print_append_notice(label, path, existing, len(output_df))
    output_df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return output_df


def write_parquet_output(path: Path, df: pd.DataFrame, label: str) -> None:
    output_df = normalize_output_song_ids(df)
    existing = path.exists() and path.stat().st_size > 0
    if existing:
        existing_df = pd.read_parquet(path)
        existing_df.columns = [col.strip() for col in existing_df.columns]
        existing_df = normalize_output_song_ids(existing_df)
        columns = list(dict.fromkeys([*existing_df.columns, *output_df.columns]))
        output_df = pd.concat(
            [existing_df.reindex(columns=columns), output_df.reindex(columns=columns)],
            ignore_index=True,
            sort=False,
        )
        output_df = merge_latest_nonempty_rows(output_df)
    print_append_notice(label, path, existing, len(output_df))
    output_df.to_parquet(path, index=False)


def write_jsonl(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    output_df, existing = merge_existing_jsonl_output(path, df)
    output_df = output_df.fillna("")
    print_append_notice("标签 JSONL", path, existing, len(output_df))
    with path.open("w", encoding="utf-8", newline="") as file:
        for record in output_df.to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_df


def load_mert_audio(mert: Any, path: Path, target_sr: int, max_seconds: float | None) -> tuple[Any, int]:
    waveform, sample_rate = mert.torchaudio.load(str(path))
    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if max_seconds is not None:
        waveform = waveform[:, : int(sample_rate * max_seconds)]
    if sample_rate != target_sr:
        resampler = mert.torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr
    return waveform.squeeze(0).contiguous(), sample_rate


def prepare_mert_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    hf_cache = BASE_DIR / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    import build_mert_emotion as mert

    mert.load_runtime_dependencies()
    device = mert.choose_device(args.mert_device)
    processor = mert.Wav2Vec2FeatureExtractor.from_pretrained(
        str(args.mert_model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = mert.AutoModel.from_pretrained(
        str(args.mert_model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().to(device)
    return mert, processor, model, device


def build_mert_outputs(tags_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    import numpy as np

    rows = tags_df[tags_df["local_audio_path"].astype(str).str.len() > 0].copy()
    if args.mert_limit > 0:
        rows = rows.head(args.mert_limit)
    if rows.empty:
        return pd.DataFrame()

    args.mert_embeddings_dir.mkdir(parents=True, exist_ok=True)
    mert, processor, model, device = prepare_mert_runtime(args)
    index_rows = []

    for _, row in progress_iter(
        rows.iterrows(),
        total=len(rows),
        description="提取 MERT",
        enabled=not getattr(args, "no_progress", False),
    ):
        song_id = safe_text(row.get("song_id"))
        audio_path = Path(safe_text(row.get("local_audio_path")))
        embedding_path = args.mert_embeddings_dir / f"{song_id}.npy"
        record: dict[str, Any] = {
            "song_id": song_id,
            "mert_embedding_path": str(embedding_path),
            "mert_error": "",
        }

        try:
            if embedding_path.exists() and not args.overwrite_mert:
                embedding_np = np.load(embedding_path)
                embedding = mert.torch.tensor(embedding_np, dtype=mert.torch.float32)
                stats = {
                    "chunks": "",
                    "embedding_dim": int(embedding.numel()),
                    "layer": args.mert_layer,
                    "chunk_seconds": args.mert_chunk_seconds,
                    "stride_seconds": args.mert_stride_seconds,
                }
                audio, sample_rate = load_mert_audio(mert, audio_path, processor.sampling_rate, args.mert_max_seconds)
            else:
                audio, sample_rate = load_mert_audio(mert, audio_path, processor.sampling_rate, args.mert_max_seconds)
                embedding, stats = mert.extract_mert_embedding(
                    model=model,
                    processor=processor,
                    audio=audio,
                    sample_rate=sample_rate,
                    device=device,
                    chunk_seconds=args.mert_chunk_seconds,
                    stride_seconds=args.mert_stride_seconds,
                    layer=args.mert_layer,
                    use_fp16=args.mert_fp16,
                )
                embedding_np = embedding.numpy().astype("float32")
                np.save(embedding_path, embedding_np)

            scores, affect = mert.heuristic_scores(audio, sample_rate)
            top_scores = mert.rank_scores(scores, args.mert_top_k)
            emotion_tags = [
                MERT_EMOTION_TAGS.get(str(item["label"]), str(item["label"]))
                for item in top_scores
                if float(item["score"]) >= args.mert_emotion_threshold
            ]
            if not emotion_tags and top_scores:
                emotion_tags = [MERT_EMOTION_TAGS.get(str(top_scores[0]["label"]), str(top_scores[0]["label"]))]

            record.update(
                {
                    "mert_chunks": stats.get("chunks", ""),
                    "mert_embedding_dim": stats.get("embedding_dim", ""),
                    "mert_layer": stats.get("layer", ""),
                    "mert_emotion_tags": " | ".join(dict.fromkeys(emotion_tags)),
                    "mert_emotion_scores": json.dumps(top_scores, ensure_ascii=False),
                    "mert_valence": round(float(affect.get("valence", 0)), 4),
                    "mert_arousal": round(float(affect.get("arousal", 0)), 4),
                }
            )
        except Exception as exc:
            record["mert_error"] = str(exc)
        index_rows.append(record)

    index_df = pd.DataFrame(index_rows)
    cluster_df = build_mert_clusters(index_df, args)
    if not cluster_df.empty:
        index_df = index_df.merge(cluster_df, on="song_id", how="left")

    index_df.to_csv(args.mert_index, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    if not cluster_df.empty:
        cluster_df.to_csv(args.mert_clusters_output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return index_df


def build_mert_clusters(index_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    valid = index_df[
        (index_df["mert_error"].fillna("") == "")
        & (index_df["mert_embedding_path"].fillna("").astype(str).str.len() > 0)
    ].copy()
    if len(valid) < 2:
        return pd.DataFrame()

    embeddings = []
    song_ids = []
    for _, row in progress_iter(
        valid.iterrows(),
        total=len(valid),
        description="读取 MERT 向量",
        enabled=not getattr(args, "no_progress", False),
    ):
        path = Path(safe_text(row.get("mert_embedding_path")))
        if not path.exists():
            continue
        vector = np.load(path).astype("float32")
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        embeddings.append(vector)
        song_ids.append(safe_text(row.get("song_id")))
    if len(embeddings) < 2:
        return pd.DataFrame()

    matrix = np.vstack(embeddings)
    cluster_count = min(args.mert_clusters, len(song_ids))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(matrix)
    neighbor_count = min(args.mert_neighbors + 1, len(song_ids))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine").fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)

    rows = []
    for idx, song_id in enumerate(song_ids):
        neighbor_ids = []
        neighbor_scores = []
        for distance, neighbor_idx in zip(distances[idx], indices[idx], strict=True):
            if neighbor_idx == idx:
                continue
            neighbor_ids.append(song_ids[neighbor_idx])
            neighbor_scores.append(round(1 - float(distance), 4))
            if len(neighbor_ids) >= args.mert_neighbors:
                break
        rows.append(
            {
                "song_id": song_id,
                "mert_cluster": int(labels[idx]),
                "mert_neighbor_song_ids": " | ".join(neighbor_ids),
                "mert_neighbor_scores": " | ".join(str(score) for score in neighbor_scores),
            }
        )
    return pd.DataFrame(rows)


def build_song_tags(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.source_separation:
        args.analyze_audio = True

    df = pd.read_csv(args.input, dtype={"song_id": "string", "artist_ids": "string", "album_id": "string"})
    df.columns = [col.strip() for col in df.columns]

    tag_rows = []
    for _, row in progress_iter(
        df.iterrows(),
        total=len(df),
        description="生成基础标签",
        enabled=not args.no_progress,
    ):
        song_id = safe_text(row.get("song_id"))
        lyric = read_lyric(args.lyrics_dir, song_id)
        tag_row = {
            "song_id": song_id,
            "name": safe_text(row.get("name")),
            "artist_names": safe_text(row.get("artist_names")),
            "has_lyric": safe_text(row.get("has_lyric")),
            **build_tags_for_row(row, lyric),
        }
        tag_rows.append(tag_row)

    tags_df = pd.DataFrame(tag_rows)
    reused_matches = bool(args.reuse_matches and args.matches_output.exists())
    if reused_matches:
        matches_df = pd.read_csv(args.matches_output, dtype={"song_id": "string"})
        matches_df.columns = [col.strip() for col in matches_df.columns]
    else:
        matches_df = match_audio_files(
            df,
            args.audio_dir,
            args.match_threshold,
            show_progress=not args.no_progress,
        )
    if not matches_df.empty:
        if not reused_matches:
            matches_df = write_csv_output(args.matches_output, matches_df, "音频匹配 CSV")
        best_matches = matches_df[
            [
                "song_id",
                "file_path",
                "match_score",
                "match_reason",
                "match_source",
                "audio_title",
                "audio_artist",
                "audio_album",
                "local_duration_seconds",
                "duration_diff_seconds",
            ]
        ].rename(
            columns={
                "file_path": "local_audio_path",
                "match_score": "audio_match_score",
                "match_reason": "audio_match_reason",
                "audio_title": "local_audio_title",
                "audio_artist": "local_audio_artist",
                "audio_album": "local_audio_album",
            }
        )
        tags_df = tags_df.merge(best_matches, on="song_id", how="left")
    else:
        tags_df["local_audio_path"] = ""
        tags_df["audio_match_score"] = ""
        tags_df["audio_match_reason"] = ""
        tags_df["match_source"] = ""
        tags_df["local_audio_title"] = ""
        tags_df["local_audio_artist"] = ""
        tags_df["local_audio_album"] = ""
        tags_df["local_duration_seconds"] = ""
        tags_df["duration_diff_seconds"] = ""

    match_columns = [
        "local_audio_path",
        "audio_match_score",
        "audio_match_reason",
        "match_source",
        "local_audio_title",
        "local_audio_artist",
        "local_audio_album",
        "local_duration_seconds",
        "duration_diff_seconds",
    ]
    tags_df[match_columns] = tags_df[
        match_columns
    ].fillna("")

    has_audio = tags_df["local_audio_path"].astype(str).str.len() > 0
    tags_df.loc[has_audio, "audio_tags"] = tags_df.loc[has_audio, "audio_tags"].apply(
        lambda value: " | ".join([item for item in [safe_text(value), "本地音频"] if item])
    )
    tags_df.loc[has_audio, "all_tags"] = tags_df.loc[has_audio, "all_tags"].apply(
        lambda value: " | ".join([item for item in [safe_text(value), "本地音频"] if item])
    )

    source_separator = None
    source_separator_error = ""
    if args.source_separation:
        try:
            source_separator = build_source_separator(
                args.source_separation_model,
                args.source_separation_device,
                args.source_separation_checkpoint,
            )
        except Exception as exc:
            source_separator_error = str(exc)
            print(f"Source separation unavailable; falling back to heuristic vocal/instrumental tags: {exc}")

    if args.analyze_audio:
        audio_feature_rows = []
        audio_rows = tags_df[has_audio]
        for _, row in progress_iter(
            audio_rows.iterrows(),
            total=len(audio_rows),
            description="分析音频/声源分离",
            enabled=not args.no_progress,
        ):
            features = analyze_audio_file(
                Path(row["local_audio_path"]),
                args.audio_feature_seconds,
                source_separator=source_separator,
                source_seconds=args.source_separation_seconds,
            )
            if args.source_separation and source_separator is None:
                features["source_separation_model"] = args.source_separation_model
                features["source_separation_error"] = source_separator_error or "source separation unavailable"
            features["song_id"] = row["song_id"]
            audio_feature_rows.append(features)
        if audio_feature_rows:
            audio_df = pd.DataFrame(audio_feature_rows)
            audio_df = write_csv_output(args.audio_features_csv, audio_df, "音频特征 CSV")
            try:
                write_parquet_output(args.audio_features_parquet, audio_df, "音频特征 Parquet")
            except Exception as exc:
                print(f"Skipped parquet audio feature output: {exc}")
            tags_df = tags_df.merge(audio_df, on="song_id", how="left")
            feature_mask = tags_df.get("audio_feature_tags", "").fillna("").astype(str).str.len() > 0
            vocal_rows = tags_df.loc[feature_mask].apply(infer_vocal_instrumental_tags, axis=1, result_type="expand")
            for col in ["vocal_presence_score", "instrumental_presence_score", "vocal_instrumental_tags"]:
                tags_df.loc[feature_mask, col] = vocal_rows[col]
            tags_df.loc[feature_mask, "audio_tags"] = tags_df.loc[feature_mask].apply(
                lambda row: " | ".join(
                    dict.fromkeys(
                        split_pipe(row.get("audio_tags"))
                        + split_pipe(row.get("audio_feature_tags"))
                        + split_pipe(row.get("vocal_instrumental_tags"))
                    )
                ),
                axis=1,
            )
            tags_df.loc[feature_mask, "all_tags"] = tags_df.loc[feature_mask].apply(
                lambda row: " | ".join(
                    dict.fromkeys(
                        split_pipe(row.get("all_tags"))
                        + split_pipe(row.get("audio_feature_tags"))
                        + split_pipe(row.get("vocal_instrumental_tags"))
                    )
                ),
                axis=1,
            )
    elif args.audio_features_csv.exists():
        audio_df = pd.read_csv(args.audio_features_csv, dtype={"song_id": "string"})
        audio_df.columns = [col.strip() for col in audio_df.columns]
        tags_df = tags_df.merge(audio_df, on="song_id", how="left")
        feature_mask = tags_df.get("audio_feature_tags", "").fillna("").astype(str).str.len() > 0
        vocal_rows = tags_df.loc[feature_mask].apply(infer_vocal_instrumental_tags, axis=1, result_type="expand")
        for col in ["vocal_presence_score", "instrumental_presence_score", "vocal_instrumental_tags"]:
            tags_df.loc[feature_mask, col] = vocal_rows[col]
        tags_df.loc[feature_mask, "audio_tags"] = tags_df.loc[feature_mask].apply(
            lambda row: " | ".join(
                dict.fromkeys(
                    split_pipe(row.get("audio_tags"))
                    + split_pipe(row.get("audio_feature_tags"))
                    + split_pipe(row.get("vocal_instrumental_tags"))
                )
            ),
            axis=1,
        )
        tags_df.loc[feature_mask, "all_tags"] = tags_df.loc[feature_mask].apply(
            lambda row: " | ".join(
                dict.fromkeys(
                    split_pipe(row.get("all_tags"))
                    + split_pipe(row.get("audio_feature_tags"))
                    + split_pipe(row.get("vocal_instrumental_tags"))
                )
            ),
            axis=1,
        )

    if args.extract_mert:
        mert_df = build_mert_outputs(tags_df, args)
        if not mert_df.empty:
            tags_df = tags_df.merge(mert_df, on="song_id", how="left")
            mert_mask = tags_df.get("mert_emotion_tags", "").fillna("").astype(str).str.len() > 0
            tags_df.loc[mert_mask, "emotion_tags"] = tags_df.loc[mert_mask].apply(
                lambda row: " | ".join(
                    dict.fromkeys(split_pipe(row.get("emotion_tags")) + split_pipe(row.get("mert_emotion_tags")))
                ),
                axis=1,
            )
            tags_df.loc[mert_mask, "all_tags"] = tags_df.loc[mert_mask].apply(
                lambda row: " | ".join(
                    dict.fromkeys(split_pipe(row.get("all_tags")) + split_pipe(row.get("mert_emotion_tags")))
                ),
                axis=1,
            )

    tags_output_df = write_csv_output(args.output, tags_df, "标签 CSV")
    write_jsonl(args.jsonl_output, tags_df)
    return tags_output_df, matches_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explainable tags for local music library.")
    parser.add_argument("--input", type=Path, default=None, help="Song CSV path. Defaults to the first data/source/*.csv.")
    parser.add_argument("--lyrics-dir", type=Path, default=DEFAULT_LYRICS_DIR, help="Lyrics directory.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR, help="Local music directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output song tag CSV. Derived from --input when omitted.")
    parser.add_argument("--jsonl-output", type=Path, default=None, help="Output song tag JSONL. Derived from --input when omitted.")
    parser.add_argument("--matches-output", type=Path, default=None, help="Output audio match CSV. Derived from --input when omitted.")
    parser.add_argument(
        "--audio-features-csv",
        type=Path,
        default=None,
        help="Output audio feature CSV when --analyze-audio is enabled. Derived from --input when omitted.",
    )
    parser.add_argument(
        "--audio-features-parquet",
        type=Path,
        default=None,
        help="Output audio feature parquet when --analyze-audio is enabled. Derived from --input when omitted.",
    )
    parser.add_argument("--match-threshold", type=float, default=0.84, help="Minimum fuzzy match score.")
    parser.add_argument("--reuse-matches", action="store_true", help="Reuse the existing audio match CSV.")
    parser.add_argument("--analyze-audio", action="store_true", help="Extract simple audio features with torchaudio.")
    parser.add_argument(
        "--audio-feature-seconds",
        type=float,
        default=45,
        help="Seconds to load per audio file when --analyze-audio is enabled.",
    )
    parser.add_argument(
        "--source-separation",
        action="store_true",
        help="Use a pretrained HDemucs source-separation model for vocal/instrumental tags.",
    )
    parser.add_argument(
        "--source-separation-model",
        choices=sorted(SOURCE_SEPARATION_MODELS),
        default="hdemucs_high_musdb_plus",
        help="Torchaudio source-separation bundle used when --source-separation is enabled.",
    )
    parser.add_argument(
        "--source-separation-checkpoint",
        type=Path,
        default=None,
        help=(
            "Local HDemucs checkpoint path. If omitted, the script first checks "
            "models/hdemucs_high_trained.pt for hdemucs_high_musdb_plus or "
            "models/hdemucs_high_musdbhq_only.pt for hdemucs_high_musdb."
        ),
    )
    parser.add_argument(
        "--source-separation-seconds",
        type=float,
        default=30.0,
        help="Seconds loaded per song for source separation; lower values are faster.",
    )
    parser.add_argument(
        "--source-separation-device",
        default="auto",
        help="Device for source separation: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument("--extract-mert", action="store_true", help="Extract MERT embeddings for matched local audio.")
    parser.add_argument("--mert-model-dir", type=Path, default=DEFAULT_MERT_MODEL_DIR, help="Local MERT model directory.")
    parser.add_argument("--mert-embeddings-dir", type=Path, default=DEFAULT_MERT_EMBEDDINGS_DIR)
    parser.add_argument("--mert-index", type=Path, default=None)
    parser.add_argument("--mert-clusters-output", type=Path, default=None)
    parser.add_argument("--mert-limit", type=int, default=0, help="Limit MERT extraction rows; 0 means all matched audio.")
    parser.add_argument("--overwrite-mert", action="store_true", help="Recompute existing MERT .npy files.")
    parser.add_argument("--mert-max-seconds", type=float, default=20.0, help="Seconds loaded per song for MERT.")
    parser.add_argument("--mert-chunk-seconds", type=float, default=5.0)
    parser.add_argument("--mert-stride-seconds", type=float, default=5.0)
    parser.add_argument("--mert-layer", default="mean")
    parser.add_argument("--mert-device", default="auto")
    parser.add_argument("--mert-fp16", action="store_true")
    parser.add_argument("--mert-top-k", type=int, default=3)
    parser.add_argument("--mert-emotion-threshold", type=float, default=0.16)
    parser.add_argument("--mert-clusters", type=int, default=12)
    parser.add_argument("--mert-neighbors", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return apply_derived_output_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    args.matches_output.parent.mkdir(parents=True, exist_ok=True)
    args.audio_features_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audio_features_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.mert_embeddings_dir.mkdir(parents=True, exist_ok=True)
    args.mert_index.parent.mkdir(parents=True, exist_ok=True)
    args.mert_clusters_output.parent.mkdir(parents=True, exist_ok=True)
    tags_df, matches_df = build_song_tags(args)
    print(f"Wrote {len(tags_df)} song tag row(s) to {args.output}")
    print(f"Wrote {len(matches_df)} audio match row(s) to {args.matches_output}")


if __name__ == "__main__":
    main()
