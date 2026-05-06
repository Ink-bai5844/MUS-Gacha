import html
import hashlib
import math
import os
import pickle
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ink_bai_liked_songs.csv"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
TAGS_FILE = BASE_DIR / "data" / "song_tags.csv"
CACHE_DIR = BASE_DIR / "datacache"
PREPROCESSED_DATA_FILE = CACHE_DIR / "preprocessed_music.pkl"
PREPROCESSED_HASH_FILE = CACHE_DIR / "data.hash"
PREPROCESS_CACHE_VERSION = "mus-gacha-preprocess-v1"
MAX_DISPLAY = 60

INITIAL_TAG_WEIGHTS = {
    "人声强": 1.4,
    "器乐强": 1.25,
    "热血": 1.25,
    "治愈": 1.2,
    "宁静": 1.15,
    "纯音乐": 1.15,
    "悲伤": 0.8,
}

LYRIC_STOP_WORDS = {
    "作词",
    "作曲",
    "编曲",
    "制作人",
    "演唱",
    "歌词",
    "纯音乐",
    "欣赏",
    "music",
    "lyrics",
    "composer",
    "arranger",
    "the",
    "and",
    "you",
    "that",
    "this",
    "with",
    "for",
    "are",
    "was",
    "were",
    "your",
    "我的",
    "我们",
    "你们",
    "他们",
    "一个",
    "没有",
    "什么",
    "怎么",
    "还是",
    "只是",
    "不要",
    "不能",
    "それ",
    "これ",
    "から",
    "まで",
    "こと",
    "もの",
}

COMMENT_RULES = [
    ("回忆共鸣", ["回忆", "童年", "青春", "小时候", "以前", "当年", "那年", "怀念", "泪目"]),
    ("治愈共鸣", ["治愈", "温暖", "安心", "舒服", "放松", "平静", "温柔"]),
    ("悲伤共鸣", ["流泪", "哭", "泪", "难过", "心酸", "遗憾", "破防", "emo"]),
    ("热血共鸣", ["燃", "热血", "励志", "加油", "力量", "勇气", "坚持"]),
    ("好听认可", ["好听", "神曲", "封神", "循环", "单曲循环", "喜欢", "爱了", "绝了"]),
    ("故事感", ["故事", "人生", "经历", "后来", "想起", "陪伴", "告别"]),
    ("幽默吐槽", ["哈哈", "笑死", "蚌埠", "绷不住", "草", "233", "hhhh"]),
    ("亲情陪伴", ["爸爸", "妈妈", "父亲", "母亲", "家人", "爸妈", "爷爷", "奶奶"]),
    ("影视回忆", ["电视剧", "电影", "动漫", "片头", "片尾", "主题曲", "插曲"]),
]

_JANOME_TOKENIZER = None

GENERATED_TAG_COLUMNS = [
    "generated_language_tags",
    "style_tags",
    "emotion_tags",
    "theme_tags",
    "scene_tags",
    "audio_tags",
    "all_tags",
    "tag_confidence",
    "local_audio_path",
    "local_audio_title",
    "local_audio_artist",
    "local_audio_album",
    "local_duration_seconds",
    "duration_diff_seconds",
    "audio_match_score",
    "audio_tempo_bpm",
    "audio_onset_strength",
    "audio_rms",
    "audio_feature_tags",
    "audio_vocal_band_ratio",
    "vocal_presence_score",
    "instrumental_presence_score",
    "vocal_instrumental_tags",
    "mert_cluster",
    "mert_neighbor_song_ids",
    "mert_emotion_tags",
    "mert_valence",
    "mert_arousal",
    "mert_embedding_path",
]

TEXT_COLUMNS = [
    "name",
    "aliases",
    "translations",
    "artist_names",
    "album_name",
    "lyric_excerpt",
    "translation_excerpt",
    "romaji_excerpt",
    "similar_song_names",
    "similar_artist_names",
    "wiki_summary_excerpt",
    "first_hot_comment",
    "first_comment",
    "style_tags",
    "emotion_tags",
    "theme_tags",
    "scene_tags",
    "audio_tags",
    "all_tags",
]


st.set_page_config(page_title="墨白的音乐仓库", layout="wide")
st.markdown(
    """
    <style>
    div[id^="gdg-overlay-"] {
        margin-left: 96px !important;
        z-index: 99999 !important;
        border-radius: 8px !important;
        box-shadow: 5px 5px 18px rgba(0, 0, 0, 0.32) !important;
        overflow: hidden !important;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: rgba(250, 250, 250, 0.72);
    }
    .song-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .muted-line {
        color: rgba(49, 51, 63, 0.70);
        margin-bottom: 0.55rem;
    }
    .lyric-box {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 1rem;
        max-height: 520px;
        overflow: auto;
        white-space: pre-wrap;
        line-height: 1.75;
        background: rgba(255, 255, 255, 0.68);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_pipe(value):
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def split_tags(value):
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def unique_items(items):
    return list(dict.fromkeys(item for item in items if item))


def extract_title_terms(row):
    original_values = [
        safe_text(row.get("name")),
        safe_text(row.get("album_name")),
        *split_pipe(row.get("aliases")),
        *split_pipe(row.get("translations")),
    ]
    original_norms = {normalize_lyric_token(value) for value in original_values if safe_text(value)}
    text = " ".join(
        [
            safe_text(row.get("name")),
            safe_text(row.get("aliases")),
            safe_text(row.get("translations")),
            safe_text(row.get("album_name")),
        ]
    ).lower()
    language_tags = set(row.get("language_tags", []))
    raw_terms = []
    if {"国语", "粤语", "华语"} & language_tags or re.search(r"[\u4e00-\u9fff]", text):
        raw_terms.extend(tokenize_chinese(text))
    if "日语" in language_tags or re.search(r"[\u3040-\u30ff]", text):
        raw_terms.extend(tokenize_japanese(text))
    if "英语" in language_tags or re.search(r"[a-zA-Z]{3,}", text):
        raw_terms.extend(tokenize_english(text))
    raw_terms.extend(re.split(r"[\s,，。/\\|·・:：;；\-—_()\[\]{}<>《》“”\"'!！?？]+", text))

    terms = []
    for term in raw_terms:
        term = normalize_lyric_token(term)
        if not term or term in LYRIC_STOP_WORDS:
            continue
        if term in original_norms:
            continue
        if re.search(r"[\u4e00-\u9fff]", term) and len(term) > 6:
            continue
        if len(term) >= 2 and len(term) <= 12:
            terms.append(term)
    return unique_items(terms)


def clean_lrc_text(text):
    text = re.sub(r"\[[^\]]+\]", " ", safe_text(text))
    text = re.sub(r"作词|作曲|编曲|制作人|OP|SP|Publisher", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_chinese(text):
    try:
        import jieba

        return jieba.lcut(text)
    except Exception:
        return re.findall(r"[\u4e00-\u9fff]{2,}", text)


def tokenize_japanese(text):
    global _JANOME_TOKENIZER
    try:
        from janome.tokenizer import Tokenizer

        if _JANOME_TOKENIZER is None:
            _JANOME_TOKENIZER = Tokenizer()
        return [token.surface for token in _JANOME_TOKENIZER.tokenize(text)]
    except Exception:
        return re.findall(r"[\u3040-\u30ff\u3400-\u9fff]{2,}", text)


def tokenize_english(text):
    try:
        from nltk.tokenize import wordpunct_tokenize

        return wordpunct_tokenize(text)
    except Exception:
        return re.findall(r"[a-zA-Z][a-zA-Z']+", text)


def normalize_lyric_token(token):
    token = safe_text(token).strip().lower()
    token = re.sub(r"^[^\w\u3040-\u30ff\u3400-\u9fff]+|[^\w\u3040-\u30ff\u3400-\u9fff]+$", "", token)
    return token


def extract_lyric_terms(row, limit=40):
    lyric = clean_lrc_text(row.get("full_lyric") or row.get("lyric_excerpt"))
    if not lyric:
        return []

    language_tags = set(row.get("language_tags", []))
    tokens = []
    if {"国语", "粤语", "华语"} & language_tags or re.search(r"[\u4e00-\u9fff]", lyric):
        tokens.extend(tokenize_chinese(lyric))
    if "日语" in language_tags or re.search(r"[\u3040-\u30ff]", lyric):
        tokens.extend(tokenize_japanese(lyric))
    if "英语" in language_tags or re.search(r"[a-zA-Z]{3,}", lyric):
        tokens.extend(tokenize_english(lyric))

    normalized = []
    for token in tokens:
        token = normalize_lyric_token(token)
        if not token or token in LYRIC_STOP_WORDS:
            continue
        if token.isdigit():
            continue
        if re.fullmatch(r"[a-z]", token):
            continue
        if len(token) < 2:
            continue
        normalized.append(token)

    counter = Counter(normalized)
    return [term for term, _count in counter.most_common(limit)]


def extract_comment_semantic_tags(row):
    text = " ".join(
        [
            safe_text(row.get("first_hot_comment")),
            safe_text(row.get("first_comment")),
        ]
    )
    tags = []
    for tag, needles in COMMENT_RULES:
        if any(needle.lower() in text.lower() for needle in needles):
            tags.append(tag)
    return unique_items(tags)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def read_lyric(song_id):
    path = LYRICS_DIR / f"{song_id}.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").strip()


def normalize_for_search(value):
    return re.sub(r"\s+", " ", safe_text(value).lower())


def open_local_file(path_text):
    path = Path(path_text)
    if not path.exists():
        return False, f"未找到本地文件：{path}"

    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        return False, f"打开失败：{exc}"
    return True, f"已打开：{path.name}"


def build_search_text(row):
    parts = [safe_text(row.get(col, "")) for col in TEXT_COLUMNS]
    parts.append(safe_text(row.get("full_lyric", "")))
    return normalize_for_search(" ".join(parts))


def minmax(series, index=None):
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    max_value = numeric.max()
    if max_value <= 0:
        return numeric * 0
    return numeric / max_value


def preferred_quality(row):
    for prefix, label in [
        ("hires", "Hi-Res"),
        ("lossless", "无损"),
        ("exhigh", "极高"),
        ("standard", "标准"),
    ]:
        url = safe_text(row.get(f"{prefix}_url", ""))
        br = pd.to_numeric(row.get(f"{prefix}_br", 0), errors="coerce")
        if url or (pd.notna(br) and br > 0):
            return label
    return "未知"


def extract_language_tags(row):
    text = safe_text(row.get("wiki_summary_excerpt", ""))
    tags = []
    for candidate in ["国语", "粤语", "英语", "日语", "韩语", "纯音乐", "华语", "欧美"]:
        if candidate in text:
            tags.append(candidate)
    for tag in split_tags(row.get("generated_language_tags", "")):
        if tag not in tags:
            tags.append(tag)
    return tags or ["未知"]


def load_generated_tags():
    if not TAGS_FILE.exists():
        return pd.DataFrame()

    tags_df = pd.read_csv(TAGS_FILE, dtype={"song_id": "string"})
    tags_df.columns = [col.strip() for col in tags_df.columns]
    if "language_tags" in tags_df.columns:
        tags_df = tags_df.rename(columns={"language_tags": "generated_language_tags"})

    keep_columns = ["song_id"] + [col for col in GENERATED_TAG_COLUMNS if col in tags_df.columns]
    return tags_df[keep_columns].drop_duplicates("song_id")


def top_counts(df, column, limit=12):
    values = []
    for items in df[column]:
        values.extend(items)
    if not values:
        return pd.DataFrame({"名称": [], "数量": []})
    counts = pd.Series(values).value_counts().head(limit)
    return counts.rename_axis("名称").reset_index(name="数量")


def filter_by_keywords(df, query):
    terms = [term.strip().lower() for term in re.split(r"[,，\s]+", query or "") if term.strip()]
    if not terms:
        return df

    mask = pd.Series(True, index=df.index)
    for term in terms:
        mask &= df["search_text"].str.contains(term, regex=False, na=False)
    return df[mask]


def frequency_counter(series):
    counter = Counter()
    for items in series:
        counter.update(items)
    return counter


def feature_base_scores(counter):
    return {name: math.log1p(count) * 10.0 for name, count in counter.items()}


def multi_feature_score(series, base_scores, weights=None, default_weight=1.0):
    weights = weights or {}
    scores = []
    for items in series:
        if not items:
            scores.append(0.0)
            continue
        total = 0.0
        seen = Counter(items)
        for item, count in seen.items():
            total += base_scores.get(item, 0.0) * float(weights.get(item, default_weight)) * count
        scores.append(total / math.sqrt(max(len(items), 1)))
    return pd.Series(scores, index=series.index, dtype="float64")


def single_feature_score(series, base_scores, weights=None, default_weight=1.0):
    weights = weights or {}
    scores = [
        base_scores.get(item, 0.0) * float(weights.get(item, default_weight))
        for item in series
    ]
    return pd.Series(scores, index=series.index, dtype="float64")


def build_scoring_resources(df):
    if df.empty:
        return build_empty_scoring_resources()

    def list_series(column):
        if column in df.columns:
            return df[column]
        return pd.Series([[] for _ in range(len(df))], index=df.index, dtype="object")

    return {
        "all_tags": frequency_counter(list_series("generated_tag_list")),
        "language": frequency_counter(list_series("language_tags")),
        "style": frequency_counter(list_series("style_tag_list")),
        "emotion": frequency_counter(list_series("emotion_tag_list")),
        "theme": frequency_counter(list_series("theme_tag_list")),
        "scene": frequency_counter(list_series("scene_tag_list")),
        "audio": frequency_counter(list_series("audio_tag_list")),
        "lyric_terms": frequency_counter(list_series("lyric_terms")),
        "comment_semantic": frequency_counter(list_series("comment_semantic_tags")),
        "artists": frequency_counter(list_series("artist_list")),
        "title_terms": frequency_counter(list_series("title_terms")),
    }


def build_empty_scoring_resources():
    return {
        "all_tags": Counter(),
        "language": Counter(),
        "style": Counter(),
        "emotion": Counter(),
        "theme": Counter(),
        "scene": Counter(),
        "audio": Counter(),
        "lyric_terms": Counter(),
        "comment_semantic": Counter(),
        "artists": Counter(),
        "title_terms": Counter(),
    }


def update_path_signature(hasher, path):
    relative_name = str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)
    hasher.update(relative_name.encode("utf-8"))
    if not path.exists():
        hasher.update(b":missing")
        return

    stat = path.stat()
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))


def get_preprocess_data_hash():
    hasher = hashlib.md5()
    hasher.update(PREPROCESS_CACHE_VERSION.encode("utf-8"))

    for path in [DATA_FILE, TAGS_FILE]:
        update_path_signature(hasher, path)

    if LYRICS_DIR.exists():
        lyric_paths = sorted(LYRICS_DIR.glob("*.txt"), key=lambda item: item.name)
        hasher.update(str(len(lyric_paths)).encode("utf-8"))
        for path in lyric_paths:
            update_path_signature(hasher, path)
    else:
        hasher.update(b"lyrics-dir:missing")

    return hasher.hexdigest()


def normalize_preprocessed_payload(payload):
    if isinstance(payload, dict) and "df" in payload:
        df = payload["df"]
        scoring_resources = payload.get("scoring_resources")
        if scoring_resources is None:
            scoring_resources = build_scoring_resources(df)
        return df, scoring_resources

    if isinstance(payload, pd.DataFrame):
        return payload, build_scoring_resources(payload)

    raise ValueError("预处理缓存文件格式无法识别，请删除 datacache 后重试。")


def apply_dynamic_music_scores(
    df,
    scoring_resources,
    dimension_weights,
    tag_weights,
    artist_weights,
    title_weights,
    lyric_weights,
    comment_weights,
):
    if df.empty:
        result = df.copy()
        result["dynamic_score"] = pd.Series(dtype=float)
        return result

    result = df.copy()
    total = pd.Series(0.0, index=result.index, dtype="float64")
    detail_parts = {}

    feature_specs = [
        ("all_tags", "generated_tag_list", "综合标签", tag_weights, 0.45),
        ("language", "language_tags", "语种", tag_weights, 0.35),
        ("style", "style_tag_list", "风格", tag_weights, 0.55),
        ("emotion", "emotion_tag_list", "情绪", tag_weights, 0.65),
        ("theme", "theme_tag_list", "主题", tag_weights, 0.45),
        ("scene", "scene_tag_list", "场景", tag_weights, 0.45),
        ("audio", "audio_tag_list", "音频标签", tag_weights, 0.60),
    ]

    for resource_key, column, label, weights, scale in feature_specs:
        counter = scoring_resources[resource_key]
        scores = multi_feature_score(result[column], feature_base_scores(counter), weights)
        contribution = scores * float(dimension_weights.get(label, 1.0)) * scale
        detail_parts[label] = contribution
        total += contribution

    artist_scores = multi_feature_score(
        result["artist_list"],
        feature_base_scores(scoring_resources["artists"]),
        artist_weights,
        default_weight=1.0,
    )
    artist_contribution = artist_scores * float(dimension_weights.get("歌手", 1.0)) * 0.85
    detail_parts["歌手"] = artist_contribution
    total += artist_contribution

    title_scores = multi_feature_score(
        result["title_terms"],
        feature_base_scores(scoring_resources["title_terms"]),
        title_weights,
        default_weight=1.0,
    )
    title_contribution = title_scores * float(dimension_weights.get("歌名关键词", 1.0)) * 0.55
    detail_parts["歌名关键词"] = title_contribution
    total += title_contribution

    lyric_scores = multi_feature_score(
        result["lyric_terms"],
        feature_base_scores(scoring_resources["lyric_terms"]),
        lyric_weights,
        default_weight=1.0,
    )
    lyric_contribution = lyric_scores * float(dimension_weights.get("歌词关键词", 1.0)) * 0.50
    detail_parts["歌词关键词"] = lyric_contribution
    total += lyric_contribution

    comment_scores = multi_feature_score(
        result["comment_semantic_tags"],
        feature_base_scores(scoring_resources["comment_semantic"]),
        comment_weights,
        default_weight=1.0,
    )
    comment_contribution = comment_scores * float(dimension_weights.get("评论语义", 1.0)) * 0.75
    detail_parts["评论语义"] = comment_contribution
    total += comment_contribution

    numeric_specs = [
        ("热度", minmax(result["popularity"]) * 35),
        ("歌词完整度", minmax(result["lyric_line_count"]) * 24),
        ("音质", result["quality"].isin(["Hi-Res", "无损"]).astype(float) * 18),
        ("可播放", result["playable"].astype(float) * 12),
        ("本地音频", result["local_audio_path"].astype(str).str.len().gt(0).astype(float) * 18),
        ("MERT", result["mert_embedding_path"].astype(str).str.len().gt(0).astype(float) * 14),
    ]
    for label, scores in numeric_specs:
        contribution = scores * float(dimension_weights.get(label, 1.0))
        detail_parts[label] = contribution
        total += contribution

    result["dynamic_score_raw"] = total
    result["dynamic_score"] = total.round().astype(int)
    top_labels = []
    for idx in result.index:
        ranked = sorted(
            ((label, float(values.loc[idx])) for label, values in detail_parts.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_labels.append(" | ".join(f"{label}:{value:.1f}" for label, value in ranked[:5] if abs(value) >= 0.1))
    result["score_breakdown"] = top_labels
    return result


def build_preprocessed_music_data():
    if not DATA_FILE.exists():
        return pd.DataFrame(), build_empty_scoring_resources()

    df = pd.read_csv(DATA_FILE, dtype={"song_id": "string", "artist_ids": "string", "album_id": "string"})
    df.columns = [col.strip() for col in df.columns]

    for col in TEXT_COLUMNS + ["song_id", "album_pic_url", "duration_text", "publish_date"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in [
        "duration_seconds",
        "popularity",
        "comment_total",
        "hot_comment_count",
        "lyric_line_count",
        "standard_br",
        "exhigh_br",
        "lossless_br",
        "hires_br",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    for col in ["playable", "has_lyric", "has_translation", "has_romaji", "check_success"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_bool)
        else:
            df[col] = False

    df["publish_year"] = pd.to_datetime(df.get("publish_date", ""), errors="coerce").dt.year
    df["duration_minutes"] = df["duration_seconds"] / 60
    df["artist_list"] = df.get("artist_names", "").apply(split_pipe)
    df["quality"] = df.apply(preferred_quality, axis=1)
    df["full_lyric"] = df["song_id"].apply(read_lyric)

    generated_tags = load_generated_tags()
    if not generated_tags.empty:
        df = df.merge(generated_tags, on="song_id", how="left")
    for col in GENERATED_TAG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["language_tags"] = df.apply(extract_language_tags, axis=1)
    df["generated_tag_list"] = df["all_tags"].apply(split_tags)
    df["style_tag_list"] = df["style_tags"].apply(split_tags)
    df["emotion_tag_list"] = df["emotion_tags"].apply(split_tags)
    df["theme_tag_list"] = df["theme_tags"].apply(split_tags)
    df["scene_tag_list"] = df["scene_tags"].apply(split_tags)
    df["audio_tag_list"] = df["audio_tags"].apply(split_tags)
    df["title_terms"] = df.apply(extract_title_terms, axis=1)
    df["lyric_terms"] = df.apply(extract_lyric_terms, axis=1)
    df["comment_semantic_tags"] = df.apply(extract_comment_semantic_tags, axis=1)
    df["search_text"] = df.apply(build_search_text, axis=1)
    df["lyrics_chars"] = df["full_lyric"].str.len()
    df["netease_url"] = "https://music.163.com/#/song?id=" + df["song_id"].astype(str)

    score = (
        minmax(df["popularity"]) * 34
        + minmax(df["comment_total"]).pow(0.45) * 28
        + minmax(df["lyric_line_count"]) * 14
        + df["has_translation"].astype(int) * 8
        + df["playable"].astype(int) * 10
        + df["has_romaji"].astype(int) * 4
        + df["quality"].isin(["Hi-Res", "无损"]).astype(int) * 8
    )
    df["recommend_score"] = score.round().clip(0, 100)
    return df, build_scoring_resources(df)


@st.cache_data(max_entries=1, show_spinner=False)
def load_music_data():
    current_hash = get_preprocess_data_hash()

    if PREPROCESSED_DATA_FILE.exists() and PREPROCESSED_HASH_FILE.exists():
        saved_hash = PREPROCESSED_HASH_FILE.read_text(encoding="utf-8").strip()
        if saved_hash == current_hash:
            print("触发文件级预处理缓存，跳过 CSV/歌词/标签重建。")
            with PREPROCESSED_DATA_FILE.open("rb") as file:
                cached_payload = pickle.load(file)
            df, scoring_resources = normalize_preprocessed_payload(cached_payload)
            if not isinstance(cached_payload, dict) or "scoring_resources" not in cached_payload:
                with PREPROCESSED_DATA_FILE.open("wb") as file:
                    pickle.dump(
                        {"df": df, "scoring_resources": scoring_resources},
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            return df, scoring_resources

    print("预处理缓存失效，重新读取 CSV、歌词与标签。")
    df, scoring_resources = build_preprocessed_music_data()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with PREPROCESSED_DATA_FILE.open("wb") as file:
        pickle.dump(
            {"df": df, "scoring_resources": scoring_resources},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    PREPROCESSED_HASH_FILE.write_text(current_hash, encoding="utf-8")
    return df, scoring_resources


def render_detail(song):
    left, right = st.columns([1.1, 2.4], gap="large")

    with left:
        cover = safe_text(song.get("album_pic_url", ""))
        if cover:
            st.image(cover, width="stretch")
        st.link_button("打开网易云页面", safe_text(song.get("netease_url", "")), width="stretch")

        local_audio_path = safe_text(song.get("local_audio_path", ""))
        if local_audio_path:
            local_path = Path(local_audio_path)
            if local_path.exists():
                st.caption("本地音频")
                st.audio(str(local_path))
                if st.button("打开本地音频", width="stretch", key=f"open-local-{song.get('song_id')}"):
                    ok, message = open_local_file(local_audio_path)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning(f"本地音频路径已失效：{local_audio_path}")
        else:
            audio_url = (
                safe_text(song.get("hires_url", ""))
                or safe_text(song.get("lossless_url", ""))
                or safe_text(song.get("exhigh_url", ""))
                or safe_text(song.get("standard_url", ""))
            )
            if audio_url:
                st.audio(audio_url)

    with right:
        st.markdown(f"<div class='song-title'>{html.escape(safe_text(song.get('name')))}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='muted-line'>"
            f"{html.escape(safe_text(song.get('artist_names')))} · {html.escape(safe_text(song.get('album_name')))}"
            "</div>",
            unsafe_allow_html=True,
        )

        meta_cols = st.columns(4)
        meta_cols[0].metric("推荐分", int(song.get("dynamic_score", song.get("recommend_score", 0))))
        meta_cols[1].metric("热度", int(song.get("popularity", 0)))
        meta_cols[2].metric("评论", int(song.get("comment_total", 0)))
        meta_cols[3].metric("时长", safe_text(song.get("duration_text", "")) or f"{song.get('duration_minutes', 0):.1f} 分")

        info = pd.DataFrame(
            [
                ("发行日期", safe_text(song.get("publish_date"))),
                ("版权/可播", f"{safe_text(song.get('copyright'))} / {'可播' if song.get('playable') else '不可播'}"),
                ("最佳音质", safe_text(song.get("quality"))),
                ("智能标签", safe_text(song.get("all_tags"))),
                ("评分拆解", safe_text(song.get("score_breakdown"))),
                ("歌词关键词", " | ".join(song.get("lyric_terms", [])[:20])),
                ("评论语义", " | ".join(song.get("comment_semantic_tags", []))),
                ("本地音频", safe_text(song.get("local_audio_path"))),
                ("本地元数据", " · ".join([item for item in [
                    safe_text(song.get("local_audio_title")),
                    safe_text(song.get("local_audio_artist")),
                    safe_text(song.get("local_audio_album")),
                ] if item])),
                ("音频特征", " · ".join([item for item in [
                    f"{safe_text(song.get('audio_tempo_bpm'))} BPM" if safe_text(song.get("audio_tempo_bpm")) else "",
                    safe_text(song.get("audio_feature_tags")),
                ] if item])),
                ("人声/器乐", " · ".join([item for item in [
                    safe_text(song.get("vocal_instrumental_tags")),
                    f"人声 {safe_text(song.get('vocal_presence_score'))}" if safe_text(song.get("vocal_presence_score")) else "",
                    f"器乐 {safe_text(song.get('instrumental_presence_score'))}" if safe_text(song.get("instrumental_presence_score")) else "",
                ] if item])),
                ("MERT", " · ".join([item for item in [
                    f"聚类 {safe_text(song.get('mert_cluster'))}" if safe_text(song.get("mert_cluster")) else "",
                    safe_text(song.get("mert_emotion_tags")),
                ] if item])),
                ("歌词行数", str(int(song.get("lyric_line_count", 0)))),
                ("热门评论", safe_text(song.get("first_hot_comment"))),
                ("相似歌曲", safe_text(song.get("similar_song_names"))),
            ],
            columns=["字段", "内容"],
        )
        st.dataframe(info, hide_index=True, width="stretch", height=250)

    lyric = safe_text(song.get("full_lyric", "")) or safe_text(song.get("lyric_excerpt", ""))
    if lyric:
        st.markdown("#### 歌词")
        st.markdown(f"<div class='lyric-box'>{html.escape(lyric)}</div>", unsafe_allow_html=True)
    else:
        st.info("这首歌暂时没有本地歌词。")


with st.spinner("正在同步预处理缓存与评分资源..."):
    df_base, scoring_resources = load_music_data()

st.sidebar.title("筛选与推荐")

if df_base.empty:
    st.error(f"未找到数据文件：{DATA_FILE}")
    st.stop()

search_kw = st.sidebar.text_input("实时检索", placeholder="歌名 / 歌手 / 专辑 / 歌词 / 评论")
lyrics_kw = st.sidebar.text_input("歌词专项检索", placeholder="只在完整歌词里查找")

with st.sidebar.expander("基础筛选", expanded=True):
    all_artists = sorted({artist for artists in df_base["artist_list"] for artist in artists})
    selected_artists = st.multiselect("歌手", options=all_artists, default=[])
    all_languages = sorted({tag for tags in df_base["language_tags"] for tag in tags})
    selected_languages = st.multiselect("语言/场景标签", options=all_languages, default=[])
    all_generated_tags = sorted({tag for tags in df_base["generated_tag_list"] for tag in tags})
    selected_generated_tags = st.multiselect("智能标签", options=all_generated_tags, default=[])
    selected_qualities = st.multiselect("音质", options=["Hi-Res", "无损", "极高", "标准", "未知"], default=[])

    playable_only = st.checkbox("只看可播放", value=True)
    with_lyrics_only = st.checkbox("只看有歌词", value=False)
    with_translation_only = st.checkbox("只看有翻译", value=False)

with st.sidebar.expander("数值范围", expanded=False):
    popularity_min = int(df_base["popularity"].min())
    popularity_max = int(df_base["popularity"].max())
    min_popularity = st.slider("最低热度", popularity_min, popularity_max, popularity_min)

    max_comments = int(df_base["comment_total"].max())
    min_comments = st.slider("最低评论数", 0, max_comments, 0, step=max(1, max_comments // 100))

    valid_years = df_base["publish_year"].dropna().astype(int)
    if valid_years.empty:
        selected_years = (1900, 2030)
    else:
        year_min, year_max = int(valid_years.min()), int(valid_years.max())
        selected_years = st.slider("发行年份", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.subheader("推荐评分配置")

with st.sidebar.expander("全局维度权重", expanded=True):
    dimension_weights = {
        "综合标签": st.slider("综合标签倍率", 0.0, 5.0, 1.0, 0.1),
        "语种": st.slider("语种倍率", 0.0, 5.0, 0.5, 0.1),
        "风格": st.slider("风格倍率", 0.0, 5.0, 1.0, 0.1),
        "情绪": st.slider("情绪倍率", 0.0, 5.0, 1.0, 0.1),
        "主题": st.slider("主题倍率", 0.0, 5.0, 0.8, 0.1),
        "场景": st.slider("场景倍率", 0.0, 5.0, 0.8, 0.1),
        "音频标签": st.slider("音频标签倍率", 0.0, 5.0, 1.0, 0.1),
        "歌手": st.slider("歌手倍率", 0.0, 5.0, 1.0, 0.1),
        "歌名关键词": st.slider("歌名/专辑关键词倍率", 0.0, 5.0, 0.7, 0.1),
        "歌词关键词": st.slider("歌词关键词倍率", 0.0, 5.0, 0.8, 0.1),
        "评论语义": st.slider("评论语义倍率", 0.0, 5.0, 0.8, 0.1),
        "热度": st.slider("热度倍率", 0.0, 5.0, 0.8, 0.1),
        "歌词完整度": st.slider("歌词完整度倍率", 0.0, 5.0, 0.5, 0.1),
        "音质": st.slider("音质倍率", 0.0, 5.0, 0.5, 0.1),
        "可播放": st.slider("可播放倍率", 0.0, 5.0, 0.6, 0.1),
        "本地音频": st.slider("本地音频倍率", 0.0, 5.0, 0.8, 0.1),
        "MERT": st.slider("MERT 可用倍率", 0.0, 5.0, 0.4, 0.1),
    }

all_weight_tags = sorted(scoring_resources["all_tags"].keys())
valid_default_tags = [tag for tag in INITIAL_TAG_WEIGHTS if tag in all_weight_tags]

with st.sidebar.expander("屏蔽标签配置", expanded=False):
    blocked_tags = st.multiselect("选择要屏蔽的标签", options=all_weight_tags, default=[])
    st.caption("命中任意屏蔽标签的歌曲会直接从结果中移除。")

with st.sidebar.expander("单标签权重配置", expanded=True):
    selected_weight_tags = st.multiselect("加权/降权标签列表", options=all_weight_tags, default=valid_default_tags)
    dynamic_tag_weights = {}
    for tag in selected_weight_tags:
        default_value = float(INITIAL_TAG_WEIGHTS.get(tag, 1.0))
        dynamic_tag_weights[tag] = st.number_input(
            f"「{tag}」权重倍率",
            value=default_value,
            step=0.1,
            format="%.1f",
            key=f"tag-weight-{tag}",
        )

with st.sidebar.expander("歌手权重配置", expanded=False):
    selected_weight_artists = st.multiselect("需要单独调整的歌手", options=all_artists, default=[])
    dynamic_artist_weights = {}
    for artist in selected_weight_artists:
        dynamic_artist_weights[artist] = st.number_input(
            f"「{artist}」倍率",
            value=2.0,
            step=0.5,
            format="%.1f",
            key=f"artist-weight-{artist}",
        )

with st.sidebar.expander("歌名关键词权重配置", expanded=False):
    all_title_terms = sorted(scoring_resources["title_terms"].keys())
    selected_title_terms = st.multiselect("关键词列表", options=all_title_terms, default=[])
    dynamic_title_weights = {}
    for term in selected_title_terms:
        dynamic_title_weights[term] = st.number_input(
            f"词汇「{term}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"title-weight-{term}",
        )

with st.sidebar.expander("歌词关键词权重配置", expanded=False):
    all_lyric_terms = sorted(scoring_resources["lyric_terms"].keys())
    selected_lyric_terms = st.multiselect("歌词关键词列表", options=all_lyric_terms, default=[])
    dynamic_lyric_weights = {}
    for term in selected_lyric_terms:
        dynamic_lyric_weights[term] = st.number_input(
            f"歌词词汇「{term}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"lyric-weight-{term}",
        )

with st.sidebar.expander("评论语义权重配置", expanded=False):
    all_comment_tags = sorted(scoring_resources["comment_semantic"].keys())
    selected_comment_tags = st.multiselect("评论语义标签", options=all_comment_tags, default=[])
    dynamic_comment_weights = {}
    for tag in selected_comment_tags:
        dynamic_comment_weights[tag] = st.number_input(
            f"评论语义「{tag}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"comment-weight-{tag}",
        )

scored_df = apply_dynamic_music_scores(
    df_base,
    scoring_resources,
    dimension_weights,
    dynamic_tag_weights,
    dynamic_artist_weights,
    dynamic_title_weights,
    dynamic_lyric_weights,
    dynamic_comment_weights,
)

if blocked_tags:
    scored_df = scored_df[
        ~scored_df["generated_tag_list"].apply(lambda tags: any(tag in tags for tag in blocked_tags))
    ]

if not scored_df.empty:
    min_possible_score = int(scored_df["dynamic_score"].min())
    max_possible_score = int(scored_df["dynamic_score"].max())
else:
    min_possible_score, max_possible_score = 0, 100
if min_possible_score >= max_possible_score:
    max_possible_score = min_possible_score + 1
default_min_score = 0 if min_possible_score <= 0 <= max_possible_score else min_possible_score
min_dynamic_score = st.sidebar.slider(
    "最低推荐评分阈值",
    min_value=min_possible_score,
    max_value=max_possible_score,
    value=default_min_score,
)

filtered_df = scored_df[scored_df["dynamic_score"] >= min_dynamic_score].copy()
filtered_df = filter_by_keywords(filtered_df, search_kw)

if lyrics_kw:
    terms = [term.strip().lower() for term in re.split(r"[,，\s]+", lyrics_kw) if term.strip()]
    lyric_mask = pd.Series(True, index=filtered_df.index)
    lyric_text = filtered_df["full_lyric"].str.lower()
    for term in terms:
        lyric_mask &= lyric_text.str.contains(term, regex=False, na=False)
    filtered_df = filtered_df[lyric_mask]

if selected_artists:
    filtered_df = filtered_df[filtered_df["artist_list"].apply(lambda artists: any(artist in artists for artist in selected_artists))]
if selected_languages:
    filtered_df = filtered_df[filtered_df["language_tags"].apply(lambda tags: any(tag in tags for tag in selected_languages))]
if selected_generated_tags:
    filtered_df = filtered_df[
        filtered_df["generated_tag_list"].apply(lambda tags: any(tag in tags for tag in selected_generated_tags))
    ]
if selected_qualities:
    filtered_df = filtered_df[filtered_df["quality"].isin(selected_qualities)]
if playable_only:
    filtered_df = filtered_df[filtered_df["playable"]]
if with_lyrics_only:
    filtered_df = filtered_df[filtered_df["has_lyric"]]
if with_translation_only:
    filtered_df = filtered_df[filtered_df["has_translation"]]

filtered_df = filtered_df[
    (filtered_df["popularity"] >= min_popularity)
    & (filtered_df["comment_total"] >= min_comments)
    & (
        filtered_df["publish_year"].isna()
        | filtered_df["publish_year"].between(selected_years[0], selected_years[1])
    )
].copy()

st.title("墨白的音乐仓库")

metric_cols = st.columns(5)
metric_cols[0].metric("当前歌曲", f"{len(filtered_df)} / {len(df_base)}")
metric_cols[1].metric("歌手数", len({artist for artists in filtered_df["artist_list"] for artist in artists}))
metric_cols[2].metric("有歌词", int(filtered_df["has_lyric"].sum()))
metric_cols[3].metric("可播放", int(filtered_df["playable"].sum()))
metric_cols[4].metric("平均推荐分", f"{filtered_df['dynamic_score'].mean():.1f}" if not filtered_df.empty else "0.0")

tab_overview, tab_library, tab_lyrics, tab_detail = st.tabs(["推荐总览", "歌曲列表", "歌词检索", "歌曲详情"])

with tab_overview:
    if filtered_df.empty:
        st.info("没有匹配当前筛选条件的歌曲。")
    else:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader("推荐候选")
            top_df = filtered_df.sort_values(["dynamic_score", "comment_total"], ascending=False).head(20)
            st.dataframe(
                top_df[
                    [
                        "album_pic_url",
                        "dynamic_score",
                        "name",
                        "artist_names",
                        "album_name",
                        "all_tags",
                        "score_breakdown",
                        "quality",
                        "duration_text",
                        "comment_total",
                        "netease_url",
                    ]
                ],
                column_config={
                    "album_pic_url": st.column_config.ImageColumn("封面"),
                    "dynamic_score": st.column_config.ProgressColumn(
                        "推荐分",
                        min_value=min_possible_score,
                        max_value=max_possible_score,
                        format="%d",
                    ),
                    "name": "歌曲",
                    "artist_names": "歌手",
                    "album_name": "专辑",
                    "all_tags": "智能标签",
                    "score_breakdown": "评分拆解",
                    "quality": "音质",
                    "duration_text": "时长",
                    "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                    "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
                },
                hide_index=True,
                width="stretch",
                height=620,
            )

        with right:
            st.subheader("分布画像")
            artist_counts = top_counts(filtered_df, "artist_list", 12)
            language_counts = top_counts(filtered_df, "language_tags", 12)
            tag_counts = top_counts(filtered_df, "generated_tag_list", 16)
            quality_counts = filtered_df["quality"].value_counts().rename_axis("音质").reset_index(name="数量")
            year_counts = (
                filtered_df.dropna(subset=["publish_year"])
                .assign(publish_year=lambda item: item["publish_year"].astype(int))
                .groupby("publish_year")
                .size()
                .reset_index(name="数量")
            )

            st.caption("热门歌手")
            st.bar_chart(artist_counts, x="名称", y="数量", height=210)
            st.caption("智能标签")
            st.bar_chart(tag_counts, x="名称", y="数量", height=240)
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.caption("语言/场景")
                st.bar_chart(language_counts, x="名称", y="数量", height=210)
            with chart_cols[1]:
                st.caption("音质")
                st.bar_chart(quality_counts, x="音质", y="数量", height=210)
            st.caption("发行年份")
            st.line_chart(year_counts, x="publish_year", y="数量", height=220)

with tab_library:
    if filtered_df.empty:
        st.info("没有可显示的数据。")
    else:
        sort_columns = {
            "推荐分": "dynamic_score",
            "评论数": "comment_total",
            "热度": "popularity",
            "发行年份": "publish_year",
            "歌名": "name",
            "歌手": "artist_names",
        }
        col_sort, col_order, col_page = st.columns([1.4, 1.0, 1.4])
        with col_sort:
            sort_label = st.selectbox("全局排序依据", options=list(sort_columns), index=0)
        with col_order:
            order_label = st.radio("顺序", ["降序", "升序"], horizontal=True)

        sorted_df = filtered_df.sort_values(
            by=sort_columns[sort_label],
            ascending=(order_label == "升序"),
            na_position="last",
        ).reset_index(drop=True)

        total_items = len(sorted_df)
        total_pages = max(1, math.ceil(total_items / MAX_DISPLAY))
        page_options = []
        for i in range(total_pages):
            start_idx = i * MAX_DISPLAY + 1
            end_idx = min((i + 1) * MAX_DISPLAY, total_items)
            page_options.append(f"{start_idx} ~ {end_idx}")

        with col_page:
            page_label = st.selectbox("显示范围", options=page_options)

        page_index = page_options.index(page_label)
        display_df = sorted_df.iloc[page_index * MAX_DISPLAY : (page_index + 1) * MAX_DISPLAY].copy()

        st.dataframe(
            display_df[
                [
                    "album_pic_url",
                    "dynamic_score",
                    "song_id",
                    "name",
                    "artist_names",
                    "album_name",
                    "all_tags",
                    "score_breakdown",
                    "publish_date",
                    "quality",
                    "duration_text",
                    "popularity",
                    "comment_total",
                    "has_lyric",
                    "has_translation",
                    "netease_url",
                ]
            ],
            column_config={
                "album_pic_url": st.column_config.ImageColumn("封面"),
                "dynamic_score": st.column_config.ProgressColumn(
                    "推荐分",
                    min_value=min_possible_score,
                    max_value=max_possible_score,
                    format="%d",
                ),
                "song_id": "ID",
                "name": "歌曲",
                "artist_names": "歌手",
                "album_name": "专辑",
                "all_tags": "智能标签",
                "score_breakdown": "评分拆解",
                "publish_date": "发行日期",
                "quality": "音质",
                "duration_text": "时长",
                "popularity": st.column_config.NumberColumn("热度", format="%d"),
                "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                "has_lyric": "歌词",
                "has_translation": "翻译",
                "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
            },
            hide_index=True,
            width="stretch",
            height=650,
        )

with tab_lyrics:
    lyric_df = filtered_df[filtered_df["full_lyric"].str.len() > 0].copy()
    if lyric_df.empty:
        st.info("当前筛选范围内没有本地歌词。")
    else:
        lyric_df = lyric_df.sort_values(["dynamic_score", "lyrics_chars"], ascending=False)
        options = {
            f"{row.name} · {row.artist_names} · {row.song_id}": idx
            for idx, row in lyric_df.head(300).iterrows()
        }
        selected_label = st.selectbox("选择歌词", options=list(options.keys()))
        selected_song = lyric_df.loc[options[selected_label]]

        st.markdown(
            f"**{safe_text(selected_song.get('name'))}** · {safe_text(selected_song.get('artist_names'))} · "
            f"{int(selected_song.get('lyric_line_count', 0))} 行"
        )

        lyric = safe_text(selected_song.get("full_lyric", ""))
        if lyrics_kw:
            for term in [term.strip() for term in re.split(r"[,，\s]+", lyrics_kw) if term.strip()]:
                lyric = re.sub(
                    re.escape(term),
                    lambda match: f"<mark>{html.escape(match.group(0))}</mark>",
                    html.escape(lyric),
                    flags=re.IGNORECASE,
                )
            st.markdown(f"<div class='lyric-box'>{lyric}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='lyric-box'>{html.escape(lyric)}</div>", unsafe_allow_html=True)

with tab_detail:
    if filtered_df.empty:
        st.info("先调整筛选条件，选出一首歌。")
    else:
        detail_df = filtered_df.sort_values(["dynamic_score", "comment_total"], ascending=False)
        song_options = {
            f"{row.name} · {row.artist_names} · {row.song_id}": idx
            for idx, row in detail_df.head(500).iterrows()
        }
        selected_song_label = st.selectbox("选择歌曲", options=list(song_options.keys()))
        render_detail(detail_df.loc[song_options[selected_song_label]])
