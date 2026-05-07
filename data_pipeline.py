import hashlib
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BASE_DIR,
    CACHE_DIR,
    GENERATED_TAG_COLUMNS,
    LYRICS_DIR,
    PREPROCESS_CACHE_VERSION,
    PREPROCESSED_DATA_FILE,
    PREPROCESSED_HASH_FILE,
    SOURCE_DATA_DIR,
    TAG_DATA_DIR,
    TEXT_COLUMNS,
)
from utils_core import build_search_text, extract_language_tags, preferred_quality, read_lyric
from utils_text import (
    extract_lyric_terms,
    extract_title_terms,
    parse_bool,
    split_pipe,
    split_tags,
)


def minmax(series, index=None):
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    max_value = numeric.max()
    if max_value <= 0:
        return numeric * 0
    return numeric / max_value


def csv_paths(directory: Path, pattern: str = "*.csv") -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def tag_csv_paths() -> list[Path]:
    return csv_paths(TAG_DATA_DIR, "*song_tags.csv")


def normalize_song_id_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "song_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame.columns = [col.strip() for col in frame.columns]
    if "language_tags" in frame.columns:
        frame = frame.rename(columns={"language_tags": "generated_language_tags"})
    frame["song_id"] = frame["song_id"].astype("string").str.strip()
    return frame[frame["song_id"].fillna("").ne("")]


def merge_song_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    normalized_frames = []
    for source_order, frame in enumerate(frames):
        frame = normalize_song_id_frame(frame)
        if frame.empty:
            continue
        keep_columns = ["song_id"] + [col for col in GENERATED_TAG_COLUMNS if col in frame.columns]
        if len(keep_columns) <= 1:
            continue
        frame = frame[keep_columns].copy()
        frame["_result_source_order"] = source_order
        frame["_result_row_order"] = range(len(frame))
        normalized_frames.append(frame)

    if not normalized_frames:
        return pd.DataFrame()

    merged_source = (
        pd.concat(normalized_frames, ignore_index=True, sort=False)
        .sort_values(["_result_source_order", "_result_row_order"], kind="stable")
        .reset_index(drop=True)
    )
    merged_source["_result_position"] = range(len(merged_source))

    song_order = (
        merged_source[["song_id", "_result_position"]]
        .drop_duplicates("song_id", keep="last")
        .sort_values("_result_position", kind="stable")
    )
    result = song_order[["song_id"]].reset_index(drop=True)

    for col in [item for item in GENERATED_TAG_COLUMNS if item in merged_source.columns]:
        values = merged_source[["song_id", col]].copy()
        values[col] = values[col].fillna("").astype(str).str.strip()
        values = values[values[col].ne("")]
        if values.empty:
            continue
        latest_values = values.drop_duplicates("song_id", keep="last")
        result = result.merge(latest_values, on="song_id", how="left")
        result[col] = result[col].fillna("")

    return result


def merge_generated_tag_sets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return merge_song_frames(frames)


def load_tag_result_frames() -> list[pd.DataFrame]:
    frames = []
    for path in tag_csv_paths():
        frames.append(pd.read_csv(path, dtype={"song_id": "string"}))
    return frames


def load_generated_tags():
    return merge_generated_tag_sets(load_tag_result_frames())


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


def get_source_csv_paths():
    if not SOURCE_DATA_DIR.exists():
        return []
    return sorted(path for path in SOURCE_DATA_DIR.glob("*.csv") if path.is_file())


def load_source_music_csvs():
    source_paths = get_source_csv_paths()
    if not source_paths:
        return pd.DataFrame()

    frames = []
    for source_order, path in enumerate(source_paths):
        frame = pd.read_csv(
            path,
            dtype={"song_id": "string", "artist_ids": "string", "album_id": "string"},
        )
        frame.columns = [col.strip() for col in frame.columns]
        frame["source_csv"] = path.name
        frame["_source_csv_order"] = source_order
        frame["_source_row_order"] = range(len(frame))
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "song_id" not in df.columns:
        return pd.DataFrame()

    df["song_id"] = df["song_id"].astype("string").str.strip()
    df = df[df["song_id"].fillna("").ne("")]
    if df.empty:
        return pd.DataFrame()

    text_df = df.fillna("").astype(str).apply(lambda column: column.str.strip())
    df["_source_row_completeness"] = text_df.ne("").sum(axis=1)
    df = (
        df.sort_values(
            ["song_id", "_source_row_completeness", "_source_csv_order", "_source_row_order"],
            kind="stable",
        )
        .drop_duplicates("song_id", keep="last")
        .sort_values(["_source_csv_order", "_source_row_order"], kind="stable")
        .drop(columns=["_source_csv_order", "_source_row_order", "_source_row_completeness"])
        .reset_index(drop=True)
    )
    return df


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
        "lyric_semantic": frequency_counter(list_series("lyric_semantic_tags")),
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
        "lyric_semantic": Counter(),
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

    source_csv_paths = get_source_csv_paths()
    hasher.update(str(len(source_csv_paths)).encode("utf-8"))
    for path in source_csv_paths:
        update_path_signature(hasher, path)

    result_path_groups = [
        ("tag-csv", tag_csv_paths()),
    ]
    for group_name, paths in result_path_groups:
        hasher.update(group_name.encode("utf-8"))
        hasher.update(str(len(paths)).encode("utf-8"))
        for path in paths:
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
    lyric_semantic_weights,
    comment_weights,
    history_preference=None,
    global_history_w=0.0,
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

    lyric_semantic_scores = multi_feature_score(
        result["lyric_semantic_tags"],
        feature_base_scores(scoring_resources["lyric_semantic"]),
        lyric_semantic_weights,
        default_weight=1.0,
    )
    lyric_semantic_contribution = lyric_semantic_scores * float(dimension_weights.get("歌词语义", 1.0)) * 0.65
    detail_parts["歌词语义"] = lyric_semantic_contribution
    total += lyric_semantic_contribution

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

    if history_preference and float(global_history_w) != 0.0:
        history_total = pd.Series(0.0, index=result.index, dtype="float64")
        history_specs = [
            ("all_tags", "generated_tag_list", 0.45),
            ("language", "language_tags", 0.35),
            ("style", "style_tag_list", 0.55),
            ("emotion", "emotion_tag_list", 0.65),
            ("theme", "theme_tag_list", 0.45),
            ("scene", "scene_tag_list", 0.45),
            ("audio", "audio_tag_list", 0.60),
            ("artists", "artist_list", 0.85),
            ("title_terms", "title_terms", 0.55),
            ("lyric_terms", "lyric_terms", 0.50),
            ("lyric_semantic", "lyric_semantic_tags", 0.65),
            ("comment_semantic", "comment_semantic_tags", 0.75),
        ]
        for resource_key, column, scale in history_specs:
            history_scores = history_preference.get(resource_key, {})
            if not history_scores:
                continue
            history_total += multi_feature_score(result[column], history_scores) * float(scale)
        history_contribution = history_total * float(global_history_w)
        detail_parts["历史偏好"] = history_contribution
        total += history_contribution

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
    df = load_source_music_csvs()
    if df.empty:
        return pd.DataFrame(), build_empty_scoring_resources()

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
    df["lyric_semantic_tags"] = df["lyric_semantic_tags"].apply(split_tags)
    df["comment_semantic_tags"] = df["comment_semantic_tags"].apply(split_tags)
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
def _load_music_data_for_hash(current_hash):
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


def load_music_data():
    return _load_music_data_for_hash(get_preprocess_data_hash())
