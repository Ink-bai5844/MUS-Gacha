from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AUDIO_FEATURE_DATA_DIR,
    CACHE_DIR,
    LYRICS_DIR,
    MERT_DATA_DIR,
    PREPROCESSED_DATA_FILE,
    PREPROCESSED_HASH_FILE,
    SOURCE_DATA_DIR,
    TAG_DATA_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_TIMEOUT = 3600

DEFAULT_JSON_DIR = SOURCE_DATA_DIR / "ink_bai_liked_json"
DEFAULT_DB_FILE = SOURCE_DATA_DIR / "ink_bai_liked.sqlite3"
DEFAULT_LIBRARY_FILE = PROJECT_ROOT / "QCloudMusicApi" / "build" / "QCloudMusicApi" / "QCloudMusicApi.dll"
DEFAULT_COOKIE_FILE = PROJECT_ROOT / "cookie.txt"
DEFAULT_AUDIO_DIR = Path(r"H:\音乐")
DEFAULT_MERT_MODEL_DIR = PROJECT_ROOT / "models" / "MERT-v1-330M"
DEFAULT_MERT_EMBEDDINGS_DIR = MERT_DATA_DIR / "embeddings"
DEFAULT_HDEMUCS_CHECKPOINT = PROJECT_ROOT / "models" / "hdemucs_high_trained.pt"

DEFAULT_RESOLVE_HOSTS = [
    "interface.music.163.com=117.135.207.67",
    "music.163.com=112.29.230.13",
]


def strip_dataset_suffix(name: str) -> str:
    clean = name.strip().replace(" ", "_")
    for suffix in ("_song_tags", "_song_features", "_song_matches", "_songs", "_json"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    return clean or name.strip() or "songs"


def dataset_name_from_path(path_text: str | Path, is_dir: bool = False) -> str:
    path = Path(str(path_text).strip())
    name = path.name if is_dir else path.stem
    if not name:
        name = path.stem or "songs"
    return strip_dataset_suffix(name)


def source_csv_default_for_dataset(dataset: str) -> str:
    return display_path(SOURCE_DATA_DIR / f"{dataset}_songs.csv")


def tag_csv_default_for_dataset(dataset: str) -> str:
    return display_path(TAG_DATA_DIR / f"{dataset}_song_tags.csv")


def tag_jsonl_default_for_dataset(dataset: str) -> str:
    return display_path(TAG_DATA_DIR / f"{dataset}_song_tags.jsonl")


def audio_matches_default_for_dataset(dataset: str) -> str:
    return display_path(AUDIO_FEATURE_DATA_DIR / f"{dataset}_song_matches.csv")


def audio_features_csv_default_for_dataset(dataset: str) -> str:
    return display_path(AUDIO_FEATURE_DATA_DIR / f"{dataset}_song_features.csv")


def audio_features_parquet_default_for_dataset(dataset: str) -> str:
    return display_path(AUDIO_FEATURE_DATA_DIR / f"{dataset}_song_features.parquet")


def mert_index_default_for_dataset(dataset: str) -> str:
    return display_path(MERT_DATA_DIR / f"{dataset}_mert_index.csv")


def mert_clusters_default_for_dataset(dataset: str) -> str:
    return display_path(MERT_DATA_DIR / f"{dataset}_mert_clusters.csv")


def project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path_value: str | Path) -> str:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
    return str(path)


def count_files(path_value: str | Path, pattern: str = "*") -> int:
    path = project_path(path_value)
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


@st.cache_data(show_spinner=False, ttl=600)
def cached_count_files(path_text: str, pattern: str, cache_token: int) -> int:
    return count_files(path_text, pattern)


@st.cache_data(show_spinner=False, ttl=600)
def cached_csv_rows(path_text: str, cache_token: int) -> int:
    path = project_path(path_text)
    if not path.exists() or not path.is_file():
        return 0
    try:
        return int(sum(1 for _ in path.open("r", encoding="utf-8-sig", errors="replace"))) - 1
    except OSError:
        return 0


@st.cache_data(show_spinner=False, ttl=600)
def cached_jsonl_rows(path_text: str, cache_token: int) -> int:
    path = project_path(path_text)
    if not path.exists() or not path.is_file():
        return 0
    try:
        return sum(1 for line in path.open("r", encoding="utf-8-sig", errors="replace") if line.strip())
    except OSError:
        return 0


def source_csv_paths() -> list[Path]:
    if not SOURCE_DATA_DIR.exists():
        return []
    return sorted(path for path in SOURCE_DATA_DIR.glob("*.csv") if path.is_file())


def default_source_csv_path() -> Path:
    paths = source_csv_paths()
    return paths[0] if paths else SOURCE_DATA_DIR / "songs.csv"


def default_source_csv_text() -> str:
    return display_path(default_source_csv_path())


def default_dataset_name() -> str:
    return dataset_name_from_path(default_source_csv_path())


def default_tag_csv_text() -> str:
    return tag_csv_default_for_dataset(default_dataset_name())


def default_tag_jsonl_text() -> str:
    return tag_jsonl_default_for_dataset(default_dataset_name())


def default_audio_matches_text() -> str:
    return audio_matches_default_for_dataset(default_dataset_name())


def default_audio_features_csv_text() -> str:
    return audio_features_csv_default_for_dataset(default_dataset_name())


def default_audio_features_parquet_text() -> str:
    return audio_features_parquet_default_for_dataset(default_dataset_name())


def default_mert_index_text() -> str:
    return mert_index_default_for_dataset(default_dataset_name())


def default_mert_clusters_text() -> str:
    return mert_clusters_default_for_dataset(default_dataset_name())


def tag_csv_paths() -> list[Path]:
    if not TAG_DATA_DIR.exists():
        return []
    return sorted(path for path in TAG_DATA_DIR.glob("*.csv") if path.is_file())


def tag_jsonl_paths() -> list[Path]:
    if not TAG_DATA_DIR.exists():
        return []
    return sorted(path for path in TAG_DATA_DIR.glob("*.jsonl") if path.is_file())


def audio_match_csv_paths() -> list[Path]:
    if not AUDIO_FEATURE_DATA_DIR.exists():
        return []
    return sorted(path for path in AUDIO_FEATURE_DATA_DIR.glob("*audio_song_matches*.csv") if path.is_file())


def audio_feature_csv_paths() -> list[Path]:
    if not AUDIO_FEATURE_DATA_DIR.exists():
        return []
    return sorted(path for path in AUDIO_FEATURE_DATA_DIR.glob("*audio_features*.csv") if path.is_file())


def mert_csv_paths() -> list[Path]:
    if not MERT_DATA_DIR.exists():
        return []
    paths = [
        *MERT_DATA_DIR.glob("*mert_index*.csv"),
        *MERT_DATA_DIR.glob("*mert_clusters*.csv"),
    ]
    return sorted(path for path in dict.fromkeys(paths) if path.is_file())


@st.cache_data(show_spinner=False, ttl=600)
def cached_source_csv_rows(source_dir_text: str, cache_token: int) -> int:
    source_dir = project_path(source_dir_text)
    if not source_dir.exists():
        return 0
    total = 0
    for path in sorted(source_dir.glob("*.csv")):
        total += max(0, cached_csv_rows(display_path(path), cache_token))
    return total


def summed_csv_rows(paths: list[Path]) -> int:
    return sum(max(0, cached_csv_rows(display_path(path), stats_token())) for path in paths)


def summed_jsonl_rows(paths: list[Path]) -> int:
    return sum(max(0, cached_jsonl_rows(display_path(path), stats_token())) for path in paths)


def stats_loaded() -> bool:
    return bool(st.session_state.get("data_processing_stats_loaded", False))


def stats_token() -> int:
    return int(st.session_state.get("data_processing_stats_token", 0))


def get_optional_count(path_value: str | Path, pattern: str = "*") -> str:
    if not stats_loaded():
        return "未统计"
    return str(cached_count_files(display_path(path_value), pattern, stats_token()))


def get_optional_csv_rows(path_value: str | Path) -> str:
    if not stats_loaded():
        return "未统计"
    return str(max(0, cached_csv_rows(display_path(path_value), stats_token())))


def get_optional_source_csv_rows() -> str:
    if not stats_loaded():
        return "未统计"
    return str(max(0, cached_source_csv_rows(display_path(SOURCE_DATA_DIR), stats_token())))


def get_optional_tag_result_rows() -> str:
    if not stats_loaded():
        return "未统计"
    return str(summed_csv_rows(tag_csv_paths()) + summed_jsonl_rows(tag_jsonl_paths()))


def get_optional_audio_result_rows() -> str:
    if not stats_loaded():
        return "未统计"
    return str(summed_csv_rows(audio_match_csv_paths()) + summed_csv_rows(audio_feature_csv_paths()))


def path_exists_text(path_value: str | Path) -> str:
    return "存在" if project_path(path_value).exists() else "不存在"


def path_size_text(path_value: str | Path) -> str:
    path = project_path(path_value)
    if not path.exists() or not path.is_file():
        return "0 KB"
    size = path.stat().st_size
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024:.1f} KB"


def result_key(key: str) -> str:
    return f"data-process-result-{key}"


def run_command(command: list[str], timeout: int = DEFAULT_TIMEOUT, live_output=None) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        return {
            "command": command,
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "run_id": str(time.time_ns()),
        }

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            if process.stdout is None:
                return
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    output_parts: list[str] = []
    timed_out = False
    reader_done = False
    started_at = time.monotonic()
    last_rendered_text = ""

    if live_output is not None:
        live_output.code("实时输出\n\n等待脚本输出...", language="text")

    while True:
        if timeout and time.monotonic() - started_at > int(timeout) and process.poll() is None:
            timed_out = True
            process.kill()
            output_parts.append(f"\n[timeout] 任务超过 {int(timeout)} 秒，已停止等待。\n")

        try:
            item = output_queue.get(timeout=0.1)
        except queue.Empty:
            item = ""

        if item is None:
            reader_done = True
        elif item:
            output_parts.append(item)

        current_text = "".join(output_parts).strip() or "等待脚本输出..."
        if live_output is not None and current_text != last_rendered_text:
            live_output.code(f"实时输出\n\n{current_text}", language="text")
            last_rendered_text = current_text

        if reader_done and process.poll() is not None:
            break
        if timed_out and process.poll() is not None:
            break

    returncode = process.wait()
    reader.join(timeout=1)

    return {
        "command": command,
        "returncode": returncode,
        "stdout": "".join(output_parts).strip(),
        "stderr": "",
        "timed_out": timed_out,
        "run_id": str(time.time_ns()),
    }


def render_result(key: str, empty_label: str | None = None) -> None:
    result = st.session_state.get(result_key(key))
    if not result:
        if empty_label:
            st.info(empty_label)
        return

    command_text = " ".join(str(part) for part in result["command"])
    with st.expander("脚本输出", expanded=True):
        st.code(command_text, language="powershell")
        if result["timed_out"]:
            st.warning("任务超时，已停止等待。")
        elif result["returncode"] == 0:
            st.success("任务完成。")
        else:
            st.error(f"任务退出码：{result['returncode']}")

        output_parts = []
        if result["stdout"]:
            output_parts.append(result["stdout"])
        if result["stderr"]:
            output_parts.append("[stderr]\n" + result["stderr"])
        output_text = "\n\n".join(output_parts) or "脚本没有输出。"
        output_key = f"{key}-script-output-{result.get('run_id', 'legacy')}"
        st.text_area("输出内容", output_text, height=320, key=output_key)


def save_inline_result(key: str, command_label: str, stdout: str, returncode: int = 0, stderr: str = "") -> None:
    st.session_state[result_key(key)] = {
        "command": [command_label],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "run_id": str(time.time_ns()),
    }


def submit_subprocess(
    button_label: str,
    key: str,
    command: list[str],
    timeout: int,
    disabled: bool = False,
    require_confirm: bool = True,
) -> None:
    if st.form_submit_button(button_label, width="stretch", disabled=disabled):
        if not require_confirm:
            st.warning("请先勾选确认项。")
            return
        with st.spinner("正在执行..."):
            live_output = st.empty()
            st.session_state[result_key(key)] = run_command(
                command,
                timeout=timeout,
                live_output=live_output,
            )
            live_output.empty()


def add_arg(command: list[str], flag: str, value: object) -> None:
    text = str(value).strip()
    if text:
        command.extend([flag, text])


def add_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def confirm_json_snapshot_defaults(input_key: str, output_key: str, lyrics_key: str) -> None:
    dataset = dataset_name_from_path(
        st.session_state.get(input_key, display_path(DEFAULT_JSON_DIR)),
        is_dir=True,
    )
    st.session_state[output_key] = source_csv_default_for_dataset(dataset)
    st.session_state[lyrics_key] = display_path(LYRICS_DIR)


def confirm_song_table_defaults(
    input_key: str,
    output_key: str,
    jsonl_key: str,
    matches_key: str,
    audio_features_csv_key: str | None = None,
    audio_features_parquet_key: str | None = None,
    mert_index_key: str | None = None,
    mert_clusters_key: str | None = None,
) -> None:
    dataset = dataset_name_from_path(st.session_state.get(input_key, default_source_csv_text()))
    st.session_state[output_key] = tag_csv_default_for_dataset(dataset)
    st.session_state[jsonl_key] = tag_jsonl_default_for_dataset(dataset)
    st.session_state[matches_key] = audio_matches_default_for_dataset(dataset)
    if audio_features_csv_key:
        st.session_state[audio_features_csv_key] = audio_features_csv_default_for_dataset(dataset)
    if audio_features_parquet_key:
        st.session_state[audio_features_parquet_key] = audio_features_parquet_default_for_dataset(dataset)
    if mert_index_key:
        st.session_state[mert_index_key] = mert_index_default_for_dataset(dataset)
    if mert_clusters_key:
        st.session_state[mert_clusters_key] = mert_clusters_default_for_dataset(dataset)


def text_input_with_confirm(
    label: str,
    default_value: str,
    key: str,
    on_confirm,
    confirm_args: tuple,
) -> str:
    input_col, confirm_col = st.columns([5, 1])
    with input_col:
        value = session_text_input(label, default_value, key)
    with confirm_col:
        st.markdown("<div style='height: 1.85rem'></div>", unsafe_allow_html=True)
        st.form_submit_button(
            "确认",
            width="stretch",
            key=f"{key}-confirm",
            on_click=on_confirm,
            args=confirm_args,
        )
    return value


def source_csv_options(default_value: str | Path) -> list[str]:
    options = [display_path(path) for path in source_csv_paths()]
    default_text = str(default_value)
    current_values = [
        str(st.session_state.get(key, "")).strip()
        for key in ("basic-input", "audio-input", "mert-input")
    ]
    for item in [default_text, *current_values]:
        if item and item not in options:
            options.append(item)
    return options or [default_text]


def source_csv_select_with_confirm(
    label: str,
    default_value: str,
    key: str,
    on_confirm,
    confirm_args: tuple,
) -> str:
    options = source_csv_options(default_value)
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[0]

    input_col, confirm_col = st.columns([5, 1])
    with input_col:
        value = st.selectbox(label, options, key=key)
    with confirm_col:
        st.markdown("<div style='height: 1.85rem'></div>", unsafe_allow_html=True)
        st.form_submit_button(
            "确认",
            width="stretch",
            key=f"{key}-confirm",
            on_click=on_confirm,
            args=confirm_args,
        )
    return value


def session_text_input(label: str, default_value: str, key: str, **kwargs) -> str:
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.text_input(label, key=key, **kwargs)


def add_resolve_hosts(command: list[str], hosts_text: str) -> None:
    for host in re.split(r"[\n,，]+", hosts_text):
        host = host.strip()
        if host:
            command.extend(["--resolve-host", host])


def normalized_song_ids(raw_text: str) -> list[str]:
    return [item for item in re.split(r"[\s,，]+", raw_text.strip()) if item]


def endpoint_text(option: str, custom_value: str) -> str:
    if option.startswith("默认"):
        return "default"
    if option.startswith("全部"):
        return "all"
    return custom_value.strip() or "default"


def levels_text(levels: list[str]) -> str:
    return ",".join(levels) if levels else "standard"


def render_overview() -> None:
    col_refresh, col_hint = st.columns([1, 4])
    with col_refresh:
        if st.button("刷新统计", width="stretch", key="data-processing-refresh-stats"):
            st.session_state["data_processing_stats_loaded"] = True
            st.session_state["data_processing_stats_token"] = stats_token() + 1
    with col_hint:
        st.caption("大目录和 CSV 行数默认不自动扫描，点刷新后再统计。")

    status_cols = st.columns(6)
    status_cols[0].metric("JSON 快照", get_optional_count(DEFAULT_JSON_DIR, "*.json"))
    status_cols[1].metric("歌词 TXT", get_optional_count(LYRICS_DIR, "*.txt"))
    status_cols[2].metric("源 CSV 总行", get_optional_source_csv_rows())
    status_cols[3].metric("标签结果行", get_optional_tag_result_rows())
    status_cols[4].metric("音频结果行", get_optional_audio_result_rows())
    status_cols[5].metric("MERT 向量", get_optional_count(DEFAULT_MERT_EMBEDDINGS_DIR, "*.npy"))

    cache_rows = [
        ("源 CSV 目录", SOURCE_DATA_DIR),
        ("首个源 CSV", default_source_csv_path()),
        ("原始 JSON 目录", DEFAULT_JSON_DIR),
        ("歌词目录", LYRICS_DIR),
        ("标签目录", TAG_DATA_DIR),
        ("推导标签 CSV", default_tag_csv_text()),
        ("推导标签 JSONL", default_tag_jsonl_text()),
        ("音频特征目录", AUDIO_FEATURE_DATA_DIR),
        ("推导本地音频匹配", default_audio_matches_text()),
        ("推导音频特征 CSV", default_audio_features_csv_text()),
        ("推导 MERT 索引", default_mert_index_text()),
        ("推导 MERT 聚类", default_mert_clusters_text()),
        ("SQLite 原始库", DEFAULT_DB_FILE),
        ("预处理 DataFrame", PREPROCESSED_DATA_FILE),
        ("预处理 Hash", PREPROCESSED_HASH_FILE),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "项目": name,
                    "路径": display_path(path),
                    "状态": path_exists_text(path),
                    "大小": path_size_text(path),
                }
                for name, path in cache_rows
            ]
        ),
        hide_index=True,
        width="stretch",
        height=420,
    )


def render_collection_tools() -> None:
    with st.expander("网易云歌曲采集", expanded=True):
        source_mode = st.selectbox(
            "来源",
            ["喜欢歌单名", "歌单 ID", "歌曲 ID 列表", "用户公开歌单"],
            key="qcloud-source-mode",
        )
        with st.form("process-qcloud-collect"):
            song_ids_raw = ""
            my_playlist_name = ""
            playlist_id = 0
            user_playlist_uid = 0
            user_playlist_name = ""

            if source_mode == "喜欢歌单名":
                my_playlist_name = st.text_input("喜欢歌单名", "Ink_bai喜欢的音乐")
            elif source_mode == "歌单 ID":
                playlist_id = st.number_input("歌单 ID", min_value=0, value=0, step=1)
            elif source_mode == "歌曲 ID 列表":
                song_ids_raw = st.text_area("歌曲 ID", placeholder="多个 ID 可用逗号、空格或换行分隔", height=90)
            else:
                user_playlist_uid = st.number_input("用户 UID", min_value=0, value=0, step=1)
                user_playlist_name = st.text_input("公开歌单名")

            json_dir = st.text_input("JSON 快照目录", display_path(DEFAULT_JSON_DIR))
            db_path = st.text_input("SQLite 输出", display_path(DEFAULT_DB_FILE))
            library_path = st.text_input("QCloudMusicApi 动态库", display_path(DEFAULT_LIBRARY_FILE))
            cookie_file = st.text_input("Cookie 文件", display_path(DEFAULT_COOKIE_FILE))

            endpoint_option = st.selectbox("接口集合", ["默认 default", "全部 all", "自定义"])
            custom_endpoints = ""
            if endpoint_option == "自定义":
                custom_endpoints = st.text_input(
                    "接口名",
                    "check_music,lyric_new,lyric,song_music_detail,comment_music,simi_song,song_url_v1",
                )
            selected_levels = st.multiselect(
                "音质层级",
                ["standard", "exhigh", "lossless", "hires"],
                default=["standard", "exhigh", "lossless", "hires"],
            )

            col_limits_1, col_limits_2, col_limits_3 = st.columns(3)
            with col_limits_1:
                max_songs = st.number_input("最多歌曲数（0 为不限）", min_value=0, value=0, step=1)
                workers = st.number_input("并发歌曲数", min_value=1, max_value=1000, value=1, step=1)
            with col_limits_2:
                comment_limit = st.number_input("评论条数", min_value=0, max_value=200, value=20, step=1)
                simi_limit = st.number_input("相似歌曲条数", min_value=0, max_value=200, value=20, step=1)
            with col_limits_3:
                sleep_seconds = st.number_input("接口间隔秒数", min_value=0.0, max_value=60.0, value=1.0, step=0.05)
                timeout = st.number_input("超时秒数", min_value=10, max_value=86400, value=3600, step=60)

            use_default_hosts = st.checkbox("使用默认网易云 Host 映射", value=True)
            extra_hosts = st.text_area(
                "额外 Host 映射",
                "",
                placeholder="interface.music.163.com=117.135.207.67",
                height=70,
            )
            confirm = st.checkbox("确认执行采集脚本", value=False)

            command = [PYTHON, str(PROJECT_ROOT / "data_get" / "qcloud_song_store.py")]
            if source_mode == "喜欢歌单名":
                add_arg(command, "--my-playlist-name", my_playlist_name)
            elif source_mode == "歌单 ID":
                add_arg(command, "--playlist-id", playlist_id)
            elif source_mode == "歌曲 ID 列表":
                command.extend(normalized_song_ids(song_ids_raw))
            else:
                add_arg(command, "--user-playlist-uid", user_playlist_uid)
                add_arg(command, "--playlist-name", user_playlist_name)

            add_arg(command, "--json-dir", json_dir)
            add_arg(command, "--db", db_path)
            add_arg(command, "--library", library_path)
            add_arg(command, "--cookie-file", cookie_file)
            add_arg(command, "--endpoints", endpoint_text(endpoint_option, custom_endpoints))
            add_arg(command, "--levels", levels_text(selected_levels))
            add_arg(command, "--workers", int(workers))
            add_arg(command, "--sleep", sleep_seconds)
            add_arg(command, "--comment-limit", int(comment_limit))
            add_arg(command, "--simi-limit", int(simi_limit))
            if int(max_songs) > 0:
                add_arg(command, "--max-songs", int(max_songs))
            if use_default_hosts:
                add_resolve_hosts(command, "\n".join(DEFAULT_RESOLVE_HOSTS))
            add_resolve_hosts(command, extra_hosts)

            submit_subprocess("开始采集", "qcloud-collect", command, int(timeout), require_confirm=confirm)

        render_result("qcloud-collect", "脚本输出会显示在这里。")

    with st.expander("操作频繁重试", expanded=False):
        with st.form("process-rate-limit-retry"):
            json_dir = st.text_input("JSON 快照目录", display_path(DEFAULT_JSON_DIR), key="retry-json-dir")
            db_path = st.text_input("SQLite 输出", display_path(DEFAULT_DB_FILE), key="retry-db")
            library_path = st.text_input("QCloudMusicApi 动态库", display_path(DEFAULT_LIBRARY_FILE), key="retry-library")
            cookie_file = st.text_input("Cookie 文件", display_path(DEFAULT_COOKIE_FILE), key="retry-cookie")
            marker = st.text_input("限流标记", "操作频繁")
            endpoints = st.text_input("接口集合", "default")
            levels = st.text_input("音质层级", "standard,exhigh,lossless,hires")

            col_retry_1, col_retry_2, col_retry_3 = st.columns(3)
            with col_retry_1:
                workers = st.number_input("并发歌曲数", 1, 1000, 1, 1, key="retry-workers")
                batch_size = st.number_input("批大小", 1, 1000, 20, 1)
            with col_retry_2:
                sleep_seconds = st.number_input("接口间隔秒数", 0.0, 60.0, 1.0, 0.1, key="retry-sleep")
                max_songs = st.number_input("最多重试歌曲（0 为不限）", 0, 100000, 0, 1)
            with col_retry_3:
                timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="retry-timeout")
                keep_going = st.checkbox("批次失败后继续", value=False)

            use_default_hosts = st.checkbox("使用默认网易云 Host 映射", value=True, key="retry-default-hosts")
            extra_hosts = st.text_area("额外 Host 映射", "", height=70, key="retry-extra-hosts")
            dry_run = st.checkbox("只扫描不重抓", value=True)
            confirm = st.checkbox("确认执行重试脚本", value=False)

            command = [
                PYTHON,
                str(PROJECT_ROOT / "data_get" / "retry_rate_limited_songs.py"),
                "--json-dir",
                str(project_path(json_dir)),
                "--db",
                str(project_path(db_path)),
                "--library",
                str(project_path(library_path)),
                "--cookie-file",
                str(project_path(cookie_file)),
                "--marker",
                marker,
                "--endpoints",
                endpoints,
                "--levels",
                levels,
                "--workers",
                str(int(workers)),
                "--sleep",
                str(float(sleep_seconds)),
                "--batch-size",
                str(int(batch_size)),
            ]
            if int(max_songs) > 0:
                command.extend(["--max-songs", str(int(max_songs))])
            if dry_run:
                command.append("--dry-run")
            if keep_going:
                command.append("--keep-going")
            if not use_default_hosts:
                command.append("--no-default-resolve-host")
            add_resolve_hosts(command, extra_hosts)

            submit_subprocess(
                "扫描/重试限流歌曲",
                "rate-limit-retry",
                command,
                int(timeout),
                require_confirm=confirm or dry_run,
            )

        render_result("rate-limit-retry", "脚本输出会显示在这里。")


def render_csv_tools() -> None:
    with st.expander("JSON 快照导出 CSV/歌词", expanded=True):
        with st.form("process-export-json"):
            input_dir = text_input_with_confirm(
                "JSON 快照目录",
                display_path(DEFAULT_JSON_DIR),
                "export-json-input-dir",
                confirm_json_snapshot_defaults,
                ("export-json-input-dir", "export-json-output", "export-json-lyrics"),
            )
            output_file = session_text_input("输出 CSV", default_source_csv_text(), key="export-json-output")
            lyrics_dir = session_text_input("歌词输出目录", display_path(LYRICS_DIR), key="export-json-lyrics")
            include_lyrics = st.checkbox("CSV 内包含完整歌词列", value=False)
            keep_no_copyright = st.checkbox("保留暂无版权歌曲", value=False)
            write_lyrics_files = st.checkbox("写出独立歌词 TXT", value=True)
            timeout = st.number_input("超时秒数", 10, 14400, 1200, 10, key="export-json-timeout")

            command = [
                PYTHON,
                str(PROJECT_ROOT / "data_processing" / "export_original_json_to_csv.py"),
                "--input-dir",
                input_dir,
                "--output",
                output_file,
                "--lyrics-dir",
                lyrics_dir,
            ]
            add_flag(command, "--include-lyrics", include_lyrics)
            add_flag(command, "--keep-no-copyright", keep_no_copyright)
            add_flag(command, "--no-lyrics-files", not write_lyrics_files)

            submit_subprocess("导出 CSV/歌词", "export-json", command, int(timeout))

        render_result("export-json", "脚本输出会显示在这里。")

    with st.expander("CSV 快速预览", expanded=False):
        preview_options = {
            "首个源 CSV": default_source_csv_path(),
            "推导 MERT 索引": project_path(default_mert_index_text()),
            "推导 MERT 聚类": project_path(default_mert_clusters_text()),
        }
        for path in source_csv_paths():
            preview_options[f"源 CSV / {path.name}"] = path
        for path in tag_csv_paths():
            preview_options[f"标签 CSV / {path.name}"] = path
        for path in audio_match_csv_paths():
            preview_options[f"音频匹配 / {path.name}"] = path
        for path in audio_feature_csv_paths():
            preview_options[f"音频特征 / {path.name}"] = path
        for path in mert_csv_paths():
            preview_options[f"MERT / {path.name}"] = path
        selected_name = st.selectbox("文件", list(preview_options), key="preview-csv-name")
        rows = st.number_input("预览行数", 5, 200, 20, 5, key="preview-csv-rows")
        preview_path = project_path(preview_options[selected_name])
        if not preview_path.exists():
            st.info(f"文件不存在：{display_path(preview_path)}")
        elif st.button("读取预览", width="stretch", key="preview-csv-button"):
            try:
                preview_df = pd.read_csv(preview_path, nrows=int(rows), dtype=str)
            except Exception as exc:
                st.error(f"读取失败：{exc}")
            else:
                st.dataframe(preview_df, hide_index=True, width="stretch")


def base_tag_command(
    input_csv: str,
    lyrics_dir: str,
    audio_dir: str,
    output_csv: str,
    jsonl_output: str,
    matches_output: str,
    audio_features_csv: str = "",
) -> list[str]:
    command = [
        PYTHON,
        str(PROJECT_ROOT / "data_processing" / "build_song_tags.py"),
        "--input",
        input_csv,
        "--lyrics-dir",
        lyrics_dir,
        "--audio-dir",
        audio_dir,
        "--output",
        output_csv,
        "--jsonl-output",
        jsonl_output,
        "--matches-output",
        matches_output,
    ]
    add_arg(command, "--audio-features-csv", audio_features_csv)
    return command


def warn_existing_outputs(outputs: list[tuple[str, str]]) -> None:
    resolved_targets: dict[Path, list[str]] = {}
    for label, path_text in outputs:
        if not str(path_text).strip():
            continue
        path = project_path(path_text)
        resolved_targets.setdefault(path.resolve(), []).append(label)
        if path.exists():
            st.warning(f"{label} 已存在：{display_path(path)}。本次会追加合并，不会覆盖旧数据。")

    duplicate_targets = {
        path: labels for path, labels in resolved_targets.items() if len(labels) > 1
    }
    for path, labels in duplicate_targets.items():
        st.warning(f"{'、'.join(labels)} 指向同一个输出文件：{display_path(path)}。")


def render_tag_tools() -> None:
    with st.expander("基础标签与本地音频匹配", expanded=True):
        with st.form("process-basic-tags"):
            input_csv = source_csv_select_with_confirm(
                "歌曲主表",
                default_source_csv_text(),
                "basic-input",
                confirm_song_table_defaults,
                (
                    "basic-input",
                    "basic-output",
                    "basic-jsonl",
                    "basic-matches",
                    "basic-audio-features",
                ),
            )
            lyrics_dir = st.text_input("歌词目录", display_path(LYRICS_DIR), key="basic-lyrics")
            audio_dir = st.text_input("本地音乐目录", str(DEFAULT_AUDIO_DIR), key="basic-audio")
            output_csv = session_text_input("输出标签 CSV", default_tag_csv_text(), key="basic-output")
            jsonl_output = session_text_input("输出标签 JSONL", default_tag_jsonl_text(), key="basic-jsonl")
            matches_output = session_text_input("输出音频匹配 CSV", default_audio_matches_text(), key="basic-matches")
            audio_features_csv = session_text_input(
                "已有音频特征 CSV",
                default_audio_features_csv_text(),
                key="basic-audio-features",
            )
            match_threshold = st.number_input("本地音频匹配阈值", 0.0, 1.0, 0.84, 0.01)
            reuse_matches = st.checkbox("复用已有音频匹配结果", value=False, key="basic-reuse")
            no_progress = st.checkbox("关闭进度条输出", value=True, key="basic-no-progress")
            timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="basic-tags-timeout")
            warn_existing_outputs(
                [
                    ("输出标签 CSV", output_csv),
                    ("输出标签 JSONL", jsonl_output),
                    ("输出音频匹配 CSV", matches_output),
                ]
            )

            command = base_tag_command(
                input_csv,
                lyrics_dir,
                audio_dir,
                output_csv,
                jsonl_output,
                matches_output,
                audio_features_csv,
            )
            command.extend(["--match-threshold", str(float(match_threshold))])
            add_flag(command, "--reuse-matches", reuse_matches)
            add_flag(command, "--no-progress", no_progress)

            submit_subprocess("生成基础标签", "basic-tags", command, int(timeout))

        render_result("basic-tags", "脚本输出会显示在这里。")

    with st.expander("音频特征与声源分离", expanded=False):
        with st.form("process-audio-features"):
            input_csv = source_csv_select_with_confirm(
                "歌曲主表",
                default_source_csv_text(),
                "audio-input",
                confirm_song_table_defaults,
                (
                    "audio-input",
                    "audio-output",
                    "audio-jsonl",
                    "audio-matches",
                    "audio-features-csv",
                    "audio-features-parquet",
                ),
            )
            lyrics_dir = st.text_input("歌词目录", display_path(LYRICS_DIR), key="audio-lyrics")
            audio_dir = st.text_input("本地音乐目录", str(DEFAULT_AUDIO_DIR), key="audio-audio")
            output_csv = session_text_input("输出标签 CSV", default_tag_csv_text(), key="audio-output")
            jsonl_output = session_text_input("输出标签 JSONL", default_tag_jsonl_text(), key="audio-jsonl")
            matches_output = session_text_input("音频匹配 CSV", default_audio_matches_text(), key="audio-matches")
            audio_features_csv = session_text_input(
                "音频特征 CSV",
                default_audio_features_csv_text(),
                key="audio-features-csv",
            )
            audio_features_parquet = session_text_input(
                "音频特征 Parquet",
                default_audio_features_parquet_text(),
                key="audio-features-parquet",
            )

            col_audio_1, col_audio_2, col_audio_3 = st.columns(3)
            with col_audio_1:
                reuse_matches = st.checkbox("复用已有音频匹配结果", value=True, key="audio-reuse")
                audio_feature_seconds = st.number_input("分析秒数", 1.0, 600.0, 20.0, 1.0)
            with col_audio_2:
                source_separation = st.checkbox("启用 HDemucs 声源分离", value=False)
                source_seconds = st.number_input("声源分离秒数", 1.0, 600.0, 30.0, 1.0)
            with col_audio_3:
                source_device = st.selectbox("设备", ["auto", "cpu", "cuda", "cuda:0"], index=0)
                timeout = st.number_input("超时秒数", 10, 86400, 7200, 60, key="audio-timeout")

            source_model = st.selectbox(
                "声源分离模型",
                ["hdemucs_high_musdb_plus", "hdemucs_high_musdb"],
                index=0,
            )
            source_checkpoint = st.text_input("HDemucs checkpoint", display_path(DEFAULT_HDEMUCS_CHECKPOINT))
            no_progress = st.checkbox("关闭进度条输出", value=True, key="audio-no-progress")
            warn_existing_outputs(
                [
                    ("输出标签 CSV", output_csv),
                    ("输出标签 JSONL", jsonl_output),
                    ("音频匹配 CSV", matches_output),
                    ("音频特征 CSV", audio_features_csv),
                    ("音频特征 Parquet", audio_features_parquet),
                ]
            )

            command = base_tag_command(input_csv, lyrics_dir, audio_dir, output_csv, jsonl_output, matches_output)
            command.extend(
                [
                    "--analyze-audio",
                    "--audio-feature-seconds",
                    str(float(audio_feature_seconds)),
                    "--audio-features-csv",
                    audio_features_csv,
                    "--audio-features-parquet",
                    audio_features_parquet,
                ]
            )
            add_flag(command, "--reuse-matches", reuse_matches)
            if source_separation:
                command.extend(
                    [
                        "--source-separation",
                        "--source-separation-model",
                        source_model,
                        "--source-separation-seconds",
                        str(float(source_seconds)),
                        "--source-separation-device",
                        source_device,
                    ]
                )
                add_arg(command, "--source-separation-checkpoint", source_checkpoint)
            add_flag(command, "--no-progress", no_progress)

            submit_subprocess("生成音频特征", "audio-features", command, int(timeout))

        render_result("audio-features", "脚本输出会显示在这里。")

    with st.expander("MERT Embedding 与聚类", expanded=False):
        with st.form("process-mert"):
            input_csv = source_csv_select_with_confirm(
                "歌曲主表",
                default_source_csv_text(),
                "mert-input",
                confirm_song_table_defaults,
                (
                    "mert-input",
                    "mert-output",
                    "mert-jsonl",
                    "mert-matches",
                    "mert-audio-features",
                    None,
                    "mert-index",
                    "mert-clusters-output",
                ),
            )
            lyrics_dir = st.text_input("歌词目录", display_path(LYRICS_DIR), key="mert-lyrics")
            audio_dir = st.text_input("本地音乐目录", str(DEFAULT_AUDIO_DIR), key="mert-audio")
            output_csv = session_text_input("输出标签 CSV", default_tag_csv_text(), key="mert-output")
            jsonl_output = session_text_input("输出标签 JSONL", default_tag_jsonl_text(), key="mert-jsonl")
            matches_output = session_text_input("音频匹配 CSV", default_audio_matches_text(), key="mert-matches")
            audio_features_csv = session_text_input(
                "已有音频特征 CSV",
                default_audio_features_csv_text(),
                key="mert-audio-features",
            )
            model_dir = st.text_input("MERT 模型目录", display_path(DEFAULT_MERT_MODEL_DIR))
            embeddings_dir = st.text_input("Embedding 输出目录", display_path(DEFAULT_MERT_EMBEDDINGS_DIR))
            mert_index = session_text_input("MERT 索引 CSV", default_mert_index_text(), key="mert-index")
            clusters_output = session_text_input("MERT 聚类 CSV", default_mert_clusters_text(), key="mert-clusters-output")

            col_mert_1, col_mert_2, col_mert_3 = st.columns(3)
            with col_mert_1:
                reuse_matches = st.checkbox("复用已有音频匹配结果", value=True, key="mert-reuse")
                overwrite_mert = st.checkbox("重算已有 embedding", value=False)
                mert_limit = st.number_input("处理条数上限（0 为全部）", 0, 100000, 0, 1)
            with col_mert_2:
                mert_max_seconds = st.number_input("单曲读取秒数", 1.0, 600.0, 12.0, 1.0)
                mert_chunk_seconds = st.number_input("Chunk 秒数", 1.0, 120.0, 5.0, 1.0)
                mert_stride_seconds = st.number_input("Stride 秒数", 1.0, 120.0, 5.0, 1.0)
            with col_mert_3:
                mert_device = st.selectbox("设备", ["auto", "cpu", "cuda", "cuda:0"], index=0, key="mert-device")
                mert_fp16 = st.checkbox("FP16", value=False)
                timeout = st.number_input("超时秒数", 10, 172800, 14400, 60, key="mert-timeout")

            col_mert_4, col_mert_5, col_mert_6 = st.columns(3)
            with col_mert_4:
                mert_layer = st.text_input("Layer", "mean")
                mert_top_k = st.number_input("情绪 Top K", 1, 20, 3, 1)
            with col_mert_5:
                emotion_threshold = st.number_input("情绪阈值", 0.0, 1.0, 0.16, 0.01)
                mert_clusters = st.number_input("聚类数", 1, 200, 12, 1)
            with col_mert_6:
                mert_neighbors = st.number_input("近邻数", 1, 200, 5, 1)
                no_progress = st.checkbox("关闭进度条输出", value=True, key="mert-no-progress")
            warn_existing_outputs(
                [
                    ("输出标签 CSV", output_csv),
                    ("输出标签 JSONL", jsonl_output),
                    ("音频匹配 CSV", matches_output),
                ]
            )

            command = base_tag_command(
                input_csv,
                lyrics_dir,
                audio_dir,
                output_csv,
                jsonl_output,
                matches_output,
                audio_features_csv,
            )
            command.extend(
                [
                    "--extract-mert",
                    "--mert-model-dir",
                    model_dir,
                    "--mert-embeddings-dir",
                    embeddings_dir,
                    "--mert-index",
                    mert_index,
                    "--mert-clusters-output",
                    clusters_output,
                    "--mert-max-seconds",
                    str(float(mert_max_seconds)),
                    "--mert-chunk-seconds",
                    str(float(mert_chunk_seconds)),
                    "--mert-stride-seconds",
                    str(float(mert_stride_seconds)),
                    "--mert-layer",
                    mert_layer,
                    "--mert-device",
                    mert_device,
                    "--mert-top-k",
                    str(int(mert_top_k)),
                    "--mert-emotion-threshold",
                    str(float(emotion_threshold)),
                    "--mert-clusters",
                    str(int(mert_clusters)),
                    "--mert-neighbors",
                    str(int(mert_neighbors)),
                ]
            )
            add_flag(command, "--reuse-matches", reuse_matches)
            add_flag(command, "--overwrite-mert", overwrite_mert)
            add_flag(command, "--mert-fp16", mert_fp16)
            add_flag(command, "--no-progress", no_progress)
            if int(mert_limit) > 0:
                command.extend(["--mert-limit", str(int(mert_limit))])

            submit_subprocess("提取 MERT", "mert", command, int(timeout))

        render_result("mert", "脚本输出会显示在这里。")


def render_cache_tools() -> None:
    with st.expander("预处理缓存清理", expanded=True):
        cache_targets = {
            "预处理 DataFrame": PREPROCESSED_DATA_FILE,
            "预处理 Hash": PREPROCESSED_HASH_FILE,
            "历史记录": CACHE_DIR / "recommendation_history.json",
            "历史设置": CACHE_DIR / "history_settings.json",
        }
        selected_targets = st.multiselect("清理对象", list(cache_targets), default=[])
        confirm = st.checkbox("确认删除选中的缓存文件", value=False)
        if st.button("删除缓存文件", width="stretch", disabled=not confirm or not selected_targets):
            deleted = []
            skipped = []
            for name in selected_targets:
                path = project_path(cache_targets[name])
                if path.exists() and path.is_file():
                    path.unlink()
                    deleted.append(str(path))
                else:
                    skipped.append(name)
            try:
                st.cache_data.clear()
            except Exception:
                pass
            save_inline_result(
                "cache-delete",
                "删除缓存文件",
                "\n".join(
                    [
                        f"已删除 {len(deleted)} 个缓存文件，跳过 {len(skipped)} 个不存在的文件。",
                        *deleted,
                    ]
                ),
            )
        render_result("cache-delete", "操作输出会显示在这里。")

    with st.expander("Streamlit 缓存", expanded=False):
        if st.button("清空 st.cache_data", width="stretch", key="clear-streamlit-cache"):
            st.cache_data.clear()
            save_inline_result("streamlit-cache-clear", "清空 st.cache_data", "已清空当前 Streamlit 进程的数据缓存。")
        render_result("streamlit-cache-clear", "操作输出会显示在这里。")


def render_data_processing_interface() -> None:
    st.subheader("数据处理")
    render_overview()

    tab_collection, tab_csv, tab_tags, tab_cache = st.tabs(
        ["数据采集", "CSV/歌词", "标签与音频", "缓存维护"]
    )

    with tab_collection:
        render_collection_tools()

    with tab_csv:
        render_csv_tools()

    with tab_tags:
        render_tag_tools()

    with tab_cache:
        render_cache_tools()
