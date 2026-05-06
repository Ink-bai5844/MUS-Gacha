import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import LYRICS_DIR, TEXT_COLUMNS
from utils_text import normalize_for_search, safe_text, split_tags


def read_lyric(song_id):
    path = LYRICS_DIR / f"{song_id}.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").strip()


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
