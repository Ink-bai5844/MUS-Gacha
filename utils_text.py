import re
from collections import Counter

import pandas as pd

from config import COMMENT_RULES, LYRIC_STOP_WORDS


_JANOME_TOKENIZER = None


def safe_text(value):
    if isinstance(value, (list, tuple, set)):
        return " | ".join(safe_text(item) for item in value if safe_text(item))
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
    text = re.sub(
        r"^\s*(作词|作曲|编曲|制作人|OP|SP|Publisher)\s*[:：].*$",
        " ",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"纯音乐[，,、\s]*请欣赏", " ", text)
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
    return extract_comment_semantic_tags_from_text(text)


def extract_comment_semantic_tags_from_text(text):
    tags = []
    for tag, needles in COMMENT_RULES:
        if any(needle.lower() in text.lower() for needle in needles):
            tags.append(tag)
    return unique_items(tags)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def normalize_for_search(value):
    return re.sub(r"\s+", " ", safe_text(value).lower())
