"""
Build lyric semantic tags with the local bge-m3 model.

The script stores lyric semantic artifacts under data/features/lyric and writes
the final fields back to data/tags/*_song_tags.csv, which is the only generated
result table read by the app.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = BASE_DIR / "data" / "source"
TAG_DATA_DIR = BASE_DIR / "data" / "tags"
LYRIC_SEMANTIC_DATA_DIR = BASE_DIR / "data" / "features" / "lyric"
DEFAULT_LYRICS_DIR = SOURCE_DATA_DIR / "lyrics"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "bge-m3"
LYRIC_MODEL_THRESHOLD = 0.48
LYRIC_MODEL_MARGIN = 0.04
LYRIC_MODEL_MAX_TAGS = 3

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import LYRIC_SEMANTIC_RULES  # noqa: E402
from utils_text import clean_lrc_text, safe_text, unique_items  # noqa: E402
from build_comment_semantics import (  # noqa: E402
    choose_device,
    encode_texts,
    merge_existing_csv,
    merge_existing_jsonl,
    print_write_notice,
    validate_model_dir,
)


def strip_dataset_suffix(name: str) -> str:
    clean = name.strip().replace(" ", "_")
    for suffix in ("_songs", "_song_tags", "_lyric_semantics", "_json"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    return clean or name.strip() or "songs"


def dataset_name_from_path(path: Path) -> str:
    return strip_dataset_suffix(path.stem or path.name)


def default_input_csv() -> Path:
    paths = sorted(path for path in SOURCE_DATA_DIR.glob("*.csv") if path.is_file())
    return paths[0] if paths else SOURCE_DATA_DIR / "songs.csv"


def apply_defaults(args: argparse.Namespace) -> None:
    if args.input is None:
        args.input = default_input_csv()
    dataset = dataset_name_from_path(args.input)
    if args.output is None:
        args.output = LYRIC_SEMANTIC_DATA_DIR / f"{dataset}_lyric_semantics.csv"
    if args.jsonl_output is None:
        args.jsonl_output = LYRIC_SEMANTIC_DATA_DIR / f"{dataset}_lyric_semantics.jsonl"
    if args.tags_output is None:
        args.tags_output = TAG_DATA_DIR / f"{dataset}_song_tags.csv"


def normalize_song_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    if "song_id" not in df.columns:
        return pd.DataFrame()
    df["song_id"] = df["song_id"].astype("string").fillna("").str.strip()
    return df[df["song_id"].ne("")]


def read_lyric_file(lyrics_dir: Path, song_id: str) -> str:
    path = lyrics_dir / f"{song_id}.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def lyric_text_from_row(row: pd.Series, lyrics_dir: Path, max_chars: int) -> str:
    song_id = safe_text(row.get("song_id"))
    text = read_lyric_file(lyrics_dir, song_id)
    if not text:
        text = " ".join(
            [
                safe_text(row.get("lyric_text")),
                safe_text(row.get("translation_text")),
                safe_text(row.get("lyric_excerpt")),
                safe_text(row.get("translation_excerpt")),
            ]
        )
    cleaned = clean_lrc_text(text)
    if max_chars > 0:
        cleaned = cleaned[:max_chars]
    return cleaned.strip()


def lyric_label_texts() -> list[str]:
    return [
        f"歌词语义标签：{tag}。相关歌词表达：{'、'.join(needles)}。"
        for tag, needles in LYRIC_SEMANTIC_RULES
    ]


def fallback_lyric_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for tag, needles in LYRIC_SEMANTIC_RULES:
        if any(needle.lower() in lowered for needle in needles):
            tags.append(tag)
    return unique_items(tags)


def load_model(model_dir: Path, device_name: str) -> tuple[Any, Any, Any]:
    complete, missing = validate_model_dir(model_dir)
    if not complete:
        raise FileNotFoundError(f"bge-m3 model is incomplete, missing: {', '.join(missing)}")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    device = choose_device(device_name)
    model.eval().to(device)
    print(f"Loaded bge-m3 lyric semantic model from {model_dir}")
    print(f"Device: {device}")
    return model, tokenizer, device


def semantic_tags_from_scores(
    scores: Any,
    threshold: float,
    margin: float,
    max_tags: int,
) -> tuple[list[str], str]:
    labels = [tag for tag, _needles in LYRIC_SEMANTIC_RULES]
    indexed_scores = [(index, float(score)) for index, score in enumerate(scores)]
    ranked = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
    if not ranked:
        return [], ""

    best_score = ranked[0][1]
    tags = []
    score_parts = []
    for index, score in ranked[:max_tags]:
        if score >= threshold and best_score - score <= margin:
            tags.append(labels[index])
            score_parts.append(f"{labels[index]}:{score:.4f}")
    return tags, " | ".join(score_parts)


def build_lyric_semantics(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"song_id": "string"})
    df = normalize_song_ids(df)
    if args.limit > 0:
        df = df.head(args.limit)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "song_id",
                "lyric_semantic_tags",
                "lyric_semantic_scores",
                "lyric_semantic_model",
                "lyric_semantic_error",
            ]
        )

    texts = [lyric_text_from_row(row, args.lyrics_dir, args.max_chars) for _idx, row in df.iterrows()]
    fallback_tags = [fallback_lyric_tags(text) for text in texts]
    nonempty_items = [(index, text) for index, text in enumerate(texts) if text]

    model, tokenizer, device = load_model(args.model_dir, args.device)
    label_embeddings = encode_texts(
        lyric_label_texts(),
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_length,
    )

    score_by_index: dict[int, Any] = {}
    if nonempty_items:
        nonempty_texts = [text for _index, text in nonempty_items]
        text_embeddings = encode_texts(
            nonempty_texts,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.max_length,
        )
        scores = text_embeddings @ label_embeddings.T
        score_by_index = {
            row_index: row_scores
            for (row_index, _text), row_scores in zip(nonempty_items, scores)
        }

    rows = []
    for row_index, (_idx, source_row) in enumerate(df.iterrows()):
        text = texts[row_index]
        tags = []
        score_text = ""
        error = ""
        if text:
            tags, score_text = semantic_tags_from_scores(
                score_by_index.get(row_index),
                args.threshold,
                args.margin,
                args.max_tags,
            )
            if not tags and args.keyword_fallback:
                tags = fallback_tags[row_index]
                if tags:
                    error = "model_low_confidence_keyword_fallback"
        rows.append(
            {
                "song_id": safe_text(source_row.get("song_id")),
                "lyric_semantic_tags": " | ".join(unique_items(tags)),
                "lyric_semantic_scores": score_text,
                "lyric_semantic_model": "bge-m3",
                "lyric_semantic_error": error,
            }
        )
        if not args.no_progress and (row_index + 1 == len(df) or (row_index + 1) % 20 == 0):
            print(f"[{row_index + 1}/{len(df)}] processed lyric semantics")

    return pd.DataFrame(rows)


def write_outputs(args: argparse.Namespace, df: pd.DataFrame) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df, existing_csv = merge_existing_csv(args.output, df)
    print_write_notice("歌词语义 CSV", args.output, existing_csv, len(output_df))
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    if args.jsonl_output:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        jsonl_df, existing_jsonl = merge_existing_jsonl(args.jsonl_output, df)
        print_write_notice("歌词语义 JSONL", args.jsonl_output, existing_jsonl, len(jsonl_df))
        with args.jsonl_output.open("w", encoding="utf-8", newline="") as file:
            for record in jsonl_df.fillna("").to_dict(orient="records"):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.tags_output:
        args.tags_output.parent.mkdir(parents=True, exist_ok=True)
        tags_df, existing_tags = merge_existing_csv(args.tags_output, df)
        print_write_notice("标签总表 CSV", args.tags_output, existing_tags, len(tags_df))
        tags_df.to_csv(args.tags_output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lyric semantic tags with local bge-m3.")
    parser.add_argument("--input", type=Path, default=None, help="Source song CSV. First data/source/*.csv is used when omitted.")
    parser.add_argument("--lyrics-dir", type=Path, default=DEFAULT_LYRICS_DIR, help="Lyrics txt directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output lyric semantic CSV.")
    parser.add_argument("--jsonl-output", type=Path, default=None, help="Output lyric semantic JSONL.")
    parser.add_argument("--tags-output", type=Path, default=None, help="Song tag CSV to update with lyric semantic columns.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local bge-m3 model directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=2400, help="Maximum lyric characters sent to the model; 0 means no trim.")
    parser.add_argument("--threshold", type=float, default=LYRIC_MODEL_THRESHOLD)
    parser.add_argument("--margin", type=float, default=LYRIC_MODEL_MARGIN)
    parser.add_argument("--max-tags", type=int, default=LYRIC_MODEL_MAX_TAGS)
    parser.add_argument("--limit", type=int, default=0, help="Limit source rows; 0 means all rows.")
    parser.add_argument("--no-keyword-fallback", dest="keyword_fallback", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--validate-model-only", action="store_true")
    parser.set_defaults(keyword_fallback=True)
    args = parser.parse_args()
    apply_defaults(args)
    return args


def main() -> None:
    args = parse_args()
    complete, missing = validate_model_dir(args.model_dir)
    print(f"bge-m3 model complete: {complete}")
    if missing:
        print(f"Missing files: {', '.join(missing)}")
    if args.validate_model_only:
        if not complete:
            raise SystemExit(2)
        return

    result_df = build_lyric_semantics(args)
    write_outputs(args, result_df)


if __name__ == "__main__":
    main()
