"""
Build comment semantic tags with the local bge-m3 model.

This script is intentionally separate from the main app preprocessing path so
the model is loaded only when the user explicitly runs comment semantic analysis.
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
COMMENT_SEMANTIC_DATA_DIR = BASE_DIR / "data" / "features" / "comment"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "bge-m3"
COMMENT_MODEL_THRESHOLD = 0.48
COMMENT_MODEL_MARGIN = 0.03
COMMENT_MODEL_MAX_TAGS = 3

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import COMMENT_RULES  # noqa: E402
from utils_text import extract_comment_semantic_tags_from_text, safe_text, unique_items  # noqa: E402


def strip_dataset_suffix(name: str) -> str:
    clean = name.strip().replace(" ", "_")
    for suffix in ("_songs", "_song_tags", "_comment_semantics", "_json"):
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
        args.output = COMMENT_SEMANTIC_DATA_DIR / f"{dataset}_comment_semantics.csv"
    if args.jsonl_output is None:
        args.jsonl_output = COMMENT_SEMANTIC_DATA_DIR / f"{dataset}_comment_semantics.jsonl"
    if args.tags_output is None:
        args.tags_output = TAG_DATA_DIR / f"{dataset}_song_tags.csv"


def normalize_song_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    if "song_id" not in df.columns:
        return pd.DataFrame()
    df["song_id"] = df["song_id"].astype("string").fillna("").str.strip()
    return df[df["song_id"].ne("")]


def merge_latest_nonempty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "song_id" not in df.columns:
        return df

    df = normalize_song_ids(df).reset_index(drop=True)
    df["_row_order"] = range(len(df))
    song_order = (
        df[["song_id", "_row_order"]]
        .drop_duplicates("song_id", keep="last")
        .sort_values("_row_order", kind="stable")
    )
    result = song_order[["song_id"]].reset_index(drop=True)

    for col in [item for item in df.columns if item not in {"song_id", "_row_order"}]:
        values = df[["song_id", col]].copy()
        values[col] = values[col].fillna("").astype(str)
        nonempty = values[values[col].str.strip().ne("")]
        if nonempty.empty:
            result[col] = ""
            continue
        latest = nonempty.drop_duplicates("song_id", keep="last")
        result = result.merge(latest, on="song_id", how="left")
        result[col] = result[col].fillna("")
    return result


def merge_existing_csv(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    output_df = normalize_song_ids(df)
    existing = path.exists() and path.stat().st_size > 0
    if not existing:
        return merge_latest_nonempty_rows(output_df), False

    existing_df = pd.read_csv(path, dtype={"song_id": "string"})
    existing_df.columns = [col.strip() for col in existing_df.columns]
    columns = list(dict.fromkeys([*existing_df.columns, *output_df.columns]))
    merged = pd.concat(
        [existing_df.reindex(columns=columns), output_df.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    return merge_latest_nonempty_rows(merged), True


def read_jsonl(path: Path) -> pd.DataFrame:
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


def merge_existing_jsonl(path: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    output_df = normalize_song_ids(df)
    existing = path.exists() and path.stat().st_size > 0
    if not existing:
        return merge_latest_nonempty_rows(output_df), False

    existing_df = read_jsonl(path)
    columns = list(dict.fromkeys([*existing_df.columns, *output_df.columns]))
    merged = pd.concat(
        [existing_df.reindex(columns=columns), output_df.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    return merge_latest_nonempty_rows(merged), True


def print_write_notice(label: str, path: Path, existing: bool, rows: int) -> None:
    if existing:
        print(f"{label} already exists; appended and merged by song_id: {path}")
    print(f"Wrote {rows} row(s) to {path}")


def write_outputs(args: argparse.Namespace, df: pd.DataFrame) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df, existing_csv = merge_existing_csv(args.output, df)
    print_write_notice("评论语义 CSV", args.output, existing_csv, len(output_df))
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    if args.jsonl_output:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        jsonl_df, existing_jsonl = merge_existing_jsonl(args.jsonl_output, df)
        print_write_notice("评论语义 JSONL", args.jsonl_output, existing_jsonl, len(jsonl_df))
        with args.jsonl_output.open("w", encoding="utf-8", newline="") as file:
            for record in jsonl_df.fillna("").to_dict(orient="records"):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.tags_output:
        args.tags_output.parent.mkdir(parents=True, exist_ok=True)
        tags_df, existing_tags = merge_existing_csv(args.tags_output, df)
        print_write_notice("标签总表 CSV", args.tags_output, existing_tags, len(tags_df))
        tags_df.to_csv(args.tags_output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def comment_text_from_row(row: pd.Series) -> str:
    return " ".join(
        [
            safe_text(row.get("first_hot_comment")),
            safe_text(row.get("first_comment")),
        ]
    ).strip()


def comment_label_texts() -> list[str]:
    return [
        f"网易云音乐评论语义标签：{tag}。相关表达：{'、'.join(needles)}。"
        for tag, needles in COMMENT_RULES
    ]


def validate_model_dir(model_dir: Path) -> tuple[bool, list[str]]:
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "1_Pooling/config.json",
    ]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    return not missing, missing


def choose_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_model(model_dir: Path, device_name: str) -> tuple[Any, Any, Any]:
    complete, missing = validate_model_dir(model_dir)
    if not complete:
        raise FileNotFoundError(f"bge-m3 model is incomplete, missing: {', '.join(missing)}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    device = choose_device(device_name)
    model.eval().to(device)
    print(f"Loaded bge-m3 comment semantic model from {model_dir}")
    print(f"Device: {device}")
    return model, tokenizer, device


def encode_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> Any:
    import torch
    import torch.nn.functional as F

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            # bge-m3's sentence-transformers config uses CLS pooling followed by normalization.
            pooled = output.last_hidden_state[:, 0]
            embeddings.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
    return torch.cat(embeddings, dim=0)


def semantic_tags_from_scores(
    scores: Any,
    threshold: float,
    margin: float,
    max_tags: int,
) -> tuple[list[str], str]:
    labels = [tag for tag, _needles in COMMENT_RULES]
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


def build_comment_semantics(args: argparse.Namespace) -> pd.DataFrame:
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
                "comment_semantic_tags",
                "comment_semantic_scores",
                "comment_semantic_model",
                "comment_semantic_error",
            ]
        )

    rows = []
    texts = [comment_text_from_row(row) for _idx, row in df.iterrows()]
    fallback_tags = [extract_comment_semantic_tags_from_text(text) for text in texts]
    nonempty_items = [(index, text) for index, text in enumerate(texts) if text]

    model, tokenizer, device = load_model(args.model_dir, args.device)
    label_embeddings = encode_texts(
        comment_label_texts(),
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
                "comment_semantic_tags": " | ".join(unique_items(tags)),
                "comment_semantic_scores": score_text,
                "comment_semantic_model": "bge-m3",
                "comment_semantic_error": error,
            }
        )
        if not args.no_progress and (row_index + 1 == len(df) or (row_index + 1) % 20 == 0):
            print(f"[{row_index + 1}/{len(df)}] processed comment semantics")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comment semantic tags with local bge-m3.")
    parser.add_argument("--input", type=Path, default=None, help="Source song CSV. First data/source/*.csv is used when omitted.")
    parser.add_argument("--output", type=Path, default=None, help="Output comment semantic CSV.")
    parser.add_argument("--jsonl-output", type=Path, default=None, help="Output comment semantic JSONL.")
    parser.add_argument("--tags-output", type=Path, default=None, help="Song tag CSV to update with comment semantic columns.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local bge-m3 model directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=COMMENT_MODEL_THRESHOLD)
    parser.add_argument("--margin", type=float, default=COMMENT_MODEL_MARGIN)
    parser.add_argument("--max-tags", type=int, default=COMMENT_MODEL_MAX_TAGS)
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

    result_df = build_comment_semantics(args)
    write_outputs(args, result_df)


if __name__ == "__main__":
    main()
