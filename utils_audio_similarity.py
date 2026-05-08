import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from config import AUDIO_SIMILARITY_TOP_K, MERT_EMBEDDING_DIR, MERT_MODEL_DIR


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


@st.cache_resource(show_spinner=False)
def load_mert_vector_index(embedding_dir=MERT_EMBEDDING_DIR):
    resolved_dir = resolve_project_path(embedding_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"未找到 MERT 向量目录: {resolved_dir}")

    item_ids = []
    embeddings = []
    for path in sorted(resolved_dir.glob("*.npy"), key=lambda item: item.stem):
        item_id = path.stem.strip()
        if not item_id:
            continue
        vector = np.load(path).astype(np.float32).reshape(-1)
        if vector.size == 0:
            continue
        item_ids.append(item_id)
        embeddings.append(vector)

    if not embeddings:
        raise ValueError(f"MERT 向量目录为空或没有可用 .npy: {resolved_dir}")

    dimensions = {vector.shape[0] for vector in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"MERT 向量维度不一致: {sorted(dimensions)}")

    matrix = normalize_vectors(np.vstack(embeddings).astype(np.float32))
    id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    return {
        "index_path": str(resolved_dir),
        "item_ids": item_ids,
        "item_embeddings": matrix,
        "id_to_index": id_to_index,
        "dimension": int(matrix.shape[1]),
    }


@st.cache_resource(show_spinner=False)
def load_mert_query_engine(model_dir=MERT_MODEL_DIR):
    import torch
    import torchaudio
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    resolved_dir = resolve_project_path(model_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"未找到 MERT 模型目录: {resolved_dir}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(resolved_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(resolved_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "torchaudio": torchaudio,
        "torch": torch,
        "model_path": str(resolved_dir),
    }


def iter_chunks(torch, audio, sample_rate, chunk_seconds=5.0, stride_seconds=5.0):
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    stride = max(1, int(round(stride_seconds * sample_rate)))

    if audio.numel() <= chunk_size:
        return [torch.nn.functional.pad(audio, (0, chunk_size - audio.numel()))]

    chunks = []
    start = 0
    while start < audio.numel():
        chunk = audio[start : start + chunk_size]
        if chunk.numel() < int(0.5 * sample_rate):
            break
        if chunk.numel() < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.numel()))
        chunks.append(chunk)
        start += stride
    return chunks


def pool_hidden_states(torch, hidden_states, layer="mean"):
    if layer == "mean":
        stacked = torch.stack(hidden_states, dim=0)
        return stacked.mean(dim=0).mean(dim=1)
    return hidden_states[int(layer)].mean(dim=1)


def load_audio_for_mert(audio_path, engine):
    torch = engine["torch"]
    torchaudio = engine["torchaudio"]
    processor = engine["processor"]

    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = int(processor.sampling_rate)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return waveform.squeeze(0).contiguous().to(dtype=torch.float32), sample_rate


def embed_audio_file(audio_path, model_dir=MERT_MODEL_DIR, layer="mean"):
    engine = load_mert_query_engine(model_dir=model_dir)
    torch = engine["torch"]
    audio, sample_rate = load_audio_for_mert(audio_path, engine)
    chunks = iter_chunks(torch, audio, sample_rate)
    embeddings = []

    with torch.inference_mode():
        for chunk in chunks:
            inputs = engine["processor"](
                chunk.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(engine["device"]) for key, value in inputs.items()}
            outputs = engine["model"](**inputs, output_hidden_states=True)
            pooled = pool_hidden_states(torch, outputs.hidden_states, layer=layer)
            embeddings.append(pooled.float().cpu())

    embedding = torch.cat(embeddings, dim=0).mean(dim=0)
    return embedding.numpy().astype(np.float32)


def embed_uploaded_audio(audio_bytes, filename="", model_dir=MERT_MODEL_DIR):
    if not audio_bytes:
        raise ValueError("未提供上传音频内容。")

    suffix = Path(filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix or ".audio", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        return embed_audio_file(tmp_path, model_dir=model_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


def get_vector_for_song_id(query_song_id, embedding_dir=MERT_EMBEDDING_DIR):
    normalized_id = str(query_song_id).strip()
    if not normalized_id:
        raise ValueError("请输入有效的歌曲 ID。")

    index_data = load_mert_vector_index(embedding_dir=embedding_dir)
    matched_index = index_data["id_to_index"].get(normalized_id)
    if matched_index is None:
        raise ValueError(
            f"向量库里没有找到 ID={normalized_id} 的 MERT 向量，请确认该歌曲已生成到 {index_data['index_path']}。"
        )
    return index_data["item_embeddings"][matched_index]


def search_similar_audio_items(
    query_song_id="",
    query_audio_bytes=None,
    query_audio_name="",
    candidate_ids=None,
    top_k=AUDIO_SIMILARITY_TOP_K,
    embedding_dir=MERT_EMBEDDING_DIR,
    model_dir=MERT_MODEL_DIR,
):
    index_data = load_mert_vector_index(embedding_dir=embedding_dir)

    if query_audio_bytes:
        query_vector = embed_uploaded_audio(
            query_audio_bytes,
            filename=query_audio_name,
            model_dir=model_dir,
        )
        query_mode = "upload"
        query_label = query_audio_name or "uploaded-audio"
    else:
        query_vector = get_vector_for_song_id(query_song_id, embedding_dir=embedding_dir)
        query_mode = "id"
        query_label = str(query_song_id).strip()

    query_vector = normalize_vectors(query_vector)[0]

    if candidate_ids is None:
        candidate_list = index_data["item_ids"]
    else:
        candidate_list = []
        seen = set()
        for item_id in candidate_ids:
            normalized_id = str(item_id).strip()
            if not normalized_id or normalized_id in seen:
                continue
            seen.add(normalized_id)
            if normalized_id in index_data["id_to_index"]:
                candidate_list.append(normalized_id)

    if not candidate_list:
        return {
            "query_mode": query_mode,
            "query_label": query_label,
            "index_path": index_data["index_path"],
            "results": [],
        }

    candidate_indices = np.array(
        [index_data["id_to_index"][item_id] for item_id in candidate_list],
        dtype=np.int32,
    )
    candidate_matrix = index_data["item_embeddings"][candidate_indices]
    scores = candidate_matrix @ query_vector

    top_k = max(1, min(int(top_k), len(candidate_list)))
    ranked_local = np.argpartition(scores, len(scores) - top_k)[-top_k:]
    ranked_local = ranked_local[np.argsort(scores[ranked_local])[::-1]]

    results = []
    for rank, local_idx in enumerate(ranked_local, start=1):
        item_id = candidate_list[int(local_idx)]
        results.append(
            {
                "rank": rank,
                "item_id": item_id,
                "score": float(scores[int(local_idx)] * 100.0),
            }
        )

    return {
        "query_mode": query_mode,
        "query_label": query_label,
        "index_path": index_data["index_path"],
        "results": results,
    }
