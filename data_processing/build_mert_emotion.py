#!/usr/bin/env python
"""
Build a local m-a-p/MERT-v1-330M emotion report for one audio file.

Important: MERT-v1-330M is a self-supervised music representation model. For
real emotion recognition you should load a fine-tuned classification head with
--head-checkpoint. Without that head this script runs a deterministic audio
affect proxy so the command is end-to-end runnable, but the proxy is not a
trained MERT emotion classifier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

np: Any = None
torch: Any = None
nn: Any = None
torchaudio: Any = None
AutoModel: Any = None
Wav2Vec2FeatureExtractor: Any = None
EmotionHead: Any = None


DEFAULT_LABELS = [
    "calm",
    "happy",
    "sad",
    "angry",
    "romantic",
    "tense",
    "energetic",
    "melancholic",
]


def load_runtime_dependencies() -> None:
    global np, torch, nn, torchaudio, AutoModel, Wav2Vec2FeatureExtractor, EmotionHead

    try:
        import numpy as _np
        import torch as _torch
        from torch import nn as _nn
        import torchaudio as _torchaudio
        from transformers import AutoModel as _AutoModel
        from transformers import Wav2Vec2FeatureExtractor as _Wav2Vec2FeatureExtractor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    class _EmotionHead(_nn.Module):
        def __init__(self, input_dim: int, labels: list[str]) -> None:
            super().__init__()
            self.labels = labels
            self.classifier = _nn.Linear(input_dim, len(labels))

        def forward(self, embedding: _torch.Tensor) -> _torch.Tensor:
            return self.classifier(embedding)

    np = _np
    torch = _torch
    nn = _nn
    torchaudio = _torchaudio
    AutoModel = _AutoModel
    Wav2Vec2FeatureExtractor = _Wav2Vec2FeatureExtractor
    EmotionHead = _EmotionHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local MERT-v1-330M on a FLAC music file for emotion analysis."
    )
    parser.add_argument("--model-dir", required=True, help="本地 MERT-v1-330M 模型目录")
    parser.add_argument("--audio", required=True, help="输入 FLAC 音乐文件")
    parser.add_argument(
        "--head-checkpoint",
        default=None,
        help="可选：微调后的情感分类头 .pt/.pth；没有则使用启发式 demo 模式",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="分类头标签，逗号分隔；checkpoint 内含 labels 时会被覆盖",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "classifier", "heuristic"],
        default="auto",
        help="auto: 有分类头用 classifier，否则 heuristic",
    )
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="分块秒数")
    parser.add_argument("--stride-seconds", type=float, default=5.0, help="分块步长秒数")
    parser.add_argument("--max-seconds", type=float, default=None, help="只分析前 N 秒")
    parser.add_argument("--layer", default="mean", help="MERT 层：mean 或整数，如 -1/24")
    parser.add_argument("--top-k", type=int, default=5, help="输出前 K 个情绪")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--fp16", action="store_true", help="CUDA 上使用半精度推理")
    parser.add_argument("--output-json", default=None, help="可选：保存 JSON 结果")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_audio(path: Path, target_sr: int, max_seconds: float | None) -> tuple[torch.Tensor, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".flac":
        raise ValueError("This demo expects a .flac input file.")

    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if max_seconds is not None:
        waveform = waveform[:, : int(sample_rate * max_seconds)]

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    return waveform.squeeze(0).contiguous(), sample_rate


def iter_chunks(
    audio: torch.Tensor,
    sample_rate: int,
    chunk_seconds: float,
    stride_seconds: float,
) -> list[torch.Tensor]:
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    stride = max(1, int(round(stride_seconds * sample_rate)))

    if audio.numel() <= chunk_size:
        return [torch.nn.functional.pad(audio, (0, chunk_size - audio.numel()))]

    chunks: list[torch.Tensor] = []
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


def layer_pool(hidden_states: tuple[torch.Tensor, ...], layer: str) -> torch.Tensor:
    if layer == "mean":
        stacked = torch.stack(hidden_states, dim=0)
        return stacked.mean(dim=0).mean(dim=1)

    layer_index = int(layer)
    return hidden_states[layer_index].mean(dim=1)


def extract_mert_embedding(
    model: nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    audio: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    chunk_seconds: float,
    stride_seconds: float,
    layer: str,
    use_fp16: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    chunks = iter_chunks(audio, sample_rate, chunk_seconds, stride_seconds)
    embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = processor(
                chunk.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", enabled=use_fp16 and device.type == "cuda"):
                outputs = model(**inputs, output_hidden_states=True)
                pooled = layer_pool(outputs.hidden_states, layer)
            embeddings.append(pooled.float().cpu())

    embedding = torch.cat(embeddings, dim=0).mean(dim=0)
    stats = {
        "chunks": len(chunks),
        "embedding_dim": int(embedding.numel()),
        "layer": layer,
        "chunk_seconds": chunk_seconds,
        "stride_seconds": stride_seconds,
    }
    return embedding, stats


def clean_state_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = raw.get("state_dict") or raw.get("model_state_dict") or raw
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        next_key = key.removeprefix("module.")
        if next_key.startswith("head."):
            next_key = "classifier." + next_key[len("head.") :]
        cleaned[next_key] = value
    return cleaned


def load_emotion_head(
    checkpoint_path: Path,
    input_dim: int,
    fallback_labels: list[str],
    device: torch.device,
) -> EmotionHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    labels = checkpoint.get("labels") or checkpoint.get("label_names") or fallback_labels
    labels = [str(label) for label in labels]
    head = EmotionHead(input_dim=input_dim, labels=labels)
    missing, _unexpected = head.load_state_dict(clean_state_dict(checkpoint), strict=False)
    if "classifier.weight" in missing or "classifier.bias" in missing:
        raise ValueError(
            "The head checkpoint did not contain classifier.weight/classifier.bias "
            "compatible with this demo."
        )
    head.eval().to(device)
    return head


def classifier_scores(
    embedding: torch.Tensor,
    head: EmotionHead,
    device: torch.device,
) -> list[dict[str, float | str]]:
    with torch.no_grad():
        logits = head(embedding.unsqueeze(0).to(device)).squeeze(0)
    probs = torch.softmax(logits.float().cpu(), dim=-1)
    return [
        {"label": label, "score": float(score)}
        for label, score in zip(head.labels, probs.tolist(), strict=True)
    ]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def audio_affect_proxy(audio: torch.Tensor, sample_rate: int) -> dict[str, float]:
    audio = audio.float()
    if audio.numel() == 0:
        return {"arousal": 0.5, "valence": 0.5}
    if audio.numel() < 2048:
        audio = torch.nn.functional.pad(audio, (0, 2048 - audio.numel()))

    rms = float(torch.sqrt(torch.mean(audio.square()) + 1e-9))
    zcr = float(torch.mean((audio[1:] * audio[:-1] < 0).float())) if audio.numel() > 1 else 0.0

    n_fft = 2048
    hop = 512
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
        center=True,
    ).abs()
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[0])
    centroid = (spec * freqs[:, None]).sum(dim=0) / (spec.sum(dim=0) + 1e-9)
    centroid_hz = float(centroid.mean())

    frame_energy = spec.square().mean(dim=0)
    flux = torch.relu(frame_energy[1:] - frame_energy[:-1])
    flux_value = float(flux.mean()) if flux.numel() else 0.0

    arousal = sigmoid(2.4 * math.log10(rms + 1e-5) + 0.0010 * centroid_hz + 10.0 * zcr + 0.8)
    arousal = max(0.0, min(1.0, arousal + min(flux_value, 1.0) * 0.05))

    brightness = sigmoid((centroid_hz - 1800.0) / 900.0)
    roughness = sigmoid((zcr - 0.08) * 18.0)
    valence = max(0.0, min(1.0, 0.62 * brightness + 0.38 * (1.0 - roughness)))

    return {
        "arousal": float(arousal),
        "valence": float(valence),
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid_hz": centroid_hz,
    }


def heuristic_scores(audio: torch.Tensor, sample_rate: int) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    features = audio_affect_proxy(audio, sample_rate)
    point = np.array([features["valence"], features["arousal"]], dtype=np.float32)
    prototypes = {
        "calm": (0.72, 0.22),
        "happy": (0.82, 0.72),
        "sad": (0.20, 0.25),
        "angry": (0.18, 0.82),
        "romantic": (0.78, 0.40),
        "tense": (0.28, 0.68),
        "energetic": (0.62, 0.88),
        "melancholic": (0.32, 0.38),
    }
    distances = []
    for label, proto in prototypes.items():
        distance = float(np.linalg.norm(point - np.array(proto, dtype=np.float32)))
        distances.append((label, distance))

    raw = np.array([-distance * 5.0 for _, distance in distances], dtype=np.float32)
    raw = np.exp(raw - raw.max())
    probs = raw / raw.sum()
    scores = [
        {"label": label, "score": float(score)}
        for (label, _), score in zip(distances, probs.tolist(), strict=True)
    ]
    return scores, features


def rank_scores(scores: list[dict[str, float | str]], top_k: int) -> list[dict[str, float | str]]:
    return sorted(scores, key=lambda item: float(item["score"]), reverse=True)[:top_k]


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()

    model_dir = Path(args.model_dir).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    device = choose_device(args.device)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().to(device)

    audio, sample_rate = load_audio(audio_path, processor.sampling_rate, args.max_seconds)
    embedding, mert_stats = extract_mert_embedding(
        model=model,
        processor=processor,
        audio=audio,
        sample_rate=sample_rate,
        device=device,
        chunk_seconds=args.chunk_seconds,
        stride_seconds=args.stride_seconds,
        layer=args.layer,
        use_fp16=args.fp16,
    )

    use_classifier = args.mode == "classifier" or (
        args.mode == "auto" and args.head_checkpoint is not None
    )
    if use_classifier:
        if not args.head_checkpoint:
            raise ValueError("--mode classifier requires --head-checkpoint")
        head = load_emotion_head(Path(args.head_checkpoint), embedding.numel(), labels, device)
        scores = classifier_scores(embedding, head, device)
        method = "mert_classifier"
        extra: dict[str, Any] = {}
    else:
        scores, affect = heuristic_scores(audio, sample_rate)
        method = "heuristic_audio_affect_proxy"
        extra = {
            "note": (
                "No fine-tuned emotion head was provided. MERT was used for local "
                "feature extraction, while emotion labels came from a deterministic "
                "audio-affect proxy for demo purposes."
            ),
            "affect_proxy": affect,
        }

    result = {
        "audio": str(audio_path),
        "model_dir": str(model_dir),
        "method": method,
        "sample_rate": sample_rate,
        "duration_seconds": round(audio.numel() / sample_rate, 3),
        "mert": mert_stats,
        "top_emotions": rank_scores(scores, args.top_k),
        **extra,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
