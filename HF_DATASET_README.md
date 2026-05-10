---
license: other
language:
  - zh
  - en
  - ja
  - ko
tags:
  - music
  - music-information-retrieval
  - netease-cloud-music
  - lyrics
  - tabular
  - embeddings
  - mert
pretty_name: Music Info Datasets
size_categories:
  - 1K<n<10K
task_categories:
  - feature-extraction
  - text-classification
  - zero-shot-classification
---

# Music Info Datasets

`Music-info-datasets` 是一个围绕歌曲元数据、歌词、评论语义、本地音频匹配、音频特征和 MERT 表征整理的音乐信息数据集。数据以网易云音乐 `song_id` 作为统一主键，适合用于个人音乐库分析、标签体系构建、音乐检索、推荐实验、歌词/评论语义分析和音乐信息检索原型。

本数据集不提供可播放音频文件。表中的 `local_audio_path`、`file_path` 等字段来自作者本地音乐库匹配结果，仅用于说明匹配关系和特征来源，通常不能在其他环境中直接访问。

## Dataset Summary

当前包含两个子数据集：

| 子数据集 | 说明 | 主表歌曲数 | 标签表歌曲数 | 本地音频匹配数 | 音频特征数 | MERT 索引数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ink_bai_liked` | 作者网易云喜欢音乐/个人曲库样本 | 1,488 | 1,488 | 870 | 870 | 870 |
| `middle_ages` | 主题歌单样本 | 219 | 219 | 12 | 12 | 12 |

额外文件：

| 类型 | 数量/说明 |
| --- | --- |
| 原始 JSON 快照 | `ink_bai_liked_json/` 1,585 个，`middle_ages_json/` 219 个 |
| 歌词 TXT | `source/lyrics/` 1,658 个，以 `{song_id}.txt` 命名 |
| SQLite 采集库 | `ink_bai_liked.sqlite3`、`middle_ages.sqlite3`，包含 `songs` 和 `api_results` 表 |
| MERT embedding | `mert/embeddings/` 871 个 `.npy` 文件，每个为 1024 维 `float32` 向量 |

## File Structure

```text
data/
  source/
    ink_bai_liked_songs.csv
    middle_ages_songs.csv
    ink_bai_liked.sqlite3
    middle_ages.sqlite3
    ink_bai_liked_json/*.json
    middle_ages_json/*.json
    lyrics/{song_id}.txt
  tags/
    ink_bai_liked_song_tags.csv
    ink_bai_liked_song_tags.jsonl
    middle_ages_song_tags.csv
    middle_ages_song_tags.jsonl
  matches/
    ink_bai_liked_song_matches.csv
    middle_ages_song_matches.csv
  features/
    audio/
      *_song_features.csv
      *_song_features.parquet
    lyric/
      *_lyric_semantics.csv
      *_lyric_semantics.jsonl
    comment/
      *_comment_semantics.csv
      *_comment_semantics.jsonl
  mert/
    embeddings/{song_id}.npy
    *_mert_index.csv
    *_mert_clusters.csv
```

## Data Fields

### `data/source/*_songs.csv`

歌曲主表，由原始 JSON 快照整理而来。两份主表均包含 95 个字段，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `song_id` | 网易云音乐歌曲 ID，数据集主键 |
| `name` | 歌曲名 |
| `aliases` / `translations` | 别名、翻译名 |
| `artist_names` / `artist_ids` | 艺人名和艺人 ID，多个值用 ` | ` 分隔 |
| `album_id` / `album_name` / `album_pic_url` | 专辑信息 |
| `duration_ms` / `duration_seconds` / `duration_text` | 时长 |
| `publish_time_ms` / `publish_date` | 发布时间 |
| `popularity` / `mv_id` / `fee` / `copyright` / `status` | 平台侧基础信息 |
| `check_success` / `playable` / `check_message` | 可播放性检查结果 |
| `max_br_level` / `max_bitrate` | 可用音质概览 |
| `comment_total` / `hot_comment_count` | 评论数量统计 |
| `first_hot_comment` / `first_comment` | 抽取的代表性评论文本 |
| `has_lyric` / `lyric_line_count` / `lyric_excerpt` | 歌词存在性和摘要 |
| `has_translation` / `translation_excerpt` | 翻译歌词信息 |
| `has_romaji` / `romaji_excerpt` | 罗马音歌词信息 |
| `similar_song_ids` / `similar_song_names` / `similar_artist_names` | 平台返回的相似歌曲信息 |
| `wiki_summary_excerpt` | 平台侧百科摘要 |
| `standard_*` / `exhigh_*` / `lossless_*` / `hires_*` | 不同音质层级的 URL、码率、大小、类型、状态码和 MD5 等信息 |

注意：音频 URL 可能会过期或受地区、账号、版权和平台策略影响。

### `data/tags/*_song_tags.csv`

融合后的标签总表，是最适合直接用于检索、推荐和建模的入口表。

| 字段组 | 说明 |
| --- | --- |
| 基础字段 | `song_id`、`name`、`artist_names`、`has_lyric` |
| 语言/风格/情绪/主题/场景 | `language_tags`、`style_tags`、`emotion_tags`、`theme_tags`、`scene_tags` |
| 音频标签 | `audio_tags`、`audio_feature_tags`、`vocal_instrumental_tags` |
| 汇总标签 | `all_tags`、`tag_confidence`、`tag_sources` |
| 本地匹配 | `local_audio_path`、`audio_match_score`、`match_source`、`local_duration_seconds`、`duration_diff_seconds` |
| 音频特征 | `audio_duration_seconds`、`audio_sample_rate`、`audio_rms`、`audio_zcr`、`audio_centroid_hz`、`audio_tempo_bpm`、`audio_tempo_source` 等 |
| 源分离特征 | `source_drums_energy_ratio`、`source_bass_energy_ratio`、`source_vocal_energy_ratio`、`source_instrumental_energy_ratio` 等 |
| 歌词语义 | `lyric_semantic_tags`、`lyric_semantic_scores`、`lyric_semantic_model`、`lyric_semantic_error` |
| 评论语义 | `comment_semantic_tags`、`comment_semantic_scores`、`comment_semantic_model`、`comment_semantic_error` |
| MERT 表征 | `mert_embedding_path`、`mert_embedding_dim`、`mert_layer`、`mert_emotion_tags`、`mert_valence`、`mert_arousal`、`mert_cluster`、`mert_neighbor_song_ids` |

多数多标签字段使用 ` | ` 分隔。

### `data/matches/*_song_matches.csv`

本地音频文件与歌曲主表的匹配结果。

| 字段 | 含义 |
| --- | --- |
| `file_path` | 本地音频路径 |
| `song_id` / `name` / `artist_names` | 匹配到的歌曲 |
| `match_score` | 综合匹配分数 |
| `match_reason` | 标题、艺人、时长等匹配细节 |
| `match_source` | 匹配来源，例如 `metadata` |
| `audio_title` / `audio_artist` / `audio_album` | 音频文件元数据 |
| `local_duration_seconds` / `duration_diff_seconds` | 本地音频时长和差值 |
| `duration_error` | 时长读取错误信息 |

### `data/features/audio/*_song_features.*`

本地音频文件上提取的音频特征，提供 CSV 和 Parquet 两种格式。

主要字段包括：

- `audio_duration_seconds`
- `audio_sample_rate`
- `audio_rms`
- `audio_zcr`
- `audio_centroid_hz`
- `audio_vocal_band_ratio`
- `audio_crest`
- `audio_tempo_bpm`
- `audio_onset_strength`
- `audio_feature_tags`
- `source_*_energy_ratio`
- `audio_tempo_raw_bpm`
- `audio_tempo_source`

### `data/features/lyric/*_lyric_semantics.*`

歌词语义标签表。语义模型字段显示为 `bge-m3`，低置信或无歌词的样本可能为空。

| 字段 | 含义 |
| --- | --- |
| `song_id` | 歌曲 ID |
| `lyric_semantic_tags` | 歌词语义标签 |
| `lyric_semantic_scores` | 标签分数 |
| `lyric_semantic_model` | 使用的语义模型 |
| `lyric_semantic_error` | 错误或回退信息 |

### `data/features/comment/*_comment_semantics.*`

评论语义标签表，结构与歌词语义相同。常见标签包括回忆共鸣、治愈共鸣、悲伤共鸣、热血共鸣、故事感等。部分样本会使用关键词规则回退，相关信息记录在 `comment_semantic_error` 中。

### `data/mert/*_mert_index.csv`

MERT 音乐表征索引表。

| 字段 | 含义 |
| --- | --- |
| `song_id` | 歌曲 ID |
| `mert_embedding_path` | 生成时记录的 embedding 路径 |
| `mert_error` | MERT 处理错误 |
| `mert_chunks` | 切片数量 |
| `mert_embedding_dim` | embedding 维度，当前为 1024 |
| `mert_layer` | 使用层，当前多为 `mean` |
| `mert_emotion_tags` | 启发式 MERT 情绪标签 |
| `mert_emotion_scores` | 情绪代理分数 |
| `mert_valence` / `mert_arousal` | 启发式效价/唤醒度 |
| `mert_cluster` | 聚类编号 |
| `mert_neighbor_song_ids` / `mert_neighbor_scores` | 近邻歌曲和相似度 |

实际 `.npy` 文件位于 `data/mert/embeddings/{song_id}.npy`。如果表内路径与仓库目录不一致，建议按 `song_id` 重新拼接 embedding 路径。

### `data/mert/*_mert_clusters.csv`

MERT 聚类和近邻结果的轻量表，仅保留：

- `song_id`
- `mert_cluster`
- `mert_neighbor_song_ids`
- `mert_neighbor_scores`

## Loading Examples

### Load CSV Tables

```python
from datasets import load_dataset

dataset = load_dataset(
    "Ink-bai/Music-info-datasets",
    data_files={
        "ink_bai_liked_source": "data/source/ink_bai_liked_songs.csv",
        "middle_ages_source": "data/source/middle_ages_songs.csv",
        "ink_bai_liked_tags": "data/tags/ink_bai_liked_song_tags.csv",
        "middle_ages_tags": "data/tags/middle_ages_song_tags.csv",
        "ink_bai_liked_matches": "data/matches/ink_bai_liked_song_matches.csv",
        "middle_ages_matches": "data/matches/middle_ages_song_matches.csv",
    },
)

print(dataset["ink_bai_liked_tags"][0])
```

### Load Parquet Audio Features

```python
from datasets import load_dataset

audio_features = load_dataset(
    "Ink-bai/Music-info-datasets",
    data_files={
        "ink_bai_liked": "data/features/audio/ink_bai_liked_song_features.parquet",
        "middle_ages": "data/features/audio/middle_ages_song_features.parquet",
    },
)
```

### Load MERT Embeddings

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("data")
mert_index = pd.read_csv(root / "mert" / "ink_bai_liked_mert_index.csv", dtype={"song_id": str})

row = mert_index.dropna(subset=["mert_embedding_dim"]).iloc[0]
song_id = row["song_id"]
embedding = np.load(root / "mert" / "embeddings" / f"{song_id}.npy")

print(song_id, embedding.shape, embedding.dtype)
```

## Intended Uses

适合：

- 个人音乐库可视化和检索系统
- 音乐标签体系构建
- 歌词和评论语义分析
- 音乐推荐原型、召回或重排特征实验
- MERT embedding 相似度检索和聚类实验
- 音频特征、歌词特征、评论特征的多模态融合研究

不适合：

- 直接训练商用音乐情绪分类器
- 作为版权清晰的歌词/音频再分发数据集
- 评估通用音乐推荐模型的无偏基准
- 依赖本地路径字段进行跨机器复现实验

## Data Processing Notes

数据由本地流水线整理，主要步骤包括：

1. 调用网易云音乐相关接口采集歌曲详情、播放可用性、评论、歌词、相似歌曲和音质信息。
2. 将每首歌的原始响应保存为 JSON 快照，并导出统一的 `*_songs.csv` 主表。
3. 抽取歌词到 `source/lyrics/{song_id}.txt`。
4. 将歌曲主表与本地音乐文件做元数据和时长匹配，生成 `matches/`。
5. 对匹配到的本地音频提取节奏、能量、谱质心、过零率、频段能量等特征，生成 `features/audio/`。
6. 对歌词和评论生成语义标签，生成 `features/lyric/` 和 `features/comment/`。
7. 使用 MERT 提取 1024 维音乐 embedding，并生成启发式情绪、valence/arousal、聚类和近邻结果。
8. 将多源结果融合回 `tags/*_song_tags.csv`，作为推荐和检索的主入口。

MERT 情绪标签、valence/arousal 和部分语义标签为启发式/模型辅助结果，适合辅助检索和初筛，不应视为人工标注的可靠事实标签。

## Limitations

- 数据主要来自个人歌单和主题歌单，分布具有明显个人偏好，不能代表全部音乐。
- `local_audio_path`、`file_path` 等字段包含作者本地环境路径，仅作追溯，不具备通用可访问性。
- 不包含音频文件；音频特征和 MERT embedding 只覆盖已匹配到本地音频的歌曲。
- 歌词、评论和平台元数据可能包含版权内容或用户生成内容，请按来源平台规则和适用法律使用。
- 平台 URL、音质信息和可播放性状态具有时效性，可能在下载后发生变化。
- 自动标签可能存在误判、漏标和语言偏差。

## License and Usage

本仓库未声明开放版权授权。数据中包含来自音乐平台的元数据、歌词摘要、歌词文本、评论和由本地音频派生的特征。请仅在符合法律、平台条款和原始权利人要求的前提下，用于研究、学习或个人分析。

如需公开发布衍生模型、商业使用或再分发包含歌词/评论/平台元数据的内容，请自行确认授权和合规性。

## Citation

如果这个数据集对你的实验有帮助，可以引用本数据集页面：

```bibtex
@misc{music_info_datasets,
  title = {Music Info Datasets},
  author = {Ink-bai},
  year = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Ink-bai/Music-info-datasets}}
}
```

## Acknowledgements

本数据集整理流程使用了本地音乐信息处理流水线，并参考/调用网易云音乐相关接口封装能力。感谢开源音乐信息检索社区、MERT 模型和中文语义模型生态提供的基础工具。
