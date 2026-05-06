# MUS-Gacha

墨白的音乐仓库：从网易云喜欢列表、歌词和本地音频中整理音乐资料，生成可解释的多维标签，并用 Streamlit 做检索、筛选、推荐和本地播放。

当前项目已经支持：

- 从网易云歌曲 JSON 导出主 CSV 和歌词文件。
- 从 `data/ink_bai_liked_songs.csv`、`data/lyrics`、`H:\音乐` 融合生成标签。
- 匹配本地音频到 `song_id`，支持一键打开本地文件。
- 提取音频特征：节奏、能量、明亮度、动态、人声/器乐估计。
- 批量提取 MERT embedding，生成聚类、近邻歌曲和启发式情绪标签。
- 在 Streamlit 页面中浏览、筛选、查看歌词、播放本地音频。

## 目录结构

```text
.
├── app.py                              # Streamlit 音乐仓库应用
├── mert_emotion_demo.py                # 单曲 MERT 情绪/embedding demo
├── requirements.txt                    # 基础依赖
├── data_get/
│   └── qcloud_song_store.py            # 抓取网易云歌曲 JSON
├── data_processing/
│   ├── export_liked_json_to_csv.py     # JSON 导出 CSV/歌词
│   └── build_song_tags.py              # 多源融合打标签
├── datacache/                          # Streamlit 预处理缓存
├── data/
│   ├── ink_bai_liked_json/             # 每首歌的原始 JSON
│   ├── lyrics/                         # song_id.txt 歌词
│   ├── ink_bai_liked_songs.csv         # 歌曲主表
│   ├── song_tags.csv                   # 标签主输出
│   ├── audio_song_matches.csv          # 本地音频匹配结果
│   ├── audio_features.csv              # 音频特征 CSV
│   ├── audio_features.parquet          # 音频特征 parquet
│   ├── mert_index.csv                  # MERT embedding/情绪/近邻索引
│   ├── mert_clusters.csv               # MERT 聚类/近邻表
│   └── mert_embeddings/                # 每首歌一个 .npy embedding
└── MERT-v1-330M/                       # 本地 MERT 模型目录
```

`data/`、`datacache/`、`MERT-v1-330M/` 和 `.cache/` 默认不进 Git。

## 安装

基础依赖：

```powershell
pip install -r requirements.txt
```

推荐额外依赖：

```powershell
pip install scikit-learn pyarrow mutagen
```

用途：

- `scikit-learn`：MERT 聚类和近邻搜索。
- `pyarrow`：写入 `audio_features.parquet`。
- `mutagen`：更稳定地读取 MP3/FLAC/M4A 等音频元数据。

如果要跑 MERT，需要本地已有 `MERT-v1-330M` 目录。当前项目默认使用：

```text
.\MERT-v1-330M
```

## 快速启动应用

已有 `data/ink_bai_liked_songs.csv` 和 `data/song_tags.csv` 时，直接启动：

```powershell
streamlit run app.py
```

打开：

```text
http://localhost:8501
```

页面功能：

- 实时检索歌名、歌手、专辑、歌词、评论。
- 按歌手、语言、智能标签、音质、年份等筛选。
- 按 XP-Gacha 风格动态调节推荐评分：全局维度倍率、标签屏蔽、单标签权重、歌手权重、歌名关键词权重、歌词关键词权重、评论语义权重。
- 查看推荐总览、歌曲列表、歌词检索、歌曲详情。
- 查看每首歌的评分拆解、歌词关键词、评论语义、本地音频/MERT 信息。
- 有本地音频时显示本地播放条，并可点击“打开本地音频”调用系统默认播放器。

### 预处理缓存

应用启动时会根据 `data/ink_bai_liked_songs.csv`、`data/song_tags.csv`、`data/lyrics/*.txt` 和预处理缓存版本生成哈希。哈希命中时，Streamlit 会直接读取 `datacache/preprocessed_music.pkl`，跳过 CSV 合并、歌词读取、分词、评论语义标签和评分资源重建。

当主 CSV、标签 CSV 或歌词文件发生变化时，`datacache/data.hash` 会失效，下一次启动会自动重建预处理缓存。若缓存结构升级导致异常，可删除 `datacache/` 后重新运行 `streamlit run app.py`。

## 数据准备

### 1. 抓取歌曲 JSON

如果已经有 `data/ink_bai_liked_json`，可以跳过。

按自己的网易云喜欢歌单名抓取：

```powershell
python data_get\qcloud_song_store.py `
  --my-playlist-name "Ink_bai喜欢的音乐" `
  --cookie-file cookie.txt `
  --json-dir data\ink_bai_liked_json `
  --db data\ink_bai_liked.sqlite3
```

也可以按歌单 ID 抓取：

```powershell
python data_get\qcloud_song_store.py `
  --playlist-id 你的歌单ID `
  --json-dir data\ink_bai_liked_json `
  --db data\ink_bai_liked.sqlite3
```

更详细的抓取说明见 `QCloudSongStore_README.md`。

### 2. 导出 CSV 和歌词

```powershell
python data_processing\export_liked_json_to_csv.py `
  --input-dir data\ink_bai_liked_json `
  --output data\ink_bai_liked_songs.csv `
  --lyrics-dir data\lyrics
```

如果希望 CSV 里也包含完整歌词列：

```powershell
python data_processing\export_liked_json_to_csv.py `
  --input-dir data\ink_bai_liked_json `
  --output data\ink_bai_liked_songs.csv `
  --lyrics-dir data\lyrics `
  --include-lyrics
```

## 生成标签

标签生成脚本是：

```text
data_processing\build_song_tags.py
```

默认输入：

```text
data\ink_bai_liked_songs.csv
data\lyrics
H:\音乐
```

### 只生成文本/歌词/本地匹配标签

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐"
```

输出：

```text
data\song_tags.csv
data\song_tags.jsonl
data\audio_song_matches.csv
```

### 生成音频特征

复用已有匹配结果，并分析每首本地音频前 20 秒：

```powershell
python data_processing\build_song_tags.py `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --analyze-audio `
  --audio-feature-seconds 20
```

额外输出：

```text
data\audio_features.csv
data\audio_features.parquet
```

会回写到 `song_tags.csv` 的字段包括：

- `audio_tempo_bpm`
- `audio_onset_strength`
- `audio_rms`
- `audio_zcr`
- `audio_centroid_hz`
- `audio_crest`
- `audio_vocal_band_ratio`
- `audio_feature_tags`
- `vocal_presence_score`
- `instrumental_presence_score`
- `vocal_instrumental_tags`

### 生成 MERT embedding 和聚类

推荐先确认 GPU 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

小批量测试：

```powershell
python data_processing\build_song_tags.py `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --extract-mert `
  --mert-limit 4 `
  --mert-max-seconds 12 `
  --mert-fp16
```

全量提取：

```powershell
python data_processing\build_song_tags.py `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --extract-mert `
  --mert-max-seconds 12 `
  --mert-fp16
```

额外输出：

```text
data\mert_embeddings\*.npy
data\mert_index.csv
data\mert_clusters.csv
```

会回写到 `song_tags.csv` 的字段包括：

- `mert_embedding_path`
- `mert_emotion_tags`
- `mert_valence`
- `mert_arousal`
- `mert_cluster`
- `mert_neighbor_song_ids`
- `mert_neighbor_scores`

注意：MERT 是自监督音乐表征模型。这里的 `mert_emotion_tags` 来自启发式情绪代理，不是经过人工标注数据训练出的可靠情绪分类器。

## 推荐评分算法

应用里的推荐评分参考 XP-Gacha 的交互方式：先把歌曲拆成多种可计分特征，再在侧边栏动态调整每个维度和每个单项的权重。分数不是固定 0-100，而是随当前权重配置动态变化。

### 评分维度

当前参与评分的维度：

```text
综合标签
语种
风格
情绪
主题
场景
音频标签
歌手
歌名/专辑关键词
歌词关键词
评论语义
热度
歌词完整度
音质
可播放
本地音频
MERT 可用
```

每个维度都有独立倍率。倍率为 `0` 时，该维度不参与推荐分。

### 标签和单项权重

侧边栏支持：

- 屏蔽标签：命中任意屏蔽标签的歌曲会直接从候选结果里移除。
- 单标签权重：对任意智能标签加权或降权，例如 `人声强 = 1.4`、`悲伤 = 0.8`。
- 歌手权重：对指定歌手额外加权。
- 歌名关键词权重：对歌名/专辑关键词加权。
- 歌词关键词权重：对歌词分词得到的关键词加权。
- 评论语义权重：对评论内容分类出的语义标签加权。

单项权重可以设置为负数，用于主动降权。

### 歌名/专辑关键词

歌名关键词不再直接使用完整歌名。应用会从：

```text
name
aliases
translations
album_name
```

里提取关键词，并按语种分词：

- 中文/粤语/华语：`jieba`
- 日语：`janome`
- 英语：`nltk.wordpunct_tokenize`
- 缺少对应库时回退到正则分词

之后会过滤完整歌名、完整别名、完整专辑名、停用词和过长中文短语。例如：

```text
光阴的故事 => 光阴 / 故事
铁血丹心 => 电视剧 / 射雕 / 英雄传 / 主题曲
你笑起来真好看 => 起来 / 好看
```

### 歌词关键词

歌词关键词来自 `data/lyrics/{song_id}.txt` 或 CSV 中的歌词摘要。处理流程：

1. 清理 LRC 时间轴、作词/作曲/编曲等元信息。
2. 按语种选择分词库。
3. 过滤停用词、纯数字、过短词、占位词。
4. 每首歌保留高频关键词，写入页面运行时的 `lyric_terms`。

这些关键词不会写回 CSV，而是在 Streamlit 加载数据时动态生成。

### 评论语义

“评论倍率”不再使用评论数量直接加分。评论数量仍可用于筛选和排序，但推荐分里的评论维度改为分析评论内容。

应用会读取：

```text
first_hot_comment
first_comment
```

并按关键词规则生成评论语义标签：

```text
回忆共鸣
治愈共鸣
悲伤共鸣
热血共鸣
好听认可
故事感
幽默吐槽
亲情陪伴
影视回忆
```

这些标签参与“评论语义倍率”和“评论语义权重配置”。

### 评分拆解

列表和详情页会显示 `评分拆解`，格式类似：

```text
综合标签:84.5 | 情绪:70.8 | 歌词关键词:46.8 | 风格:57.7
```

它展示当前权重下贡献最高的几个维度，方便判断为什么这首歌被排到前面。

## 标签体系

每首歌是多标签、多来源融合，不只打一个标签。

主要维度：

```text
基础标签：国语 / 粤语 / 英语 / 日语 / 韩语 / 纯音乐 / 本地音频
风格标签：华语流行 / 粤语流行 / 轻音乐 / 古风 / 摇滚 / 电子 / 民谣 / 钢琴 / 管弦 / 游戏音乐 / 影视原声
情绪标签：治愈 / 欢快 / 悲伤 / 怀旧 / 热血 / 宁静 / 孤独 / 浪漫 / 史诗 / 紧张 / 忧郁
主题标签：爱情 / 离别 / 成长 / 时光 / 江湖 / 旅途 / 梦想 / 思念 / 自然
场景标签：学习 / 睡前 / 通勤 / 运动 / 工作 / 回忆杀 / 放松 / 燃向 / 年代标签
音频标签：快节奏 / 慢节奏 / 高能量 / 低能量 / 明亮 / 柔和 / 动态大 / 人声强 / 器乐强
评论语义：回忆共鸣 / 治愈共鸣 / 悲伤共鸣 / 热血共鸣 / 好听认可 / 故事感 / 幽默吐槽
```

标签来源会记录在 `tag_sources` 字段中，方便解释“为什么有这个标签”。

## 本地音频匹配逻辑

匹配目标是把 `H:\音乐` 中的文件对应到 CSV 的 `song_id`。

优先级：

1. 读取音频元数据 `title / artist / album`。
2. 从文件名解析 `艺术家 - 歌名`。
3. 与 CSV 中 `name / aliases / artist_names` 做模糊匹配。
4. 用本地音频时长和 CSV 的 `duration_seconds` 做加分或降权。
5. 低于阈值的文件不会自动写入匹配表。

匹配结果在：

```text
data\audio_song_matches.csv
```

重要字段：

- `file_path`
- `song_id`
- `name`
- `artist_names`
- `match_score`
- `match_reason`
- `match_source`
- `audio_title`
- `audio_artist`
- `audio_album`
- `local_duration_seconds`
- `duration_diff_seconds`

## 单曲 MERT demo

`mert_emotion_demo.py` 可以单独分析一首 FLAC：

```powershell
python .\mert_emotion_demo.py `
  --model-dir .\MERT-v1-330M `
  --audio "H:\音乐\demo.flac" `
  --max-seconds 30 `
  --fp16
```

保存 JSON：

```powershell
python .\mert_emotion_demo.py `
  --model-dir .\MERT-v1-330M `
  --audio "H:\音乐\demo.flac" `
  --max-seconds 30 `
  --output-json emotion_result.json
```

使用微调分类头：

```powershell
python .\mert_emotion_demo.py `
  --model-dir .\MERT-v1-330M `
  --audio "H:\音乐\demo.flac" `
  --mode classifier `
  --head-checkpoint .\emotion_head.pt
```

## 常用命令速查

启动应用：

```powershell
streamlit run app.py
```

重新生成基础标签：

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐"
```

重跑音频特征：

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐" --reuse-matches --analyze-audio --audio-feature-seconds 20
```

重跑 MERT：

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐" --reuse-matches --extract-mert --mert-max-seconds 12 --mert-fp16
```

强制重算已有 MERT embedding：

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐" --reuse-matches --extract-mert --overwrite-mert --mert-max-seconds 12 --mert-fp16
```

查看脚本参数：

```powershell
python data_processing\build_song_tags.py --help
python data_processing\export_liked_json_to_csv.py --help
python data_get\qcloud_song_store.py --help
python mert_emotion_demo.py --help
```

## 常见问题

### 部分 M4A 分析失败

`torchaudio` 可能无法识别某些 `.m4a`：

```text
Format not recognised
```

这种文件仍可能被匹配到本地路径，但音频特征和 MERT embedding 会为空。可以转成 FLAC/MP3 后重新跑。

### MERT 第一次运行写缓存失败

项目会把 HuggingFace 动态模块缓存写到：

```text
.cache\huggingface
```

如果看到权限错误，确认项目目录可写，并删除异常的 `.cache` 后重跑。

### MERT 情绪标签不准

当前 `mert_emotion_tags` 是启发式标签，适合辅助检索和初筛。要做可靠情绪分类，需要准备人工标注数据并训练分类头。

### 人声强/器乐强不是源分离

当前逻辑使用歌词信息、纯音乐提示和频段能量做启发式估计。它不是 Demucs、Spleeter 这类源分离模型的结果。

### 后台启动 Streamlit 不常驻

某些执行环境会杀掉后台进程。推荐在终端前台运行：

```powershell
streamlit run app.py
```

## 当前推荐工作流

日常使用：

```powershell
streamlit run app.py
```

新增或移动了本地音乐后：

```powershell
python data_processing\build_song_tags.py --audio-dir "H:\音乐"
python data_processing\build_song_tags.py --audio-dir "H:\音乐" --reuse-matches --analyze-audio --audio-feature-seconds 20
python data_processing\build_song_tags.py --audio-dir "H:\音乐" --reuse-matches --extract-mert --mert-max-seconds 12 --mert-fp16
```

然后重新打开或刷新 Streamlit 页面。
