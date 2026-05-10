# MUS-Gacha / 墨白的音乐仓库

一个基于 `Streamlit` 的本地音乐资料整理、检索与推荐系统。

它把「网易云歌曲采集 / 原始 JSON 导出 CSV / 歌词整理 / 本地音频匹配 / 音频特征分析 / MERT embedding / 多维标签推荐 / 历史偏好加权」串成一条完整流程，用来搭建一个更懂自己口味的本地音乐仓库。

当前实现明显偏向 Windows 本地环境使用：

- 支持直接打开本地音频文件（`os.startfile`）
- 默认本地音乐目录示例为 `H:\音乐`
- QCloudMusicApi 动态库路径按 Windows 编译产物组织
- MERT、HDemucs 等模型默认放在项目根目录的 `models/` 下

## ✨ 功能概览

- 启动时自动汇总读取 `data/source/*.csv`，按 `song_id` 去重后进入曲库。
- 从网易云歌曲 JSON 导出歌曲 CSV、歌词 TXT、评论摘要、相似歌曲和音质信息。
- 从源 CSV、`data/source/lyrics` 和本地音乐目录融合生成可解释标签。
- 支持歌名、歌手、专辑、歌词、评论的实时关键词检索。
- 支持歌手、语种/场景、智能标签、音质、年份、热度、评论数等筛选。
- 支持全局维度倍率、标签屏蔽、单标签权重、歌手权重、标题关键词权重、歌词关键词权重、歌词语义权重、评论语义权重。
- 支持记录最近 N 次选中/打开行为，包含勾选歌曲、本地音频、网易云详情页和表格链接，并基于历史偏好自动加权推荐。
- 支持推荐总览、歌曲列表、歌曲详情、历史记录和数据处理独立页面。
- 支持全局曲库画像、当前筛选画像和个人历史画像图表。
- 匹配本地音频到 `song_id`，详情页可播放本地文件并一键打开系统播放器。
- 支持普通音频特征：节奏、能量、明亮度、动态、人声/器乐估计。
- 支持 HDemucs 声源分离，输出 `vocals / drums / bass / other` 能量占比。
- 支持 MERT embedding，生成启发式情绪、valence/arousal、聚类和近邻歌曲。
- 数据处理页支持网易云采集、限流重试、JSON 导 CSV/歌词、标签生成、音频特征、MERT、缓存维护和脚本实时输出。

## 🧮 推荐评分算法

推荐分不是固定 0-100，而是随着侧边栏权重配置动态变化。应用会先把每首歌拆成多种可计分特征，再把这些特征按当前偏好组合成 `dynamic_score`。

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

每个维度都有独立倍率。倍率为 `0` 时，该维度不参与推荐分。列表和详情页会显示 `评分拆解`，格式类似：

```text
综合标签:84.5 | 情绪:70.8 | 歌词关键词:46.8 | 风格:57.7
```

它展示当前权重下贡献最高的几个维度，方便判断为什么这首歌被排到前面。

### 历史偏好加权

应用会把最近 `HISTORY_RECOMMENDATION_CACHE_SIZE` 次交互记录保存到：

```text
datacache/recommendation_history.json
```

默认保留最近 80 次记录。当前会写入历史的动作包括：

- 在推荐总览或歌曲列表中勾选 `选中`
- 在歌曲详情页打开本地音频
- 在歌曲详情页打开网易云页面
- 在推荐总览或歌曲列表中点击网易云表格链接

网易云链接会先经过本地跳转追踪地址记录一次历史，再重定向到真实网易云页面。这个服务默认只监听本机 `127.0.0.1:8766`，用于区分“只是看到推荐”和“实际点开歌曲”。

历史偏好会基于这些记录统计综合标签、语种、风格、情绪、主题、场景、音频标签、歌手、歌名关键词、歌词关键词、歌词语义和评论语义。越少见但又反复出现在历史里的特征，额外加分越高。

历史分会出现在 `评分拆解` 的 `历史偏好` 中。`历史偏好总分倍率` 为 `0` 时完全关闭历史偏好加权。历史记录页也支持刷新、清空、删除选中记录，以及关闭“选中歌曲算入历史记录”。这个开关只影响勾选歌曲；打开本地音频和网易云链接仍会记入历史。

### 单项权重

侧边栏支持：

- 屏蔽标签：命中任意屏蔽标签的歌曲会直接从候选结果里移除。
- 单标签权重：对任意智能标签加权或降权，例如 `人声强 = 1.4`、`悲伤 = 0.8`。
- 歌手权重：对指定歌手额外加权。
- 歌名关键词权重：对歌名/专辑关键词加权。
- 歌词关键词权重：对歌词分词得到的关键词加权。
- 歌词语义权重：对歌词内容分析出的语义意象加权。
- 评论语义权重：对评论内容分类出的语义标签加权。

单项权重可以设置为负数，用于主动降权。

## 🗂️ 项目结构

```text
MUS-Gacha/
├─ app.py                                  # Streamlit 主界面
├─ config.py                               # 路径、缓存、标签列和默认权重配置
├─ data_pipeline.py                        # 源 CSV 汇总、预处理缓存、评分资源和动态评分
├─ ui_components.py                        # 页面样式和歌曲详情组件
├─ ui_data_processing.py                   # 数据处理可视化页面与脚本实时输出
├─ utils_core.py                           # 本地文件打开、音质/语种和展示字段辅助
├─ utils_text.py                           # 文本清洗、分词和歌词关键词提取
├─ utils_charts.py                         # 全局/筛选/历史画像图表
├─ utils_history.py                        # 历史记录、历史设置和历史偏好加权
├─ QCloudSongStore_README.md               # 网易云采集脚本详细说明
├─ requirements.txt                        # 基础依赖
├─ .streamlit/
│  └─ config.toml                          # Streamlit 主题配置
├─ data_get/
│  ├─ qcloud_song_store.py                 # 调用 QCloudMusicApi 抓取网易云歌曲 JSON
│  └─ retry_rate_limited_songs.py          # 重试包含“操作频繁”的 JSON 快照
├─ data_processing/
│  ├─ export_original_json_to_csv.py       # 原始 JSON 导出 CSV/歌词
│  ├─ build_song_tags.py                   # 多源融合打标签、音频分析、MERT
│  ├─ build_mert_emotion.py                # 单曲 MERT 情绪/embedding 分析
│  ├─ build_lyric_semantics.py             # 使用本地 bge-m3 单独生成歌词语义
│  └─ build_comment_semantics.py           # 使用本地 bge-m3 单独生成评论语义
├─ data/
│  ├─ source/                              # 源数据：JSON、SQLite、可聚合歌曲 CSV、歌词
│  │  ├─ my_playlist_json/                 # 每首歌的原始 JSON 快照示例
│  │  ├─ lyrics/                           # song_id.txt 歌词
│  │  ├─ *.csv                             # 应用会读取全部 CSV 并按 song_id 去重
│  │  ├─ my_playlist_songs.csv             # export_original_json_to_csv.py 输出示例
│  │  └─ my_playlist.sqlite3               # 抓取缓存数据库示例
│  ├─ tags/
│  │  ├─ *_song_tags.csv                   # 最终标签总表，应用只从这里读取生成结果
│  │  └─ *_song_tags.jsonl                 # JSONL 版标签输出，供查看/归档
│  ├─ matches/
│  │  └─ *_song_matches.csv                # 本地音频匹配功能产物
│  └─ features/
│     ├─ lyric/
│     │  ├─ *_lyric_semantics.csv          # 歌词语义功能产物
│     │  └─ *_lyric_semantics.jsonl
│     ├─ comment/
│     │  ├─ *_comment_semantics.csv        # 评论语义功能产物
│     │  └─ *_comment_semantics.jsonl
│     ├─ audio/
│     │  ├─ *_song_features.csv            # 普通音频特征功能产物
│     │  └─ *_song_features.parquet
│     └─ mert/
│        ├─ embeddings/                    # 每首歌一个 .npy embedding
│        ├─ *_mert_index.csv               # MERT 情绪与 embedding 索引功能产物
│        └─ *_mert_clusters.csv            # 聚类与近邻歌曲功能产物
├─ datacache/                              # Streamlit 预处理缓存与历史记录
├─ models/                                 # 本地模型目录，含 MERT 和 HDemucs
└─ QCloudMusicApi/                         # QCloudMusicApi 源码/编译产物
```

`data/`、`datacache/`、`models/`、`.cache/`、`.streamlit/` 和 `cookie.txt` 默认不进 Git。

### 数据流

1. 用 `data_get/qcloud_song_store.py` 调用 QCloudMusicApi 抓取网易云歌曲信息，写入 SQLite 和每首歌一个 JSON 快照。
2. 用 `data_processing/export_original_json_to_csv.py` 把 JSON 快照导出为歌曲 CSV，并抽取歌词到 `data/source/lyrics/{song_id}.txt`。
3. 把一个或多个歌曲 CSV 放到 `data/source/` 下。
4. 启动 `app.py` 后，应用扫描 `data/source/*.csv`，按 `song_id` 去重汇总为曲库。
5. 应用只批量合并 `data/tags/*song_tags.csv` 和本地歌词，生成预处理缓存。匹配、音频特征、歌词语义、评论语义和 MERT 功能产物需要先回写到对应的 `*_song_tags.csv`。
6. 页面中按照动态推荐分、关键词检索、侧边栏筛选条件进行展示。
7. 用户选中歌曲、打开本地音频、打开网易云页面或点击表格链接时，应用写入历史记录，用于历史偏好加权和历史画像图表。
8. 新增本地音乐、重跑标签、重跑音频特征或 MERT 后，刷新 Streamlit 页面即可重新读取结果。

## 💻 运行环境

建议环境：

- Python `3.10+`
- Windows
- 本地可用的 `QCloudMusicApi.dll/.so/.dylib`
- 如果要抓取登录态歌单，需要有效的网易云 Cookie
- 如果要跑本地音频匹配，准备本地音乐目录，例如 `H:\音乐`
- 如果要跑 MERT，准备本地 `models/MERT-v1-330M`
- 如果要跑 HDemucs 声源分离，准备对应 checkpoint，或允许 torchaudio 使用默认缓存

基础依赖：

```powershell
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

```text
streamlit
pandas
numpy
torch
torchaudio
transformers
```

推荐额外依赖：

```powershell
pip install scikit-learn pyarrow mutagen jieba janome nltk tqdm
```

用途：

- `scikit-learn`：MERT 聚类和近邻搜索。
- `pyarrow`：写入 `audio_features.parquet`。
- `mutagen`：更稳定地读取 MP3/FLAC/M4A 等音频元数据。
- `jieba` / `janome` / `nltk`：中文、日语、英语分词；缺少时会回退到正则分词。
- `tqdm`：脚本命令行进度条。

## ⚙️ 如何开始

### 0. 可选：直接下载示例数据集

如果只是想先体验应用，或不想从网易云重新采集数据，可以直接下载已经整理好的 Hugging Face 数据集：

[Ink-bai/Music-info-datasets](https://huggingface.co/datasets/Ink-bai/Music-info-datasets)

下载后把数据集仓库里的 `data/` 目录放到本项目根目录，保持下面这些路径存在即可：

```text
data/source/*.csv
data/tags/*_song_tags.csv
data/source/lyrics/*.txt
```

如果需要音频特征和 MERT 相似度，也保留：

```text
data/matches/
data/features/
data/mert/
```

命令行方式示例：

```powershell
git lfs install
git clone https://huggingface.co/datasets/Ink-bai/Music-info-datasets hf_music_info_datasets
Copy-Item -Recurse hf_music_info_datasets\data .\data
```

如果你已经有自己的 `data/` 目录，建议先改名备份，或只拷贝需要的子目录，避免覆盖自己的采集结果。

### 1. 准备 QCloudMusicApi

项目里的 `data_get/qcloud_song_store.py` 会调用 QCloudMusicApi 动态库。默认会搜索：

```text
QCloudMusicApi/build/QCloudMusicApi/QCloudMusicApi.dll
QCloudMusicApi/build/bin/QCloudMusicApi.dll
QCloudMusicApi/build/Release/QCloudMusicApi.dll
```

如果未找到动态库，按提示编译：

```powershell
cmake -S QCloudMusicApi -B QCloudMusicApi/build -DQCLOUDMUSICAPI_BUILD_TEST=OFF -DQCLOUDMUSICAPI_BUILD_SHARED=ON
cmake --build QCloudMusicApi/build --config Release
```

抓取脚本还支持显式传入：

```powershell
--library QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll
```

更详细的采集说明见 `QCloudSongStore_README.md`。

### 2. 准备 Cookie

如果要抓自己的喜欢歌单或私有歌单，把网易云 Cookie 放到项目根目录：

```text
cookie.txt
```

`cookie.txt` 已在 `.gitignore` 中，避免误提交。

### 3. 准备模型资源

MERT 默认目录：

```text
models/
└─ MERT-v1-330M/
```

HDemucs 声源分离默认优先读取：

```text
models/hdemucs_high_trained.pt
models/hdemucs_high_musdbhq_only.pt
```

如果本地 checkpoint 不存在，脚本会尝试使用 torchaudio 的默认模型加载路径。离线环境建议提前把权重放到 `models/` 下。

### 4. 自定义主题色

项目使用 Streamlit 项目级主题配置：

```text
.streamlit/config.toml
```

当前主题配置：

```toml
[theme]
primaryColor = "#755bbb"

[theme.light]
backgroundColor = "#FFFDF8"
secondaryBackgroundColor = "#F3EEE7"
textColor = "#1F1F1F"
borderColor = "#D9D1C7"

[theme.dark]
backgroundColor = "#121714"
secondaryBackgroundColor = "#1D2520"
textColor = "#EAF2EC"
borderColor = "#334039"
```

保存后刷新页面；如果没有立即生效，重启 `streamlit run app.py`。

## 🚀 启动应用

已有至少一个 `data/source/*.csv` 时，直接启动：

```powershell
streamlit run app.py
```

打开：

```text
http://localhost:8501
```

页面当前支持：

- `推荐总览`：展示 Top 推荐候选、全局曲库画像、个人历史画像。
- `歌曲列表`：分页展示、全局排序、当前筛选画像、勾选歌曲。
- `歌曲详情`：展示封面、网易云链接、本地音频播放/打开、评分拆解、歌词、音频特征、MERT 信息。
- `历史记录`：查看、刷新、清空、删除历史条目，控制“选中歌曲是否算入历史记录”。打开本地音频和网易云链接仍会记录。
- `数据处理`：数据采集、CSV/歌词导出、标签与音频处理、缓存维护，以及脚本实时输出。
- 侧边栏：关键词检索、歌词专项检索、基础筛选、数值范围、动态评分权重、历史偏好倍率。

## 📖 文本、标签与语义说明

### 歌名/专辑关键词

应用会从这些字段中提取歌名关键词：

```text
name
aliases
translations
album_name
```

处理方式：

- 中文/粤语/华语：优先用 `jieba`
- 日语：优先用 `janome`
- 英语：优先用 `nltk.wordpunct_tokenize`
- 缺少对应库时回退到正则分词

随后会过滤完整歌名、完整别名、完整专辑名、停用词和过长中文短语。

### 歌词关键词

歌词关键词来自：

```text
data/source/lyrics/{song_id}.txt
```

或 CSV 中的歌词摘要。处理流程：

1. 清理 LRC 时间轴、作词/作曲/编曲等元信息。
2. 按语种选择分词库。
3. 过滤停用词、纯数字、过短词、占位词。
4. 每首歌保留高频关键词，写入运行时字段 `lyric_terms`。

这些关键词不会写回 CSV，而是在 Streamlit 加载数据时动态生成。

### 歌词语义

歌词语义不会在应用启动或主预处理时运行模型。需要在 `数据处理 -> 标签与音频 -> 歌词语义分析` 中单独执行，或使用命令行脚本：

```powershell
python data_processing\build_lyric_semantics.py `
  --input data\source\my_playlist_songs.csv `
  --lyrics-dir data\source\lyrics `
  --model-dir models\bge-m3
```

默认输出会按输入文件名推导，例如：

```text
data\features\lyric\my_playlist_lyric_semantics.csv
data\features\lyric\my_playlist_lyric_semantics.jsonl
data\tags\my_playlist_song_tags.csv
```

其中 `data\features\lyric` 保存歌词语义功能产物，`data\tags\*_song_tags.csv` 保存应用最终读取的字段。

### 评论语义

推荐分里的“评论语义”不直接使用评论数量，而是分析评论内容。应用读取：

```text
first_hot_comment
first_comment
```

评论语义不会在应用启动或主预处理时运行模型。需要在 `数据处理 -> 标签与音频 -> 评论语义分析` 中单独执行，或使用命令行脚本：

```powershell
python data_processing\build_comment_semantics.py `
  --input data\source\my_playlist_songs.csv `
  --model-dir models\bge-m3
```

脚本会读取本地 `bge-m3`，根据评论文本和语义标签描述的相似度生成：

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

默认输出会按输入文件名推导，例如：

```text
data\features\comment\my_playlist_comment_semantics.csv
data\features\comment\my_playlist_comment_semantics.jsonl
data\tags\my_playlist_song_tags.csv
```

其中 `data\features\comment` 保存评论语义功能产物，`data\tags\*_song_tags.csv` 保存应用最终读取的字段。如果模型置信度不足，默认会使用关键词规则回退；可以加 `--no-keyword-fallback` 关闭。输出文件如果已经存在，会按 `song_id` 追加合并，不覆盖旧数据。

这些标签参与“评论语义倍率”和“评论语义权重配置”。

### 标签体系

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

## 🛠️ 数据准备与维护

推荐优先使用应用内的 `数据处理` 页面操作。该页面已经把常用流程做成可视化表单，并提供脚本实时输出。

在标签、歌词语义、评论语义、音频特征和 MERT 表单中，`歌曲主表` 使用 `data/source/*.csv` 下拉选择。选择某个 CSV 后点击右侧 `确认`，页面会按文件名自动填充对应的标签总表、JSONL、音频匹配、歌词语义、评论语义、音频特征和 MERT 输出路径。

### 网易云歌曲采集

按自己的网易云喜欢歌单名抓取：

```powershell
python data_get\qcloud_song_store.py `
  --my-playlist-name "你的喜欢歌单名" `
  --cookie-file cookie.txt `
  --json-dir data\source\my_playlist_json `
  --db data\source\my_playlist.sqlite3 `
  --library QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll
```

也可以按歌单 ID 抓取：

```powershell
python data_get\qcloud_song_store.py `
  --playlist-id 你的歌单ID `
  --json-dir data\source\my_playlist_json `
  --db data\source\my_playlist.sqlite3
```

常用可选项：

- `--max-songs`：最多抓取多少首。
- `--workers`：并发歌曲数。
- `--endpoints default|all|接口列表`：控制抓取接口集合。
- `--levels standard,exhigh,lossless,hires`：控制音质 URL 抓取层级。
- `--comment-limit` / `--simi-limit`：评论和相似歌曲数量。
- `--resolve-host HOST=IP`：通过本地代理固定解析某些网易云域名。

### 操作频繁重试

扫描 JSON 快照中包含“操作频繁”的歌曲，并批量重抓：

```powershell
python data_get\retry_rate_limited_songs.py `
  --json-dir data\source\my_playlist_json `
  --db data\source\my_playlist.sqlite3 `
  --library QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --cookie-file cookie.txt
```

先只扫描不重抓：

```powershell
python data_get\retry_rate_limited_songs.py --json-dir data\source\my_playlist_json --dry-run
```

### 原始 JSON 导出 CSV 和歌词

```powershell
python data_processing\export_original_json_to_csv.py `
  --input-dir data\source\my_playlist_json `
  --output data\source\my_playlist_songs.csv `
  --lyrics-dir data\source\lyrics
```

如果希望 CSV 中也包含完整歌词列：

```powershell
python data_processing\export_original_json_to_csv.py `
  --input-dir data\source\my_playlist_json `
  --output data\source\my_playlist_songs.csv `
  --lyrics-dir data\source\lyrics `
  --include-lyrics
```

导出的 CSV 只要放在 `data/source` 下，就会被应用自动纳入汇总。多个 CSV 同时存在时不需要手动合并，应用会在启动时按 `song_id` 去重。

如果省略 `--input-dir`，脚本会选择第一个 `data/source/*_json` 目录；如果省略 `--output`，会按 JSON 目录名推导输出，例如 `data\source\my_playlist_json` 会生成 `data\source\my_playlist_songs.csv`。

### 生成基础标签和本地音频匹配

`build_song_tags.py` 一次处理一个源 CSV。Streamlit 曲库会读取 `data/source/*.csv` 的汇总结果，并只从 `data/tags/*song_tags.csv` 读取标签、匹配、音频特征、歌词语义、评论语义和 MERT 字段。`data/matches`、`data/features/audio`、`data/features/lyric`、`data/features/comment`、`data/features/mert` 只保留各自功能产物；运行对应脚本时会把最终字段回写到 `*_song_tags.csv`。

如果省略输出路径，各数据处理脚本会按 `--input` 的文件名推导数据集名前缀。例如 `data\source\my_playlist_songs.csv` 对应的默认结果路径是：

```text
data\tags\my_playlist_song_tags.csv
data\tags\my_playlist_song_tags.jsonl
data\matches\my_playlist_song_matches.csv
data\features\lyric\my_playlist_lyric_semantics.csv
data\features\lyric\my_playlist_lyric_semantics.jsonl
data\features\comment\my_playlist_comment_semantics.csv
data\features\comment\my_playlist_comment_semantics.jsonl
data\features\audio\my_playlist_song_features.csv
data\features\audio\my_playlist_song_features.parquet
data\features\mert\my_playlist_mert_index.csv
data\features\mert\my_playlist_mert_clusters.csv
```

```powershell
python data_processing\build_song_tags.py `
  --input data\source\my_playlist_songs.csv `
  --lyrics-dir data\source\lyrics `
  --audio-dir "H:\音乐"
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--match-threshold` | `0.84` | 本地音频与歌曲的模糊匹配最低分，越高越严格。 |
| `--reuse-matches` | 关闭 | 复用已有匹配表，不重新扫描 `--audio-dir`。日常重跑建议开启。 |
| `--no-progress` | 关闭 | 关闭进度条，适合后台日志或自动化脚本。 |

输出文件如果已经存在，脚本会追加合并而不是覆盖；同一 `song_id` 的同一字段会保留最新非空值。

### 生成音频特征

复用已有匹配结果，并分析每首本地音频前 20 秒：

```powershell
python data_processing\build_song_tags.py `
  --input data\source\my_playlist_songs.csv `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --analyze-audio `
  --audio-feature-seconds 20
```

额外输出会按输入 CSV 名推导，例如：

```text
data\features\audio\my_playlist_song_features.csv
data\features\audio\my_playlist_song_features.parquet
```

会回写到对应 `*_song_tags.csv` 的字段包括：

- `audio_tempo_bpm`
- `audio_tempo_raw_bpm`
- `audio_tempo_source`
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

BPM 的优先级是：

1. 先使用源 CSV 中已有的 BPM，例如 `bpm`、`wiki_bpm`，或从 `wiki_summary_excerpt` 里解析到的 BPM。
2. 源表没有可用 BPM 时，才用音频 onset 做节奏估计。
3. 音频估计会同时保存 `audio_tempo_raw_bpm` 和 `audio_tempo_bpm`。前者是原始估计值，后者是用于推荐和展示的听感节奏值；真实源表里的高 BPM 不会被统一除以二。

`audio_tempo_source` 用来说明 BPM 来源：`source_csv` 表示来自源表，`audio_estimate` 表示来自音频估计。

### 声源分离

如果要用 HDemucs 拆出人声、鼓、贝斯和其它伴奏：

```powershell
python data_processing\build_song_tags.py `
  --input data\source\my_playlist_songs.csv `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --source-separation `
  --source-separation-seconds 30
```

可显式指定 checkpoint：

```powershell
python data_processing\build_song_tags.py `
  --input data\source\my_playlist_songs.csv `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --source-separation `
  --source-separation-checkpoint "D:\Python\MUS-Gacha\models\hdemucs_high_trained.pt"
```

声源分离会输出：

- `source_separation_model`
- `source_drums_energy_ratio`
- `source_bass_energy_ratio`
- `source_other_energy_ratio`
- `source_vocals_energy_ratio`
- `source_vocal_energy_ratio`
- `source_instrumental_energy_ratio`

### 生成 MERT embedding 和聚类

推荐先确认 GPU 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

小批量测试：

```powershell
python data_processing\build_song_tags.py `
  --input data\source\my_playlist_songs.csv `
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
  --input data\source\my_playlist_songs.csv `
  --audio-dir "H:\音乐" `
  --reuse-matches `
  --extract-mert `
  --mert-max-seconds 12 `
  --mert-fp16
```

额外输出会按输入 CSV 名推导，例如：

```text
data\features\mert\embeddings\*.npy
data\features\mert\my_playlist_mert_index.csv
data\features\mert\my_playlist_mert_clusters.csv
```

会回写到对应 `*_song_tags.csv` 的字段包括：

- `mert_embedding_path`
- `mert_emotion_tags`
- `mert_valence`
- `mert_arousal`
- `mert_cluster`
- `mert_neighbor_song_ids`
- `mert_neighbor_scores`

注意：MERT 是自监督音乐表征模型。这里的 `mert_emotion_tags` 来自启发式情绪代理，不是经过人工标注数据训练出的可靠情绪分类器。

### 单曲 MERT 情绪分析

`data_processing\build_mert_emotion.py` 可以单独分析一首音频：

```powershell
python data_processing\build_mert_emotion.py `
  --model-dir .\models\MERT-v1-330M `
  --audio "H:\音乐\demo.flac" `
  --max-seconds 30 `
  --fp16
```

保存 JSON：

```powershell
python data_processing\build_mert_emotion.py `
  --model-dir .\models\MERT-v1-330M `
  --audio "H:\音乐\demo.flac" `
  --max-seconds 30 `
  --output-json emotion_result.json
```

## 🔑 核心约定

### `song_id` 唯一标识

当前项目统一以网易云 `song_id` 作为歌曲唯一标识：

- 源 CSV 的 `song_id` 列
- `data/source/lyrics/{song_id}.txt`
- `data/tags/*_song_tags.csv`
- `data/matches/*_song_matches.csv`
- `data/features/mert/embeddings/{song_id}.npy`
- `data/features/mert/*_mert_index.csv`
- `data/features/mert/*_mert_clusters.csv`
- Streamlit 页面选中状态
- 历史记录和历史偏好统计

### 源 CSV 汇总规则

应用会扫描 `data/source` 目录下所有 `.csv` 文件并拼接成曲库。去重依据是 `song_id`：

1. 丢弃没有 `song_id` 或 `song_id` 为空的行。
2. 同一首歌在多个 CSV 中出现时，优先保留非空字段更多的记录。
3. 完整度相同则保留排序靠后的源文件/行。
4. 最终保留的源文件名会写入运行时字段 `source_csv`，方便追溯。

### 本地音频匹配逻辑

匹配目标是把本地音乐目录中的文件对应到 CSV 的 `song_id`。

优先级：

1. 读取音频元数据 `title / artist / album`。
2. 从文件名解析 `艺术家 - 歌名`。
3. 与 CSV 中 `name / aliases / artist_names` 做模糊匹配。
4. 用本地音频时长和 CSV 的 `duration_seconds` 做加分或降权。
5. 低于阈值的文件不会自动写入匹配表。

匹配结果在：

```text
data\matches\*_song_matches.csv
```

## 缓存说明

项目当前主要有这几类缓存：

- `datacache/`
  预处理后的主缓存和用户历史记录目录。
  - `preprocessed_music.pkl`：预处理后的 DataFrame 和评分资源。
  - `data.hash`：源 CSV、标签、MERT、歌词和缓存版本生成的哈希。
  - `recommendation_history.json`：最近交互记录，用于历史偏好加权和图表。
  - `history_settings.json`：历史记录相关开关设置。
- `data/source/*.csv`
  源歌曲 CSV。任意文件变化都会触发下次启动时重建预处理缓存。
- `data/source/lyrics/*.txt`
  本地歌词。歌词增删改也会触发预处理缓存重建。
- `data/tags/*song_tags.csv`
  最终标签总表。应用只读取这些 CSV；任意文件变化后会触发预处理缓存重建。
- `data/tags/*song_tags.jsonl`
  JSONL 版标签输出，供查看或归档，不参与应用读取和预处理哈希。
- `data/matches/*song_matches*.csv`
  本地音频匹配功能产物。标签脚本会读取/复用它，并把最终字段回写到 `*_song_tags.csv`。
- `data/features/audio/*song_features*.csv`
  普通音频特征功能产物。应用不直接读取，最终字段以 `*_song_tags.csv` 为准。
- `data/features/lyric/*lyric_semantics*.csv`
  歌词语义功能产物。应用不直接读取，最终字段以 `*_song_tags.csv` 为准。
- `data/features/comment/*comment_semantics*.csv`
  评论语义功能产物。应用不直接读取，最终字段以 `*_song_tags.csv` 为准。
- `data/features/mert/embeddings/*.npy`
  每首歌的 MERT embedding。
- `data/features/mert/*mert_index*.csv`
  MERT embedding、启发式情绪、valence/arousal 索引功能产物。
- `data/features/mert/*mert_clusters*.csv`
  MERT 聚类和近邻歌曲功能产物。
- `.cache/huggingface`
  HuggingFace 动态模块和模型相关缓存。

当任意源 CSV、`data/tags/*song_tags.csv` 或歌词文件发生变化时，`datacache/data.hash` 会失效，下一次启动会自动重建预处理缓存。若缓存结构升级导致异常，可删除 `datacache/` 后重新运行：

```powershell
streamlit run app.py
```

## 常用命令速查

启动应用：

```powershell
streamlit run app.py
```

导出原始 JSON：

```powershell
python data_processing\export_original_json_to_csv.py --input-dir data\source\my_playlist_json --output data\source\my_playlist_songs.csv --lyrics-dir data\source\lyrics
```

重新生成基础标签：

```powershell
python data_processing\build_song_tags.py --input data\source\my_playlist_songs.csv --audio-dir "H:\音乐"
```

重跑音频特征：

```powershell
python data_processing\build_song_tags.py --input data\source\my_playlist_songs.csv --audio-dir "H:\音乐" --reuse-matches --analyze-audio --audio-feature-seconds 20
```

重跑 MERT：

```powershell
python data_processing\build_song_tags.py --input data\source\my_playlist_songs.csv --audio-dir "H:\音乐" --reuse-matches --extract-mert --mert-max-seconds 12 --mert-fp16
```

查看脚本参数：

```powershell
python data_get\qcloud_song_store.py --help
python data_get\retry_rate_limited_songs.py --help
python data_processing\export_original_json_to_csv.py --help
python data_processing\build_song_tags.py --help
python data_processing\build_lyric_semantics.py --help
python data_processing\build_mert_emotion.py --help
```

## ⚠️ 注意事项

- 当前实现对 Windows 更友好，尤其是本地文件打开和默认路径。
- `cookie.txt` 不应提交到 Git；当前 `.gitignore` 已忽略它。
- `data/`、`datacache/`、`models/`、`.cache/` 默认不进 Git。
- `build_song_tags.py` 一次处理一个 CSV；省略 `--input` 时会选择第一个 `data/source/*.csv`，输出文件按输入名自动推导。
- 如果 `data/source/*.csv` 一个都没有，标签/音频/MERT 没有可处理的歌曲主表；请先从 JSON 快照导出 CSV，或手动放入带 `song_id` 列的源 CSV。
- Streamlit 曲库会自动读取 `data/source/*.csv` 并去重；生成结果只读取 `data/tags/*song_tags.csv`。JSONL、音频匹配、音频特征、歌词语义、评论语义和 MERT CSV 都是功能产物，需要先回写到标签总表。
- QCloudMusicApi 抓取接口可能遇到限流；可以用 `retry_rate_limited_songs.py` 或数据处理页重试。
- `torchaudio` 可能无法识别某些 `.m4a`。这种文件仍可能被匹配到本地路径，但音频特征和 MERT embedding 会为空。
- MERT 情绪标签是启发式标签，适合辅助检索和初筛；可靠情绪分类需要人工标注数据和分类头训练。
- 如果没有开启 `--source-separation`，或模型权重下载/加载失败，人声/器乐会退回歌词信息、纯音乐提示和频段能量的启发式估计。
- 某些执行环境会杀掉后台 Streamlit 进程，推荐在终端前台运行 `streamlit run app.py`。

## 💡 适合谁用

如果你想把网易云喜欢列表、本地音乐文件、歌词、评论、音频特征和个人历史偏好整合成一个可解释、可调权重、可本地播放的私人音乐推荐系统，这个项目会很顺手。

它更像一个“可调音色的音乐仓库”，不是一个追求通用流行榜单的推荐器：你可以把自己想听的标签、歌手、歌词意象、评论共鸣和历史偏好一点点拧到合适的位置。

## 致谢

本项目的网易云音乐数据采集能力基于 [QCloudMusicApi](https://github.com/s12mmm3/QCloudMusicApi)。感谢原项目提供的接口封装与本地调用能力。
