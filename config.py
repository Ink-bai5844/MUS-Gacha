from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
TAG_DATA_DIR = DATA_DIR / "tags"
MATCH_DATA_DIR = DATA_DIR / "matches"
AUDIO_FEATURE_DATA_DIR = DATA_DIR / "features" / "audio"
MERT_DATA_DIR = DATA_DIR / "features" / "mert"
COMMENT_SEMANTIC_DATA_DIR = DATA_DIR / "features" / "comment"
LYRIC_SEMANTIC_DATA_DIR = DATA_DIR / "features" / "lyric"
LYRICS_DIR = SOURCE_DATA_DIR / "lyrics"
CACHE_DIR = BASE_DIR / "datacache"
PREPROCESSED_DATA_FILE = CACHE_DIR / "preprocessed_music.pkl"
PREPROCESSED_HASH_FILE = CACHE_DIR / "data.hash"
PREPROCESS_CACHE_VERSION = "mus-gacha-preprocess-v7"
MAX_DISPLAY = 60
HISTORY_CACHE_FILE = CACHE_DIR / "recommendation_history.json"
HISTORY_SETTINGS_FILE = CACHE_DIR / "history_settings.json"
HISTORY_RECOMMENDATION_CACHE_SIZE = 80
HISTORY_LINK_TRACKING_HOST = "127.0.0.1"
HISTORY_LINK_TRACKING_PORT = 8766

INITIAL_TAG_WEIGHTS = {
    "人声强": 1.4,
    "器乐强": 1.25,
    "热血": 1.25,
    "治愈": 1.2,
    "宁静": 1.15,
    "纯音乐": 1.15,
    "悲伤": 0.8,
}

LYRIC_STOP_WORDS = {
    "作词",
    "作曲",
    "编曲",
    "制作人",
    "演唱",
    "歌词",
    "纯音乐",
    "请欣赏",
    "欣赏",
    "music",
    "lyrics",
    "composer",
    "arranger",
    "the",
    "and",
    "you",
    "that",
    "this",
    "with",
    "for",
    "are",
    "was",
    "were",
    "your",
    "我的",
    "我们",
    "你们",
    "他们",
    "一个",
    "没有",
    "什么",
    "怎么",
    "还是",
    "只是",
    "不要",
    "不能",
    "それ",
    "これ",
    "から",
    "まで",
    "こと",
    "もの",
}

COMMENT_RULES = [
    ("回忆共鸣", ["回忆", "童年", "青春", "小时候", "以前", "当年", "那年", "怀念", "泪目"]),
    ("治愈共鸣", ["治愈", "温暖", "安心", "舒服", "放松", "平静", "温柔"]),
    ("悲伤共鸣", ["流泪", "哭", "泪", "难过", "心酸", "遗憾", "破防", "emo"]),
    ("热血共鸣", ["燃", "热血", "励志", "加油", "力量", "勇气", "坚持"]),
    ("好听认可", ["好听", "神曲", "封神", "循环", "单曲循环", "喜欢", "爱了", "绝了"]),
    ("故事感", ["故事", "人生", "经历", "后来", "想起", "陪伴", "告别"]),
    ("幽默吐槽", ["哈哈", "笑死", "蚌埠", "绷不住", "草", "233", "hhhh"]),
    ("亲情陪伴", ["爸爸", "妈妈", "父亲", "母亲", "家人", "爸妈", "爷爷", "奶奶"]),
    ("影视回忆", ["电视剧", "电影", "动漫", "片头", "片尾", "主题曲", "插曲"]),
]

LYRIC_SEMANTIC_RULES = [
    ("爱情思念", ["爱", "想你", "思念", "拥抱", "心动", "恋人", "等待", "重逢"]),
    ("离别遗憾", ["离别", "再见", "告别", "遗憾", "错过", "眼泪", "远走", "分开"]),
    ("成长追寻", ["成长", "梦想", "远方", "寻找", "奔跑", "未来", "答案", "旅程"]),
    ("孤独夜晚", ["孤独", "寂寞", "夜", "月光", "一个人", "沉默", "失眠", "黑暗"]),
    ("治愈温柔", ["温柔", "治愈", "安静", "微风", "阳光", "陪伴", "安心", "平静"]),
    ("热血信念", ["热血", "勇气", "信念", "战斗", "燃烧", "坚持", "不放弃", "希望"]),
    ("怀旧时光", ["回忆", "青春", "童年", "从前", "时光", "旧梦", "昨日", "怀念"]),
    ("城市旅途", ["城市", "街道", "车站", "旅途", "海边", "路上", "远行", "风景"]),
    ("自然意象", ["天空", "大海", "星星", "雨", "风", "花", "雪", "山川"]),
    ("梦境幻想", ["梦", "幻想", "童话", "奇迹", "魔法", "宇宙", "星河", "光"]),
    ("江湖古风", ["江湖", "山河", "刀剑", "红尘", "故人", "长安", "明月", "青丝"]),
    ("亲情陪伴", ["妈妈", "爸爸", "家", "故乡", "孩子", "亲人", "归来", "守候"]),
    ("纯音乐留白", ["纯音乐", "请欣赏", "无词", "instrumental", "piano", "钢琴"]),
]

GENERATED_TAG_COLUMNS = [
    "generated_language_tags",
    "style_tags",
    "emotion_tags",
    "theme_tags",
    "scene_tags",
    "audio_tags",
    "lyric_semantic_tags",
    "comment_semantic_tags",
    "all_tags",
    "tag_confidence",
    "tag_sources",
    "bpm",
    "local_audio_path",
    "local_audio_title",
    "local_audio_artist",
    "local_audio_album",
    "local_duration_seconds",
    "duration_diff_seconds",
    "audio_match_score",
    "audio_match_reason",
    "match_source",
    "audio_duration_seconds",
    "audio_sample_rate",
    "audio_tempo_bpm",
    "audio_tempo_raw_bpm",
    "audio_tempo_source",
    "audio_onset_strength",
    "audio_rms",
    "audio_zcr",
    "audio_centroid_hz",
    "audio_crest",
    "audio_feature_tags",
    "audio_error",
    "audio_vocal_band_ratio",
    "source_separation_model",
    "source_separation_error",
    "source_drums_energy_ratio",
    "source_bass_energy_ratio",
    "source_other_energy_ratio",
    "source_vocals_energy_ratio",
    "source_vocal_energy_ratio",
    "source_instrumental_energy_ratio",
    "vocal_presence_score",
    "instrumental_presence_score",
    "vocal_instrumental_tags",
    "lyric_semantic_scores",
    "lyric_semantic_model",
    "lyric_semantic_error",
    "comment_semantic_scores",
    "comment_semantic_model",
    "comment_semantic_error",
    "mert_cluster",
    "mert_neighbor_song_ids",
    "mert_neighbor_scores",
    "mert_emotion_tags",
    "mert_emotion_scores",
    "mert_valence",
    "mert_arousal",
    "mert_embedding_path",
    "mert_error",
    "mert_chunks",
    "mert_embedding_dim",
    "mert_layer",
]

TEXT_COLUMNS = [
    "name",
    "aliases",
    "translations",
    "artist_names",
    "album_name",
    "lyric_excerpt",
    "translation_excerpt",
    "romaji_excerpt",
    "similar_song_names",
    "similar_artist_names",
    "wiki_summary_excerpt",
    "first_hot_comment",
    "first_comment",
    "style_tags",
    "emotion_tags",
    "theme_tags",
    "scene_tags",
    "audio_tags",
    "lyric_semantic_tags",
    "all_tags",
]
