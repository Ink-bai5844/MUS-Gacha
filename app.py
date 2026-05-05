import html
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ink_bai_liked_songs.csv"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
MAX_DISPLAY = 60

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
]


st.set_page_config(page_title="墨白的音乐仓库", layout="wide")
st.markdown(
    """
    <style>
    div[id^="gdg-overlay-"] {
        margin-left: 96px !important;
        z-index: 99999 !important;
        border-radius: 8px !important;
        box-shadow: 5px 5px 18px rgba(0, 0, 0, 0.32) !important;
        overflow: hidden !important;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: rgba(250, 250, 250, 0.72);
    }
    .song-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .muted-line {
        color: rgba(49, 51, 63, 0.70);
        margin-bottom: 0.55rem;
    }
    .lyric-box {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 1rem;
        max-height: 520px;
        overflow: auto;
        white-space: pre-wrap;
        line-height: 1.75;
        background: rgba(255, 255, 255, 0.68);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_pipe(value):
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def read_lyric(song_id):
    path = LYRICS_DIR / f"{song_id}.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").strip()


def normalize_for_search(value):
    return re.sub(r"\s+", " ", safe_text(value).lower())


def build_search_text(row):
    parts = [safe_text(row.get(col, "")) for col in TEXT_COLUMNS]
    parts.append(safe_text(row.get("full_lyric", "")))
    return normalize_for_search(" ".join(parts))


def minmax(series, index=None):
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    max_value = numeric.max()
    if max_value <= 0:
        return numeric * 0
    return numeric / max_value


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
    return tags or ["未知"]


def top_counts(df, column, limit=12):
    values = []
    for items in df[column]:
        values.extend(items)
    if not values:
        return pd.DataFrame({"名称": [], "数量": []})
    counts = pd.Series(values).value_counts().head(limit)
    return counts.rename_axis("名称").reset_index(name="数量")


def filter_by_keywords(df, query):
    terms = [term.strip().lower() for term in re.split(r"[,，\s]+", query or "") if term.strip()]
    if not terms:
        return df

    mask = pd.Series(True, index=df.index)
    for term in terms:
        mask &= df["search_text"].str.contains(term, regex=False, na=False)
    return df[mask]


@st.cache_data(show_spinner=False)
def load_music_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE, dtype={"song_id": "string", "artist_ids": "string", "album_id": "string"})
    df.columns = [col.strip() for col in df.columns]

    for col in TEXT_COLUMNS + ["song_id", "album_pic_url", "duration_text", "publish_date"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in [
        "duration_seconds",
        "popularity",
        "comment_total",
        "hot_comment_count",
        "lyric_line_count",
        "standard_br",
        "exhigh_br",
        "lossless_br",
        "hires_br",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    for col in ["playable", "has_lyric", "has_translation", "has_romaji", "check_success"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_bool)
        else:
            df[col] = False

    df["publish_year"] = pd.to_datetime(df.get("publish_date", ""), errors="coerce").dt.year
    df["duration_minutes"] = df["duration_seconds"] / 60
    df["artist_list"] = df.get("artist_names", "").apply(split_pipe)
    df["language_tags"] = df.apply(extract_language_tags, axis=1)
    df["quality"] = df.apply(preferred_quality, axis=1)
    df["full_lyric"] = df["song_id"].apply(read_lyric)
    df["search_text"] = df.apply(build_search_text, axis=1)
    df["lyrics_chars"] = df["full_lyric"].str.len()
    df["netease_url"] = "https://music.163.com/#/song?id=" + df["song_id"].astype(str)

    score = (
        minmax(df["popularity"]) * 34
        + minmax(df["comment_total"]).pow(0.45) * 28
        + minmax(df["lyric_line_count"]) * 14
        + df["has_translation"].astype(int) * 8
        + df["playable"].astype(int) * 10
        + df["has_romaji"].astype(int) * 4
        + df["quality"].isin(["Hi-Res", "无损"]).astype(int) * 8
    )
    df["recommend_score"] = score.round().clip(0, 100)
    return df


def render_detail(song):
    left, right = st.columns([1.1, 2.4], gap="large")

    with left:
        cover = safe_text(song.get("album_pic_url", ""))
        if cover:
            st.image(cover, width="stretch")
        st.link_button("打开网易云页面", safe_text(song.get("netease_url", "")), width="stretch")

        audio_url = (
            safe_text(song.get("hires_url", ""))
            or safe_text(song.get("lossless_url", ""))
            or safe_text(song.get("exhigh_url", ""))
            or safe_text(song.get("standard_url", ""))
        )
        if audio_url:
            st.audio(audio_url)

    with right:
        st.markdown(f"<div class='song-title'>{html.escape(safe_text(song.get('name')))}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='muted-line'>"
            f"{html.escape(safe_text(song.get('artist_names')))} · {html.escape(safe_text(song.get('album_name')))}"
            "</div>",
            unsafe_allow_html=True,
        )

        meta_cols = st.columns(4)
        meta_cols[0].metric("推荐分", int(song.get("dynamic_score", song.get("recommend_score", 0))))
        meta_cols[1].metric("热度", int(song.get("popularity", 0)))
        meta_cols[2].metric("评论", int(song.get("comment_total", 0)))
        meta_cols[3].metric("时长", safe_text(song.get("duration_text", "")) or f"{song.get('duration_minutes', 0):.1f} 分")

        info = pd.DataFrame(
            [
                ("发行日期", safe_text(song.get("publish_date"))),
                ("版权/可播", f"{safe_text(song.get('copyright'))} / {'可播' if song.get('playable') else '不可播'}"),
                ("最佳音质", safe_text(song.get("quality"))),
                ("歌词行数", str(int(song.get("lyric_line_count", 0)))),
                ("热门评论", safe_text(song.get("first_hot_comment"))),
                ("相似歌曲", safe_text(song.get("similar_song_names"))),
            ],
            columns=["字段", "内容"],
        )
        st.dataframe(info, hide_index=True, width="stretch", height=250)

    lyric = safe_text(song.get("full_lyric", "")) or safe_text(song.get("lyric_excerpt", ""))
    if lyric:
        st.markdown("#### 歌词")
        st.markdown(f"<div class='lyric-box'>{html.escape(lyric)}</div>", unsafe_allow_html=True)
    else:
        st.info("这首歌暂时没有本地歌词。")


df_base = load_music_data()

st.sidebar.title("筛选与推荐")

if df_base.empty:
    st.error(f"未找到数据文件：{DATA_FILE}")
    st.stop()

search_kw = st.sidebar.text_input("实时检索", placeholder="歌名 / 歌手 / 专辑 / 歌词 / 评论")
lyrics_kw = st.sidebar.text_input("歌词专项检索", placeholder="只在完整歌词里查找")

with st.sidebar.expander("基础筛选", expanded=True):
    all_artists = sorted({artist for artists in df_base["artist_list"] for artist in artists})
    selected_artists = st.multiselect("歌手", options=all_artists, default=[])
    all_languages = sorted({tag for tags in df_base["language_tags"] for tag in tags})
    selected_languages = st.multiselect("语言/场景标签", options=all_languages, default=[])
    selected_qualities = st.multiselect("音质", options=["Hi-Res", "无损", "极高", "标准", "未知"], default=[])

    playable_only = st.checkbox("只看可播放", value=True)
    with_lyrics_only = st.checkbox("只看有歌词", value=False)
    with_translation_only = st.checkbox("只看有翻译", value=False)

with st.sidebar.expander("数值范围", expanded=False):
    popularity_min = int(df_base["popularity"].min())
    popularity_max = int(df_base["popularity"].max())
    min_popularity = st.slider("最低热度", popularity_min, popularity_max, popularity_min)

    max_comments = int(df_base["comment_total"].max())
    min_comments = st.slider("最低评论数", 0, max_comments, 0, step=max(1, max_comments // 100))

    valid_years = df_base["publish_year"].dropna().astype(int)
    if valid_years.empty:
        selected_years = (1900, 2030)
    else:
        year_min, year_max = int(valid_years.min()), int(valid_years.max())
        selected_years = st.slider("发行年份", year_min, year_max, (year_min, year_max))

with st.sidebar.expander("推荐权重", expanded=True):
    score_weight = st.slider("推荐分权重", 0.0, 3.0, 1.0, 0.1)
    comment_weight = st.slider("评论热度权重", 0.0, 3.0, 1.0, 0.1)
    lyric_weight = st.slider("歌词完整度权重", 0.0, 3.0, 0.8, 0.1)
    quality_weight = st.slider("音质权重", 0.0, 3.0, 0.8, 0.1)

filtered_df = filter_by_keywords(df_base.copy(), search_kw)

if lyrics_kw:
    terms = [term.strip().lower() for term in re.split(r"[,，\s]+", lyrics_kw) if term.strip()]
    lyric_mask = pd.Series(True, index=filtered_df.index)
    lyric_text = filtered_df["full_lyric"].str.lower()
    for term in terms:
        lyric_mask &= lyric_text.str.contains(term, regex=False, na=False)
    filtered_df = filtered_df[lyric_mask]

if selected_artists:
    filtered_df = filtered_df[filtered_df["artist_list"].apply(lambda artists: any(artist in artists for artist in selected_artists))]
if selected_languages:
    filtered_df = filtered_df[filtered_df["language_tags"].apply(lambda tags: any(tag in tags for tag in selected_languages))]
if selected_qualities:
    filtered_df = filtered_df[filtered_df["quality"].isin(selected_qualities)]
if playable_only:
    filtered_df = filtered_df[filtered_df["playable"]]
if with_lyrics_only:
    filtered_df = filtered_df[filtered_df["has_lyric"]]
if with_translation_only:
    filtered_df = filtered_df[filtered_df["has_translation"]]

filtered_df = filtered_df[
    (filtered_df["popularity"] >= min_popularity)
    & (filtered_df["comment_total"] >= min_comments)
    & (
        filtered_df["publish_year"].isna()
        | filtered_df["publish_year"].between(selected_years[0], selected_years[1])
    )
].copy()

if not filtered_df.empty:
    weighted = (
        filtered_df["recommend_score"] * score_weight
        + minmax(filtered_df["comment_total"]).pow(0.45) * 100 * comment_weight
        + minmax(filtered_df["lyric_line_count"]) * 100 * lyric_weight
        + filtered_df["quality"].isin(["Hi-Res", "无损"]).astype(int) * 100 * quality_weight
    )
    divider = max(0.1, score_weight + comment_weight + lyric_weight + quality_weight)
    filtered_df["dynamic_score"] = (weighted / divider).round().clip(0, 100)
else:
    filtered_df["dynamic_score"] = pd.Series(dtype=float)

st.title("墨白的音乐仓库")

metric_cols = st.columns(5)
metric_cols[0].metric("当前歌曲", f"{len(filtered_df)} / {len(df_base)}")
metric_cols[1].metric("歌手数", len({artist for artists in filtered_df["artist_list"] for artist in artists}))
metric_cols[2].metric("有歌词", int(filtered_df["has_lyric"].sum()))
metric_cols[3].metric("可播放", int(filtered_df["playable"].sum()))
metric_cols[4].metric("平均推荐分", f"{filtered_df['dynamic_score'].mean():.1f}" if not filtered_df.empty else "0.0")

tab_overview, tab_library, tab_lyrics, tab_detail = st.tabs(["推荐总览", "歌曲列表", "歌词检索", "歌曲详情"])

with tab_overview:
    if filtered_df.empty:
        st.info("没有匹配当前筛选条件的歌曲。")
    else:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader("推荐候选")
            top_df = filtered_df.sort_values(["dynamic_score", "comment_total"], ascending=False).head(20)
            st.dataframe(
                top_df[
                    [
                        "album_pic_url",
                        "dynamic_score",
                        "name",
                        "artist_names",
                        "album_name",
                        "quality",
                        "duration_text",
                        "comment_total",
                        "netease_url",
                    ]
                ],
                column_config={
                    "album_pic_url": st.column_config.ImageColumn("封面"),
                    "dynamic_score": st.column_config.ProgressColumn("推荐分", min_value=0, max_value=100, format="%d"),
                    "name": "歌曲",
                    "artist_names": "歌手",
                    "album_name": "专辑",
                    "quality": "音质",
                    "duration_text": "时长",
                    "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                    "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
                },
                hide_index=True,
                width="stretch",
                height=620,
            )

        with right:
            st.subheader("分布画像")
            artist_counts = top_counts(filtered_df, "artist_list", 12)
            language_counts = top_counts(filtered_df, "language_tags", 12)
            quality_counts = filtered_df["quality"].value_counts().rename_axis("音质").reset_index(name="数量")
            year_counts = (
                filtered_df.dropna(subset=["publish_year"])
                .assign(publish_year=lambda item: item["publish_year"].astype(int))
                .groupby("publish_year")
                .size()
                .reset_index(name="数量")
            )

            st.caption("热门歌手")
            st.bar_chart(artist_counts, x="名称", y="数量", height=210)
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.caption("语言/场景")
                st.bar_chart(language_counts, x="名称", y="数量", height=210)
            with chart_cols[1]:
                st.caption("音质")
                st.bar_chart(quality_counts, x="音质", y="数量", height=210)
            st.caption("发行年份")
            st.line_chart(year_counts, x="publish_year", y="数量", height=220)

with tab_library:
    if filtered_df.empty:
        st.info("没有可显示的数据。")
    else:
        sort_columns = {
            "推荐分": "dynamic_score",
            "评论数": "comment_total",
            "热度": "popularity",
            "发行年份": "publish_year",
            "歌名": "name",
            "歌手": "artist_names",
        }
        col_sort, col_order, col_page = st.columns([1.4, 1.0, 1.4])
        with col_sort:
            sort_label = st.selectbox("全局排序依据", options=list(sort_columns), index=0)
        with col_order:
            order_label = st.radio("顺序", ["降序", "升序"], horizontal=True)

        sorted_df = filtered_df.sort_values(
            by=sort_columns[sort_label],
            ascending=(order_label == "升序"),
            na_position="last",
        ).reset_index(drop=True)

        total_items = len(sorted_df)
        total_pages = max(1, math.ceil(total_items / MAX_DISPLAY))
        page_options = []
        for i in range(total_pages):
            start_idx = i * MAX_DISPLAY + 1
            end_idx = min((i + 1) * MAX_DISPLAY, total_items)
            page_options.append(f"{start_idx} ~ {end_idx}")

        with col_page:
            page_label = st.selectbox("显示范围", options=page_options)

        page_index = page_options.index(page_label)
        display_df = sorted_df.iloc[page_index * MAX_DISPLAY : (page_index + 1) * MAX_DISPLAY].copy()

        st.dataframe(
            display_df[
                [
                    "album_pic_url",
                    "dynamic_score",
                    "song_id",
                    "name",
                    "artist_names",
                    "album_name",
                    "publish_date",
                    "quality",
                    "duration_text",
                    "popularity",
                    "comment_total",
                    "has_lyric",
                    "has_translation",
                    "netease_url",
                ]
            ],
            column_config={
                "album_pic_url": st.column_config.ImageColumn("封面"),
                "dynamic_score": st.column_config.ProgressColumn("推荐分", min_value=0, max_value=100, format="%d"),
                "song_id": "ID",
                "name": "歌曲",
                "artist_names": "歌手",
                "album_name": "专辑",
                "publish_date": "发行日期",
                "quality": "音质",
                "duration_text": "时长",
                "popularity": st.column_config.NumberColumn("热度", format="%d"),
                "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                "has_lyric": "歌词",
                "has_translation": "翻译",
                "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
            },
            hide_index=True,
            width="stretch",
            height=650,
        )

with tab_lyrics:
    lyric_df = filtered_df[filtered_df["full_lyric"].str.len() > 0].copy()
    if lyric_df.empty:
        st.info("当前筛选范围内没有本地歌词。")
    else:
        lyric_df = lyric_df.sort_values(["dynamic_score", "lyrics_chars"], ascending=False)
        options = {
            f"{row.name} · {row.artist_names} · {row.song_id}": idx
            for idx, row in lyric_df.head(300).iterrows()
        }
        selected_label = st.selectbox("选择歌词", options=list(options.keys()))
        selected_song = lyric_df.loc[options[selected_label]]

        st.markdown(
            f"**{safe_text(selected_song.get('name'))}** · {safe_text(selected_song.get('artist_names'))} · "
            f"{int(selected_song.get('lyric_line_count', 0))} 行"
        )

        lyric = safe_text(selected_song.get("full_lyric", ""))
        if lyrics_kw:
            for term in [term.strip() for term in re.split(r"[,，\s]+", lyrics_kw) if term.strip()]:
                lyric = re.sub(
                    re.escape(term),
                    lambda match: f"<mark>{html.escape(match.group(0))}</mark>",
                    html.escape(lyric),
                    flags=re.IGNORECASE,
                )
            st.markdown(f"<div class='lyric-box'>{lyric}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='lyric-box'>{html.escape(lyric)}</div>", unsafe_allow_html=True)

with tab_detail:
    if filtered_df.empty:
        st.info("先调整筛选条件，选出一首歌。")
    else:
        detail_df = filtered_df.sort_values(["dynamic_score", "comment_total"], ascending=False)
        song_options = {
            f"{row.name} · {row.artist_names} · {row.song_id}": idx
            for idx, row in detail_df.head(500).iterrows()
        }
        selected_song_label = st.selectbox("选择歌曲", options=list(song_options.keys()))
        render_detail(detail_df.loc[song_options[selected_song_label]])
