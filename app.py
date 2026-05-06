import html
import math
import re

import pandas as pd
import streamlit as st

from config import DATA_FILE, HISTORY_RECOMMENDATION_CACHE_SIZE, INITIAL_TAG_WEIGHTS, MAX_DISPLAY
from data_pipeline import apply_dynamic_music_scores, filter_by_keywords, load_music_data
from ui_components import render_detail, render_page_style
from utils_charts import (
    build_dataframe_chart_data,
    build_global_preference_chart_data,
    build_history_preference_chart_data,
    render_dataframe_chart_section,
    render_preference_chart_grid,
)
from utils_history import (
    build_history_preference_maps,
    clear_history_entries,
    load_history_entries,
    record_recommendation_history,
    save_history_entries,
)
from utils_text import safe_text


SELECTED_SONG_STATE_KEY = "selected_song_id"
HISTORY_CHART_ENTRIES_STATE_KEY = "history_chart_entries"
PRESERVE_DISTRIBUTION_ONCE_STATE_KEY = "preserve_distribution_once"
SELECTION_WRITES_HISTORY_STATE_KEY = "selection_writes_history"


def current_selected_song_id():
    selected_song_id = st.session_state.get(SELECTED_SONG_STATE_KEY)
    return str(selected_song_id) if selected_song_id else ""


def make_selectable_table(table_df):
    editable_df = table_df.copy()
    selected_song_id = current_selected_song_id()
    editable_df["选中"] = editable_df["song_id"].astype(str).eq(selected_song_id)
    return editable_df


def apply_table_selection(edited_df, source_df):
    if "选中" not in edited_df.columns or "song_id" not in edited_df.columns:
        return

    selected_rows = edited_df[edited_df["选中"].fillna(False)]
    if selected_rows.empty:
        return

    current_song_id = current_selected_song_id()
    newly_selected = selected_rows[selected_rows["song_id"].astype(str) != current_song_id]
    chosen_row = newly_selected.iloc[0] if not newly_selected.empty else selected_rows.iloc[0]
    chosen_song_id = str(chosen_row["song_id"])

    if chosen_song_id != current_song_id:
        matched_rows = source_df[source_df["song_id"].astype(str) == chosen_song_id]
        history_row = matched_rows.iloc[0] if not matched_rows.empty else chosen_row
        st.session_state[SELECTED_SONG_STATE_KEY] = chosen_song_id
        if st.session_state.get(SELECTION_WRITES_HISTORY_STATE_KEY, True):
            st.session_state[PRESERVE_DISTRIBUTION_ONCE_STATE_KEY] = True
            record_recommendation_history(history_row, "select")
        st.toast(f"已选中：{safe_text(history_row.get('name'))}", icon="✅")
        st.rerun()


def get_history_chart_entries(history_entries):
    if HISTORY_CHART_ENTRIES_STATE_KEY not in st.session_state:
        st.session_state[HISTORY_CHART_ENTRIES_STATE_KEY] = history_entries

    if st.session_state.pop(PRESERVE_DISTRIBUTION_ONCE_STATE_KEY, False):
        return st.session_state[HISTORY_CHART_ENTRIES_STATE_KEY]

    st.session_state[HISTORY_CHART_ENTRIES_STATE_KEY] = history_entries
    return history_entries


def selected_or_first(df, sort_columns):
    if df.empty:
        return None, False

    selected_song_id = current_selected_song_id()
    if selected_song_id:
        matched = df[df["song_id"].astype(str) == selected_song_id]
        if not matched.empty:
            return matched.iloc[0], True

    return df.sort_values(sort_columns, ascending=False).iloc[0], False


def build_history_table(history_entries):
    rows = []
    for index, entry in enumerate(history_entries):
        if not isinstance(entry, dict):
            continue
        tag_summary = " | ".join((entry.get("all_tags") or [])[:6])
        rows.append(
            {
                "删除": False,
                "序号": index + 1,
                "选中时间": safe_text(entry.get("selected_at")),
                "动作": safe_text(entry.get("action")),
                "歌曲ID": safe_text(entry.get("song_id")),
                "歌曲": safe_text(entry.get("name")),
                "歌手": safe_text(entry.get("artist_names")),
                "专辑": safe_text(entry.get("album_name")),
                "标签": tag_summary,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="墨白的音乐仓库", layout="wide")
render_page_style()
st.session_state.setdefault(SELECTION_WRITES_HISTORY_STATE_KEY, True)


with st.spinner("正在同步预处理缓存与评分资源..."):
    df_base, scoring_resources = load_music_data()

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
    all_generated_tags = sorted({tag for tags in df_base["generated_tag_list"] for tag in tags})
    selected_generated_tags = st.multiselect("智能标签", options=all_generated_tags, default=[])
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

st.sidebar.markdown("---")
st.sidebar.subheader("推荐评分配置")

with st.sidebar.expander("全局维度权重", expanded=True):
    dimension_weights = {
        "综合标签": st.slider("综合标签倍率（汇总所有智能标签的整体匹配度）", 0.0, 5.0, 1.0, 0.1),
        "语种": st.slider("语种倍率（国语/粤语/英语/日语等语言偏好）", 0.0, 5.0, 0.5, 0.1),
        "风格": st.slider("风格倍率（流行/古风/摇滚/电子等曲风偏好）", 0.0, 5.0, 1.0, 0.1),
        "情绪": st.slider("情绪倍率（治愈/热血/悲伤/宁静等听感倾向）", 0.0, 5.0, 1.0, 0.1),
        "主题": st.slider("主题倍率（爱情/成长/时光/旅途等内容主题）", 0.0, 5.0, 0.8, 0.1),
        "场景": st.slider("场景倍率（学习/睡前/通勤/运动等使用场景）", 0.0, 5.0, 0.8, 0.1),
        "音频标签": st.slider("音频标签倍率（快慢节奏、能量、人声/器乐等音频特征）", 0.0, 5.0, 1.0, 0.1),
        "歌手": st.slider("歌手倍率（常见或指定歌手带来的推荐加成）", 0.0, 5.0, 1.0, 0.1),
        "歌名关键词": st.slider("歌名/专辑关键词倍率（从歌名、别名、专辑名提取的关键词）", 0.0, 5.0, 0.7, 0.1),
        "歌词关键词": st.slider("歌词关键词倍率（从完整歌词中提取的高频语义词）", 0.0, 5.0, 0.8, 0.1),
        "评论语义": st.slider("评论语义倍率（热门评论里的回忆、治愈、热血等共鸣）", 0.0, 5.0, 0.8, 0.1),
        "热度": st.slider("热度倍率（网易云热度数值带来的基础加分）", 0.0, 5.0, 0.8, 0.1),
        "歌词完整度": st.slider("歌词完整度倍率（歌词行数和本地歌词完整程度）", 0.0, 5.0, 0.5, 0.1),
        "音质": st.slider("音质倍率（Hi-Res/无损等高音质资源加成）", 0.0, 5.0, 0.5, 0.1),
        "可播放": st.slider("可播放倍率（当前歌曲是否可在线播放）", 0.0, 5.0, 0.6, 0.1),
        "本地音频": st.slider("本地音频倍率（已匹配到本地音乐文件的加成）", 0.0, 5.0, 0.8, 0.1),
        "MERT": st.slider("MERT 可用倍率（已提取 MERT embedding/情绪信息的加成）", 0.0, 5.0, 0.4, 0.1),
    }
    global_history_weight = st.slider("历史偏好总分倍率", 0.0, 5.0, 1.0, 0.1)

all_weight_tags = sorted(scoring_resources["all_tags"].keys())
valid_default_tags = [tag for tag in INITIAL_TAG_WEIGHTS if tag in all_weight_tags]

with st.sidebar.expander("屏蔽标签配置", expanded=False):
    blocked_tags = st.multiselect("选择要屏蔽的标签", options=all_weight_tags, default=[])
    st.caption("命中任意屏蔽标签的歌曲会直接从结果中移除。")

with st.sidebar.expander("单标签权重配置", expanded=True):
    selected_weight_tags = st.multiselect("加权/降权标签列表", options=all_weight_tags, default=valid_default_tags)
    dynamic_tag_weights = {}
    for tag in selected_weight_tags:
        default_value = float(INITIAL_TAG_WEIGHTS.get(tag, 1.0))
        dynamic_tag_weights[tag] = st.number_input(
            f"「{tag}」权重倍率",
            value=default_value,
            step=0.1,
            format="%.1f",
            key=f"tag-weight-{tag}",
        )

with st.sidebar.expander("歌手权重配置", expanded=False):
    selected_weight_artists = st.multiselect("需要单独调整的歌手", options=all_artists, default=[])
    dynamic_artist_weights = {}
    for artist in selected_weight_artists:
        dynamic_artist_weights[artist] = st.number_input(
            f"「{artist}」倍率",
            value=2.0,
            step=0.5,
            format="%.1f",
            key=f"artist-weight-{artist}",
        )

with st.sidebar.expander("歌名关键词权重配置", expanded=False):
    all_title_terms = sorted(scoring_resources["title_terms"].keys())
    selected_title_terms = st.multiselect("关键词列表", options=all_title_terms, default=[])
    dynamic_title_weights = {}
    for term in selected_title_terms:
        dynamic_title_weights[term] = st.number_input(
            f"词汇「{term}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"title-weight-{term}",
        )

with st.sidebar.expander("歌词关键词权重配置", expanded=False):
    all_lyric_terms = sorted(scoring_resources["lyric_terms"].keys())
    selected_lyric_terms = st.multiselect("歌词关键词列表", options=all_lyric_terms, default=[])
    dynamic_lyric_weights = {}
    for term in selected_lyric_terms:
        dynamic_lyric_weights[term] = st.number_input(
            f"歌词词汇「{term}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"lyric-weight-{term}",
        )

with st.sidebar.expander("评论语义权重配置", expanded=False):
    all_comment_tags = sorted(scoring_resources["comment_semantic"].keys())
    selected_comment_tags = st.multiselect("评论语义标签", options=all_comment_tags, default=[])
    dynamic_comment_weights = {}
    for tag in selected_comment_tags:
        dynamic_comment_weights[tag] = st.number_input(
            f"评论语义「{tag}」权重",
            value=1.0,
            step=0.1,
            format="%.1f",
            key=f"comment-weight-{tag}",
        )

history_entries = load_history_entries()
history_chart_entries = get_history_chart_entries(history_entries)

history_preference = (
    build_history_preference_maps(history_entries, scoring_resources, dimension_weights)
    if global_history_weight > 0 and history_entries
    else None
)

scored_df = apply_dynamic_music_scores(
    df_base,
    scoring_resources,
    dimension_weights,
    dynamic_tag_weights,
    dynamic_artist_weights,
    dynamic_title_weights,
    dynamic_lyric_weights,
    dynamic_comment_weights,
    history_preference=history_preference,
    global_history_w=global_history_weight,
)

if blocked_tags:
    scored_df = scored_df[
        ~scored_df["generated_tag_list"].apply(lambda tags: any(tag in tags for tag in blocked_tags))
    ]

if not scored_df.empty:
    min_possible_score = int(scored_df["dynamic_score"].min())
    max_possible_score = int(scored_df["dynamic_score"].max())
else:
    min_possible_score, max_possible_score = 0, 100
if min_possible_score >= max_possible_score:
    max_possible_score = min_possible_score + 1
default_min_score = 0 if min_possible_score <= 0 <= max_possible_score else min_possible_score
min_dynamic_score = st.sidebar.slider(
    "最低推荐评分阈值",
    min_value=min_possible_score,
    max_value=max_possible_score,
    value=default_min_score,
)

filtered_df = scored_df[scored_df["dynamic_score"] >= min_dynamic_score].copy()
filtered_df = filter_by_keywords(filtered_df, search_kw)

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
if selected_generated_tags:
    filtered_df = filtered_df[
        filtered_df["generated_tag_list"].apply(lambda tags: any(tag in tags for tag in selected_generated_tags))
    ]
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

st.title("墨白的音乐仓库")

metric_cols = st.columns(5)
metric_cols[0].metric("当前歌曲", f"{len(filtered_df)} / {len(df_base)}")
metric_cols[1].metric("歌手数", len({artist for artists in filtered_df["artist_list"] for artist in artists}))
metric_cols[2].metric("有歌词", int(filtered_df["has_lyric"].sum()))
metric_cols[3].metric("可播放", int(filtered_df["playable"].sum()))
metric_cols[4].metric("平均推荐分", f"{filtered_df['dynamic_score'].mean():.1f}" if not filtered_df.empty else "0.0")

tab_overview, tab_library, tab_lyrics, tab_detail, tab_history = st.tabs(
    ["推荐总览", "歌曲列表", "歌词详情", "歌曲详情", "历史记录"]
)

with tab_overview:
    if filtered_df.empty:
        st.info("没有匹配当前筛选条件的歌曲。")
    else:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader("推荐候选")
            top_df = filtered_df.sort_values(["dynamic_score", "comment_total"], ascending=False).head(20)
            overview_columns = [
                "album_pic_url",
                "选中",
                "dynamic_score",
                "name",
                "artist_names",
                "album_name",
                "all_tags",
                "score_breakdown",
                "quality",
                "duration_text",
                "comment_total",
                "netease_url",
            ]
            overview_table = make_selectable_table(
                top_df[
                    [
                        "song_id",
                        "album_pic_url",
                        "dynamic_score",
                        "name",
                        "artist_names",
                        "album_name",
                        "all_tags",
                        "score_breakdown",
                        "quality",
                        "duration_text",
                        "comment_total",
                        "netease_url",
                    ]
                ]
            )
            edited_overview = st.data_editor(
                overview_table,
                column_config={
                    "album_pic_url": st.column_config.ImageColumn("封面"),
                    "选中": st.column_config.CheckboxColumn("选中", width="small"),
                    "dynamic_score": st.column_config.ProgressColumn(
                        "推荐分",
                        min_value=min_possible_score,
                        max_value=max_possible_score,
                        format="%d",
                    ),
                    "name": "歌曲",
                    "artist_names": "歌手",
                    "album_name": "专辑",
                    "all_tags": "智能标签",
                    "score_breakdown": "评分拆解",
                    "quality": "音质",
                    "duration_text": "时长",
                    "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                    "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
                },
                column_order=overview_columns,
                disabled=[col for col in overview_table.columns if col != "选中"],
                hide_index=True,
                width="stretch",
                height=620,
                key=f"overview-select-{current_selected_song_id()}",
            )
            apply_table_selection(edited_overview, top_df)

        with right:
            st.subheader("分布画像")
            chart_tab_global, chart_tab_history = st.tabs(["全局曲库", "个人历史"])

            with chart_tab_global:
                st.caption("数据来源：完整曲库预处理缓存")
                render_preference_chart_grid(build_global_preference_chart_data(scoring_resources))

            with chart_tab_history:
                st.caption("数据来源：datacache/recommendation_history.json")
                render_preference_chart_grid(build_history_preference_chart_data(history_chart_entries))

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

        library_columns = [
            "album_pic_url",
            "选中",
            "dynamic_score",
            "song_id",
            "name",
            "artist_names",
            "album_name",
            "all_tags",
            "score_breakdown",
            "publish_date",
            "quality",
            "duration_text",
            "popularity",
            "comment_total",
            "has_lyric",
            "has_translation",
            "netease_url",
        ]
        library_table = make_selectable_table(
            display_df[
                [
                    "album_pic_url",
                    "dynamic_score",
                    "song_id",
                    "name",
                    "artist_names",
                    "album_name",
                    "all_tags",
                    "score_breakdown",
                    "publish_date",
                    "quality",
                    "duration_text",
                    "popularity",
                    "comment_total",
                    "has_lyric",
                    "has_translation",
                    "netease_url",
                ]
            ]
        )
        edited_library = st.data_editor(
            library_table,
            column_config={
                "album_pic_url": st.column_config.ImageColumn("封面"),
                "选中": st.column_config.CheckboxColumn("选中", width="small"),
                "dynamic_score": st.column_config.ProgressColumn(
                    "推荐分",
                    min_value=min_possible_score,
                    max_value=max_possible_score,
                    format="%d",
                ),
                "song_id": "ID",
                "name": "歌曲",
                "artist_names": "歌手",
                "album_name": "专辑",
                "all_tags": "智能标签",
                "score_breakdown": "评分拆解",
                "publish_date": "发行日期",
                "quality": "音质",
                "duration_text": "时长",
                "popularity": st.column_config.NumberColumn("热度", format="%d"),
                "comment_total": st.column_config.NumberColumn("评论", format="%d"),
                "has_lyric": "歌词",
                "has_translation": "翻译",
                "netease_url": st.column_config.LinkColumn("链接", display_text="打开"),
            },
            column_order=library_columns,
            disabled=[col for col in library_table.columns if col != "选中"],
            hide_index=True,
            width="stretch",
            height=650,
            key=f"library-select-{current_selected_song_id()}-{page_index}-{sort_label}-{order_label}",
        )
        apply_table_selection(edited_library, display_df)

        st.markdown("---")
        st.subheader("当前筛选分布画像")
        render_dataframe_chart_section(build_dataframe_chart_data(filtered_df, prefix="当前筛选"))

with tab_lyrics:
    lyric_df = filtered_df[filtered_df["full_lyric"].str.len() > 0].copy()
    if lyric_df.empty:
        st.info("当前筛选范围内没有本地歌词。")
    else:
        selected_song, matched_selected = selected_or_first(lyric_df, ["dynamic_score", "lyrics_chars"])
        if not matched_selected and current_selected_song_id():
            st.caption("当前选中的歌曲不在歌词结果中，已显示当前筛选范围内推荐最高的有歌词歌曲。")

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
        selected_song, matched_selected = selected_or_first(filtered_df, ["dynamic_score", "comment_total"])
        if not matched_selected and current_selected_song_id():
            st.caption("当前选中的歌曲不在详情结果中，已显示当前筛选范围内推荐最高的歌曲。")
        render_detail(selected_song)

with tab_history:
    st.subheader("历史记录")
    st.checkbox(
        "选中歌曲算入历史记录",
        key=SELECTION_WRITES_HISTORY_STATE_KEY,
        help="关闭后，勾选歌曲只会切换当前歌曲，不会新增历史偏好记录。",
    )
    st.caption(f"缓存最近 {HISTORY_RECOMMENDATION_CACHE_SIZE} 次选中记录，当前已保存 {len(history_entries)} 条。")

    col_refresh_history, col_clear_history = st.columns(2)
    with col_refresh_history:
        if st.button("刷新记录", width="stretch", key="history-refresh"):
            st.rerun()
    with col_clear_history:
        if st.button("清空记录", width="stretch", key="history-clear"):
            clear_history_entries()
            st.toast("已清空历史偏好记录", icon="✅")
            st.rerun()

    history_table = build_history_table(history_entries)
    if history_table.empty:
        st.info("暂时还没有保存的历史记录。")
    else:
        edited_history = st.data_editor(
            history_table,
            column_config={
                "删除": st.column_config.CheckboxColumn("删除", width="small"),
                "序号": st.column_config.NumberColumn("序号", format="%d", width="small"),
                "选中时间": "选中时间",
                "动作": "动作",
                "歌曲ID": "歌曲ID",
                "歌曲": "歌曲",
                "歌手": "歌手",
                "专辑": "专辑",
                "标签": "标签",
            },
            column_order=["删除", "序号", "选中时间", "歌曲ID", "歌曲", "歌手", "专辑", "标签", "动作"],
            disabled=[col for col in history_table.columns if col != "删除"],
            hide_index=True,
            width="stretch",
            height=560,
            key=f"history-record-editor-{len(history_entries)}",
        )

        rows_to_delete = edited_history[edited_history["删除"].fillna(False)]
        selected_delete_count = len(rows_to_delete)
        if st.button(
            f"删除选中的 {selected_delete_count} 条记录",
            width="stretch",
            disabled=selected_delete_count == 0,
            key="history-delete-selected",
        ):
            indices_to_delete = set((rows_to_delete["序号"].astype(int) - 1).tolist())
            remaining_entries = [
                entry for index, entry in enumerate(history_entries) if index not in indices_to_delete
            ]
            save_history_entries(remaining_entries)
            st.toast(f"已删除 {selected_delete_count} 条历史记录", icon="✅")
            st.rerun()
