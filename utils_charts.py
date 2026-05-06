from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [part.strip() for part in stripped.split("|") if part.strip()]
    return []


def _unique_items(items):
    seen_items = set()
    unique_items = []
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        unique_items.append(item)
    return unique_items


def _counter_from_series(series):
    counter = Counter()
    for value in series:
        counter.update(_unique_items(_coerce_list(value)))
    return counter


def _counter_from_history(history_entries, field_name):
    counter = Counter()
    for entry in history_entries:
        if isinstance(entry, dict):
            counter.update(_unique_items(_coerce_list(entry.get(field_name))))
    return counter


def _counter_from_value_counts(series):
    counter = Counter()
    for value, count in series.value_counts(dropna=False).items():
        label = str(value).strip()
        if label and label.lower() != "nan":
            counter[label] += int(count)
    return counter


def top_counts(df, column, limit=12):
    counter = _counter_from_series(df[column])
    if not counter:
        return pd.DataFrame({"名称": [], "数量": []})
    return pd.DataFrame(counter.most_common(limit), columns=["名称", "数量"])


def build_chart_meta(title, counter, label_col, value_col, top_limit=15, table_limit=150):
    return {
        "title": title,
        "top_items": counter.most_common(top_limit),
        "table_items": counter.most_common(table_limit),
        "label_col": label_col,
        "value_col": value_col,
        "table_label_col": label_col,
        "table_value_col": value_col,
        "expander_label": f"查看 Top {table_limit} {label_col}",
    }


def build_dataframe_chart_data(df, prefix="当前筛选"):
    if df.empty:
        empty = Counter()
        return {
            "artists": build_chart_meta(f"{prefix}热门歌手", empty, "歌手", "数量"),
            "tags": build_chart_meta(f"{prefix}智能标签", empty, "标签", "数量"),
            "languages": build_chart_meta(f"{prefix}语种/场景", empty, "标签", "数量"),
            "quality": build_chart_meta(f"{prefix}音质分布", empty, "音质", "数量"),
            "years": pd.DataFrame({"publish_year": [], "数量": []}),
        }

    year_counts = (
        df.dropna(subset=["publish_year"])
        .assign(publish_year=lambda item: item["publish_year"].astype(int))
        .groupby("publish_year")
        .size()
        .reset_index(name="数量")
        .sort_values("publish_year")
    )

    return {
        "artists": build_chart_meta(
            f"{prefix}热门歌手",
            _counter_from_series(df["artist_list"]),
            "歌手",
            "数量",
        ),
        "tags": build_chart_meta(
            f"{prefix}智能标签",
            _counter_from_series(df["generated_tag_list"]),
            "标签",
            "数量",
        ),
        "languages": build_chart_meta(
            f"{prefix}语种/场景",
            _counter_from_series(df["language_tags"]),
            "语种/场景",
            "数量",
        ),
        "quality": build_chart_meta(
            f"{prefix}音质分布",
            _counter_from_value_counts(df["quality"]),
            "音质",
            "数量",
        ),
        "years": year_counts,
    }


def build_global_preference_chart_data(scoring_resources):
    return {
        "tags": build_chart_meta("全局智能标签", scoring_resources["all_tags"], "标签", "出现频次"),
        "artists": build_chart_meta("全局歌手", scoring_resources["artists"], "歌手", "歌曲数"),
        "title_terms": build_chart_meta("全局歌名关键词", scoring_resources["title_terms"], "关键词", "出现频次"),
        "lyric_terms": build_chart_meta("全局歌词关键词", scoring_resources["lyric_terms"], "关键词", "出现频次"),
        "comment_semantic": build_chart_meta("全局评论语义", scoring_resources["comment_semantic"], "语义", "出现频次"),
        "emotion": build_chart_meta("全局情绪标签", scoring_resources["emotion"], "情绪", "出现频次"),
    }


def build_history_preference_chart_data(history_entries):
    return {
        "tags": build_chart_meta(
            "历史偏好智能标签",
            _counter_from_history(history_entries, "all_tags"),
            "标签",
            "选中频次",
        ),
        "artists": build_chart_meta(
            "历史偏好歌手",
            _counter_from_history(history_entries, "artists"),
            "歌手",
            "选中频次",
        ),
        "title_terms": build_chart_meta(
            "历史偏好歌名关键词",
            _counter_from_history(history_entries, "title_terms"),
            "关键词",
            "选中频次",
        ),
        "lyric_terms": build_chart_meta(
            "历史偏好歌词关键词",
            _counter_from_history(history_entries, "lyric_terms"),
            "关键词",
            "选中频次",
        ),
        "comment_semantic": build_chart_meta(
            "历史偏好评论语义",
            _counter_from_history(history_entries, "comment_semantic"),
            "语义",
            "选中频次",
        ),
        "emotion": build_chart_meta(
            "历史偏好情绪标签",
            _counter_from_history(history_entries, "emotion"),
            "情绪",
            "选中频次",
        ),
    }


def render_ranked_bar_chart(items, label_col, value_col, height=220):
    if not items:
        st.caption("暂无数据")
        return

    chart_df = pd.DataFrame(items, columns=[label_col, value_col])
    sort_order = chart_df[label_col].tolist()
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                f"{label_col}:N",
                sort=sort_order,
                axis=alt.Axis(labelAngle=-45, labelLimit=0, labelOverlap=False, title=None),
            ),
            y=alt.Y(f"{value_col}:Q", title=None),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title=label_col),
                alt.Tooltip(f"{value_col}:Q", title=value_col),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def render_preference_chart_block(chart_meta, height=220):
    st.caption(chart_meta["title"])
    render_ranked_bar_chart(
        chart_meta["top_items"],
        chart_meta["label_col"],
        chart_meta["value_col"],
        height=height,
    )
    with st.expander(chart_meta["expander_label"], expanded=False):
        if chart_meta["table_items"]:
            st.dataframe(
                pd.DataFrame(
                    chart_meta["table_items"],
                    columns=[chart_meta["table_label_col"], chart_meta["table_value_col"]],
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("暂无数据")


def render_year_line_chart(year_counts, height=220):
    st.caption("发行年份")
    if year_counts.empty:
        st.caption("暂无数据")
        return
    chart = (
        alt.Chart(year_counts)
        .mark_line(point=True)
        .encode(
            x=alt.X("publish_year:O", title=None, sort=year_counts["publish_year"].astype(str).tolist()),
            y=alt.Y("数量:Q", title=None),
            tooltip=[
                alt.Tooltip("publish_year:O", title="年份"),
                alt.Tooltip("数量:Q", title="数量"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def render_dataframe_chart_section(chart_data):
    top_cols = st.columns(2)
    with top_cols[0]:
        render_preference_chart_block(chart_data["artists"], height=210)
    with top_cols[1]:
        render_preference_chart_block(chart_data["tags"], height=210)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        render_preference_chart_block(chart_data["languages"], height=180)
    with chart_cols[1]:
        render_preference_chart_block(chart_data["quality"], height=180)
    render_year_line_chart(chart_data["years"], height=190)


def render_preference_chart_grid(chart_data):
    first_row = st.columns(3)
    for column, key in zip(first_row, ["tags", "artists", "emotion"]):
        with column:
            render_preference_chart_block(chart_data[key])

    second_row = st.columns(3)
    for column, key in zip(second_row, ["title_terms", "lyric_terms", "comment_semantic"]):
        with column:
            render_preference_chart_block(chart_data[key])
