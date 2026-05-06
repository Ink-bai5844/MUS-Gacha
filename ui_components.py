import html
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_core import open_local_file
from utils_history import record_recommendation_history
from utils_text import safe_text


def render_page_style():
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


def render_detail(song):
    left, right = st.columns([1.1, 2.4], gap="large")

    with left:
        cover = safe_text(song.get("album_pic_url", ""))
        if cover:
            st.image(cover, width="stretch")
        st.link_button("打开网易云页面", safe_text(song.get("netease_url", "")), width="stretch")

        local_audio_path = safe_text(song.get("local_audio_path", ""))
        if local_audio_path:
            local_path = Path(local_audio_path)
            if local_path.exists():
                st.caption("本地音频")
                st.audio(str(local_path))
                if st.button("打开本地音频", width="stretch", key=f"open-local-{song.get('song_id')}"):
                    ok, message = open_local_file(local_audio_path)
                    if ok:
                        record_recommendation_history(song, "local_audio")
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning(f"本地音频路径已失效：{local_audio_path}")
        else:
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
                ("智能标签", safe_text(song.get("all_tags"))),
                ("评分拆解", safe_text(song.get("score_breakdown"))),
                ("歌词关键词", " | ".join(song.get("lyric_terms", [])[:20])),
                ("评论语义", " | ".join(song.get("comment_semantic_tags", []))),
                ("本地音频", safe_text(song.get("local_audio_path"))),
                ("本地元数据", " · ".join([item for item in [
                    safe_text(song.get("local_audio_title")),
                    safe_text(song.get("local_audio_artist")),
                    safe_text(song.get("local_audio_album")),
                ] if item])),
                ("音频特征", " · ".join([item for item in [
                    f"{safe_text(song.get('audio_tempo_bpm'))} BPM" if safe_text(song.get("audio_tempo_bpm")) else "",
                    safe_text(song.get("audio_feature_tags")),
                ] if item])),
                ("人声/器乐", " · ".join([item for item in [
                    safe_text(song.get("vocal_instrumental_tags")),
                    f"人声 {safe_text(song.get('vocal_presence_score'))}" if safe_text(song.get("vocal_presence_score")) else "",
                    f"器乐 {safe_text(song.get('instrumental_presence_score'))}" if safe_text(song.get("instrumental_presence_score")) else "",
                ] if item])),
                ("声源分离", " · ".join([item for item in [
                    safe_text(song.get("source_separation_model")),
                    safe_text(song.get("source_separation_error")),
                ] if item])),
                ("声源分离占比", " · ".join([item for item in [
                    f"人声 {safe_text(song.get('source_vocals_energy_ratio') or song.get('source_vocal_energy_ratio'))}" if safe_text(song.get("source_vocals_energy_ratio") or song.get("source_vocal_energy_ratio")) else "",
                    f"鼓 {safe_text(song.get('source_drums_energy_ratio'))}" if safe_text(song.get("source_drums_energy_ratio")) else "",
                    f"贝斯 {safe_text(song.get('source_bass_energy_ratio'))}" if safe_text(song.get("source_bass_energy_ratio")) else "",
                    f"其它 {safe_text(song.get('source_other_energy_ratio'))}" if safe_text(song.get("source_other_energy_ratio")) else "",
                    f"伴奏合计 {safe_text(song.get('source_instrumental_energy_ratio'))}" if safe_text(song.get("source_instrumental_energy_ratio")) else "",
                ] if item])),
                ("MERT", " · ".join([item for item in [
                    f"聚类 {safe_text(song.get('mert_cluster'))}" if safe_text(song.get("mert_cluster")) else "",
                    safe_text(song.get("mert_emotion_tags")),
                ] if item])),
                ("歌词行数", str(int(song.get("lyric_line_count", 0)))),
                ("热门评论", safe_text(song.get("first_hot_comment"))),
                ("相似歌曲", safe_text(song.get("similar_song_names"))),
            ],
            columns=["字段", "内容"],
        )
        st.dataframe(info, hide_index=True, width="stretch", height=630)

    lyric = safe_text(song.get("full_lyric", "")) or safe_text(song.get("lyric_excerpt", ""))
    if lyric:
        st.markdown("#### 歌词")
        st.markdown(f"<div class='lyric-box'>{html.escape(lyric)}</div>", unsafe_allow_html=True)
    else:
        st.info("这首歌暂时没有本地歌词。")
