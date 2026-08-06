"""图片筛选结果展示页（Streamlit）。

独立运行：
    streamlit run guardian/streamlit_app.py -- --results /path/to/gallery_results.json

或在代码中：
    from guardian.gallery import GalleryManager
    GalleryManager.launch_streamlit(results, data_dir)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Gallery Viewer")
    parser.add_argument("--results", required=True, help="Gallery results JSON path")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: cannot load results: {e}", file=sys.stderr)
        sys.exit(1)

    # Streamlit import (may fail if not installed)
    try:
        import streamlit as st
    except ImportError:
        print(
            "streamlit 未安装。请: pip install streamlit",
            file=sys.stderr,
        )
        print(f"结果文件: {results_path}", file=sys.stderr)
        sys.exit(1)

    st.set_page_config(
        page_title="图片筛选展示",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("🖼️ 训练成果展示")

    gallery_names = list(data.keys())
    if not gallery_names:
        st.warning("没有找到图集数据")
        return

    # 侧边栏：图集选择
    with st.sidebar:
        st.header("图集")
        selected_gallery = st.radio(
            "选择图集",
            gallery_names,
            format_func=lambda x: f"{x} ({len(data.get(x, []))} 张)",
        )

        st.divider()

        # 置信度过滤滑块
        st.header("全局过滤")
        min_conf = st.slider("最小置信度", 0.0, 1.0, 0.0, 0.05)
        max_conf = st.slider("最大置信度", 0.0, 1.0, 1.0, 0.05)

        st.divider()
        st.caption("Training Guardian · Gallery Viewer")

    # 主区域
    images = data.get(selected_gallery, [])

    # 全局过滤
    images = [
        img for img in images
        if img.get("confidence", img.get("score", 0)) is not None
        and min_conf <= (img.get("confidence", img.get("score", 0)) or 0) <= max_conf
    ]

    if not images:
        st.info("没有匹配的图片（请调整过滤条件）")
        return

    st.subheader(f"{selected_gallery} — {len(images)} 张")

    # 网格展示
    cols_per_row = 4
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(images):
                break
            img = images[idx]
            with col:
                img_path = img.get("image_path", img.get("path", ""))
                if img_path and Path(img_path).exists():
                    st.image(str(img_path), use_container_width=True)

                # 元数据
                conf = img.get("confidence", img.get("score"))
                pred = img.get("predicted_class", img.get("prediction"))
                label = img.get("true_label", img.get("label"))

                meta_lines = []
                if conf is not None:
                    conf_color = "green" if conf >= 0.9 else ("orange" if conf >= 0.5 else "red")
                    meta_lines.append(f"置信度: :{conf_color}[{conf:.3f}]")
                if pred is not None:
                    meta_lines.append(f"预测: {pred}")
                if label is not None:
                    meta_lines.append(f"标签: {label}")

                if meta_lines:
                    st.markdown("  \n".join(meta_lines))

    # 底栏统计
    st.divider()
    if images:
        confs = [
            (img.get("confidence", img.get("score", 0)) or 0)
            for img in images
            if img.get("confidence", img.get("score")) is not None
        ]
        if confs:
            col1, col2, col3 = st.columns(3)
            col1.metric("平均置信度", f"{sum(confs) / len(confs):.3f}")
            col2.metric("最高置信度", f"{max(confs):.3f}")
            col3.metric("最低置信度", f"{min(confs):.3f}")


if __name__ == "__main__":
    main()
