import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
from langchain_core.messages import AIMessage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _setup_chinese_font():
    """設定 matplotlib 中文字型。"""
    font_candidates = [
        "Microsoft JhengHei",  # 微軟正黑體 (Windows)
        "Microsoft YaHei",     # 微軟雅黑 (Windows)
        "SimHei",              # 黑體 (Windows)
        "PingFang TC",         # macOS
        "Noto Sans CJK TC",   # Linux
        "WenQuanYi Micro Hei", # Linux
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return

    # 如果找不到中文字型，設定 fallback
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _generate_pie_chart(sentiment_result: dict) -> str:
    """生成情感分析圓餅圖。"""
    _setup_chinese_font()

    labels = ["正面", "負面", "中性"]
    sizes = [
        sentiment_result.get("正面", 0),
        sentiment_result.get("負面", 0),
        sentiment_result.get("中性", 0),
    ]
    colors = ["#4CAF50", "#F44336", "#FFC107"]
    explode = (0.05, 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 14},
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
    ax.set_title("情感分析結果", fontsize=16, fontweight="bold")

    output_path = os.path.join(OUTPUT_DIR, "chart_pie.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _generate_line_chart(raw_data: list[dict]) -> str:
    """生成每月回饋趨勢折線圖。"""
    _setup_chinese_font()

    df = pd.DataFrame(raw_data)
    if "回饋日期" not in df.columns:
        return None

    df["回饋日期"] = pd.to_datetime(df["回饋日期"], errors="coerce")
    df = df.dropna(subset=["回饋日期"])
    df["月份"] = df["回饋日期"].dt.to_period("M").astype(str)

    monthly = df.groupby("月份").size().reset_index(name="回饋數量")
    monthly = monthly.sort_values("月份")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly["月份"], monthly["回饋數量"], marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(monthly["月份"], monthly["回饋數量"], alpha=0.1, color="#2196F3")
    ax.set_title("每月回饋數量趨勢", fontsize=16, fontweight="bold")
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("回饋數量", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(OUTPUT_DIR, "chart_line.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _generate_bar_chart(clusters: dict) -> str:
    """生成各分群回饋數量長條圖。"""
    _setup_chinese_font()

    cluster_labels = clusters.get("cluster_labels", {})
    items = clusters.get("items", [])

    # 計算每群數量
    cluster_counts = {}
    for item in items:
        cid = str(item.get("cluster_id", 0))
        label = cluster_labels.get(cid, f"群組 {cid}")
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    labels = list(cluster_counts.keys())
    values = list(cluster_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(labels))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # 在每根長條上方顯示數值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("各分群回饋數量", fontsize=16, fontweight="bold")
    ax.set_xlabel("分群", fontsize=12)
    ax.set_ylabel("回饋數量", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    output_path = os.path.join(OUTPUT_DIR, "chart_bar.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def chart_node(state: dict) -> dict:
    """Chart 節點：依 query_context 生成指定圖表，未指定則生成全部。"""
    raw_data = state.get("raw_data")
    sentiment_result = state.get("sentiment_result")
    clusters = state.get("clusters")

    # 讀取 query_context 中的 chart_types
    query_context = state.get("query_context") or {}
    requested_types = query_context.get("chart_types", [])
    # 若未指定，預設生成全部三種圖表
    if not requested_types:
        requested_types = ["pie", "line", "bar"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chart_paths = {}
    generated = []
    errors = []

    # 圓餅圖
    if "pie" in requested_types and sentiment_result:
        try:
            pie_path = _generate_pie_chart(sentiment_result)
            chart_paths["pie"] = pie_path
            generated.append("情感分析圓餅圖")
        except Exception as e:
            errors.append(f"圓餅圖生成失敗：{e}")

    # 折線圖
    if "line" in requested_types and raw_data:
        try:
            line_path = _generate_line_chart(raw_data)
            if line_path:
                chart_paths["line"] = line_path
                generated.append("月度趨勢折線圖")
        except Exception as e:
            errors.append(f"折線圖生成失敗：{e}")

    # 長條圖
    if "bar" in requested_types and clusters:
        try:
            bar_path = _generate_bar_chart(clusters)
            chart_paths["bar"] = bar_path
            generated.append("分群數量長條圖")
        except Exception as e:
            errors.append(f"長條圖生成失敗：{e}")

    msg_parts = []
    if generated:
        msg_parts.append(f"📊 已生成圖表：{'、'.join(generated)}")
    if errors:
        msg_parts.append(f"⚠️ 部分圖表生成失敗：{'；'.join(errors)}")
    if not generated and not errors:
        msg_parts.append("❌ 沒有足夠資料生成圖表，請先執行分析。")

    return {
        "chart_paths": chart_paths,
        "messages": [AIMessage(content="\n".join(msg_parts))],
    }
