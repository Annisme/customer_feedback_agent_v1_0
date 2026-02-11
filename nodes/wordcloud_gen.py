import os
import jieba
from collections import Counter
from wordcloud import WordCloud
from langchain_core.messages import AIMessage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# 中文停用詞
STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一個",
    "上", "也", "很", "到", "說", "要", "去", "你", "會", "著", "沒有", "看", "好",
    "自己", "這", "他", "她", "它", "們", "這個", "那個", "什麼", "怎麼", "如果",
    "但是", "因為", "所以", "可以", "已經", "還是", "或者", "而且", "雖然", "不過",
    "可能", "應該", "比較", "非常", "真的", "覺得", "知道", "其實", "然後", "之後",
    "就是", "還有", "但", "與", "及", "等", "或", "讓", "被", "把", "從", "用",
    "對", "做", "能", "會", "想", "給", "跟", "個", "那", "這", "些", "吧", "嗎",
    "呢", "啊", "喔", "哦", "耶", "吶", "唷",
}


def wordcloud_node(state: dict) -> dict:
    """Wordcloud 節點：生成關鍵字文字雲。"""
    raw_data = state.get("raw_data")

    if not raw_data:
        return {
            "messages": [AIMessage(content="❌ 尚無原始資料，請先執行資料讀取。")],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 合併所有回饋內容
    texts = [row.get("回饋內容", "") for row in raw_data if row.get("回饋內容")]
    all_text = " ".join(texts)

    # jieba 斷詞並過濾停用詞
    words = jieba.cut(all_text)
    filtered_words = [w.strip() for w in words if w.strip() and len(w.strip()) > 1 and w.strip() not in STOP_WORDS]

    if not filtered_words:
        return {
            "messages": [AIMessage(content="❌ 斷詞後沒有有效詞彙可生成文字雲。")],
        }

    word_freq = Counter(filtered_words)

    # 嘗試找中文字型
    font_path = _find_chinese_font()

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        colormap="viridis",
        collocations=False,
    )
    wc.generate_from_frequencies(word_freq)

    output_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    wc.to_file(output_path)

    return {
        "wordcloud_path": output_path,
        "messages": [AIMessage(content=f"☁️ 文字雲已生成：{output_path}")],
    }


def _find_chinese_font() -> str:
    """尋找可用的中文字型路徑。"""
    possible_paths = [
        # Windows
        "C:/Windows/Fonts/msjh.ttc",        # 微軟正黑體
        "C:/Windows/Fonts/mingliu.ttc",      # 新細明體
        "C:/Windows/Fonts/kaiu.ttf",         # 標楷體
        "C:/Windows/Fonts/simsun.ttc",       # 宋體
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # 專案目錄
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NotoSansTC-Regular.ttf"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # 未找到，回傳 None（WordCloud 會使用預設字型，中文可能無法顯示）
    return None
