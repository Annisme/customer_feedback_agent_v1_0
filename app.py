import os
import glob
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from graph import build_graph

load_dotenv()


def _clear_outputs():
    """清除 outputs/ 目錄中的所有產出檔案。"""
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    if os.path.exists(outputs_dir):
        for f in glob.glob(os.path.join(outputs_dir, "*")):
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


# ─── 頁面設定 ───
st.set_page_config(
    page_title="客服回饋智能分析助理",
    page_icon="🤖",
    layout="wide",
)

# ─── 初始化 session state ───
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph" not in st.session_state:
    graph, checkpointer = build_graph()
    st.session_state.graph = graph
    st.session_state.checkpointer = checkpointer
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {}
if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = False
if "sheet_url" not in st.session_state:
    st.session_state.sheet_url = ""
if "interrupted" not in st.session_state:
    st.session_state.interrupted = False

graph = st.session_state.graph
config = {"configurable": {"thread_id": st.session_state.thread_id}}


def get_latest_state() -> dict:
    """取得最新的 graph state。"""
    try:
        snapshot = graph.get_state(config)
        return snapshot.values if snapshot else {}
    except Exception:
        return {}


def run_graph(user_input: str = None, resume_value: str = None):
    """執行或恢復 graph。"""
    try:
        if resume_value is not None:
            # 恢復被中斷的執行
            result = graph.invoke(
                Command(resume=resume_value),
                config=config,
            )
        else:
            # 新的執行
            state = get_latest_state()
            input_state = {
                "user_input": user_input,
                "messages": [HumanMessage(content=user_input)],
                "sheet_url": st.session_state.sheet_url or state.get("sheet_url"),
            }
            result = graph.invoke(input_state, config=config)

        # 更新 state
        st.session_state.graph_state = get_latest_state()

        # 檢查是否被中斷
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next:
            st.session_state.interrupted = True
            st.session_state.awaiting_input = True
        else:
            st.session_state.interrupted = False
            st.session_state.awaiting_input = False

        return result

    except Exception as e:
        error_msg = f"執行時發生錯誤：{str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {error_msg}"})
        st.session_state.interrupted = False
        st.session_state.awaiting_input = False
        return None


def extract_messages_from_state(state: dict) -> list[str]:
    """從 state 中擷取 AI 訊息。"""
    messages = state.get("messages", [])
    ai_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            ai_messages.append(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            ai_messages.append(msg.get("content", ""))
    return ai_messages


# ─── 側邊欄 ───
with st.sidebar:
    st.title("📋 設定")

    # Google Sheet URL 輸入
    sheet_url = st.text_input(
        "Google Sheet URL",
        value=st.session_state.sheet_url,
        placeholder="https://docs.google.com/spreadsheets/d/...",
    )
    if sheet_url != st.session_state.sheet_url:
        st.session_state.sheet_url = sheet_url

    st.divider()

    # 執行步驟進度
    st.subheader("執行進度")
    state = get_latest_state()
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    plan_approved = state.get("plan_approved", False)

    step_names = {
        "fetch": "抓取資料",
        "cluster": "情感分析與分群",
        "knowledge_map": "Knowledge Map",
        "wordcloud": "文字雲",
        "chart": "生成圖表",
        "report": "生成報告",
        "evaluate": "品質評估",
    }

    if plan:
        for i, step in enumerate(plan):
            name = step_names.get(step, step)
            if plan_approved and i < current_step:
                st.markdown(f"✅ {name}")
            elif plan_approved and i == current_step:
                st.markdown(f"⏳ {name}")
            else:
                st.markdown(f"⬚ {name}")
    else:
        st.caption("尚未開始")

    st.divider()

    # 圖片縮圖預覽
    st.subheader("輸出預覽")
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    for img_name, label in [
        ("wordcloud.png", "文字雲"),
        ("chart_pie.png", "圓餅圖"),
        ("chart_line.png", "折線圖"),
        ("chart_bar.png", "長條圖"),
    ]:
        img_path = os.path.join(outputs_dir, img_name)
        if os.path.exists(img_path):
            st.caption(label)
            st.image(img_path, use_container_width=True)

    # 重置按鈕
    st.divider()
    if st.button("🔄 重新開始", use_container_width=True):
        _clear_outputs()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.graph_state = {}
        st.session_state.awaiting_input = False
        st.session_state.interrupted = False
        st.session_state.sheet_url = ""
        graph, checkpointer = build_graph()
        st.session_state.graph = graph
        st.session_state.checkpointer = checkpointer
        st.rerun()


# ─── 主對話區 ───
st.title("🤖 客服回饋智能分析助理")
st.caption("輸入自然語言指令，自動完成客戶回饋的資料分析、視覺化與報告生成。")

# ─── 功能導覽卡片（僅首次載入時顯示）───
if not st.session_state.chat_history:
    st.markdown("### 歡迎使用！以下是本系統支援的分析功能：")

    _features = [
        ("📥 資料讀取", "從 Google Sheet 讀取客戶回饋資料"),
        ("💬 情感分析與分群", "自動辨識正面/負面/中性情緒，並將相似意見歸類"),
        ("📊 視覺化圖表", "生成圓餅圖、折線圖、長條圖等統計圖表"),
        ("☁️ 文字雲", "以關鍵字頻率呈現客戶回饋的重點詞彙"),
        ("🗺️ Knowledge Map", "建立階層式知識地圖，呈現問題全貌"),
        ("📝 改善建議報告", "自動生成含短中長期行動建議的完整報告"),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)
    _rows = [row1, row2]
    for idx, (title, desc) in enumerate(_features):
        with _rows[idx // 3][idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)

    st.markdown("#### 💡 常用指令範例")
    st.markdown(
        "- **完整分析**：`請幫我做完整的客戶回饋分析`\n"
        "- **特定查詢**：`給我 2024Q4 的回饋數量統計`\n"
        "- **僅視覺化**：`幫我畫一張情感分析的圓餅圖`\n"
        "- **僅報告**：`根據目前的分析結果，產出改善建議報告`"
    )
    st.divider()

# 顯示對話記錄
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 如果訊息包含圖表路徑，顯示圖片
        if msg["role"] == "assistant":
            state = get_latest_state()

            # 嵌入圖表
            if "圖表已生成" in msg["content"] or "chart" in msg["content"].lower():
                chart_paths = state.get("chart_paths", {})
                for chart_type, path in chart_paths.items():
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)

            # 嵌入文字雲
            if "文字雲已生成" in msg["content"]:
                wc_path = state.get("wordcloud_path", "")
                if wc_path and os.path.exists(wc_path):
                    st.image(wc_path, use_container_width=True)

            # 嵌入 Knowledge Map
            if "Knowledge Map 已生成" in msg["content"]:
                km_path = state.get("knowledge_map_path", "")
                if km_path and os.path.exists(km_path):
                    with open(km_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)

            # 嵌入報告
            if "報告已生成" in msg["content"]:
                report = state.get("report", "")
                if report:
                    with st.expander("📄 查看完整報告", expanded=False):
                        st.markdown(report)

# ─── 中斷互動區 ───
if st.session_state.interrupted and st.session_state.awaiting_input:
    state = get_latest_state()
    interrupt_msg = state.get("interrupt_message", "")
    is_completed = "所有分析已完成" in interrupt_msg

    if interrupt_msg:
        st.info(interrupt_msg)

    if is_completed:
        # ─── 分析完成：顯示「結束」與「繼續提問」───
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 結束，開始新的分析", use_container_width=True):
                _clear_outputs()
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.session_state.graph_state = {}
                st.session_state.awaiting_input = False
                st.session_state.interrupted = False
                st.session_state.sheet_url = ""
                graph, checkpointer = build_graph()
                st.session_state.graph = graph
                st.session_state.checkpointer = checkpointer
                st.rerun()
        with col2:
            if st.button("💬 繼續提問", use_container_width=True, type="primary"):
                # 恢復 graph 讓它結束當前 interrupt，然後開放輸入框
                st.session_state.awaiting_input = False
                st.session_state.interrupted = False
                run_graph(resume_value="approved")
                st.rerun()
    else:
        # ─── Plan 確認卡片 ───
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 同意，開始執行", use_container_width=True, type="primary"):
                st.session_state.chat_history.append({"role": "user", "content": "同意，開始執行"})
                st.session_state.awaiting_input = False

                with st.spinner("執行中..."):
                    run_graph(resume_value="approved")

                # 更新對話記錄
                new_state = get_latest_state()
                messages = new_state.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if not any(h["content"] == content for h in st.session_state.chat_history if h["role"] == "assistant"):
                            st.session_state.chat_history.append({"role": "assistant", "content": content})

                st.rerun()

        with col2:
            modify_input = st.text_input("✏️ 輸入修改意見：", key="modify_input")
            if st.button("送出修改", use_container_width=True):
                if modify_input:
                    st.session_state.chat_history.append({"role": "user", "content": modify_input})
                    st.session_state.awaiting_input = False

                    with st.spinner("重新規劃中..."):
                        run_graph(resume_value=modify_input)

                    new_state = get_latest_state()
                    messages = new_state.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            content = msg.content
                            if not any(h["content"] == content for h in st.session_state.chat_history if h["role"] == "assistant"):
                                st.session_state.chat_history.append({"role": "assistant", "content": content})

                    st.rerun()

# ─── 使用者輸入 ───
if not st.session_state.awaiting_input:
    user_input = st.chat_input("請輸入指令，例如：幫我做完整分析 / 給我 2024Q4 的回饋統計 / 畫一張圓餅圖")

    if user_input:
        # 添加到對話記錄
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("思考中..."):
            result = run_graph(user_input=user_input)

        # 從 state 中擷取新的 AI 訊息
        new_state = get_latest_state()
        messages = new_state.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = msg.content
                if not any(h["content"] == content for h in st.session_state.chat_history if h["role"] == "assistant"):
                    st.session_state.chat_history.append({"role": "assistant", "content": content})

        st.rerun()
