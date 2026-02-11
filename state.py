from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    # 對話管理
    messages: Annotated[list[BaseMessage], add_messages]  # LangChain message 歷史
    user_input: str                                       # 本輪使用者輸入的原始文字

    # 執行計畫
    plan: Optional[list[str]]             # Supervisor 生成的步驟清單
    plan_approved: bool                   # 使用者是否已確認 plan
    current_step: int                     # 目前執行到第幾步（0-indexed）
    awaiting_human: bool                  # 是否正在等待使用者回應
    interrupt_message: Optional[str]      # 顯示給使用者的提示訊息

    # 原始資料
    sheet_url: Optional[str]             # Google Sheet URL
    raw_data: Optional[list[dict]]       # 從 Sheet 讀取的原始資料（list of row dicts）
    dataframe_summary: Optional[str]     # 資料摘要文字，供 LLM 理解資料概況

    # 分析結果
    sentiment_result: Optional[dict]     # 情感分析結果
    clusters: Optional[dict]             # 分群結果

    # Knowledge Map 資料
    knowledge_map_data: Optional[dict]   # 階層樹狀結構

    # 視覺化輸出路徑
    wordcloud_path: Optional[str]        # 文字雲圖片路徑
    chart_paths: Optional[dict]          # {"pie": str, "line": str, "bar": str}
    knowledge_map_path: Optional[str]    # Knowledge Map HTML 路徑

    # 使用者意圖解析
    query_context: Optional[dict]       # {"intent", "time_range", "chart_types", "needs_clarification", "clarification_question"}

    # 最終報告
    report: Optional[str]               # Markdown 格式的改善建議報告

    # 品質評估
    evaluation_result: Optional[dict]   # 品質評估結果（score, passed, strengths, issues, summary）
