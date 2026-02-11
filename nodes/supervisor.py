import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command


INTENT_PARSE_PROMPT = """你是一個客服回饋分析助理的意圖解析器。請分析使用者的指令，判斷他們想要什麼。

請以 JSON 格式回應，包含以下欄位：
{
    "intent": "full_analysis | specific_query | visualization_only | report_only",
    "time_range": "使用者提到的時間範圍，例如 '2024Q4'、'最近三個月'，若未提及則為 null",
    "chart_types": ["pie", "line", "bar"] 中使用者需要的圖表類型，若未指定則為 [],
    "needs_clarification": true/false，指令是否模糊不清需要釐清,
    "clarification_question": "若需要釐清，要問使用者的問題，否則為 null"
}

意圖判斷規則：
- full_analysis: 使用者要求完整分析、全面分析、或未明確指定範圍的分析
- specific_query: 使用者只想知道某個特定數據（如回饋數量、某時段的統計）
- visualization_only: 使用者只要求圖表或視覺化
- report_only: 使用者只要求產出報告

needs_clarification 判斷規則：
- 若指令過於模糊（例如只說「分析」但沒有其他脈絡），設為 true
- 若指令明確（例如「幫我做完整分析」、「畫一張圓餅圖」），設為 false
- 若使用者已提供 Google Sheet URL 或之前已有資料，指令通常足夠清楚"""

PLAN_SYSTEM_PROMPT = """你是一個客服回饋分析助理的 Supervisor。你的職責是：
1. 根據使用者意圖和解析結果，生成精準的執行計畫（plan）
2. 計畫應該只包含使用者真正需要的步驟

可用的步驟有：
- fetch: 從 Google Sheet 讀取資料
- cluster: 進行情感分析與意見分群
- knowledge_map: 生成 Knowledge Map 階層樹狀圖
- wordcloud: 生成文字雲
- chart: 生成圖表（圓餅圖、折線圖、長條圖）
- report: 生成改善建議報告

使用者意圖解析結果：
{query_context}

計畫規則：
- 每次執行都必須以 fetch 步驟開頭，因為需要先讀取資料才能進行任何分析
- intent 為 full_analysis 時，依序執行所有步驟：fetch → cluster → knowledge_map → wordcloud → chart → report
- intent 為 specific_query 時，通常只需 fetch → 相關分析步驟 → report
- intent 為 visualization_only 時，需要 fetch → 相關分析 → 圖表 → report
- intent 為 report_only 時，需要 fetch → 前置分析步驟 → report
- 若使用者指定了 chart_types，在 explanation 中說明只會生成指定的圖表
- 每次執行都必須以 report 步驟結尾，即使使用者只要求部分分析，也要生成報告

請以 JSON 格式回應：
{{
    "plan": ["步驟1", "步驟2", ...],
    "explanation": "對計畫的簡短說明"
}}

步驟名稱必須是上述可用步驟的英文名稱。"""

ROUTE_SYSTEM_PROMPT = """你是一個客服回饋分析助理的 Supervisor。
目前的執行計畫是：{plan}
目前已執行到第 {current_step} 步。

請根據目前狀態決定下一步要做什麼。
如果所有步驟都已完成，回傳 "done"。
否則回傳下一步的步驟名稱。

只回傳步驟名稱，不要其他文字。"""


def _get_llm():
    return ChatOpenAI(model="gpt-4.1", temperature=0)


def _parse_json_response(content: str) -> dict:
    """從 LLM 回應中擷取 JSON。"""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())


def supervisor_node(state: dict) -> dict:
    """Supervisor 節點：兩階段意圖解析 + 計畫生成。"""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    plan = state.get("plan")
    plan_approved = state.get("plan_approved", False)
    current_step = state.get("current_step", 0)
    query_context = state.get("query_context")

    llm = _get_llm()

    # 情境一：尚未有 plan，需要生成新計畫
    if not plan:
        # ── 第一階段：意圖解析 ──
        if not query_context:
            intent_response = llm.invoke([
                {"role": "system", "content": INTENT_PARSE_PROMPT},
                {"role": "user", "content": f"使用者指令：{user_input}"},
            ])

            try:
                query_context = _parse_json_response(intent_response.content)
            except (json.JSONDecodeError, IndexError):
                query_context = {
                    "intent": "full_analysis",
                    "time_range": None,
                    "chart_types": [],
                    "needs_clarification": False,
                    "clarification_question": None,
                }

        # 若需要釐清，透過 interrupt 向使用者提問
        if query_context.get("needs_clarification"):
            question = query_context.get("clarification_question", "請問您具體想要什麼樣的分析？")
            # 清除 needs_clarification 以避免回覆後再次觸發
            query_context["needs_clarification"] = False
            return {
                "query_context": query_context,
                "awaiting_human": True,
                "interrupt_message": f"🤔 {question}",
                "messages": [AIMessage(content=f"🤔 {question}")],
            }

        # ── 第二階段：計畫生成 ──
        plan_prompt = PLAN_SYSTEM_PROMPT.format(query_context=json.dumps(query_context, ensure_ascii=False))
        response = llm.invoke([
            {"role": "system", "content": plan_prompt},
            {"role": "user", "content": f"使用者指令：{user_input}"},
        ])

        try:
            plan_data = _parse_json_response(response.content)
            new_plan = plan_data.get("plan", [])
            explanation = plan_data.get("explanation", "")
        except (json.JSONDecodeError, IndexError):
            new_plan = ["fetch", "cluster", "knowledge_map", "wordcloud", "chart", "report"]
            explanation = "將執行完整分析流程"

        # 強制確保 fetch 為計畫的第一步（沒有資料就無法分析）
        if "fetch" not in new_plan:
            new_plan.insert(0, "fetch")
        elif new_plan[0] != "fetch":
            new_plan.remove("fetch")
            new_plan.insert(0, "fetch")

        # 強制確保 report 為計畫的倒數第二步
        if "report" not in new_plan:
            new_plan.append("report")
        elif new_plan[-1] != "report":
            new_plan.remove("report")
            new_plan.append("report")

        # 強制確保 evaluate 為計畫的最後一步（report → evaluate）
        if "evaluate" in new_plan:
            new_plan.remove("evaluate")
        new_plan.append("evaluate")

        # 格式化計畫顯示
        step_names = {
            "fetch": "讀取 Google Sheet 資料",
            "cluster": "進行情感分析與意見分群",
            "knowledge_map": "生成 Knowledge Map",
            "wordcloud": "生成文字雲",
            "chart": "生成圖表（圓餅圖、折線圖、長條圖）",
            "report": "輸出改善建議報告",
            "evaluate": "品質評估",
        }
        plan_display = "\n".join(
            f"  {i+1}. {step_names.get(s, s)}" for i, s in enumerate(new_plan)
        )
        interrupt_msg = f"📋 執行計畫：\n{plan_display}\n\n{explanation}\n\n是否同意開始執行？"

        return {
            "plan": new_plan,
            "query_context": query_context,
            "plan_approved": False,
            "current_step": 0,
            "awaiting_human": True,
            "interrupt_message": interrupt_msg,
            "messages": [AIMessage(content=interrupt_msg)],
        }

    # 情境二：有 plan 但尚未被確認 → 等待確認結果
    if plan and not plan_approved:
        return {
            "plan_approved": True,
            "awaiting_human": False,
            "current_step": 0,
            "messages": [AIMessage(content="✅ 計畫已確認，開始執行...")],
        }

    # 情境三：plan 已確認，決定下一步路由
    if plan_approved and current_step < len(plan):
        next_step = plan[current_step]
        step_names = {
            "fetch": "讀取資料",
            "cluster": "分析分群",
            "knowledge_map": "生成 Knowledge Map",
            "wordcloud": "生成文字雲",
            "chart": "生成圖表",
            "report": "生成報告",
            "evaluate": "品質評估",
        }
        return {
            "current_step": current_step,
            "awaiting_human": False,
            "messages": [AIMessage(content=f"⏳ 正在執行：{step_names.get(next_step, next_step)}...")],
        }

    # 情境四：所有步驟已完成
    return {
        "awaiting_human": True,
        "interrupt_message": "✅ 所有分析已完成！是否需要調整任何部分？",
        "messages": [AIMessage(content="✅ 所有分析已完成！是否需要調整任何部分？")],
    }
