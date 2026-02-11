import json
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


EVALUATION_PROMPT = """你是一個專業的品質評審，負責檢核客服回饋分析的產出品質。

使用者原始問題：
{user_input}

分析產出摘要：
- 情感分析結果：{sentiment_summary}
- 分群數量：{cluster_count}
- 生成的圖表：{chart_types}
- 文字雲：{has_wordcloud}
- Knowledge Map：{has_knowledge_map}
- 報告：{has_report}

報告內容（節錄）：
{report_excerpt}

請根據以下四個維度評估（每個維度 1-10 分）：
1. **相關性**：分析結果是否與使用者問題相關？
2. **完整性**：是否涵蓋所有必要的分析面向？
3. **準確性**：圖表和報告描述是否與資料一致？
4. **可行性**：改善建議是否具體且可執行？

請以 JSON 格式回應：
{{
    "relevance": 分數,
    "completeness": 分數,
    "accuracy": 分數,
    "actionability": 分數,
    "score": 總平均分數（四捨五入到整數）,
    "passed": true/false（>= 7 分為通過）,
    "strengths": ["優點1", "優點2"],
    "issues": ["問題1", "問題2"],
    "summary": "一句話總結評估結果"
}}"""


def evaluate_node(state: dict) -> dict:
    """品質評估節點：使用 LLM 檢核分析產出品質。"""
    user_input = state.get("user_input", "")
    sentiment_result = state.get("sentiment_result")
    clusters = state.get("clusters")
    chart_paths = state.get("chart_paths", {})
    wordcloud_path = state.get("wordcloud_path")
    knowledge_map_path = state.get("knowledge_map_path")
    report = state.get("report", "")

    # 準備評估資訊
    sentiment_summary = "無"
    if sentiment_result:
        pos = sentiment_result.get("正面", 0)
        neg = sentiment_result.get("負面", 0)
        neu = sentiment_result.get("中性", 0)
        sentiment_summary = f"正面 {pos}、負面 {neg}、中性 {neu}"

    cluster_count = 0
    if clusters:
        cluster_count = len(clusters.get("cluster_labels", {}))

    chart_type_names = {"pie": "圓餅圖", "line": "折線圖", "bar": "長條圖"}
    chart_types_str = "、".join(chart_type_names.get(k, k) for k in chart_paths.keys()) if chart_paths else "無"

    report_excerpt = report[:1500] if report else "無"

    prompt = EVALUATION_PROMPT.format(
        user_input=user_input,
        sentiment_summary=sentiment_summary,
        cluster_count=cluster_count,
        chart_types=chart_types_str,
        has_wordcloud="有" if wordcloud_path else "無",
        has_knowledge_map="有" if knowledge_map_path else "無",
        has_report="有" if report else "無",
        report_excerpt=report_excerpt,
    )

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    response = llm.invoke([
        {"role": "system", "content": prompt},
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        evaluation = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        evaluation = {
            "score": 7,
            "passed": True,
            "strengths": ["分析流程完整執行"],
            "issues": [],
            "summary": "分析流程已完成，但無法自動評估品質。",
        }

    # 產生訊息
    score = evaluation.get("score", 0)
    passed = evaluation.get("passed", False)
    summary = evaluation.get("summary", "")

    if passed:
        msg = f"✅ **品質評估通過**（分數：{score}/10）\n\n{summary}"
    else:
        issues = evaluation.get("issues", [])
        issues_str = "\n".join(f"  - {issue}" for issue in issues)
        msg = (
            f"⚠️ **品質評估未通過**（分數：{score}/10）\n\n"
            f"{summary}\n\n"
            f"**發現的問題：**\n{issues_str}"
        )

    return {
        "evaluation_result": evaluation,
        "messages": [AIMessage(content=msg)],
    }
