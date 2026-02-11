import json
from datetime import datetime
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def report_node(state: dict) -> dict:
    """Report 節點：根據所有分析結果，由 LLM 生成改善建議報告。"""
    clusters = state.get("clusters")
    sentiment_result = state.get("sentiment_result")
    chart_paths = state.get("chart_paths", {})
    knowledge_map_path = state.get("knowledge_map_path")
    raw_data = state.get("raw_data", [])
    dataframe_summary = state.get("dataframe_summary", "")

    llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)

    # 組裝分析摘要給 LLM
    analysis_parts = []

    analysis_parts.append(f"資料概況：{dataframe_summary}")
    analysis_parts.append(f"總資料筆數：{len(raw_data)} 筆")

    if sentiment_result:
        analysis_parts.append(
            f"情感分析：正面 {sentiment_result.get('正面', 0)} 筆、"
            f"負面 {sentiment_result.get('負面', 0)} 筆、"
            f"中性 {sentiment_result.get('中性', 0)} 筆"
        )

    if clusters:
        cluster_labels = clusters.get("cluster_labels", {})
        items = clusters.get("items", [])
        cluster_summary_parts = []
        for cid, label in cluster_labels.items():
            count = sum(1 for item in items if str(item.get("cluster_id")) == str(cid))
            # 取幾則範例
            examples = [item["content"] for item in items if str(item.get("cluster_id")) == str(cid)][:3]
            examples_text = "；".join(examples)
            cluster_summary_parts.append(f"「{label}」({count}筆)：{examples_text}")
        analysis_parts.append("分群結果：\n" + "\n".join(cluster_summary_parts))

    if chart_paths:
        analysis_parts.append(f"已生成圖表：{', '.join(chart_paths.keys())}")

    if knowledge_map_path:
        analysis_parts.append("已生成 Knowledge Map")

    analysis_text = "\n\n".join(analysis_parts)

    prompt = f"""你是一位專業的客服品質分析顧問。請根據以下分析結果，生成一份完整的客戶回饋分析報告。

分析結果：
{analysis_text}

請使用以下 Markdown 格式輸出報告：

# 顧客回饋分析報告
**分析日期**：{datetime.now().strftime('%Y-%m-%d')}
**資料筆數**：{len(raw_data)} 筆

## 執行摘要
（簡要說明本次分析的整體發現，3-5 句話）

## 情感分析結果
（描述正面/負面/中性比例及其含義）

## 主要問題分群
（列出每個分群的名稱、數量、主要問題及具體回饋範例）

## Knowledge Map 洞察
（從 Knowledge Map 的階層結構中提取的關鍵洞察）

## 改善建議
### 短期（1 個月內）
（列出 2-3 項可立即執行的改善措施）

### 中期（3 個月內）
（列出 2-3 項需要規劃執行的改善措施）

### 長期（6 個月以上）
（列出 1-2 項策略性的改善方向）

請確保報告內容具體、可執行，並基於實際數據。"""

    try:
        response = llm.invoke([
            {"role": "system", "content": "你是一位專業的客服品質分析顧問，擅長從數據中提取洞察並提出可執行的建議。"},
            {"role": "user", "content": prompt},
        ])
        report = response.content
    except Exception as e:
        report = f"# 顧客回饋分析報告\n\n**分析日期**：{datetime.now().strftime('%Y-%m-%d')}\n\n❌ 報告生成失敗：{str(e)}"

    # 儲存報告到檔案
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "report": report,
        "messages": [AIMessage(content=f"📝 改善建議報告已生成！\n\n{report[:200]}...\n\n完整報告已儲存至：{report_path}")],
    }
