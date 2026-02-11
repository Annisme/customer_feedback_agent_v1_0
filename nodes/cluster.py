import json
import jieba
import numpy as np
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline


def _sentiment_analysis(texts: list[str]) -> dict:
    """使用 transformers 進行中文情感分析。"""
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model="uer/roberta-base-finetuned-jd-binary-chinese",
            truncation=True,
            max_length=512,
        )
    except Exception:
        # 如果模型載入失敗，使用基本的替代方案
        classifier = None

    positive = 0
    negative = 0
    neutral = 0
    details = []

    for text in texts:
        if classifier:
            try:
                result = classifier(text[:512])[0]
                label = result["label"]
                score = result["score"]

                if label == "positive" or label == "LABEL_1":
                    if score > 0.6:
                        sentiment = "正面"
                        positive += 1
                    else:
                        sentiment = "中性"
                        neutral += 1
                else:
                    if score > 0.6:
                        sentiment = "負面"
                        negative += 1
                    else:
                        sentiment = "中性"
                        neutral += 1

                details.append({"sentiment": sentiment, "score": round(score, 4)})
            except Exception:
                details.append({"sentiment": "中性", "score": 0.5})
                neutral += 1
        else:
            details.append({"sentiment": "中性", "score": 0.5})
            neutral += 1

    return {
        "正面": positive,
        "負面": negative,
        "中性": neutral,
        "details": details,
    }


def _cluster_texts(texts: list[str], n_clusters: int = 5) -> tuple[list[int], np.ndarray]:
    """使用 sentence-transformers + KMeans 進行文字分群。"""
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    # 確保分群數不超過資料數
    actual_k = min(n_clusters, len(texts))
    if actual_k < 2:
        return [0] * len(texts), embeddings

    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.tolist(), embeddings


def _name_clusters(cluster_items: dict[int, list[str]]) -> dict[str, str]:
    """使用 LLM 為每個分群命名。"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    cluster_labels = {}
    for cluster_id, items in cluster_items.items():
        sample = items[:10]  # 取最多 10 則作為範例
        sample_text = "\n".join(f"- {item}" for item in sample)

        response = llm.invoke([
            {"role": "system", "content": "你是一個客服回饋分析專家。請根據以下回饋內容，給這個分群一個簡短的中文標籤名稱（5個字以內）。只回傳標籤名稱。"},
            {"role": "user", "content": f"回饋內容：\n{sample_text}"},
        ])
        cluster_labels[str(cluster_id)] = response.content.strip()

    return cluster_labels


def _build_knowledge_map(cluster_labels: dict, cluster_items: dict) -> dict:
    """使用 LLM 建立階層樹狀 Knowledge Map 資料結構。"""
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # 準備分群摘要
    cluster_summary = ""
    for cid, label in cluster_labels.items():
        items = cluster_items.get(int(cid), [])[:5]
        items_text = "、".join(items)
        cluster_summary += f"\n分群「{label}」：{items_text}"

    prompt = f"""根據以下客服回饋分群結果，建立一個三層階層的 Knowledge Map：
大分類 → 小分類 → 關鍵字

分群資料：
{cluster_summary}

請以 JSON 格式回應，結構如下：
{{
    "name": "顧客回饋",
    "children": [
        {{
            "name": "大分類名稱",
            "children": [
                {{"name": "小分類名稱", "keywords": ["關鍵字1", "關鍵字2"]}},
                ...
            ]
        }},
        ...
    ]
}}

只回傳 JSON，不要其他文字。"""

    response = llm.invoke([
        {"role": "system", "content": "你是一個資料分析專家，擅長將客服回饋資料結構化。"},
        {"role": "user", "content": prompt},
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        # 回退方案：直接從分群建立簡單結構
        children = []
        for cid, label in cluster_labels.items():
            items = cluster_items.get(int(cid), [])
            # 使用 jieba 擷取關鍵字
            all_words = []
            for item in items:
                words = [w for w in jieba.cut(item) if len(w) > 1]
                all_words.extend(words)
            # 取前 5 個高頻字
            from collections import Counter
            top_words = [w for w, _ in Counter(all_words).most_common(5)]
            children.append({
                "name": label,
                "children": [{"name": label, "keywords": top_words}],
            })
        return {"name": "顧客回饋", "children": children}


def cluster_node(state: dict) -> dict:
    """Cluster 節點：意見分群 + 情感分析。"""
    raw_data = state.get("raw_data")

    if not raw_data:
        return {
            "messages": [AIMessage(content="❌ 尚無原始資料，請先執行資料讀取。")],
        }

    # 擷取回饋內容
    texts = [row.get("回饋內容", "") for row in raw_data if row.get("回饋內容")]
    feedback_ids = [row.get("回饋編號", f"FB{i}") for i, row in enumerate(raw_data) if row.get("回饋內容")]

    if not texts:
        return {
            "messages": [AIMessage(content="❌ 資料中沒有回饋內容可分析。")],
        }

    # 情感分析
    sentiment_result = _sentiment_analysis(texts)
    for i, detail in enumerate(sentiment_result["details"]):
        if i < len(feedback_ids):
            detail["回饋編號"] = feedback_ids[i]

    # 分群
    n_clusters = 5
    labels, _ = _cluster_texts(texts, n_clusters)

    # 組織分群項目
    cluster_items: dict[int, list[str]] = {}
    for i, (text, label) in enumerate(zip(texts, labels)):
        cluster_items.setdefault(label, []).append(text)

    # 命名分群
    cluster_labels = _name_clusters(cluster_items)

    # 組裝分群結果
    items_list = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        items_list.append({
            "回饋編號": feedback_ids[i] if i < len(feedback_ids) else f"FB{i}",
            "cluster_id": label,
            "content": text,
        })

    clusters = {
        "n_clusters": len(cluster_items),
        "cluster_labels": cluster_labels,
        "items": items_list,
    }

    # 建立 Knowledge Map 資料
    knowledge_map_data = _build_knowledge_map(cluster_labels, cluster_items)

    # 組裝摘要訊息
    cluster_summary = "\n".join(
        f"  - {cluster_labels.get(str(cid), f'群組{cid}')}：{len(items)} 筆"
        for cid, items in cluster_items.items()
    )
    sentiment_summary = f"正面 {sentiment_result['正面']} 筆、負面 {sentiment_result['負面']} 筆、中性 {sentiment_result['中性']} 筆"

    msg = f"🔍 分析完成：\n\n📊 情感分析：{sentiment_summary}\n\n📋 分群結果（共 {len(cluster_items)} 群）：\n{cluster_summary}"

    return {
        "sentiment_result": sentiment_result,
        "clusters": clusters,
        "knowledge_map_data": knowledge_map_data,
        "messages": [AIMessage(content=msg)],
    }
