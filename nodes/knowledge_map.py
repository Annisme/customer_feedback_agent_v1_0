import os
import plotly.graph_objects as go
from langchain_core.messages import AIMessage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _flatten_tree(node: dict, parent: str = "") -> tuple[list[str], list[str], list[str]]:
    """將階層樹狀結構展平為 plotly treemap 所需的格式。"""
    ids = []
    labels = []
    parents = []

    current_id = f"{parent}/{node['name']}" if parent else node["name"]
    ids.append(current_id)
    labels.append(node["name"])
    parents.append(parent)

    for child in node.get("children", []):
        if "children" in child:
            child_ids, child_labels, child_parents = _flatten_tree(child, current_id)
            ids.extend(child_ids)
            labels.extend(child_labels)
            parents.extend(child_parents)
        elif "keywords" in child:
            # 葉節點 - 小分類
            child_id = f"{current_id}/{child['name']}"
            ids.append(child_id)
            labels.append(child["name"])
            parents.append(current_id)
            # 添加關鍵字作為更深的葉節點
            for kw in child.get("keywords", []):
                kw_id = f"{child_id}/{kw}"
                ids.append(kw_id)
                labels.append(kw)
                parents.append(child_id)
        else:
            child_id = f"{current_id}/{child['name']}"
            ids.append(child_id)
            labels.append(child["name"])
            parents.append(current_id)

    return ids, labels, parents


def knowledge_map_node(state: dict) -> dict:
    """Knowledge Map 節點：生成階層樹狀圖。"""
    knowledge_map_data = state.get("knowledge_map_data")

    if not knowledge_map_data:
        return {
            "messages": [AIMessage(content="❌ 尚無 Knowledge Map 資料，請先執行分群分析。")],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ids, labels, parents = _flatten_tree(knowledge_map_data)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        root_color="lightgrey",
        branchvalues="total",
        textinfo="label",
        hovertemplate="<b>%{label}</b><extra></extra>",
        marker=dict(
            colorscale="RdYlGn",
        ),
    ))

    fig.update_layout(
        title="顧客回饋 Knowledge Map",
        font=dict(size=14),
        margin=dict(t=50, l=25, r=25, b=25),
        width=900,
        height=600,
    )

    output_path = os.path.join(OUTPUT_DIR, "knowledge_map.html")
    fig.write_html(output_path, include_plotlyjs="cdn")

    return {
        "knowledge_map_path": output_path,
        "messages": [AIMessage(content=f"🗺️ Knowledge Map 已生成：{output_path}")],
    }
