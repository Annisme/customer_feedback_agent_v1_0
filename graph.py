from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from state import State
from nodes.supervisor import supervisor_node
from nodes.fetch import fetch_node
from nodes.cluster import cluster_node
from nodes.knowledge_map import knowledge_map_node
from nodes.wordcloud_gen import wordcloud_node
from nodes.chart import chart_node
from nodes.report import report_node
from nodes.evaluate import evaluate_node


def human_approval_node(state: dict) -> dict:
    """Human-in-the-loop 節點：等待使用者確認。"""
    message = state.get("interrupt_message", "是否繼續？")
    response = interrupt(message)

    # 使用者回應 "approved" 或同意類的文字
    if response in ("approved", "同意", "確認", "繼續", "好", "是"):
        return {
            "plan_approved": True,
            "awaiting_human": False,
        }
    else:
        # 使用者給出修改意見，交回 supervisor 重新規劃
        return {
            "user_input": response,
            "plan": None,  # 清除舊 plan，讓 supervisor 重新規劃
            "query_context": None,  # 清除舊意圖，讓 supervisor 重新解析
            "plan_approved": False,
            "awaiting_human": False,
        }


def step_complete_node(state: dict) -> dict:
    """步驟完成後的中繼節點，推進 current_step。"""
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    raw_data = state.get("raw_data")

    # 如果剛執行完 fetch 但 raw_data 仍為 None，代表讀取失敗
    # 直接跳到最後，觸發完成流程
    if plan and current_step < len(plan) and plan[current_step] == "fetch" and raw_data is None:
        return {
            "current_step": len(plan),  # 跳到結尾，停止後續步驟
        }

    return {
        "current_step": current_step + 1,
    }


def _route_from_supervisor(state: dict) -> str:
    """從 Supervisor 決定下一步路由。"""
    plan = state.get("plan")
    plan_approved = state.get("plan_approved", False)
    awaiting_human = state.get("awaiting_human", False)
    current_step = state.get("current_step", 0)

    # 需要等待人類確認
    if awaiting_human:
        return "human_approval"

    # plan 尚未確認
    if plan and not plan_approved:
        return "human_approval"

    # 所有步驟已完成
    if plan and plan_approved and current_step >= len(plan):
        return "human_approval"

    # 路由到對應的 worker 節點
    if plan and plan_approved and current_step < len(plan):
        next_step = plan[current_step]
        node_map = {
            "fetch": "fetch",
            "cluster": "cluster",
            "knowledge_map": "knowledge_map",
            "wordcloud": "wordcloud",
            "chart": "chart",
            "report": "report",
            "evaluate": "evaluate",
        }
        return node_map.get(next_step, "supervisor")

    return "human_approval"


def _route_from_human(state: dict) -> str:
    """從 human approval 節點決定下一步。"""
    plan_approved = state.get("plan_approved", False)
    plan = state.get("plan")

    if not plan_approved or not plan:
        # 使用者不同意或需要重新規劃
        return "supervisor"

    return "supervisor"


def _route_after_step(state: dict) -> str:
    """每個 worker 步驟完成後回到 supervisor。"""
    return "supervisor"


def build_graph():
    """組裝 LangGraph 工作流程圖。"""
    workflow = StateGraph(State)

    # 添加節點
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("cluster", cluster_node)
    workflow.add_node("knowledge_map", knowledge_map_node)
    workflow.add_node("wordcloud", wordcloud_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("report", report_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("step_complete", step_complete_node)

    # 設定入口
    workflow.add_edge(START, "supervisor")

    # Supervisor 的條件路由
    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "human_approval": "human_approval",
            "fetch": "fetch",
            "cluster": "cluster",
            "knowledge_map": "knowledge_map",
            "wordcloud": "wordcloud",
            "chart": "chart",
            "report": "report",
            "evaluate": "evaluate",
        },
    )

    # Human approval 後的路由
    workflow.add_conditional_edges(
        "human_approval",
        _route_from_human,
        {
            "supervisor": "supervisor",
        },
    )

    # 每個 worker 節點完成後 → step_complete → supervisor
    for node_name in ["fetch", "cluster", "knowledge_map", "wordcloud", "chart", "report", "evaluate"]:
        workflow.add_edge(node_name, "step_complete")

    workflow.add_edge("step_complete", "supervisor")

    # 編譯
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph, checkpointer
