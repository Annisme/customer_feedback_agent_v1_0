import re
import pandas as pd
from langchain_core.messages import AIMessage
from tools.gsheet import read_sheet


def _parse_time_range(time_range: str):
    """將自然語言時間範圍轉為 (start_date, end_date)。

    支援格式：
    - 季度：2024Q4, 2024 Q4, 2024q4
    - 年份：2024, 2024年
    - 月份：2024-10, 2024/10, 2024年10月
    - 近期：最近三個月, 最近半年 等
    """
    text = time_range.strip().upper()

    # 季度：2024Q4
    m = re.match(r"(\d{4})\s*Q(\d)", text)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        if q in quarter_starts:
            sm, sd = quarter_starts[q]
            em, ed = quarter_ends[q]
            return pd.Timestamp(year, sm, sd), pd.Timestamp(year, em, ed)

    # 年份：2024 或 2024年
    m = re.match(r"(\d{4})\s*年?$", text)
    if m:
        year = int(m.group(1))
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)

    # 月份：2024-10, 2024/10, 2024年10月
    m = re.match(r"(\d{4})[\-/年](\d{1,2})月?$", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthEnd(1)
        return start, end

    # 最近 N 個月
    m = re.search(r"最近\s*(\d+|[一二三四五六七八九十]+)\s*個月", time_range)
    if m:
        num_str = m.group(1)
        cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        n = cn_map.get(num_str, None) or int(num_str)
        end = pd.Timestamp.now()
        start = end - pd.DateOffset(months=n)
        return start, end

    # 最近半年
    if "半年" in time_range:
        end = pd.Timestamp.now()
        start = end - pd.DateOffset(months=6)
        return start, end

    return None, None


def fetch_node(state: dict) -> dict:
    """Fetch 節點：從 Google Sheet 讀取資料。"""
    sheet_url = state.get("sheet_url")

    if not sheet_url:
        return {
            "messages": [AIMessage(content="❌ 尚未提供 Google Sheet URL，請在側邊欄輸入。")],
            "raw_data": None,
            "dataframe_summary": None,
        }

    try:
        raw_data = read_sheet(sheet_url)
    except FileNotFoundError as e:
        return {
            "messages": [AIMessage(content=f"❌ {str(e)}")],
            "raw_data": None,
            "dataframe_summary": None,
        }
    except ValueError as e:
        return {
            "messages": [AIMessage(content=f"❌ 資料讀取錯誤：{str(e)}")],
            "raw_data": None,
            "dataframe_summary": None,
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"❌ 讀取 Google Sheet 時發生錯誤：{str(e)}")],
            "raw_data": None,
            "dataframe_summary": None,
        }

    if not raw_data:
        return {
            "messages": [AIMessage(content="❌ Google Sheet 中沒有資料。")],
            "raw_data": None,
            "dataframe_summary": None,
        }

    # 依 query_context.time_range 篩選資料
    df = pd.DataFrame(raw_data)
    total_before_filter = len(df)
    filter_desc = ""

    query_context = state.get("query_context") or {}
    time_range = query_context.get("time_range")

    if time_range and "回饋日期" in df.columns:
        df["回饋日期_dt"] = pd.to_datetime(df["回饋日期"], errors="coerce")
        before_count = len(df)

        start, end = _parse_time_range(time_range)
        if start and end:
            mask = (df["回饋日期_dt"] >= start) & (df["回饋日期_dt"] <= end)
            df = df[mask]
            filter_desc = f"（已篩選 {time_range}：{len(df)}/{before_count} 筆）"

        df = df.drop(columns=["回饋日期_dt"])
        raw_data = df.to_dict("records")

        if not raw_data:
            return {
                "messages": [AIMessage(content=f"⚠️ 在 {time_range} 範圍內找不到任何資料。")],
                "raw_data": None,
                "dataframe_summary": None,
            }

    # 生成資料摘要
    total_rows = len(df)
    summary_parts = [f"資料筆數：{total_rows} 筆{filter_desc}"]

    if "回饋日期" in df.columns:
        dates = pd.to_datetime(df["回饋日期"], errors="coerce").dropna()
        if not dates.empty:
            summary_parts.append(f"日期範圍：{dates.min().strftime('%Y-%m-%d')} ~ {dates.max().strftime('%Y-%m-%d')}")

    if "回饋類別" in df.columns:
        categories = df["回饋類別"].value_counts()
        cat_str = "、".join(f"{k}({v}筆)" for k, v in categories.items())
        summary_parts.append(f"回饋類別分佈：{cat_str}")

    if "評分" in df.columns:
        try:
            scores = pd.to_numeric(df["評分"], errors="coerce").dropna()
            if not scores.empty:
                summary_parts.append(f"平均評分：{scores.mean():.2f}（範圍 {scores.min():.0f} ~ {scores.max():.0f}）")
        except Exception:
            pass

    summary = "\n".join(summary_parts)
    msg = f"📊 已成功讀取資料：\n{summary}"

    return {
        "raw_data": raw_data,
        "dataframe_summary": summary,
        "messages": [AIMessage(content=msg)],
    }
