"""
Al Brooks 结构训练器 V20
================================
数据接口：akshare 新浪财经期货数据
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
from datetime import datetime
import time
import json
import random

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="Al Brooks 结构训练器", layout="wide")

# ── 品种映射 ──────────────────────────────────────────────
SYMBOL_NAMES = {
    "RB": "螺纹钢", "HC": "热卷", "AU": "黄金", "AG": "白银",
    "CU": "铜", "AL": "铝", "ZN": "锌", "NI": "镍",
    "RU": "橡胶", "BU": "沥青", "FU": "燃油", "SC": "原油",
    "PB": "铅", "SN": "锡", "SS": "不锈钢", "SP": "纸浆",
    "I": "铁矿石", "J": "焦炭", "JM": "焦煤", "A": "豆一",
    "M": "豆粕", "Y": "豆油", "P": "棕榈油", "C": "玉米",
    "L": "塑料", "PP": "PP", "EG": "乙二醇", "EB": "苯乙烯",
    "PG": "LPG", "V": "PVC", "B": "豆二", "JD": "鸡蛋",
    "CF": "棉花", "SR": "白糖", "TA": "PTA", "MA": "甲醇",
    "FG": "玻璃", "SA": "纯碱", "OI": "菜油", "RM": "菜粕",
    "AP": "苹果", "ZC": "动力煤", "SF": "硅铁", "SM": "锰硅",
    "UR": "尿素", "PF": "短纤", "SH": "烧碱", "PX": "对二甲苯",
    "IF": "沪深300", "IC": "中证500", "IM": "中证1000",
    "IH": "上证50", "T": "十债", "TF": "五债", "TS": "两债",
    "SI": "工业硅", "LC": "碳酸锂",
}

# 品种分类显示
PRODUCT_CATEGORIES = {
    "金融": ["IF", "IH", "IC", "IM", "T", "TF", "TS"],
    "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
    "黑色": ["RB", "HC", "I", "JM", "J"],
    "化工": ["V", "PP", "L", "TA", "MA", "RU", "BU", "FU", "EG", "EB", "PG", "SA", "UR", "PF"],
    "农产品": ["M", "Y", "P", "A", "B", "C", "CS", "JD", "AP", "CF", "SR", "OI", "RM"],
    "能源": ["SC"],
}
DEFAULT_EXPANDED = ["金融"]

# ── 5个技能定义 ─────────────────────────────────────────
SKILLS = [
    {"id": 1, "name": "背景阅读",   "question": "当前市场背景是什么？"},
    {"id": 2, "name": "控制权识别", "question": "现在谁在控制市场？"},
    {"id": 3, "name": "推进质量",   "question": "最近推进的质量如何？"},
    {"id": 4, "name": "回调vs转换", "question": "这是正常回调还是控制权转换？"},
    {"id": 5, "name": "市场接受",   "question": "市场是否接受了新价格？"},
]

# ── AI配置 ──────────────────────────────────────────────
def get_ai_config():
    try:
        if "ai" in st.secrets:
            return {
                "base_url": st.secrets["ai"].get("base_url", "https://api.deepseek.com/v1"),
                "api_key": st.secrets["ai"].get("api_key", ""),
                "model": st.secrets["ai"].get("model", "deepseek-chat"),
            }
        else:
            return {
                "base_url": st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                "api_key": st.secrets.get("OPENAI_API_KEY", ""),
                "model": st.secrets.get("OPENAI_MODEL", "deepseek-chat"),
            }
    except Exception:
        return {
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
        }

AI_CONFIG = get_ai_config()

# ── 训练阶段配置 ─────────────────────────────────────────
LEVEL_CONFIG = {
    "level1": {"name": "观察阶段", "desc": "识别市场结构特征", "n_bars": 40},
    "level2": {"name": "行为细化阶段", "desc": "分析K线行为细节", "n_bars": 30},
    "level3": {"name": "结构验证阶段", "desc": "验证结构预期", "n_bars": 60},
}

# ── AI Prompt 模板 ──────────────────────────────────────
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为交易教练。

【核心职责】
1. 训练师：根据训练阶段和当前技能目的，提供结构化的分析指导
2. 点评师：对用户的分析给出专业点评

当前训练阶段：{level_name} - {level_desc}
当前技能：{skill_name}
核心提问：{skill_question}

【技能核心维度】
1. 背景阅读 → 趋势方向、高低点序列、通道斜率、区间边界
2. 控制权识别 → 谁在主导、推进力度、对手方反击
3. 推进质量 → 实体大小、重叠程度、影线长度、动能衰减
4. 回调vs转换 → 回调K线数量、对手方实体连续性、跟进情况
5. 市场接受 → 突破后停留时间、是否被推回、是否继续推进

【对话流程】
- 第1轮：描述市场结构，提出引导性问题，不给答案
- 第2轮：点评用户回答，亮出自己判断（引用具体K线编号）

【回答风格】
- 简短专业，不超过150字
- 基于具体K线分析"""


# ── 数据加载 ──────────────────────────────────────────────
def fetch_kline_data(symbol: str, period: str = "30"):
    """获取期货K线数据"""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df = df.rename(columns={
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
        
    except Exception as e:
        st.warning(f"数据获取失败: {e}")
        return None


def get_main_contract(symbol: str) -> str:
    """获取主力合约代码"""
    try:
        # 尝试直接获取主力合约
        main = ak.match_main_contract(symbol=symbol)
        if main and isinstance(main, str) and len(main) > 2:
            return main
    except Exception:
        pass
    
    # 尝试常见后缀格式
    for suffix in ["888", "0", "99"]:
        test_symbol = f"{symbol}{suffix}"
        try:
            df = ak.futures_zh_minute_sina(symbol=test_symbol, period="30")
            if df is not None and not df.empty:
                return test_symbol
        except Exception:
            continue
    
    # 默认返回原代码
    return symbol


@st.cache_data(ttl=300, show_spinner=False)
def load_symbol_data(symbol: str):
    """加载品种数据"""
    main_code = get_main_contract(symbol)
    df = fetch_kline_data(main_code)
    if df is not None and len(df) > 0:
        return df, main_code
    return None, None


# ── 市场消息描述 ──────────────────────────────────────────
def market_msg(kline_df: pd.DataFrame, n_bars: int = 40) -> str:
    """生成市场描述文本"""
    if kline_df is None or kline_df.empty:
        return "暂无数据"
    
    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return "数据不足"
    
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    
    lines = [f"【当前K线】第{n}号K线", "", "【最近行情描述】"]
    
    start = max(0, n - 15)
    for i in range(start, n):
        body = abs(c[i] - o[i])
        total_range = h[i] - l[i]
        body_ratio = body / total_range if total_range > 0 else 0
        direction = "阳" if c[i] >= o[i] else "阴"
        
        if body_ratio >= 0.7:
            k_type = "大阳线" if direction == "阳" else "大阴线"
        elif body_ratio >= 0.4:
            k_type = "中阳线" if direction == "阳" else "中阴线"
        elif body_ratio >= 0.1:
            k_type = "小阳线" if direction == "阳" else "小阴线"
        else:
            k_type = "十字星"
        
        lines.append(f"  K{i+1}: {k_type} 开{o[i]:.0f} 收{c[i]:.0f} 高{h[i]:.0f} 低{l[i]:.0f}")
    
    lines.extend(["", "【整体感知】"])
    
    if n >= 10:
        yang = sum(1 for j in range(n-10, n) if c[j] >= o[j])
        lines.append(f"  • 最近10根中{yang}阳{10-yang}阴")
        change = c[-1] - c[0]
        if change > 0:
            lines.append(f"  • 整体上涨{change:.1f}")
        elif change < 0:
            lines.append(f"  • 整体下跌{abs(change):.1f}")
    
    return "\n".join(lines)


# ── AI调用 ──────────────────────────────────────────────
def ask_coach(skill_name, skill_question, market_desc, level_name, level_desc, user_input="", is_second_round=False):
    """调用AI教练"""
    if not AI_CONFIG["api_key"]:
        return "⚠️ API Key未配置"
    
    client = OpenAI(
        base_url=AI_CONFIG["base_url"],
        api_key=AI_CONFIG["api_key"],
    )
    
    system_prompt = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        skill_question=skill_question,
        level_name=level_name,
        level_desc=level_desc,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"市场状况：\n{market_desc}"},
    ]
    
    if "chat_history" in st.session_state:
        for m in st.session_state.chat_history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
    
    if is_second_round:
        messages.append({
            "role": "user",
            "content": f"【第2轮】用户回答：{user_input}\n\n请点评并给出你的判断"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"【第1轮】技能：「{skill_name}」，提问：「{skill_question}」\n请描述市场并提出引导性问题"
        })
    
    try:
        resp = client.chat.completions.create(
            model=AI_CONFIG["model"],
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[AI错误] {str(e)[:100]}"


# ── 图表绘制 ──────────────────────────────────────────────
def plot_kline(kline_df: pd.DataFrame, n_bars: int = 40):
    """绘制K线图"""
    if kline_df is None or kline_df.empty:
        return go.Figure()
    
    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )
    
    if "volume" in df.columns:
        colors = ["#ef5350" if row["close"] >= row["open"] else "#26a69a" for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], marker_color=colors, opacity=0.5),
            row=2, col=1,
        )
    
    # K线编号
    for idx in range(0, len(df), 5):
        row = df.iloc[idx]
        fig.add_annotation(
            x=idx, y=row["high"],
            text=str(len(df) - idx),
            showarrow=False, yshift=8, font_size=8, font_color="#666",
            row=1, col=1,
        )
    
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig


# ── 主程序 ──────────────────────────────────────────────
def main():
    # 初始化 session_state
    defaults = {
        "data_loaded": False,
        "kline_data": None,
        "current_symbol": "RB",
        "current_skill": None,
        "skill_round": 1,
        "chat_history": [],
        "train_level": "level1",
        "main_contract": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 品种选择")
        
        for cat_name, symbols in PRODUCT_CATEGORIES.items():
            expanded = cat_name in DEFAULT_EXPANDED
            with st.expander(cat_name, expanded=expanded):
                cols = st.columns(min(4, len(symbols)))
                for i, sym in enumerate(symbols):
                    if cols[i % len(cols)].button(sym, key=f"sym_{cat_name}_{sym}"):
                        st.session_state.current_symbol = sym
                        st.session_state.data_loaded = False
                        st.session_state.kline_data = None
                        st.session_state.chat_history = []
                        st.session_state.skill_round = 1
                        st.rerun()
        
        st.markdown("---")
        
        # 训练阶段
        level_labels = ["观察阶段", "行为细化阶段", "结构验证阶段"]
        selected_level = st.selectbox("训练阶段", level_labels)
        level_map = {"观察阶段": "level1", "行为细化阶段": "level2", "结构验证阶段": "level3"}
        st.session_state.train_level = level_map[selected_level]
        
        level_info = LEVEL_CONFIG[st.session_state.train_level]
        n_bars = level_info["n_bars"]
        level_name = level_info["name"]
        level_desc = level_info["desc"]
        
        st.markdown("---")
        
        # 技能选择
        skill_labels = [s["name"] for s in SKILLS]
        selected_skill_name = st.radio("选择技能", skill_labels, index=None)
        
        if selected_skill_name:
            skill_obj = next(s for s in SKILLS if s["name"] == selected_skill_name)
            if st.session_state.current_skill is None or st.session_state.current_skill["id"] != skill_obj["id"]:
                st.session_state.current_skill = skill_obj
                st.session_state.skill_round = 1
                st.session_state.chat_history = []
                st.rerun()
    
    # 数据加载
    if not st.session_state.data_loaded:
        with st.spinner(f"加载 {st.session_state.current_symbol} 数据..."):
            df, main_code = load_symbol_data(st.session_state.current_symbol)
            if df is not None and len(df) > 0:
                st.session_state.kline_data = df
                st.session_state.main_contract = main_code
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.error(f"无法加载 {st.session_state.current_symbol} 数据")
                st.info("请检查网络或稍后重试")
                return
    
    # 主界面
    if st.session_state.current_skill:
        skill = st.session_state.current_skill
        round_label = "第1轮" if st.session_state.skill_round == 1 else "第2轮"
        st.caption(f"品种: {st.session_state.current_symbol} | 技能: {skill['name']} | {level_name} | {round_label}")
    
    # 图表
    if st.session_state.kline_data is not None:
        fig = plot_kline(st.session_state.kline_data, n_bars)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 对话区
    if st.session_state.current_skill is None:
        st.info("👈 从左侧选择品种、阶段和技能开始训练")
        return
    
    skill = st.session_state.current_skill
    is_round2 = st.session_state.skill_round == 2
    
    # 显示历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 自动发送引导
    if st.session_state.skill_round == 1 and len(st.session_state.chat_history) == 0:
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_desc = market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill["name"], skill["question"], market_desc,
                    level_name, level_desc, is_second_round=False
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
    # 用户输入
    prompt = "你的观察？" if not is_round2 else "你的回答？(第2轮)"
    user_input = st.chat_input(prompt)
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_desc = market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill["name"], skill["question"], market_desc,
                    level_name, level_desc, user_input, is_round2
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        
        if is_round2:
            st.session_state.skill_round = 1
            st.success("✅ 本轮完成，可切换其他技能")
        else:
            st.session_state.skill_round = 2


if __name__ == "__main__":
    main()
