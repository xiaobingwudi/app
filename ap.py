"""
Al Brooks 结构训练器 V24
修复：
1. 品种按钮key冲突（L0同时出现在化工和能源）→ key加品类前缀
2. API Key改用st.secrets管理，支持deepseek/其他API
3. 侧栏三块内容：品种选择 + 训练阶段 + 技能选择
"""
import json, time, random
from datetime import datetime, date
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="Al Brooks 结构训练器", layout="wide")

# ── 品种分类定义 ─────────────────────────────────────────
PRODUCT_CATEGORIES = {
    "金融": ["IF", "IH", "IC", "IM", "TS", "TF"],
    "有色": ["CU0", "AL0", "ZN0", "PB0", "NI0", "SN0"],
    "黑色": ["RB0", "HC0", "I0", "JM0", "J0"],
    "化工": ["V0", "PP0", "L0", "TA0", "MA0", "RU0", "BU0", "FU0", "EG0"],
    "农产品": ["M0", "Y0", "P0", "A0", "B0", "C0", "CS0", "JD0", "AP0", "CF0"],
    "能源": ["SC0", "L0"],
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
# 优先用st.secrets，否则fallback到硬编码（仅开发用）
try:
    AI_CONFIG = {
        "base_url": st.secrets.get("ai", {}).get("base_url", "https://www.right.codes/codex/v1"),
        "api_key": st.secrets.get("ai", {}).get("api_key", "sk-KIhnn3eQ0A8mR1eI0a8fC7bBe3d3FfD1BfD3FfD1BfD3FfD1BfD1BfD1BfD1"),
        "model": st.secrets.get("ai", {}).get("model", "gpt-5.5"),
    }
except Exception:
    AI_CONFIG = {
        "base_url": "https://www.right.codes/codex/v1",
        "api_key": "sk-KIhnn3eQ0A8mR1eI0a8fC7bBe3d3FfD1BfD3FfD1BfD3FfD1BfD1BfD1BfD1",
        "model": "gpt-5.5",
    }

# ── 侧栏 ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**品种选择**")

    current_symbol = st.session_state.get("current_symbol", "RB0")
    for cat_name, symbols in PRODUCT_CATEGORIES.items():
        expanded = cat_name in DEFAULT_EXPANDED
        with st.expander(cat_name, expanded=expanded):
            cols = st.columns(min(4, len(symbols)))
            for i, sym in enumerate(symbols):
                col = cols[i % len(cols)]
                btn_style = "primary" if sym == current_symbol else "secondary"
                # 用品类+品种作为唯一key，避免L0在化工和能源中重复
                if col.button(sym, key=f"sym_{cat_name}_{sym}", type=btn_style, use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.session_state.data_loaded = False
                    st.session_state.kline_data = None
                    st.session_state.structural_features = {}
                    st.rerun()

    st.markdown("---")

    # 训练阶段
    st.markdown("**训练阶段**")
    TRAIN_LEVEL_OPTIONS = {
        "阶段1: 观察阶段": "level1",
        "阶段2: 行为细化阶段": "level2",
        "阶段3: 结构验证阶段": "level3",
    }
    selected_level_label = st.selectbox(
        "", list(TRAIN_LEVEL_OPTIONS.keys()), label_visibility="collapsed"
    )
    train_level = TRAIN_LEVEL_OPTIONS[selected_level_label]

    LEVEL_CONFIG = {
        "level1": {
            "name": "观察阶段",
            "desc": "识别市场结构特征（趋势/震荡/通道/双重顶底）",
            "n_bars": 40,
        },
        "level2": {
            "name": "行为细化阶段",
            "desc": "分析K线行为细节（影线/实体/嵌套/突破）",
            "n_bars": 30,
        },
        "level3": {
            "name": "结构验证阶段",
            "desc": "验证结构预期与多时间框架一致性",
            "n_bars": 60,
        },
    }
    level_info = LEVEL_CONFIG[train_level]
    n_bars = level_info["n_bars"]
    level_name = level_info["name"]
    level_desc = level_info["desc"]

    st.markdown("---")

    # ── 技能选择（侧栏，垂直 radio） ──
    st.markdown("**选择技能目的**")
    skill_labels = [f"{s['name']}" for s in SKILLS]
    selected_skill_name = st.radio(
        "",
        skill_labels,
        index=None,
        label_visibility="collapsed",
    )

    st.markdown("---")
    data_display = st.empty()

# ── AI Prompt 模板 ──────────────────────────────────────
AI_SYSTEM_PROMPT_TEMPLATE = """你是一个Al Brooks价格行为交易教练，当前训练阶段为「{level_name}」：{level_desc}

## 核心职责
1. **训练师**：根据训练阶段和当前技能目的，提供结构化的分析指导
2. **点评师**：对用户的分析给出专业点评，指出对错与遗漏

## 对话流程（两轮制）
- **第1轮（技能引导）**：围绕技能「{skill_name}」的核心提问「{skill_question}」进行引导，先描述当前市场发生了什么，再提出有针对性的问题。不要直接给答案，用追问促使用户自己发现结构特征。
- **第2轮（点评反馈）**：基于用户第1轮的回答，给出结构化的点评，明确指出正确和需要改进的地方，最后给出清晰的判断结论。

## 5个技能的核心分析维度
1. 背景阅读 → 趋势方向、震荡区间、关键支撑阻力、近期价格行为模式
2. 控制权识别 → 趋势线的角度和持续性、突破K线的力度（实体大小/影线长度）、连续同向K线的数量
3. 推进质量 → 推进波的幅度（点数/ATR比例）、推进波的斜率（每单位时间移动距离）、回调深度（浅回调<38.2% vs 深回调>61.8%）
4. 回调vs转换 → 回调的时间/幅度特征、转换的确认信号（突破结构/趋势线/均线）、微观结构的破坏
5. 市场接受 → 价格对新区域的停留时间、重叠K线的数量、测试关键价位后的反应

## 回答风格
- 简洁、专业、直击要点
- 使用具体的价格行为术语
- 基于实际K线结构分析，不泛泛而谈"""


def _market_msg(kline_df: pd.DataFrame) -> str:
    """将K线数据转化为自然语言的市场描述"""
    if kline_df is None or kline_df.empty:
        return "暂无数据"

    recent = kline_df.tail(10)
    o, h, l, c = recent["Open"], recent["High"], recent["Low"], recent["Close"]

    direction = "上涨" if c.iloc[-1] > o.iloc[-1] else "下跌" if c.iloc[-1] < o.iloc[-1] else "平收"
    body = abs(c.iloc[-1] - o.iloc[-1])
    upper = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
    lower = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
    range_val = h.iloc[-1] - l.iloc[-1]

    total_range = (h.max() - l.min()) / (l.min() or 1) * 100
    atr = (h - l).rolling(5).mean().iloc[-1]

    lines = [
        f"最新K线: {direction}，实体{body:.1f}，上影线{upper:.1f}，下影线{lower:.1f}，振幅{range_val:.1f}",
        f"近10根K线范围: {total_range:.2f}%，5期ATR: {atr:.1f}",
    ]

    ma5 = c.rolling(5).mean()
    if all(c.iloc[-i] > ma5.iloc[-i] for i in range(1, 4)):
        lines.append("短期趋势: 多头排列")
    elif all(c.iloc[-i] < ma5.iloc[-i] for i in range(1, 4)):
        lines.append("短期趋势: 空头排列")
    else:
        lines.append("短期趋势: 震荡")

    cons_up = 0
    cons_dn = 0
    for i in range(len(c) - 1, 0, -1):
        if c.iloc[i] > c.iloc[i - 1]:
            cons_up += 1
            cons_dn = 0
        else:
            cons_dn += 1
            cons_up = 0
    if cons_up >= 3:
        lines.append(f"连续{cons_up}根上涨，多头推进中")
    elif cons_dn >= 3:
        lines.append(f"连续{cons_dn}根下跌，空头推进中")

    return "\n".join(lines)


def ask_coach(
    skill_name: str,
    skill_question: str,
    market_msg: str,
    user_input: str = "",
    is_second_round: bool = False,
) -> str:
    """调用AI教练"""
    from openai import OpenAI

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

    recent_msgs = []
    if "chat_history" in st.session_state:
        for m in st.session_state.chat_history[-10:]:
            recent_msgs.append({"role": m["role"], "content": m["content"]})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"当前市场状况：\n{market_msg}"},
    ]
    messages.extend(recent_msgs)

    if is_second_round:
        messages.append({
            "role": "user",
            "content": f"【第2轮】用户对上一轮引导的回应：{user_input}\n\n请按以下结构给出点评：\n1. 肯定正确的部分\n2. 指出遗漏或偏差\n3. 给出清晰的判断结论（趋势方向/控制权归属/结构类型）",
        })
    else:
        messages.append({
            "role": "user",
            "content": f"【第1轮】当前技能目的：「{skill_name}」，核心提问：「{skill_question}」。\n请描述当前市场结构，并提出引导性问题促使我思考。",
        })

    try:
        resp = client.chat.completions.create(
            model=AI_CONFIG["model"],
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[AI调用失败] {str(e)}"


# ── 数据获取（缓存） ──────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_all_contracts(symbol: str):
    """获取全合约数据并找出主力"""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period="60")
        if df is None or df.empty:
            return None, None
        df["date"] = pd.to_datetime(df["date"])
        main_code = ak.match_main_contract(symbol=symbol)
        return df, main_code
    except Exception:
        return None, None


def load_data(symbol: str = "RB0"):
    df, main_code = _fetch_all_contracts(symbol)
    if df is None:
        return None, None
    if main_code and main_code in df["symbol"].values:
        df_main = df[df["symbol"] == main_code].copy()
    else:
        df_main = df.copy()
    df_main.sort_values("date", inplace=True)
    df_main.reset_index(drop=True, inplace=True)
    return df_main, main_code


# ── 结构特征预计算 ──────────────────────────────────────
def calc_structural_features(kline_df: pd.DataFrame) -> dict:
    """预计算K线结构特征"""
    if kline_df is None or len(kline_df) < 10:
        return {}
    o = kline_df["Open"].values
    h = kline_df["High"].values
    l = kline_df["Low"].values
    c = kline_df["Close"].values
    n = len(kline_df)
    features = {}

    ma20 = pd.Series(c).rolling(20).mean().values
    slope = (ma20[-1] - ma20[-5]) / 5 if not np.isnan(ma20[-1]) and not np.isnan(ma20[-5]) else 0
    features["trend"] = "up" if slope > 0 else "down" if slope < 0 else "flat"

    features["volatility"] = float(np.std((h - l) / (l + 1e-10)))

    lookback = min(20, n)
    up_count = sum(1 for i in range(n - lookback, n) if c[i] > c[i - 1])
    features["up_ratio"] = up_count / lookback

    recent_high = max(h[n - 21 : n - 1]) if n >= 21 else max(h[: n - 1])
    recent_low = min(l[n - 21 : n - 1]) if n >= 21 else min(l[: n - 1])
    features["breakout_up"] = bool(c[-1] > recent_high and h[-1] > recent_high)
    features["breakout_dn"] = bool(c[-1] < recent_low and l[-1] < recent_low)

    if n >= 10:
        seg_high = max(h[-10:])
        seg_low = min(l[-10:])
        seg_range = seg_high - seg_low
        retrace = abs(c[-1] - seg_low) / seg_range if seg_range > 0 else 0.5
        features["retrace_depth"] = float(retrace)

    return features


# ── 图表绘制 ────────────────────────────────────────────
def plot_kline(kline_df: pd.DataFrame, features: dict):
    """绘制K线图+结构标注"""
    if kline_df is None or kline_df.empty:
        return go.Figure()

    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )

    colors = ["#ef5350" if row["Close"] >= row["Open"] else "#26a69a" for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df["date"], y=df["Volume"], name="成交量", marker_color=colors),
        row=2, col=1,
    )

    for i, (_, row) in enumerate(df.iterrows()):
        label = str(len(df) - i)
        fig.add_annotation(
            x=row["date"], y=row["High"],
            text=label, showarrow=False,
            yshift=8, font_size=8, font_color="#666",
            row=1, col=1,
        )

    if features:
        last_c = df["Close"].iloc[-1]
        if features.get("breakout_up"):
            fig.add_hline(y=last_c, line_color="red", line_dash="dot", opacity=0.5, row=1, col=1)
            fig.add_annotation(x=df["date"].iloc[-1], y=last_c, text="突破↑", showarrow=False, yshift=10, row=1, col=1)
        if features.get("breakout_dn"):
            fig.add_hline(y=last_c, line_color="green", line_dash="dot", opacity=0.5, row=1, col=1)
            fig.add_annotation(x=df["date"].iloc[-1], y=last_c, text="突破↓", showarrow=False, yshift=-10, row=1, col=1)

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(matches="x", row=2, col=1)
    return fig


# ── Session State 初始化 ──────────────────────────────
for key, default in [
    ("current_skill", None),
    ("skill_round", 1),
    ("chat_history", []),
    ("last_skill_id", None),
    ("data_loaded", False),
    ("kline_data", None),
    ("main_contract", None),
    ("structural_features", {}),
    ("prev_skill_name", None),
    ("current_symbol", "RB0"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════════════

# ── 技能选择处理 ────────────────────────────────────
if selected_skill_name is not None and selected_skill_name != st.session_state.prev_skill_name:
    skill_obj = next(s for s in SKILLS if s["name"] == selected_skill_name)

    if st.session_state.prev_skill_name == selected_skill_name:
        st.session_state.skill_round = 2
    else:
        st.session_state.current_skill = skill_obj
        st.session_state.last_skill_id = skill_obj["id"]
        st.session_state.skill_round = 1
        st.session_state.chat_history = []

    st.session_state.prev_skill_name = selected_skill_name

# 状态栏
if st.session_state.current_skill:
    skill = st.session_state.current_skill
    round_label = "第1轮(引导)" if st.session_state.skill_round == 1 else "第2轮(点评)"
    st.caption(
        f"品种: {st.session_state.current_symbol} | "
        f"技能: {skill['name']} — {skill['question']} | "
        f"阶段: {level_name} | {round_label}"
    )

# ── 数据加载 ────────────────────────────────────────
if not st.session_state.data_loaded:
    with st.spinner(f"加载 {st.session_state.current_symbol} 数据..."):
        df, main_code = load_data(st.session_state.current_symbol)
        if df is not None:
            st.session_state.kline_data = df
            st.session_state.main_contract = main_code
            st.session_state.structural_features = calc_structural_features(df)
            st.session_state.data_loaded = True

# ── 图表区 ──────────────────────────────────────────
if st.session_state.data_loaded and st.session_state.kline_data is not None:
    fig = plot_kline(st.session_state.kline_data, st.session_state.structural_features)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
else:
    st.info("数据加载失败，请检查网络或合约代码")

# 更新侧栏数据信息
if st.session_state.data_loaded and st.session_state.kline_data is not None:
    data_display.caption(
        f"合约: {st.session_state.main_contract or st.session_state.current_symbol} | "
        f"数据: {len(st.session_state.kline_data)} 根K线 | Bar: {n_bars}"
    )

st.markdown("---")

# ── 对话区 ──────────────────────────────────────────
if st.session_state.current_skill is None:
    st.info("👈 从左侧侧栏选择品种、阶段和技能目的开始训练")
else:
    skill = st.session_state.current_skill
    is_round2 = st.session_state.skill_round == 2

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    has_guide = any(
        m["role"] == "assistant" and "当前技能目的" in m.get("content", "")
        for m in st.session_state.chat_history[-5:]
    )
    if not is_round2 and not has_guide:
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    is_second_round=False,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    prompt = "你的观察是？" if not is_round2 else "你的回答是？(第2轮)"
    user_input = st.chat_input(prompt)
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    user_input=user_input,
                    is_second_round=is_round2,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if is_round2:
            st.session_state.skill_round = 1

# ── 紧凑样式 ────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp { margin: 0; padding: 0; }
    .block-container { padding: 0.8rem 2rem 0.8rem 2rem !important; max-width: 100%; }
    section[data-testid="stSidebar"] > div { padding: 0.5rem !important; }
    section[data-testid="stSidebar"] .block-container { padding: 0.5rem !important; }
    hr { margin: 6px 0 !important; }
    .stPlotlyChart { margin: 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)
