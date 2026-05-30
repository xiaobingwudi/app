"""
Al Brooks 结构训练器 V19
================================
核心优化：
  1. 修复品种代码格式（CU0 → CU）
  2. 修复重复品种L0冲突
  3. 增加数据缓存和错误处理
  4. 优化AI Prompt，增加两轮制控制
  5. 修复技能切换时轮次重置
  6. 增加训练总结功能
"""

import json
import time
import random
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="Al Brooks 结构训练器", layout="wide")

# ── 品种分类定义（修复代码格式）─────────────────────────
PRODUCT_CATEGORIES = {
    "金融": ["IF", "IH", "IC", "IM", "TS", "TF", "T"],
    "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
    "黑色": ["RB", "HC", "I", "JM", "J"],
    "化工": ["V", "PP", "TA", "MA", "RU", "BU", "FU", "EG", "EB", "PG", "SA", "UR"],
    "农产品": ["M", "Y", "P", "A", "B", "C", "CS", "JD", "AP", "CF", "SR", "OI", "RM"],
    "能源": ["SC", "LU", "NR"],
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

# ── AI配置（从Streamlit Cloud Secrets读取） ────────────
try:
    AI_CONFIG = {
        "base_url": st.secrets["ai"]["base_url"],
        "api_key": st.secrets["ai"]["api_key"],
        "model": st.secrets["ai"]["model"],
    }
except Exception:
    # 兼容旧配置方式
    AI_CONFIG = {
        "base_url": st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": st.secrets.get("OPENAI_API_KEY", ""),
        "model": st.secrets.get("OPENAI_MODEL", "deepseek-chat"),
    }

# ── 训练阶段配置 ─────────────────────────────────────────
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

# ── AI Prompt 模板 ──────────────────────────────────────
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为交易教练。

【核心职责】
1. 训练师：根据训练阶段和当前技能目的，提供结构化的分析指导
2. 点评师：对用户的分析给出专业点评，指出对错与遗漏

当前训练阶段：{level_name} - {level_desc}
当前技能：{skill_name}
核心提问：{skill_question}

【5个技能的核心分析维度】
1. 背景阅读 → 趋势方向、震荡区间、关键支撑阻力、高低点序列(HH/HL或LH/LL)
2. 控制权识别 → 趋势线的角度和持续性、突破K线的力度（实体大小/影线长度）、连续同向K线的数量
3. 推进质量 → 推进波的幅度、K线重叠程度、影线长度、动能是否衰减
4. 回调vs转换 → 回调的时间/幅度特征、转换的确认信号、对手方是否连续出现
5. 市场接受 → 价格对新区域的停留时间、重叠K线的数量、测试关键价位后的反应

【对话流程 - 严格执行】
- 第1轮：围绕技能核心提问进行引导，描述当前市场结构，提出有针对性的问题。不要直接给答案。
- 第2轮：基于用户回答给出结构化点评：1)肯定正确的部分 2)指出遗漏或偏差 3)给出清晰的判断结论

【回答风格】
- 简洁、专业、直击要点，不超过150字
- 使用具体的K线编号作为依据
- 基于实际结构分析，不泛泛而谈"""


def _market_msg(kline_df: pd.DataFrame, n_bars: int = 40) -> str:
    """将K线数据转化为自然语言的市场描述（增强版）"""
    if kline_df is None or kline_df.empty:
        return "暂无数据"

    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return "数据不足"

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    lines = []
    lines.append(f"【当前K线】第{n}号K线")
    lines.append("")
    lines.append("【最近行情描述】")

    # 最近10根K线详细描述
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

        # 与前一根对比
        change_desc = ""
        if i > start:
            price_change = c[i] - c[i-1]
            if abs(price_change) > total_range * 0.8 and total_range > 0:
                change_desc = f"，相比前一根{'大涨' if price_change > 0 else '大跌'}了{abs(price_change):.1f}"
            elif price_change > 0:
                change_desc = f"，比前一根涨了{price_change:.1f}"
            elif price_change < 0:
                change_desc = f"，比前一根跌了{abs(price_change):.1f}"

        # 影线描述
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]
        wick_parts = []
        if body > 0:
            if upper_wick > body * 2:
                wick_parts.append("上影线很长")
            if lower_wick > body * 2:
                wick_parts.append("下影线很长")
        wick_text = f"，{','.join(wick_parts)}" if wick_parts else ""

        lines.append(f"  K{i+1}: {k_type}，开{o[i]:.0f} 收{c[i]:.0f} 高{h[i]:.0f} 低{l[i]:.0f}{wick_text}{change_desc}")

    lines.append("")
    lines.append("【整体市场感知】")

    if n >= 10:
        last_10 = df.tail(10)
        yang_count = sum(1 for j in range(len(last_10)) if last_10.iloc[j]["Close"] >= last_10.iloc[j]["Open"])
        yin_count = 10 - yang_count

        if yang_count >= 7:
            bias = "近期明显偏多，阳线占主导"
        elif yin_count >= 7:
            bias = "近期明显偏空，阴线占主导"
        elif yang_count >= 6:
            bias = "近期略偏多"
        elif yin_count >= 6:
            bias = "近期略偏空"
        else:
            bias = "近期多空平衡"
        lines.append(f"  • {bias}（最近10根中{yang_count}阳{yin_count}阴）")

        total_change = c[-1] - c[0]
        if total_change > 0:
            lines.append(f"  • 整体向上，累计上涨{total_change:.1f}")
        elif total_change < 0:
            lines.append(f"  • 整体向下，累计下跌{abs(total_change):.1f}")

        # 连续方向检测
        cons_up, cons_dn = 0, 0
        for j in range(n-1, max(0, n-11), -1):
            if c[j] > c[j-1]:
                cons_up += 1
                cons_dn = 0
            else:
                cons_dn += 1
                cons_up = 0
            if cons_up >= 3 or cons_dn >= 3:
                break
        if cons_up >= 3:
            lines.append(f"  • 连续{cons_up}根上涨，多头推进中")
        elif cons_dn >= 3:
            lines.append(f"  • 连续{cons_dn}根下跌，空头推进中")

    return "\n".join(lines)


def ask_coach(
    skill_name: str,
    skill_question: str,
    market_msg: str,
    level_name: str,
    level_desc: str,
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"当前市场状况：\n{market_msg}"},
    ]

    # 添加历史对话
    if "chat_history" in st.session_state:
        for m in st.session_state.chat_history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})

    if is_second_round:
        messages.append({
            "role": "user",
            "content": f"【第2轮】用户对上一轮的回答：{user_input}\n\n请按以下结构给出点评：\n1. 肯定正确的部分\n2. 指出遗漏或偏差\n3. 给出清晰的判断结论"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"【第1轮】当前技能：「{skill_name}」，核心提问：「{skill_question}」。\n请描述当前市场结构，并提出引导性问题帮助用户思考。"
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
        return f"[AI调用失败] {str(e)}"


def ask_summary(dialogue, skill_name: str) -> str:
    """生成训练总结"""
    from openai import OpenAI

    client = OpenAI(
        base_url=AI_CONFIG["base_url"],
        api_key=AI_CONFIG["api_key"],
    )

    summary_prompt = f"""你是训练总结分析师。根据以下关于「{skill_name}」的训练对话，分析用户的阅读习惯。

【对话记录】
{dialogue}

请输出JSON格式：
{{
    "observations": ["用户的阅读习惯和特点"],
    "strong_areas": ["用户表现好的方面"],
    "weak_areas": ["用户需要加强的方面"],
    "next_focus": ["下一阶段训练建议"]
}}

要求：具体引用训练中的实际表现，不要笼统评价。"""

    try:
        resp = client.chat.completions.create(
            model=AI_CONFIG["model"],
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        # 清理markdown标记
        content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'


# ── 数据获取（缓存） ──────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kline_data(symbol: str, period: str = "30"):
    """获取K线数据"""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "date": "datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_main_contract(symbol: str):
    """获取主力合约代码"""
    try:
        main = ak.match_main_contract(symbol=symbol)
        return main
    except Exception:
        return symbol


# ── 图表绘制 ────────────────────────────────────────────
def plot_kline(kline_df: pd.DataFrame, n_bars: int = 40):
    """绘制K线图"""
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
            x=df["datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )

    colors = ["#ef5350" if row["Close"] >= row["Open"] else "#26a69a" for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df["datetime"], y=df["Volume"], name="成交量", marker_color=colors, opacity=0.5),
        row=2, col=1,
    )

    # K线编号（每5根标记）
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 5 == 0:
            fig.add_annotation(
                x=row["datetime"],
                y=row["High"],
                text=str(len(df) - idx),
                showarrow=False,
                yshift=8,
                font_size=8,
                font_color="#666",
                row=1, col=1,
            )

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
defaults = {
    "current_skill": None,
    "skill_round": 1,
    "chat_history": [],
    "last_skill_id": None,
    "data_loaded": False,
    "kline_data": None,
    "current_symbol": "RB",
    "train_level": "level1",
    "training_summary": None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 品种选择")

    current_symbol = st.session_state.current_symbol

    for cat_name, symbols in PRODUCT_CATEGORIES.items():
        expanded = cat_name in DEFAULT_EXPANDED
        with st.expander(cat_name, expanded=expanded):
            cols = st.columns(min(4, len(symbols)))
            for i, sym in enumerate(symbols):
                col = cols[i % len(cols)]
                if col.button(sym, key=f"sym_{cat_name}_{sym}", type="primary" if sym == current_symbol else "secondary", use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.session_state.data_loaded = False
                    st.session_state.kline_data = None
                    st.session_state.chat_history = []
                    st.session_state.skill_round = 1
                    st.session_state.training_summary = None
                    st.rerun()

    st.markdown("---")

    # 训练阶段
    st.markdown("**训练阶段**")
    level_options = list(LEVEL_CONFIG.keys())
    level_labels = [f"阶段{i+1}: {LEVEL_CONFIG[l]['name']}" for i, l in enumerate(level_options)]
    selected_level_label = st.selectbox("", level_labels, label_visibility="collapsed")
    train_level = level_options[level_labels.index(selected_level_label)]
    st.session_state.train_level = train_level

    level_info = LEVEL_CONFIG[train_level]
    n_bars = level_info["n_bars"]
    level_name = level_info["name"]
    level_desc = level_info["desc"]

    st.markdown("---")

    # 技能选择
    st.markdown("**选择技能**")
    skill_labels = [f"{s['name']}" for s in SKILLS]
    selected_skill_name = st.radio(
        "",
        skill_labels,
        index=None,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # 训练总结按钮
    if len(st.session_state.chat_history) > 2:
        if st.button("📊 生成训练总结", use_container_width=True):
            with st.spinner("生成总结中..."):
                dialogue_text = "\n".join([
                    f"{'用户' if m['role']=='user' else '教练'}: {m['content']}"
                    for m in st.session_state.chat_history[-20:]
                ])
                skill_name = st.session_state.current_skill["name"] if st.session_state.current_skill else "价格行为"
                st.session_state.training_summary = ask_summary(dialogue_text, skill_name)
                st.rerun()

    # 显示总结
    if st.session_state.training_summary:
        with st.expander("📋 训练总结", expanded=True):
            try:
                summary = json.loads(st.session_state.training_summary)
                st.markdown("**观察习惯**")
                for o in summary.get("observations", []):
                    st.markdown(f"- {o}")
                st.markdown("**优势**")
                for s in summary.get("strong_areas", []):
                    st.markdown(f"- ✅ {s}")
                st.markdown("**待加强**")
                for w in summary.get("weak_areas", []):
                    st.markdown(f"- ⚠️ {w}")
                st.markdown("**下一步建议**")
                for n in summary.get("next_focus", []):
                    st.markdown(f"- 🎯 {n}")
            except:
                st.text(st.session_state.training_summary)

    st.caption(f"数据: {len(st.session_state.kline_data) if st.session_state.kline_data is not None else 0} 根K线")


# ═══════════════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════════════

# 技能切换处理
if selected_skill_name is not None:
    skill_obj = next(s for s in SKILLS if s["name"] == selected_skill_name)
    if st.session_state.current_skill is None or st.session_state.current_skill["id"] != skill_obj["id"]:
        st.session_state.current_skill = skill_obj
        st.session_state.skill_round = 1
        st.session_state.chat_history = []
        st.session_state.training_summary = None
        st.rerun()

# 状态栏
if st.session_state.current_skill:
    skill = st.session_state.current_skill
    round_label = "第1轮(引导)" if st.session_state.skill_round == 1 else "第2轮(点评)"
    st.caption(
        f"📊 {st.session_state.current_symbol} | "
        f"🎯 {skill['name']} — {skill['question']} | "
        f"📖 {level_name} | {round_label}"
    )

# 数据加载
if not st.session_state.data_loaded:
    with st.spinner(f"加载 {st.session_state.current_symbol} 数据..."):
        df = fetch_kline_data(st.session_state.current_symbol, period="30")
        if df is not None and len(df) > 0:
            st.session_state.kline_data = df
            st.session_state.data_loaded = True
        else:
            st.error(f"数据加载失败: {st.session_state.current_symbol}")

# 图表区
if st.session_state.data_loaded and st.session_state.kline_data is not None:
    fig = plot_kline(st.session_state.kline_data, n_bars)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
else:
    st.info("请从左侧选择品种开始训练")

st.markdown("---")

# 对话区
if st.session_state.current_skill is None:
    st.info("👈 从左侧选择品种、阶段和技能开始训练")
else:
    skill = st.session_state.current_skill

    # 显示历史对话
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 第1轮自动发送引导
    is_round2 = st.session_state.skill_round == 2
    has_guide = any(
        "第1轮" in m.get("content", "") for m in st.session_state.chat_history
    )

    if not is_round2 and not has_guide and st.session_state.data_loaded:
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    level_name=level_name,
                    level_desc=level_desc,
                    is_second_round=False,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 用户输入
    prompt_text = "你的观察是？" if not is_round2 else "你的回答是？(第2轮)"
    user_input = st.chat_input(prompt_text)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    level_name=level_name,
                    level_desc=level_desc,
                    user_input=user_input,
                    is_second_round=is_round2,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # 第2轮结束后，下一轮回到第1轮（等待切换技能）
        if is_round2:
            st.session_state.skill_round = 1
            st.info("✅ 本轮训练完成，可切换其他技能继续训练")

# ── 紧凑样式 ────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp { margin: 0; padding: 0; }
    .block-container { padding: 0.5rem 1.5rem 0.5rem 1.5rem !important; max-width: 100%; }
    section[data-testid="stSidebar"] > div { padding: 0.5rem !important; }
    hr { margin: 6px 0 !important; }
    .stPlotlyChart { margin: 0 !important; }
    button[kind="primary"] { background-color: #ef5350 !important; }
</style>
""",
    unsafe_allow_html=True,
)
