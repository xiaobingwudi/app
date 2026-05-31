"""
Al Brooks 价格行为读盘训练器 V19
基于 V18_fixed 修改，仅改动：
1. 数据加载：{exchange}{code} → {CODE}0 格式
2. 交互：chart_df固定 + 每技能最多2轮 + 图表只在换图时移动
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from datetime import datetime, timedelta
import requests
import json
import re
import random
import os

st.set_page_config(page_title="Al Brooks 读盘训练器", layout="wide", initial_sidebar_state="expanded")

# ========== 品种配置 ==========
CONTRACTS = {
    "螺纹钢": "RB0", "铁矿石": "I0", "豆粕": "M0", "豆油": "Y0",
    "棕榈油": "P0", "焦煤": "JM0", "焦炭": "J0", "甲醇": "MA0",
    "PTA": "TA0", "纯碱": "SA0", "玻璃": "FG0", "棉花": "CF0",
    "白糖": "SR0", "菜油": "OI0", "菜粕": "RM0", "沪铜": "CU0",
    "沪铝": "AL0", "沪锌": "ZN0", "沪镍": "NI0", "沪金": "AU0",
    "沪银": "AG0", "原油": "SC0", "燃油": "FU0", "沥青": "BU0",
    "橡胶": "RU0", "塑料": "L0", "PVC": "V0", "PP": "PP0",
    "乙二醇": "EG0", "苯乙烯": "EB0", "尿素": "UR0", "硅铁": "SF0",
    "锰硅": "SM0", "热卷": "HC0", "纸浆": "SP0", "红枣": "CJ0",
    "苹果": "AP0", "花生": "PK0", "生猪": "LH0", "鸡蛋": "JD0",
    "玉米": "C0", "淀粉": "CS0", "豆二": "B0", "碳酸锂": "LC0",
    "工业硅": "SI0"
}
PERIODS = {"1分钟": "1", "5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
PERIOD_LABELS = {"1": "1分钟", "5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟"}
BARS_TO_SHOW = 30
MAX_ROUNDS_PER_SKILL = 2

SKILLS = ["背景阅读", "控制权识别", "推进质量判断", "回调vs转换", "市场接受度判断"]
SKILL_PROMPTS = {
    "背景阅读": "请分析当前K线图的整体背景：趋势方向（上涨/下跌/震荡）、关键支撑阻力位、主要摆动高低点、均线排列状态。",
    "控制权识别": "请分析当前市场的控制权状态：多头控制/空头控制/双向交易/震荡无方向，依据是什么？",
    "推进质量判断": "请评估最近几根K线的推进质量：K线实体大小、影线长度、成交量配合情况、是否出现连续推进或衰竭信号。",
    "回调vs转换": "请判断最近的回调是正常的趋势回调还是趋势转换信号，依据是什么？",
    "市场接受度判断": "请分析市场对当前价格的接受程度：是否在某个价位出现拒绝、是否形成双底/双顶、是否有信号K线。"
}

# ========== 数据加载 ==========
@st.cache_data(ttl=3600)
def load_data(product_name, period="30"):
    try:
        symbol = CONTRACTS[product_name]
        raw = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if raw is None or raw.empty:
            return None
        df = raw.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            return None
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None

def random_bar(df, n_bars=30):
    if df is None or len(df) < n_bars:
        return 0, 0
    max_start = len(df) - n_bars
    start = random.randint(0, max_start)
    return start, start + n_bars

def get_ai_comment(skill_name, kline_data, user_input=None):
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY", st.secrets.get("DEEPSEEK_API_KEY", ""))
        if not api_key:
            return "请设置 DEEPSEEK_API_KEY 环境变量"
        df_sample = kline_data.tail(20)
        kline_summary = []
        for _, row in df_sample.iterrows():
            kline_summary.append(
                f"{row['datetime'].strftime('%m-%d %H:%M')} "
                f"O:{row['open']:.1f} H:{row['high']:.1f} "
                f"L:{row['low']:.1f} C:{row['close']:.1f} "
                f"V:{int(row['volume'])}"
            )
        kline_text = "\n".join(kline_summary)
        prompt = f"""你是一个Al Brooks价格行为交易教练。学员正在训练技能：{skill_name}。

【技能训练目标】
{SKILL_PROMPTS[skill_name]}

【最近20根K线数据】
{kline_text}

"""
        if user_input:
            prompt += f"""【学员的观察/回答】
{user_input}

请对学员的回答进行点评：
1. 指出学员回答中的正确之处
2. 指出学员遗漏的关键信息
3. 给出Al Brooks价格行为角度的专业分析
4. 给出1-2个具体的改进建议
"""
        else:
            prompt += """请从Al Brooks价格行为角度分析当前市场状态，给出你的专业观察。"""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        resp = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"API请求失败: {resp.status_code}"
    except Exception as e:
        return f"AI点评出错: {str(e)}"

def plot_candlestick(df_segment, width=900, height=500):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2], shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=list(range(len(df_segment))),
        open=df_segment["open"], high=df_segment["high"],
        low=df_segment["low"], close=df_segment["close"], name="K线",
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(range(len(df_segment))), y=df_segment["volume"], name="成交量",
        marker_color='#90a4ae', opacity=0.7
    ), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"{st.session_state.product} {PERIOD_LABELS.get(st.session_state.period, '30分钟')}", font=dict(size=14)),
        template="plotly_white", height=height, width=width,
        margin=dict(l=40, r=20, t=40, b=20), showlegend=False, hovermode="x unified"
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(title_text="K线序号", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig

# ========== 初始化 ==========
def init_session():
    defaults = {
        "chart_df": None, "display_start": 0, "display_end": 30,
        "product": "螺纹钢", "period": "30",
        "current_skill": None,
        "skill_rounds": {s: 0 for s in SKILLS},
        "skill_responses": {},
        "total_bars": 0, "data_loaded": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { width: 280px !important; }
        section[data-testid="stSidebar"] .stButton button { height: 28px; font-size: 12px; padding: 0 10px; }
        div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("### 设置")
    sel_product = st.selectbox("品种", list(CONTRACTS.keys()), index=0)
    sel_period = st.selectbox("周期", list(PERIODS.keys()), index=3)

    if st.button("加载/重载数据", use_container_width=True):
        period_val = PERIODS[sel_period]
        with st.spinner(f"加载 {sel_product} {sel_period} 数据..."):
            df = load_data(sel_product, period_val)
            if df is not None and len(df) > BARS_TO_SHOW:
                start, end = random_bar(df, BARS_TO_SHOW)
                st.session_state.chart_df = df
                st.session_state.display_start = start
                st.session_state.display_end = end
                st.session_state.total_bars = len(df)
                st.session_state.product = sel_product
                st.session_state.period = period_val
                st.session_state.data_loaded = True
                st.session_state.skill_rounds = {s: 0 for s in SKILLS}
                st.session_state.skill_responses = {}
                st.session_state.current_skill = None
                st.rerun()
            else:
                st.error("数据加载失败，请尝试其他品种或周期")

    st.markdown("### 品种")
    categories = {
        "黑色": ["螺纹钢", "铁矿石", "焦煤", "焦炭", "热卷", "硅铁", "锰硅"],
        "化工": ["甲醇", "PTA", "纯碱", "玻璃", "塑料", "PVC", "PP", "乙二醇", "苯乙烯", "尿素"],
        "农产品": ["豆粕", "豆油", "棕榈油", "棉花", "白糖", "菜油", "菜粕", "红枣", "苹果", "花生", "玉米", "淀粉"],
        "有色": ["沪铜", "沪铝", "沪锌", "沪镍", "沪金", "沪银"],
        "能源": ["原油", "燃油", "沥青"],
        "其他": ["橡胶", "纸浆", "鸡蛋", "生猪", "碳酸锂", "工业硅", "豆二"]
    }
    for cat, products in categories.items():
        with st.expander(cat, expanded=False):
            cols = st.columns(2)
            for i, p in enumerate(products):
                if cols[i % 2].button(p, key=f"cat_{p}", use_container_width=True):
                    period_val = PERIODS[sel_period]
                    with st.spinner(f"加载 {p} {sel_period} 数据..."):
                        df = load_data(p, period_val)
                        if df is not None and len(df) > BARS_TO_SHOW:
                            start, end = random_bar(df, BARS_TO_SHOW)
                            st.session_state.chart_df = df
                            st.session_state.display_start = start
                            st.session_state.display_end = end
                            st.session_state.total_bars = len(df)
                            st.session_state.product = p
                            st.session_state.period = period_val
                            st.session_state.data_loaded = True
                            st.session_state.skill_rounds = {s: 0 for s in SKILLS}
                            st.session_state.skill_responses = {}
                            st.session_state.current_skill = None
                            st.rerun()
                        else:
                            st.error("数据加载失败，请尝试其他品种或周期")

# ========== 主界面 ==========
st.markdown("""
<style>
    .stApp { background: #f8f9fa; }
    .main-header { font-size: 22px; font-weight: 600; padding: 10px 0; }
    .chat-bubble { padding: 10px; border-radius: 8px; margin: 5px 0; }
    .user-bubble { background: #e3f2fd; }
    .ai-bubble { background: #f5f5f5; border-left: 3px solid #1976d2; }
</style>
""", unsafe_allow_html=True)

col_title, col_info = st.columns([3, 2])
with col_title:
    st.markdown(f"<div class='main-header'>{st.session_state.product} {PERIOD_LABELS.get(st.session_state.period, '30分钟')} · Al Brooks 读盘训练</div>", unsafe_allow_html=True)
with col_info:
    if st.session_state.data_loaded:
        st.markdown(f"共 {st.session_state.total_bars} 根K线 | 当前 {st.session_state.display_start+1}-{st.session_state.display_end}")

# ========== 图表 ==========
if st.session_state.data_loaded and st.session_state.chart_df is not None:
    df = st.session_state.chart_df
    start = st.session_state.display_start
    end = st.session_state.display_end
    total = st.session_state.total_bars

    col_prev, col_next, col_random, col_spacer, col_status = st.columns([1, 1, 1.5, 3, 3])
    with col_prev:
        if st.button("◀ 上一根", use_container_width=True, disabled=(start <= 0)):
            st.session_state.display_start = start - 1
            st.session_state.display_end = end - 1
            st.session_state.skill_rounds = {s: 0 for s in SKILLS}
            st.session_state.skill_responses = {}
            st.session_state.current_skill = None
            st.rerun()
    with col_next:
        if st.button("下一根 ▶", use_container_width=True, disabled=(end >= total)):
            st.session_state.display_start = start + 1
            st.session_state.display_end = end + 1
            st.session_state.skill_rounds = {s: 0 for s in SKILLS}
            st.session_state.skill_responses = {}
            st.session_state.current_skill = None
            st.rerun()
    with col_random:
        if st.button("随机跳转", use_container_width=True):
            new_start, new_end = random_bar(df, BARS_TO_SHOW)
            st.session_state.display_start = new_start
            st.session_state.display_end = new_end
            st.session_state.skill_rounds = {s: 0 for s in SKILLS}
            st.session_state.skill_responses = {}
            st.session_state.current_skill = None
            st.rerun()
    with col_status:
        st.markdown(f"第 {end}/{total} 根K线")

    df_segment = df.iloc[start:end].copy()
    fig = plot_candlestick(df_segment, width=1000, height=520)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("请在侧边栏选择品种并点击「加载/重载数据」开始训练")

# ========== 技能训练 ==========
if st.session_state.data_loaded and st.session_state.chart_df is not None:
    st.markdown("---")
    st.markdown("### 技能训练")
    skill_cols = st.columns(len(SKILLS))
    for i, skill in enumerate(SKILLS):
        rounds = st.session_state.skill_rounds.get(skill, 0)
        disabled = rounds >= MAX_ROUNDS_PER_SKILL
        label = f"{skill}" if not disabled else f"{skill} ✅"
        with skill_cols[i]:
            if st.button(label, key=f"skill_{skill}", use_container_width=True, disabled=disabled):
                st.session_state.current_skill = skill
                st.rerun()

    current_skill = st.session_state.current_skill
    if current_skill:
        rounds = st.session_state.skill_rounds.get(current_skill, 0)
        remaining = MAX_ROUNDS_PER_SKILL - rounds
        st.markdown(f"---\n#### 当前技能：{current_skill}（剩余 {remaining} 轮）")
        st.markdown(f"训练目标：{SKILL_PROMPTS[current_skill]}")

        if current_skill in st.session_state.skill_responses:
            for resp in st.session_state.skill_responses[current_skill]:
                if resp.get("user_input"):
                    st.markdown(f"<div class='chat-bubble user-bubble'><b>你的观察：</b><br>{resp['user_input']}</div>", unsafe_allow_html=True)
                if resp.get("ai_reply"):
                    st.markdown(f"<div class='chat-bubble ai-bubble'><b>AI教练：</b><br>{resp['ai_reply']}</div>", unsafe_allow_html=True)

        if remaining > 0:
            user_input = st.text_area("你的观察：", key=f"input_{current_skill}_{rounds}", height=100,
                                      placeholder="描述你看到的K线形态、趋势、信号...")
            col_submit, col_skip = st.columns([1, 1])
            with col_submit:
                if st.button("提交给AI点评", key=f"submit_{current_skill}_{rounds}", use_container_width=True):
                    if user_input.strip():
                        with st.spinner("AI教练正在分析..."):
                            df_seg = st.session_state.chart_df.iloc[st.session_state.display_start:st.session_state.display_end]
                            ai_reply = get_ai_comment(current_skill, df_seg, user_input)
                            if current_skill not in st.session_state.skill_responses:
                                st.session_state.skill_responses[current_skill] = []
                            st.session_state.skill_responses[current_skill].append({"user_input": user_input, "ai_reply": ai_reply})
                            st.session_state.skill_rounds[current_skill] = rounds + 1
                            st.rerun()
                    else:
                        st.warning("请输入观察后再提交")
            with col_skip:
                if st.button("跳过（看AI分析）", key=f"skip_{current_skill}_{rounds}", use_container_width=True):
                    with st.spinner("AI教练正在分析..."):
                        df_seg = st.session_state.chart_df.iloc[st.session_state.display_start:st.session_state.display_end]
                        ai_reply = get_ai_comment(current_skill, df_seg, None)
                        if current_skill not in st.session_state.skill_responses:
                            st.session_state.skill_responses[current_skill] = []
                        st.session_state.skill_responses[current_skill].append({"user_input": None, "ai_reply": ai_reply})
                        st.session_state.skill_rounds[current_skill] = rounds + 1
                        st.rerun()
        else:
            st.success(f"✅ {current_skill} 已完成 {MAX_ROUNDS_PER_SKILL} 轮训练！")

    st.markdown("---")
    st.markdown("### 训练进度")
    progress_cols = st.columns(len(SKILLS))
    for i, skill in enumerate(SKILLS):
        rounds = st.session_state.skill_rounds.get(skill, 0)
        with progress_cols[i]:
            st.progress(rounds / MAX_ROUNDS_PER_SKILL, text=f"{skill} ({rounds}/{MAX_ROUNDS_PER_SKILL})")
