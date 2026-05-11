# =========================================================
# Al Brooks 逐K训练系统 v3
# 重点：
# 1. 市场控制权训练
# 2. 推进质量引擎
# 3. Push / Pullback 识别
# 4. Failed Breakout 识别
# 5. AI偏差分析
# 6. TradingView风格布局
#
# 运行:
# streamlit run app.py
#
# 安装:
# pip install streamlit pandas plotly openai numpy
#
# CSV格式:
# Open High Low Close
# =========================================================

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# =========================================================
# 页面
# =========================================================

st.set_page_config(
    page_title="Al Brooks 逐K训练系统",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-size: 13px;
}

.block-container{
    padding-top:0.4rem;
    padding-bottom:0.4rem;
    max-width:1800px;
}

div[data-testid="stHorizontalBlock"]{
    gap:0.5rem;
}

div[data-testid="stMetric"]{
    background:#ffffff;
    border:1px solid #2f2f2f;
    border-radius:5px;
    padding:4px;
}

div[data-testid="stMetricLabel"]{
    font-size:11px;
}

div[data-testid="stMetricValue"]{
    font-size:16px;
}

.stTextArea textarea{
    background:#ffffff;
    color:#eaeaea;
    border-radius:8px;
    font-size:13px !important;
    line-height:1.5;
}

.stSelectbox label{
    font-size:12px !important;
}

button[kind="primary"]{
    border-radius:8px !important;
    height:15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# API
# =========================================================

API_KEY = "你的API_KEY"

BASE_URL = "https://你的中转地址/v1"

MODEL_NAME = "gpt-5-mini"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# =========================================================
# 数据目录
# =========================================================

DATA_DIR = Path("data")

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) == 0:

    st.error("data文件夹为空")

    st.stop()

# =========================================================
# Session
# =========================================================

WINDOW_SIZE = 140

if "selected_file" not in st.session_state:

    st.session_state.selected_file = random.choice(csv_files)

if "logs" not in st.session_state:

    st.session_state.logs = []

# =========================================================
# 加载数据
# =========================================================

file_path = st.session_state.selected_file

df = pd.read_csv(file_path)

required_cols = [
    "Open",
    "High",
    "Low",
    "Close"
]

for col in required_cols:

    if col not in df.columns:

        st.error(f"缺少字段: {col}")

        st.stop()

df = df.reset_index(drop=True)

# =========================================================
# 随机起点
# =========================================================

if "random_start" not in st.session_state:

    st.session_state.random_start = random.randint(
        120,
        len(df)-50
    )

if "current_index" not in st.session_state:

    st.session_state.current_index = (
        st.session_state.random_start
    )

current_index = st.session_state.current_index

start_index = max(
    0,
    current_index - WINDOW_SIZE
)

visible_df = df.iloc[
    start_index:current_index
]

# =========================================================
# Swing识别
# =========================================================

def detect_swings(data):

    swings = []

    highs = data["High"].tolist()
    lows = data["Low"].tolist()

    for i in range(2, len(data)-2):

        # swing high
        if (
            highs[i] > highs[i-1]
            and highs[i] > highs[i-2]
            and highs[i] > highs[i+1]
            and highs[i] > highs[i+2]
        ):

            swings.append({
                "index": i,
                "type": "SH",
                "price": highs[i]
            })

        # swing low
        if (
            lows[i] < lows[i-1]
            and lows[i] < lows[i-2]
            and lows[i] < lows[i+1]
            and lows[i] < lows[i+2]
        ):

            swings.append({
                "index": i,
                "type": "SL",
                "price": lows[i]
            })

    return swings

swings = detect_swings(
    visible_df
)

# =========================================================
# 结构识别
# =========================================================

def detect_structure(swings):

    if len(swings) < 6:

        return "区间"

    highs = [
        x for x in swings
        if x["type"] == "SH"
    ]

    lows = [
        x for x in swings
        if x["type"] == "SL"
    ]

    if len(highs) < 2 or len(lows) < 2:

        return "区间"

    hh = highs[-1]["price"] > highs[-2]["price"]

    hl = lows[-1]["price"] > lows[-2]["price"]

    ll = lows[-1]["price"] < lows[-2]["price"]

    lh = highs[-1]["price"] < highs[-2]["price"]

    if hh and hl:

        return "HH/HL"

    if ll and lh:

        return "LL/LH"

    return "区间"

market_structure = detect_structure(
    swings
)

# =========================================================
# Push / Pullback
# =========================================================

def detect_push_pullback(data):

    recent = data.tail(20)

    closes = recent["Close"].tolist()

    direction = closes[-1] - closes[0]

    highest = recent["High"].max()

    lowest = recent["Low"].min()

    total_move = highest - lowest

    recent_pullback = abs(
        closes[-1] - max(closes)
    )

    if total_move == 0:

        return {
            "phase":"平衡",
            "depth":"无"
        }

    ratio = recent_pullback / total_move

    if direction > 0:

        if ratio < 0.25:

            return {
                "phase":"Push",
                "depth":"浅回调"
            }

        if ratio < 0.5:

            return {
                "phase":"Pullback",
                "depth":"中等回调"
            }

        return {
            "phase":"深回调",
            "depth":"深回调"
        }

    else:

        if ratio < 0.25:

            return {
                "phase":"Push",
                "depth":"浅回调"
            }

        if ratio < 0.5:

            return {
                "phase":"Pullback",
                "depth":"中等回调"
            }

        return {
            "phase":"深回调",
            "depth":"深回调"
        }

push_pullback = detect_push_pullback(
    visible_df
)

# =========================================================
# 推进质量引擎
# =========================================================

def momentum_engine(data):

    recent = data.tail(15)

    score = 0

    closes = recent["Close"].tolist()

    highs = recent["High"].tolist()

    lows = recent["Low"].tolist()

    # =====================================================
    # 1. 净推进距离
    # =====================================================

    net_move = abs(
        closes[-1] - closes[0]
    )

    total_range = max(highs) - min(lows)

    if total_range > 0:

        efficiency = net_move / total_range

        score += efficiency * 30

    # =====================================================
    # 2. 收盘质量
    # =====================================================

    close_quality = 0

    for _, row in recent.iterrows():

        h = row["High"]
        l = row["Low"]
        c = row["Close"]
        o = row["Open"]

        if h - l == 0:
            continue

        if c > o:

            pos = (c - l) / (h - l)

            close_quality += pos

        else:

            pos = (h - c) / (h - l)

            close_quality += pos

    close_quality = (
        close_quality / len(recent)
    )

    score += close_quality * 25

    # =====================================================
    # 3. 回调深度
    # =====================================================

    pullbacks = []

    for i in range(1, len(closes)):

        diff = closes[i] - closes[i-1]

        if diff < 0:

            pullbacks.append(abs(diff))

    if len(pullbacks) > 0:

        avg_pullback = np.mean(
            pullbacks
        )

        score += max(
            0,
            25 - avg_pullback * 5
        )

    # =====================================================
    # 4. 连续性
    # =====================================================

    bull_count = 0
    bear_count = 0

    for _, row in recent.iterrows():

        if row["Close"] > row["Open"]:

            bull_count += 1

        elif row["Close"] < row["Open"]:

            bear_count += 1

    continuity = abs(
        bull_count - bear_count
    )

    score += continuity * 2

    # =====================================================
    # 分类
    # =====================================================

    if score >= 70:

        return {
            "score": round(score,1),
            "state":"强推进"
        }

    if score >= 50:

        return {
            "score": round(score,1),
            "state":"一般推进"
        }

    if score >= 35:

        return {
            "score": round(score,1),
            "state":"推进困难"
        }

    return {
        "score": round(score,1),
        "state":"混乱"
    }

momentum_data = momentum_engine(
    visible_df
)

# =========================================================
# Failed Breakout
# =========================================================

def detect_failed_breakout(data):

    recent = data.tail(6)

    last = recent.iloc[-1]

    prev_high = recent["High"].iloc[:-1].max()

    prev_low = recent["Low"].iloc[:-1].min()

    rng = (
        last["High"] - last["Low"]
    )

    if rng == 0:

        return "无"

    # 假向上突破
    if (
        last["High"] > prev_high
        and last["Close"] <
        last["High"] - rng*0.5
    ):

        return "向上失败突破"

    # 假向下突破
    if (
        last["Low"] < prev_low
        and last["Close"] >
        last["Low"] + rng*0.5
    ):

        return "向下失败突破"

    return "无"

failed_breakout = detect_failed_breakout(
    visible_df
)

# =========================================================
# 顶部栏
# =========================================================

st.title("Al Brooks 逐K训练系统")

top1, top2, top3, top4 = st.columns(
    [1,1,1,2]
)

with top1:

    if st.button("随机图表"):

        st.session_state.selected_file = (
            random.choice(csv_files)
        )

        new_df = pd.read_csv(
            st.session_state.selected_file
        )

        st.session_state.random_start = (
            random.randint(
                120,
                len(new_df)-50
            )
        )

        st.session_state.current_index = (
            st.session_state.random_start
        )

        st.rerun()

with top2:

    if st.button("下一根K线"):

        if current_index < len(df):

            st.session_state.current_index += 1

            st.rerun()

with top3:

    st.metric(
        "位置",
        f"{current_index}/{len(df)}"
    )

with top4:

    st.write(
        f"当前文件：{file_path.name}"
    )

# =========================================================
# 左右布局
# =========================================================

left, right = st.columns(
    [3.2,1]
)

# =========================================================
# 左侧图表
# =========================================================

with left:

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=list(range(len(visible_df))),
            open=visible_df["Open"],
            high=visible_df["High"],
            low=visible_df["Low"],
            close=visible_df["Close"]
        )
    )

    # swing标记
    for swing in swings[-20:]:

        fig.add_annotation(
            x=swing["index"],
            y=swing["price"],
            text=swing["type"],
            showarrow=True,
            font=dict(size=9)
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=5,
            r=5,
            t=5,
            b=5
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

# =========================================================
# 右侧面板
# =========================================================

with right:

    st.subheader("系统读图")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "结构",
            market_structure
        )

        st.metric(
            "阶段",
            push_pullback["phase"]
        )

    with c2:

        st.metric(
            "推进",
            momentum_data["state"]
        )

        st.metric(
            "失败突破",
            failed_breakout
        )

    st.metric(
        "推进分数",
        momentum_data["score"]
    )

    st.divider()

    st.subheader("你的判断")

    structure = st.selectbox(
        "结构",
        [
            "HH/HL",
            "LL/LH",
            "区间",
            "转换"
        ]
    )

    control = st.selectbox(
        "控制权",
        [
            "多头明显控制",
            "多头轻微控制",
            "平衡",
            "空头轻微控制",
            "空头明显控制"
        ]
    )

    momentum_user = st.selectbox(
        "推进质量",
        [
            "强推进",
            "一般推进",
            "推进困难",
            "混乱"
        ]
    )

    market_state = st.selectbox(
        "当前状态",
        [
            "趋势延续",
            "正常回调",
            "震荡",
            "可能反转"
        ]
    )

    notes = st.text_area(
        "市场观察",
        placeholder="""1. 谁控制市场？
2. 推进是否轻松？
3. 回调是否破坏结构？
4. 是否出现失败突破？
5. 当前更像趋势还是转换？
""",
        height=140
    )

# =========================================================
# 上下文
# =========================================================

def build_context(data):

    recent = data.tail(20)

    candles = []

    for _, row in recent.iterrows():

        candles.append({
            "open": round(float(row["Open"]),2),
            "high": round(float(row["High"]),2),
            "low": round(float(row["Low"]),2),
            "close": round(float(row["Close"]),2)
        })

    return candles

# =========================================================
# GPT分析
# =========================================================

def analyze_with_gpt(
    candles,
    system_read,
    user_answer
):

    system_prompt = """
你是严格的Al Brooks价格行为导师。

目标：

分析用户是否真正理解市场控制权。

重点：

1. 是否忽略背景结构
2. 是否提前猜反转
3. 是否被单根K线误导
4. 是否忽略持续性
5. 是否错误定义回调

不要预测未来。

不要安慰。

不要模糊表达。

输出JSON：

{
 "score":0,
 "market_read":"",
 "is_reasonable":true,
 "hard_errors":[],
 "mistakes":[],
 "coach_feedback":"",
 "focus_next":""
}
"""

    user_prompt = f"""
真实市场：

{json.dumps(system_read, ensure_ascii=False)}

最近20根K线：

{json.dumps(candles, ensure_ascii=False)}

用户判断：

{json.dumps(user_answer, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":user_prompt
            }
        ],
        temperature=0.1,
        response_format={
            "type":"json_object"
        }
    )

    return json.loads(
        response.choices[0]
        .message.content
    )

# =========================================================
# AI分析
# =========================================================

if st.button(
    "开始AI分析",
    use_container_width=True
):

    candles = build_context(
        visible_df
    )

    system_read = {
        "structure": market_structure,
        "momentum": momentum_data,
        "push_pullback": push_pullback,
        "failed_breakout": failed_breakout,
        "swings": swings[-10:]
    }

    user_answer = {
        "structure": structure,
        "control": control,
        "momentum": momentum_user,
        "market_state": market_state,
        "notes": notes
    }

    with st.spinner("AI分析中..."):

        try:

            ai_result = analyze_with_gpt(
                candles,
                system_read,
                user_answer
            )

        except Exception as e:

            ai_result = {
                "score":0,
                "market_read":"AI失败",
                "is_reasonable":False,
                "hard_errors":[str(e)],
                "mistakes":[],
                "coach_feedback":"检查API",
                "focus_next":"修复API"
            }

    st.session_state.logs.append({
        "bar":current_index,
        "ai_result":ai_result
    })

    st.divider()

    st.subheader("AI反馈")

    st.metric(
        "评分",
        ai_result.get("score",0)
    )

    st.info(
        ai_result.get(
            "market_read",
            ""
        )
    )

    if ai_result.get(
        "is_reasonable",
        False
    ):

        st.success("整体逻辑合理")

    else:

        st.error("市场理解存在偏差")

    hard_errors = ai_result.get(
        "hard_errors",
        []
    )

    if hard_errors:

        st.subheader("严重错误")

        for e in hard_errors:

            st.error(e)

    mistakes = ai_result.get(
        "mistakes",
        []
    )

    if mistakes:

        st.subheader("发现的问题")

        for m in mistakes:

            st.warning(m)

    st.subheader("导师反馈")

    st.write(
        ai_result.get(
            "coach_feedback",
            ""
        )
    )

    st.subheader("下一步重点")

    st.write(
        ai_result.get(
            "focus_next",
            ""
        )
    )

# =========================================================
# 错误统计
# =========================================================

st.divider()

st.subheader("长期错误统计")

mistake_pool = []

for item in st.session_state.logs:

    mistakes = (
        item["ai_result"]
        .get("mistakes",[])
    )

    for m in mistakes:

        mistake_pool.append(m)

if len(mistake_pool) > 0:

    stats = pd.Series(
        mistake_pool
    ).value_counts()

    stats_df = pd.DataFrame({
        "错误类型": stats.index,
        "次数": stats.values
    })

    st.dataframe(
        stats_df,
        use_container_width=True
    )

else:

    st.info("暂无统计")

# =========================================================
# 下载记录
# =========================================================

if len(st.session_state.logs) > 0:

    json_data = json.dumps(
        st.session_state.logs,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        "下载训练记录",
        json_data,
        file_name="training_logs.json",
        mime="application/json"
    )

# =========================================================
