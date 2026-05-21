# =========================================================
# Al Brooks Structure Trainer V2
# =========================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak

from datetime import datetime
from openai import OpenAI

# =========================================================
# 页面配置
# =========================================================

st.set_page_config(
    page_title="Al Brooks 结构训练器",
    layout="wide"
)

# =========================================================
# API 配置
# =========================================================

api_key = st.secrets["OPENAI_API_KEY"]

BASE_URL = "https://api.videocaptioner.cn/v1"
MODEL_NAME = "gpt-5.4-nano"

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)

# =========================================================
# 样式
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-size: 13px !important;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    max-width: 100%;
}

.stButton button {
    width: 100%;
    height: 42px;
    border-radius: 8px;
}

section[data-testid="stSidebar"] {
    width: 320px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 数据加载
# =========================================================

@st.cache_data(ttl=300)
def load_data(symbol="IF0"):

    df = ak.futures_zh_minute_sina(
        symbol=symbol,
        period="30"
    )

    raw_columns = df.columns.tolist()

    rename_map = {}

    for i, col in enumerate(raw_columns):

        if i == 0:
            rename_map[col] = "datetime"

        elif i == 1:
            rename_map[col] = "open"

        elif i == 2:
            rename_map[col] = "high"

        elif i == 3:
            rename_map[col] = "low"

        elif i == 4:
            rename_map[col] = "close"

        elif i == 5:
            rename_map[col] = "volume"

        elif i == 6:
            rename_map[col] = "hold"

    df.rename(columns=rename_map, inplace=True)

    keep_cols = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    existing_cols = [
        c for c in keep_cols
        if c in df.columns
    ]

    df = df[existing_cols].copy()

    df["datetime"] = pd.to_datetime(df["datetime"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df

# =========================================================
# Session
# =========================================================

if "logs" not in st.session_state:
    st.session_state.logs = []

if "current_index" not in st.session_state:
    st.session_state.current_index = 120

# =========================================================
# 顶部
# =========================================================
st.title("----------------------------")
st.title("Al Brooks 结构训练器 V2【不是预测下一根K线。核心问题：当前市场，真的发生控制权转换了吗？】")

top1, top2, top3 = st.columns([1, 1, 2])

with top1:

    symbol = st.selectbox(
        "期货品种",
        [
            "IF0",
            "IC0",
            "IH0",
            "RB0",
            "AU0",
            "AG0",
            "SC0"
        ]
    )

with top2:

    if st.button("重新加载数据"):

        st.cache_data.clear()
        st.rerun()

# =========================================================
# 获取数据
# =========================================================

try:

    df = load_data(symbol)

except Exception as e:

    st.error(f"数据加载失败：{e}")
    st.stop()

MAX_INDEX = len(df) - 10

if st.session_state.current_index > MAX_INDEX:
    st.session_state.current_index = 120

current_index = st.session_state.current_index

# =========================================================
# K线窗口
# =========================================================

WINDOW = 80

start_index = max(
    0,
    current_index - WINDOW
)

chart_df = df.iloc[
    start_index:current_index + 1
].copy()

chart_df.reset_index(
    drop=True,
    inplace=True
)

# =========================================================
# 图表控制
# =========================================================

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:

    if st.button("上一根"):

        if st.session_state.current_index > 50:

            st.session_state.current_index -= 1
            st.rerun()

with c2:

    if st.button("下一根"):

        if st.session_state.current_index < MAX_INDEX:

            st.session_state.current_index += 1
            st.rerun()

with c3:

    if st.button("随机位置"):

        st.session_state.current_index = np.random.randint(
            120,
            MAX_INDEX
        )

        st.rerun()

 with c4:

    st.caption(
        f"当前位置：{current_index}/{len(df)}"
    )

# =========================================================
# 绘图
# =========================================================

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"]
    )
)

fig.add_vline(
    x=len(chart_df)-1,
    line_width=2
)

for i in range(len(chart_df)):

    fig.add_annotation(
        x=i,
        y=chart_df.iloc[i]["high"],
        text=str(i),
        showarrow=False,
        font=dict(size=8)
    )

fig.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False}
)

# =========================================================
# 主训练区域
# =========================================================

st.markdown("---")

left, right = st.columns([1, 1])

# =========================================================
# 结构判断（3列2行布局）
# =========================================================

st.markdown("---")
st.subheader("结构判断")

# =========================================================
# 第一行
# =========================================================

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:

    market_control = st.radio(
        "当前谁控制市场？",
        [
            "多头控制",
            "空头控制",
            "多空平衡"
        ],
        horizontal=False
    )

with row1_col2:

    market_type = st.radio(
        "当前市场类型？",
        [
            "趋势",
            "区间",
            "突破尝试",
            "反转尝试"
        ],
        horizontal=False
    )

with row1_col3:

    momentum_quality = st.radio(
        "当前推进质量？",
        [
            "强推进",
            "健康推进",
            "弱推进",
            "推进衰减"
        ],
        horizontal=False
    )

# =========================================================
# 第二行
# =========================================================

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:

    expectation = st.radio(
        "当前更可能？",
        [
            "延续",
            "反转",
            "继续区间"
        ],
        horizontal=False
    )

with row2_col2:

    breakout_quality = st.radio(
        "当前突破质量？",
        [
            "突破成功概率高",
            "突破失败概率高",
            "暂时不明确"
        ],
        horizontal=False
    )

with row2_col3:

    structure_events = st.multiselect(
        "结构事件",
        [
            "失败突破",
            "楔形",
            "紧密通道",
            "扩张三角形",
            "微型双顶",
            "微型双底",
            "高潮衰竭",
            "重叠增加",
            "尾巴增加",
            "突破后跟进弱"
        ]
    )

# =========================================================
# 一句话记录
# =========================================================

short_note = st.text_area(
    "一句话记录",
    max_chars=100,
    height=100,
    placeholder="例如：多头趋势开始失去连续性"
)

# =========================================================
# 提交按钮
# =========================================================

submit = st.button("提交当前判断")

# =========================================================
# 未来验证
# =========================================================

def validate_future(df, current_index):

    future = df.iloc[
        current_index+1:current_index+6
    ]

    current_close = df.iloc[
        current_index
    ]["close"]

    future_close = future.iloc[-1]["close"]

    move = future_close - current_close

    if move > 0:

        direction = "未来偏多"

    elif move < 0:

        direction = "未来偏空"

    else:

        direction = "未来平衡"

    volatility = (
        future["high"].max()
        - future["low"].min()
    )

    body_efficiency = np.mean(
        abs(future["close"] - future["open"]) /
        (
            future["high"]
            - future["low"]
            + 0.0001
        )
    )

    return {
        "direction": direction,
        "move": round(move, 2),
        "volatility": round(volatility, 2),
        "body_efficiency": round(body_efficiency, 2)
    }

# =========================================================
# AI反馈
# =========================================================

def get_ai_feedback(user_data, validation):

    prompt = f"""
你是Al Brooks价格行为训练教练。

你的任务：

不是预测市场。

而是指出用户结构阅读偏差。

用户判断：

市场控制：
{user_data['market_control']}

市场类型：
{user_data['market_type']}

推进质量：
{user_data['momentum_quality']}

预期：
{user_data['expectation']}

突破质量：
{user_data['breakout_quality']}

结构事件：
{user_data['events']}

备注：
{user_data['note']}

未来验证：

方向：
{validation['direction']}

价格变化：
{validation['move']}

波动：
{validation['volatility']}

实体效率：
{validation['body_efficiency']}

要求：

只输出：

1. 用户忽略了什么
2. 哪个判断偏差最大
3. 下次重点观察什么

限制150字。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# =========================================================
# 偏差统计
# =========================================================

def build_bias_statistics(logs):

    stats = {
        "过早猜反转": 0,
        "趋势误判": 0,
        "区间识别不足": 0,
        "失败突破遗漏": 0
    }

    for log in logs:

        validation = log["validation"]

        if (
            log["expectation"] == "反转"
            and validation["direction"] in ["未来偏多", "未来偏空"]
            and abs(validation["move"]) > 20
        ):

            stats["过早猜反转"] += 1

        if (
            log["market_type"] == "区间"
            and abs(validation["move"]) > 30
        ):

            stats["趋势误判"] += 1

        if (
            log["market_type"] == "趋势"
            and validation["body_efficiency"] < 0.35
        ):

            stats["区间识别不足"] += 1

        if (
            validation["body_efficiency"] < 0.3
            and "失败突破" not in log["events"]
        ):

            stats["失败突破遗漏"] += 1

    return stats

# =========================================================
# 提交逻辑
# =========================================================

if submit:

    validation = validate_future(
        df,
        current_index
    )

    log = {
        "time": str(datetime.now()),
        "bar_index": current_index,
        "market_control": market_control,
        "market_type": market_type,
        "momentum_quality": momentum_quality,
        "expectation": expectation,
        "breakout_quality": breakout_quality,
        "events": structure_events,
        "note": short_note,
        "validation": validation
    }

    st.session_state.logs.append(log)

    try:

        ai_feedback = get_ai_feedback(
            log,
            validation
        )

    except Exception as e:

        ai_feedback = f"AI反馈失败：{e}"

    # =====================================================
    # 展示结果
    # =====================================================

    st.markdown("---")

    st.subheader("未来验证结果")

    v1, v2, v3, v4 = st.columns(4)

    with v1:

        st.metric(
            "未来方向",
            validation["direction"]
        )

    with v2:

        st.metric(
            "价格变化",
            validation["move"]
        )

    with v3:

        st.metric(
            "波动范围",
            validation["volatility"]
        )

    with v4:

        st.metric(
            "实体效率",
            validation["body_efficiency"]
        )

    st.markdown("---")

    st.subheader("AI偏差反馈")

    st.warning(ai_feedback)

# =========================================================
# 偏差画像
# =========================================================

st.markdown("---")

st.subheader("你的结构偏差画像")

if len(st.session_state.logs) > 0:

    stats = build_bias_statistics(
        st.session_state.logs
    )

    b1, b2 = st.columns(2)

    with b1:

        st.markdown(f"""
### 高频错误

- 过早猜反转：{stats['过早猜反转']}
- 趋势误判：{stats['趋势误判']}
""")

    with b2:

        st.markdown(f"""
### 结构识别问题

- 区间识别不足：{stats['区间识别不足']}
- 失败突破遗漏：{stats['失败突破遗漏']}
""")

# =========================================================
# 导出日志
# =========================================================

if st.button("导出训练日志"):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    with open(
        "logs/training_logs.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            st.session_state.logs,
            f,
            ensure_ascii=False,
            indent=4
        )

    st.success(
        "日志已导出到 logs/training_logs.json"
    )

# =========================================================
# 底部
# =========================================================

st.markdown("---")

st.caption("""
系统目标：

不是预测下一根K线。

而是训练：

1. 市场控制权阅读
2. 推进质量识别
3. 趋势衰减识别
4. 区间化识别
5. 失败突破识别
6. continuation vs reversal 判断

核心问题：

当前市场，
真的发生控制权转换了吗？
""")
