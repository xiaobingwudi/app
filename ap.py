# =========================================================
# Al Brooks 结构训练器 V4
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

    df = df[keep_cols].copy()

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
    st.session_state.current_index = None

# =========================================================
# 顶部
# =========================================================

st.title("Al Brooks 结构训练器 V4")

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

MIN_INDEX = 120
MAX_INDEX = len(df) - 10

# =========================================================
# 首次随机
# =========================================================

if st.session_state.current_index is None:

    st.session_state.current_index = np.random.randint(
        MIN_INDEX,
        MAX_INDEX
    )

# =========================================================
# 防止越界
# =========================================================

if st.session_state.current_index > MAX_INDEX:

    st.session_state.current_index = np.random.randint(
        MIN_INDEX,
        MAX_INDEX
    )

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

current_bar_number = len(chart_df) - 1

# =========================================================
# Swing 检测
# =========================================================

def detect_swings(df):

    swings = []

    for i in range(2, len(df)-2):

        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]

        # Swing High

        if (
            current_high > df.iloc[i-1]["high"]
            and current_high > df.iloc[i-2]["high"]
            and current_high > df.iloc[i+1]["high"]
            and current_high > df.iloc[i+2]["high"]
        ):

            swings.append({
                "index": i,
                "type": "SH",
                "price": current_high
            })

        # Swing Low

        if (
            current_low < df.iloc[i-1]["low"]
            and current_low < df.iloc[i-2]["low"]
            and current_low < df.iloc[i+1]["low"]
            and current_low < df.iloc[i+2]["low"]
        ):

            swings.append({
                "index": i,
                "type": "SL",
                "price": current_low
            })

    return swings

# =========================================================
# HH HL LH LL
# =========================================================

def detect_market_structure(swings):

    labels = []

    highs = [
        s for s in swings
        if s["type"] == "SH"
    ]

    lows = [
        s for s in swings
        if s["type"] == "SL"
    ]

    for i in range(1, len(highs)):

        prev_price = highs[i-1]["price"]
        current_price = highs[i]["price"]

        if current_price > prev_price:

            labels.append({
                "index": highs[i]["index"],
                "label": "HH"
            })

        else:

            labels.append({
                "index": highs[i]["index"],
                "label": "LH"
            })

    for i in range(1, len(lows)):

        prev_price = lows[i-1]["price"]
        current_price = lows[i]["price"]

        if current_price > prev_price:

            labels.append({
                "index": lows[i]["index"],
                "label": "HL"
            })

        else:

            labels.append({
                "index": lows[i]["index"],
                "label": "LL"
            })

    return labels

# =========================================================
# Tight Channel
# =========================================================

def detect_tight_channel(df):

    recent = df.tail(10)

    bull_count = 0
    bear_count = 0
    overlap_count = 0

    for i in range(1, len(recent)):

        current = recent.iloc[i]
        prev = recent.iloc[i-1]

        if current["close"] > current["open"]:

            bull_count += 1

        if current["close"] < current["open"]:

            bear_count += 1

        if current["low"] < prev["high"]:

            overlap_count += 1

    if (
        bull_count >= 8
        and overlap_count <= 3
    ):

        return "紧密多头通道"

    if (
        bear_count >= 8
        and overlap_count <= 3
    ):

        return "紧密空头通道"

    return None

# =========================================================
# Failed Breakout
# =========================================================

def detect_failed_breakout(df):

    recent = df.tail(8)

    highs = recent["high"].tolist()

    breakout_high = max(highs[:-1])

    last_bar = recent.iloc[-1]

    if (
        last_bar["high"] > breakout_high
        and last_bar["close"] < breakout_high
    ):

        return "向上失败突破"

    lows = recent["low"].tolist()

    breakout_low = min(lows[:-1])

    if (
        last_bar["low"] < breakout_low
        and last_bar["close"] > breakout_low
    ):

        return "向下失败突破"

    return None

# =========================================================
# 自动结构
# =========================================================

swings = detect_swings(chart_df)

market_labels = detect_market_structure(
    swings
)

tight_channel = detect_tight_channel(
    chart_df
)

failed_breakout = detect_failed_breakout(
    chart_df
)

auto_structures = []

if tight_channel:
    auto_structures.append(tight_channel)

if failed_breakout:
    auto_structures.append(failed_breakout)

# =========================================================
# 顶部控制
# =========================================================

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:

    if st.button("上一根"):

        if st.session_state.current_index > MIN_INDEX:

            st.session_state.current_index -= 1
            st.rerun()

with c2:

    if st.button("下一根"):

        if st.session_state.current_index < MAX_INDEX:

            st.session_state.current_index += 1
            st.rerun()

with c3:

    if st.button("随机位置"):

        old_index = st.session_state.current_index

        while True:

            new_index = np.random.randint(
                MIN_INDEX,
                MAX_INDEX
            )

            if abs(new_index - old_index) > 50:
                break

        st.session_state.current_index = new_index

        st.rerun()

with c4:

    st.markdown(f"""
当前训练K线：

- 全局编号：{current_index}
- 当前窗口编号：{current_bar_number}
""")

# =========================================================
# 图表
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

# 当前K线

fig.add_vline(
    x=current_bar_number,
    line_width=3
)

# K线编号

for i in range(len(chart_df)):

    fig.add_annotation(
        x=i,
        y=chart_df.iloc[i]["high"],
        text=str(i),
        showarrow=False,
        font=dict(size=8)
    )

# Swing

for swing in swings:

    fig.add_annotation(
        x=swing["index"],
        y=swing["price"],
        text=swing["type"],
        showarrow=True,
        font=dict(size=9)
    )

# HH HL LH LL

for label in market_labels:

    row = chart_df.iloc[label["index"]]

    fig.add_annotation(
        x=label["index"],
        y=row["high"],
        text=label["label"],
        showarrow=False,
        font=dict(size=10)
    )

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False}
)

# =========================================================
# 自动结构识别
# =========================================================

st.markdown("---")

st.subheader("自动结构识别")

if len(auto_structures) == 0:

    st.info("当前未检测到明显结构")

else:

    for item in auto_structures:

        st.warning(item)

# =========================================================
# 结构判断
# =========================================================

st.markdown("---")

st.subheader(
    f"当前正在判断：窗口编号 {current_bar_number}"
)

# =========================================================
# 第一行
# =========================================================

r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:

    market_control = st.radio(
        "当前谁控制市场？",
        [
            "多头控制",
            "空头控制",
            "多空平衡"
        ]
    )

with r1c2:

    market_type = st.radio(
        "当前市场类型？",
        [
            "趋势",
            "区间",
            "突破尝试",
            "反转尝试"
        ]
    )

with r1c3:

    momentum_quality = st.radio(
        "当前推进质量？",
        [
            "强推进",
            "健康推进",
            "弱推进",
            "推进衰减"
        ]
    )

# =========================================================
# 第二行
# =========================================================

r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:

    expectation = st.radio(
        "当前更可能？",
        [
            "延续",
            "反转",
            "继续区间"
        ]
    )

with r2c2:

    breakout_quality = st.radio(
        "当前突破质量？",
        [
            "突破成功概率高",
            "突破失败概率高",
            "暂时不明确"
        ]
    )

with r2c3:

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
    max_chars=120,
    height=100,
    placeholder="例如：多头趋势开始失去连续性"
)

# =========================================================
# 提交
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
# 市场背景
# =========================================================

def build_market_context(chart_df):

    lines = []

    for i, row in chart_df.iterrows():

        direction = (
            "阳线"
            if row["close"] >= row["open"]
            else "阴线"
        )

        total_range = (
            row["high"] - row["low"]
        )

        body = abs(
            row["close"] - row["open"]
        )

        upper_tail = (
            row["high"]
            - max(
                row["open"],
                row["close"]
            )
        )

        lower_tail = (
            min(
                row["open"],
                row["close"]
            )
            - row["low"]
        )

        body_ratio = 0

        if total_range > 0:

            body_ratio = round(
                body / total_range,
                2
            )

        lines.append(
            f"""
编号:{i}
{direction}
开:{row['open']}
高:{row['high']}
低:{row['low']}
收:{row['close']}
实体占比:{body_ratio}
上影:{round(upper_tail,2)}
下影:{round(lower_tail,2)}
"""
        )

    return "\n".join(lines)

# =========================================================
# AI反馈
# =========================================================

def get_ai_feedback(
        user_data,
        validation,
        chart_df,
        current_bar_number,
        auto_structures
):

    market_context = build_market_context(
        chart_df
    )

    prompt = f"""
你是Al Brooks价格行为结构训练教练。

你已经获得：

1. 当前屏幕显示的全部K线背景
2. 每根K线编号
3. 用户正在判断的K线编号
4. 用户的结构判断
5. 自动结构识别结果
6. 后续5根K验证

======================
当前正在判断的K线编号
======================

{current_bar_number}

======================
自动结构识别
======================

{auto_structures}

======================
当前屏幕全部K线背景
======================

{market_context}

======================
用户判断
======================

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

======================
未来5根K验证
======================

方向：
{validation['direction']}

价格变化：
{validation['move']}

波动：
{validation['volatility']}

实体效率：
{validation['body_efficiency']}

======================
要求
======================

只输出：

1. 用户忽略了什么
2. 哪个判断偏差最大
3. 下次重点观察什么

额外要求：

1. 如果用户忽略了自动结构，要明确指出
2. 如果用户和结构引擎冲突，要指出冲突
3. 不允许模糊表达
4. 必须具体指出是哪类结构

禁止：

- 安慰
- 废话
- 长篇理论

限制200字。
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
        "window_bar_number": current_bar_number,
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
            validation,
            chart_df,
            current_bar_number,
            auto_structures
        )

    except Exception as e:

        ai_feedback = f"AI反馈失败：{e}"

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
