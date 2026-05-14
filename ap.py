# =====================================================
# Al Brooks 五层逐K训练系统（优化版）
# 目标：训练“控制权阅读”而非预测
# =====================================================

import json
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI

# =====================================================
# 页面配置
# =====================================================

st.set_page_config(
    page_title="Al Brooks 五层训练系统",
    layout="wide"
)

# =====================================================

# API 配置 - 安全版
# =========================================================
# 1. 从 st.secrets 中安全地读取 API Key
# 这里的 "OPENAI_API_KEY" 是你在 Streamlit 后台设置的变量名
api_key = st.secrets["OPENAI_API_KEY"]

# 2. 其他配置可以保持不变，或者也用 secrets 管理
BASE_URL = "https://api.videocaptioner.cn/v1"
MODEL_NAME = "gpt-5.4-nano"

# 3. 将读取到的 api_key 传给 OpenAI 客户端
client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)

# =====================================================
# CSS 小屏优化
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-size: 12px !important;
}

.block-container {
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
    max-width: 100%;
}

h1 {
    font-size: 20px !important;
    margin-bottom: 0.3rem !important;
}

h2,h3 {
    font-size: 15px !important;
}

textarea {
    font-size: 12px !important;
    line-height: 1.4 !important;
}

.stButton button {
    height: 32px;
    font-size: 12px;
    border-radius: 6px;
}

.stSelectbox label,
.stTextArea label {
    font-size: 11px !important;
}

.custom-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.small-tip {
    font-size: 11px;
    color: #6b7280;
    line-height: 1.4;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 数据生成
# =====================================================


def generate_mock_data(num_bars=300):
    np.random.seed(random.randint(1, 999999))
    data = []
    price = 100
    trend = 0
    for i in range(num_bars):
        if i % 40 == 0:
            trend = random.choice([1.2, -1.2, 0.2, -0.2, 0.0])
        noise = np.random.normal(0, 1.2)
        change = trend + noise
        open_price = price
        close_price = price + change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
        data.append({
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(close_price, 2)
        })
        price = close_price
    return pd.DataFrame(data)


# =====================================================
# Session
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = generate_mock_data()

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 120

if "logs" not in st.session_state:
    st.session_state.logs = []

# =====================================================
# 数据,背景默认为60跟K线
# =====================================================

df = st.session_state.df
current_idx = st.session_state.current_idx
WINDOW = 60
start = max(0, current_idx - WINDOW)
chart_df = df.iloc[start:current_idx + 1].copy()
chart_df["index"] = range(len(chart_df))


# =====================================================
# Swing 识别
# =====================================================

def detect_swings(data):
    swings = []
    highs = data["High"].tolist()
    lows = data["Low"].tolist()
    for i in range(2, len(data) - 2):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            swings.append({"index": i, "type": "SH", "price": highs[i]})
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            swings.append({"index": i, "type": "SL", "price": lows[i]})
    return swings


swings = detect_swings(chart_df)


# =====================================================
# 市场结构识别
# =====================================================

def detect_structure(swings):
    highs = [x for x in swings if x["type"] == "SH"]
    lows = [x for x in swings if x["type"] == "SL"]
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


market_structure = detect_structure(swings)


# =====================================================
# 推进质量
# =====================================================

def momentum_analysis(data):
    recent = data.tail(15)
    total_move = abs(recent.iloc[-1]["Close"] - recent.iloc[0]["Close"])
    total_range = (recent["High"].max() - recent["Low"].min())
    efficiency = 0
    if total_range > 0:
        efficiency = total_move / total_range
    bull_close = 0
    bear_close = 0
    for _, row in recent.iterrows():
        body = abs(row["Close"] - row["Open"])
        range_size = row["High"] - row["Low"]
        if range_size == 0:
            continue
        close_position = (row["Close"] - row["Low"]) / range_size
        if row["Close"] > row["Open"] and close_position > 0.7:
            bull_close += 1
        if row["Close"] < row["Open"] and close_position < 0.3:
            bear_close += 1
    if efficiency > 0.7:
        state = "强推进"
    elif efficiency > 0.5:
        state = "健康推进"
    elif efficiency > 0.3:
        state = "推进困难"
    else:
        state = "混乱/区间"
    return {
        "efficiency": round(efficiency, 2),
        "state": state,
        "bull_close": bull_close,
        "bear_close": bear_close
    }


momentum = momentum_analysis(chart_df)

# =====================================================
# 顶部栏
# =====================================================

st.title("Al Brooks 五层控制权训练")

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:
    if st.button("随机新走势"):
        st.session_state.df = generate_mock_data()
        st.session_state.current_idx = 120
        st.rerun()

with c2:
    if st.button("上一根"):
        if st.session_state.current_idx > 30:
            st.session_state.current_idx -= 1
            st.rerun()

with c3:
    if st.button("下一根"):
        if st.session_state.current_idx < len(df) - 2:
            st.session_state.current_idx += 1
            st.rerun()

with c4:
    st.caption(f"当前位置：{current_idx}/{len(df)}")

# =====================================================
# A区域：图表区（正常布局，不加容器限制）
# =====================================================

# 绘制K线图
fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=chart_df["index"],
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"]
    )
)
# 新增：为每根K线标上序号
for i in range(len(chart_df)):
    fig.add_annotation(
        x=i,
        y=chart_df["High"].iloc[i],  # 标注在每根K线的最高价上方
        text=str(i),                 # 显示当前K线的序号
        showarrow=False,             # 不显示箭头
        font=dict(size=9, color="gray"), # 字体大小和颜色
        yshift=5                     # 稍微向上偏移5个像素，防止遮挡K线
    )
# 网格线
# for x in range(0, len(chart_df), 5):
#     fig.add_vline(x=x, line_dash="dot", opacity=0.15)
#
# max_price = chart_df["High"].max()
# min_price = chart_df["Low"].min()
# step = (max_price - min_price) * 0.01
# current = min_price
# while current <= max_price:
#     fig.add_hline(y=current, line_dash="dot", opacity=0.12)
#     current += step

# Swing 标记
for s in swings[-15:]:
    fig.add_annotation(
        x=s["index"],
        y=s["price"],
        text=s["type"],
        showarrow=True,
        font=dict(size=9)
    )

fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_rangeslider_visible=False,
    dragmode="pan"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =====================================================
# B区域：表单区（放入固定高度容器，独立滚动）
# =====================================================

st.subheader("✍️ 五层强制思考工作表")

# 核心：使用容器包裹表单，实现独立滚动
with st.container(height=400, border=True):
    # 表单内容
    with st.form("perception_form"):
        # 第一层
        st.markdown("### 第一层｜背景")
        st.markdown("<div class='small-tip'>先看背景，再看单根K线</div>", unsafe_allow_html=True)
        layer1 = st.text_area("市场背景", placeholder="现在是HH/HL？LL/LH？还是区间？谁控制市场？", height=70, key="l1")

        # 第二层
        st.markdown("### 第二层｜当前K线")
        layer2 = st.text_area("当前K线含义", placeholder="当前K线是否改变控制权？", height=70, key="l2")

        # 第三层
        st.markdown("### 第三层｜推进质量")
        layer3 = st.text_area("推进/回调质量", placeholder="推进轻松吗？回调深吗？恢复快吗？", height=70, key="l3")

        # 第四层
        st.markdown("### 第四层｜转换检查")
        layer4 = st.text_area("是否真正转换", placeholder="结构是否真正被破坏？对手是否真正接管？", height=70, key="l4")

        # 第五层
        st.markdown("### 第五层｜最终校验")
        control = st.selectbox("控制权", ["原趋势继续控制", "控制权争夺", "新方向接管"], key="l5c")
        action = st.selectbox("倾向", ["顺势", "观望", "逆势试探"], key="l5a")

        submit = st.form_submit_button("AI批改", use_container_width=True)


# =====================================================
# AI 分析
# =====================================================

def build_market_context(data):
    recent = data.tail(15)
    desc = ""
    for i, row in recent.iterrows():
        direction = "阳线" if row["Close"] >= row["Open"] else "阴线"
        body = abs(row["Close"] - row["Open"])
        upper = row["High"] - max(row["Open"], row["Close"])
        lower = min(row["Open"], row["Close"]) - row["Low"]
        desc += f"K线{i}:{direction} O:{row['Open']} H:{row['High']} L:{row['Low']} C:{row['Close']} 实体:{round(body, 2)} 上影:{round(upper, 2)} 下影:{round(lower, 2)}\n"
    return desc


def get_ai_analysis():
    market_desc = build_market_context(df.iloc[:current_idx + 1])
    system_prompt = """
    你是一位极度严格的 Al Brooks 价格行为导师。
    你的任务：不是预测市场。而是检查学生是否真正按照：
    1. 背景 2. 当前K线 3. 推进质量 4. 是否真正转换 5. 最终控制权 这个顺序思考。
    重点：严禁把推进困难直接等于反转，严禁单根K线决定趋势，严禁忽略背景结构。
    请返回JSON：{"score": 0-100, "summary": "...", "layer_review": "...", "thinking_error": "...", "next_focus": "..."}
    """
    user_prompt = f"""
    最近15根K线：
    {market_desc}
    学生五层分析：
    第一层：{layer1}
    第二层：{layer2}
    第三层：{layer3}
    第四层：{layer4}
    第五层：控制权[{control}] 倾向[{action}]
    请严格批改。
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# =====================================================
# 提交分析
# =====================================================

if submit:
    if not layer1 or not layer2 or not layer3 or not layer4:
        st.error("请完成五层分析")
    else:
        with st.spinner("AI导师正在审查你的思维链..."):
            try:
                result = get_ai_analysis()
                st.markdown("---")
                st.subheader("AI 导师批改")
                score = result["score"]
                if score >= 80:
                    st.success(f"评分：{score}")
                elif score >= 60:
                    st.warning(f"评分：{score}")
                else:
                    st.error(f"评分：{score}")
                st.info(result["summary"])
                st.warning(result["layer_review"])
                st.error(result["thinking_error"])
                st.success(result["next_focus"])
                st.session_state.logs.append({
                    "score": score,
                    "structure": market_structure,
                    "momentum": momentum["state"]
                })
            except Exception as e:
                st.error(str(e))

# =====================================================
# 底部训练提醒
# =====================================================

st.markdown("---")
st.caption("""
训练目标：
不是预测下一根K线。
而是训练：
1. 背景阅读 2. 控制权识别 3. 推进质量判断 4. 区分正常回调与真正转换 5. 理解市场是否接受新价格
核心问题：当前K线，真的改变控制权了吗？
""")
