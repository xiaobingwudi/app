# Al Brooks 读盘训练器 V16
# =========================================================
# 认知训练工程 — 不是软件工程
# 用户 = 训练者 | GPT = 教练 | 软件 = 训练场
# =========================================================

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak
from openai import OpenAI

# =========================================================
# 常量
# =========================================================
CHUNK_SIZE = 300
SWING_LOOKBACK = 3

SKILLS = {
    1: {"name": "背景阅读",   "question": "当前市场背景是什么？"},
    2: {"name": "控制权识别", "question": "现在谁在控制市场？"},
    3: {"name": "推进质量",   "question": "最近推进的质量如何？"},
    4: {"name": "回调vs转换", "question": "这是正常回调还是控制权转换？"},
    5: {"name": "市场接受",   "question": "市场是否接受了新价格？"},
}

AI_SYSTEM_PROMPT = """
你是 Al Brooks 价格行为训练教练。

你不是交易员。
你不是分析师。
你不是预测模型。

你唯一职责：

帮助用户训练以下5项核心能力：

1. 背景阅读
2. 控制权识别
3. 推进质量判断
4. 区分正常回调与真正转换
5. 理解市场是否接受新价格

--------------------------------------------------

【核心原则】

真正获得能力的人只能是用户。

你永远不能替用户：
- 观察
- 推理
- 下结论
- 判断市场

你只能：
- 引导
- 追问
- 纠偏
- 强迫用户回到具体K线行为

--------------------------------------------------

【你必须严格禁止】

禁止：

- 告诉用户市场方向
- 告诉用户趋势/区间/反转
- 告诉用户谁控制市场
- 告诉用户用户是否正确
- 给交易建议
- 给买卖建议
- 预测后续走势
- 替用户总结市场结论

--------------------------------------------------

【最重要规则】

当用户使用以下抽象词：

- 转强
- 转弱
- 趋势
- 反转
- 突破
- 控制
- 接受
- 拒绝
- 多头
- 空头
- 强势
- 弱势

你绝对不能围绕这些词讨论。

你必须：

强制用户重新回到：

具体K线行为。

例如：

- 哪几根K线？
- 行为从哪里开始变化？
- 后续有没有跟进？
- 对手有没有回应？
- 重叠有没有增加？
- 收盘位置有没有变化？
- 推进是否持续？
- 行为是否连续？

--------------------------------------------------

【你的真正职责】

你只能做5件事：

1. 强迫用户引用具体K线

例如：
- 从哪几根开始？
- 哪一段？
- 哪里的行为发生变化？

--------------------------------------------------

2. 强迫用户描述行为

只能讨论：
- 实体变化
- 收盘位置变化
- 高低点变化
- 重叠变化
- 跟进行为
- 对手回应
- 推进连续性

禁止讨论抽象市场定义。

--------------------------------------------------

3. 强迫用户提供依据

用户每一个观点，
都必须要求：

"依据是什么？"

--------------------------------------------------

4. 强迫用户面对矛盾

例如：

用户说：
"这里开始转强"

你必须追问：

- 为什么后续没有持续跟进？
- 为什么价格仍然频繁重叠？
- 为什么对手仍然持续回应？

--------------------------------------------------

5. 强迫用户观察连续性

你必须不断提醒用户：

不要只看：
- 单根K线
- 单个形态
- 单次突破

而要观察：

- 行为是否持续
- 跟进是否衰减
- 对手是否回应
- 市场是否真正接受价格

--------------------------------------------------

【你的回答风格】

- 简短
- 直接
- 一次只推进一步
- 不长篇解释
- 不分析市场
- 不总结市场
- 不教学式讲解

--------------------------------------------------

【最关键规则】

如果你发现：

用户开始：
- 下定义
- 猜趋势
- 猜反转
- 猜方向

你必须立即：

把用户拉回：

"具体发生了什么行为？"

这是你的最高优先级。
"""

# =========================================================
# 样式注入
# =========================================================
def _inject_css():
    st.markdown("""<style>
/* 全局 */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
div[data-testid="stSidebar"] { background: #1e1e2e; }
div[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* 标题 */
h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #1e1e2e !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #313244 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

/* 按钮 */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid #bac2de !important;
    font-size: 0.85rem !important;
    padding: 0.25rem 0.75rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #89b4fa !important;
    box-shadow: 0 0 0 2px rgba(137,180,250,0.15);
}

/* 主按钮 */
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #89b4fa !important;
    color: #1e1e2e !important;
    border: none !important;
    font-weight: 600;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #74c7ec !important;
}

/* 输入框 */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #bac2de !important;
}
.stTextArea > div > div > textarea {
    border-radius: 8px !important;
    border: 1px solid #bac2de !important;
}

/* 对话气泡 */
.chat-user {
    background: #89b4fa;
    color: #1e1e2e;
    padding: 10px 14px;
    border-radius: 12px 12px 4px 12px;
    margin: 4px 0;
    font-size: 0.9rem;
    max-width: 95%;
    display: inline-block;
    font-weight: 500;
}
.chat-coach {
    background: #f5f5f5;
    color: #313244;
    padding: 10px 14px;
    border-radius: 12px 12px 12px 4px;
    margin: 4px 0;
    font-size: 0.9rem;
    max-width: 95%;
    display: inline-block;
    border-left: 3px solid #89b4fa;
}
.chat-label {
    font-size: 0.75rem;
    color: #6c7086;
    margin: 8px 0 2px 0;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Slider */
.stSlider > div > div > div { color: #89b4fa !important; }

/* Expander */
.streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.9rem !important; }

/* 侧栏标题 */
div[data-testid="stSidebar"] h1 { color: #cdd6f4 !important; font-size: 1.3rem !important; }
div[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem !important; }
</style>""", unsafe_allow_html=True)

# =========================================================
# 数据类
# =========================================================
@dataclass
class Observation:
    skill_id: int
    bar: int
    text: str
    timestamp: str

@dataclass
class TimelineEvent:
    bar: int
    text: str
    timestamp: str

@dataclass
class SwingPoint:
    index: int
    kind: str   # "SH" or "SL"
    price: float

# =========================================================
# 数据加载
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_raw(symbol):
    for _ in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={
                "datetime": "datetime", "open": "open",
                "high": "high", "low": "low", "close": "close"})
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.reset_index(drop=True)
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


def load_data(symbol, seed=None):
    raw = _fetch_raw(symbol)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    n = len(raw)
    if n <= CHUNK_SIZE:
        return raw.reset_index(drop=True)
    rng = random.Random(seed)
    start = rng.randint(0, n - CHUNK_SIZE)
    return raw.iloc[start:start + CHUNK_SIZE].reset_index(drop=True)

# =========================================================
# Swing 检测（仅用于图表标注）
# =========================================================
def detect_swings(df):
    N = SWING_LOOKBACK
    swings = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(N, len(df) - N):
        if all(highs[i] > highs[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SH", price=float(highs[i])))
        if all(lows[i] < lows[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SL", price=float(lows[i])))
    return swings

# =========================================================
# 图表
# =========================================================
def build_chart(chart_df, bar, swings):
    fig = go.Figure()
    vis = chart_df.iloc[:bar + 1]
    if len(vis) == 0:
        return fig

    fig.add_trace(go.Candlestick(
        x=vis.index, open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"],
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71"))

    annotations = []

    # SH/SL 标注 + 价格
    for s in swings:
        if s.index <= bar:
            is_sh = s.kind == "SH"
            color = "#c0392b" if is_sh else "#27ae60"
            symbol = "\u25b2" if is_sh else "\u25bc"
            annotations.append(dict(
                x=s.index, y=s.price,
                text="{} {:.0f}".format(symbol, s.price),
                showarrow=False,
                font=dict(size=9, color=color),
                xanchor="center",
                yshift=14 if is_sh else -14,
            ))

    # 当前K线编号
    cur = chart_df.iloc[bar]
    annotations.append(dict(
        x=bar, y=cur["high"],
        text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#9399b2",
        font=dict(size=9, color="#6c7086"), ax=0, ay=28))

    fig.update_layout(
        annotations=annotations,
        height=480,
        margin=dict(l=20, r=20, t=15, b=5),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#eff1f5", zeroline=False,
                   tickfont=dict(size=10), side="right",
                   title_text="", title_font=dict(size=10)),
        template="plotly_white",
        font=dict(family="system-ui, sans-serif"),
    )
    return fig

# =========================================================
# GPT 教练
# =========================================================
def _build_market_msg(chart_df, bar, skill_name):
    start = max(0, bar - 30)
    recent = []
    for i in range(start, bar + 1):
        r = chart_df.iloc[i]
        recent.append({
            "bar": i,
            "open": round(float(r["open"]), 1),
            "high": round(float(r["high"]), 1),
            "low": round(float(r["low"]), 1),
            "close": round(float(r["close"]), 1),
        })
    return json.dumps({
        "current_bar": bar,
        "total_bars": len(chart_df),
        "skill": skill_name,
        "market": recent,
    }, ensure_ascii=False)


def _call_gpt(messages):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=messages,
                temperature=0.4,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2 and "429" in str(e):
                time.sleep(2 ** (attempt + 1))
                continue
            return "AI\u8c03\u7528\u5931\u8d25: {}".format(e)


def ask_coach(chart_df, bar, skill_name, dialogue, extra=None):
    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": _build_market_msg(chart_df, bar, skill_name)},
    ]
    for msg in dialogue:
        messages.append({"role": msg["role"], "content": msg["content"]})
    if extra:
        messages.append({"role": "user", "content": extra})
    return _call_gpt(messages)


def ask_summary(chart_df, observations, dialogue):
    obs_text = "\n".join("[K{}] {}".format(o.bar, o.text) for o in observations)
    dlg_text = "\n".join(
        "{}: {}".format("\u7528\u6237" if m["role"] == "user" else "\u6559\u7ec3", m["content"])
        for m in dialogue[-20:])
    prompt = (
        "\u4ee5\u4e0b\u662f\u7528\u6237\u672c\u6b21\u8bad\u7ec3\u7684\u5168\u90e8\u89c2\u5bdf\u548c\u6559\u7ec3\u5bf9\u8bdd\u3002\n\n"
        "\u3010\u89c2\u5bdf\u8bb0\u5f55\u3011\n{}\n\n"
        "\u3010\u6559\u7ec3\u5bf9\u8bdd\u3011\n{}\n\n"
        "\u8bf7\u5206\u6790\u7528\u6237\u7684\u8bfb\u76d8\u80fd\u529b\uff0c\u8f93\u51fa\uff1a\n"
        "1. \u7528\u6237\u957f\u671f\u95ee\u9898\uff08\u5177\u4f53\u5230\u884c\u4e3a\u5c42\u9762\uff09\n"
        "2. \u4e60\u60ef\u6027\u9519\u8bef\uff08\u5f15\u7528\u5bf9\u8bdd\u4e2d\u7684\u5b9e\u9645\u8868\u73b0\uff09\n"
        "3. \u4e0b\u4e00\u9636\u6bb5\u8bad\u7ec3\u91cd\u70b9\n\n"
        "\u57fa\u4e8e\u5bf9\u8bdd\u4e2d\u7684\u5b9e\u9645\u8868\u73b0\u5206\u6790\uff0c\u4e0d\u8981\u6cdb\u6cdb\u800c\u8c08\u3002"
    ).format(obs_text, dlg_text)
    return _call_gpt([
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])


def ask_memory_test(chart_df, bar, observations):
    market = _build_market_msg(chart_df, bar, "")
    obs_text = "\n".join("[K{}] {}".format(o.bar, o.text) for o in observations[-10:])
    prompt = (
        "\u8fd9\u662f\u4e00\u6b21\u5ef6\u8fdf\u8bb0\u5fc6\u8bad\u7ec3\u3002\n\n"
        "\u5f53\u524d\u76d8\u9762\uff1a\n{}\n\n"
        "\u7528\u6237\u7684\u89c2\u5bdf\u8bb0\u5f55\uff1a\n{}\n\n"
        "\u8bf7\u6839\u636e\u7528\u6237\u7684\u89c2\u5bdf\u8bb0\u5f55\uff0c\u51fa1-2\u4e2a\u8bb0\u5fc6\u6d4b\u8bd5\u95ee\u9898\uff1a\n"
        "\u6d4b\u8bd5\u7528\u6237\u662f\u5426\u8bb0\u5f97\u4e4b\u524d\u89c2\u5bdf\u5230\u7684\u5177\u4f53\u884c\u4e3a\u53d8\u5316\u3002\n"
        "\u53ea\u95ee\u95ee\u9898\uff0c\u4e0d\u7ed9\u51fa\u7b54\u6848\u3002\n"
        "\u95ee\u9898\u8981\u5177\u4f53\u5230K\u7ebf\u884c\u4e3a\uff0c\u4e0d\u8981\u95ee\u62bd\u8c61\u6982\u5ff5\u3002"
    ).format(market, obs_text)
    return _call_gpt([
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])


def ask_contradiction(chart_df, bar, skill_name, dialogue):
    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": _build_market_msg(chart_df, bar, skill_name)},
    ]
    for msg in dialogue:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": (
        "\u8bf7\u627e\u51fa\u7528\u6237\u89c2\u5bdf\u4e2d\u7684\u77db\u76fe\u4e4b\u5904\u3002\n"
        "\u7528\u6237\u4e4b\u524d\u8bf4\u4e86\u4e00\u4e9b\u89c2\u5bdf\uff0c\u73b0\u5728\u76d8\u9762\u5df2\u7ecf\u53d8\u5316\u3002\n"
        "\u6307\u51fa\u7528\u6237\u89c2\u5bdf\u4e0e\u5b9e\u9645K\u7ebf\u884c\u4e3a\u4e4b\u95f4\u7684\u77db\u76fe\u3002\n"
        "\u4e0d\u8981\u76f4\u63a5\u544a\u8bc9\u7b54\u6848\uff0c\u7528\u63d0\u95ee\u7684\u65b9\u5f0f\u8ba9\u7528\u6237\u81ea\u5df1\u53d1\u73b0\u77db\u76fe\u3002"
    )})
    return _call_gpt(messages)

# =========================================================
# 对话气泡渲染
# =========================================================
def _render_bubble(role, content):
    label = "\u4f60" if role == "user" else "\u6559\u7ec3"
    cls = "chat-user" if role == "user" else "chat-coach"
    safe = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(
        '<div class="chat-label">{}</div>'
        '<div class="{}">{}</div>'.format(label, cls, safe),
        unsafe_allow_html=True)

# =========================================================
# 主程序
# =========================================================
def main():
    _inject_css()

    for key, default in [
        ("data_loaded", False), ("observations", []),
        ("train_mode", 1), ("timeline", []),
        ("replay_mode", "\u590d\u76d8\u6a21\u5f0f"), ("coach_dialogue", []),
        ("send_counter", 0), ("training_summary", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ===== 侧栏 =====
    with st.sidebar:
        st.title("\u8bfb\u76d8\u8bad\u7ec3\u5668")
        st.caption("V16 \u00b7 \u8ba4\u77e5\u8bad\u7ec3\u5de5\u7a0b")

        symbol = st.text_input(
            "\u5408\u7ea6\u4ee3\u7801", value="rb2510", key="sym")
        sc = st.columns(2)
        with sc[0]:
            if st.button("\u52a0\u8f7d", key="load", use_container_width=True):
                _do_load(symbol)
        with sc[1]:
            if st.button("\u6362\u4e00\u6bb5", key="rand", use_container_width=True):
                _do_load(symbol)

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.session_state["replay_mode"] = st.radio(
                "Replay \u6a21\u5f0f",
                ["\u590d\u76d8\u6a21\u5f0f", "\u4e25\u683c\u6a21\u5f0f"],
                key="rm_radio",
                captions=["\u53ef\u56de\u9000\u3001\u5feb\u8fdb", "\u53ea\u80fd +1"],
            )
            st.markdown("---")
            st.markdown("**\u8bad\u7ec3\u76ee\u6807**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                active = st.session_state.get("train_mode") == sid
                prefix = "\u25b6 " if active else "  "
                if st.button(
                    "{}{}. {}".format(prefix, sid, name),
                    key="mode_{}".format(sid), use_container_width=True,
                ):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            st.markdown("---")
            if st.button(
                "\u7ed3\u675f\u8bad\u7ec3 \u2192 \u603b\u7ed3",
                key="end_train", use_container_width=True, type="primary",
            ):
                _do_summary()

            obs_n = len(st.session_state.get("observations", []))
            dlg_n = len(st.session_state.get("coach_dialogue", [])) // 2
            st.caption("\u89c2\u5bdf {} \u6b21  |  \u5bf9\u8bdd {} \u8f6e".format(obs_n, dlg_n))

    # ===== 欢迎页 =====
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks \u8bfb\u76d8\u8bad\u7ec3\u5668")
        st.markdown("")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** \u2014 {}".format(sid, s["name"], s["question"]))
        st.markdown("")
        st.markdown("> \u4f60\u770b\u56fe\u3002\u4f60\u89c2\u5bdf\u3002\u6559\u7ec3\u53ea\u63d0\u95ee\uff0c\u4e0d\u7ed9\u7b54\u6848\u3002")
        st.markdown("")
        st.markdown("**\u8bad\u7ec3\u67b6\u6784\uff1a**")
        st.markdown("- \u7528\u6237 = \u771f\u6b63\u8bad\u7ec3\u8005")
        st.markdown("- GPT = \u6559\u7ec3\uff08\u4e0e\u4f60\u770b\u540c\u4e00\u4e2a\u76d8\u9762\uff09")
        st.markdown("- \u8f6f\u4ef6 = \u8bad\u7ec3\u573a")
        return

    # ===== 训练总结页 =====
    if st.session_state.get("training_summary"):
        st.markdown("## \u8bad\u7ec3\u603b\u7ed3")
        st.markdown(st.session_state["training_summary"])
        if st.button("\u7ee7\u7eed\u8bad\u7ec3", key="resume"):
            st.session_state["training_summary"] = ""
            st.rerun()
        return

    # ===== 主布局 =====
    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar", 0)
    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    swings = st.session_state.get("swings", [])
    mode = st.session_state.get("train_mode", 1)
    skill = SKILLS[mode]
    strict = st.session_state.get("replay_mode") == "\u4e25\u683c\u6a21\u5f0f"

    # -- 图表（全宽）--
    chart = build_chart(chart_df, bar, swings)
    st.plotly_chart(chart, use_container_width=True)

    # -- OHLC + Slider + 导航（一行）--
    cur = chart_df.iloc[bar]
    chg = cur["close"] - cur["open"]
    sign = "+" if chg >= 0 else ""
    ohlc = (
        "<span style='font-size:0.85rem; color:#6c7086; font-weight:600'>"
        "K{} &nbsp; O <b>{:.0f}</b> &nbsp; H <b>{:.0f}</b> "
        "&nbsp; L <b>{:.0f}</b> &nbsp; C <b>{:.0f}</b> "
        "&nbsp; <span style='color:{}'>{:+.0f}</span>"
        "</span>"
    ).format(
        bar, cur["open"], cur["high"], cur["low"], cur["close"],
        "#27ae60" if chg >= 0 else "#e74c3c", chg)
    st.markdown(ohlc, unsafe_allow_html=True)

    # slider
    if strict:
        new_bar = bar
    else:
        new_bar = st.slider(
            "\u4f4d\u7f6e", 0, len(chart_df) - 1, bar, key="bar_slider")
    if not strict and new_bar != bar:
        st.session_state["current_bar"] = new_bar
        st.rerun()

    # 导航按钮
    steps = [(-5, "-5", "b_p5"), (-1, "-1", "b_p1"), (1, "+1", "b_n1"),
             (5, "+5", "b_n5"), (15, "+15", "b_n15"), (None, "\u672b", "b_end")]
    nav = st.columns(len(steps))
    for i, (step, label, key) in enumerate(steps):
        show = (step is not None and (not strict or step > 0)) or (step is None and not strict)
        if show:
            if nav[i].button(label, key=key, use_container_width=True):
                if step is not None:
                    st.session_state["current_bar"] = max(0, min(len(chart_df) - 1, bar + step))
                else:
                    st.session_state["current_bar"] = len(chart_df) - 1
                st.rerun()

    st.markdown("<hr style='margin:4px 0; border-color:#eff1f5'>", unsafe_allow_html=True)

    # -- 主区域：左图表信息 + 右对话 --
    col_left, col_right = st.columns([3, 2], gap="large")

    # ===== 左列：输入区 =====
    with col_left:
        st.markdown("**{}** &nbsp;<span style='font-size:0.8rem; color:#6c7086'>{}</span>".format(
            skill["name"], skill["question"]), unsafe_allow_html=True)

        cnt = st.session_state.get("send_counter", 0)
        obs_text = st.text_area(
            "\u4f60\u89c2\u5bdf\u5230\u4e86\u4ec0\u4e48\uff1f",
            height=100, key="obs_{}".format(cnt),
            placeholder="\u63cf\u8ff0\u4f60\u89c2\u5bdf\u5230\u7684\u5177\u4f53\u884c\u4e3a\u53d8\u5316...",
            label_visibility="visible",
        )

        # 按钮行
        bc = st.columns(4)
        with bc[0]:
            if st.button("\u53d1\u9001", key="send_obs", use_container_width=True, type="primary"):
                if obs_text.strip():
                    _send_observation(obs_text.strip(), chart_df, bar, skill)
        with bc[1]:
            if st.button("\u91cd\u7f6e\u5bf9\u8bdd", key="new_round", use_container_width=True):
                st.session_state["coach_dialogue"] = []
                st.rerun()
        with bc[2]:
            if st.button("\u8bb0\u5fc6\u6d4b\u8bd5", key="btn_memory", use_container_width=True):
                _do_memory(chart_df, bar)
        with bc[3]:
            if st.button("\u627e\u77db\u76fe", key="btn_contra", use_container_width=True):
                _do_contradiction(chart_df, bar, skill)

        # 时间轴
        st.markdown("<hr style='margin:12px 0 6px; border-color:#eff1f5'>", unsafe_allow_html=True)
        with st.expander("\u884c\u4e3a\u53d8\u5316\u8bb0\u5f55 ({})".format(
                len(st.session_state.get("timeline", [])))):
            tl = st.session_state.get("timeline", [])
            for ev in tl[-8:]:
                st.caption("[K{}] {}".format(ev.bar, ev.text))
            tc = st.columns([5, 1])
            with tc[0]:
                tl_input = st.text_input(
                    "\u8bb0\u5f55", key="tl_input",
                    placeholder="\u63cf\u8ff0\u884c\u4e3a\u53d8\u5316...")
            with tc[1]:
                if st.button("\u8bb0", key="tl_add"):
                    if tl_input.strip():
                        st.session_state.setdefault("timeline", []).append(
                            TimelineEvent(bar=bar, text=tl_input.strip(),
                                         timestamp=datetime.now().strftime("%H:%M:%S")))
                        st.rerun()
            if tl:
                if st.button("\u6e05\u7a7a", key="tl_clear"):
                    st.session_state["timeline"] = []
                    st.rerun()

    # ===== 右列：对话区 =====
    with col_right:
        st.markdown("**\u6559\u7ec3\u5bf9\u8bdd**")
        st.markdown("<hr style='margin:0 0 8px; border-color:#eff1f5'>", unsafe_allow_html=True)

        dialogue = st.session_state["coach_dialogue"]
        if not dialogue:
            st.caption("\u53d1\u9001\u4f60\u7684\u89c2\u5bdf\uff0c\u6559\u7ec3\u4f1a\u8ffd\u95ee\u3002")

        for msg in dialogue:
            _render_bubble(msg["role"], msg["content"])

        # 用户总数统计
        if dialogue:
            user_count = sum(1 for m in dialogue if m["role"] == "user")
            coach_count = sum(1 for m in dialogue if m["role"] == "assistant")
            st.markdown(
                "<div style='font-size:0.75rem; color:#9399b2; text-align:right; margin-top:8px'>"
                "\u4f60 {} \u6b21 | \u6559\u7ec3 {} \u6b21"
                "</div>".format(user_count, coach_count),
                unsafe_allow_html=True)


# =========================================================
# 辅助函数
# =========================================================
def _send_observation(text, chart_df, bar, skill):
    session = st.session_state
    dialogue = session["coach_dialogue"]
    mode = session.get("train_mode", 1)

    dialogue.append({"role": "user", "content": text})

    session["observations"].append(Observation(
        skill_id=mode, bar=bar, text=text,
        timestamp=datetime.now().strftime("%H:%M:%S")))

    with st.spinner("\u6559\u7ec3\u601d\u8003\u4e2d..."):
        response = ask_coach(chart_df, bar, skill["name"], dialogue)

    dialogue.append({"role": "assistant", "content": response})
    session["coach_dialogue"] = dialogue
    session["send_counter"] = session.get("send_counter", 0) + 1
    st.rerun()


def _do_memory(chart_df, bar):
    observations = st.session_state.get("observations", [])
    if len(observations) < 3:
        st.warning("\u81f3\u5c11\u89c2\u5bdf 3 \u6b21\u540e\u53ef\u7528")
        return
    with st.spinner("\u51fa\u9898\u4e2d..."):
        q = ask_memory_test(chart_df, bar, observations)
    st.session_state["coach_dialogue"].append(
        {"role": "assistant", "content": "[\u8bb0\u5fc6\u6d4b\u8bd5] " + q})
    st.rerun()


def _do_contradiction(chart_df, bar, skill):
    dialogue = st.session_state["coach_dialogue"]
    if len(dialogue) < 4:
        st.warning("\u81f3\u5c11\u5bf9\u8bdd 2 \u8f6e\u540e\u53ef\u7528")
        return
    with st.spinner("\u5206\u6790\u4e2d..."):
        q = ask_contradiction(chart_df, bar, skill["name"], dialogue)
    st.session_state["coach_dialogue"].append(
        {"role": "assistant", "content": q})
    st.rerun()


def _do_load(symbol):
    with st.spinner("\u52a0\u8f7d\u4e2d..."):
        seed = random.randint(0, 999999)
        df = load_data(symbol, seed=seed)
        if df is not None and len(df) > 0:
            sw = detect_swings(df)
            st.session_state.update({
                "chart_df": df, "swings": sw,
                "current_bar": min(40, len(df) - 1),
                "data_loaded": True, "observations": [],
                "timeline": [], "train_mode": 1,
                "coach_dialogue": [], "training_summary": "",
                "send_counter": 0,
            })
            st.success("{} \u6839 K \u7ebf".format(len(df)))
        else:
            st.error("\u52a0\u8f7d\u5931\u8d25")


def _do_summary():
    session = st.session_state
    chart_df = session.get("chart_df")
    observations = session.get("observations", [])
    dialogue = session.get("coach_dialogue", [])
    if not observations:
        st.warning("\u8fd8\u6ca1\u6709\u89c2\u5bdf\u8bb0\u5f55")
        return
    with st.spinner("\u751f\u6210\u8bad\u7ec3\u603b\u7ed3..."):
        summary = ask_summary(chart_df, observations, dialogue)
    session["training_summary"] = summary


if __name__ == "__main__":
    main()
