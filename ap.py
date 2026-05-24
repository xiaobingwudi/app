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
# 样式
# =========================================================
def _css():
    st.markdown("""<style>
/* -- 全局 -- */
.block-container{padding-top:1rem!important;padding-bottom:1rem!important;max-width:100%!important}
.main .block-container{padding-left:1.5rem!important;padding-right:1.5rem!important}

/* -- 侧栏紧凑 -- */
[data-testid="stSidebar"]{width:210px!important;min-width:210px!important;background:#1b1b2f}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]>div{padding-top:4px!important;padding-bottom:4px!important}
[data-testid="stSidebar"] h1{font-size:1rem!important;margin:0}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] p{font-size:.75rem!important;line-height:1.3!important}
[data-testid="stSidebar"] .stTextInput>div>div>input{font-size:.8rem!important;padding:.15rem .4rem!important}
[data-testid="stSidebar"] .stRadio>div>label>div>span{font-size:.8rem!important}
[data-testid="stSidebar"] .stRadio>div>label>p{font-size:.7rem!important}
[data-testid="stSidebar"] .stMarkdown{font-size:.8rem!important}
[data-testid="stSidebar"] .stButton>button{font-size:.75rem!important;padding:.15rem .3rem!important;margin:1px 0}
[data-testid="stSidebar"] .stCheckbox>label>div>div>span{font-size:.8rem!important}
[data-testid="stSidebar"] hr{margin:6px 0!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]+div{margin-top:0!important}

/* -- 按钮 -- */
.stButton>button{border-radius:6px!important;font-size:.82rem!important;padding:.2rem .6rem!important;border:1px solid #d0d7e3!important;white-space:nowrap}
.stButton>button:hover{border-color:#89b4fa!important;box-shadow:0 0 0 2px rgba(137,180,250,.12)}
.stButton>button[data-testid="stBaseButton-primary"]{background:#89b4fa!important;color:#1e1e2e!important;border:none!important;font-weight:600}

/* -- 输入 -- */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{border-radius:6px!important;border:1px solid #d0d7e3!important;font-size:.85rem!important}
.stTextArea>div{height:fit-content!important}

/* -- Slider -- */
.stSlider>div>div>div{font-size:.8rem!important}

/* -- 对话气泡 -- */
.bubble-u{background:#dce8ff;color:#1e1e2e;padding:6px 10px;border-radius:10px 10px 2px 10px;margin:2px 0;font-size:.85rem;max-width:98%;display:inline-block;font-weight:500;line-height:1.4}
.bubble-c{background:#f4f4f6;color:#313244;padding:6px 10px;border-radius:10px 10px 10px 2px;margin:2px 0;font-size:.85rem;max-width:98%;display:inline-block;border-left:3px solid #89b4fa;line-height:1.4}
.lbl-u{font-size:.7rem;color:#6c7086;margin:6px 0 1px;font-weight:600;letter-spacing:.3px}
.lbl-c{font-size:.7rem;color:#89b4fa;margin:6px 0 1px;font-weight:600;letter-spacing:.3px}
.dlg-stat{font-size:.7rem;color:#9399b2;text-align:right;margin-top:6px}

/* -- Expander -- */
.streamlit-expanderHeader{font-size:.85rem!important;font-weight:600!important}
[data-testid="stExpander"]>div>div{font-size:.82rem!important}

/* -- ohlc -- */
.ohlc{font-size:.8rem;color:#6c7086;font-weight:600}
.ohlc b{color:#313244}
.ohlc .up{color:#27ae60}
.ohlc .dn{color:#e74c3c}

/* -- skill tag -- */
.skill-tag{display:inline-block;background:#eef2ff;color:#4a6fa5;padding:2px 10px;border-radius:12px;font-size:.85rem;font-weight:600}
.skill-q{font-size:.8rem;color:#6c7086;margin-left:6px}

/* -- sep -- */
.sep{border:none;border-top:1px solid #e8ecf2;margin:6px 0}
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
    kind: str
    price: float

# =========================================================
# 数据加载
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_raw(symbol):
    for _ in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={"datetime":"datetime","open":"open","high":"high","low":"low","close":"close"})
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for c in ["open","high","low","close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df = df.dropna(subset=["open","high","low","close"])
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
# Swing 检测
# =========================================================
def detect_swings(df):
    N = SWING_LOOKBACK
    swings = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(N, len(df) - N):
        if all(highs[i] > highs[j] for j in range(i-N,i+N+1) if j!=i):
            swings.append(SwingPoint(index=i, kind="SH", price=float(highs[i])))
        if all(lows[i] < lows[j] for j in range(i-N,i+N+1) if j!=i):
            swings.append(SwingPoint(index=i, kind="SL", price=float(lows[i])))
    return swings

# =========================================================
# 图表
# =========================================================
def build_chart(chart_df, bar, swings):
    fig = go.Figure()
    vis = chart_df.iloc[:bar+1]
    if len(vis)==0:
        return fig
    fig.add_trace(go.Candlestick(
        x=vis.index, open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"],
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71"))
    ann = []
    for s in swings:
        if s.index <= bar:
            is_sh = s.kind == "SH"
            c = "#c0392b" if is_sh else "#27ae60"
            sym = "\u25b2" if is_sh else "\u25bc"
            ann.append(dict(x=s.index, y=s.price,
                text="{} {:.0f}".format(sym, s.price),
                showarrow=False, font=dict(size=9, color=c),
                xanchor="center", yshift=14 if is_sh else -14))
    cur = chart_df.iloc[bar]
    ann.append(dict(x=bar, y=cur["high"], text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#9399b2",
        font=dict(size=9, color="#6c7086"), ax=0, ay=28))
    fig.update_layout(annotations=ann, height=420,
        margin=dict(l=15, r=60, t=8, b=5),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#eff1f5", zeroline=False,
                   tickfont=dict(size=10), side="right"),
        template="plotly_white",
        font=dict(family="system-ui,sans-serif"))
    return fig

# =========================================================
# GPT 教练
# =========================================================
def _build_market_msg(chart_df, bar, skill_name):
    start = max(0, bar - 30)
    recent = []
    for i in range(start, bar+1):
        r = chart_df.iloc[i]
        recent.append({"bar":i,"open":round(float(r["open"]),1),
                       "high":round(float(r["high"]),1),
                       "low":round(float(r["low"]),1),
                       "close":round(float(r["close"]),1)})
    return json.dumps({"current_bar":bar,"total_bars":len(chart_df),
                        "skill":skill_name,"market":recent}, ensure_ascii=False)

def _call_gpt(messages):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")
    for att in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano", messages=messages,
                temperature=0.4, max_tokens=400)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if att<2 and "429" in str(e):
                time.sleep(2**(att+1)); continue
            return "AI\u8c03\u7528\u5931\u8d25: {}".format(e)

def ask_coach(chart_df, bar, skill_name, dialogue, extra=None):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_build_market_msg(chart_df, bar, skill_name)}]
    for m in dialogue:
        msgs.append({"role":m["role"],"content":m["content"]})
    if extra:
        msgs.append({"role":"user","content":extra})
    return _call_gpt(msgs)

def ask_summary(chart_df, observations, dialogue):
    ot = "\n".join("[K{}] {}".format(o.bar, o.text) for o in observations)
    dt = "\n".join("{}: {}".format("\u7528\u6237" if m["role"]=="user" else "\u6559\u7ec3", m["content"]) for m in dialogue[-20:])
    return _call_gpt([
        {"role":"system","content":AI_SYSTEM_PROMPT},
        {"role":"user","content":(
            "\u4ee5\u4e0b\u662f\u7528\u6237\u672c\u6b21\u8bad\u7ec3\u7684\u5168\u90e8\u89c2\u5bdf\u548c\u6559\u7ec3\u5bf9\u8bdd\u3002\n\n"
            "\u3010\u89c2\u5bdf\u8bb0\u5f55\u3011\n{}\n\n\u3010\u6559\u7ec3\u5bf9\u8bdd\u3011\n{}\n\n"
            "\u8bf7\u5206\u6790\u7528\u6237\u8bfb\u76d8\u80fd\u529b\uff0c\u8f93\u51fa\uff1a\n"
            "1. \u7528\u6237\u957f\u671f\u95ee\u9898\uff08\u5177\u4f53\u5230\u884c\u4e3a\u5c42\u9762\uff09\n"
            "2. \u4e60\u60ef\u6027\u9519\u8bef\uff08\u5f15\u7528\u5bf9\u8bdd\u4e2d\u7684\u5b9e\u9645\u8868\u73b0\uff09\n"
            "3. \u4e0b\u4e00\u9636\u6bb5\u8bad\u7ec3\u91cd\u70b9\n\n"
            "\u57fa\u4e8e\u5bf9\u8bdd\u4e2d\u7684\u5b9e\u9645\u8868\u73b0\u5206\u6790\uff0c\u4e0d\u8981\u6cdb\u6cdb\u800c\u8c08\u3002"
        ).format(ot, dt)}
    ])

def ask_memory_test(chart_df, bar, observations):
    return _call_gpt([
        {"role":"system","content":AI_SYSTEM_PROMPT},
        {"role":"user","content":(
            "\u8fd9\u662f\u4e00\u6b21\u5ef6\u8fdf\u8bb0\u5fc6\u8bad\u7ec3\u3002\n\n"
            "\u5f53\u524d\u76d8\u9762\uff1a\n{}\n\n"
            "\u7528\u6237\u7684\u89c2\u5bdf\u8bb0\u5f55\uff1a\n{}\n\n"
            "\u8bf7\u6839\u636e\u7528\u6237\u89c2\u5bdf\u8bb0\u5f55\uff0c\u51fa1-2\u4e2a\u8bb0\u5fc6\u6d4b\u8bd5\u95ee\u9898\u3002"
            "\u53ea\u95ee\u95ee\u9898\uff0c\u4e0d\u7ed9\u7b54\u6848\u3002\u95ee\u9898\u8981\u5177\u4f53\u5230K\u7ebf\u884c\u4e3a\u3002"
        ).format(_build_market_msg(chart_df, bar, ""),
                 "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations[-10:]))}
    ])

def ask_contradiction(chart_df, bar, skill_name, dialogue):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_build_market_msg(chart_df, bar, skill_name)}]
    for m in dialogue:
        msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":(
        "\u8bf7\u627e\u51fa\u7528\u6237\u89c2\u5bdf\u4e2d\u7684\u77db\u76fe\u4e4b\u5904\u3002"
        "\u4e0d\u8981\u76f4\u63a5\u544a\u8bc9\u7b54\u6848\uff0c\u7528\u63d0\u95ee\u65b9\u5f0f\u8ba9\u7528\u6237\u81ea\u5df1\u53d1\u73b0\u3002"
    )})
    return _call_gpt(msgs)

# =========================================================
# 气泡渲染
# =========================================================
def _bubble(role, content):
    lbl = "\u4f60" if role=="user" else "\u6559\u7ec3"
    cls = "bubble-u" if role=="user" else "bubble-c"
    lc  = "lbl-u" if role=="user" else "lbl-c"
    safe = content.replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
    st.markdown('<div class="{}">{}</div><div class="{}">{}</div>'.format(lc,lbl,cls,safe), unsafe_allow_html=True)

# =========================================================
# 主程序
# =========================================================
def main():
    _css()

    for k, d in [("data_loaded",False),("observations",[]),("train_mode",1),
                  ("timeline",[]),("replay_mode","\u590d\u76d8\u6a21\u5f0f"),
                  ("coach_dialogue",[]),("send_counter",0),("training_summary","")]:
        if k not in st.session_state:
            st.session_state[k] = d

    # ===== 侧栏（紧凑）=====
    with st.sidebar:
        st.title("\u8bfb\u76d8\u8bad\u7ec3\u5668")
        symbol = st.text_input("\u5408\u7ea6", value="rb2510", key="sym")
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
                "Replay", ["\u590d\u76d8\u6a21\u5f0f","\u4e25\u683c\u6a21\u5f0f"],
                key="rm_radio", captions=["\u53ef\u56de\u9000","\u53ea+1"])
            st.markdown("---")
            st.markdown("**\u8bad\u7ec3\u76ee\u6807**")
            for sid in range(1,6):
                name = SKILLS[sid]["name"]
                pf = "\u25b6 " if st.session_state.get("train_mode")==sid else "  "
                if st.button("{}{}. {}".format(pf,sid,name), key="m{}".format(sid), use_container_width=True):
                    st.session_state["train_mode"]=sid; st.rerun()
            st.markdown("---")
            if st.button("\u7ed3\u675f\u8bad\u7ec3\u2192\u603b\u7ed3", key="end_train", use_container_width=True, type="primary"):
                _do_summary()
            on = len(st.session_state.get("observations",[]))
            dn = len(st.session_state.get("coach_dialogue",[]))//2
            st.caption("\u89c2\u5bdf {} | \u5bf9\u8bdd {}".format(on, dn))

    # ===== 欢迎页 =====
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks \u8bfb\u76d8\u8bad\u7ec3\u5668")
        st.markdown("")
        for sid in range(1,6):
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

    # ===== 总结页 =====
    if st.session_state.get("training_summary"):
        st.markdown("## \u8bad\u7ec3\u603b\u7ed3")
        st.markdown(st.session_state["training_summary"])
        if st.button("\u7ee7\u7eed\u8bad\u7ec3", key="resume"):
            st.session_state["training_summary"] = ""; st.rerun()
        return

    # ===== 主布局 =====
    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar", 0)
    if bar >= len(chart_df):
        bar = len(chart_df)-1; st.session_state["current_bar"]=bar
    swings = st.session_state.get("swings",[])
    mode = st.session_state.get("train_mode",1)
    skill = SKILLS[mode]
    strict = st.session_state.get("replay_mode")=="\u4e25\u683c\u6a21\u5f0f"

    # ---- K线图 全宽 ----
    chart = build_chart(chart_df, bar, swings)
    st.plotly_chart(chart, use_container_width=True)

    # ---- OHLC 信息 ----
    cur = chart_df.iloc[bar]
    chg = cur["close"]-cur["open"]
    cc = "up" if chg>=0 else "dn"
    st.markdown('<div class="ohlc">K<b>{}</b> &nbsp; O<b>{:.0f}</b> &nbsp; H<b>{:.0f}</b> &nbsp; L<b>{:.0f}</b> &nbsp; C<b>{:.0f}</b> &nbsp; <span class="{}">{:+.0f}</span></div>'.format(
        bar, cur["open"], cur["high"], cur["low"], cur["close"], cc, chg), unsafe_allow_html=True)

    # ---- Slider + 导航 一行 ----
    nav1, nav2 = st.columns([3, 2], vertical_alignment="center")
    with nav1:
        if strict:
            new_bar = bar
        else:
            new_bar = st.slider("", 0, len(chart_df)-1, bar, key="bsl", label_visibility="collapsed")
        if not strict and new_bar != bar:
            st.session_state["current_bar"]=new_bar; st.rerun()
    with nav2:
        steps = [(-5,"-5","bp5"),(-1,"-1","bp1"),(1,"+1","bn1"),(5,"+5","bn5"),(15,"+15","bn15"),(None,"\u672b","bend")]
        bc = st.columns(len(steps))
        for i,(step,label,key) in enumerate(steps):
            show = (step is not None and (not strict or step>0)) or (step is None and not strict)
            if show:
                if bc[i].button(label, key=key, use_container_width=True):
                    st.session_state["current_bar"] = max(0, min(len(chart_df)-1, bar+step)) if step is not None else len(chart_df)-1
                    st.rerun()

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # ---- 下方区域：左输入 + 右对话 ----
    col_in, col_dlg = st.columns([2, 3], gap="medium")

    with col_in:
        # 训练目标标签
        st.markdown('<span class="skill-tag">{}</span><span class="skill-q">{}</span>'.format(
            skill["name"], skill["question"]), unsafe_allow_html=True)

        # 观察输入
        cnt = st.session_state.get("send_counter",0)
        obs_text = st.text_area("\u4f60\u89c2\u5bdf\u5230\u4e86\u4ec0\u4e48\uff1f", height=80,
                                key="obs_{}".format(cnt),
                                placeholder="\u63cf\u8ff0\u5177\u4f53\u884c\u4e3a\u53d8\u5316...")

        # 操作按钮（2行紧凑）
        r1 = st.columns(2)
        with r1[0]:
            if st.button("\u53d1\u9001\u89c2\u5bdf", key="send_obs", use_container_width=True, type="primary"):
                if obs_text.strip():
                    _send(obs_text.strip(), chart_df, bar, skill)
        with r1[1]:
            if st.button("\u91cd\u7f6e\u5bf9\u8bdd", key="new_round", use_container_width=True):
                st.session_state["coach_dialogue"]=[]; st.rerun()

        r2 = st.columns(2)
        with r2[0]:
            if st.button("\u8bb0\u5fc6\u6d4b\u8bd5", key="btn_mem", use_container_width=True):
                _do_memory(chart_df, bar)
        with r2[1]:
            if st.button("\u627e\u77db\u76fe", key="btn_con", use_container_width=True):
                _do_contra(chart_df, bar, skill)

        # 时间轴（可折叠）
        tl = st.session_state.get("timeline",[])
        with st.expander("\u884c\u4e3a\u53d8\u5316\u8bb0\u5f55 ({})".format(len(tl))):
            for ev in tl[-6:]:
                st.caption("[K{}] {}".format(ev.bar, ev.text))
            tc = st.columns([4,1])
            with tc[0]:
                tli = st.text_input("\u8bb0\u5f55", key="tli", placeholder="\u884c\u4e3a\u53d8\u5316...")
            with tc[1]:
                if st.button("\u8bb0", key="tla"):
                    if tli.strip():
                        st.session_state.setdefault("timeline",[]).append(
                            TimelineEvent(bar=bar, text=tli.strip(), timestamp=datetime.now().strftime("%H:%M:%S")))
                        st.rerun()
            if tl and st.button("\u6e05\u7a7a", key="tlc"):
                st.session_state["timeline"]=[]; st.rerun()

    with col_dlg:
        st.markdown("**\u6559\u7ec3\u5bf9\u8bdd**")
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        dialogue = st.session_state["coach_dialogue"]
        if not dialogue:
            st.caption("\u53d1\u9001\u89c2\u5bdf\uff0c\u6559\u7ec3\u4f1a\u8ffd\u95ee\u3002")
        for msg in dialogue:
            _bubble(msg["role"], msg["content"])
        if dialogue:
            uc = sum(1 for m in dialogue if m["role"]=="user")
            dc = sum(1 for m in dialogue if m["role"]=="assistant")
            st.markdown('<div class="dlg-stat">\u4f60 {} | \u6559\u7ec3 {}</div>'.format(uc,dc), unsafe_allow_html=True)

# =========================================================
# 辅助
# =========================================================
def _send(text, chart_df, bar, skill):
    s = st.session_state
    dlg = s["coach_dialogue"]
    dlg.append({"role":"user","content":text})
    s["observations"].append(Observation(
        skill_id=s.get("train_mode",1), bar=bar, text=text,
        timestamp=datetime.now().strftime("%H:%M:%S")))
    with st.spinner("\u6559\u7ec3\u601d\u8003\u4e2d..."):
        resp = ask_coach(chart_df, bar, skill["name"], dlg)
    dlg.append({"role":"assistant","content":resp})
    s["coach_dialogue"]=dlg
    s["send_counter"]=s.get("send_counter",0)+1
    st.rerun()

def _do_memory(chart_df, bar):
    obs = st.session_state.get("observations",[])
    if len(obs)<3:
        st.warning("\u81f3\u5c11\u89c2\u5bdf3\u6b21"); return
    with st.spinner("\u51fa\u9898\u4e2d..."):
        q = ask_memory_test(chart_df, bar, obs)
    st.session_state["coach_dialogue"].append({"role":"assistant","content":"[\u8bb0\u5fc6\u6d4b\u8bd5] "+q})
    st.rerun()

def _do_contra(chart_df, bar, skill):
    dlg = st.session_state["coach_dialogue"]
    if len(dlg)<4:
        st.warning("\u81f3\u5c11\u5bf9\u8bdd2\u8f6e"); return
    with st.spinner("\u5206\u6790\u4e2d..."):
        q = ask_contradiction(chart_df, bar, skill["name"], dlg)
    st.session_state["coach_dialogue"].append({"role":"assistant","content":q})
    st.rerun()

def _do_load(symbol):
    with st.spinner("\u52a0\u8f7d\u4e2d..."):
        seed = random.randint(0,999999)
        df = load_data(symbol, seed=seed)
        if df is not None and len(df)>0:
            sw = detect_swings(df)
            st.session_state.update({
                "chart_df":df,"swings":sw,"current_bar":min(40,len(df)-1),
                "data_loaded":True,"observations":[],"timeline":[],
                "train_mode":1,"coach_dialogue":[],"training_summary":"","send_counter":0})
            st.success("{} \u6839K\u7ebf".format(len(df)))
        else:
            st.error("\u52a0\u8f7d\u5931\u8d25")

def _do_summary():
    s = st.session_state
    if not s.get("observations"):
        st.warning("\u8fd8\u6ca1\u6709\u89c2\u5bdf\u8bb0\u5f55"); return
    with st.spinner("\u751f\u6210\u603b\u7ed3..."):
        s["training_summary"] = ask_summary(s["chart_df"], s["observations"], s["coach_dialogue"])

if __name__ == "__main__":
    main()
