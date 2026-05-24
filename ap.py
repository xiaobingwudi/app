# Al Brooks 读盘训练器 V16
# =========================================================
# 用户 = 训练者 | GPT = 教练 | 软件 = 训练场
# 布局原则：一屏内所有内容可见，零滚动
# =========================================================

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass

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

你不是交易员。你不是分析师。你不是预测模型。

你唯一职责：帮助用户训练5项核心能力：
1. 背景阅读 2. 控制权识别 3. 推进质量判断
4. 区分正常回调与真正转换 5. 理解市场是否接受新价格

【核心原则】
真正获得能力的人只能是用户。
你永远不能替用户：观察、推理、下结论、判断市场。
你只能：引导、追问、纠偏、强迫用户回到具体K线行为。

【严格禁止】
- 告诉用户市场方向、趋势/区间/反转、谁控制市场
- 告诉用户判断是否正确、给交易建议、预测后续走势
- 替用户总结市场结论

【最重要规则】
当用户使用抽象词（转强、转弱、趋势、反转、突破、控制、接受、拒绝、多头、空头、强势、弱势），
你绝对不能围绕这些词讨论。必须强制用户回到具体K线行为：
哪几根K线？行为从哪里开始变化？后续有没有跟进？
对手有没有回应？重叠有没有增加？收盘位置有没有变化？

【你的职责】
1. 强迫用户引用具体K线（从哪几根开始？哪一段？）
2. 强迫用户描述行为（实体变化、收盘位置、高低点、重叠、跟进、对手回应）
3. 强迫用户提供依据（"依据是什么？"）
4. 强迫用户面对矛盾
5. 强迫用户观察连续性（不要只看单根K线/单个形态）

【回答风格】简短、直接、一次只推进一步、不长篇解释。
如果用户开始下定义/猜趋势/猜方向，必须立即拉回"具体发生了什么行为？"
这是你的最高优先级。
"""

# =========================================================
# 样式
# =========================================================
def _css():
    st.markdown("""<style>
/* 全局：去掉多余padding，最大化内容区 */
.main .block-container{padding-top:1rem!important;padding-bottom:0!important;padding-left:1rem!important;padding-right:1rem!important}

/* 侧栏 */
[data-testid="stSidebar"]{width:220px!important;min-width:220px!important;background:#f7f8fa}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]>div{padding-top:2px!important;padding-bottom:2px!important}
[data-testid="stSidebar"] h1{font-size:1.15rem!important;margin:0!important}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] p{font-size:.9rem!important;line-height:1.4!important;margin:0!important}
[data-testid="stSidebar"] .stTextInput>div>div>input{font-size:.92rem!important;padding:.2rem .5rem!important;height:32px!important}
[data-testid="stSidebar"] .stRadio>div>label>div>span{font-size:.9rem!important}
[data-testid="stSidebar"] .stRadio>div>label>p{font-size:.78rem!important}
[data-testid="stSidebar"] .stMarkdown{font-size:.9rem!important;margin:0!important}
[data-testid="stSidebar"] .stButton>button{font-size:.88rem!important;padding:.2rem .3rem!important;margin:0!important;height:30px!important;line-height:1!important}
[data-testid="stSidebar"] .stHorizontalBlock{gap:5px!important}
[data-testid="stSidebar"] .stHorizontalBlock>div>div{gap:5px!important}
[data-testid="stSidebar"] hr{margin:4px 0!important}
[data-testid="stSidebar"] .stCheckbox>label{font-size:.9rem!important}

/* 全局按钮：小巧紧凑 */
.stButton>button{border-radius:4px!important;font-size:.75rem!important;padding:.1rem .4rem!important;border:1px solid #d0d7e3!important;height:26px!important;line-height:1!important}
.stButton>button:hover{border-color:#89b4fa!important}
.stButton>button[data-testid="stBaseButton-primary"]{background:#89b4fa!important;color:#1e1e2e!important;border:none!important;font-weight:600}
div[data-testid="stHorizontalBlock"]>div>div{gap:3px!important}
[data-testid="stHorizontalBlock"]{gap:3px!important}

/* 输入框 */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{border-radius:4px!important;border:1px solid #d0d7e3!important;font-size:.8rem!important}

/* 对话气泡 */
.bu{background:#dce8ff;color:#1e1e2e;padding:4px 7px;border-radius:7px 7px 1px 7px;margin:1px 0;font-size:.8rem;max-width:98%;display:inline-block;font-weight:500;line-height:1.3}
.bc{background:#f4f4f6;color:#313244;padding:4px 7px;border-radius:7px 7px 7px 1px;margin:1px 0;font-size:.8rem;max-width:98%;display:inline-block;border-left:2px solid #89b4fa;line-height:1.3}
.lu{font-size:.6rem;color:#6c7086;margin:4px 0 0;font-weight:600}
.lc{font-size:.6rem;color:#89b4fa;margin:4px 0 0;font-weight:600}
.ds{font-size:.6rem;color:#9399b2;text-align:right;margin-top:3px}

/* Expander */
.streamlit-expanderHeader{font-size:.78rem!important;font-weight:600!important;padding:2px 0!important}
[data-testid="stExpander"]>div>div{font-size:.75rem!important}
[data-testid="stExpander"] details{margin:0!important}

/* OHLC */
.ohlc{font-size:.75rem;color:#6c7086;font-weight:600}
.ohlc b{color:#313244}
.ohlc .up{color:#27ae60}
.ohlc .dn{color:#e74c3c}

/* 标签 */
.stag{display:inline-block;background:#eef2ff;color:#4a6fa5;padding:1px 7px;border-radius:8px;font-size:.8rem;font-weight:600}
.sq{font-size:.75rem;color:#6c7086;margin-left:4px}

/* 分隔 */
.sep{border:none;border-top:1px solid #e8ecf2;margin:3px 0}

/* 隐藏Streamlit自带间距 */
.stApp>header{visibility:hidden}
div[data-testid="stStatusWidget"]{visibility:hidden}
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
# 图表（紧凑）
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
                showarrow=False, font=dict(size=8, color=c),
                xanchor="center", yshift=12 if is_sh else -12))
    cur = chart_df.iloc[bar]
    ann.append(dict(x=bar, y=cur["high"], text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#9399b2",
        font=dict(size=8, color="#6c7086"), ax=0, ay=22))
    fig.update_layout(annotations=ann, height=600,
        margin=dict(l=12, r=55, t=5, b=2),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#eff1f5", zeroline=False,
                   tickfont=dict(size=9), side="right"),
        template="plotly_white",
        font=dict(family="system-ui,sans-serif"))
    return fig

# =========================================================
# GPT
# =========================================================
def _market_msg(chart_df, bar, skill_name):
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

def _gpt(messages):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")
    for a in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-mini", messages=messages,
                temperature=0.4, max_tokens=400)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if a<2 and "429" in str(e):
                time.sleep(2**(a+1)); continue
            return "AI\u8c03\u7528\u5931\u8d25: {}".format(e)

def ask_coach(chart_df, bar, skill_name, dialogue, extra=None):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_market_msg(chart_df, bar, skill_name)}]
    for m in dialogue: msgs.append({"role":m["role"],"content":m["content"]})
    if extra: msgs.append({"role":"user","content":extra})
    return _gpt(msgs)

def ask_summary(chart_df, observations, dialogue):
    ot = "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations)
    dt = "\n".join("{}: {}".format("\u7528\u6237" if m["role"]=="user" else "\u6559\u7ec3",m["content"]) for m in dialogue[-20:])
    return _gpt([{"role":"system","content":AI_SYSTEM_PROMPT},{"role":"user","content":(
        "\u4ee5\u4e0b\u662f\u7528\u6237\u672c\u6b21\u8bad\u7ec3\u7684\u5168\u90e8\u89c2\u5bdf\u548c\u6559\u7ec3\u5bf9\u8bdd\u3002\n\n"
        "\u3010\u89c2\u5bdf\u3011\n{}\n\n\u3010\u5bf9\u8bdd\u3011\n{}\n\n"
        "1.\u7528\u6237\u957f\u671f\u95ee\u9898\uff08\u884c\u4e3a\u5c42\u9762\uff09 2.\u4e60\u60ef\u6027\u9519\u8bef\uff08\u5f15\u7528\u5b9e\u9645\u8868\u73b0\uff09 3.\u4e0b\u9636\u6bb5\u8bad\u7ec3\u91cd\u70b9"
    ).format(ot,dt)}])

def ask_memory_test(chart_df, bar, observations):
    return _gpt([{"role":"system","content":AI_SYSTEM_PROMPT},{"role":"user","content":(
        "\u5ef6\u8fdf\u8bb0\u5fc6\u8bad\u7ec3\u3002\u76d8\u9762\uff1a\n{}\n\n\u89c2\u5bdf\uff1a\n{}\n\n"
        "\u51fa1-2\u4e2a\u8bb0\u5fc6\u6d4b\u8bd5\u95ee\u9898\uff0c\u5177\u4f53\u5230K\u7ebf\u884c\u4e3a\u3002\u53ea\u95ee\u4e0d\u7b54\u3002"
    ).format(_market_msg(chart_df,bar,""),
             "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations[-10:]))}])

def ask_contradiction(chart_df, bar, skill_name, dialogue):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_market_msg(chart_df,bar,skill_name)}]
    for m in dialogue: msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":"\u627e\u51fa\u7528\u6237\u89c2\u5bdf\u4e2d\u7684\u77db\u76fe\u3002\u7528\u63d0\u95ee\u8ba9\u7528\u6237\u81ea\u5df1\u53d1\u73b0\u3002"})
    return _gpt(msgs)

# =========================================================
# 主程序
# =========================================================
def main():
    _css()
    for k, d in [("data_loaded",False),("observations",[]),("train_mode",1),
                  ("timeline",[]),("replay_mode","\u590d\u76d8\u6a21\u5f0f"),
                  ("coach_dialogue",[]),("send_counter",0),("training_summary","")]:
        if k not in st.session_state: st.session_state[k] = d

    # ========== 侧栏 ==========
    with st.sidebar:
        st.title("\u8bfb\u76d8\u8bad\u7ec3\u5668")
        symbol = st.text_input("\u5408\u7ea6", value="rb2510", key="sym")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("\u52a0\u8f7d",key="ld",use_container_width=True): _do_load(symbol)
        with c2:
            if st.button("\u6362\u6bb5",key="rn",use_container_width=True): _do_load(symbol)
        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.session_state["replay_mode"] = st.radio(
                "Replay",["\u590d\u76d8\u6a21\u5f0f","\u4e25\u683c\u6a21\u5f0f"],
                key="rmr",captions=["\u53ef\u56de\u9000","\u53ea+1"])
            st.markdown("---")
            st.markdown("**\u76ee\u6807**")
            for sid in range(1,6):
                nm = SKILLS[sid]["name"]
                pf = "\u25b6 " if st.session_state.get("train_mode")==sid else "  "
                if st.button("{}{}. {}".format(pf,sid,nm),key="m{}".format(sid),use_container_width=True):
                    st.session_state["train_mode"]=sid; st.rerun()
            st.markdown("---")
            if st.button("\u7ed3\u675f\u8bad\u7ec3\u2192\u603b\u7ed3",key="end",use_container_width=True,type="primary"):
                _do_summary()
            st.caption("\u89c2\u5bdf {} | \u5bf9\u8bdd {}".format(
                len(st.session_state.get("observations",[])),
                len(st.session_state.get("coach_dialogue",[]))//2))

    # ========== 欢迎页 ==========
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks \u8bfb\u76d8\u8bad\u7ec3\u5668")
        for sid in range(1,6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** \u2014 {}".format(sid,s["name"],s["question"]))
        st.markdown("> \u4f60\u770b\u56fe\u3002\u4f60\u89c2\u5bdf\u3002\u6559\u7ec3\u53ea\u63d0\u95ee\uff0c\u4e0d\u7ed9\u7b54\u6848\u3002")
        return

    # ========== 总结页 ==========
    if st.session_state.get("training_summary"):
        st.markdown("## \u8bad\u7ec3\u603b\u7ed3")
        st.markdown(st.session_state["training_summary"])
        if st.button("\u7ee7\u7eed\u8bad\u7ec3",key="res"):
            st.session_state["training_summary"]=""; st.rerun()
        return

    # ========== 主界面 ==========
    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar",0)
    if bar >= len(chart_df):
        bar = len(chart_df)-1; st.session_state["current_bar"]=bar
    swings = st.session_state.get("swings",[])
    skill = SKILLS[st.session_state.get("train_mode",1)]
    strict = st.session_state.get("replay_mode")=="\u4e25\u683c\u6a21\u5f0f"

    # ---- 第1行：图表（全宽，紧凑）----
    chart = build_chart(chart_df, bar, swings)
    st.plotly_chart(chart, use_container_width=True)

    # ---- 第2行：OHLC + slider + 导航按钮 全部一行 ----
    cur = chart_df.iloc[bar]
    chg = cur["close"]-cur["open"]
    cc = "up" if chg>=0 else "dn"
    ohlc = '<span class="ohlc">K<b>{}</b> O<b>{:.0f}</b> H<b>{:.0f}</b> L<b>{:.0f}</b> C<b>{:.0f}</b> <span class="{}">{:+.0f}</span></span>'.format(
        bar,cur["open"],cur["high"],cur["low"],cur["close"],cc,chg)

    c_ohlc, c_sl, c_nav = st.columns([3, 4, 2], vertical_alignment="center")
    with c_ohlc:
        st.markdown(ohlc, unsafe_allow_html=True)
    with c_sl:
        if strict:
            nb = bar
        else:
            nb = st.slider("",0,len(chart_df)-1,bar,key="bsl",label_visibility="collapsed")
        if not strict and nb != bar:
            st.session_state["current_bar"]=nb; st.rerun()
    with c_nav:
        steps = [(-5,"-5","bp5"),(-1,"-1","bp1"),(1,"+1","bn1"),(5,"+5","bn5")]
        nc = st.columns(len(steps))
        for i,(step,label,key) in enumerate(steps):
            show = not strict or step > 0
            if show:
                if nc[i].button(label,key=key,use_container_width=True):
                    st.session_state["current_bar"]=max(0,min(len(chart_df)-1,bar+step)); st.rerun()

    # ---- 第3行：标签 + 输入 + 按钮 一行搞定 ----
    st.markdown('<span class="stag">{}</span><span class="sq">{}</span>'.format(
        skill["name"],skill["question"]), unsafe_allow_html=True)

    c_input, c_btns = st.columns([4, 1], vertical_alignment="bottom")
    with c_input:
        cnt = st.session_state.get("send_counter",0)
        obs_text = st.text_area("\u4f60\u89c2\u5bdf\u5230\u4e86\u4ec0\u4e48\uff1f",height=60,
                                key="obs_{}".format(cnt),
                                placeholder="\u63cf\u8ff0\u5177\u4f53\u884c\u4e3a\u53d8\u5316...",
                                label_visibility="collapsed")
    with c_btns:
        if st.button("\u53d1\u9001",key="send_obs",use_container_width=True,type="primary"):
            if obs_text.strip(): _send(obs_text.strip(),chart_df,bar,skill)

    # 操作按钮一行
    bc = st.columns(4)
    btns = [
        ("\u91cd\u7f6e","new_round", lambda: _reset_dlg()),
        ("\u8bb0\u5fc6\u6d4b\u8bd5","btn_mem", lambda: _do_memory(chart_df,bar)),
        ("\u627e\u77db\u76fe","btn_con", lambda: _do_contra(chart_df,bar,skill)),
        ("\u8bb0\u5f55\u884c\u4e3a","btn_tl", lambda: None),
    ]
    for i,(label,key,fn) in enumerate(btns):
        if bc[i].button(label,key=key,use_container_width=True):
            if fn: fn()

    # ---- 第4行：对话区（全宽，限制高度）----
    dialogue = st.session_state["coach_dialogue"]
    if dialogue:
        # 用container限制高度，内部滚动
        with st.container(height=180):
            for msg in dialogue:
                role = msg["role"]
                lbl = "\u4f60" if role=="user" else "\u6559\u7ec3"
                cls = "bu" if role=="user" else "bc"
                lc = "lu" if role=="user" else "lc"
                safe = msg["content"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                st.markdown('<div class="{}">{}</div><div class="{}">{}</div>'.format(lc,lbl,cls,safe), unsafe_allow_html=True)
            uc = sum(1 for m in dialogue if m["role"]=="user")
            dc = sum(1 for m in dialogue if m["role"]=="assistant")
            st.markdown('<div class="ds">\u4f60 {} | \u6559\u7ec3 {}</div>'.format(uc,dc), unsafe_allow_html=True)
    else:
        st.caption("\u53d1\u9001\u89c2\u5bdf\uff0c\u6559\u7ec3\u4f1a\u8ffd\u95ee\u3002")

    # ---- 时间轴（折叠）----
    tl = st.session_state.get("timeline",[])
    with st.expander("\u884c\u4e3a\u53d8\u5316\u8bb0\u5f55 ({})".format(len(tl))):
        for ev in tl[-5:]:
            st.caption("[K{}] {}".format(ev.bar,ev.text))
        tc = st.columns([5,1])
        with tc[0]:
            tli = st.text_input("\u8bb0\u5f55",key="tli",placeholder="\u63cf\u8ff0\u884c\u4e3a\u53d8\u5316...")
        with tc[1]:
            if st.button("\u8bb0",key="tla"):
                if tli.strip():
                    st.session_state.setdefault("timeline",[]).append(
                        TimelineEvent(bar=bar,text=tli.strip(),timestamp=datetime.now().strftime("%H:%M:%S")))
                    st.rerun()
        if tl and st.button("\u6e05\u7a7a",key="tlc"):
            st.session_state["timeline"]=[]; st.rerun()

# =========================================================
# 辅助
# =========================================================
def _send(text, chart_df, bar, skill):
    s = st.session_state
    dlg = s["coach_dialogue"]
    dlg.append({"role":"user","content":text})
    s["observations"].append(Observation(
        skill_id=s.get("train_mode",1),bar=bar,text=text,
        timestamp=datetime.now().strftime("%H:%M:%S")))
    with st.spinner("\u6559\u7ec3\u601d\u8003\u4e2d..."):
        resp = ask_coach(chart_df,bar,skill["name"],dlg)
    dlg.append({"role":"assistant","content":resp})
    s["coach_dialogue"]=dlg
    s["send_counter"]=s.get("send_counter",0)+1
    st.rerun()

def _reset_dlg():
    st.session_state["coach_dialogue"]=[]; st.rerun()

def _do_memory(chart_df, bar):
    obs = st.session_state.get("observations",[])
    if len(obs)<3: st.warning("\u81f3\u5c11\u89c2\u5bdf3\u6b21"); return
    with st.spinner("\u51fa\u9898\u4e2d..."):
        q = ask_memory_test(chart_df,bar,obs)
    st.session_state["coach_dialogue"].append({"role":"assistant","content":"[\u8bb0\u5fc6\u6d4b\u8bd5] "+q})
    st.rerun()

def _do_contra(chart_df, bar, skill):
    dlg = st.session_state["coach_dialogue"]
    if len(dlg)<4: st.warning("\u81f3\u5c11\u5bf9\u8bdd2\u8f6e"); return
    with st.spinner("\u5206\u6790\u4e2d..."):
        q = ask_contradiction(chart_df,bar,skill["name"],dlg)
    st.session_state["coach_dialogue"].append({"role":"assistant","content":q})
    st.rerun()

def _do_load(symbol):
    with st.spinner("\u52a0\u8f7d\u4e2d..."):
        seed = random.randint(0,999999)
        df = load_data(symbol,seed=seed)
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
    if not s.get("observations"): st.warning("\u8fd8\u6ca1\u6709\u89c2\u5bdf\u8bb0\u5f55"); return
    with st.spinner("\u751f\u6210\u603b\u7ed3..."):
        s["training_summary"] = ask_summary(s["chart_df"],s["observations"],s["coach_dialogue"])

if __name__ == "__main__":
    main()
