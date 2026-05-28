# Al Brooks 读盘训练器 V17
# =========================================================
# 用户 = 训练者 | GPT = 教练 | 软件 = 训练场
# 本版改动：
# 1. max_tokens 400→700，避免回答截断
# 2. 市场上下文从30根扩展至60根
# 3. 总结对话历史从20轮扩展至40轮
# 4. 删除 ex_map 冗余代码
# 5. OHLC 高点格式 {:.0g} → {:.0f}，修复科学计数法显示bug
# 6. 训练模式切换按钮改为 st.rerun()，即时刷新
# 7. _fetch_raw 去掉 fillna(0)，避免异常K线
# 8. 品种下拉与取值统一用同一排序列表，修复取错品种bug
# =========================================================

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

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

SYMBOL_NAMES = {
    "RB": "螺纹钢", "HC": "热卷", "AU": "黄金", "AG": "白银",
    "CU": "铜", "AL": "铝", "ZN": "锌", "NI": "镍",
    "RU": "橡胶", "BU": "沥青", "FU": "燃油", "SC": "原油",
    "PB": "铅", "SN": "锡", "SS": "不锈钢", "SP": "纸浆",
    "I": "铁矿石", "J": "焦炭", "JM": "焦煤", "A": "豆一",
    "M": "豆粕", "Y": "豆油", "P": "棕榈油", "C": "玉米",
    "L": "塑料", "PP": "PP", "EG": "乙二醇", "EB": "苯乙烯",
    "PG": "LPG", "V": "PVC", "B": "豆二", "JD": "鸡蛋",
    "CF": "棉花", "SR": "白糖", "TA": "PTA", "MA": "甲醇",
    "FG": "玻璃", "SA": "纯碱", "OI": "菜油", "RM": "菜粕",
    "AP": "苹果", "ZC": "动力煤", "SF": "硅铁", "SM": "锰硅",
    "UR": "尿素", "PF": "短纤", "SH": "烧碱", "PX": "对二甲苯",
    "IF": "沪深300", "IC": "中证500", "IM": "中证1000",
    "IH": "上证50", "T": "十债", "TF": "五债", "TS": "两债",
    "SI": "工业硅", "LC": "碳酸锂", "PS": "聚烯烃", "PD": "铂钯",
}

SKILLS = {
    1: {"name": "背景阅读",   "question": "当前市场背景是什么？"},
    2: {"name": "控制权识别", "question": "现在谁在控制市场？"},
    3: {"name": "推进质量",   "question": "最近推进的质量如何？"},
    4: {"name": "回调vs转换", "question": "这是正常回调还是控制权转换？"},
    5: {"name": "市场接受",   "question": "市场是否接受了新价格？"},
}

AI_SYSTEM_PROMPT = """
你是 Al Brooks 价格行为观察训练教练。你的职责不是分析市场，而是观察用户的阅读过程。

【核心原则】
不判断对错。只问："市场具体做了什么？"
如果用户给的是结论、标签、方向预判，把他拉回具体K线行为。
每次只追问一个点，不要一次列多个问题。
回应不超过100字。不给答案，不给引导性选项。

【每个技能的专属追问焦点】

技能1 背景阅读：
- 只关注：整体结构是怎么形成的？是谁建立的背景？
- 追问方向：多头/空头背景从哪根K线开始变明显？用具体bar编号说话。
- 禁止：不要追问控制权、推进质量、回调质量，那是其他技能的事。

技能2 控制权识别：
- 只关注：当前这个位置，谁在主导？证据是什么？
- 追问方向：最近3-5根K线，哪一方行为更强？有没有对手方反抗？反抗有没有跟进？
- 禁止：不要追问背景怎么形成的，只看当前位置。

技能3 推进质量：
- 只关注：最近一段推进，力度如何？
- 追问方向：实体大小、重叠程度、影线方向、收盘位置。强还是弱？为什么？
- 禁止：不要追问背景，不要追问谁控制，只看这段推进本身的质量。

技能4 回调vs转换：
- 只关注：当前的回调，是暂停还是反转信号？
- 追问方向：回调了几根K线？空头实体有没有连续出现？对手方有没有得到跟进？
- 禁止：不要重新讨论背景，聚焦在回调这个事件本身。

技能5 市场接受：
- 只关注：价格到了新区域，市场有没有留在那里？
- 追问方向：突破后有没有立刻被推回？在新价格区域停留了几根K线？有没有继续跟进？
- 禁止：不要讨论谁控制，只看新价格有没有被接受。

【你会收到的数据】
每次对话开始时，你会收到一条包含当前盘面OHLC数据的消息，以及用户正在训练的技能名称。
请严格按照该技能的追问焦点回应，不要跨技能混答。
"""

# =========================================================
# 样式
# =========================================================
def _page_config():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def _css():
    st.markdown("""
    <style>
    /* 全局重置与字体优化 */
    html, body, [class*="css"] { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        scrollbar-width: thin;
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        width: 220px !important;
        min-width: 220px !important;
        background: #f7f8fa !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        padding-top: 1px !important;
        padding-bottom: 1px !important;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #313244 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea {
        border-radius: 4px !important;
        border: 1px solid #d0d7e3 !important;
        font-size: 0.85rem !important;
        padding: 0.15rem 0.4rem !important;
        height: 28px !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 4px !important;
        font-size: 0.82rem !important;
        padding: 0.15rem 0.2rem !important;
        margin: 0 !important;
        height: 28px !important;
        line-height: 1 !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 4px 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > div > span {
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > p {
        font-size: 0.72rem !important;
    }
    
    /* 主按钮主题色（Al Brooks 风格红） */
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #c0392b !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
    }
    
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #a03023 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(192, 57, 43, 0.3);
    }
    
    /* 对话气泡美化 */
    .bu, .bc {
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bu {
        background: #e3f2fd;
        color: #0d47a1;
        border-bottom-right-radius: 4px !important;
    }
    
    .bc {
        background: #f0f0f0;
        color: #333333;
        border-bottom-left-radius: 4px !important;
    }
    
    /* OHLC 信息栏 */
    .ohlc {
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    
    .ohlc .up { color: #27ae60; }
    .ohlc .dn { color: #e74c3c; }
    
    /* 选项卡样式 */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #333333;
    }
    
    /* --- 图表容器强化 --- */
    .main .block-container {
        max-width: 100% !important;
        padding-top: 0.1rem !important;
        padding-bottom: 0.1rem !important;
        padding-right: 0.3rem !important;
        padding-left: 0.3rem !important;
        display: flex;
        flex-direction: column;
    }

    .js-plotly-plot, .plotly-graph-div {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-width: 100% !important;
    }

    /* --- 顶部状态栏优化 --- */
    .ohlc {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.9rem;
    }

    /* 导航按钮微调 */
    .stButton>button {
        min-width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
    }

    /* Plotly 图表容器 */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_raw(symbol):
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period="30")
        df = df.rename(columns={"datetime":"datetime","open":"open","high":"high","low":"low","close":"close"})
        df = df.reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"])
        return df.reset_index(drop=True)
    except Exception:
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
            sym = "▲" if is_sh else "▼"
            ann.append(dict(x=s.index, y=s.price,
                text="{} {:.0f}".format(sym, s.price),
                showarrow=False, font=dict(size=9, color=c),
                xanchor="center", yshift=14 if is_sh else -14))
    cur = chart_df.iloc[bar]
    ann.append(dict(x=bar, y=cur["high"], text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#9399b2",
        font=dict(size=9, color="#6c7086"), ax=0, ay=25))
    # K线编号（每5根显示）
    bar_nums = [dict(x=idx, y=cur["low"] if chart_df.iloc[idx]["close"] >= chart_df.iloc[idx]["open"] else chart_df.iloc[idx]["high"],
                     text=str(idx), showarrow=False,
                     font=dict(size=8, color="#9399b2"),
                     xanchor="center", yshift=-12 if chart_df.iloc[idx]["close"] >= chart_df.iloc[idx]["open"] else 12)
                for idx in range(0, bar + 1, 5)]
    ann.extend(bar_nums)
    fig.update_layout(annotations=ann, height=330,
        margin=dict(l=40, r=60, t=10, b=5),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10), showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#eff1f5", zeroline=False,
                   tickfont=dict(size=10), side="right"),
        template="plotly_white",
        font=dict(family="system-ui,sans-serif"))
    return fig

# =========================================================
# GPT
# =========================================================
def _market_msg(chart_df, bar, skill_name):
    start = max(0, bar - 60)
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
    client = OpenAI(api_key=api_key, base_url="https://www.right.codes/codex/v1")
    for a in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.5", messages=messages,
                temperature=0.4, max_tokens=700)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if a<2 and "429" in str(e):
                time.sleep(2**(a+1)); continue
            return "AI调用失败: {}".format(e)

def ask_coach(chart_df, bar, skill_name, dialogue, extra=None):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_market_msg(chart_df, bar, skill_name)}]
    for m in dialogue: msgs.append({"role":m["role"],"content":m["content"]})
    if extra: msgs.append({"role":"user","content":extra})
    return _gpt(msgs)

def ask_summary(chart_df, observations, dialogue):
    ot = "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations)
    dt = "\n".join("{}: {}".format("用户" if m["role"]=="user" else "教练",m["content"]) for m in dialogue[-40:])
    return _gpt([{"role":"system","content":AI_SYSTEM_PROMPT},{"role":"user","content":(
        "以下是用户本次训练的全部观察和教练对话。\n\n"
        "【观察】\n{}\n\n【对话】\n{}\n\n"
        "1.用户长期问题（行为层面） 2.习惯性错误（引用实际表现） 3.下阶段训练重点"
    ).format(ot,dt)}])

def ask_memory_test(chart_df, bar, observations):
    return _gpt([{"role":"system","content":AI_SYSTEM_PROMPT},{"role":"user","content":(
        "延迟记忆训练。盘面：\n{}\n\n观察：\n{}\n\n"
        "出1-2个记忆测试问题，具体到K线行为。只问不答。"
    ).format(_market_msg(chart_df,bar,""),
             "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations[-10:]))}])

def ask_contradiction(chart_df, bar, skill_name, dialogue):
    msgs = [{"role":"system","content":AI_SYSTEM_PROMPT},
            {"role":"user","content":_market_msg(chart_df,bar,skill_name)}]
    for m in dialogue: msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":"找出用户观察中的矛盾。用提问让用户自己发现。"})
    return _gpt(msgs)

# =========================================================
# 主程序
# =========================================================
def main():
    _page_config()
    _css()

    for k, d in [("data_loaded",False),("observations",[]),("train_mode",1),
                  ("timeline",[]),("replay_mode","复盘模式"),
                  ("coach_dialogue",[]),("send_counter",0),("training_summary","")]:
        if k not in st.session_state: st.session_state[k] = d

    # ========== 侧栏 ==========
    with st.sidebar:
        st.title("读盘训练器")

        if "main_contracts" not in st.session_state:
            st.session_state["main_contracts"] = {}
        mc = st.session_state["main_contracts"]
        if not mc:
            with st.spinner("获取主力合约..."):
                _load_all_main_contracts(mc)
        if mc:
            sorted_items = sorted(mc.items())
            labels = ["{} ({})".format(SYMBOL_NAMES.get(code, code), sym) for code, sym in sorted_items]
            sym_idx = st.selectbox("品种", range(len(labels)),
                                   format_func=lambda i: labels[i])
            sym_code, sym_main = sorted_items[sym_idx]
        else:
            st.warning("主力合约获取失败，请刷新重试")
            return
        c1, c2 = st.columns(2)
        with c1:
            if st.button("加载", key="ld", use_container_width=True):
                _do_load(sym_code, sym_main)
        with c2:
            if st.button("换一段", key="rn", use_container_width=True):
                _do_load(sym_code, sym_main)

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.session_state["replay_mode"] = st.radio(
                "Replay", [ "严格模式"],
                key="rmr", captions=[ "只能+1"])

            st.markdown("---")
            st.markdown("**训练目标**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                pf = "▶ " if st.session_state.get("train_mode") == sid else "  "
                if st.button("{}{}. {}".format(pf, sid, name),
                             key="m{}".format(sid),
                             use_container_width=True):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            st.markdown("---")
            if st.button("结束训练 → 总结",
                         key="end", use_container_width=True, type="primary"):
                _do_summary()

            on = len(st.session_state.get("observations", []))
            dn = len(st.session_state.get("coach_dialogue", [])) // 2
            st.caption("观察 {} 次  |  对话 {} 轮".format(on, dn))

    # ========== 欢迎页 ==========
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器")
        st.markdown("")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** — {}".format(sid, s["name"], s["question"]))
        st.markdown("")
        st.markdown("> 你看图。你观察。教练只提问，不给答案。")
        st.markdown("")
        st.markdown("**训练架构：**")
        st.markdown("- 用户 = 真正训练者")
        st.markdown("- GPT = 教练（与你看同一个盘面）")
        st.markdown("- 软件 = 训练场")
        return

    # ========== 总结页 ==========
    if st.session_state.get("training_summary"):
        st.markdown("## 训练总结")
        st.markdown(st.session_state["training_summary"])
        if st.button("继续训练", key="res"):
            st.session_state["training_summary"] = ""
            st.rerun()
        return

    # ========== 主界面 ==========
    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar", 0)
    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    swings = st.session_state.get("swings", [])
    skill = SKILLS[st.session_state.get("train_mode", 1)]
    strict = st.session_state.get("replay_mode") == "严格模式"

    # ===== 图表（全宽）=====
    chart = build_chart(chart_df, bar, swings)
    st.plotly_chart(
    chart, 
    use_container_width=True, 
    config={'displayModeBar': False},
    key="main_chart"
)

    # ===== OHLC + Slider + 导航（一行）=====
    cur = chart_df.iloc[bar]
    chg = cur["close"] - cur["open"]
    cc = "up" if chg >= 0 else "dn"
    ohlc = (
        '<span class="ohlc">'
        '<b>K{}</b> | '
        'O<b>{:.0f}</b> '
        'H<b>{:.0f}</b> '
        'L<b>{:.0f}</b> '
        'C<b>{:.0f}</b> '
        '<span class="{}">{:+.0f}</span>'
        '</span>'
    ).format(bar, cur["open"], cur["high"], cur["low"], cur["close"], cc, chg)

    c_info, c_sl, c_nav = st.columns([3, 2, 1], vertical_alignment="center")
    with c_info:
        st.markdown(ohlc, unsafe_allow_html=True)
    with c_sl:
        st.markdown("K{} / {}".format(bar, len(chart_df) - 1))
    with c_nav:
        def _on_next_bar():
            st.session_state["current_bar"] = min(len(chart_df) - 1, st.session_state["current_bar"] + 1)
        st.button("下一根", key="bn1", on_click=_on_next_bar, use_container_width=True)

    # ===== Tab 分组 =====
    tab_train, tab_dlg, tab_tl = st.tabs([
        "训练场",
        "教练对话 ({})".format(len(st.session_state.get("coach_dialogue", []))),
        "行为记录 ({})".format(len(st.session_state.get("timeline", []))),
    ])

    # ---- Tab 1: 训练场 ----
    with tab_train:
        st.markdown('<span class="stag">{}</span><span class="sq">{}</span>'.format(
            skill["name"], skill["question"]), unsafe_allow_html=True)

        cnt = st.session_state.get("send_counter", 0)
        obs_text = st.text_area(
            "你观察到了什么？", height=80,
            key="obs_{}".format(cnt),
            placeholder=skill.get("hints", ""))

        bc = st.columns(2)
        with bc[0]:
            if st.button("发送观察", key="send_obs",
                         use_container_width=True, type="primary"):
                if obs_text.strip():
                    _send(obs_text.strip(), chart_df, bar, skill)
        with bc[1]:
            if st.button("重置对话", key="new_round",
                         use_container_width=True):
                st.session_state["coach_dialogue"] = []
                st.rerun()

        # 最近一轮对话预览
        dialogue = st.session_state["coach_dialogue"]
        if dialogue:
            st.markdown("---")
            last = dialogue[-1]
            role = "教练" if last["role"] == "assistant" else "你"
            cls = "bc" if last["role"] == "assistant" else "bu"
            safe = last["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown('<div class="lc">{}</div><div class="{}">{}</div>'.format(role, cls, safe), unsafe_allow_html=True)

    # ---- Tab 2: 教练对话 ----
    with tab_dlg:
        dialogue = st.session_state["coach_dialogue"]
        if not dialogue:
            st.caption("发送观察后，教练会在这里追问。")
        for msg in dialogue:
            role = msg["role"]
            lbl = "你" if role == "user" else "教练"
            cls = "bu" if role == "user" else "bc"
            lc = "lu" if role == "user" else "lc"
            safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown(
                '<div class="{}">{}</div><div class="{}">{}</div>'.format(lc, lbl, cls, safe),
                unsafe_allow_html=True)
        if dialogue:
            uc = sum(1 for m in dialogue if m["role"] == "user")
            dc = sum(1 for m in dialogue if m["role"] == "assistant")
            st.markdown('<div class="ds">你 {} 次 | 教练 {} 次</div>'.format(uc, dc), unsafe_allow_html=True)

    # ---- Tab 3: 行为记录 ----
    with tab_tl:
        tl = st.session_state.get("timeline", [])
        if tl:
            for ev in tl:
                st.caption("[K{}] {}".format(ev.bar, ev.text))
        else:
            st.caption("在训练过程中记录你观察到的行为变化。")
        st.markdown("---")
        tc = st.columns([5, 1])
        with tc[0]:
            tli = st.text_input("记录", key="tli",
                                placeholder="描述行为变化...")
        with tc[1]:
            if st.button("记", key="tla"):
                if tli.strip():
                    st.session_state.setdefault("timeline", []).append(
                        TimelineEvent(bar=bar, text=tli.strip(),
                                     timestamp=datetime.now().strftime("%H:%M:%S")))
                    st.rerun()
        if tl and st.button("清空", key="tlc"):
            st.session_state["timeline"] = []
            st.rerun()


# =========================================================
# 辅助函数
# =========================================================
def _send(text, chart_df, bar, skill):
    s = st.session_state
    dlg = s["coach_dialogue"]
    dlg.append({"role": "user", "content": text})
    s["observations"].append(Observation(
        skill_id=s.get("train_mode", 1), bar=bar, text=text,
        timestamp=datetime.now().strftime("%H:%M:%S")))
    with st.spinner("教练思考中..."):
        resp = ask_coach(chart_df, bar, skill["name"], dlg)
    dlg.append({"role": "assistant", "content": resp})
    s["coach_dialogue"] = dlg
    s["send_counter"] = s.get("send_counter", 0) + 1
    st.rerun()


def _do_memory(chart_df, bar):
    obs = st.session_state.get("observations", [])
    if len(obs) < 3:
        st.warning("至少观察 3 次后可用")
        return
    with st.spinner("出题中..."):
        q = ask_memory_test(chart_df, bar, obs)
    st.session_state["coach_dialogue"].append(
        {"role": "assistant", "content": "[记忆测试] " + q})
    st.rerun()


def _do_contra(chart_df, bar, skill):
    dlg = st.session_state["coach_dialogue"]
    if len(dlg) < 4:
        st.warning("至少对话 2 轮后可用")
        return
    with st.spinner("分析中..."):
        q = ask_contradiction(chart_df, bar, skill["name"], dlg)
    st.session_state["coach_dialogue"].append(
        {"role": "assistant", "content": q})
    st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_all_contracts():
    """并发请求5个交易所主力合约，结果缓存1小时"""
    def _fetch_one(ex):
        try:
            result = ak.match_main_contract(symbol=ex)
            return str(result).split(",")
        except Exception:
            return []

    mc = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, ex): ex
                   for ex in ["shfe", "dce", "czce", "cffex", "gfex"]}
        for future in as_completed(futures):
            for c in future.result():
                c = c.strip()
                if len(c) < 3:
                    continue
                code = "".join(ch for ch in c[:4] if ch.isalpha()).upper()
                if code in SYMBOL_NAMES and code not in mc:
                    mc[code] = c
    return mc


def _load_all_main_contracts(mc):
    """从缓存函数取结果，写入 session_state 的 mc 字典"""
    result = _fetch_all_contracts()
    mc.update(result)

def _do_load(sym_code, sym_main):
    with st.spinner("加载中..."):
        seed = random.randint(0, 999999)
        df = load_data(sym_main, seed=seed)
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
        else:
            st.error("加载失败")


def _do_summary():
    s = st.session_state
    if not s.get("observations"):
        st.warning("还没有观察记录")
        return
    with st.spinner("生成总结..."):
        s["training_summary"] = ask_summary(
            s["chart_df"], s["observations"], s["coach_dialogue"])


if __name__ == "__main__":
    main()
