# Al Brooks 读盘训练器 V17
# =========================================================
# 用户 = 训练者 | GPT = 教练 | 软件 = 训练场

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
AI_SYSTEM_PROMPT = """你是 Al Brooks 价格行为训练教练。

你会收到当前K线的OHLC数据（最近60根），以及用户正在训练的技能名称。
你已经能看到图表。根据数据，你对这项技能已有自己的判断。

【训练流程 - 严格执行】

第1轮（用户首次作答）：
判断用户的回答是否触及该技能的核心观察维度。
- 如果到位：直接进入第2轮点评流程。
- 如果不到位：给一次提示，提示只指向用户遗漏的具体维度，不给答案。

第2轮（用户二次作答）：
无论用户答得如何，执行以下两步：
1. 对用户的二次作答给出点评（肯定到位的部分，指出仍然缺失的部分）。
2. 亮出你自己对这项技能的判断（基于你看到的K线数据，说清楚你的观察依据）。
然后结束，不再追问。

【各技能的核心观察维度】

技能1 背景阅读：
核心：摆动高低点序列（HH/HL 还是 LH/LL）、整体是趋势还是区间、通道倾斜方向。
提示方向：让用户说清楚高低点的排列方式。
你的判断模板：「背景：[趋势/区间]。依据：bar__到bar__，[高低点序列描述]，[通道/节奏描述]。」

技能2 控制权识别：
核心：当前位置谁在持续推进、对手方有没有得到跟进。
提示方向：让用户回到最近3-5根K线的具体行为。
你的判断模板：「当前控制方：[多/空]。依据：bar__到bar__，[推进描述]，对手方[有/无]跟进。」

技能3 推进质量：
核心：实体大小、K线重叠程度、影线方向、收盘位置、动能是否衰减。
提示方向：让用户描述实体和重叠情况。
你的判断模板：「推进质量：[强/中/弱]。依据：[实体描述]，[重叠描述]，[影线描述]。」

技能4 回调vs转换：
核心：回调了几根K线、对手方实体是否连续出现、有没有得到跟进。
提示方向：让用户数回调K线数量，判断对手方实体质量。
你的判断模板：「判断：[正常回调/控制权转换]。依据：回调[N]根，对手方实体[连续/不连续]，跟进[有/无]。」

技能5 市场接受：
核心：突破后有没有立刻被推回、在新价格区域停留了几根K线、有没有继续跟进买入/卖出。
提示方向：让用户说清楚突破后停留了几根K线。
你的判断模板：「市场[接受/拒绝]新价格。依据：突破后停留[N]根，[有/无]继续跟进，[有/无]立刻推回。」

【约束】
- 只在该技能维度内分析，不跨技能
- 提示只给一次，第2轮必须亮出自己判断
- 你的判断要有具体bar编号作为依据
- 回答简短，不列大纲，不写长篇
"""

TRAIN_LEVEL = {
    1: {"name": "观察阶段", "desc": "允许模糊、整体感觉、通道、节奏、倾向。禁止结构辩论与精确确认。"},
    2: {"name": "行为细化阶段", "desc": "开始关注具体K线行为、推进连续性、重叠程度。"},
    3: {"name": "结构验证阶段", "desc": "允许讨论失败突破、摆动确认、Always In转换、结构争议。"},
}

AI_SUMMARY_PROMPT = """你是训练总结分析师。

你的职责：
分析用户的观察习惯和行为模式。

不要：
- 继续追问
- 像教练一样提问题
- 评判对错

只需：
1. 用户长期问题（行为层面）
2. 习惯性错误（引用实际表现）
3. 下阶段训练重点
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
    bar_nums = []
    for idx in range(0, bar + 1, 5):
        row = chart_df.iloc[idx]
        ny = row["low"] if row["close"] >= row["open"] else row["high"]
        ns = -12 if row["close"] >= row["open"] else 12
        bar_nums.append(dict(x=idx, y=ny, text=str(idx), showarrow=False,
                             font=dict(size=8, color="#9399b2"),
                             xanchor="center", yshift=ns))
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
    """
    给AI提供增强的K线数据
    只添加真正有用的特征，不做过度处理
    """
    # 取最近40根K线（您确认过的合适数量）
    start = max(0, bar - 40)
    
    recent = []
    for i in range(start, bar + 1):
        row = chart_df.iloc[i]
        
        # 原始数据
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        
        # === 计算有用特征 ===
        
        # 1. 实体大小 (绝对值)
        body = abs(c - o)
        
        # 2. 整体波幅
        total_range = h - l
        
        # 3. 实体占比 (判断K线强弱)
        #    0.8以上 = 实体很大，影线很短 -> 强趋势K线
        #    0.3以下 = 实体很小，影线很长 -> 犹豫/反转K线
        body_ratio = body / total_range if total_range > 0 else 0
        
        # 4. 上影线占比 (判断上方压力)
        if c >= o:  # 阳线
            upper_wick = h - c
        else:       # 阴线
            upper_wick = h - o
        upper_ratio = upper_wick / total_range if total_range > 0 else 0
        
        # 5. 下影线占比 (判断下方支撑)
        if c >= o:  # 阳线
            lower_wick = o - l
        else:       # 阴线
            lower_wick = c - l
        lower_ratio = lower_wick / total_range if total_range > 0 else 0
        
        # 6. 相对于前一根收盘价的变化 (判断动量)
        if i > 0:
            prev_c = float(chart_df.iloc[i-1]["close"])
            price_change = c - prev_c
        else:
            price_change = 0
        
        recent.append({
            "bar": i,
            "open": round(o, 1),
            "high": round(h, 1),
            "low": round(l, 1),
            "close": round(c, 1),
            "body_ratio": round(body_ratio, 2),      # 新增：实体占比
            "upper_wick": round(upper_ratio, 2),     # 新增：上影线占比
            "lower_wick": round(lower_ratio, 2),     # 新增：下影线占比
            "change": round(price_change, 1)         # 新增：相对变化
        })
    
    # 额外添加一个简单的整体统计
    closes = [r["close"] for r in recent]
    
    market_msg = {
        "current_bar": bar,
        "skill": skill_name,
        "bars": recent,
        "summary": {
            "high_40": max([r["high"] for r in recent]),      # 40根最高点
            "low_40": min([r["low"] for r in recent]),        # 40根最低点
            "start_price": recent[0]["close"],                # 起始价格
            "end_price": recent[-1]["close"],                 # 当前价格
            "net_change": round(recent[-1]["close"] - recent[0]["close"], 1)  # 整体涨跌
        }
    }
    
    return json.dumps(market_msg, ensure_ascii=False)
    
def _gpt(messages):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")
    for a in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano-2026-03-17", messages=messages,
                temperature=0.2, max_tokens=700)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if a<2 and "429" in str(e):
                time.sleep(2**(a+1)); continue
            return "AI调用失败: {}".format(e)

def ask_coach(chart_df, bar, skill_name, dialogue, level=1):
    system_prompt = AI_SYSTEM_PROMPT
    msgs = [{"role":"system","content":system_prompt},
            {"role":"user","content":_market_msg(chart_df, bar, skill_name)}]
    for m in dialogue[-10:]: msgs.append({"role":m["role"],"content":m["content"]})
    return _gpt(msgs)


def ask_summary(chart_df, observations, dialogue):
    ot = "\n".join("[K{}] {}".format(o.bar,o.text) for o in observations)
    dt = "\n".join("{}: {}".format("用户" if m["role"]=="user" else "教练",m["content"]) for m in dialogue[-40:])
    return _gpt([{"role":"system","content":AI_SUMMARY_PROMPT},{"role":"user","content":(
        "以下是用户本次训练的全部观察记录和教练对话。\n\n"
        "【观察】\n{}\n\n【对话】\n{}\n\n"
        "1.用户长期问题（行为层面） 2.习惯性错误（引用实际表现） 3.下阶段训练重点"
    ).format(ot,dt)}])


def main():
    _page_config()
    _css()

    for k, d in [("data_loaded",False),("observations",[]),("train_mode",1),
                  ("timeline",[]),("replay_mode","严格模式"),
                  ("coach_dialogue",[]),("send_counter",0),("training_summary",""),("skill_round",0),("train_level",1)]:
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
            st.session_state["replay_mode"] = "严格模式"

            st.markdown("---")
            st.markdown("**训练目标**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                pf = "▶ " if st.session_state.get("train_mode") == sid else "  "
                if st.button("{}{}. {}".format(pf, sid, name),
                             key="m{}".format(sid),
                             use_container_width=True):
                    st.session_state["train_mode"] = sid
                    st.session_state["coach_dialogue"] = []
                    st.session_state["skill_round"] = 0
                    st.rerun()

            st.markdown("---")
            st.markdown("**训练阶段**")
            level_opts = {1: "观察阶段", 2: "行为细化", 3: "结构验证"}
            st.selectbox("", list(level_opts.keys()),
                         format_func=lambda k: "{}. {}".format(k, level_opts[k]),
                         key="level_selector")
            st.session_state["train_level"] = st.session_state.get("level_selector", 1)
            
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
        resp = ask_coach(chart_df, bar, skill["name"], dlg, level=s.get("train_level", 1))
    s["skill_round"] += 1

    # 第2轮结束后加结束提示（AI自己的判断已在prompt里要求输出）
    if s["skill_round"] >= 2:
        resp += "\n\n---\n本项技能训练结束，可切换下一项继续。"

    dlg.append({"role": "assistant", "content": resp})
    s["coach_dialogue"] = dlg
    s["send_counter"] = s.get("send_counter", 0) + 1
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
                "coach_dialogue": [], "training_summary": "", "skill_round": 0,
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
