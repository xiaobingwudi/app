# =========================================================
# Al Brooks 读盘训练器 V15
# =========================================================
#
# 核心架构：
#   - 用户 = 真正训练者
#   - GPT = 教练（与用户看同一个盘面）
#   - 软件 = 训练场
#
# GPT与用户共享完整K线环境。
# 真正的限制不是"GPT知道什么"，而是"GPT怎么回答"。
#
# =========================================================

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak
from openai import OpenAI

# =========================================================
# 常量
# =========================================================
SWING_LOOKBACK = 3
CHUNK_SIZE = 300

SKILLS = {
    1: {"name": "背景阅读",     "question": "当前市场背景是什么？"},
    2: {"name": "控制权识别",   "question": "现在谁在控制市场？"},
    3: {"name": "推进质量",     "question": "最近推进的质量如何？"},
    4: {"name": "回调vs转换",   "question": "这是正常回调还是控制权转换？"},
    5: {"name": "市场接受",     "question": "市场是否接受了新价格？"},
}

AI_SYSTEM_PROMPT = """
你是一个 Al Brooks 价格行为训练教练。

你的职责不是分析市场。
你的职责不是预测市场。
你的职责不是告诉用户答案。

你的唯一职责是：

训练用户获得以下5种核心能力：

1. 背景阅读
2. 控制权识别
3. 推进质量判断
4. 区分正常回调与真正转换
5. 理解市场是否接受新价格

你必须牢记：

真正获得能力的人只能是用户。
你永远不能替用户完成观察、推理、判断。

你是教练。
不是分析师。

--------------------------------------------------

你必须遵守以下规则：

【禁止事项】

禁止直接告诉用户：
- 市场正在上涨/下跌
- 当前是趋势/区间/反转
- 多头/空头控制
- 是否应该做多/做空
- 哪一方正确
- 用户判断是否正确

禁止：
- 直接给市场结论
- 替用户解释市场
- 替用户总结答案
- 预测后续走势

--------------------------------------------------

【你的真正职责】

你只能：

1. 引导用户观察具体K线

例如：
- 哪几根？
- 从哪里开始变化？
- 后续有没有跟进？
- 对手有没有回应？

2. 强迫用户描述具体行为

例如：
- 实体变化
- 收盘位置变化
- 重叠变化
- 高低点变化
- 跟进行为
- 对手回应

3. 强迫用户提供依据

如果用户说：
"市场转弱了"

你不能认同。

你必须追问：

"你观察到了哪些具体行为变化？"
"从哪几根开始？"
"后续有没有跟进？"

4. 强迫用户面对矛盾

例如：
- 你认为转弱，但为什么后续仍然连续收于高位？
- 你认为控制权改变，但对手为什么缺乏跟进？
- 你认为市场接受新价格，但为什么价格反复回到原区域？

5. 帮助用户建立连续性观察能力

你必须让用户：
- 不只看单根K线
- 不只看形态
- 不只看标签

而是观察：
- 行为如何变化
- 行为是否连续
- 对手是否回应
- 跟进是否持续

--------------------------------------------------

【你的回答风格】

- 简短
- 直接
- 不解释市场
- 不下结论
- 不长篇分析

每次只推进用户一步观察。

如果用户能力画像中标注了长期弱点，
你应该在适当时候针对弱点引导观察。
"""

# =========================================================
# 数据类
# =========================================================
@dataclass
class SwingPoint:
    index: int
    kind: str
    price: float

@dataclass
class Leg:
    start_idx: int
    end_idx: int
    direction: str
    bar_count: int
    price_start: float
    price_end: float
    price_range: float
    body_avg: float

@dataclass
class Observation:
    skill_id: int
    bar: int
    judgment: str
    bar_from: int
    bar_to: int
    behaviors: List[str]
    timestamp: str

@dataclass
class TimelineEvent:
    bar: int
    description: str
    event_type: str  # "inflection" or "observe"
    timestamp: str

@dataclass
class UserWeakness:
    pattern: str
    count: int = 0
    examples: List[str] = field(default_factory=list)

# =========================================================
# 数据加载
# =========================================================
@st.cache_data(ttl=300, show_spinner="加载中...")
def _fetch_raw(symbol: str) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={
                "datetime": "datetime", "open": "open",
                "high": "high", "low": "low", "close": "close"})
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.reset_index(drop=True)
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

def load_data(symbol: str, seed: int = None) -> pd.DataFrame:
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
# Swing / Leg 检测
# =========================================================
def detect_swings(df: pd.DataFrame) -> list:
    N = SWING_LOOKBACK
    swings = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(N, len(df) - N):
        if all(highs[i] > highs[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SH", price=float(highs[i])))
        if all(lows[i] < lows[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SL", price=float(lows[i])))
    return swings

def detect_legs(df: pd.DataFrame, swings: list) -> list:
    if not swings:
        return []
    legs = []
    direction = None
    start_idx = 0
    price_start = float(df.iloc[0]["close"])
    for s in swings:
        if s.kind == "SH" and direction != "bull":
            if direction is not None and s.index > start_idx + 2:
                _al(legs, df, start_idx, s.index - 1, direction, price_start,
                    float(df.iloc[s.index - 1]["close"]))
            direction = "bull"
            start_idx = s.index
            price_start = float(df.iloc[s.index]["high"])
        elif s.kind == "SL" and direction != "bear":
            if direction is not None and s.index > start_idx + 2:
                _al(legs, df, start_idx, s.index - 1, direction, price_start,
                    float(df.iloc[s.index - 1]["close"]))
            direction = "bear"
            start_idx = s.index
            price_start = float(df.iloc[s.index]["low"])
    if direction is not None and start_idx < len(df) - 2:
        _al(legs, df, start_idx, len(df) - 1, direction, price_start,
            float(df.iloc[len(df) - 1]["close"]))
    return legs

def _al(legs, df, start, end, direction, ps, pe):
    bc = sum(abs(float(df.iloc[i]["close"]) - float(df.iloc[i]["open"]))
             for i in range(start, end + 1))
    c = max(end - start + 1, 1)
    legs.append(Leg(start_idx=start, end_idx=end, direction=direction,
        bar_count=c, price_start=ps, price_end=pe,
        price_range=abs(pe - ps), body_avg=bc / c))

# =========================================================
# 图表
# =========================================================
def build_chart(chart_df, bar, hotzones=None, show_swings=False, show_legs=False):
    fig = go.Figure()
    vis = chart_df.iloc[:bar + 1]
    if len(vis) == 0:
        return fig
    fig.add_trace(go.Candlestick(
        x=vis.index, open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"],
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71"))
    annotations = []
    if show_swings:
        for s in st.session_state.get("swings", []):
            if hasattr(s, "index") and s.index <= bar:
                lab = "SH" if s.kind == "SH" else "SL"
                annotations.append(dict(x=s.index, y=s.price, text=lab,
                    showarrow=True, arrowhead=1, arrowcolor="#888",
                    font=dict(size=10, color="#888"),
                    ax=0, ay=-30 if s.kind == "SH" else 30))
    if show_legs:
        for leg in [l for l in st.session_state.get("legs", [])
                    if hasattr(l, "end_idx") and l.end_idx <= bar][-6:]:
            mb = (leg.start_idx + leg.end_idx) // 2
            mp = (leg.price_start + leg.price_end) / 2
            annotations.append(dict(x=mb, y=mp,
                text="{}({}根 {:.0f}pt)".format(leg.direction, leg.bar_count, leg.price_range),
                showarrow=False, font=dict(size=9, color="#999")))
    cur = chart_df.iloc[bar]
    annotations.append(dict(x=bar, y=cur["high"], text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#aaa",
        font=dict(size=8, color="#aaa"), ax=0, ay=25))
    fig.update_layout(annotations=annotations, height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f5f5f5", zeroline=False),
        template="plotly_white")
    return fig

# =========================================================
# GPT 教练 — 与用户共享完整K线环境
# =========================================================
def build_context(chart_df, bar, user_observation, skill_name,
                  behaviors, bar_from, bar_to, profile, history):
    """构建GPT上下文：完整OHLC + 用户观察 + 能力画像"""
    start = max(0, bar - 30)
    recent = []
    for i in range(start, bar + 1):
        r = chart_df.iloc[i]
        recent.append({
            "bar": i,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"])
        })

    ctx = {
        "current_bar": bar,
        "total_bars": len(chart_df),
        "skill": skill_name,
        "market": recent,
        "user_judgment": user_observation,
    }
    if behaviors:
        ctx["user_behaviors"] = behaviors
    if bar_from is not None and bar_to is not None:
        ctx["user_bar_range"] = [bar_from, bar_to]
    if profile:
        ctx["user_weaknesses"] = [
            {"pattern": w.pattern, "frequency": w.count} for w in profile if w.count > 0]
    if history:
        ctx["recent_training"] = [
            {"bar": h.bar, "skill": SKILLS.get(h.skill_id, {}).get("name", "?"),
             "judgment": h.judgment}
            for h in history[-5:]
        ]
    return ctx


def ask_coach(context, extra_prompt=None):
    """调用GPT教练"""
    max_retries = 3
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")
    user_msg = json.dumps(context, ensure_ascii=False)
    if extra_prompt:
        user_msg += "\n\n" + extra_prompt
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {"role": "system", "content": AI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if attempt < max_retries - 1 and ("429" in err or "rate" in err.lower()):
                time.sleep(2 ** (attempt + 1))
                continue
            return "AI调用失败: {}".format(e)
    return "AI调用失败: 请求过于频繁，请稍后再试"

# =========================================================
# 用户能力画像
# =========================================================
WEAKNESS_PATTERNS = {
    "conclusion_without_evidence": "结论缺乏依据",
    "ignore_followthrough": "忽略跟进",
    "single_bar_focus": "过度关注单根K线",
    "no_opponent": "忽略对手反应",
    "no_continuity": "缺乏连续性观察",
    "premature_reversal": "过早判断转换",
    "no_acceptance": "忽略接受度",
    "no_background": "忽略背景",
}

def detect_weakness(judgment, behaviors):
    """从用户观察中检测潜在弱点"""
    full = judgment + " " + " ".join(behaviors)
    hits = []
    if behaviors and not judgment.strip():
        hits.append("conclusion_without_evidence")
    if len(behaviors) == 0:
        hits.append("conclusion_without_evidence")
    if not any(k in full for k in ["跟进", "第二根", "第三根", "后续", "延续"]):
        hits.append("ignore_followthrough")
    if re.search(r'第\d+根', judgment) and "到" not in judgment and "-" not in judgment:
        hits.append("single_bar_focus")
    if not any(k in full for k in ["对手", "反攻", "反击", "回应", "买方", "卖方"]):
        hits.append("no_opponent")
    if not any(k in full for k in ["变化", "连续", "过程", "逐渐", "开始"]):
        hits.append("no_continuity")
    if any(k in full for k in ["转换", "反转", "转向"]):
        if not any(k in full for k in ["跟进", "延续", "持续"]):
            hits.append("premature_reversal")
    return hits


def update_profile(session, judgment, behaviors):
    """更新用户能力画像"""
    hits = detect_weakness(judgment, behaviors)
    profile = session.get("user_profile", {})
    for h in hits:
        if h not in profile:
            profile[h] = UserWeakness(pattern=WEAKNESS_PATTERNS.get(h, h))
        profile[h].count += 1
        if len(profile[h].examples) < 3:
            profile[h].examples.append(judgment[:60])
    session["user_profile"] = profile

# =========================================================
# 训练面板
# =========================================================
def render_training(session, bar, chart_df):
    mode = session.get("train_mode", 1)
    skill = SKILLS[mode]
    st.markdown("---")
    st.markdown("### 训练 {}/5: {}".format(mode, skill["name"]))
    st.markdown("*{}*".format(skill["question"]))

    # 用户判断
    judgment = st.text_area(
        "你的判断", height=55, key="judg_{}_{}".format(bar, mode),
        placeholder="描述你对当前市场的判断...")

    # K线范围
    st.markdown("**具体观察：**")
    c1, c2 = st.columns(2)
    with c1:
        bar_from = st.number_input("起始K线", min_value=0, max_value=bar,
            value=max(0, bar - 10), key="bf_{}_{}".format(bar, mode))
    with c2:
        bar_to = st.number_input("结束K线", min_value=0, max_value=bar,
            value=bar, key="bt_{}_{}".format(bar, mode))

    # 行为观察
    st.markdown("*你在这个范围内观察到的具体行为变化：*")
    c1, c2, c3 = st.columns(3)
    with c1:
        b1 = st.text_input("行为1", key="b1_{}_{}".format(bar, mode), placeholder="例如：实体缩小")
    with c2:
        b2 = st.text_input("行为2", key="b2_{}_{}".format(bar, mode), placeholder="例如：重叠增加")
    with c3:
        b3 = st.text_input("行为3", key="b3_{}_{}".format(bar, mode), placeholder="例如：跟进减少")

    # 按钮行
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        submitted = st.button("提交观察", key="submit_{}_{}".format(bar, mode))
    with c2:
        observe_btn = st.button("教练: 我该看哪里？", key="obs_{}_{}".format(bar, mode))
    with c3:
        challenge_btn = st.button("教练: 质疑我的观察", key="ch_{}_{}".format(bar, mode))
    with c4:
        if st.button("教练: 我有什么盲点？", key="blind_{}_{}".format(bar, mode)):
            profile = session.get("user_profile", {})
            hits = detect_weakness(judgment.strip(), [b1.strip(), b2.strip(), b3.strip()])
            if hits:
                weakness_names = [WEAKNESS_PATTERNS.get(h, h) for h in hits]
                st.warning("本次检测到的观察盲点：" + "、".join(weakness_names))
            else:
                st.success("本次观察暂未检测到明显盲点")

    history = session.get("observations", [])
    profile = session.get("user_profile", {})

    # 提交观察
    if submitted and judgment.strip():
        behaviors = [x.strip() for x in [b1, b2, b3] if x.strip()]
        obs = Observation(
            skill_id=mode, bar=bar, judgment=judgment.strip(),
            bar_from=bar_from, bar_to=bar_to,
            behaviors=behaviors, timestamp=datetime.now().strftime("%H:%M:%S"))
        session.setdefault("observations", []).append(obs)
        session["last_judgment_{}".format(mode)] = judgment.strip()
        session["last_behaviors_{}".format(mode)] = behaviors
        session["last_obs_bar_{}".format(mode)] = bar

        update_profile(session, judgment.strip(), behaviors)
        st.success("已记录")

    # 教练：我该看哪里？
    if observe_btn:
        profile = session.get("user_profile", {})
        ctx = build_context(chart_df, bar, "", skill["name"], [], None, None, profile, history)
        extra = "用户请求教练指出应该观察的K线范围。只指向，不解释。"
        with st.spinner("教练思考中..."):
            result = ask_coach(ctx, extra)
        st.markdown("**教练：** " + result)

    # 教练：质疑我的观察
    if challenge_btn:
        lj = session.get("last_judgment_{}".format(mode), "")
        lb = session.get("last_behaviors_{}".format(mode), [])
        lbf = session.get("last_obs_bar_{}".format(mode))
        if not lj:
            st.warning("请先提交你的观察")
        else:
            profile = session.get("user_profile", {})
            # 获取之前的bar范围
            bf_key = "bf_{}_{}".format(lbf, mode)
            bt_key = "bt_{}_{}".format(lbf, mode)
            bf_val = session.get(bf_key, None)
            bt_val = session.get(bt_key, None)
            ctx = build_context(chart_df, bar, lj, skill["name"], lb,
                                bf_val, bt_val, profile, history)
            extra = "请质疑用户的具体观察。找出可能忽略的地方。"
            with st.spinner("教练思考中..."):
                result = ask_coach(ctx, extra)
            st.markdown("**教练：** " + result)

    # 教练：回顾之前的判断
    last_bar = session.get("last_obs_bar_{}".format(mode))
    if last_bar is not None and bar > last_bar + 5:
        lj = session.get("last_judgment_{}".format(mode), "")
        if lj:
            st.markdown("---")
            if st.button("教练: 回顾我之前的判断", key="reveal_{}".format(mode)):
                lb = session.get("last_behaviors_{}".format(mode), [])
                profile = session.get("user_profile", {})
                ctx = build_context(chart_df, bar, "", skill["name"], [], None, None, profile, history)
                extra = (
                    "用户在K{}的判断是：{}\n"
                    "用户的行为观察：{}\n"
                    "现在已走到K{}。"
                    "请引导用户回顾：后续发生了什么变化？"
                    "不要评价对错。只问关于可观察事实的问题。"
                ).format(last_bar, lj, "、".join(lb) if lb else "无", bar)
                with st.spinner("教练思考中..."):
                    result = ask_coach(ctx, extra)
                st.markdown("**教练回顾：** " + result)

# =========================================================
# 时间轴
# =========================================================
def render_timeline(session, bar):
    tl = session.get("timeline", [])
    with st.expander("行为变化时间轴 ({}条)".format(len(tl))):
        for ev in tl:
            icon = "●" if ev.event_type == "inflection" else "○"
            st.text("{} [K{}] {} {}".format(icon, ev.bar, ev.timestamp, ev.description))
        if not tl:
            st.caption("在Replay中记录你观察到的行为变化")
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        with c1:
            new_desc = st.text_input("记录变化", key="tl_{}".format(bar),
                placeholder="描述这里的行为变化...")
        with c2:
            etype = st.selectbox("类型", ["拐点", "观察"], key="tlt_{}_{}".format(bar))
        if st.button("添加", key="tl_add_{}".format(bar)) and new_desc.strip():
            tl.append(TimelineEvent(bar=bar, description=new_desc.strip(),
                event_type="inflection" if etype == "拐点" else "observe",
                timestamp=datetime.now().strftime("%H:%M:%S")))
            session["timeline"] = tl
            st.rerun()
        if tl and st.button("清空", key="tl_clear"):
            session["timeline"] = []
            st.rerun()

# =========================================================
# 能力画像面板
# =========================================================
def render_profile(session):
    profile = session.get("user_profile", {})
    if not profile:
        return
    with st.expander("能力画像"):
        sorted_w = sorted(profile.items(), key=lambda x: x[1].count, reverse=True)
        for key, w in sorted_w[:5]:
            bar_count = min(w.count, 20)
            bar_str = "█" * bar_count + "░" * (20 - bar_count)
            st.text("{}: {} ({}次)".format(w.pattern, bar_str, w.count))
            for ex in w.examples[-1:]:
                st.caption("  最近：{}".format(ex))

# =========================================================
# 训练记录
# =========================================================
def render_history(session):
    observations = session.get("observations", [])
    if not observations:
        return
    with st.expander("训练记录 ({}次)".format(len(observations))):
        for obs in observations[-8:]:
            name = SKILLS.get(obs.skill_id, {}).get("name", "?")
            line = "[{}] K{}-{} {}: {}".format(
                name, obs.bar_from, obs.bar_to, obs.timestamp, obs.judgment[:35])
            if obs.behaviors:
                line += " → " + " | ".join(obs.behaviors)
            st.text(line)

# =========================================================
# 主程序
# =========================================================
def main():
    for key, default in [
        ("data_loaded", False), ("observations", []),
        ("train_mode", 1), ("timeline", []),
        ("replay_mode", "复盘模式"), ("user_profile", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.title("读盘训练器 V15")
        st.caption("教练制 | Replay | 能力画像")

        symbol = st.text_input("合约代码", value="rb2510", key="sym")
        c_load, c_rand = st.columns(2)
        with c_load:
            if st.button("加载数据", key="load"):
                _do_load(symbol)
        with c_rand:
            if st.button("换一段", key="rand_chunk"):
                _do_load(symbol)

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.markdown("**Replay模式：**")
            st.session_state["replay_mode"] = st.radio(
                "", ["复盘模式", "严格模式"], key="rm_radio",
                label_visibility="collapsed",
                captions=["可回退、快进", "只能+1，不可回退"])

            st.markdown("**训练目标：**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                active = st.session_state.get("train_mode") == sid
                btn = ">> {}. {} <<".format(sid, name) if active else "{}. {}".format(sid, name)
                if st.button(btn, key="mode_{}".format(sid)):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            obs_list = st.session_state.get("observations", [])
            st.markdown("---")
            st.text("总观察: {}次".format(len(obs_list)))

            st.markdown("---")
            st.markdown("**标注（默认关闭）：**")
            st.session_state["show_swings"] = st.checkbox("Swing点", False, key="cb_sw")
            st.session_state["show_legs"] = st.checkbox("波段线", False, key="cb_lg")

    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器 V15")
        st.markdown("")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** — {}".format(sid, s["name"], s["question"]))
        st.markdown("")
        st.markdown("> 你看图。你判断。教练只提问，不给答案。")
        st.markdown("")
        st.markdown("**训练架构：**")
        st.markdown("- 用户 = 真正训练者")
        st.markdown("- GPT = 教练（与你看同一个盘面）")
        st.markdown("- 软件 = 训练场")
        return

    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar", 0)
    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    show_sw = st.session_state.get("show_swings", False)
    show_lg = st.session_state.get("show_legs", False)
    chart = build_chart(chart_df, bar, show_swings=show_sw, show_legs=show_lg)
    st.plotly_chart(chart, use_container_width=True)

    strict = st.session_state.get("replay_mode") == "严格模式"
    if strict:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("+1 →", key="b_n1"):
                st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 1)
                st.rerun()
        with c2:
            pct = bar / max(len(chart_df) - 1, 1)
            st.progress(pct)
            st.caption("Replay: {}/{} ({:.0f}%)".format(bar, len(chart_df) - 1, pct * 100))
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            if st.button("<<-5", key="b_p5"):
                st.session_state["current_bar"] = max(0, bar - 5); st.rerun()
        with c2:
            if st.button("<-1", key="b_p1"):
                st.session_state["current_bar"] = max(0, bar - 1); st.rerun()
        with c3:
            if st.button("+1->", key="b_n1r"):
                st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 1); st.rerun()
        with c4:
            if st.button("+5->", key="b_n5"):
                st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 5); st.rerun()
        with c5:
            if st.button("+15->", key="b_n15"):
                st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 15); st.rerun()
        with c6:
            if st.button("末尾", key="b_end"):
                st.session_state["current_bar"] = len(chart_df) - 1; st.rerun()

    cur = chart_df.iloc[bar]
    st.caption("#{}/{}  O:{:.1f}  H:{:.1f}  L:{:.1f}  C:{:.1f}  {:+.1f}".format(
        bar, len(chart_df) - 1, cur["open"], cur["high"], cur["low"], cur["close"],
        cur["close"] - cur["open"]))

    col_train, col_tl = st.columns([2, 1])
    with col_train:
        render_training(st.session_state, bar, chart_df)
    with col_tl:
        render_timeline(st.session_state, bar)
        render_profile(st.session_state)


def _do_load(symbol):
    with st.spinner("加载中..."):
        seed = random.randint(0, 999999)
        df = load_data(symbol, seed=seed)
        if df is not None and len(df) > 0:
            sw = detect_swings(df)
            lg = detect_legs(df, sw)
            st.session_state.update({
                "chart_df": df, "swings": sw, "legs": lg,
                "current_bar": min(40, len(df) - 1),
                "data_loaded": True, "observations": [],
                "timeline": [], "train_mode": 1, "user_profile": {},
            })
            for k in list(st.session_state.keys()):
                if k.startswith("last_"):
                    del st.session_state[k]
            st.success("{}根K线".format(len(df)))
        else:
            st.error("加载失败")


if __name__ == "__main__":
    main()
