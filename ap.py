# =========================================================
# Al Brooks 读盘训练器 V12
# =========================================================
#
# 训练目标：
#   1. 背景阅读     — 识别市场背景
#   2. 控制权识别   — 判断谁在控制市场
#   3. 推进质量判断 — 评估推进波强弱
#   4. 回调 vs 转换 — 区分正常回调与真正转换
#   5. 市场接受     — 市场是否接受新价格
#
# 设计原则：
#   - 用户直接看 K 线图，系统不写文字总结
#   - AI 是苏格拉底式教练：不给答案，通过提问发现盲点
#   - Replay 是唯一训练方式
#
# =========================================================

import os
import time
from datetime import datetime
from dataclasses import dataclass

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

SKILLS = {
    1: {"name": "背景阅读",     "question": "当前市场背景是什么？"},
    2: {"name": "控制权识别",   "question": "现在谁在控制市场？"},
    3: {"name": "推进质量",     "question": "最近推进的质量如何？"},
    4: {"name": "回调vs转换",   "question": "这是正常回调还是控制权转换？"},
    5: {"name": "市场接受",     "question": "市场是否接受了新价格？"},
}

AI_SYSTEM_PROMPT = (
    "你是一个 Al Brooks 价格行为读盘教练。"
    "你的角色不是给学生答案，而是通过提问帮助他们发现自己可能忽略的地方。"
    "规则：绝对不能告诉学生正确答案；绝对不能解释市场行为；"
    "只能提问和指向具体的K线位置；问题要简短、直接、具体；"
    "每次最多2-3个问题；用中文回复。"
    "指向具体K线范围时用第X根到第Y根。不要加前缀或编号。"
)


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
class SkillAnswer:
    skill_id: int
    bar: int
    answer: str
    timestamp: str

# =========================================================
# 数据加载
# =========================================================
@st.cache_data(ttl=300, show_spinner="加载中...")
def load_data(symbol: str = "IF0") -> pd.DataFrame:
    for _ in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={
                "datetime": "datetime", "open": "open",
                "high": "high", "low": "low", "close": "close",
            })
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.reset_index(drop=True)
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


# =========================================================
# 检测函数
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
    if len(swings) < 2:
        return []
    legs = []
    for i in range(len(swings) - 1):
        s1, s2 = swings[i], swings[i + 1]
        if s2.index <= s1.index:
            continue
        segment = df.iloc[s1.index: s2.index + 1]
        if len(segment) < 2:
            continue
        if s1.kind == "SL" and s2.kind == "SH":
            direction, ps, pe = "bull", s1.price, s2.price
        elif s1.kind == "SH" and s2.kind == "SL":
            direction, ps, pe = "bear", s1.price, s2.price
        else:
            continue
        bodies = []
        for j in range(len(segment)):
            bar = segment.iloc[j]
            rng = bar["high"] - bar["low"]
            bodies.append(abs(bar["close"] - bar["open"]) / rng if rng > 1e-9 else 0)
        legs.append(Leg(
            start_idx=s1.index, end_idx=s2.index, direction=direction,
            bar_count=s2.index - s1.index + 1, price_start=ps, price_end=pe,
            price_range=abs(pe - ps), body_avg=round(float(np.mean(bodies)), 3),
        ))
    return legs



# =========================================================
# 图表上下文提取（给 AI 用）
# =========================================================
def extract_chart_context(chart_df, swings, legs, bar):
    n = len(chart_df)
    if bar < 0 or bar >= n:
        return None
    lookback = min(25, bar)
    recent = chart_df.iloc[bar - lookback: bar + 1].copy()

    bars_text = []
    for idx in range(len(recent)):
        row = recent.iloc[idx]
        actual_bar = bar - lookback + idx
        d = "+" if row["close"] > row["open"] else "-" if row["close"] < row["open"] else "="
        bars_text.append(
            "#{0} O:{1:.1f} H:{2:.1f} L:{3:.1f} C:{4:.1f} {5}".format(
                actual_bar, row["open"], row["high"], row["low"], row["close"], d
            )
        )

    safe_legs = [l for l in legs if hasattr(l, "end_idx") and l.end_idx <= bar]
    legs_text = []
    for l in safe_legs[-4:]:
        d = "多" if l.direction == "bull" else "空"
        legs_text.append("波段{0}: #{1}-{2} {3}根 范围{4:.1f}".format(
            d, l.start_idx, l.end_idx, l.bar_count, l.price_range))

    safe_swings = [s for s in swings if hasattr(s, "index") and s.index <= bar]
    swings_text = []
    for s in safe_swings[-4:]:
        label = "SH" if s.kind == "SH" else "SL"
        swings_text.append("{0}: #{1} {2:.1f}".format(label, s.index, s.price))

    return {
        "current_bar": bar,
        "total_bars": n,
        "bars": bars_text,
        "legs": legs_text,
        "swings": swings_text,
    }


# =========================================================
# AI 教练 — 苏格拉底式提问
# =========================================================
def call_ai(user_message: str, skill_name: str, context: dict) -> str:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.videocaptioner.cn/v1",
        )
        bars_summary = context.get("bars", [])
        recent_bars = bars_summary[-15:]

        parts = [
            "当前训练目标：{}".format(skill_name),
            "当前在第{}根K线，共{}根。".format(
                context.get("current_bar", 0), context.get("total_bars", 0)),
            "",
            "最近K线数据：",
            "\n".join(recent_bars),
        ]
        if context.get("legs"):
            parts += ["", "波段："] + context["legs"]
        if context.get("swings"):
            parts += ["", "Swing："] + context["swings"]
        parts.append("")
        parts.append(user_message)

        full_prompt = "\n".join(parts)

        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "AI调用失败: {}".format(e)


def ai_observe(context: dict, skill_name: str) -> str:
    prompt = (
        "用户正在训练「{}」。".format(skill_name)
        + "请指出在这个画面中，用户应该重点观察哪几根K线。"
        + "只指出K线范围和应该关注什么特征，不要给出结论。"
    )
    return call_ai(prompt, skill_name, context)


def ai_challenge(answer: str, context: dict, skill_name: str) -> str:
    prompt = (
        "训练目标「{}」\n用户的判断：{}\n\n".format(skill_name, answer)
        + "请针对这个判断提出2-3个追问，帮助用户发现自己可能忽略的地方。"
    )
    return call_ai(prompt, skill_name, context)


def ai_reveal(context_after: dict, answer: str, skill_name: str) -> str:
    bars_after = context_after.get("bars", [])[-10:]
    prompt = (
        "训练目标「{}」\n用户的判断：{}\n\n".format(skill_name, answer)
        + "之后走出的K线：\n" + "\n".join(bars_after) + "\n\n"
        + "价格已经走出来了。请提出2-3个问题，"
        + "引导学生回顾自己的判断过程，不要直接说对错。"
    )
    return call_ai(prompt, skill_name, context_after)



# =========================================================
# 图表
# =========================================================
def build_chart(chart_df, swings, legs, bar, show_swings=True, show_legs=True):
    fig = go.Figure()
    visible = chart_df.iloc[:bar + 1]
    if len(visible) == 0:
        return fig

    fig.add_trace(go.Candlestick(
        x=visible.index, open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"],
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",
    ))

    annotations = []
    if show_swings:
        for s in swings:
            if hasattr(s, "index") and s.index <= bar:
                label = "SH" if s.kind == "SH" else "SL"
                annotations.append(dict(
                    x=s.index, y=s.price, text=label,
                    showarrow=True, arrowhead=1, arrowcolor="#888",
                    font=dict(size=10, color="#888"),
                    ax=0, ay=-30 if s.kind == "SH" else 30,
                ))
        swing_highs = [(s.index, s.price) for s in swings
                       if hasattr(s, "kind") and s.kind == "SH" and s.index <= bar]
        swing_lows = [(s.index, s.price) for s in swings
                      if hasattr(s, "kind") and s.kind == "SL" and s.index <= bar]
        if len(swing_highs) >= 2:
            sx, sy = zip(*sorted(swing_highs))
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers",
                line=dict(color="#3498db", width=1, dash="dot"),
                marker=dict(size=3), showlegend=False, hoverinfo="skip"))
        if len(swing_lows) >= 2:
            sx, sy = zip(*sorted(swing_lows))
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers",
                line=dict(color="#e67e22", width=1, dash="dot"),
                marker=dict(size=3), showlegend=False, hoverinfo="skip"))

    if show_legs:
        safe_legs = [l for l in legs if hasattr(l, "end_idx") and l.end_idx <= bar]
        for leg in safe_legs[-6:]:
            mid_bar = (leg.start_idx + leg.end_idx) // 2
            mid_price = (leg.price_start + leg.price_end) / 2
            label = "多" if leg.direction == "bull" else "空"
            color = "#8e44ad" if leg.direction == "bull" else "#c0392b"
            annotations.append(dict(
                x=mid_bar, y=mid_price,
                text="{}({}根 {:.0f}pt)".format(label, leg.bar_count, leg.price_range),
                showarrow=False, font=dict(size=9, color=color),
            ))

    cur = chart_df.iloc[bar]
    annotations.append(dict(
        x=bar, y=cur["high"], text="#{}".format(bar),
        showarrow=True, arrowhead=0, arrowcolor="#aaa",
        font=dict(size=8, color="#aaa"), ax=0, ay=25,
    ))

    fig.update_layout(annotations=annotations, height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f5f5f5", zeroline=False),
        template="plotly_white",
    )
    return fig


# =========================================================
# 训练面板
# =========================================================
def render_training(session, bar):
    mode = session.get("train_mode", 1)
    skill = SKILLS[mode]

    st.markdown("---")
    st.markdown("### 训练 {}/5: {}".format(mode, skill["name"]))
    st.markdown("*{}*".format(skill["question"]))

    answer = st.text_area(
        "你的判断", height=80,
        key="answer_{}_{}".format(bar, mode),
        placeholder="直接写出你的观察和结论...",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        submitted = st.button("提交", key="submit_{}_{}".format(bar, mode))
    with c2:
        observe_clicked = st.button("AI: 我该看哪里？", key="obs_{}_{}".format(bar, mode))
    with c3:
        challenge_clicked = st.button("AI: 质疑我的判断", key="chal_{}_{}".format(bar, mode))

    chart_df = session.get("chart_df")
    swings = session.get("swings", [])
    legs = session.get("legs", [])
    context = extract_chart_context(chart_df, swings, legs, bar) if chart_df is not None else None

    if submitted and answer.strip():
        sa = SkillAnswer(
            skill_id=mode, bar=bar,
            answer=answer.strip(),
            timestamp=datetime.now().strftime("%H:%M:%S"),
        )
        session.setdefault("answers", []).append(sa)
        session["last_answer_{}".format(mode)] = answer.strip()
        session["last_answer_bar_{}".format(mode)] = bar
        st.success("已记录")

    if observe_clicked and context:
        with st.spinner("AI思考中..."):
            result = ai_observe(context, skill["name"])
        st.markdown("**AI教练：**\n{}".format(result))

    if challenge_clicked:
        last_answer = session.get("last_answer_{}".format(mode), "")
        if not last_answer:
            st.warning("请先提交你的判断")
        elif context:
            with st.spinner("AI思考中..."):
                result = ai_challenge(last_answer, context, skill["name"])
            st.markdown("**AI教练：**\n{}".format(result))

    # 揭示后回顾
    last_bar = session.get("last_answer_bar_{}".format(mode))
    if last_bar is not None and bar > last_bar + 5:
        last_answer = session.get("last_answer_{}".format(mode), "")
        if last_answer:
            st.markdown("---")
            if st.button("AI: 回顾我之前的判断", key="reveal_{}".format(mode)):
                if chart_df is not None:
                    context_after = extract_chart_context(chart_df, swings, legs, bar)
                    with st.spinner("AI思考中..."):
                        result = ai_reveal(context_after, last_answer, skill["name"])
                    st.markdown("**AI回顾：**\n{}".format(result))


def render_history(session):
    answers = session.get("answers", [])
    if not answers:
        return
    with st.expander("训练记录"):
        by_skill = {}
        for a in answers:
            by_skill.setdefault(a.skill_id, []).append(a)
        for sid in sorted(by_skill.keys()):
            name = SKILLS[sid]["name"]
            st.text("[{}] {}次".format(name, len(by_skill[sid])))
        for a in answers[-8:]:
            name = SKILLS.get(a.skill_id, {}).get("name", "?")
            st.text("[{}] #{} {}: {}".format(a.timestamp, a.bar, name, a.answer[:50]))


# =========================================================
# 主函数
# =========================================================
def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器 V12", layout="wide")

    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
        st.session_state["answers"] = []
        st.session_state["train_mode"] = 1

    with st.sidebar:
        st.title("读盘训练器 V12")
        st.caption("5个训练目标 | AI苏格拉底式教练 | Replay")

        symbol = st.text_input("合约代码", value="rb2510", key="sym")
        if st.button("加载数据", key="load"):
            with st.spinner("加载中..."):
                df = load_data(symbol)
                if df is not None and len(df) > 0:
                    new_swings = detect_swings(df)
                    new_legs = detect_legs(df, new_swings)
                    st.session_state["chart_df"] = df
                    st.session_state["swings"] = new_swings
                    st.session_state["legs"] = new_legs
                    st.session_state["current_bar"] = min(40, len(df) - 1)
                    st.session_state["data_loaded"] = True
                    st.session_state["answers"] = []
                    st.session_state["train_mode"] = 1
                    for k in list(st.session_state.keys()):
                        if k.startswith("last_answer"):
                            del st.session_state[k]
                    st.success("{}根K线, {}个波段".format(len(df), len(new_legs)))
                else:
                    st.error("加载失败")

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.markdown("**训练目标：**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                is_active = st.session_state.get("train_mode") == sid
                btn = ">> {}. {} <<".format(sid, name) if is_active else "{}. {}".format(sid, name)
                if st.button(btn, key="mode_{}".format(sid)):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            answers = st.session_state.get("answers", [])
            st.markdown("---")
            st.text("总训练: {}次".format(len(answers)))
            st.markdown("---")
            st.markdown("**标注：**")
            st.session_state["show_swings"] = st.checkbox("Swing", True, key="cb_sw")
            st.session_state["show_legs"] = st.checkbox("波段", True, key="cb_lg")

    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器 V12")
        st.markdown("")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** - {}".format(sid, s["name"], s["question"]))
        st.markdown("")
        st.markdown("> 你看图，你判断。AI只提问，不给答案。")
        return

    chart_df = st.session_state["chart_df"]
    swings = st.session_state["swings"]
    legs = st.session_state["legs"]
    bar = st.session_state.get("current_bar", 0)

    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    show_sw = st.session_state.get("show_swings", True)
    show_lg = st.session_state.get("show_legs", True)
    chart = build_chart(chart_df, swings, legs, bar, show_swings=show_sw, show_legs=show_lg)
    st.plotly_chart(chart, use_container_width=True)

    # Replay
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        if st.button("<<-5", key="b_p5"):
            st.session_state["current_bar"] = max(0, bar - 5); st.rerun()
    with c2:
        if st.button("<-1", key="b_p1"):
            st.session_state["current_bar"] = max(0, bar - 1); st.rerun()
    with c3:
        if st.button("+1->", key="b_n1"):
            st.session_state["current_bar"] = min(len(chart_df)-1, bar+1); st.rerun()
    with c4:
        if st.button("+5->", key="b_n5"):
            st.session_state["current_bar"] = min(len(chart_df)-1, bar+5); st.rerun()
    with c5:
        if st.button("+15->", key="b_n15"):
            st.session_state["current_bar"] = min(len(chart_df)-1, bar+15); st.rerun()
    with c6:
        if st.button("末尾", key="b_end"):
            st.session_state["current_bar"] = len(chart_df)-1; st.rerun()

    cur = chart_df.iloc[bar]
    st.caption("#{}/{}  O:{:.1f}  H:{:.1f}  L:{:.1f}  C:{:.1f}  {:+.1f}".format(
        bar, len(chart_df)-1, cur["open"], cur["high"], cur["low"], cur["close"],
        cur["close"]-cur["open"]))

    col_train, col_hist = st.columns([2, 1])
    with col_train:
        render_training(st.session_state, bar)
    with col_hist:
        render_history(st.session_state)


if __name__ == "__main__":
    main()
