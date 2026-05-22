# =========================================================
# Al Brooks 读盘训练器 V12
# =========================================================
#
# 训练目标（不是预测下一根K线）：
#   1. 背景阅读     — 识别市场背景（趋势/区间/关键位）
#   2. 控制权识别   — 判断谁在控制市场
#   3. 推进质量判断 — 评估推进波的强弱
#   4. 回调vs转换   — 区分正常回调与真正控制权转换
#   5. 市场接受     — 市场是否接受新价格
#
# 设计原则：
#   - 用户做判断，系统做辅助
#   - AI 不给答案，只提问和指向
#   - Replay 是唯一的训练方式
#   - 系统不展示"结论"，只提供"观察材料"
#
# =========================================================

import os
import time
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak

# =========================================================
# 常量
# =========================================================
SWING_LOOKBACK = 3

# 5个训练目标定义
SKILLS = {
    1: {
        "name": "背景阅读",
        "question": "当前市场背景是什么？",
        "hint": "看整体结构：是趋势还是区间？关键位在哪里？",
    },
    2: {
        "name": "控制权识别",
        "question": "现在谁在控制市场？",
        "hint": "看谁在推进：买方还是卖方？推进是否持续？",
    },
    3: {
        "name": "推进质量",
        "question": "最近推进的质量如何？",
        "hint": "看实体大小、连续性、尾巴比例、是否被反包",
    },
    4: {
        "name": "回调 vs 转换",
        "question": "这是正常回调还是控制权转换？",
        "hint": "回调深度、是否回到起点、反包力度、跟进行为",
    },
    5: {
        "name": "市场接受",
        "question": "市场是否接受了新价格？",
        "hint": "价格是否维持在新水平？后续K线是否确认？",
    },
}


# =========================================================
# 数据类
# =========================================================
@dataclass
class SwingPoint:
    index: int
    kind: str    # "SH" / "SL"
    price: float


@dataclass
class Leg:
    start_idx: int
    end_idx: int
    direction: str   # "bull" / "bear"
    bar_count: int
    price_start: float
    price_end: float
    price_range: float
    body_avg: float


@dataclass
class SkillAnswer:
    """用户对某个训练目标的回答"""
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
                "datetime": "datetime",
                "open": "open", "high": "high",
                "low": "low", "close": "close",
            })
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.reset_index(drop=True)
        except Exception as e:
            time.sleep(1)
    return pd.DataFrame()


# =========================================================
# 检测函数 — 只提供原始数据，不做判断
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


def get_raw_materials(chart_df, swings, legs, bar):
    """提取当前 bar 附近的原始观察材料（给用户看，不是给系统分析的）"""
    materials = {}
    n = len(chart_df)
    if bar < 0 or bar >= n:
        return materials

    cur = chart_df.iloc[bar]

    # 1. 最近 K 线的原始数据
    lookback = min(20, bar)
    recent = chart_df.iloc[bar - lookback: bar + 1]

    # HC/LC 统计（原始数字）
    hc, lc = 0, 0
    for i in range(1, len(recent)):
        if recent.iloc[i]["close"] > recent.iloc[i - 1]["close"]:
            hc += 1
        else:
            lc += 1
    materials["hc"] = hc
    materials["lc"] = lc

    # 实体均值（前半 vs 后半）
    mid = len(recent) // 2
    first_bodies = (recent.iloc[:mid]["close"] - recent.iloc[:mid]["open"]).abs()
    second_bodies = (recent.iloc[mid:]["close"] - recent.iloc[mid:]["open"]).abs()
    materials["body_avg_first"] = first_bodies.mean()
    materials["body_avg_second"] = second_bodies.mean()

    # 最近波段
    safe_legs = [l for l in legs if hasattr(l, "end_idx") and l.end_idx <= bar]
    if safe_legs:
        last_leg = safe_legs[-1]
        materials["last_leg"] = {
            "direction": last_leg.direction,
            "bars": last_leg.bar_count,
            "range": round(last_leg.price_range, 2),
            "body": round(last_leg.body_avg, 3),
        }
        if len(safe_legs) >= 2:
            prev_leg = safe_legs[-2]
            materials["prev_leg"] = {
                "direction": prev_leg.direction,
                "bars": prev_leg.bar_count,
                "range": round(prev_leg.price_range, 2),
            }
            # 回调占推进的比例
            if last_leg.price_range > 1e-9:
                ratio = prev_leg.price_range / last_leg.price_range
                materials["pullback_ratio"] = round(ratio, 2)

    # 最近 Swing
    safe_swings = [s for s in swings if hasattr(s, "index") and s.index <= bar]
    if safe_swings:
        materials["last_swing"] = {
            "kind": safe_swings[-1].kind,
            "index": safe_swings[-1].index,
            "price": round(safe_swings[-1].price, 2),
        }
        if len(safe_swings) >= 2:
            materials["prev_swing"] = {
                "kind": safe_swings[-2].kind,
                "index": safe_swings[-2].index,
                "price": round(safe_swings[-2].price, 2),
            }

    # 全局位置
    full_high = chart_df["high"].max()
    full_low = chart_df["low"].min()
    full_range = full_high - full_low
    if full_range > 1e-9:
        materials["price_position"] = round(
            (cur["close"] - full_low) / full_range * 100, 0)

    # 当前 bar 的实体信息
    body = abs(cur["close"] - cur["open"])
    total = cur["high"] - cur["low"]
    materials["cur_body_pct"] = round(body / total * 100, 0) if total > 1e-9 else 0
    materials["cur_direction"] = ("阳线" if cur["close"] > cur["open"]
                                  else "阴线" if cur["close"] < cur["open"]
                                  else "十字")

    return materials



# =========================================================
# 图表 — 可切换标注层
# =========================================================
def build_chart(chart_df, swings, legs, bar, show_swings=True, show_legs=True):
    """构建 K 线图，用户选择要看哪些标注"""
    fig = go.Figure()
    visible = chart_df.iloc[:bar + 1]

    if len(visible) == 0:
        return fig

    fig.add_trace(go.Candlestick(
        x=visible.index,
        open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"],
        increasing_line_color="#e74c3c",
        decreasing_line_color="#2ecc71",
    ))

    annotations = []

    # Swing 标注
    if show_swings:
        for s in swings:
            if hasattr(s, "index") and s.index <= bar:
                label = "SH" if s.kind == "SH" else "SL"
                y_off = -30 if s.kind == "SH" else 30
                annotations.append(dict(
                    x=s.index, y=s.price, text=label,
                    showarrow=True, arrowhead=1, arrowcolor="#555",
                    font=dict(size=10, color="#555"),
                    ax=0, ay=y_off,
                ))
                # Swing 连线
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

    # 波段标注
    if show_legs:
        safe_legs = [l for l in legs if hasattr(l, "end_idx") and l.end_idx <= bar]
        for leg in safe_legs[-6:]:  # 只显示最近6个波段
            mid_bar = (leg.start_idx + leg.end_idx) // 2
            mid_price = (leg.price_start + leg.price_end) / 2
            label = "多" if leg.direction == "bull" else "空"
            annotations.append(dict(
                x=mid_bar, y=mid_price,
                text=f"{label}({leg.bar_count}根 {leg.price_range:.0f}pt)",
                showarrow=False,
                font=dict(size=9, color="#8e44ad" if leg.direction == "bull" else "#c0392b"),
            ))

    # 当前 bar 标记
    cur = chart_df.iloc[bar]
    annotations.append(dict(
        x=bar, y=cur["high"],
        text=f"#{bar}",
        showarrow=True, arrowhead=0, arrowcolor="#999",
        font=dict(size=8, color="#999"), ax=0, ay=25,
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
# 训练面板 — 围绕 5 个训练目标
# =========================================================

def render_skill_panel(materials, session, current_bar):
    """核心训练面板：用户逐个完成 5 个判断"""

    if not materials:
        st.warning("没有足够的观察材料，请先移动到有数据的K线位置")
        return

    # 当前训练模式
    mode = session.get("train_mode", 1)

    st.markdown("---")
    st.markdown(f"**训练 {mode}/5: {SKILLS[mode]['name']}**")
    st.markdown(f"*{SKILLS[mode]['question']}*")

    # 展示原始观察材料（用户需要的"原料"）
    st.markdown("**观察材料：**")
    render_materials(materials, mode)

    st.markdown("")

    # 用户输入区域
    if mode == 1:  # 背景阅读
        col1, col2 = st.columns(2)
        with col1:
            structure = st.radio("市场结构", ["趋势（多）", "趋势（空）", "区间", "不确定"],
                                key="s1_struct")
        with col2:
            key_level = st.text_input("关键价格位", placeholder="如: 前高3850, 前低3780",
                                      key="s1_key")
        user_answer = f"结构={structure}, 关键位={key_level}"

    elif mode == 2:  # 控制权识别
        control = st.radio("谁在控制？", ["多头控制", "空头控制", "争夺中", "不确定"],
                           key="s2_ctrl")
        reason = st.text_input("你的依据", key="s2_reason", placeholder="简洁描述依据")
        user_answer = f"{control}, 依据: {reason}"

    elif mode == 3:  # 推进质量
        quality = st.radio("推进质量", ["强", "中等", "弱"], key="s3_qual")
        detail = st.text_input("关键特征", key="s3_detail",
                               placeholder="如: 实体饱满、连续阳线、无反包")
        user_answer = f"质量={quality}, 特征: {detail}"

    elif mode == 4:  # 回调 vs 转换
        judgment = st.radio("这是？", ["正常回调", "控制权转换", "不确定"], key="s4_judg")
        evidence = st.text_input("证据", key="s4_evid",
                                 placeholder="如: 回调浅、未破前低、快速恢复")
        user_answer = f"{judgment}, 证据: {evidence}"

    elif mode == 5:  # 市场接受
        accept = st.radio("市场态度", ["接受", "拒绝", "待观察"], key="s5_acc")
        explain = st.text_input("你的观察", key="s5_exp",
                                placeholder="如: 价格维持在新高位、后续K线确认")
        user_answer = f"{accept}, 观察: {explain}"

    col_submit, col_skip, col_next = st.columns(3)
    with col_submit:
        if st.button("提交判断", key="submit_skill"):
            answer = SkillAnswer(
                skill_id=mode, bar=current_bar, answer=user_answer,
                timestamp=datetime.now().strftime("%H:%M:%S"),
            )
            session.setdefault("answers", []).append(answer)
            session["last_submit"] = user_answer
            st.success("已记录")
    with col_skip:
        if st.button("跳过", key="skip_skill"):
            st.info("跳过此题")
    with col_next:
        if st.button("下一题 ->" if mode < 5 else "完成本轮", key="next_skill"):
            session["train_mode"] = min(mode + 1, 5)
            st.rerun()

    # 显示上次提交
    if "last_submit" in session:
        st.caption(f"上次提交: {session['last_submit']}")


def render_materials(materials, mode):
    """根据当前训练模式，展示相关的原始观察材料"""
    col1, col2 = st.columns(2)

    with col1:
        # K线基础数据（所有模式都需要）
        if "hc" in materials:
            st.text(f"最近: HC={materials['hc']} LC={materials['lc']}")
        if "cur_direction" in materials:
            st.text(f"当前K线: {materials['cur_direction']}, 实体占比{materials.get('cur_body_pct', 0)}%")
        if "price_position" in materials:
            st.text(f"全局位置: {materials['price_position']}%")

    with col2:
        # 波段数据
        if "last_leg" in materials:
            ll = materials["last_leg"]
            st.text(f"最近波段: {'多' if ll['direction']=='bull' else '空'} "
                    f"{ll['bars']}根 范围{ll['range']}pt 实体{ll['body']}")
        if "prev_leg" in materials:
            pl = materials["prev_leg"]
            st.text(f"前一波段: {'多' if pl['direction']=='bull' else '空'} "
                    f"{pl['bars']}根 范围{pl['range']}pt")

    # 额外材料（根据模式）
    if mode == 1:
        # 背景：需要 Swing 位置
        if "last_swing" in materials:
            ls = materials["last_swing"]
            label = "高" if ls["kind"] == "SH" else "低"
            st.text(f"最近Swing{label}: #{ls['index']} ({ls['price']})")
        if "prev_swing" in materials:
            ps = materials["prev_swing"]
            label = "高" if ps["kind"] == "SH" else "低"
            st.text(f"前一个Swing{label}: #{ps['index']} ({ps['price']})")

    if mode in (3, 4, 5):
        # 推进质量/回调/接受：需要回调比例
        if "pullback_ratio" in materials:
            st.text(f"回调/推进比: {materials['pullback_ratio']}x")
        if "body_avg_first" in materials and "body_avg_second" in materials:
            b1 = materials["body_avg_first"]
            b2 = materials["body_avg_second"]
            if b1 > 1e-9:
                st.text(f"实体趋势: 前={b1:.2f} 后={b2:.2f} ({b2/b1:.1f}x)")


def render_answer_history(session):
    """展示训练记录"""
    answers = session.get("answers", [])
    if not answers:
        return

    st.markdown("---")
    with st.expander("训练记录"):
        # 按技能统计
        by_skill = {}
        for a in answers:
            by_skill.setdefault(a.skill_id, []).append(a)

        for sid in sorted(by_skill.keys()):
            skill_name = SKILLS[sid]["name"]
            count = len(by_skill[sid])
            st.text(f"[{skill_name}] {count} 次训练")

        # 最近记录
        st.markdown("**最近 10 条：**")
        for a in answers[-10:]:
            skill_name = SKILLS.get(a.skill_id, {}).get("name", "?")
            st.text(f"[{a.timestamp}] #{a.bar} {skill_name}: {a.answer[:60]}")



# =========================================================
# 主函数
# =========================================================
def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器 V12", layout="wide")

    # Session 初始化
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
        st.session_state["answers"] = []
        st.session_state["train_mode"] = 1

    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("读盘训练器 V12")
        st.caption("训练目标：背景阅读、控制权识别、推进质量、回调vs转换、市场接受")

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
                    st.success(f"{len(df)}根K线, {len(new_legs)}个波段")
                else:
                    st.error("加载失败")

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.markdown("**训练模式选择：**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                if st.button(f"{sid}. {name}", key=f"mode_{sid}"):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            # 训练统计
            answers = st.session_state.get("answers", [])
            st.markdown("---")
            st.text(f"总训练次数: {len(answers)}")

            # 标注控制
            st.markdown("---")
            st.markdown("**标注显示：**")
            st.session_state["show_swings"] = st.checkbox(
                "Swing High/Low", value=True, key="cb_swings")
            st.session_state["show_legs"] = st.checkbox(
                "波段", value=True, key="cb_legs")

    # ---- 主区域 ----
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器 V12")
        st.markdown("")
        st.markdown("## 训练目标")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown(f"**{sid}. {s['name']}** — {s['question']}")
        st.markdown("")
        st.markdown("> 系统不给你答案。你观察、你判断、你记录。")
        st.markdown("> 用 Replay 一根根推进，训练你的**读盘能力**。")
        return

    chart_df = st.session_state["chart_df"]
    swings = st.session_state["swings"]
    legs = st.session_state["legs"]
    bar = st.session_state.get("current_bar", 0)

    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    # K 线图
    show_sw = st.session_state.get("show_swings", True)
    show_lg = st.session_state.get("show_legs", True)
    chart = build_chart(chart_df, swings, legs, bar, show_swings=show_sw, show_legs=show_lg)
    st.plotly_chart(chart, use_container_width=True)

    # Replay 控制（核心交互）
    col_p3, col_p1, col_n1, col_n3, col_n10, col_end = st.columns(6)
    with col_p3:
        if st.button("<<-3", key="b_p3"):
            st.session_state["current_bar"] = max(0, bar - 3)
            st.rerun()
    with col_p1:
        if st.button("<-1", key="b_p1"):
            st.session_state["current_bar"] = max(0, bar - 1)
            st.rerun()
    with col_n1:
        if st.button("+1->", key="b_n1"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 1)
            st.rerun()
    with col_n3:
        if st.button("+3->", key="b_n3"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 3)
            st.rerun()
    with col_n10:
        if st.button("+10->", key="b_n10"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, bar + 10)
            st.rerun()
    with col_end:
        if st.button("末尾", key="b_end"):
            st.session_state["current_bar"] = len(chart_df) - 1
            st.rerun()

    # 当前 bar 信息
    cur = chart_df.iloc[bar]
    st.caption(
        f"#{bar}/{len(chart_df)-1}  "
        f"O:{cur['open']:.2f}  H:{cur['high']:.2f}  "
        f"L:{cur['low']:.2f}  C:{cur['close']:.2f}  "
        f"{cur['close']-cur['open']:+.2f}"
    )

    # 获取原始观察材料
    materials = get_raw_materials(chart_df, swings, legs, bar)

    # 训练面板
    col_train, col_hist = st.columns([2, 1])
    with col_train:
        render_skill_panel(materials, st.session_state, bar)
    with col_hist:
        render_answer_history(st.session_state)


if __name__ == "__main__":
    main()
