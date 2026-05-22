# =========================================================
# Al Brooks 读盘训练器 V11
# =========================================================
#
# 从 V10 提炼：修复所有参数传递 bug，统一函数签名规范。
#
# 核心原则（与 V10 相同）：
#   - 系统不分类，只描述行为变化
#   - 用户自己判断，系统只展示"发生了什么"
#   - 控制权转移是过程（6阶段），不是布尔值
#   - AI 不解释，只指向行为变化位置
#   - Replay 是核心训练方式
#   - 删除：评分/分类器/三档偏差/案例库/压力统计
#
# V11 修复：
#   - 统一所有函数的参数顺序: (chart_df, swings, legs, ...)
#   - safe_legs 在 build_snapshot 顶部统一构建，所有子函数共享
#   - 所有 legs 元素访问前有 hasattr 保护
#
# =========================================================

import os
import json
import time
import textwrap
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter

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

OPENAI_BASE_URL = "https://api.videocaptioner.cn/v1"
OPENAI_MODEL = "gpt-5.4-nano"


# =========================================================
# 数据类
# =========================================================

@dataclass
class SwingPoint:
    index: int
    kind: str    # "SH" or "SL"
    price: float


@dataclass
class StructureLabel:
    index: int
    label: str


@dataclass
class Leg:
    """波段 — 只存原始数据，不分类 momentum"""
    start_idx: int
    end_idx: int
    direction: str
    bar_count: int
    price_start: float
    price_end: float
    price_range: float
    overlap_ratio: float
    body_avg: float
    tail_avg: float


@dataclass
class BehaviorChange:
    """一次行为变化 — 不分类，只描述"""
    what: str           # 什么在变化
    direction: str      # 增加 / 减少 / 稳定
    from_desc: str      # 变化前
    to_desc: str        # 变化后
    bars: str           # K线范围


@dataclass
class DecayTracker:
    """行为衰减追踪 — 连续恶化过程"""
    body_shrinking: int
    hc_decreasing: int
    tail_growing: int
    reversal_frequency: int
    pullback_deepening: str
    breakout_distance: str


@dataclass
class ControlShift:
    """控制权转移 — 6阶段过程，不是布尔值"""
    push_failed: bool
    opposite_testing: str
    original_ft_gone: bool
    opposite_accepted: str
    trapped_formed: bool
    second_attempt: str
    stage: int
    description: str
    phase: str = "无"


@dataclass
class Viewpoint:
    """观点生命周期 — 极简版"""
    bar: int
    direction: str
    expectation: str
    timestamp: str
    status: str       # "active" / "expired"


@dataclass
class Outcome:
    """行为验证 — 只展示原始行为"""
    predict_bar: int
    outcome_bar: int
    move: float
    move_pct: float
    path_observations: list
    suggest_replay_range: tuple


@dataclass
class MarketSnapshot:
    """快照 — 只有原始观察，没有分类"""
    bar_index: int
    time: str
    open: float
    high: float
    low: float
    close: float
    control: list
    location: list
    behavior_changes: list
    decay: list
    control_shift: list
    legs: list
    swings: list


# =========================================================
# 数据加载
# =========================================================

@st.cache_data(ttl=300, show_spinner="正在加载行情数据...")
def load_data(symbol: str = "IF0") -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={
                "datetime": "datetime",
                "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            })
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1)
    st.error(f"数据加载失败: {last_err}")
    return pd.DataFrame()


# =========================================================
# Session 初始化
# =========================================================

def init_session():
    defaults = {
        "logs": [],
        "mode": "自由浏览",
        "viewpoints": [],
        "timeline": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =========================================================
# 检测函数
# 统一参数规范: (chart_df, swings, legs, current_bar)
# 只接收 legs 时用: (chart_df, legs)
# =========================================================

def detect_swings(df: pd.DataFrame) -> list:
    """Swing High/Low 检测"""
    N = SWING_LOOKBACK
    swings = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(N, len(df) - N):
        if all(highs[i] > highs[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SH", price=float(highs[i])))
        if all(lows[i] < lows[j] for j in range(i - N, i + N + 1) if j != i):
            swings.append(SwingPoint(index=i, kind="SL", price=float(lows[i])))
    return swings


def detect_market_structure(swings: list) -> list:
    """HH/HL/LH/LL 标记"""
    labels = []
    for i in range(1, len(swings)):
        prev, curr = swings[i - 1], swings[i]
        if prev.kind == "SH" and curr.kind == "SH":
            labels.append(StructureLabel(index=curr.index,
                            label="HH" if curr.price > prev.price else "LH"))
        elif prev.kind == "SL" and curr.kind == "SL":
            labels.append(StructureLabel(index=curr.index,
                            label="HL" if curr.price > prev.price else "LL"))
    return labels


def detect_legs(df: pd.DataFrame, swings: list) -> list:
    """波段检测"""
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
        bodies, tails = [], []
        for j in range(len(segment)):
            bar = segment.iloc[j]
            rng = bar["high"] - bar["low"]
            if rng > 1e-9:
                bodies.append(abs(bar["close"] - bar["open"]) / rng)
                tails.append((rng - abs(bar["close"] - bar["open"])) / rng)
            else:
                bodies.append(0)
                tails.append(1.0)
        body_avg = float(np.mean(bodies))
        tail_avg = float(np.mean(tails))
        overlap_ratio = 0.0
        if legs:
            pl = legs[-1]
            pt = max(pl.price_start, pl.price_end)
            pb = min(pl.price_start, pl.price_end)
            ct = max(ps, pe)
            cb = min(ps, pe)
            ov = max(0, min(pt, ct) - max(pb, cb))
            un = max(pt, ct) - min(pb, cb)
            overlap_ratio = ov / un if un > 1e-9 else 0
        legs.append(Leg(
            start_idx=s1.index, end_idx=s2.index, direction=direction,
            bar_count=s2.index - s1.index + 1, price_start=ps, price_end=pe,
            price_range=abs(pe - ps), overlap_ratio=round(overlap_ratio, 3),
            body_avg=round(body_avg, 3), tail_avg=round(tail_avg, 3),
        ))
    return legs


def detect_behavior_changes(chart_df: pd.DataFrame, legs: list) -> list:
    """行为变化检测 — 只描述，不分类"""
    changes = []
    n = min(15, len(chart_df))
    if n < 5:
        return []
    recent = chart_df.tail(n)
    mid = n // 2

    # 1. 实体大小变化
    first_half = recent.iloc[:mid]
    second_half = recent.iloc[mid:]
    body_first = (first_half["close"] - first_half["open"]).abs().mean()
    body_second = (second_half["close"] - second_half["open"]).abs().mean()
    if body_first > 1e-9:
        ratio = body_second / body_first
        if ratio < 0.65:
            changes.append(BehaviorChange(
                what="实体大小", direction="减少",
                from_desc=f"{body_first:.2f}", to_desc=f"{body_second:.2f}",
                bars=f"#{mid}-{n-1}"))
        elif ratio > 1.35:
            changes.append(BehaviorChange(
                what="实体大小", direction="增加",
                from_desc=f"{body_first:.2f}", to_desc=f"{body_second:.2f}",
                bars=f"#{mid}-{n-1}"))

    # 2. 重叠变化
    for i in range(max(0, len(recent) - 8), len(recent) - 2):
        b1, b2 = recent.iloc[i], recent.iloc[i + 1]
        overlap = max(0, min(b1["close"], b2["close"]) - max(b1["open"], b2["open"]))
        rng = max(b1["high"], b2["high"]) - min(b1["low"], b2["low"])
        if rng > 1e-9 and overlap / rng > 0.5:
            changes.append(BehaviorChange(
                what="重叠", direction="增加",
                from_desc="正常", to_desc=f"高重叠{overlap/rng:.0%}",
                bars=f"#{i}-{i+1}"))
            break

    # 3. 尾巴长度变化
    tails_first = []
    tails_second = []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng > 1e-9:
            tails_first.append((rng - abs(bar["close"] - bar["open"])) / rng)
        else:
            tails_first.append(1.0)
        if i >= mid:
            tails_second.append(tails_first[-1])
    if tails_first and tails_second:
        avg1 = np.mean(tails_first[:mid]) if mid > 0 else 0
        avg2 = np.mean(tails_second) if tails_second else 0
        if avg1 > 0.1 and avg2 / avg1 > 1.4:
            changes.append(BehaviorChange(
                what="尾长", direction="增加",
                from_desc=f"{avg1:.2f}", to_desc=f"{avg2:.2f}",
                bars=f"后半段"))

    # 4. HC/LC 变化
    if len(recent) >= 10:
        first_10 = recent.iloc[:10]
        second_10 = recent.iloc[-10:] if len(recent) >= 20 else recent.iloc[len(recent)//2:]
        hc1 = sum(1 for i in range(1, len(first_10)) if first_10.iloc[i]["close"] > first_10.iloc[i-1]["close"])
        hc2 = sum(1 for i in range(1, len(second_10)) if second_10.iloc[i]["close"] > second_10.iloc[i-1]["close"])
        if hc1 > 0 and hc2 / hc1 < 0.5:
            changes.append(BehaviorChange(
                what="HC", direction="减少",
                from_desc=str(hc1), to_desc=str(hc2),
                bars="前半 vs 后半"))

    # 5. 波段对比
    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]
    if len(safe_legs) >= 3:
        recent_legs = [l for l in safe_legs if l.end_idx <= len(chart_df) - 1]
        if len(recent_legs) >= 3:
            last3 = recent_legs[-3:]
            body_avgs = [l.body_avg for l in last3]
            if body_avgs[0] > 1e-9 and body_avgs[-1] / body_avgs[0] < 0.65:
                changes.append(BehaviorChange(
                    what="波段实体", direction="减少",
                    from_desc=f"{body_avgs[0]:.2f}", to_desc=f"{body_avgs[-1]:.2f}",
                    bars=f"最近3波段"))

    return changes


def track_decay(chart_df: pd.DataFrame, legs: list) -> DecayTracker:
    """衰减追踪 — 连续恶化过程"""
    n = min(15, len(chart_df))
    if n < 5:
        return DecayTracker(0, 0, 0, 0, "数据不足", "数据不足")

    recent = chart_df.tail(n)
    bodies = (recent["close"] - recent["open"]).abs()

    # 实体连续缩小
    body_shrinking = 0
    for i in range(len(bodies) - 1, 0, -1):
        if bodies.iloc[i] < bodies.iloc[i-1] * 0.85:
            body_shrinking += 1
        else:
            break

    # HC 连续减少
    hc_decreasing = 0
    for i in range(len(recent) - 1, max(0, len(recent) - 8), -1):
        if i > 0 and recent.iloc[i]["close"] < recent.iloc[i-1]["close"]:
            hc_decreasing += 1
        else:
            break

    # 尾巴连续增加
    tail_growing = 0
    tails = []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        tails.append((rng - abs(bar["close"] - bar["open"])) / rng if rng > 1e-9 else 1.0)
    for i in range(len(tails) - 1, 0, -1):
        if tails[i] > tails[i-1] * 1.1:
            tail_growing += 1
        else:
            break

    # 反包频率
    reversal_frequency = 0
    for i in range(1, len(recent)):
        prev, cur = recent.iloc[i-1], recent.iloc[i]
        if (prev["close"] > prev["open"] and cur["close"] < cur["open"]
            and cur["close"] < prev["open"]):
            reversal_frequency += 1
        elif (prev["close"] < prev["open"] and cur["close"] > cur["open"]
              and cur["close"] > prev["open"]):
            reversal_frequency += 1

    # 回调深度趋势
    pullback_deepening = "稳定"
    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]
    if len(safe_legs) >= 3:
        recent_legs = [l for l in safe_legs if l.end_idx <= len(chart_df) - 1]
        if len(recent_legs) >= 2:
            for i in range(len(recent_legs) - 1):
                push = recent_legs[i]
                pull = recent_legs[i + 1]
                if push.price_range > 1e-9:
                    ratio = pull.price_range / push.price_range
                    if ratio > 0.7:
                        pullback_deepening = "加深"
                    elif ratio < 0.3:
                        pullback_deepening = "减弱"

    # 突破距离趋势
    breakout_distance = "稳定"
    if len(safe_legs) >= 3:
        recent_legs = [l for l in safe_legs if l.end_idx <= len(chart_df) - 1]
        if len(recent_legs) >= 2:
            dists = [l.price_range for l in recent_legs[-3:]]
            if len(dists) >= 2 and dists[0] > 1e-9:
                if dists[-1] / dists[0] < 0.5:
                    breakout_distance = "缩小"
                elif dists[-1] / dists[0] > 1.5:
                    breakout_distance = "扩大"

    return DecayTracker(
        body_shrinking=body_shrinking,
        hc_decreasing=hc_decreasing,
        tail_growing=tail_growing,
        reversal_frequency=reversal_frequency,
        pullback_deepening=pullback_deepening,
        breakout_distance=breakout_distance,
    )



# =========================================================
# 观察函数
# 统一参数: (chart_df, swings, legs, current_bar)
# =========================================================

def observe_control(chart_df, swings, legs, current_bar):
    """控制权观察 — 替代 AlwaysIn 分类，只输出原始事实"""
    observations = []
    if len(chart_df) == 0 or current_bar < 0 or current_bar >= len(chart_df):
        return ["数据不足"]

    cur = chart_df.iloc[current_bar]

    # 防御：只取 Leg 实例
    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]

    # 最近波段
    last_leg = next((l for l in reversed(safe_legs) if l.end_idx <= current_bar), None)
    if last_leg:
        observations.append(
            f"最近波段: {last_leg.direction}方向, {last_leg.bar_count}根, "
            f"实体{last_leg.body_avg:.0%}")

    if len(safe_legs) >= 2:
        prev = next((l for l in reversed(safe_legs)
                     if l != last_leg and l.end_idx <= current_bar), None)
        if prev and last_leg:
            if last_leg.direction != prev.direction:
                if last_leg.price_range > 1e-9:
                    ratio = prev.price_range / last_leg.price_range
                    observations.append(f"回调/推进比: {ratio:.1f}x")
            else:
                if last_leg.body_avg > 1e-9 and prev.body_avg > 1e-9:
                    ratio = last_leg.body_avg / prev.body_avg
                    observations.append(f"同方向波段实体比: {ratio:.1f}x")

    # HC/LC
    recent = chart_df.tail(10)
    if len(recent) >= 5:
        hc = sum(1 for i in range(1, len(recent))
                 if recent.iloc[i]["close"] > recent.iloc[i - 1]["close"])
        lc = sum(1 for i in range(1, len(recent))
                 if recent.iloc[i]["close"] < recent.iloc[i - 1]["close"])
        observations.append(f"最近10根: HC={hc}, LC={lc}, 净差={hc - lc:+d}")

    # 最近 Swing
    recent_swings = [s for s in swings if hasattr(s, 'index') and s.index <= current_bar]
    if recent_swings:
        last_swing = recent_swings[-1]
        observations.append(
            f"最近Swing: {'High' if last_swing.kind == 'SH' else 'Low'} "
            f"#{last_swing.index}")

    return observations


def observe_location(chart_df, swings, legs, current_bar):
    """位置观察 — 替代 LocationContext 布尔值，只描述位置事实"""
    observations = []
    if len(chart_df) == 0 or current_bar < 0 or current_bar >= len(chart_df):
        return ["数据不足"]

    cur = chart_df.iloc[current_bar]
    full_high = chart_df["high"].max()
    full_low = chart_df["low"].min()
    full_range = full_high - full_low

    if full_range < 1e-9:
        return ["无足够价格变动"]

    pos = (cur["close"] - full_low) / full_range
    observations.append(f"价格位置: 全局{pos:.0%}")

    # 距前高前低
    prior_highs = [s.price for s in swings if hasattr(s, 'kind') and s.kind == "SH"
                   and hasattr(s, 'index') and s.index <= current_bar]
    prior_lows = [s.price for s in swings if hasattr(s, 'kind') and s.kind == "SL"
                  and hasattr(s, 'index') and s.index <= current_bar]

    if prior_highs:
        highest = max(prior_highs)
        dist = (highest - cur["close"]) / full_range
        observations.append(f"距前高: {dist:.1%}")
        if dist < 0.02:
            observations.append("接近前高")

    if prior_lows:
        lowest = min(prior_lows)
        dist = (cur["close"] - lowest) / full_range
        observations.append(f"距前低: {dist:.1%}")
        if dist < 0.02:
            observations.append("接近前低")

    # 区间检测
    if len(prior_highs) >= 2 and len(prior_lows) >= 2:
        rh = sorted(prior_highs)[-2:]
        rl = sorted(prior_lows)[-2:]
        rt, rb = min(rh), max(rl)
        if rt > rb + full_range * 0.01:
            if rb <= cur["close"] <= rt:
                range_pos = (cur["close"] - rb) / (rt - rb)
                observations.append(
                    f"在区间内: {range_pos:.0%}（区间宽{rt - rb:.2f}）")

    return observations


def detect_control_shift(chart_df, swings, legs, current_bar):
    """控制权转移 — 6阶段过程，不是布尔值"""
    n = min(15, len(chart_df))
    recent = chart_df.tail(n)
    if n < 5:
        return ControlShift(False, "无", False, "无", False, "无", 0, "数据不足")

    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]
    last_leg = next((l for l in reversed(safe_legs) if l.end_idx <= current_bar), None)
    if last_leg is None:
        return ControlShift(False, "无", False, "无", False, "无", 0, "无波段数据")

    original_dir = last_leg.direction

    # 阶段 1: 原方向推进失败
    push_failed = False
    if n >= 3:
        cur = recent.iloc[-1]
        if original_dir == "bull":
            push_failed = cur["close"] < cur["open"]
            if not push_failed and len(recent) >= 3:
                big = recent.iloc[-3]
                if big["close"] > big["open"]:
                    big_range = big["high"] - big["low"]
                    big_body = abs(big["close"] - big["open"])
                    if big_range > 1e-9 and big_body / big_range > 0.5:
                        if cur["close"] < big["open"]:
                            push_failed = True
        else:
            push_failed = cur["close"] > cur["open"]
            if not push_failed and len(recent) >= 3:
                big = recent.iloc[-3]
                if big["close"] < big["open"]:
                    big_range = big["high"] - big["low"]
                    big_body = abs(big["close"] - big["open"])
                    if big_range > 1e-9 and big_body / big_range > 0.5:
                        if cur["close"] > big["open"]:
                            push_failed = True

    # 阶段 2: 反方向测试
    opposite_testing = "无"
    if push_failed:
        if original_dir == "bull":
            prior_lows = [s.price for s in swings
                          if hasattr(s, 'kind') and s.kind == "SL"
                          and hasattr(s, 'index') and s.index <= current_bar]
            if prior_lows:
                lowest = min(prior_lows[-3:])
                if recent.iloc[-1]["low"] <= lowest:
                    opposite_testing = "测试中"
                if recent.iloc[-1]["close"] < lowest:
                    opposite_testing = "测试成功"
        else:
            prior_highs = [s.price for s in swings
                           if hasattr(s, 'kind') and s.kind == "SH"
                           and hasattr(s, 'index') and s.index <= current_bar]
            if prior_highs:
                highest = max(prior_highs[-3:])
                if recent.iloc[-1]["high"] >= highest:
                    opposite_testing = "测试中"
                if recent.iloc[-1]["close"] > highest:
                    opposite_testing = "测试成功"

    # 阶段 3: 原方向跟进消失
    original_ft_gone = False
    if original_dir == "bull":
        hc_count = sum(1 for i in range(max(0, len(recent) - 6), len(recent) - 1)
                       if recent.iloc[i + 1]["close"] > recent.iloc[i]["close"])
        if hc_count <= 1 and len(recent) >= 6:
            original_ft_gone = True
    else:
        lc_count = sum(1 for i in range(max(0, len(recent) - 6), len(recent) - 1)
                       if recent.iloc[i + 1]["close"] < recent.iloc[i]["close"])
        if lc_count <= 1 and len(recent) >= 6:
            original_ft_gone = True

    # 阶段 4: 反方向获得接受
    opposite_accepted = "无"
    if original_dir == "bull" and original_ft_gone:
        bear_bars = []
        for i in range(len(recent) - 1, max(len(recent) - 6, -1), -1):
            if recent.iloc[i]["close"] < recent.iloc[i]["open"]:
                bear_bars.append(i)
        if len(bear_bars) >= 2:
            first_bear_low = recent.iloc[bear_bars[0]]["low"]
            maintained = sum(1 for i in range(bear_bars[0] + 1, len(recent))
                            if recent.iloc[i]["low"] <= first_bear_low * 1.005)
            total = len(recent) - bear_bars[0] - 1
            if total > 0 and maintained / total >= 0.8:
                opposite_accepted = "完全接受"
            elif total > 0 and maintained / total >= 0.6:
                opposite_accepted = "部分接受"
    elif original_dir == "bear" and original_ft_gone:
        bull_bars = []
        for i in range(len(recent) - 1, max(len(recent) - 6, -1), -1):
            if recent.iloc[i]["close"] > recent.iloc[i]["open"]:
                bull_bars.append(i)
        if len(bull_bars) >= 2:
            first_bull_high = recent.iloc[bull_bars[0]]["high"]
            maintained = sum(1 for i in range(bull_bars[0] + 1, len(recent))
                            if recent.iloc[i]["high"] >= first_bull_high * 0.995)
            total = len(recent) - bull_bars[0] - 1
            if total > 0 and maintained / total >= 0.8:
                opposite_accepted = "完全接受"
            elif total > 0 and maintained / total >= 0.6:
                opposite_accepted = "部分接受"

    # 阶段 5: Trapped Trader
    trapped_formed = False
    if opposite_testing == "测试成功" or opposite_accepted in ("部分接受", "完全接受"):
        if len(recent) >= 4:
            for i in range(len(recent) - 2, max(len(recent) - 6, -1), -1):
                bar = recent.iloc[i]
                if original_dir == "bull" and bar["close"] > bar["open"]:
                    if bar["high"] > recent.iloc[-1]["high"]:
                        if recent.iloc[-1]["close"] < bar["close"]:
                            trapped_formed = True
                            break
                elif original_dir == "bear" and bar["close"] < bar["open"]:
                    if bar["low"] < recent.iloc[-1]["low"]:
                        if recent.iloc[-1]["close"] > bar["close"]:
                            trapped_formed = True
                            break

    # 阶段 6: 二次原方向尝试
    second_attempt = "无"
    if trapped_formed and len(recent) >= 2:
        last_bar = recent.iloc[-1]
        if original_dir == "bull" and last_bar["close"] > last_bar["open"]:
            prior_highs = [s.price for s in swings
                           if hasattr(s, 'kind') and s.kind == "SH"
                           and hasattr(s, 'index') and s.index <= current_bar]
            if prior_highs and last_bar["high"] < max(prior_highs[-3:]):
                second_attempt = "尝试中"
        elif original_dir == "bear" and last_bar["close"] < last_bar["open"]:
            prior_lows = [s.price for s in swings
                          if hasattr(s, 'kind') and s.kind == "SL"
                          and hasattr(s, 'index') and s.index <= current_bar]
            if prior_lows and last_bar["low"] > min(prior_lows[-3:]):
                second_attempt = "尝试中"

    # 确定阶段
    stage = 0
    if push_failed: stage = 1
    if opposite_testing != "无": stage = 2
    if original_ft_gone: stage = max(stage, 3)
    if opposite_accepted != "无": stage = max(stage, 4)
    if trapped_formed: stage = max(stage, 5)
    if second_attempt != "无": stage = max(stage, 6)

    # 叙述
    dir_name = "多头" if original_dir == "bull" else "空头"
    opp_name = "空头" if original_dir == "bull" else "多头"
    parts = []
    if stage == 0:
        narrative = f"{dir_name}主导，无转移迹象"
    else:
        if push_failed: parts.append(f"{dir_name}推进失败")
        if opposite_testing != "无": parts.append(f"{opp_name}{opposite_testing}")
        if original_ft_gone: parts.append(f"{dir_name}跟进消失")
        if opposite_accepted != "无": parts.append(f"{opp_name}被{opposite_accepted}")
        if trapped_formed: parts.append("Trapped Trader 形成")
        if second_attempt != "无": parts.append(f"{dir_name}二次尝试{second_attempt}")
        narrative = " -> ".join(parts) if parts else "无明确转移过程"

    phase_map = {
        0: "无转移", 1: "推进失败", 2: "反向测试", 3: "跟进消失",
        4: "反向接受", 5: "Trapped", 6: "二次尝试",
    }

    return ControlShift(
        push_failed=push_failed,
        opposite_testing=opposite_testing,
        original_ft_gone=original_ft_gone,
        opposite_accepted=opposite_accepted,
        trapped_formed=trapped_formed,
        second_attempt=second_attempt,
        stage=stage,
        description=narrative,
        phase=phase_map.get(stage, "未知"),
    )



# =========================================================
# 核心组装函数
# =========================================================

def build_snapshot(chart_df, swings, legs, current_bar, session):
    """组装当前 bar 的市场快照 — 只有原始观察"""
    if current_bar < 0 or current_bar >= len(chart_df):
        return None

    cur = chart_df.iloc[current_bar]

    # 统一 safe_legs，所有子调用共享
    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]

    # 各维度观察
    control_obs = observe_control(chart_df, swings, safe_legs, current_bar)
    location_obs = observe_location(chart_df, swings, safe_legs, current_bar)

    # 行为变化
    behavior_changes = detect_behavior_changes(chart_df, safe_legs)
    bc_texts = [f"{bc.what} {bc.direction}: {bc.from_desc} -> {bc.to_desc}"
                for bc in behavior_changes]

    # 衰减追踪
    decay = track_decay(chart_df, safe_legs)
    decay_texts = []
    if decay:
        max_decay = max(decay.body_shrinking, decay.hc_decreasing,
                       decay.tail_growing, decay.reversal_frequency)
        if max_decay >= 3:
            decay_texts.append(
                f"衰减: 实体缩{decay.body_shrinking}次, HC减{decay.hc_decreasing}次, "
                f"尾增{decay.tail_growing}次, 反包{decay.reversal_frequency}次")
            if decay.pullback_deepening not in ("稳定", "数据不足"):
                decay_texts.append(f"  回调: {decay.pullback_deepening}")

    # 控制权转移
    control_shift = detect_control_shift(chart_df, swings, safe_legs, current_bar)
    shift_texts = []
    if control_shift and control_shift.stage > 0:
        shift_texts.append(f"控制权阶段: {control_shift.phase}")
        shift_texts.append(f"  {control_shift.description}")

    # 波段信息
    leg_texts = []
    recent_legs = [l for l in safe_legs if l.end_idx <= current_bar]
    if recent_legs:
        last = recent_legs[-1]
        leg_texts.append(
            f"最近波段: {'多' if last.direction == 'bull' else '空'} "
            f"#{last.start_idx}-{last.end_idx} ({last.price_range:.2f})")

    # Swing 信息
    swing_texts = []
    recent_swings = [s for s in swings if hasattr(s, 'index') and s.index <= current_bar]
    if len(recent_swings) >= 2:
        last2 = recent_swings[-2:]
        for s in last2:
            swing_texts.append(
                f"Swing{'高' if s.kind == 'SH' else '低'} #{s.index} ({s.price:.2f})")

    return MarketSnapshot(
        bar_index=current_bar,
        time=str(cur.get("datetime", "")),
        open=cur["open"],
        high=cur["high"],
        low=cur["low"],
        close=cur["close"],
        control=control_obs,
        location=location_obs,
        behavior_changes=bc_texts,
        decay=decay_texts,
        control_shift=shift_texts,
        legs=leg_texts,
        swings=swing_texts,
    )


def validate_outcome(chart_df, swings, legs, session, predict_bar, outcome_bar):
    """只展示原始行为，不说对错"""
    if predict_bar is None or outcome_bar is None:
        return None
    if predict_bar >= len(chart_df) or outcome_bar >= len(chart_df):
        return None
    if predict_bar < 0 or outcome_bar < 0:
        return None

    span_start = min(predict_bar, outcome_bar)
    span_end = max(predict_bar, outcome_bar)
    span = chart_df.iloc[span_start:span_end + 1]

    if len(span) < 2:
        return None

    start_price = chart_df.iloc[predict_bar]["close"]
    end_price = chart_df.iloc[outcome_bar]["close"]
    move = end_price - start_price
    move_pct = move / start_price * 100 if start_price > 0 else 0

    path_obs = []

    span_high = span["high"].max()
    span_low = span["low"].min()
    path_obs.append(f"区间: {span_low:.2f} ~ {span_high:.2f}")
    path_obs.append(f"价格: {start_price:.2f} -> {end_price:.2f} ({move:+.2f}, {move_pct:+.1f}%)")

    # 连续同向
    close_changes = span["close"].diff().dropna()
    if len(close_changes) > 0:
        pos_count = (close_changes > 0).sum()
        neg_count = (close_changes < 0).sum()
        path_obs.append(f"收盘变化: {pos_count}次上涨, {neg_count}次下跌")

    # 实体趋势
    bodies = span["close"] - span["open"]
    if len(bodies) >= 3:
        first_third = bodies.iloc[:len(bodies)//3].abs().mean()
        last_third = bodies.iloc[-len(bodies)//3:].abs().mean()
        if first_third > 1e-9:
            ratio = last_third / first_third
            if ratio < 0.6:
                path_obs.append(f"实体趋势: 缩小 ({ratio:.1f}x)")
            elif ratio > 1.4:
                path_obs.append(f"实体趋势: 扩大 ({ratio:.1f}x)")

    # 反方向拒绝
    if move > 0:
        for i in range(span_start, span_end + 1):
            if chart_df.iloc[i]["low"] < start_price - abs(move) * 0.3:
                path_obs.append(f"价格曾被拒绝至 {chart_df.iloc[i]['low']:.2f}")
                break
    elif move < 0:
        for i in range(span_start, span_end + 1):
            if chart_df.iloc[i]["high"] > start_price + abs(move) * 0.3:
                path_obs.append(f"价格曾有反扑至 {chart_df.iloc[i]['high']:.2f}")
                break

    # 控制权
    safe_legs = [l for l in legs if hasattr(l, 'end_idx')]
    shift = detect_control_shift(chart_df, swings, safe_legs, outcome_bar)
    if shift and shift.stage > 0:
        path_obs.append(f"控制权阶段: {shift.phase}")

    # 衰减
    decay = track_decay(chart_df, safe_legs)
    if decay:
        max_decay = max(decay.body_shrinking, decay.hc_decreasing,
                       decay.tail_growing, decay.reversal_frequency)
        if max_decay >= 3:
            path_obs.append(f"存在连续衰减（最强指标: {max_decay}次）")

    suggest_start = max(0, predict_bar - 3)
    suggest_end = min(len(chart_df) - 1, outcome_bar + 3)

    return Outcome(
        predict_bar=predict_bar,
        outcome_bar=outcome_bar,
        move=move,
        move_pct=move_pct,
        path_observations=path_obs,
        suggest_replay_range=(suggest_start, suggest_end),
    )



# =========================================================
# 图表 + AI 指向
# =========================================================

def build_chart(chart_df, swings, legs, current_bar, snapshot, blind_mode=False):
    """K 线图 — 盲测模式下隐藏所有标注"""
    fig = go.Figure()
    visible_end = current_bar + 1
    visible = chart_df.iloc[:visible_end].copy()

    if len(visible) == 0:
        return fig

    fig.add_trace(go.Candlestick(
        x=visible.index,
        open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"],
        name="K线",
        increasing_line_color="#e74c3c",
        decreasing_line_color="#2ecc71",
    ))

    if not blind_mode and snapshot:
        annotations = []

        # 波段标注
        for s in swings:
            if hasattr(s, 'index') and s.index < visible_end:
                label = "SH" if s.kind == "SH" else "SL"
                annotations.append(dict(
                    x=s.index, y=s.price, text=label,
                    showarrow=True, arrowhead=1, arrowcolor="#888",
                    font=dict(size=9, color="#888"),
                    ax=0, ay=-25 if s.kind == "SH" else 25,
                ))

        # 行为变化
        if snapshot.behavior_changes:
            annotations.append(dict(
                x=current_bar, y=chart_df.iloc[current_bar]["high"],
                text="行为变化", showarrow=True, arrowhead=1,
                arrowcolor="#f39c12", font=dict(size=10, color="#f39c12"),
                ax=0, ay=40,
            ))

        # 控制权转移
        if snapshot.control_shift:
            annotations.append(dict(
                x=current_bar, y=chart_df.iloc[current_bar]["low"],
                text="控制权变化", showarrow=True, arrowhead=1,
                arrowcolor="#e74c3c", font=dict(size=10, color="#e74c3c"),
                ax=0, ay=-40,
            ))

        # Swing 连线
        swing_highs = [(s.index, s.price) for s in swings
                       if hasattr(s, 'kind') and s.kind == "SH" and s.index < visible_end]
        swing_lows = [(s.index, s.price) for s in swings
                      if hasattr(s, 'kind') and s.kind == "SL" and s.index < visible_end]

        if len(swing_highs) >= 2:
            sx, sy = zip(*sorted(swing_highs))
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="lines+markers",
                line=dict(color="#3498db", width=1, dash="dash"),
                marker=dict(size=4), showlegend=False))
        if len(swing_lows) >= 2:
            sx, sy = zip(*sorted(swing_lows))
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="lines+markers",
                line=dict(color="#e67e22", width=1, dash="dash"),
                marker=dict(size=4), showlegend=False))

        fig.update_layout(annotations=annotations)

    fig.update_layout(
        height=500,
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        template="plotly_white",
    )
    return fig


def get_ai_pointing(snapshot):
    """AI 只指向行为变化位置，不解释"""
    if snapshot is None:
        return None
    bar = snapshot.bar_index
    suggestions = []

    if snapshot.behavior_changes:
        suggestions.append((max(0, bar - 5), bar))

    if snapshot.decay:
        for d in snapshot.decay:
            if "衰减:" in d:
                try:
                    num = int(''.join(c for c in d if c.isdigit()))
                    if num >= 3:
                        suggestions.append((max(0, bar - num), bar))
                except ValueError:
                    pass

    if snapshot.control_shift:
        suggestions.append((max(0, bar - 3), bar))

    return suggestions if suggestions else None


# =========================================================
# UI 组件（极简）
# =========================================================

def render_observation_panel(snapshot):
    """只展示 5 个维度的原始观察"""
    if snapshot is None:
        st.info("移动K线到某个位置开始观察")
        return

    st.markdown("**控制权**")
    for obs in snapshot.control:
        st.text(obs)

    st.markdown("**位置**")
    for obs in snapshot.location:
        st.text(obs)

    st.markdown("**行为变化**")
    if snapshot.behavior_changes:
        for bc in snapshot.behavior_changes:
            st.text(bc)
    else:
        st.text("暂未检测到显著行为变化")

    if snapshot.decay:
        st.markdown("**衰减**")
        for d in snapshot.decay:
            st.text(d)

    if snapshot.control_shift:
        st.markdown("**控制权转移**")
        for s in snapshot.control_shift:
            st.text(s)

    with st.expander("波段 / Swing"):
        for t in snapshot.legs:
            st.text(t)
        for t in snapshot.swings:
            st.text(t)


def render_viewpoint_panel(session, current_bar):
    """观点面板 — 强制形成预期，系统只记录"""
    st.markdown("---")
    st.markdown("**你的观点**")

    active = [v for v in session.get("viewpoints", []) if v.status == "active"]
    if active:
        for v in active:
            st.text(f"#{v.bar} 预期{v.direction}: {v.expectation}")
            if st.button(f"失效观点 #{v.bar}", key=f"expire_{v.bar}"):
                v.status = "expired"
                st.rerun()
    else:
        st.text("尚未形成观点")

    st.markdown("**记录观点**")
    col1, col2 = st.columns(2)
    with col1:
        direction = st.selectbox("方向", ["多", "空", "观望"], key="vp_dir")
    with col2:
        expectation = st.text_input(
            "预期描述", key="vp_exp",
            placeholder="例如：突破前高后回踩不破，预期继续上涨")

    if st.button("记录", key="record_vp") and expectation.strip():
        vp = Viewpoint(
            bar=current_bar,
            direction=direction,
            expectation=expectation.strip(),
            timestamp=datetime.now().strftime("%H:%M:%S"),
            status="active",
        )
        session.setdefault("viewpoints", []).append(vp)
        st.rerun()


def render_timeline(session):
    """行为演化时间线"""
    timeline = session.get("timeline", [])
    if not timeline:
        return

    st.markdown("---")
    st.markdown("**时间线**")
    for entry in timeline[-15:]:
        st.text(f"[{entry.get('time', '')}] #{entry.get('bar', '')} {entry.get('text', '')}")


def render_outcome_panel(outcome, session):
    """结果面板 — 只展示原始行为"""
    if outcome is None:
        return

    st.markdown("---")
    st.markdown("**发生了什么**")
    for obs in outcome.path_observations:
        st.text(obs)

    if outcome.suggest_replay_range:
        start, end = outcome.suggest_replay_range
        st.markdown(f"**建议重新观察**: 第{start}根 ~ 第{end}根")
        if st.button("跳转到建议范围"):
            session["current_bar"] = start
            st.rerun()



# =========================================================
# 主函数 — Replay 是核心
# =========================================================

def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器 V11", layout="wide")
    init_session()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("V11 读盘训练器")

        symbol = st.text_input("合约代码", value="rb2510", key="symbol_input")
        blind_mode = st.checkbox("盲测模式（隐藏标注）", value=False, key="blind_toggle")

        if st.button("加载数据", key="load_btn"):
            with st.spinner("加载中..."):
                df = load_data(symbol)
                if df is not None and len(df) > 0:
                    new_swings = detect_swings(df)
                    new_structure = detect_market_structure(new_swings)
                    new_legs = detect_legs(df, new_swings)
                    st.session_state["chart_df"] = df
                    st.session_state["swings"] = new_swings
                    st.session_state["structure"] = new_structure
                    st.session_state["legs"] = new_legs
                    st.session_state["current_bar"] = min(30, len(df) - 1)
                    st.session_state["viewpoints"] = []
                    st.session_state["timeline"] = []
                    st.session_state["data_loaded"] = True
                    st.success(f"已加载 {len(df)} 根K线, {len(new_legs)} 个波段")
                else:
                    st.error("数据加载失败")

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            vp_active = len([v for v in st.session_state.get("viewpoints", [])
                            if v.status == "active"])
            vp_total = len(st.session_state.get("viewpoints", []))
            tl = len(st.session_state.get("timeline", []))
            st.text(f"活跃观点: {vp_active}")
            st.text(f"总观点: {vp_total}")
            st.text(f"时间线: {tl}")

    # ---- 主区域 ----
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器 V11")
        st.markdown("加载合约数据开始训练。")
        st.markdown("---")
        st.markdown("**核心原则：**")
        st.markdown("- 系统只展示行为变化，不分类、不评分")
        st.markdown("- 控制权是连续过程，不是布尔值")
        st.markdown("- Replay 是核心训练方式")
        st.markdown("- 你看的是市场，不是系统")
        return

    chart_df = st.session_state["chart_df"]
    swings = st.session_state["swings"]
    legs = st.session_state["legs"]
    current_bar = st.session_state["current_bar"]

    if current_bar is None or current_bar >= len(chart_df):
        return

    # 构建快照
    snapshot = build_snapshot(chart_df, swings, legs, current_bar, st.session_state)

    # ---- K 线图 ----
    chart = build_chart(chart_df, swings, legs, current_bar, snapshot, blind_mode)
    st.plotly_chart(chart, use_container_width=True)

    # ---- Replay 控制 ----
    col_prev, col_next, col_jump, col_reveal = st.columns(4)
    with col_prev:
        if st.button("<< 前5根", key="prev5"):
            st.session_state["current_bar"] = max(0, current_bar - 5)
            st.rerun()
    with col_next:
        if st.button("后5根 >>", key="next5"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 5)
            st.rerun()
    with col_jump:
        if st.button("后15根 >>>", key="next15"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 15)
            st.rerun()
    with col_reveal:
        if st.button("揭示后20根", key="reveal20"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 20)
            st.rerun()

    # 当前 bar 信息
    cur = chart_df.iloc[current_bar]
    st.text(f"#{current_bar}  O:{cur['open']:.2f}  H:{cur['high']:.2f}  "
            f"L:{cur['low']:.2f}  C:{cur['close']:.2f}")

    # ---- 面板 ----
    col_obs, col_vp = st.columns([1, 1])

    with col_obs:
        render_observation_panel(snapshot)

        if not blind_mode:
            pointing = get_ai_pointing(snapshot)
            if pointing:
                st.markdown("---")
                st.markdown("**建议重新观察**")
                for start, end in pointing:
                    st.text(f"第{start}根 ~ 第{end}根")

    with col_vp:
        render_viewpoint_panel(st.session_state, current_bar)

        # 验证结果
        active_vp = [v for v in st.session_state.get("viewpoints", []) if v.status == "active"]
        if active_vp and st.button("查看结果", key="check_outcome"):
            vp = active_vp[-1]
            outcome_bar = min(len(chart_df) - 1, current_bar + 20)
            outcome = validate_outcome(
                chart_df, swings, legs, st.session_state, vp.bar, outcome_bar)
            if outcome:
                render_outcome_panel(outcome, st.session_state)
                vp.status = "expired"
                tl_entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "bar": current_bar,
                    "text": (f"观点#{vp.bar}({vp.direction}) -> #{outcome_bar} "
                             f"({outcome.move:+.2f})"),
                }
                st.session_state.setdefault("timeline", []).append(tl_entry)

    render_timeline(st.session_state)


if __name__ == "__main__":
    main()
