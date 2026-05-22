# =========================================================
# Al Brooks 读盘训练器 V9
# =========================================================
#
# V9 核心重构：
#   1. 状态转移系统 — 不分类市场，描述市场正在变成什么
#   2. Follow Through Acceptance Engine — 不数K线，观察市场是否接受价格
#   3. 观点生命周期 — 强制预期 -> 观点更新 -> 观点失效 -> 复盘
#   4. 失败后行为追踪 — 不只是失败突破，而是失败后发生了什么
#   5. 盲测模式 — 隐藏系统辅助，用户自己读图
#   6. 偏差统计三档 — 短期5次/中期20次/长期100次
#   7. "为什么失败" — 指出忽略的行为变化，不只是对错
#
# V9 核心理念：
#   市场不是"是什么"，而是"正在变成什么"
#   不追求数值精确，追求相对变化
#   观察 > 指标，训练 > 面板，闭环 > 展示
#
# =========================================================

import os
import json
import time
import textwrap
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import Counter
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak
from openai import OpenAI

# =========================================================
# 常量
# =========================================================

CHART_WINDOW = 80
LOOKBACK_MIN = 120
LOOKAHEAD_RESERVE = 10
SWING_LOOKBACK = 2
RANDOM_MIN_DISTANCE = 50

FUTURES_SYMBOLS = [
    "IF0", "IC0", "IH0",
    "RB0",
    "AU0", "AG0",
    "SC0",
]

STRUCTURE_EVENTS = [
    "失败突破", "楔形", "紧密通道", "扩张三角形",
    "微型双顶", "微型双底", "高潮衰竭",
    "重叠增加", "尾巴增加", "突破后跟进弱",
]

BULL_PRESSURE_PATTERNS = [
    "突破距离缩短",
    "实体缩小",
    "跟进减少（HC减少）",
    "重叠增加",
    "尾盘收不住（上影线增加）",
    "二次突破失败",
    "通道上轨受压",
    "成交量萎缩",
]

BEAR_PRESSURE_PATTERNS = [
    "下跌后快速拉回",
    "空头无法收盘新低",
    "阳线反包增加",
    "下影线增多",
    "底部支撑测试频繁",
    "空头推进变短",
    "区间下沿反弹",
    "买盘涌入",
]

CASE_SCENARIOS = {
    "趋势衰减": "检测到推进衰减的K线位置",
    "假突破": "检测到失败突破的K线位置",
    "区间交易": "检测到区间行为的K线位置",
    "Always In转换": "检测到Always In从Long/Short切换的位置",
    "反转尝试": "检测到趋势末端可能反转的位置",
    "跟进衰竭": "检测到跟进消失的K线位置",
}


# =========================================================
# V9 核心：状态转移系统
# =========================================================

class MarketState(Enum):
    """市场状态 — 不是分类，而是描述性标签"""
    STRONG_TREND = "强趋势"
    TREND = "趋势"
    TREND_DECAYING = "趋势衰减"
    TWO_SIDED = "双向交易"
    RANGE_FORMING = "正在区间化"
    RANGE = "区间"
    BREAKOUT_ATTEMPT = "突破尝试"
    REVERSAL_ATTEMPT = "反转尝试"


@dataclass
class StateTransition:
    """V9 核心：状态转移。不回答'市场是什么'，而是'市场正在变成什么'。"""
    current_state: str
    previous_state: str
    transition_direction: str    # 加强 / 衰减 / 转换 / 不变
    trigger_events: list
    confidence: str              # 明确 / 模糊 / 矛盾
    state_history: list          # 最近转移历史（最多5条）

    def to_display(self) -> str:
        if self.transition_direction == "不变":
            return f"{self.current_state}（维持）"
        arrow_map = {"加强": "→ 加速", "衰减": "→ 减速", "转换": f"→ {self.current_state}"}
        arrow = arrow_map.get(self.transition_direction, "")
        return f"{self.previous_state} {arrow}（{self.confidence}）"


@dataclass
class FollowThroughAcceptance:
    """V9: Follow Through Acceptance Engine。观察市场是否接受了价格。"""
    acceptance_level: str        # 被接受 / 部分接受 / 被拒绝 / 无明确方向
    breakthrough_prior: bool     # 突破前高/前低
    maintaining_breakzone: bool  # 维持在突破区域
    quick_rejection: bool        # 快速回撤
    trapped_opposite: bool       # 对手被困
    opposite_pressure: bool      # 反向压力
    detail: dict


@dataclass
class PostFailureBehavior:
    """V9: 失败后行为追踪。不只是标记失败，而是追踪失败之后市场做了什么。"""
    failure_detected: bool
    failure_type: str
    rapid_reversal: bool
    strong_opposite_ft: bool
    trapped_traders_formed: bool
    measured_move_failure: bool
    second_failure: bool
    continuation_after_failure: bool
    description: str


@dataclass
class Viewpoint:
    """V9: 观点生命周期。用户持续更新观点，不是一次判断。"""
    state: str
    direction: str
    expectation: str
    invalidate_cond: str
    ft_cond: str
    created_at: str
    updated_at: str
    bars_alive: int
    updates_count: int


# =========================================================
# 基础数据类
# =========================================================

@dataclass
class SwingPoint:
    index: int
    kind: str
    price: float

@dataclass
class StructureLabel:
    index: int
    label: str

@dataclass
class Leg:
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
    momentum: str

@dataclass
class LocationContext:
    near_prior_high: bool
    near_prior_low: bool
    in_range: bool
    range_position: str
    near_channel_line: str
    breakout_pullback_area: bool
    climactic_extension: bool
    measured_move_level: str

@dataclass
class AlwaysIn:
    status: str
    evidence: list
    conviction: str

@dataclass
class PressureSnapshot:
    follow_through: str
    ft_detail: dict
    momentum_shift: str
    momentum_detail: dict
    range_progression: str
    range_detail: dict

@dataclass
class MarketTendency:
    primary: str
    secondary: str
    mixed_signals: list
    state_transition: StateTransition

@dataclass
class Outcome:
    got_follow_through: bool
    trapped_traders: bool
    breakout_succeeded: bool
    reversal_held: bool
    range_continued: bool
    description: str
    what_you_missed: str
    failure_category: str

@dataclass
class MarketSnapshot:
    swings: list
    labels: list
    legs: list
    location: LocationContext
    always_in: AlwaysIn
    pressure: PressureSnapshot
    tendency: MarketTendency
    auto_tags: list
    ft_acceptance: FollowThroughAcceptance
    post_failure: PostFailureBehavior
    state_transition: StateTransition

# =========================================================
# 数据加载
# =========================================================

@st.cache_data(ttl=300, show_spinner="正在加载行情数据...")
def load_data(symbol: str = "IF0") -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="30")
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1)
    else:
        raise RuntimeError(f"数据加载失败（重试 3 次）：{last_err}")
    if df.empty:
        raise ValueError("返回数据为空")
    expected = ["datetime", "open", "high", "low", "close", "volume", "hold"]
    raw_cols = df.columns.tolist()
    rename_map = {col: expected[i] for i, col in enumerate(raw_cols) if i < len(expected)}
    df.rename(columns=rename_map, inplace=True)
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{missing}")
    df = df[keep].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) < LOOKBACK_MIN + LOOKAHEAD_RESERVE:
        raise ValueError(f"数据量不足（{len(df)} 行）")
    return df


# =========================================================
# Session
# =========================================================

def init_session():
    defaults = {
        "logs": [],
        "current_index": None,
        "submit_count": 0,
        "mode": "自由浏览",
        "replay_positions": [],
        "replay_cursor": 0,
        "replay_judgments": {},
        "replay_count": 0,
        "forced_review": False,
        "force_review_bar": None,
        "case_mode": None,
        "case_positions": [],
        # V9 新增
        "active_viewpoint": None,
        "viewpoint_history": [],
        "replay_sub_mode": "标准",
        "blind_mode": False,
        "state_history": [],
        "last_transition": None,
        "case_cursor": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =========================================================
# Swing + HH/HL/LH/LL
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


def detect_market_structure(swings: list) -> list:
    labels = []
    for seq, tag_pos, tag_neg in [
        ([s for s in swings if s.kind == "SH"], "HH", "LH"),
        ([s for s in swings if s.kind == "SL"], "HL", "LL"),
    ]:
        for i in range(1, len(seq)):
            tag = tag_pos if seq[i].price > seq[i - 1].price else tag_neg
            labels.append(StructureLabel(index=seq[i].index, label=tag))
    return labels

# =========================================================
# Leg Engine
# =========================================================

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
        momentum = "强推进" if body_avg > 0.6 and tail_avg < 0.5 else (
            "正常推进" if body_avg > 0.4 and tail_avg < 0.8 else "弱推进"
        )
        legs.append(Leg(
            start_idx=s1.index, end_idx=s2.index, direction=direction,
            bar_count=s2.index - s1.index + 1, price_start=ps, price_end=pe,
            price_range=abs(pe - ps), overlap_ratio=round(overlap_ratio, 3),
            body_avg=round(body_avg, 3), tail_avg=round(tail_avg, 3), momentum=momentum,
        ))
    return legs


# =========================================================
# Location Engine
# =========================================================

def detect_location(chart_df, swings, legs, current_bar) -> LocationContext:
    if len(chart_df) == 0 or current_bar < 0:
        return LocationContext(False, False, False, "", "", False, False, "")
    cur = chart_df.iloc[current_bar]
    full_high = chart_df["high"].max()
    full_low = chart_df["low"].min()
    full_range = full_high - full_low
    recent_swings = [s for s in swings if s.index <= current_bar]
    prior_highs = [s.price for s in recent_swings if s.kind == "SH"]
    prior_lows = [s.price for s in recent_swings if s.kind == "SL"]

    near_prior_high = False
    near_prior_low = False
    if prior_highs and full_range > 1e-9:
        highest = max(prior_highs)
        near_prior_high = (cur["high"] - highest) > -full_range * 0.02 and cur["high"] <= highest
    if prior_lows and full_range > 1e-9:
        lowest = min(prior_lows)
        near_prior_low = (cur["low"] - lowest) < full_range * 0.02 and cur["low"] >= lowest

    in_range, range_position = False, ""
    if len(prior_highs) >= 2 and len(prior_lows) >= 2 and full_range > 1e-9:
        rh = sorted(prior_highs)[-2:]
        rl = sorted(prior_lows)[-2:]
        rt, rb = min(rh), max(rl)
        if rt > rb + full_range * 0.01:
            in_range = rb <= cur["close"] <= rt
            pos = (cur["close"] - rb) / (rt - rb)
            range_position = "区间上部" if pos > 0.7 else ("区间下部" if pos < 0.3 else "区间中部")

    near_channel_line = ""
    if len(legs) >= 3:
        for direction_set, line_name, compare_fn in [
            ([l for l in legs if l.direction == "bull" and l.end_idx <= current_bar],
             "通道上沿", lambda c, p: c["high"] >= p * 0.998),
            ([l for l in legs if l.direction == "bear" and l.end_idx <= current_bar],
             "通道下沿", lambda c, p: c["low"] <= p * 1.002),
        ]:
            if len(direction_set) >= 2:
                edge = max(direction_set[-1].price_start, direction_set[-1].price_end)
                if compare_fn(cur, edge):
                    near_channel_line = line_name
                    break

    breakout_pullback_area = (near_prior_high and cur["close"] < cur["open"]) or \
                              (near_prior_low and cur["close"] > cur["open"])

    climactic_extension = False
    if len(legs) >= 2:
        last = next((l for l in reversed(legs) if l.end_idx <= current_bar), None)
        if last:
            for pl in reversed(legs):
                if pl != last and pl.direction == last.direction:
                    if last.price_range > pl.price_range * 2 and last.bar_count < pl.bar_count * 0.7:
                        climactic_extension = True
                    break

    measured_move_level = ""
    if len(legs) >= 2:
        recent_legs = [l for l in legs if l.end_idx <= current_bar]
        for i in range(len(recent_legs) - 1, 0, -1):
            if recent_legs[i].direction == recent_legs[i - 1].direction:
                r1, r2 = recent_legs[i - 1].price_range, recent_legs[i].price_range
                if r1 > 1e-9:
                    ratio = r2 / r1
                    measured_move_level = "等幅目标已到" if ratio >= 0.95 else (
                        "接近等幅目标" if ratio >= 0.75 else "")
                break

    return LocationContext(
        near_prior_high, near_prior_low, in_range, range_position,
        near_channel_line, breakout_pullback_area, climactic_extension, measured_move_level,
    )


# =========================================================
# Always In Engine
# =========================================================

def detect_always_in(chart_df, legs, swings, current_bar) -> AlwaysIn:
    evidence = []
    last_leg = next((l for l in reversed(legs) if l.end_idx <= current_bar), None)
    if last_leg is None:
        return AlwaysIn("Always In Transition", ["数据不足"], "弱")
    if last_leg.direction == "bull":
        evidence.append(f"最近波段多头推进（{last_leg.bar_count}根，实体{last_leg.body_avg:.0%}）")
    else:
        evidence.append(f"最近波段空头推进（{last_leg.bar_count}根，实体{last_leg.body_avg:.0%}）")

    if len(legs) >= 2:
        prev = next((l for l in reversed(legs) if l != last_leg and l.end_idx <= current_bar), None)
        if prev:
            if last_leg.direction == "bull" and prev.direction == "bear":
                mid = (last_leg.price_start + last_leg.price_end) / 2
                if prev.price_end > mid:
                    evidence.append("空头回调深入推进区域")
                else:
                    evidence.append("空头回调未覆盖推进50%")
            elif last_leg.direction == "bear" and prev.direction == "bull":
                mid = (last_leg.price_start + last_leg.price_end) / 2
                if prev.price_end < mid:
                    evidence.append("多头反弹深入推进区域")
                else:
                    evidence.append("多头反弹未覆盖推进50%")

    recent = chart_df.tail(10)
    if len(recent) >= 3:
        hc = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["close"] > recent.iloc[i - 1]["close"])
        lc = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["close"] < recent.iloc[i - 1]["close"])
        if hc >= 7:
            evidence.append(f"Higher Close 偏多（{hc}/10）")
        elif lc >= 7:
            evidence.append(f"Lower Close 偏空（{lc}/10）")
        elif abs(hc - lc) <= 1:
            evidence.append(f"HC/LC 均衡（{hc}v{lc}/10）")

    recent_swings = [s for s in swings if s.index <= current_bar]
    if recent_swings:
        ls = recent_swings[-1]
        evidence.append(f"最近Swing={'High' if ls.kind == 'SH' else 'Low'} #{ls.index}")

    if len(legs) >= 3:
        same = [l for l in legs if l.end_idx <= current_bar and l.direction == last_leg.direction]
        if len(same) >= 2:
            r1, r2 = same[-2].price_range, same[-1].price_range
            if r1 > 1e-9:
                ratio = r2 / r1
                if ratio < 0.5:
                    evidence.append(f"同方向波段缩小至{ratio:.0%}")
                elif ratio > 1.3:
                    evidence.append(f"同方向波段扩大至{ratio:.0%}")

    bull_e = sum(1 for e in evidence if any(w in e for w in ["多头", "Higher Close", "偏多", "空头回调未覆盖"]))
    bear_e = sum(1 for e in evidence if any(w in e for w in ["空头", "Lower Close", "偏空", "多头反弹未覆盖"]))
    if bull_e > bear_e + 1:
        return AlwaysIn("Always In Long", evidence, "强" if bull_e >= 3 else "中")
    elif bear_e > bull_e + 1:
        return AlwaysIn("Always In Short", evidence, "强" if bear_e >= 3 else "中")
    else:
        return AlwaysIn("Always In Transition", evidence, "中" if (bull_e + bear_e) >= 4 else "弱")

# =========================================================
# 压力观察（只描述，不打分）
# =========================================================

def observe_follow_through(df: pd.DataFrame) -> tuple:
    n = min(5, len(df))
    recent = df.tail(n)
    if len(recent) < 2:
        return "数据不足", {}
    net = recent.iloc[-1]["close"] - recent.iloc[0]["open"]
    direction = "bull" if net > 0 else ("bear" if net < 0 else "neutral")
    consecutive = 0
    for i in range(len(recent) - 1, -1, -1):
        bar = recent.iloc[i]
        if (direction == "bull" and bar["close"] > bar["open"]) or \
           (direction == "bear" and bar["close"] < bar["open"]):
            consecutive += 1
        else:
            break
    extreme_count = 0
    hc_count = reversal = 0
    for i in range(1, len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng > 1e-9:
            if direction == "bull" and (bar["close"] - bar["low"]) / rng > 0.66:
                extreme_count += 1
            elif direction == "bear" and (bar["high"] - bar["close"]) / rng > 0.66:
                extreme_count += 1
        prev = recent.iloc[i - 1]
        if direction == "bull" and bar["close"] > prev["close"]:
            hc_count += 1
        elif direction == "bear" and bar["close"] < prev["close"]:
            hc_count += 1
        if direction == "bull" and bar["close"] < prev["open"] and prev["close"] > prev["open"]:
            reversal += 1
        elif direction == "bear" and bar["close"] > prev["open"] and prev["close"] < prev["open"]:
            reversal += 1
    detail = {"方向": direction, "连续同向实体": consecutive,
              "收盘靠近极端": f"{extreme_count}/{n-1}",
              "HC/LC": f"{hc_count}/{n-1}", "反包": reversal}
    if direction == "neutral":
        return "无明确方向", detail
    if reversal >= 2 and consecutive <= 1:
        return "反包", detail
    if consecutive >= 3 and extreme_count >= 2:
        return "跟进强", detail
    if consecutive >= 2 or extreme_count >= 2:
        return "跟进弱", detail
    return "无跟进", detail


def observe_momentum(df: pd.DataFrame) -> tuple:
    n = min(10, len(df))
    recent = df.tail(n)
    if n < 4:
        return "数据不足", {}
    body_r, tail_r, ranges, overlaps = [], [], [], []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        body = abs(bar["close"] - bar["open"])
        body_r.append(body / rng if rng > 1e-9 else 0)
        tail_r.append((rng - body) / rng if rng > 1e-9 else 0)
        ranges.append(rng)
        if i > 0:
            prev = recent.iloc[i - 1]
            ov = max(0, min(prev["high"], bar["high"]) - max(prev["low"], bar["low"]))
            un = max(prev["high"], bar["high"]) - min(prev["low"], bar["low"])
            overlaps.append(ov / un if un > 1e-9 else 0)
        else:
            overlaps.append(0)
    mid = n // 2
    fb, sb = np.mean(body_r[:mid]), np.mean(body_r[mid:])
    ft, st_ = np.mean(tail_r[:mid]), np.mean(tail_r[mid:])
    fo, so = np.mean(overlaps[:mid]) if mid > 0 else 0, np.mean(overlaps[mid:]) if n - mid > 0 else 0
    fr, sr = np.mean(ranges[:mid]), np.mean(ranges[mid:])
    detail = {"实体": f"{fb:.2f}->{sb:.2f}", "尾巴": f"{ft:.2f}->{st_:.2f}",
              "重叠": f"{fo:.2f}->{so:.2f}", "振幅": f"{fr:.2f}->{sr:.2f}"}
    d = e = 0
    if sb < fb * 0.75: d += 1
    elif sb > fb * 1.2: e += 1
    if st_ > ft * 1.3: d += 1
    if so > fo * 1.3 and fo > 0.1: d += 1
    if sr < fr * 0.75: d += 1
    elif sr > fr * 1.2: e += 1
    if e >= 2 and d == 0: return "推进增强", detail
    if d >= 3: return "严重衰减", detail
    if d >= 2: return "明显衰减", detail
    if d >= 1: return "轻微衰减", detail
    return "推进稳定", detail


def observe_range_formation(df: pd.DataFrame) -> tuple:
    n = min(15, len(df))
    recent = df.tail(n)
    if n < 5:
        return "数据不足", {}
    overlaps, tail_r, bodies = [], [], []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        body = abs(bar["close"] - bar["open"])
        rng = bar["high"] - bar["low"]
        tail_r.append((rng - body) / body if body > 1e-9 else 5.0)
        bodies.append(body)
        if i > 0:
            prev = recent.iloc[i - 1]
            ov = max(0, min(prev["high"], bar["high"]) - max(prev["low"], bar["low"]))
            un = max(prev["high"], bar["high"]) - min(prev["low"], bar["low"])
            overlaps.append(ov / un if un > 1e-9 else 0)
    avg_ov = float(np.mean(overlaps)) if overlaps else 0
    avg_tail = float(np.mean(tail_r))
    mid = n // 2
    bs = np.mean(bodies[mid:]) / np.mean(bodies[:mid]) if np.mean(bodies[:mid]) > 1e-9 else 1.0
    ref = recent.iloc[:int(n * 0.6)]
    test = recent.iloc[int(n * 0.6):]
    att = fail = 0
    if len(ref) > 0 and len(test) > 0:
        rh, rl = ref["high"].max(), ref["low"].min()
        for i in range(len(test)):
            bar = test.iloc[i]
            if bar["high"] > rh:
                att += 1
                if bar["close"] < rh: fail += 1
            if bar["low"] < rl:
                att += 1
                if bar["close"] > rl: fail += 1
    bf = fail / att if att > 0 else 0
    detail = {"重叠": f"{avg_ov:.0%}", "尾巴/实体": f"{avg_tail:.1f}x",
              "实体缩小": f"{bs:.0%}", "突破失败率": f"{bf:.0%}"}
    s = 0
    if avg_ov > 0.5: s += 1
    if avg_tail > 1.5: s += 1
    if bs < 0.7: s += 1
    if bf > 0.5: s += 1
    if s >= 3: return "明确区间", detail
    if s >= 2: return "正在区间化", detail
    if s >= 1: return "趋势偏区间", detail
    return "明确趋势", detail


# =========================================================
# V9 核心：状态转移引擎
# =========================================================

def compute_state_transition(
    momentum_shift: str, ft: str, range_prog: str,
    always_in: AlwaysIn, location: LocationContext,
    legs: list, chart_df: pd.DataFrame, current_bar: int,
    prev_state: str = "",
) -> StateTransition:
    """
    V9 核心：判断市场正在经历什么状态转移。
    不输出'市场是什么'，而是'市场正在变成什么'。
    每个判断都是相对变化，不是绝对阈值。
    """
    trigger_events = []
    current = ""
    direction = "不变"
    confidence = "模糊"

    # --- 综合观察相对变化 ---

    # 推进在变化
    momentum_changing = momentum_shift in ("轻微衰减", "明显衰减", "严重衰减", "推进增强")

    # 跟进在变化
    ft_changing = ft in ("跟进强", "跟进弱", "反包")

    # 区间化在变化
    range_changing = range_prog in ("正在区间化", "明确区间", "趋势偏区间")

    # Always In 方向
    ai_long = "Long" in always_in.status
    ai_short = "Short" in always_in.status
    ai_transition = "Transition" in always_in.status

    # --- 确定当前状态 ---

    # 趋势加速
    if momentum_shift == "推进增强" and ft == "跟进强" and (ai_long or ai_short):
        current = "强趋势"
        direction = "加强"
        trigger_events.append("推进增强 + 跟进强")
        confidence = "明确"

    # 趋势衰减中
    elif momentum_changing and (momentum_shift in ("轻微衰减", "明显衰减", "严重衰减")):
        if ai_long or ai_short:
            current = "趋势衰减"
            direction = "衰减"
            trigger_events.append(f"推进{momentum_shift}")
            if ft in ("无跟进", "反包"):
                trigger_events.append(f"跟进{ft}")
            confidence = "明确" if len(trigger_events) >= 2 else "模糊"
        else:
            current = "双向交易"
            direction = "转换"
            trigger_events.append("Always In 过渡 + 推进衰减")
            confidence = "模糊"

    # 双向交易
    elif ai_transition:
        current = "双向交易"
        direction = "转换"
        trigger_events.append("Always In 过渡")
        if ft in ("反包", "无跟进"):
            trigger_events.append(f"跟进{ft}")
        confidence = "明确" if len(trigger_events) >= 2 else "模糊"

    # 区间化
    elif range_changing:
        current = "正在区间化" if range_prog == "正在区间化" else (
            "区间" if range_prog == "明确区间" else "趋势偏区间")
        direction = "转换" if range_prog in ("正在区间化", "区间") else "衰减"
        trigger_events.append(f"区间化: {range_prog}")
        confidence = "明确" if range_prog == "明确区间" else "模糊"

    # 反转尝试
    elif location.climactic_extension or location.measured_move_level == "等幅目标已到":
        current = "反转尝试"
        direction = "转换"
        trigger_events.append("高潮延伸或等幅目标已到")
        confidence = "模糊"

    # 突破尝试
    elif location.near_prior_high or location.near_prior_low:
        current = "突破尝试"
        direction = "不变"
        trigger_events.append("接近关键价位")
        confidence = "模糊"

    # 强趋势维持
    elif (ai_long or ai_short) and always_in.conviction == "强":
        current = "强趋势"
        direction = "不变"
        trigger_events.append(f"Always In {'Long' if ai_long else 'Short'} 强")
        confidence = "明确"

    # 趋势维持
    elif (ai_long or ai_short) and not momentum_changing and not range_changing:
        current = "趋势"
        direction = "不变"
        confidence = "模糊"

    else:
        current = "双向交易"
        direction = "不变"
        confidence = "模糊"

    # --- 矛盾检测 ---
    if confidence == "明确":
        if ft == "跟进强" and momentum_shift in ("明显衰减", "严重衰减"):
            confidence = "矛盾"
            trigger_events.append("矛盾：跟进强但推进衰减")
        if (ai_long or ai_short) and range_prog in ("正在区间化", "明确区间"):
            confidence = "矛盾"
            trigger_events.append(f"矛盾：Always In 但{range_prog}")

    # --- 状态历史 ---
    # Use external state history if available (from session_state), else empty
    state_history_external = []
    if hasattr(st, 'session_state'):
        state_history_external = list(st.session_state.get("state_history", []))
    history = list(state_history_external)
    history_summary = []
    if prev_state:
        history.append(f"{prev_state}->{current}")
        if len(history) > 10:
            history = history[-10:]
        if hasattr(st, 'session_state'):
            st.session_state["state_history"] = history
    for h in history[-5:]:
        history_summary.append(h)

    return StateTransition(
        current_state=current,
        previous_state=prev_state if prev_state else current,
        transition_direction=direction,
        trigger_events=trigger_events,
        confidence=confidence,
        state_history=history_summary,
    )


# =========================================================
# V9 核心：Follow Through Acceptance Engine
# =========================================================

def compute_ft_acceptance(df: pd.DataFrame, swings: list, current_bar: int) -> FollowThroughAcceptance:
    """
    V9 重写：不数连续同向K线，而是观察市场是否接受了当前价格。
    
    核心问题：最后一根K线的突破/推进，被市场接受了吗？
    """
    n = min(8, len(df))
    if n < 3:
        return FollowThroughAcceptance("数据不足", False, False, False, False, False, {})

    recent = df.tail(n)
    cur = recent.iloc[-1]
    prev = recent.iloc[-2] if len(recent) >= 2 else cur

    # 判断最后一根K线的方向意图
    bar_body = abs(cur["close"] - cur["open"])
    bar_range = cur["high"] - cur["low"]
    if bar_range < 1e-9:
        return FollowThroughAcceptance("无明确方向", False, False, False, False, False, {"body": 0})
    body_ratio = bar_body / bar_range
    is_bull_bar = cur["close"] > cur["open"]
    is_bear_bar = cur["close"] < cur["open"]

    # 1. 是否突破前高/前低？
    prior_highs = [s.price for s in swings if s.index <= current_bar and s.kind == "SH"]
    prior_lows = [s.price for s in swings if s.index <= current_bar and s.kind == "SL"]
    breakthrough_prior = False
    if is_bull_bar and prior_highs:
        if cur["close"] >= max(prior_highs[-3:]):
            breakthrough_prior = True
    if is_bear_bar and prior_lows:
        if cur["close"] <= min(prior_lows[-3:]):
            breakthrough_prior = True

    # 2. 是否维持在突破区域？
    maintaining_breakzone = False
    if breakthrough_prior:
        if len(recent) >= 3:
            # 看突破后的K线是否维持
            for i in range(len(recent) - 2, max(len(recent) - 5, -1), -1):
                bar = recent.iloc[i]
                if is_bull_bar and bar["low"] > cur["open"]:
                    maintaining_breakzone = True
                    break
                if is_bear_bar and bar["high"] < cur["open"]:
                    maintaining_breakzone = True
                    break

    # 3. 是否快速回撤？
    quick_rejection = False
    if len(recent) >= 2 and body_ratio > 0.4:
        next_bar = recent.iloc[-2] if len(recent) >= 2 else None
        # 用前一根来模拟（因为我们看不到未来，这里用倒数第二根模拟突破后的反应）
        if is_bull_bar and prev["close"] < prev["open"] and prev["close"] < cur["open"]:
            quick_rejection = True
        elif is_bear_bar and prev["close"] > prev["open"] and prev["close"] > cur["open"]:
            quick_rejection = True

    # 4. 是否出现对手被困？
    trapped_opposite = False
    if body_ratio > 0.6:
        # 大实体K线后紧跟着反方向K线，但反方向K线被当前K线覆盖 = trapped
        if len(recent) >= 3:
            bar_before = recent.iloc[-3]
            if is_bull_bar:
                if bar_before["close"] < bar_before["open"] and cur["low"] <= bar_before["low"]:
                    trapped_opposite = True
            elif is_bear_bar:
                if bar_before["close"] > bar_before["open"] and cur["high"] >= bar_before["high"]:
                    trapped_opposite = True

    # 5. 是否出现反向压力？
    opposite_pressure = False
    if len(recent) >= 3:
        # 检查最近3根K线中是否有反方向K线的尾巴明显增长
        opposite_bodies = []
        for i in range(-3, 0):
            bar = recent.iloc[i]
            rng = bar["high"] - bar["low"]
            if rng > 1e-9:
                if is_bull_bar and bar["close"] < bar["open"]:
                    opposite_bodies.append(abs(bar["close"] - bar["open"]) / rng)
                elif is_bear_bar and bar["close"] > bar["open"]:
                    opposite_bodies.append(abs(bar["close"] - bar["open"]) / rng)
        if opposite_bodies and max(opposite_bodies) > 0.5:
            opposite_pressure = True

    # 综合判断
    detail = {
        "实体比": f"{body_ratio:.0%}",
        "突破前极值": breakthrough_prior,
        "维持突破区": maintaining_breakzone,
        "快速回撤": quick_rejection,
        "对手被困": trapped_opposite,
        "反向压力": opposite_pressure,
    }

    if body_ratio < 0.2:
        return FollowThroughAcceptance("无明确方向", False, False, False, False, False, detail)

    if breakthrough_prior and maintaining_breakzone:
        return FollowThroughAcceptance("被接受", True, True, False, False, False, detail)
    if breakthrough_prior and quick_rejection:
        return FollowThroughAcceptance("被拒绝", True, False, True, False, False, detail)
    if breakthrough_prior and opposite_pressure:
        return FollowThroughAcceptance("部分接受", True, False, False, False, True, detail)
    if body_ratio > 0.5 and trapped_opposite:
        return FollowThroughAcceptance("部分接受", False, False, False, True, False, detail)

    return FollowThroughAcceptance("无明确方向", False, False, False, False, False, detail)

# =========================================================
# V9 核心：失败后行为追踪
# =========================================================

def compute_post_failure(
    df: pd.DataFrame, swings: list, legs: list,
    chart_df: pd.DataFrame, current_bar: int,
) -> PostFailureBehavior:
    """
    V9 新增：不只是标记失败突破，而是追踪失败之后市场做了什么。
    """
    if len(chart_df) < 5:
        return PostFailureBehavior(False, "", False, False, False, False, False, False, "数据不足")

    recent = chart_df.tail(min(15, len(chart_df)))
    n = len(recent)
    if n < 5:
        return PostFailureBehavior(False, "", False, False, False, False, False, False, "数据不足")

    cur = recent.iloc[-1]
    # 检测最近是否发生了突破失败
    prior_highs = [s.price for s in swings if s.kind == "SH"]
    prior_lows = [s.price for s in swings if s.kind == "SL"]

    failure_detected = False
    failure_type = ""

    # 突破失败：K线突破前高但收盘在前高之下
    if prior_highs and cur["high"] >= max(prior_highs[-3:]) and cur["close"] < max(prior_highs[-3:]):
        failure_detected = True
        failure_type = "突破失败"
    elif prior_lows and cur["low"] <= min(prior_lows[-3:]) and cur["close"] > min(prior_lows[-3:]):
        failure_detected = True
        failure_type = "突破失败"

    # 推进失败：大实体K线后完全被反包
    if not failure_detected and n >= 3:
        big_bar = recent.iloc[-3]
        big_body = abs(big_bar["close"] - big_bar["open"])
        big_range = big_bar["high"] - big_bar["low"]
        if big_range > 1e-9 and big_body / big_range > 0.6:
            reversal_bar = recent.iloc[-1]
            if (big_bar["close"] > big_bar["open"] and reversal_bar["close"] < big_bar["open"]) or \
               (big_bar["close"] < big_bar["open"] and reversal_bar["close"] > big_bar["open"]):
                failure_detected = True
                failure_type = "推进失败"

    if not failure_detected:
        # 跟进消失
        if n >= 6:
            direction = "bull" if recent.iloc[-1]["close"] > recent.iloc[-6]["close"] else "bear"
            same_dir = 0
            for i in range(-6, 0):
                bar = recent.iloc[i]
                if (direction == "bull" and bar["close"] > bar["open"]) or \
                   (direction == "bear" and bar["close"] < bar["open"]):
                    same_dir += 1
            if same_dir <= 1:
                failure_detected = True
                failure_type = "跟进消失"

    if not failure_detected:
        return PostFailureBehavior(False, "", False, False, False, False, False, False, "未检测到失败行为")

    # --- 追踪失败后行为 ---
    rapid_reversal = False
    strong_opposite_ft = False
    trapped_formed = False
    measured_move_fail = False
    second_failure = False
    continuation = False

    # 快速反包：失败K线后紧接着反向K线覆盖
    if n >= 2:
        next_bar = recent.iloc[-2]
        if failure_type == "突破失败":
            if cur["high"] >= max(prior_highs[-3:]) and next_bar["close"] < cur["open"]:
                rapid_reversal = True
            elif prior_lows and cur["low"] <= min(prior_lows[-3:]) and next_bar["close"] > cur["open"]:
                rapid_reversal = True

    # 强反向跟进：失败后出现连续反向推进
    if n >= 4 and failure_type in ("突破失败", "推进失败"):
        failed_bull = cur["close"] > cur["open"]  # 失败的多头突破
        opposite_count = 0
        for i in range(-4, -1):
            bar = recent.iloc[i]
            if failed_bull and bar["close"] < bar["open"]:
                opposite_count += 1
            elif not failed_bull and bar["close"] > bar["open"]:
                opposite_count += 1
        if opposite_count >= 2:
            strong_opposite_ft = True

    # Trapped trader：失败后价格再次测试失败区域又失败
    if n >= 5 and prior_highs and failure_type == "突破失败":
        resistance = max(prior_highs[-3:])
        tests = 0
        fails = 0
        for i in range(n - 5, n):
            bar = recent.iloc[i]
            if bar["high"] >= resistance:
                tests += 1
                if bar["close"] < resistance:
                    fails += 1
        if tests >= 2 and fails >= 2:
            trapped_formed = True
    elif n >= 5 and prior_lows and failure_type == "突破失败":
        support = min(prior_lows[-3:])
        tests = 0
        fails = 0
        for i in range(n - 5, n):
            bar = recent.iloc[i]
            if bar["low"] <= support:
                tests += 1
                if bar["close"] > support:
                    fails += 1
        if tests >= 2 and fails >= 2:
            trapped_formed = True

    # 等幅目标失败
    if legs and failure_type == "推进失败":
        last_leg = next((l for l in reversed(legs) if l.end_idx <= current_bar), None)
        if last_leg:
            # 如果失败后反向推进超过了原leg的range，算measured move failure
            if n >= 4:
                reversal_range = abs(recent.iloc[-1]["close"] - cur["close"])
                if reversal_range > last_leg.price_range * 0.8:
                    measured_move_fail = True

    # 二次失败：再次尝试同一方向但再次失败
    if n >= 8 and failure_type == "突破失败":
        if prior_highs:
            resistance = max(prior_highs[-3:])
            first_test = False
            second_test = False
            for i in range(n - 8, n - 3):
                bar = recent.iloc[i]
                if bar["high"] >= resistance and bar["close"] < resistance:
                    first_test = True
            for i in range(n - 3, n):
                bar = recent.iloc[i]
                if bar["high"] >= resistance and bar["close"] < resistance:
                    second_test = True
            if first_test and second_test:
                second_failure = True

    # 失败后反而继续原方向
    if n >= 4 and failure_type == "突破失败":
        if prior_highs and cur["close"] > cur["open"]:
            # 多头突破失败但随后价格继续上涨
            for i in range(-4, -1):
                bar = recent.iloc[i]
                if bar["close"] > bar["open"] and bar["close"] > max(prior_highs[-3:]):
                    continuation = True
                    break

    parts = []
    if failure_detected:
        parts.append(f"检测到{failure_type}")
    if rapid_reversal: parts.append("快速反包")
    if strong_opposite_ft: parts.append("强反向跟进")
    if trapped_formed: parts.append("Trapped trader 形成")
    if measured_move_fail: parts.append("等幅目标失败")
    if second_failure: parts.append("二次失败")
    if continuation: parts.append("失败后继续原方向")

    return PostFailureBehavior(
        failure_detected, failure_type, rapid_reversal, strong_opposite_ft,
        trapped_formed, measured_move_fail, second_failure, continuation,
        "；".join(parts) if parts else "失败但行为不明显",
    )


# =========================================================
# 市场倾向（V9: 基于状态转移）
# =========================================================

def compute_tendency(
    momentum_shift: str, ft: str, range_prog: str,
    always_in: AlwaysIn, location: LocationContext, legs: list,
    chart_df: pd.DataFrame, current_bar: int,
    prev_state: str = "",
) -> MarketTendency:
    """V9: 倾向 + 状态转移"""
    transition = compute_state_transition(
        momentum_shift, ft, range_prog, always_in, location,
        legs, chart_df, current_bar, prev_state,
    )
    primary = transition.current_state
    secondary = transition.previous_state if transition.transition_direction == "转换" else ""

    # 混合信号
    mixed = []
    if ft == "跟进强" and momentum_shift in ("明显衰减", "严重衰减"):
        mixed.append("跟进强但推进衰减")
    if ("Long" in always_in.status or "Short" in always_in.status) and \
       range_prog in ("正在区间化", "明确区间"):
        mixed.append(f"Always In 但{range_prog}")
    if transition.confidence == "矛盾":
        for ev in transition.trigger_events:
            if "矛盾" in ev:
                mixed.append(ev.replace("矛盾：", ""))

    return MarketTendency(primary=primary, secondary=secondary, mixed_signals=mixed,
                          state_transition=transition)


# =========================================================
# V9 重写：validate_outcome()
# =========================================================

def validate_outcome(df, current_index, user_expectation, snapshot=None) -> Outcome:
    """
    V9 重写：不只判断对错，还要回答'你忽略了什么'。
    """
    future = df.iloc[current_index + 1: current_index + 11]
    if len(future) == 0:
        return Outcome(False, False, False, False, False, "数据不足", "数据不足", "数据不足")

    cur = df.iloc[current_index]
    direction = "bull" if cur["close"] > cur["open"] else "bear"
    cur_range = cur["high"] - cur["low"]
    cur_body = abs(cur["close"] - cur["open"])
    body_ratio = cur_body / cur_range if cur_range > 1e-9 else 0

    # --- Follow Through：使用 Acceptance Engine 的逻辑 ---
    got_ft = False
    ft_detail_parts = []

    # 突破前极值后的跟进
    if len(future) >= 2:
        # 价格是否被接受？
        if direction == "bull":
            # 看收盘是否维持在当前K线实体中点以上
            mid = (cur["open"] + cur["close"]) / 2
            accepted = sum(1 for i in range(min(3, len(future))) if future.iloc[i]["close"] > mid)
            if accepted >= 2:
                got_ft = True
                ft_detail_parts.append(f"价格维持在突破区（{accepted}/3）")
            # 看是否有更高收盘
            higher_closes = sum(1 for i in range(min(3, len(future))) if future.iloc[i]["close"] > cur["close"])
            if higher_closes >= 1:
                got_ft = True
                ft_detail_parts.append(f"有更高收盘（{higher_closes}/3）")
            # 大阳线后小幅回调但未覆盖实体50% = 接受
            if body_ratio > 0.5 and len(future) >= 2:
                pullback_low = min(future.iloc[0]["low"], future.iloc[1]["low"])
                if pullback_low > cur["open"]:
                    got_ft = True
                    ft_detail_parts.append("回调未覆盖实体50%，价格被接受")
        else:
            mid = (cur["open"] + cur["close"]) / 2
            accepted = sum(1 for i in range(min(3, len(future))) if future.iloc[i]["close"] < mid)
            if accepted >= 2:
                got_ft = True
                ft_detail_parts.append(f"价格维持在突破区（{accepted}/3）")
            lower_closes = sum(1 for i in range(min(3, len(future))) if future.iloc[i]["close"] < cur["close"])
            if lower_closes >= 1:
                got_ft = True
                ft_detail_parts.append(f"有更低收盘（{lower_closes}/3）")
            if body_ratio > 0.5 and len(future) >= 2:
                pullback_high = max(future.iloc[0]["high"], future.iloc[1]["high"])
                if pullback_high < cur["open"]:
                    got_ft = True
                    ft_detail_parts.append("回调未覆盖实体50%，价格被接受")

    # --- Trapped traders ---
    trapped = False
    if body_ratio > 0.5 and len(future) >= 2:
        f0, f1 = future.iloc[0], future.iloc[1]
        if direction == "bull":
            # 大阳线后空头入场但被多头发起的新高覆盖
            if f0["close"] < f0["open"] and f1["high"] > cur["high"] and f1["close"] > cur["close"]:
                trapped = True
        else:
            if f0["close"] > f0["open"] and f1["low"] < cur["low"] and f1["close"] < cur["close"]:
                trapped = True

    # --- Breakout ---
    breakout_ok = False
    if user_expectation == "延续":
        for i in range(min(5, len(future))):
            if direction == "bull" and future.iloc[i]["close"] > cur["close"]:
                breakout_ok = True; break
            elif direction == "bear" and future.iloc[i]["close"] < cur["close"]:
                breakout_ok = True; break

    # --- Reversal ---
    reversal_held = False
    if user_expectation == "反转":
        fc = future.iloc[-1]["close"]
        if (direction == "bull" and fc < cur["close"]) or (direction == "bear" and fc > cur["close"]):
            reversal_held = True

    # --- Range ---
    range_cont = False
    if user_expectation == "继续区间":
        rh = df.iloc[max(0, current_index - 5): current_index + 1]["high"].max()
        rl = df.iloc[max(0, current_index - 5): current_index + 1]["low"].min()
        if rh > rl + 1e-9:
            range_cont = all(
                rl * 0.998 <= future.iloc[i]["close"] <= rh * 1.002
                for i in range(min(5, len(future)))
            )

    # --- V9: 你忽略了什么？ ---
    what_you_missed = ""
    failure_category = ""

    # 检测各种"你忽略的行为变化"
    missed_parts = []

    # 1. 用户猜延续，但跟进消失了
    if user_expectation == "延续" and not got_ft and not breakout_ok:
        missed_parts.append("你预期延续，但市场没有接受推进价格")
        failure_category = "跟进误判"

    # 2. 用户猜反转，但Always In仍然很强
    if snapshot and user_expectation == "反转" and not reversal_held:
        ai = snapshot.always_in
        if "Long" in ai.status or "Short" in ai.status:
            if ai.conviction in ("强", "中"):
                missed_parts.append(f"你猜反转，但 Always In {ai.status}（{ai.conviction}）仍然有效")
                failure_category = "逆Always In反转"

    # 3. 用户在区间猜方向
    if snapshot and user_expectation in ("延续", "反转") and snapshot.location.in_range:
        if range_cont or (not got_ft and not breakout_ok):
            missed_parts.append("你在区间中追方向，但市场维持区间行为")
            failure_category = "区间中追方向"

    # 4. 推进在衰减但用户没注意到
    if snapshot and user_expectation == "延续" and \
       snapshot.pressure.momentum_shift in ("明显衰减", "严重衰减") and not got_ft:
        missed_parts.append(f"推进正在{snapshot.pressure.momentum_shift}，但你的预期没有反映这个变化")
        failure_category = "忽略衰减"

    # 5. 用户忽略了失败后行为
    if snapshot and snapshot.post_failure.failure_detected:
        pf = snapshot.post_failure
        if pf.rapid_reversal and user_expectation != "反转":
            missed_parts.append(f"检测到{pf.failure_type}后的快速反包，但你没有据此调整预期")
            failure_category = "忽略失败后行为"
        elif pf.trapped_traders_formed:
            missed_parts.append(f"检测到 trapped trader 形成，这可能预示反向跟进")
            failure_category = "忽略 trapped trader"

    if missed_parts:
        what_you_missed = "；".join(missed_parts)
    else:
        what_you_missed = "无明显忽略" if (got_ft or breakout_ok or reversal_held or range_cont) else "判断与市场行为不一致，请回顾K线细节"

    parts = []
    if got_ft: parts.append("跟进成立")
    else: parts.append("跟进消失")
    if trapped: parts.append("对手被困")
    if breakout_ok: parts.append("延续成立")
    if reversal_held: parts.append("反转成立")
    if range_cont: parts.append("区间继续")

    return Outcome(
        got_ft, trapped, breakout_ok, reversal_held, range_cont,
        "；".join(parts) if parts else "无明确行为",
        what_you_missed, failure_category,
    )

# =========================================================
# 构建快照
# =========================================================

def build_snapshot(chart_df, current_bar, global_index=0) -> MarketSnapshot:
    swings = detect_swings(chart_df)
    labels = detect_market_structure(swings)
    legs = detect_legs(chart_df, swings)
    location = detect_location(chart_df, swings, legs, current_bar)
    always_in = detect_always_in(chart_df, legs, swings, current_bar)
    ft_s, ft_d = observe_follow_through(chart_df)
    mom_s, mom_d = observe_momentum(chart_df)
    rng_s, rng_d = observe_range_formation(chart_df)
    pressure = PressureSnapshot(ft_s, ft_d, mom_s, mom_d, rng_s, rng_d)

    prev_state = ""
    if hasattr(st, 'session_state'):
        prev_state = st.session_state.get("last_transition", "")
    tendency = compute_tendency(mom_s, ft_s, rng_s, always_in, location, legs,
                                chart_df, current_bar, prev_state)
    if hasattr(st, 'session_state'):
        st.session_state["last_transition"] = tendency.state_transition.current_state

    ft_acceptance = compute_ft_acceptance(chart_df, swings, current_bar)
    post_failure = compute_post_failure(chart_df, swings, legs, chart_df, current_bar)
    state_transition = tendency.state_transition

    tags = []
    if ft_s in ("跟进强", "跟进弱", "反包"): tags.append(f"跟进:{ft_s}")
    if mom_s in ("轻微衰减", "明显衰减", "严重衰减"): tags.append(f"推进:{mom_s}")
    if rng_s in ("正在区间化", "明确区间"): tags.append(f"区间化:{rng_s}")
    if always_in.status != "Always In Transition":
        tags.append(f"AI {always_in.status.split()[-1]}（{always_in.conviction}）")
    else:
        tags.append("AI 过渡期")
    if location.measured_move_level: tags.append(location.measured_move_level)
    if location.climactic_extension: tags.append("高潮延伸")
    if location.breakout_pullback_area: tags.append("突破回踩")
    if tendency.mixed_signals: tags.append(f"矛盾:{tendency.mixed_signals[0]}")
    if ft_acceptance.acceptance_level != "无明确方向":
        tags.append(f"FT:{ft_acceptance.acceptance_level}")
    if post_failure.failure_detected:
        tags.append(f"失败:{post_failure.failure_type}")

    return MarketSnapshot(
        swings, labels, legs, location, always_in, pressure, tendency, tags,
        ft_acceptance, post_failure, state_transition,
    )


# =========================================================
# V9: 训练闭环 — 错误追踪 + 三档偏差统计
# =========================================================

def detect_consecutive_errors(logs: list) -> list:
    if len(logs) < 3:
        return []
    results = []
    n = len(logs)

    # 1. 连续在区间中追方向
    for length in range(min(5, n), 2, -1):
        tail = logs[-length:]
        if all(l.get("outcome", {}).get("range_continued", False) and
               l["expectation"] in ("延续", "反转") for l in tail):
            results.append({
                "pattern": "区间中追方向",
                "count": length,
                "bars": [l["bar_index"] for l in tail],
                "suggestion": "你连续在区间行为中追方向。区间中市场没有方向偏好，延续概率不高于反转。重新观察这些位置的压力信号。",
            })
            break

    # 2. 连续猜反转失败
    for length in range(min(5, n), 2, -1):
        tail = logs[-length:]
        if all(l["expectation"] == "反转" and not l.get("outcome", {}).get("reversal_held", False) for l in tail):
            results.append({
                "pattern": "连续猜反转失败",
                "count": length,
                "bars": [l["bar_index"] for l in tail],
                "suggestion": "你连续猜反转但市场没有反转。检查：Always In 是否仍指向原方向？回调是否覆盖了推进的50%？",
            })
            break

    # 3. 连续忽略跟进衰竭
    for length in range(min(5, n), 2, -1):
        tail = logs[-length:]
        if all(l.get("outcome", {}).get("got_follow_through", True) == False and
               "跟进弱" not in str(l.get("events", [])) and
               "无跟进" not in str(l.get("events", [])) for l in tail):
            results.append({
                "pattern": "连续忽略跟进衰竭",
                "count": length,
                "bars": [l["bar_index"] for l in tail],
                "suggestion": "你连续没有识别到跟进消失。跟进衰竭是趋势结束的早期信号，关注：连续实体数、收盘靠近极端的比例。",
            })
            break

    # 4. Always In 判断连续偏差
    for length in range(min(5, n), 2, -1):
        tail = logs[-length:]
        mismatches = 0
        for l in tail:
            ai = l.get("always_in", "")
            mc = l.get("market_control", "")
            if (("Long" in ai and mc == "空头控制") or
                ("Short" in ai and mc == "多头控制") or
                ("Transition" in ai and mc != "多空平衡")):
                mismatches += 1
        if mismatches == length:
            results.append({
                "pattern": "Always In判断连续偏差",
                "count": length,
                "bars": [l["bar_index"] for l in tail],
                "suggestion": "你的市场控制判断与 Always In 状态连续不一致。回顾这些位置的证据：推进 leg 质量、回调覆盖程度、HC/LC 比例。",
            })
            break

    # 5. V9: 连续忽略失败后行为
    for length in range(min(5, n), 2, -1):
        tail = logs[-length:]
        if all(l.get("outcome", {}).get("failure_category", "") == "忽略失败后行为" for l in tail):
            results.append({
                "pattern": "连续忽略失败后行为",
                "count": length,
                "bars": [l["bar_index"] for l in tail],
                "suggestion": "你连续忽略失败突破后的行为变化。失败后市场通常会出现快速反包或trapped trader，这是控制权转移的关键信号。",
            })
            break

    return results


def build_bias_statistics_three_tier(logs: list) -> dict:
    """
    V9: 三档偏差统计 — 短期5次/中期20次/长期100次。
    不再用固定窗口，避免短期随机性污染。
    """
    tiers = {
        "短期（最近5次）": logs[-5:] if len(logs) >= 5 else logs,
        "中期（最近20次）": logs[-20:] if len(logs) >= 20 else logs,
        "长期（最近100次）": logs[-100:] if len(logs) >= 100 else logs,
    }

    result = {}
    for tier_name, tier_logs in tiers.items():
        if not tier_logs:
            continue
        stats = {k: 0 for k in [
            "过早猜反转", "趋势误判", "区间识别不足",
            "失败突破遗漏", "位置盲区", "跟进误判",
            "忽略衰减", "忽略失败后行为", "忽略trapped trader",
        ]}
        for log in tier_logs:
            oc = log.get("outcome", {})
            if log["expectation"] == "反转" and not oc.get("reversal_held", True):
                stats["过早猜反转"] += 1
            if log["market_type"] == "区间" and not oc.get("range_continued", True):
                stats["趋势误判"] += 1
            if (log.get("engine_tendency_range", 0) > 0.4 and log["market_type"] == "趋势"):
                stats["区间识别不足"] += 1
            if not oc.get("got_follow_through", True) and "跟进弱" not in str(log.get("events", [])):
                stats["跟进误判"] += 1
            if (log.get("location_special", False) and log["expectation"] == "延续"
                    and not oc.get("breakout_succeeded", True)):
                stats["位置盲区"] += 1
            if not oc.get("got_follow_through", True) and "失败突破" not in log.get("events", []):
                stats["失败突破遗漏"] += 1
            fc = oc.get("failure_category", "")
            if fc == "忽略衰减":
                stats["忽略衰减"] += 1
            elif fc == "忽略失败后行为":
                stats["忽略失败后行为"] += 1
            elif fc == "忽略 trapped trader":
                stats["忽略trapped trader"] += 1

        total = max(1, sum(stats.values()))
        # 找出该档最突出的2个偏差
        top2 = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:2]
        result[tier_name] = {
            "total": len(tier_logs),
            "top_errors": top2,
            "all_stats": stats,
        }
    return result


def get_pressure_pattern_stats(logs: list) -> dict:
    selected = Counter()
    missed = Counter()
    for log in logs:
        selected_bull = log.get("bull_pressure", [])
        selected_bear = log.get("bear_pressure", [])
        for p in selected_bull + selected_bear:
            selected[p] += 1
        outcome = log.get("outcome", {})
        if not outcome.get("got_follow_through", True):
            if "跟进减少（HC减少）" not in selected_bull + selected_bear:
                missed["跟进减少（HC减少）"] += 1
            if "实体缩小" not in selected_bull + selected_bear:
                missed["实体缩小"] += 1
        if log["expectation"] == "反转" and not outcome.get("reversal_held", False):
            if "二次突破失败" not in selected_bull:
                missed["二次突破失败"] += 1
            if "下跌后快速拉回" not in selected_bear:
                missed["下跌后快速拉回"] += 1
    return {"most_selected": selected.most_common(5), "most_missed": missed.most_common(5)}


# =========================================================
# 案例库
# =========================================================

def find_case_positions(df, scenario, snapshot_cache) -> list:
    positions = []
    min_i = LOOKBACK_MIN
    max_i = len(df) - LOOKAHEAD_RESERVE
    for i in range(min_i, max_i, 3):
        if i in snapshot_cache:
            snap = snapshot_cache[i]
        else:
            start = max(0, i - CHART_WINDOW)
            window = df.iloc[start: i + 1].copy().reset_index(drop=True)
            snap = build_snapshot(window, len(window) - 1, global_index=i)
            snapshot_cache[i] = snap
        if scenario == "趋势衰减":
            if snap.pressure.momentum_shift in ("明显衰减", "严重衰减"):
                positions.append(i)
        elif scenario == "假突破":
            if snap.post_failure.failure_detected and snap.post_failure.failure_type == "突破失败":
                positions.append(i)
        elif scenario == "区间交易":
            if snap.location.in_range or snap.pressure.range_progression in ("正在区间化", "明确区间"):
                positions.append(i)
        elif scenario == "Always In转换":
            if "Transition" in snap.always_in.status and snap.always_in.conviction in ("中", "强"):
                positions.append(i)
        elif scenario == "反转尝试":
            if snap.location.climactic_extension or snap.location.measured_move_level:
                positions.append(i)
        elif scenario == "跟进衰竭":
            if snap.pressure.follow_through in ("无跟进", "反包"):
                positions.append(i)
    return positions[:30]


# =========================================================
# 观点生命周期管理
# =========================================================

def create_viewpoint(direction, expectation, invalidate_cond, ft_cond):
    now = str(datetime.now())
    return Viewpoint(
        state="活跃", direction=direction, expectation=expectation,
        invalidate_cond=invalidate_cond, ft_cond=ft_cond,
        created_at=now, updated_at=now, bars_alive=1, updates_count=0,
    )


def update_viewpoint(vp, new_state):
    now = str(datetime.now())
    vp.state = new_state
    vp.updated_at = now
    vp.bars_alive += 1
    vp.updates_count += 1
    return vp

# =========================================================
# 市场背景（给AI用）
# =========================================================

def build_market_context(chart_df) -> str:
    n = len(chart_df)
    if n == 0:
        return "无数据"
    rng = chart_df["high"] - chart_df["low"]
    body = np.abs(chart_df["close"] - chart_df["open"])
    upper = chart_df["high"] - np.maximum(chart_df["open"], chart_df["close"])
    lower = np.minimum(chart_df["open"], chart_df["close"]) - chart_df["low"]
    br = np.where(rng > 0, body / rng, 0)
    dr = np.where(chart_df["close"] >= chart_df["open"], "阳", "阴")
    lines = []
    for i in range(n):
        lines.append(
            f"#{i} {dr[i]} O:{chart_df.iloc[i]['open']:.2f} H:{chart_df.iloc[i]['high']:.2f} "
            f"L:{chart_df.iloc[i]['low']:.2f} C:{chart_df.iloc[i]['close']:.2f} "
            f"体:{br[i]:.2f} 上:{upper[i]:.2f} 下:{lower[i]:.2f}"
        )
    return "\\n".join(lines)


# =========================================================
# AI 偏差纠正（V9: 三档统计，100字限制）
# =========================================================

def get_bias_correction(client, model, consecutive_errors, pressure_stats, recent_logs) -> str:
    if not consecutive_errors and not pressure_stats["most_missed"]:
        return "近期未检测到明显偏差模式。继续保持观察训练。"

    errors_text = ""
    for e in consecutive_errors:
        errors_text += f"- {e['pattern']}（连续{e['count']}次）：{e['suggestion']}\\n"

    missed_text = ""
    for p, count in pressure_stats["most_missed"]:
        if count >= 2:
            missed_text += f"- 「{p}」被忽略{count}次\\n"

    # V9: 三档统计摘要
    three_tier = build_bias_statistics_three_tier(recent_logs)
    tier_summary = ""
    for tier_name, tier_data in three_tier.items():
        top = tier_data["top_errors"]
        if top and top[0][1] > 0:
            tier_summary += f"- {tier_name}({tier_data['total']}次): 最突出「{top[0][0]}」{top[0][1]}次\\n"

    system_prompt = textwrap.dedent("""\\
        你是交易训练偏差分析器。你的唯一任务是指出用户反复出现的读盘偏差。

        严格规则：
        1. 只基于下方【错误数据】输出
        2. 每个偏差给一个具体改进动作
        3. 禁止解盘、禁止分析市场、禁止解释理论
        4. 严格限制 100 字
        5. 如果没有明显偏差，直接说"近期偏差在减少"
    """)

    user_msg = f"""\\
        ===== 连续错误模式 =====
        {errors_text if errors_text else "无连续错误"}

        ===== 三档偏差统计 =====
        {tier_summary if tier_summary else "无"}

        ===== 最常忽略的压力模式 =====
        {missed_text if missed_text else "无"}

        ===== 最近5次判断 =====
        {json.dumps([{
            "ai": l.get("always_in", ""),
            "expectation": l.get("expectation", ""),
            "what_missed": l.get("outcome", {}).get("what_you_missed", ""),
            "failure_cat": l.get("outcome", {}).get("failure_category", ""),
        } for l in recent_logs[-5:]], ensure_ascii=False, indent=2)}
    """

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[分析失败: {e}]"


# =========================================================
# 图表
# =========================================================

_BULL = "#26A69A"
_BEAR = "#EF5350"
_LBL = {"HH": "#00C853", "HL": "#69F0AE", "LH": "#FF5252", "LL": "#FF8A80"}


def build_chart(chart_df, snapshot, current_bar, case_highlight=False, blind_mode=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"], close=chart_df["close"],
        increasing_line_color=_BULL, decreasing_line_color=_BEAR,
    ))

    fig.add_vrect(x0=current_bar - 0.4, x1=current_bar + 0.4,
                  fillcolor="rgba(255,235,59,0.15)" if not case_highlight else "rgba(255,82,82,0.25)",
                  line_width=0)
    fig.add_vline(x=current_bar, line_dash="dash", line_width=1.5,
                  line_color="#FFC107" if not case_highlight else "#FF5252")

    loc = snapshot.location
    if loc.in_range:
        fig.add_hrect(y0=chart_df["low"].min(), y1=chart_df["high"].max(),
                      fillcolor="rgba(156,39,176,0.06)", line_width=0)
        fig.add_annotation(x=len(chart_df) - 1, y=chart_df["high"].max(),
                           text=f"区间（{loc.range_position}）", showarrow=False,
                           font=dict(size=9, color="#CE93D8"), xanchor="right", yshift=18)

    for leg in snapshot.legs[-4:]:
        clr = _BULL if leg.direction == "bull" else _BEAR
        op = 0.3 if leg.momentum == "弱推进" else (0.5 if leg.momentum == "正常推进" else 0.7)
        fig.add_vrect(x0=leg.start_idx - 0.5, x1=leg.end_idx + 0.5,
                      fillcolor=clr, line_width=0, opacity=op)

    for i in range(0, len(chart_df), 10):
        fig.add_annotation(x=i, y=chart_df.iloc[i]["high"], text=str(i),
                           showarrow=False, font=dict(size=7, color="#757575"), yshift=8)

    # V9: 盲测模式 — 隐藏系统标注
    if not blind_mode:
        for sw in snapshot.swings:
            clr = "#00E676" if sw.kind == "SH" else "#FF5252"
            fig.add_annotation(x=sw.index, y=sw.price, text=sw.kind, showarrow=True,
                               font=dict(size=8, color=clr), arrowhead=2, arrowsize=0.8)

        for lb in snapshot.labels:
            row = chart_df.iloc[lb.index]
            clr = _LBL.get(lb.label, "#FFF")
            y_pos = row["low"] if lb.label in ("HL", "LL") else row["high"]
            y_off = -14 if lb.label in ("HL", "LL") else 14
            fig.add_annotation(x=lb.index, y=y_pos, text=lb.label, showarrow=False,
                               font=dict(size=9, color=clr, family="monospace"), yshift=y_off)

        ai = snapshot.always_in
        ai_clr = "#26A69A" if "Long" in ai.status else ("#EF5350" if "Short" in ai.status else "#FFC107")
        fig.add_annotation(x=0, y=chart_df["high"].max(),
                           text=f"{ai.status} ({ai.conviction})",
                           showarrow=False, font=dict(size=11, color=ai_clr, family="monospace"),
                           xanchor="left", yshift=18)

    fig.update_layout(
        height=540, xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="#16161a", plot_bgcolor="#16161a",
        font=dict(color="#E0E0E0"),
        xaxis=dict(gridcolor="#222228", zeroline=False),
        yaxis=dict(gridcolor="#222228", zeroline=False),
    )
    return fig


# =========================================================
# UI 组件
# =========================================================

def inject_css():
    st.markdown("""<style>
    html, body, [class*="css"] { font-size: 13px !important; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
    .stButton > button { width: 100%; height: 40px; border-radius: 8px; }
    </style>""", unsafe_allow_html=True)


def render_observation_panel(snapshot: MarketSnapshot, blind_mode: bool):
    """V9: 观察面板。盲测模式下隐藏系统辅助。"""
    if blind_mode:
        st.markdown("### 盲测模式")
        st.info("系统标注已隐藏。请凭自己的观察做出判断。提交后可查看系统标注。")
        return

    st.markdown("### 市场观察")

    # 状态转移（V9 核心）
    st = snapshot.state_transition
    st_color = {
        "明确": "#00C853", "模糊": "#FFC107", "矛盾": "#FF5252",
    }.get(st.confidence, "#E0E0E0")
    st.markdown(f"**状态转移**: {st.to_display()}")
    st.caption(f"触发事件：{', '.join(st.trigger_events) if st.trigger_events else '无'}")
    if st.state_history:
        st.caption(f"历史：{' -> '.join(st.state_history[-5:])}")

    # Always In
    ai = snapshot.always_in
    ai_icon = "🟢" if "Long" in ai.status else ("🔴" if "Short" in ai.status else "🟡")
    st.markdown(f"**{ai_icon} {ai.status}**（{ai.conviction}）")
    with st.expander("依据", expanded=False):
        for e in ai.evidence:
            st.caption(f"- {e}")

    # FT Acceptance（V9 新增）
    ft_acc = snapshot.ft_acceptance
    if ft_acc.acceptance_level != "无明确方向" and ft_acc.acceptance_level != "数据不足":
        ft_icon = {"被接受": "✅", "部分接受": "⚠️", "被拒绝": "❌"}.get(ft_acc.acceptance_level, "❓")
        st.markdown(f"**{ft_icon} FT Acceptance**: {ft_acc.acceptance_level}")

    # 失败后行为（V9 新增）
    pf = snapshot.post_failure
    if pf.failure_detected:
        st.markdown(f"**失败后行为**: {pf.description}")

    # 位置
    loc = snapshot.location
    loc_parts = []
    if loc.near_prior_high: loc_parts.append("接近前高")
    if loc.near_prior_low: loc_parts.append("接近前低")
    if loc.in_range: loc_parts.append(f"区间（{loc.range_position}）")
    if loc.near_channel_line: loc_parts.append(loc.near_channel_line)
    if loc.breakout_pullback_area: loc_parts.append("突破回踩")
    if loc.climactic_extension: loc_parts.append("高潮延伸")
    if loc.measured_move_level: loc_parts.append(loc.measured_move_level)
    st.markdown(f"**位置**：{'，'.join(loc_parts) if loc_parts else '无特殊位置'}")

    # 三个引擎
    c1, c2, c3 = st.columns(3)
    p = snapshot.pressure
    with c1:
        st.markdown(f"**跟进**: {p.follow_through}")
        for k, v in p.ft_detail.items():
            st.caption(f"{k}: {v}")
    with c2:
        st.markdown(f"**推进**: {p.momentum_shift}")
        for k, v in p.momentum_detail.items():
            st.caption(f"{k}: {v}")
    with c3:
        st.markdown(f"**区间化**: {p.range_progression}")
        for k, v in p.range_detail.items():
            st.caption(f"{k}: {v}")


def render_viewpoint_panel():
    """V9: 观点生命周期面板。"""
    vp = st.session_state.get("active_viewpoint")
    if vp is None:
        return

    st.markdown("---")
    state_icon = {
        "活跃": "👁️", "加强": "💪", "减弱": "📉", "失效": "❌", "转换": "🔄",
    }.get(vp.state, "👁️")
    st.markdown(f"**当前观点** {state_icon}")
    c1, c2, c3 = st.columns(4)
    with c1:
        st.markdown(f"**方向**: {vp.direction}")
    with c2:
        st.markdown(f"**预期**: {vp.expectation}")
    with c3:
        st.markdown(f"**状态**: {vp.state}")
    with c3:
        st.markdown(f"**存活**: {vp.bars_alive}根")

    st.caption(f"失效条件：{vp.invalidate_cond}")
    st.caption(f"确认条件：{vp.ft_cond}")


def export_logs(logs: list) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"training_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    return path

# =========================================================
# 主函数
# =========================================================

def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器", layout="wide")
    init_session()
    inject_css()

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("请在 .streamlit/secrets.toml 中配置 OPENAI_API_KEY")
        st.stop()

    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")

    # --- 顶部 ---
    st.title("Al Brooks 读盘训练器 V9")

    top1, top2, top3, top4 = st.columns([1, 1, 1, 1])
    with top1:
        symbol = st.selectbox("期货品种", FUTURES_SYMBOLS)
    with top2:
        if st.button("重新加载数据"):
            st.cache_data.clear()
            st.session_state.case_mode = None
            st.session_state.active_viewpoint = None
            st.session_state.state_history = []
            st.session_state.last_transition = ""
            st.rerun()

    # 模式选择
    modes = ["自由浏览", "Replay训练", "专项训练"]
    mode_idx = 0
    if st.session_state.mode == "Replay训练":
        mode_idx = 1
    elif st.session_state.mode == "专项训练":
        mode_idx = 2

    with top3:
        new_mode = st.radio("训练模式", modes, index=mode_idx)
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode
            st.session_state.case_mode = None
            st.session_state.replay_positions = []
            st.session_state.replay_cursor = 0
            st.session_state.active_viewpoint = None
            st.rerun()

    with top4:
        st.markdown(f"**已完成：{st.session_state.submit_count} 次**")

    # --- 数据 ---
    try:
        df = load_data(symbol)
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        st.stop()

    min_idx, max_idx = LOOKBACK_MIN, len(df) - LOOKAHEAD_RESERVE
    mode = st.session_state.mode
    blind_mode = st.session_state.blind_mode

    # --- Replay 子模式选择 ---
    sub_modes = ["标准", "强制预期", "观点更新", "盲测"]
    if mode == "Replay训练":
        st.markdown("---")
        sm_cols = st.columns(4)
        with sm_cols[0]:
            new_sub = st.radio("Replay 子模式", sub_modes, index=sub_modes.index(st.session_state.replay_sub_mode))
            if new_sub != st.session_state.replay_sub_mode:
                st.session_state.replay_sub_mode = new_sub
                st.session_state.blind_mode = (new_sub == "盲测")
                st.session_state.active_viewpoint = None
                st.rerun()
        with sm_cols[1]:
            st.caption("标准：看图判断 | 强制预期：必须写预期和条件 | 观点更新：每根更新观点 | 盲测：隐藏标注")

    # --- Replay 模式 ---
    if mode == "Replay训练":
        if not st.session_state.replay_positions:
            positions = sorted(np.random.randint(min_idx, max_idx, size=20).tolist())
            st.session_state.replay_positions = positions
            st.session_state.replay_cursor = 0

        positions = st.session_state.replay_positions
        cursor = st.session_state.replay_cursor

        rp1, rp2, rp3, rp4 = st.columns([1, 1, 1, 2])
        with rp1:
            st.markdown(f"**进度：{cursor + 1} / {len(positions)}**")
        with rp2:
            if cursor > 0:
                if st.button("◀ 上一题"):
                    st.session_state.replay_cursor -= 1
                    st.rerun()
        with rp3:
            if cursor < len(positions) - 1:
                if st.button("下一题 ▶"):
                    st.session_state.replay_cursor += 1
                    st.rerun()
            else:
                st.info("Replay 已完成")
        with rp4:
            st.caption("按「下一题」推进到下一根 K 线。不要用自动播放——真正训练需要你在每根 K 线前停下来观察。")

        st.session_state.current_index = positions[cursor]

    # --- 专项训练模式 ---
    elif mode == "专项训练":
        st.markdown("---")
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            scenario = st.selectbox("训练场景", list(CASE_SCENARIOS.keys()))
        with sc2:
            if st.button("生成案例"):
                st.session_state.case_mode = scenario
                st.session_state.case_positions = find_case_positions(df, scenario, {})
                st.session_state.case_cursor = 0
                st.rerun()

        if st.session_state.case_mode:
            case_positions = st.session_state.case_positions
            case_cursor = st.session_state.get("case_cursor", 0)
            if case_positions:
                st.markdown(f"**{st.session_state.case_mode}** — 找到 {len(case_positions)} 个案例")
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                with cc1:
                    st.markdown(f"**进度：{case_cursor + 1} / {len(case_positions)}**")
                with cc2:
                    if case_cursor < len(case_positions) - 1:
                        if st.button("下一个案例 ▶"):
                            st.session_state.case_cursor = case_cursor + 1
                            st.rerun()
                    else:
                        st.info("案例已全部完成")
                with cc3:
                    st.caption(CASE_SCENARIOS[st.session_state.case_mode])
                st.session_state.current_index = case_positions[case_cursor]
            else:
                st.warning("当前数据中未找到符合条件的案例。")

    # --- 自由浏览 ---
    else:
        if st.session_state.current_index is None:
            st.session_state.current_index = np.random.randint(min_idx, max_idx)

    # --- 越界修正 ---
    if st.session_state.current_index is None or st.session_state.current_index > max_idx:
        st.session_state.current_index = np.random.randint(min_idx, max_idx)

    current_index = st.session_state.current_index

    # --- 强制复盘 ---
    if st.session_state.forced_review and st.session_state.force_review_bar is not None:
        current_index = st.session_state.force_review_bar
        st.warning(
            f"**强制复盘**：请重新观察 #{st.session_state.force_review_bar}。"
            "你在这个位置连续犯错——重新判断后再提交。"
        )

    # --- 构建窗口 ---
    start_idx = max(0, current_index - CHART_WINDOW)
    chart_df = df.iloc[start_idx: current_index + 1].copy().reset_index(drop=True)
    current_bar = len(chart_df) - 1
    snapshot = build_snapshot(chart_df, current_bar, global_index=current_index)

    # --- 导航（自由浏览模式）---
    if mode == "自由浏览":
        nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 2])
        with nav1:
            if st.button("◀ 上一根"):
                if current_index > min_idx:
                    st.session_state.current_index -= 1
                    st.rerun()
        with nav2:
            if st.button("▶ 下一根"):
                if current_index < max_idx:
                    st.session_state.current_index += 1
                    st.rerun()
        with nav3:
            if st.button("🎲 随机"):
                old = st.session_state.current_index
                for _ in range(100):
                    new = np.random.randint(min_idx, max_idx)
                    if abs(new - old) > RANDOM_MIN_DISTANCE:
                        break
                st.session_state.current_index = new
                st.rerun()
        with nav4:
            st.markdown(f"**全局 #{current_index}** | **窗口 #{current_bar}**")
            # 盲测开关（自由浏览模式）
            blind_toggle = st.checkbox("盲测模式（隐藏系统标注）", value=st.session_state.blind_mode)
            if blind_toggle != st.session_state.blind_mode:
                st.session_state.blind_mode = blind_toggle
                st.rerun()

    # --- 图表 ---
    case_hl = mode == "专项训练"
    fig = build_chart(chart_df, snapshot, current_bar, case_highlight=case_hl,
                      blind_mode=blind_mode)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- 观察面板 ---
    st.markdown("---")
    render_observation_panel(snapshot, blind_mode)

    # --- 自动标签（盲测模式下也折叠）---
    with st.expander("自动标签", expanded=False):
        for t in snapshot.auto_tags:
            st.markdown(f"- {t}")

    # --- 观点生命周期面板 ---
    render_viewpoint_panel()

    # --- 用户判断 ---
    st.markdown("---")
    st.subheader(f"你的读盘 — 窗口 #{current_bar}")

    # V9: 强制预期模式
    replay_sub = st.session_state.get("replay_sub_mode", "标准")
    force_expectation = (mode == "Replay训练" and replay_sub == "强制预期")
    force_viewpoint_update = (mode == "Replay训练" and replay_sub == "观点更新")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        market_control = st.radio("谁控制市场？", ["多头控制", "空头控制", "多空平衡"])
    with r1c2:
        market_type = st.radio("市场类型？", ["趋势", "区间", "突破尝试", "反转尝试"])
    with r1c3:
        momentum_quality = st.radio("推进质量？", ["强推进", "健康推进", "弱推进", "推进衰减"])

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        expectation = st.radio("更可能？", ["延续", "反转", "继续区间"])
    with r2c2:
        breakout_quality = st.radio("突破质量？", ["突破成功概率高", "突破失败概率高", "暂时不明确"])
    with r2c3:
        structure_events = st.multiselect("结构事件", STRUCTURE_EVENTS)

    # V9: 强制预期输入
    invalidate_cond = ""
    ft_cond = ""
    if force_expectation or force_viewpoint_update:
        st.markdown("**强制预期**（必须填写）")
        ec1, ec2 = st.columns(2)
        with ec1:
            invalidate_cond = st.text_input("失效条件（什么情况下你的判断错了？）",
                                            max_chars=100, placeholder="如：跌破xxx / 反包前一根阳线")
        with ec2:
            ft_cond = st.text_input("跟进确认条件（什么情况下你判断对了？）",
                                    max_chars=100, placeholder="如：下一根Higher Close / 突破前高")

        if force_expectation and (not invalidate_cond or not ft_cond):
            st.warning("请先填写失效条件和跟进确认条件，否则无法提交。")

    # V9: 观点更新
    viewpoint_action = ""
    if force_viewpoint_update and st.session_state.active_viewpoint is not None:
        st.markdown("**观点更新**")
        viewpoint_action = st.radio("你的观点如何变化？",
                                     ["观点加强", "观点减弱", "观点失效", "转入双向交易"])

    # 结构化压力模式
    st.markdown("**你观察到的压力信号**")
    bp1, bp2 = st.columns(2)
    with bp1:
        st.markdown("**多头压力**（多头推进变难的证据）")
        bull_pressure = st.multiselect("选择你看到的多头压力", BULL_PRESSURE_PATTERNS)
    with bp2:
        st.markdown("**空头压力**（空头推进变难的证据）")
        bear_pressure = st.multiselect("选择你看到的空头压力", BEAR_PRESSURE_PATTERNS)

    short_note = st.text_area("一句话总结", max_chars=120, height=60,
                              placeholder="结合位置和压力变化做判断")

    # 提交校验
    can_submit = True
    if force_expectation and (not invalidate_cond or not ft_cond):
        can_submit = False

    submit = st.button("提交判断", disabled=not can_submit)

    # --- 提交 ---
    if submit:
        outcome = validate_outcome(df, current_index, expectation, snapshot)
        loc = snapshot.location
        location_special = any([
            loc.climactic_extension, loc.measured_move_level,
            loc.breakout_pullback_area, loc.near_channel_line,
        ])

        log = {
            "time": str(datetime.now()),
            "bar_index": current_index,
            "window_bar_number": current_bar,
            "market_control": market_control,
            "market_type": market_type,
            "momentum_quality": momentum_quality,
            "expectation": expectation,
            "breakout_quality": breakout_quality,
            "events": structure_events,
            "bull_pressure": bull_pressure,
            "bear_pressure": bear_pressure,
            "note": short_note,
            "always_in": snapshot.always_in.status,
            "always_in_conviction": snapshot.always_in.conviction,
            "tendency_primary": snapshot.tendency.primary,
            "tendency_secondary": snapshot.tendency.secondary,
            "state_transition": snapshot.state_transition.to_display(),
            "ft_acceptance": snapshot.ft_acceptance.acceptance_level,
            "post_failure": snapshot.post_failure.description,
            "location_special": location_special,
            "engine_ft": snapshot.pressure.follow_through,
            "engine_momentum": snapshot.pressure.momentum_shift,
            "engine_range": snapshot.pressure.range_progression,
            "engine_tendency_range": 1.0 if snapshot.tendency.primary in ("区间", "双向交易") else 0.0,
            "outcome": asdict(outcome),
            "forced_review": st.session_state.forced_review,
            "viewpoint_action": viewpoint_action,
            "blind_mode": blind_mode,
        }

        st.session_state.logs.append(log)
        st.session_state.submit_count += 1

        if mode == "Replay训练":
            st.session_state.replay_judgments[current_bar] = log

        # V9: 观点生命周期更新
        if force_viewpoint_update and st.session_state.active_viewpoint is not None:
            vp = st.session_state.active_viewpoint
            state_map = {"观点加强": "加强", "观点减弱": "减弱", "观点失效": "失效", "转入双向交易": "转换"}
            new_state = state_map.get(viewpoint_action, vp.state)
            update_viewpoint(vp, new_state)
            if new_state == "失效":
                st.session_state.active_viewpoint = None
        elif force_expectation or force_viewpoint_update:
            # 创建新观点
            direction = "多头" if market_control == "多头控制" else (
                "空头" if market_control == "空头控制" else "中性")
            vp = create_viewpoint(direction, expectation,
                                  invalidate_cond or "未设定", ft_cond or "未设定")
            st.session_state.active_viewpoint = vp
            st.session_state.viewpoint_history.append(asdict(vp))

        # 退出强制复盘
        st.session_state.forced_review = False
        st.session_state.force_review_bar = None

        # --- 行为验证 ---
        st.markdown("---")
        st.subheader("行为验证")
        o_cols = st.columns(5)
        checks = [
            ("跟进", outcome.got_follow_through),
            ("对手被困", outcome.trapped_traders),
            ("突破成功", outcome.breakout_succeeded),
            ("反转成立", outcome.reversal_held),
            ("区间继续", outcome.range_continued),
        ]
        for col, (label, val) in zip(o_cols, checks):
            icon = "✅" if val else "❌"
            col.markdown(f"{icon} **{label}**")
        st.caption(outcome.description)

        # V9: "为什么失败" — 你忽略了什么
        if outcome.what_you_missed != "无明显忽略":
            st.markdown("---")
            st.subheader("你忽略了什么")
            st.error(f"**{outcome.what_you_missed}**")

    # --- 连续错误检测 + 强制复盘 ---
    consecutive_errors = detect_consecutive_errors(st.session_state.logs)
    if consecutive_errors:
        st.markdown("---")
        st.error("**连续错误检测**")
        for err in consecutive_errors:
            st.markdown(f"**{err['pattern']}**（连续 {err['count']} 次）")
            st.warning(err["suggestion"])
            if err["count"] >= 3 and not st.session_state.forced_review:
                review_bar = err["bars"][-1]
                if st.button("进入强制复盘"):
                    st.session_state.forced_review = True
                    st.session_state.force_review_bar = review_bar
                    st.rerun()

    # --- 压力模式统计 ---
    if len(st.session_state.logs) >= 5:
        st.markdown("---")
        pressure_stats = get_pressure_pattern_stats(st.session_state.logs)
        ps1, ps2 = st.columns(2)
        with ps1:
            st.markdown("**你最常识别的压力**")
            for p, cnt in pressure_stats["most_selected"][:5]:
                st.caption(f"- 「{p}」{cnt}次")
        with ps2:
            if pressure_stats["most_missed"]:
                st.markdown("**你最容易忽略的压力**")
                for p, cnt in pressure_stats["most_missed"][:5]:
                    st.caption(f"- 「{p}」被忽略{cnt}次")
            else:
                st.markdown("**忽略统计** — 暂无")

    # --- V9: 三档偏差画像 ---
    st.markdown("---")
    st.subheader("偏差画像（三档统计）")

    if len(st.session_state.logs) > 0:
        three_tier = build_bias_statistics_three_tier(st.session_state.logs)
        if three_tier:
            for tier_name, tier_data in three_tier.items():
                with st.expander(tier_name, expanded=(tier_name.startswith("短期"))):
                    total = tier_data["total"]
                    st.markdown(f"样本量：{total} 次")
                    top = tier_data["top_errors"]
                    if top and top[0][1] > 0:
                        for err_name, err_count in top:
                            if err_count > 0:
                                bar_pct = min(100, int(err_count / total * 100 * 3))
                                st.markdown(f"- **{err_name}**: {err_count}/{total}")
                    else:
                        st.caption("该档暂无突出偏差")
        else:
            st.caption("样本不足，无法统计。至少需要 5 次判断。")

    # --- AI 偏差纠正 ---
    if len(st.session_state.logs) >= 5 and consecutive_errors:
        st.markdown("---")
        st.subheader("偏差纠正")
        ai_correction = get_bias_correction(
            client, "gpt-5.4-nano",
            consecutive_errors, get_pressure_pattern_stats(st.session_state.logs),
            st.session_state.logs,
        )
        st.info(ai_correction)

    # --- Replay 总结 ---
    if mode == "Replay训练" and st.session_state.replay_cursor >= len(st.session_state.replay_positions) - 1:
        n = len(st.session_state.replay_judgments)
        if n > 0:
            st.markdown("---")
            st.subheader("Replay 训练总结")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("logs", exist_ok=True)
            rp_path = os.path.join("logs", f"replay_{ts}.json")
            with open(rp_path, "w", encoding="utf-8") as f:
                json.dump(list(st.session_state.replay_judgments.values()), f, ensure_ascii=False, indent=4)
            st.success(f"Replay {n} 根已完成，记录已保存到 {rp_path}")

            # V9: Replay 中观点生命周期统计
            if st.session_state.viewpoint_history:
                vh = st.session_state.viewpoint_history
                avg_alive = sum(v["bars_alive"] for v in vh) / len(vh)
                expired = sum(1 for v in vh if v["state"] == "失效")
                st.markdown(f"**观点统计**：平均存活 {avg_alive:.1f} 根，失效 {expired}/{len(vh)}")

    # --- 导出 ---
    if st.button("导出训练日志"):
        path = export_logs(st.session_state.logs)
        st.success(f"已导出到 {path}")

    # --- 底部 ---
    st.markdown("---")
    st.caption("""**V9 理念：市场正在变成什么？**

核心变化（V8 -> V9）：
1. 状态转移系统 — 不分类市场，描述市场正在变成什么
2. FT Acceptance Engine — 不数K线，观察市场是否接受价格
3. 观点生命周期 — 强制预期 -> 持续更新 -> 失效检测
4. 失败后行为追踪 — 不只标记失败，追踪失败之后发生了什么
5. 盲测模式 — 隐藏所有系统标注，你自己读
6. 三档偏差统计 — 短期/中期/长期，避免被短期随机性污染
7. "为什么失败" — 不只说你错了，说你忽略了什么行为变化

**核心问题：当前市场，真的发生控制权转移了吗？转移后市场做了什么？**""")


if __name__ == "__main__":
    main()
