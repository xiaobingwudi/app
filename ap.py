# =========================================================
# Al Brooks 读盘训练器 V10
# =========================================================
#
# V10 核心理念（对 V9 的方向修正）：
#
#   V9 的问题：
#     - 嘴上说"不分类"，代码内部仍在分类/评分/阈值判断
#     - bull_e > bear_e + 1 仍是"证据累计评分"
#     - hc >= 7, ratio > 1.3 仍是"静态阈值分类器"
#     - 功能越来越多，UI越来越复杂，用户在看系统而非市场
#
#   V10 的核心转变：
#     1. 系统不分类，只描述行为变化（"实体从0.65缩小到0.38"）
#     2. 用户自己判断，系统只展示"发生了什么"
#     3. 行为是连续的，不是离散快照
#     4. 控制权转移是过程（6个阶段），不是 boolean
#     5. AI 不解释，只指向（"重新观察第132-145根"）
#     6. 删除一半标签：只保留 控制权 / 接受 / 衰减 / trapped / 失败后行为
#     7. Replay 是核心，不是附加功能
#
#   删除的内容：
#     - MarketTendency（分类器）
#     - StateTransition with confidence/trigger_events（过度结构化）
#     - FollowThroughAcceptance（太多 boolean）
#     - PressureSnapshot + 3个observe函数（分类标签）
#     - PostFailureBehavior（太多 boolean）
#     - get_bias_correction（AI不应解释）
#     - 三档偏差统计（噪音）
#     - 压力模式统计（噪音）
#     - 连续错误检测（过度结构化）
#     - 案例库（Replay才是核心）
#     - AlwaysIn 的 bull_e > bear_e + 1 评分
#
#   新增的内容（精简）：
#     - BehaviorChange：一次行为变化的描述
#     - DecayTracker：连续行为衰减追踪
#     - ControlShift：多阶段控制权转移
#     - StateTimeline：行为演化时间线
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

# V10: 删除一半标签，只保留核心
STRUCTURE_EVENTS = [
    "失败突破", "Trapped Trader",
    "推进衰减", "跟进消失",
    "控制权转移", "二次失败",
]

BULL_PRESSURE_PATTERNS = [
    "突破距离缩短",
    "实体缩小",
    "跟进减少（HC减少）",
    "重叠增加",
    "上影线增加",
    "二次突破失败",
]

BEAR_PRESSURE_PATTERNS = [
    "下跌后快速拉回",
    "空头无法收盘新低",
    "阳线反包增加",
    "下影线增多",
    "空头推进变短",
    "买盘涌入",
]


# =========================================================
# V10 数据类 — 最小化
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
    """
    V10 核心：一次行为变化。
    不分类，只描述"什么在变化"和"变化方向"。
    """
    what: str           # 什么在变化（如"实体大小"、"HC"、"重叠"）
    direction: str      # 增加 / 减少 / 稳定
    from_desc: str      # 变化前描述
    to_desc: str        # 变化后描述
    bars: str           # 涉及的K线范围


@dataclass
class DecayTracker:
    """
    V10 新增：行为衰减追踪。
    追踪连续恶化过程，不是瞬时判断。
    """
    body_shrinking: int           # 实体连续缩小次数
    hc_decreasing: int            # HC连续减少次数
    tail_growing: int             # 尾巴连续增加次数
    reversal_frequency: int       # 最近N根中反包次数
    pullback_deepening: str       # 回调深度趋势
    breakout_distance: str        # 突破距离趋势
    summary: str                  # 衰减过程描述（自然语言）


@dataclass
class ControlShift:
    """
    V10 新增：控制权转移。
    不是 boolean，是 6 个阶段的连续过程。
    """
    push_failed: bool
    opposite_testing: str         # 无 / 测试中 / 测试成功
    original_ft_gone: bool
    opposite_accepted: str         # 无 / 部分接受 / 完全接受
    trapped_formed: bool
    second_attempt: str           # 无 / 尝试中 / 失败
    stage: int                    # 0=无转移 1-6=转移各阶段
    description: str              # 过程描述
    phase: str = "无"             # 便捷属性


@dataclass
class Viewpoint:
    """观点生命周期 — V10 极简版"""
    bar: int
    direction: str            # 多 / 空 / 观望
    expectation: str          # 用户自己的描述
    timestamp: str            # 创建时间
    status: str               # active / expired


@dataclass
class Outcome:
    """行为验证 — 只展示原始行为，不说你错了"""
    predict_bar: int
    outcome_bar: int
    move: float
    move_pct: float
    path_observations: list       # 原始行为观察列表
    suggest_replay_range: tuple   # (start, end) 建议重新观察的K线范围


@dataclass
class MarketSnapshot:
    """
    V10: 快照 — 只有原始观察，没有分类。
    """
    bar_index: int
    time: str
    open: float
    high: float
    low: float
    close: float
    control: list                 # 控制权观察
    location: list                # 位置观察
    behavior_changes: list        # 行为变化描述文本列表
    decay: list                   # 衰减描述文本列表
    control_shift: list           # 控制权转移描述文本列表
    legs: list                    # 波段描述文本列表
    swings: list                  # Swing描述文本列表

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
        "replay_sub_mode": "标准",
        "blind_mode": False,
        "active_viewpoint": None,
        "viewpoint_history": [],
        "timeline_history": [],      # V10: 全局行为演化时间线
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =========================================================
# 基础检测：Swing + HH/HL/LH/LL
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
# Leg Engine — 只存原始数据，不分类
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
        # V10: 不分类 momentum，只存原始数据
        legs.append(Leg(
            start_idx=s1.index, end_idx=s2.index, direction=direction,
            bar_count=s2.index - s1.index + 1, price_start=ps, price_end=pe,
            price_range=abs(pe - ps), overlap_ratio=round(overlap_ratio, 3),
            body_avg=round(body_avg, 3), tail_avg=round(tail_avg, 3),
        ))
    return legs

# =========================================================
# V10 核心：行为变化检测
# 不分类，只描述"什么在变化"和"变化方向"
# =========================================================

def detect_behavior_changes(chart_df: pd.DataFrame, legs: list) -> list:
    """
    V10 核心：检测最近K线中的行为变化。
    只描述变化，不做判断。
    
    返回 BehaviorChange 列表。
    """
    n = min(15, len(chart_df))
    if n < 5:
        return []
    
    recent = chart_df.tail(n)
    mid = n // 2
    first_half = recent.iloc[:mid]
    second_half = recent.iloc[mid:]
    
    changes = []
    
    # 1. 实体大小变化
    bodies_1 = []
    bodies_2 = []
    for i in range(len(first_half)):
        bar = first_half.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng > 1e-9:
            bodies_1.append(abs(bar["close"] - bar["open"]) / rng)
    for i in range(len(second_half)):
        bar = second_half.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng > 1e-9:
            bodies_2.append(abs(bar["close"] - bar["open"]) / rng)
    
    if bodies_1 and bodies_2:
        avg1, avg2 = np.mean(bodies_1), np.mean(bodies_2)
        if abs(avg2 - avg1) > 0.1:
            changes.append(BehaviorChange(
                what="实体大小",
                direction="增加" if avg2 > avg1 else "减少",
                from_desc=f"{avg1:.2f}",
                to_desc=f"{avg2:.2f}",
                bars=f"前半 {len(bodies_1)}根 vs 后半 {len(bodies_2)}根",
            ))
    
    # 2. HC/LC 变化
    hc_1 = sum(1 for i in range(1, len(first_half)) if first_half.iloc[i]["close"] > first_half.iloc[i - 1]["close"])
    lc_1 = sum(1 for i in range(1, len(first_half)) if first_half.iloc[i]["close"] < first_half.iloc[i - 1]["close"])
    hc_2 = sum(1 for i in range(1, len(second_half)) if second_half.iloc[i]["close"] > second_half.iloc[i - 1]["close"])
    lc_2 = sum(1 for i in range(1, len(second_half)) if second_half.iloc[i]["close"] < second_half.iloc[i - 1]["close"])
    
    net_1 = hc_1 - lc_1
    net_2 = hc_2 - lc_2
    if abs(net_2 - net_1) >= 2:
        changes.append(BehaviorChange(
            what="HC/LC净差",
            direction="偏多增强" if net_2 > net_1 else "偏空增强",
            from_desc=f"HC-LC={net_1:+d}",
            to_desc=f"HC-LC={net_2:+d}",
            bars=f"#{chart_df.index[0]}-{chart_df.index[mid]} vs #{chart_df.index[mid]}-{chart_df.index[-1]}",
        ))
    
    # 3. 重叠变化
    overlaps_1, overlaps_2 = [], []
    for i in range(1, len(first_half)):
        prev, cur = first_half.iloc[i - 1], first_half.iloc[i]
        ov = max(0, min(prev["high"], cur["high"]) - max(prev["low"], cur["low"]))
        un = max(prev["high"], cur["high"]) - min(prev["low"], cur["low"])
        overlaps_1.append(ov / un if un > 1e-9 else 0)
    for i in range(1, len(second_half)):
        prev, cur = second_half.iloc[i - 1], second_half.iloc[i]
        ov = max(0, min(prev["high"], cur["high"]) - max(prev["low"], cur["low"]))
        un = max(prev["high"], cur["high"]) - min(prev["low"], cur["low"])
        overlaps_2.append(ov / un if un > 1e-9 else 0)
    
    if overlaps_1 and overlaps_2:
        avg_ov_1, avg_ov_2 = np.mean(overlaps_1), np.mean(overlaps_2)
        if abs(avg_ov_2 - avg_ov_1) > 0.15:
            changes.append(BehaviorChange(
                what="K线重叠",
                direction="增加" if avg_ov_2 > avg_ov_1 else "减少",
                from_desc=f"{avg_ov_1:.0%}",
                to_desc=f"{avg_ov_2:.0%}",
                bars=f"前半 vs 后半",
            ))
    
    # 4. 振幅变化
    ranges_1 = [(first_half.iloc[i]["high"] - first_half.iloc[i]["low"]) for i in range(len(first_half))]
    ranges_2 = [(second_half.iloc[i]["high"] - second_half.iloc[i]["low"]) for i in range(len(second_half))]
    if ranges_1 and ranges_2:
        avg_r1, avg_r2 = np.mean(ranges_1), np.mean(ranges_2)
        if abs(avg_r2 - avg_r1) / max(avg_r1, 1e-9) > 0.2:
            changes.append(BehaviorChange(
                what="振幅",
                direction="放大" if avg_r2 > avg_r1 else "缩小",
                from_desc=f"{avg_r1:.2f}",
                to_desc=f"{avg_r2:.2f}",
                bars=f"前半 vs 后半",
            ))
    
    # 5. 波段对比（如果有足够的legs）
    recent_legs = [l for l in legs if l.end_idx <= len(chart_df) - 1]
    if len(recent_legs) >= 3:
        last3 = recent_legs[-3:]
        body_avgs = [l.body_avg for l in last3]
        if len(set(round(b, 2) for b in body_avgs)) >= 2:
            trend = "缩小" if body_avgs[-1] < body_avgs[0] else "放大"
            changes.append(BehaviorChange(
                what="波段实体",
                direction=trend,
                from_desc=f"波段1={body_avgs[0]:.2f}",
                to_desc=f"波段3={body_avgs[-1]:.2f}",
                bars=f"最近3个波段",
            ))
    
    return changes


# =========================================================
# V10 核心：衰减追踪器
# 追踪连续恶化过程
# =========================================================

def track_decay(chart_df: pd.DataFrame, legs: list) -> DecayTracker:
    """
    V10: 不是判断"是否衰减"，而是描述"衰减过程"。
    每个指标都是连续观察，不是阈值判断。
    """
    n = min(20, len(chart_df))
    recent = chart_df.tail(n)
    if n < 5:
        return DecayTracker(0, 0, 0, 0, "数据不足", "数据不足", "数据不足")
    
    # 1. 实体连续缩小
    body_shrinking = 0
    bodies = []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        if rng > 1e-9:
            bodies.append(abs(bar["close"] - bar["open"]) / rng)
        else:
            bodies.append(0)
    # 从后往前数连续缩小的次数
    for i in range(len(bodies) - 1, 0, -1):
        if bodies[i] < bodies[i - 1] * 0.9:  # 相对缩小
            body_shrinking += 1
        else:
            break
    
    # 2. HC连续减少
    hc_decreasing = 0
    hc_series = []
    for i in range(1, len(recent)):
        if recent.iloc[i]["close"] > recent.iloc[i - 1]["close"]:
            hc_series.append(1)
        else:
            hc_series.append(0)
    # 从后往前数连续0的次数
    for i in range(len(hc_series) - 1, -1, -1):
        if hc_series[i] == 0:
            hc_decreasing += 1
        else:
            break
    
    # 3. 尾巴连续增加
    tail_growing = 0
    tails = []
    for i in range(len(recent)):
        bar = recent.iloc[i]
        rng = bar["high"] - bar["low"]
        body = abs(bar["close"] - bar["open"])
        tails.append((rng - body) / body if body > 1e-9 else 5.0)
    for i in range(len(tails) - 1, 0, -1):
        if tails[i] > tails[i - 1] * 1.1:
            tail_growing += 1
        else:
            break
    
    # 4. 反包频率
    reversal_count = 0
    for i in range(2, len(recent)):
        prev = recent.iloc[i - 1]
        cur = recent.iloc[i]
        if (prev["close"] > prev["open"] and cur["close"] < cur["open"] and cur["close"] < prev["open"]) or \
           (prev["close"] < prev["open"] and cur["close"] > cur["open"] and cur["close"] > prev["open"]):
            reversal_count += 1
    
    # 5. 回调深度趋势
    pullback_deepening = "稳定"
    if len(legs) >= 3:
        recent_legs = [l for l in legs if l.end_idx <= len(chart_df) - 1]
        if len(recent_legs) >= 2:
            # 计算回调占推进的比例
            pullback_ratios = []
            for i in range(len(recent_legs) - 1):
                push = recent_legs[i]
                pull = recent_legs[i + 1]
                if push.price_range > 1e-9:
                    # 如果方向相反，计算回调比例
                    if push.direction != pull.direction:
                        ratio = pull.price_range / push.price_range
                        pullback_ratios.append(ratio)
            if len(pullback_ratios) >= 2:
                if pullback_ratios[-1] > pullback_ratios[-2] * 1.2:
                    pullback_deepening = "回调在变深"
                elif pullback_ratios[-1] < pullback_ratios[-2] * 0.8:
                    pullback_deepening = "回调在变浅"
    
    # 6. 突破距离趋势
    breakout_distance = "稳定"
    if len(legs) >= 3:
        recent_legs = [l for l in legs if l.end_idx <= len(chart_df) - 1]
        if len(recent_legs) >= 2:
            dists = [l.price_range for l in recent_legs[-3:]]
            if len(dists) >= 2 and dists[-1] < dists[-2] * 0.7:
                breakout_distance = "推进距离在缩短"
            elif len(dists) >= 2 and dists[-1] > dists[-2] * 1.3:
                breakout_distance = "推进距离在扩大"
    
    # 汇总描述
    parts = []
    if body_shrinking >= 3:
        parts.append(f"实体连续{body_shrinking}根缩小")
    if hc_decreasing >= 3:
        parts.append(f"HC连续{hc_decreasing}根减少")
    if tail_growing >= 2:
        parts.append(f"尾巴连续{tail_growing}根增长")
    if reversal_count >= 3:
        parts.append(f"最近{n}根中{reversal_count}次反包")
    if pullback_deepening != "稳定":
        parts.append(pullback_deepening)
    if breakout_distance != "稳定":
        parts.append(breakout_distance)
    
    summary = "；".join(parts) if parts else "未检测到明显衰减过程"
    
    return DecayTracker(
        body_shrinking, hc_decreasing, tail_growing,
        reversal_count, pullback_deepening, breakout_distance, summary,
    )

# =========================================================
# V10 核心：控制权转移检测
# 不是 boolean，是 6 个阶段的连续过程
# =========================================================

def detect_control_shift(
    chart_df: pd.DataFrame, swings: list, legs: list, current_bar: int,
) -> ControlShift:
    """
    V10: 控制权转移是过程，不是判断。
    
    阶段 0: 无转移迹象
    阶段 1: 原方向推进失败
    阶段 2: 反方向开始测试
    阶段 3: 原方向跟进消失
    阶段 4: 反方向获得接受
    阶段 5: Trapped Trader 形成
    阶段 6: 二次原方向尝试失败
    """
    n = min(15, len(chart_df))
    recent = chart_df.tail(n)
    if n < 5:
        return ControlShift(False, "无", False, "无", False, "无", 0, "数据不足")
    
    # 确定当前主导方向
    last_leg = next((l for l in reversed(legs) if l.end_idx <= current_bar), None)
    if last_leg is None:
        return ControlShift(False, "无", False, "无", False, "无", 0, "无波段数据")
    
    original_dir = last_leg.direction  # "bull" or "bear"
    
    # 阶段 1: 原方向推进失败
    push_failed = False
    if n >= 3:
        # 最近的方向推进是否失败（收盘在推进起点以下）
        cur = recent.iloc[-1]
        if original_dir == "bull":
            push_failed = cur["close"] < cur["open"] and len(recent) >= 2
            if not push_failed:
                # 检查大阳线后是否被反包
                big = recent.iloc[-3] if len(recent) >= 3 else None
                if big and big["close"] > big["open"]:
                    big_range = big["high"] - big["low"]
                    big_body = abs(big["close"] - big["open"])
                    if big_range > 1e-9 and big_body / big_range > 0.5:
                        if cur["close"] < big["open"]:
                            push_failed = True
        else:
            push_failed = cur["close"] > cur["open"] and len(recent) >= 2
            if not push_failed:
                big = recent.iloc[-3] if len(recent) >= 3 else None
                if big and big["close"] < big["open"]:
                    big_range = big["high"] - big["low"]
                    big_body = abs(big["close"] - big["open"])
                    if big_range > 1e-9 and big_body / big_range > 0.5:
                        if cur["close"] > big["open"]:
                            push_failed = True
    
    # 阶段 2: 反方向测试
    opposite_testing = "无"
    if push_failed:
        if original_dir == "bull":
            # 空头是否在测试前低？
            prior_lows = [s.price for s in swings if s.kind == "SL" and s.index <= current_bar]
            if prior_lows:
                lowest = min(prior_lows[-3:])
                if recent.iloc[-1]["low"] <= lowest:
                    opposite_testing = "测试中"
                if recent.iloc[-1]["close"] < lowest:
                    opposite_testing = "测试成功"
        else:
            prior_highs = [s.price for s in swings if s.kind == "SH" and s.index <= current_bar]
            if prior_highs:
                highest = max(prior_highs[-3:])
                if recent.iloc[-1]["high"] >= highest:
                    opposite_testing = "测试中"
                if recent.iloc[-1]["close"] > highest:
                    opposite_testing = "测试成功"
    
    # 阶段 3: 原方向跟进消失
    original_ft_gone = False
    if original_dir == "bull":
        hc_count = 0
        for i in range(max(0, len(recent) - 6), len(recent) - 1):
            if recent.iloc[i + 1]["close"] > recent.iloc[i]["close"]:
                hc_count += 1
        if hc_count <= 1 and len(recent) >= 6:
            original_ft_gone = True
    else:
        lc_count = 0
        for i in range(max(0, len(recent) - 6), len(recent) - 1):
            if recent.iloc[i + 1]["close"] < recent.iloc[i]["close"]:
                lc_count += 1
        if lc_count <= 1 and len(recent) >= 6:
            original_ft_gone = True
    
    # 阶段 4: 反方向获得接受
    opposite_accepted = "无"
    if original_dir == "bull" and original_ft_gone:
        # 空头K线是否被后续K线接受（价格维持在新低位附近）
        bear_bars = []
        for i in range(len(recent) - 1, max(len(recent) - 6, -1), -1):
            if recent.iloc[i]["close"] < recent.iloc[i]["open"]:
                bear_bars.append(i)
        if len(bear_bars) >= 2:
            first_bear = bear_bars[0]
            first_bear_low = recent.iloc[first_bear]["low"]
            # 后续K线是否维持在低位
            maintained = sum(1 for i in range(first_bear + 1, len(recent))
                           if recent.iloc[i]["low"] <= first_bear_low * 1.005)
            total = len(recent) - first_bear - 1
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
            first_bull = bull_bars[0]
            first_bull_high = recent.iloc[first_bull]["high"]
            maintained = sum(1 for i in range(first_bull + 1, len(recent))
                           if recent.iloc[i]["high"] >= first_bull_high * 0.995)
            total = len(recent) - first_bull - 1
            if total > 0 and maintained / total >= 0.8:
                opposite_accepted = "完全接受"
            elif total > 0 and maintained / total >= 0.6:
                opposite_accepted = "部分接受"
    
    # 阶段 5: Trapped Trader
    trapped_formed = False
    if opposite_testing == "测试成功" or opposite_accepted in ("部分接受", "完全接受"):
        # 检测原方向再次尝试但失败
        if len(recent) >= 4:
            # 找原方向的尝试
            for i in range(len(recent) - 2, max(len(recent) - 6, -1), -1):
                bar = recent.iloc[i]
                if original_dir == "bull" and bar["close"] > bar["open"] and bar["high"] > recent.iloc[-1]["high"]:
                    # 多头试图创新高但最终失败
                    if recent.iloc[-1]["close"] < bar["close"]:
                        trapped_formed = True
                        break
                elif original_dir == "bear" and bar["close"] < bar["open"] and bar["low"] < recent.iloc[-1]["low"]:
                    if recent.iloc[-1]["close"] > bar["close"]:
                        trapped_formed = True
                        break
    
    # 阶段 6: 二次原方向尝试
    second_attempt = "无"
    if trapped_formed:
        # 原方向是否再次尝试？
        if len(recent) >= 2:
            last = recent.iloc[-1]
            if original_dir == "bull" and last["close"] > last["open"]:
                prior_highs = [s.price for s in swings if s.kind == "SH" and s.index <= current_bar]
                if prior_highs and last["high"] < max(prior_highs[-3:]):
                    second_attempt = "尝试中"
                elif prior_highs and last["close"] < max(prior_highs[-3:]):
                    second_attempt = "失败"
            elif original_dir == "bear" and last["close"] < last["open"]:
                prior_lows = [s.price for s in swings if s.kind == "SL" and s.index <= current_bar]
                if prior_lows and last["low"] > min(prior_lows[-3:]):
                    second_attempt = "尝试中"
                elif prior_lows and last["close"] > min(prior_lows[-3:]):
                    second_attempt = "失败"
    
    # 确定阶段
    stage = 0
    if push_failed: stage = 1
    if opposite_testing != "无": stage = 2
    if original_ft_gone: stage = max(stage, 3)
    if opposite_accepted != "无": stage = max(stage, 4)
    if trapped_formed: stage = max(stage, 5)
    if second_attempt != "无": stage = max(stage, 6)
    
    # 生成过程描述
    narrative_parts = []
    dir_name = "多头" if original_dir == "bull" else "空头"
    opp_name = "空头" if original_dir == "bull" else "多头"
    if stage == 0:
        narrative = f"{dir_name}主导，无转移迹象"
    else:
        if push_failed: narrative_parts.append(f"{dir_name}推进失败")
        if opposite_testing != "无": narrative_parts.append(f"{opp_name}{opposite_testing}")
        if original_ft_gone: narrative_parts.append(f"{dir_name}跟进消失")
        if opposite_accepted != "无": narrative_parts.append(f"{opp_name}被{opposite_accepted}")
        if trapped_formed: narrative_parts.append("Trapped Trader 形成")
        if second_attempt != "无": narrative_parts.append(f"{dir_name}二次尝试{second_attempt}")
        narrative = " → ".join(narrative_parts) if narrative_parts else "无明确转移过程"
    
    phase_map = {0: "无转移", 1: "推进失败", 2: "反向测试", 3: "跟进消失",
                4: "反向接受", 5: "Trapped", 6: "二次尝试"}
    
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
# V10: 控制权观察（替代 AlwaysIn 分类）
# 不输出"Always In Long"，只描述观察到的证据
# =========================================================

def observe_control(chart_df: pd.DataFrame, legs: list, swings: list, current_bar: int) -> list:
    """
    V10: 不分类 AlwaysIn，只描述原始观察。
    用户自己从观察中推断控制权。
    """
    observations = []
    
    last_leg = next((l for l in reversed(legs) if l.end_idx <= current_bar), None)
    if last_leg:
        observations.append(f"最近波段: {last_leg.direction}方向, {last_leg.bar_count}根, 实体{last_leg.body_avg:.0%}")
    
    if len(legs) >= 2:
        prev = next((l for l in reversed(legs) if l != last_leg and l.end_idx <= current_bar), None)
        if prev and last_leg:
            if last_leg.direction != prev.direction:
                # 回调分析
                if last_leg.price_range > 1e-9:
                    ratio = prev.price_range / last_leg.price_range
                    observations.append(f"回调/推进比: {ratio:.1f}x")
            else:
                # 同方向
                if last_leg.body_avg > 1e-9 and prev.body_avg > 1e-9:
                    ratio = last_leg.body_avg / prev.body_avg
                    observations.append(f"同方向波段实体比: {ratio:.1f}x")
    
    # HC/LC 观察
    recent = chart_df.tail(10)
    if len(recent) >= 5:
        hc = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["close"] > recent.iloc[i - 1]["close"])
        lc = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["close"] < recent.iloc[i - 1]["close"])
        observations.append(f"最近10根: HC={hc}, LC={lc}, 净差={hc - lc:+d}")
    
    # 最近Swing
    recent_swings = [s for s in swings if s.index <= current_bar]
    if recent_swings:
        last_swing = recent_swings[-1]
        observations.append(f"最近Swing: {'High' if last_swing.kind == 'SH' else 'Low'} #{last_swing.index}")
    
    return observations


# =========================================================
# V10: 位置观察（替代 LocationContext 布尔值）
# 不输出布尔值，只描述位置事实
# =========================================================

def observe_location(chart_df, swings, legs, current_bar) -> list:
    """
    V10: 不输出布尔值，只描述位置事实。
    """
    observations = []
    if len(chart_df) == 0 or current_bar < 0:
        return ["数据不足"]
    
    cur = chart_df.iloc[current_bar]
    full_high = chart_df["high"].max()
    full_low = chart_df["low"].min()
    full_range = full_high - full_low
    
    if full_range < 1e-9:
        return ["无足够价格变动"]
    
    # 当前位置在全局范围中的位置
    pos = (cur["close"] - full_low) / full_range
    observations.append(f"价格位置: 全局{pos:.0%}")
    
    # 距离前高前低
    prior_highs = [s.price for s in swings if s.kind == "SH" and s.index <= current_bar]
    prior_lows = [s.price for s in swings if s.kind == "SL" and s.index <= current_bar]
    
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
                observations.append(f"在区间内: {range_pos:.0%}（区间宽{rt - rb:.2f}）")
    
    return observations


# =========================================================
# V10: build_snapshot — 组装所有观察（不分类、不评分）
# =========================================================

def build_snapshot(chart_df, swings, legs, current_bar, session):
    """V10: 组装当前bar的市场快照，只有原始观察，没有分类标签。"""
    if current_bar < 0 or current_bar >= len(chart_df):
        return None
    
    # 基础数据
    cur = chart_df.iloc[current_bar]
    visible = chart_df.iloc[:current_bar + 1]
    
    # 各维度观察（都是原始事实描述）
    control_obs = observe_control(chart_df, swings, legs, current_bar)
    location_obs = observe_location(chart_df, swings, legs, current_bar)
    
    # 行为变化
    behavior_changes = detect_behavior_changes(chart_df, swings, legs, current_bar)
    bc_texts = [f"{bc.what} {bc.direction}: {bc.from_desc} → {bc.to_desc}" for bc in behavior_changes]
    
    # 衰减追踪
    decay = track_decay(chart_df, swings, legs, current_bar)
    decay_texts = []
    if decay:
        max_decay = max(decay.body_shrinking, decay.hc_decreasing,
                       decay.tail_growing, decay.reversal_frequency)
        if max_decay >= 3:
            decay_texts.append(f"衰减指标: 实体缩小{decay.body_shrinking}次, HC减少{decay.hc_decreasing}次, "
                             f"尾长增加{decay.tail_growing}次, 反包{decay.reversal_frequency}次")
            if decay.pullback_deepening and decay.pullback_deepening != "数据不足":
                decay_texts.append(f"  回调: {decay.pullback_deepening}")
    
    # 控制权转移
    control_shift = detect_control_shift(chart_df, swings, legs, current_bar)
    shift_texts = []
    if control_shift:
        shift_texts.append(f"控制权转移阶段: {control_shift.phase}")
        if control_shift.description:
            shift_texts.append(f"  {control_shift.description}")
    
    # 波段信息
    leg_texts = []
    recent_legs = [l for l in legs if l.end <= current_bar]
    if recent_legs:
        last = recent_legs[-1]
        leg_texts.append(f"最近波段: {'多' if last.direction == 'bull' else '空'} #{last.start_idx}-{last.end_idx} ({last.price_range:.2f})")
    
    # Swing信息
    swing_texts = []
    recent_swings = [s for s in swings if s.index <= current_bar]
    if len(recent_swings) >= 2:
        last2 = recent_swings[-2:]
        for s in last2:
            swing_texts.append(f"Swing{'高' if s.kind == 'SH' else '低'} #{s.index} ({s.price:.2f})")
    
    snapshot = MarketSnapshot(
        bar_index=current_bar,
        time=cur.get("datetime", ""),
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
    return snapshot


# =========================================================
# V10: validate_outcome — 只展示原始行为，不说"你错了"
# =========================================================

def validate_outcome(chart_df, swings, legs, session, predict_bar, outcome_bar):
    """
    V10 核心改变：
    - 不判断对错
    - 不给分数
    - 只展示从predict_bar到outcome_bar之间发生了什么
    - 建议用户重新观察的K线范围
    """
    if predict_bar is None or outcome_bar is None:
        return None
    if predict_bar >= len(chart_df) or outcome_bar >= len(chart_df):
        return None
    
    span_start = min(predict_bar, outcome_bar)
    span_end = max(predict_bar, outcome_bar)
    span = chart_df.iloc[span_start:span_end + 1]
    
    if len(span) < 2:
        return None
    
    # 原始价格行为
    start_price = chart_df.iloc[predict_bar]["close"]
    end_price = chart_df.iloc[outcome_bar]["close"]
    move = end_price - start_price
    move_pct = move / start_price * 100 if start_price > 0 else 0
    
    # 路径观察（不是评价，是事实）
    path_obs = []
    
    # 最高最低
    span_high = span["high"].max()
    span_low = span["low"].min()
    path_obs.append(f"区间: {span_low:.2f} ~ {span_high:.2f}")
    path_obs.append(f"价格: {start_price:.2f} → {end_price:.2f} ({move:+.2f}, {move_pct:+.1f}%)")
    
    # 连续同向K线
    close_changes = span["close"].diff().dropna()
    if len(close_changes) > 0:
        pos_count = (close_changes > 0).sum()
        neg_count = (close_changes < 0).sum()
        path_obs.append(f"收盘变化: {pos_count}次上涨, {neg_count}次下跌")
    
    # 实体变化趋势
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
    
    # 是否有反方向突破
    predict_close = chart_df.iloc[predict_bar]["close"]
    if move > 0:
        # 原预期上涨，看是否有被拒绝的时刻
        for i in range(span_start, span_end + 1):
            if chart_df.iloc[i]["low"] < predict_close - abs(move) * 0.3:
                path_obs.append(f"价格曾被拒绝至 {chart_df.iloc[i]['low']:.2f}")
                break
    elif move < 0:
        for i in range(span_start, span_end + 1):
            if chart_df.iloc[i]["high"] > predict_close + abs(move) * 0.3:
                path_obs.append(f"价格曾有反扑至 {chart_df.iloc[i]['high']:.2f}")
                break
    
    # 控制权变化
    shift = detect_control_shift(chart_df, swings, legs, outcome_bar)
    if shift:
        path_obs.append(f"当前控制权阶段: {shift.phase}")
    
    # 衰减
    decay = track_decay(chart_df, swings, legs, outcome_bar)
    max_decay = max(decay.body_shrinking, decay.hc_decreasing,
                   decay.tail_growing, decay.reversal_frequency)
    if max_decay >= 3:
        path_obs.append(f"存在连续衰减（最强指标: {max_decay}次）")
    
    # 建议重新观察的K线范围
    suggest_start = max(0, predict_bar - 3)
    suggest_end = min(len(chart_df) - 1, outcome_bar + 3)
    
    result = Outcome(
        predict_bar=predict_bar,
        outcome_bar=outcome_bar,
        move=move,
        move_pct=move_pct,
        path_observations=path_obs,
        suggest_replay_range=(suggest_start, suggest_end),
    )
    return result



# =========================================================
# V10: 图表构建（盲测模式隐藏标注）
# =========================================================

def build_chart(chart_df, swings, legs, current_bar, snapshot, blind_mode=False):
    """构建Plotly K线图。盲测模式下隐藏所有辅助标注。"""
    fig = go.Figure()
    
    visible_end = current_bar + 1
    visible = chart_df.iloc[:visible_end].copy()
    
    if len(visible) == 0:
        return fig
    
    # K线
    fig.add_trace(go.Candlestick(
        x=visible.index if visible.index.name else range(len(visible)),
        open=visible["open"],
        high=visible["high"],
        low=visible["low"],
        close=visible["close"],
        name="K线",
        increasing_line_color="#e74c3c",
        decreasing_line_color="#2ecc71",
    ))
    
    if not blind_mode and snapshot:
        annotations = []
        
        # 波段标注（极简）
        for s in swings:
            if s.index < visible_end:
                y = s.price
                label = "SH" if s.kind == "SH" else "SL"
                annotations.append(dict(
                    x=s.index, y=y,
                    text=label,
                    showarrow=True, arrowhead=1, arrowcolor="#888",
                    font=dict(size=9, color="#888"),
                    ax=0, ay=-25 if s.kind == "SH" else 25,
                ))
        
        # 行为变化标注（只标位置，不标分类）
        if snapshot.behavior_changes:
            for bc_text in snapshot.behavior_changes[:3]:
                annotations.append(dict(
                    x=current_bar, y=chart_df.iloc[current_bar]["high"],
                    text="行为变化",
                    showarrow=True, arrowhead=1, arrowcolor="#f39c12",
                    font=dict(size=10, color="#f39c12"),
                    ax=0, ay=40,
                ))
        
        # 控制权转移标注
        if snapshot.control_shift:
            annotations.append(dict(
                x=current_bar, y=chart_df.iloc[current_bar]["low"],
                text="控制权变化",
                showarrow=True, arrowhead=1, arrowcolor="#e74c3c",
                font=dict(size=10, color="#e74c3c"),
                ax=0, ay=-40,
            ))
        
        # Swing连线
        swing_highs = [(s.index, s.price) for s in swings if s.kind == "SH" and s.index < visible_end]
        swing_lows = [(s.index, s.price) for s in swings if s.kind == "SL" and s.index < visible_end]
        
        if len(swing_highs) >= 2:
            sx, sy = zip(*sorted(swing_highs))
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers",
                line=dict(color="#3498db", width=1, dash="dash"),
                marker=dict(size=4), showlegend=False))
        
        if len(swing_lows) >= 2:
            sx, sy = zip(*sorted(swing_lows))
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers",
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


def get_ai_pointing(snapshot, session):
    """
    V10: AI 不解释市场，只指向行为变化位置。
    返回建议重新观察的K线索引范围（不是文字解释）。
    """
    if snapshot is None:
        return None
    
    bar = snapshot.bar_index
    suggestions = []
    
    # 有行为变化 → 指向前5根
    if snapshot.behavior_changes:
        suggestions.append((max(0, bar - 5), bar))
    
    # 有衰减 → 指向衰减开始位置
    if snapshot.decay:
        decay_len = 0
        for d in snapshot.decay:
            if "连续衰减:" in d:
                try:
                    decay_len = int(d.split(":")[1].strip().replace("根", ""))
                except:
                    pass
        if decay_len >= 3:
            suggestions.append((max(0, bar - decay_len), bar))
    
    # 有控制权转移 → 指向当前±3根
    if snapshot.control_shift:
        suggestions.append((max(0, bar - 3), min(bar + 3, bar)))
    
    return suggestions if suggestions else None



# =========================================================
# V10: UI 组件（极简 — 用户看市场，不看系统）
# =========================================================

def render_observation_panel(snapshot):
    """
    V10: 只展示5个维度的原始观察，不做分类。
    删除所有标签、分数、百分比置信度。
    """
    if snapshot is None:
        st.info("移动K线到某个位置开始观察")
        return
    
    # 控制权
    st.markdown("**控制权**")
    for obs in snapshot.control:
        st.text(obs)
    
    # 位置
    st.markdown("**位置**")
    for obs in snapshot.location:
        st.text(obs)
    
    # 行为变化（核心）
    st.markdown("**行为变化**")
    if snapshot.behavior_changes:
        for bc in snapshot.behavior_changes:
            st.text(bc)
    else:
        st.text("暂未检测到显著行为变化")
    
    # 衰减
    if snapshot.decay:
        st.markdown("**衰减**")
        for d in snapshot.decay:
            st.text(d)
    
    # 控制权转移
    if snapshot.control_shift:
        st.markdown("**控制权转移**")
        for s in snapshot.control_shift:
            st.text(s)
    
    # 波段 + Swing（折叠）
    with st.expander("波段 / Swing"):
        for t in snapshot.legs:
            st.text(t)
        for t in snapshot.swings:
            st.text(t)


def render_viewpoint_panel(session, current_bar):
    """
    V10: 观点面板 — 强制用户形成预期，系统只记录。
    生命周期：形成 → 更新 → 失效
    """
    st.markdown("---")
    st.markdown("**你的观点**")
    
    # 当前活跃观点
    active = [v for v in session.get("viewpoints", []) if v.status == "active"]
    expired = [v for v in session.get("viewpoints", []) if v.status == "expired"]
    
    if active:
        for v in active:
            st.text(f"#{v.bar} 预期{v.direction}: {v.expectation}")
            if st.button(f"更新观点 #{v.bar}", key=f"update_vp_{v.bar}"):
                v.status = "expired"
                st.rerun()
    else:
        st.text("尚未形成观点")
    
    # 形成新观点
    st.markdown("**记录观点**")
    col1, col2 = st.columns(2)
    with col1:
        direction = st.selectbox("方向", ["多", "空", "观望"], key="vp_direction")
    with col2:
        expectation = st.text_input("预期描述（用你自己的话）", key="vp_expectation",
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
    """
    V10: 行为演化时间线 — 记录每次推进时发生了什么。
    不是统计面板，是过程记录。
    """
    timeline = session.get("timeline", [])
    if not timeline:
        return
    
    st.markdown("---")
    st.markdown("**时间线**")
    
    # 只显示最近15条
    for entry in timeline[-15:]:
        time_str = entry.get("time", "")
        bar_idx = entry.get("bar", "")
        text = entry.get("text", "")
        st.text(f"[{time_str}] #{bar_idx} {text}")


def render_outcome_panel(outcome, session):
    """
    V10: 结果面板 — 只展示原始行为，不判断对错。
    """
    if outcome is None:
        return
    
    st.markdown("---")
    st.markdown("**发生了什么**")
    
    for obs in outcome.path_observations:
        st.text(obs)
    
    # 建议重新观察
    if outcome.suggest_replay_range:
        start, end = outcome.suggest_replay_range
        st.markdown(f"**建议重新观察**: 第{start}根 ~ 第{end}根")
        if st.button("跳转到建议范围"):
            session["current_bar"] = start
            st.rerun()



# =========================================================
# V10: 主函数 — Replay 是核心
# =========================================================

def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器 V10", layout="wide")
    
    init_session()
    
    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("V10 读盘训练器")
        
        symbol = st.text_input("合约代码", value="rb2510", key="symbol_input")
        
        blind_mode = st.checkbox("盲测模式（隐藏标注）", value=False, key="blind_toggle")
        
        if st.button("加载数据", key="load_btn"):
            with st.spinner("加载中..."):
                df = load_data(symbol)
                if df is not None and len(df) > 0:
                    st.session_state["chart_df"] = df
                    st.session_state["swings"] = detect_swings(df)
                    st.session_state["structure"] = detect_market_structure(
                        st.session_state["swings"])
                    st.session_state["legs"] = detect_legs(df)
                    st.session_state["current_bar"] = min(30, len(df) - 1)
                    st.session_state["viewpoints"] = []
                    st.session_state["timeline"] = []
                    st.session_state["data_loaded"] = True
                    st.success(f"已加载 {len(df)} 根K线")
                else:
                    st.error("数据加载失败")
        
        # 训练统计（极简）
        if st.session_state.get("data_loaded"):
            st.markdown("---")
            vp_count = len([v for v in st.session_state.get("viewpoints", []) 
                          if v.status == "active"])
            total_vp = len(st.session_state.get("viewpoints", []))
            tl_count = len(st.session_state.get("timeline", []))
            st.text(f"活跃观点: {vp_count}")
            st.text(f"总观点: {total_vp}")
            st.text(f"时间线条目: {tl_count}")
    
    # ---- 主区域 ----
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器 V10")
        st.markdown("加载合约数据开始训练。")
        st.markdown("---")
        st.markdown("**V10 核心原则：**")
        st.markdown("- 系统只展示行为变化，不分类、不评分、不说你错了")
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
    
    # ---- K线图 ----
    chart = build_chart(chart_df, swings, legs, current_bar, snapshot, blind_mode)
    st.plotly_chart(chart, use_container_width=True)
    
    # ---- Replay 控制（核心交互） ----
    col_prev, col_next, col_jump, col_reveal = st.columns(4)
    with col_prev:
        if st.button("◀ 前5根", key="prev5"):
            st.session_state["current_bar"] = max(0, current_bar - 5)
            st.rerun()
    with col_next:
        if st.button("后5根 ▶", key="next5"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 5)
            st.rerun()
    with col_jump:
        if st.button("后15根 ▶▶", key="next15"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 15)
            st.rerun()
    with col_reveal:
        if st.button("揭示后20根", key="reveal20"):
            st.session_state["current_bar"] = min(len(chart_df) - 1, current_bar + 20)
            st.rerun()
    
    # 键盘快捷键提示
    st.caption("快捷键: ← 前5根 | → 后5根 | Shift+→ 后15根 | Space 揭示后20根")
    
    # 当前bar信息
    cur = chart_df.iloc[current_bar]
    st.text(f"#{current_bar}  O:{cur['open']:.2f} H:{cur['high']:.2f} L:{cur['low']:.2f} C:{cur['close']:.2f}")
    
    # ---- 观察面板 ----
    col_obs, col_vp = st.columns([1, 1])
    
    with col_obs:
        render_observation_panel(snapshot)
        
        # AI pointing（盲测模式下不显示）
        if not blind_mode:
            pointing = get_ai_pointing(snapshot, st.session_state)
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
            outcome = validate_outcome(chart_df, swings, legs, st.session_state, vp.bar, outcome_bar)
            if outcome:
                render_outcome_panel(outcome, st.session_state)
                vp.status = "expired"
                # 记录到时间线
                tl_entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "bar": current_bar,
                    "text": f"观点#{vp.bar}({vp.direction}) → #{outcome_bar} ({outcome.move:+.2f})"
                }
                st.session_state.setdefault("timeline", []).append(tl_entry)
    
    # 时间线
    render_timeline(st.session_state)
    
    # ---- 键盘控制 ----
    # 使用 st.components + JS 实现键盘快捷键
    keyboard_js = """
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const btns = window.parent.document.querySelectorAll('button[kind="primary"]');
        // 通过 Streamlit 的方式触发按钮比较复杂，这里只做提示
    });
    </script>
    """
    # 注：Streamlit 中键盘控制的可靠实现需要 streamlit-js-eval 或类似方案
    # 当前版本通过按钮操作即可


if __name__ == "__main__":
    main()
