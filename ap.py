"""
Al Brooks 结构训练器 V21 - 深度优化版

核心改进（基于专业点评反馈）：
1. 结构化输入：分段式填空，降低认知负载，形成观察条件反射
2. Brooks原生术语标签：将K线数字翻译为趋势K线/震荡K线/紧迫度等语义标签
3. Al Brooks铁律注入：80%原则、惯性原理、重叠度判断
4. 画像实时喂回：AI根据用户弱项进行针对性提问
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random, json, time, re

# ==================== 品种映射表 ====================
SYMBOL_NAMES = {
    "IF": "沪深300股指", "IH": "上证50股指", "IC": "中证500股指", "IM": "中证1000股指",
    "CU": "沪铜", "AL": "沪铝", "ZN": "沪锌", "PB": "沪铅", "NI": "沪镍", "SN": "沪锡",
    "AU": "黄金", "AG": "白银", "RB": "螺纹钢", "HC": "热轧卷板", "SS": "不锈钢", "WR": "线材",
    "FU": "燃料油", "BU": "沥青", "RU": "天然橡胶", "SC": "原油", "NR": "20号胶", "BC": "国际铜", "LU": "低硫燃油",
    "A": "豆一", "B": "豆二", "M": "豆粕", "Y": "豆油", "P": "棕榈油", "C": "玉米", "CS": "玉米淀粉",
    "JD": "鸡蛋", "L": "聚乙烯", "PP": "聚丙烯", "V": "PVC", "J": "焦炭", "JM": "焦煤", "I": "铁矿石",
    "EG": "乙二醇", "RR": "粳米", "EB": "苯乙烯", "PG": "液化气", "LH": "生猪",
    "CF": "棉花", "SR": "白糖", "TA": "PTA", "ZC": "动力煤", "MA": "甲醇", "RM": "菜粕",
    "OI": "菜油", "FG": "玻璃", "RS": "油菜籽", "WH": "强麦", "PM": "普通小麦",
    "JR": "粳稻", "LR": "晚籼稻", "RI": "早籼稻", "TC": "棉纱", "AP": "苹果", "CJ": "红枣",
    "UR": "尿素", "SA": "纯碱", "SF": "硅铁", "SM": "锰硅", "CY": "棉纱", "PF": "短纤", "PK": "花生",
    "TS": "2年期国债", "TF": "5年期国债", "T": "10年期国债", "TL": "30年期国债",
}

# ==================== 技能定义 ====================
SKILL_CONSTRAINTS = {
    1: {
        "name": "背景阅读",
        "question": "当前市场背景是什么？",
        "forbidden": ["买", "卖", "做多", "做空", "进场", "止损", "目标位", "开仓", "平仓", "做单"],
        "allowed": "趋势方向、高低点序列(HH/HL或LH/LL)、通道斜率、震荡区间边界、结构位(前高/前低/支撑/阻力)",
        "desc": "只允许描述市场结构，禁止任何交易决策类词汇。允许使用结构位词汇。"
    },
    2: {
        "name": "控制权识别",
        "question": "现在谁在控制市场？",
        "forbidden": ["趋势", "方向", "预测", "目标位", "进场", "止损", "做多", "做空"],
        "allowed": "最近3-5根K线谁在主导、推进方实体质量、对手方有无有效反击、结构位附近的行为",
        "desc": "只关注最近几根K线的力量对比，禁止谈论大趋势和预测。"
    },
    3: {
        "name": "推进质量",
        "question": "最近推进的质量如何？",
        "forbidden": ["趋势", "方向", "多空", "预测", "进场", "止损", "目标"],
        "allowed": "K线实体大小、重叠程度、影线长度、收盘位置、动能变化、成交量配合",
        "desc": "只描述K线本身的质量特征，禁止判断方向。"
    },
    4: {
        "name": "回调vs转换",
        "question": "这是正常回调还是控制权转换？",
        "forbidden": ["预测", "目标位", "进场", "止损", "开仓"],
        "allowed": "回调K线数量、回调实体强弱、对手方连续性、有无跟进、结构位是否被突破",
        "desc": "区分回调与转换，禁止谈论后续走势预测。"
    },
    5: {
        "name": "市场接受",
        "question": "市场是否接受了新价格？",
        "forbidden": ["进场", "止损", "目标位", "预测", "做多", "做空"],
        "allowed": "突破后停留几根、有无推回、有无继续推进、成交量确认",
        "desc": "只关注价格突破后的市场反应，禁止交易建议。"
    }
}

# ==================== 技能合格回答标准定义 ====================
# 每个维度的key用于结构化输入，desc用于显示提示
SKILL_QUALITY_STANDARDS = {
    1: {
        "name": "背景阅读",
        "dimensions": [
            {"key": "sequence", "label": "高低点序列", "placeholder": "观察最近10-15根K线的高点和低点，是HH/HL（上升序列）还是LH/LL（下降序列）？"},
            {"key": "boundary", "label": "区间/通道边界", "placeholder": "当前结构的高点区间和低点区间分别在哪个价格区域？"},
            {"key": "position", "label": "当前价格位置", "placeholder": "当前价格在结构中的什么位置？接近上沿/下沿/中间？"}
        ],
        "must_contain": ["高低点序列", "区间/通道边界", "当前价格位置"],
        "reject_if_only": ["上升趋势", "下降趋势", "震荡"],
        "placeholder": "请观察以下维度"
    },
    2: {
        "name": "控制权识别",
        "dimensions": [
            {"key": "initiator", "label": "推进方是谁", "placeholder": "最近3-5根K线，是买方在主导还是卖方在主导？"},
            {"key": "quality", "label": "推进方实体质量", "placeholder": "推进方的K线实体占波幅多少？收盘在什么位置（高位/中位/低位）？"},
            {"key": "counter", "label": "对手方反击情况", "placeholder": "对手方有没有反击？反击的K线特征是什么（实体大小、影线）？"},
            {"key": "follow", "label": "反击是否被跟进", "placeholder": "反击发生后，是否有后续K线跟进？还是被立即反包？"}
        ],
        "must_contain": ["推进方是谁", "推进方实体质量", "对手方反击情况", "反击是否被跟进"],
        "reject_if_only": ["多头控制", "空头控制"],
        "placeholder": "请观察以下维度"
    },
    3: {
        "name": "推进质量",
        "dimensions": [
            {"key": "entity_change", "label": "实体大小变化", "placeholder": "连续K线的实体大小是放大还是缩小？"},
            {"key": "overlap", "label": "K线重叠程度", "placeholder": "K线之间是紧密重叠还是有跳空？重叠比例约为多少？"},
            {"key": "shadow", "label": "影线特征", "placeholder": "有无长上影线或长下影线？影线长度占波幅多少？"},
            {"key": "close", "label": "收盘位置", "placeholder": "收盘价在K线实体的高位、中位还是低位？"}
        ],
        "must_contain": ["实体大小变化", "K线重叠程度", "影线特征", "收盘位置"],
        "reject_if_only": ["推进很强", "推进很弱"],
        "placeholder": "请观察以下维度"
    },
    4: {
        "name": "回调vs转换",
        "dimensions": [
            {"key": "duration", "label": "回调/反向持续时间", "placeholder": "回调/反向运动持续了几根K线？"},
            {"key": "strength", "label": "回调K线实体强弱", "placeholder": "回调K线的实体是弱回调的小K线还是强反向的大K线？"},
            {"key": "structure", "label": "是否触及关键结构位", "placeholder": "是否触及或突破了关键结构位（前高/前低/密集区）？"},
            {"key": "reply", "label": "原方向方是否反击", "placeholder": "原方向方是否有立即的反击？反击K线特征如何？"}
        ],
        "must_contain": ["回调/反向持续时间", "回调K线实体强弱", "是否触及关键结构位", "原方向方是否反击"],
        "reject_if_only": ["这是回调", "这是转换"],
        "placeholder": "请观察以下维度"
    },
    5: {
        "name": "市场接受",
        "dimensions": [
            {"key": "stay", "label": "突破后停留时间", "placeholder": "突破后价格停留了几根K线？（1根/2根/3根以上）"},
            {"key": "reject", "label": "是否被推回", "placeholder": "是否被立即推回结构内？推回的幅度多大？"},
            {"key": "continuation", "label": "是否继续推进", "placeholder": "推回后是否继续向突破方向推进？"},
            {"key": "entity_quality", "label": "突破后K线实体特征", "placeholder": "突破后的K线实体特征如何？实体大小、成交量是否放大？"}
        ],
        "must_contain": ["突破后停留时间", "是否被推回", "是否继续推进", "突破后K线实体特征"],
        "reject_if_only": ["市场接受了", "市场没接受"],
        "placeholder": "请观察以下维度"
    }
}

# ==================== AI系统提示词模板（V21优化版）====================
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为训练教练。

【你的双重职责】
1. 分析市场 - 你和其他教练一样，能看到K线数据和结构位信息，对当前行情有自己的判断
2. 训练用户 - 通过提问和反馈，帮助用户提升观察能力

【核心原则】
你的判断是"参考答案"，不是"标准答案"。

【Al Brooks 核心解盘心法 - 必须作为你的点评依据】
1. 80%原则：市场有80%的时间试图突破震荡区间都会遭遇失败并反转；市场同样有80%的时间试图让一个强趋势反转都会失败并变成顺势回调。
2. 惯性原理（Inertia）：当前的K线行为大概率会延续前几根K线的惯性，直到关键结构位出现破坏性力量（强趋势K线群）。
3. 观察重叠度：K线之间高度重叠、带有长影线、阴阳交错，代表这是震荡区间（Trading Range），此时多空双方都没有控制权，任何单根K线的突破都不可信。
4. 顺势思维：在强通道或大趋势中，任何反向运动在第1轮和第2轮尝试中，都必须先视为"回调（Pullback）"而非"反转（Reversal）"。

当前用户正在训练：{skill_name}
训练阶段：{level_name}
{level_desc}

当前技能的核心问题：{skill_question}

【当前技能的语言约束 - 必须遵守】
{skill_constraints}

【当前技能的合格回答标准 - 必须严格执行】
{quality_standards}

用户回答必须包含上述 must_contain 中的所有维度。
如果用户只给出结论性判断（如 {reject_keywords}），视为无效回答。

【用户阅读画像 - 了解用户的薄弱点，用于针对性教学】
{reading_profile_text}

【个性化教学注入 - 必须严格执行】
请仔细阅读上述【用户阅读画像】。
- 如果用户历史中【喜欢提前预测】的分数较高，你在本轮提问中，要严厉质疑他是否有事实依据，逼迫他停留在对当下K线实体的客观描述上。
- 如果用户历史中【忽略背景阅读】的分数较高，在技能2、3的提问中，顺带问一句："你当前观察到的几根K线，处于大周期的什么结构位置？"
- 如果用户历史中【描述过于笼统】的分数较高，要求他给出具体的K线编号和量化数据（如"第X根K线实体占比XX%"）。

【训练流程 - 严格执行】

第1轮（用户首次作答）：
你的任务是：**只提问，不亮答案**

检查用户的回答：
- 是否包含了【合格回答标准】中 must_contain 的所有维度？
- 是否只给出了结论性判断（如"多头控制"、"空头控制"）？

如果回答**不到位**（缺少维度或只有结论）：
- 你只能提出**引导性问题**，帮助用户发现遗漏的维度
- **绝对禁止**引用任何具体K线编号（如K258、K259）
- **绝对禁止**描述任何具体K线特征（如"长下影"、"实体78%"、"吞没形态"）
- **绝对禁止**给出任何趋势判断或方向判断
- **根据用户画像的薄弱点，针对性提问**

允许的提问方式示例：
- "你是根据哪几根K线得出这个判断的？"
- "最近3-5根K线里，推进方有没有连续性？"
- "对手方是否有反击？反击之后发生了什么？"
- "你观察到收盘位置有什么特征吗？"

禁止的提问方式示例（泄露答案）：
- "你注意到K259的长下影了吗？"  ❌
- "K260的吞没形态说明什么？"  ❌

如果回答**到位**（覆盖了所有必须维度）：
- 只说："好的，我明白了"
- 然后直接进入第2轮流程

第2轮（用户二次作答）：
无论用户答得如何，执行以下两步：
1. 对用户的回答给出简短点评（肯定到位的部分，指出仍可补充的维度）
2. **亮出你自己的判断**（现在可以引用具体K线编号和结构位，说清楚你的观察依据）

【约束】
- 提示只给一次，第2轮必须亮出自己判断
- 第1轮绝对禁止引用具体K线特征
- 回答简短，不列大纲，不超过200字
- 禁止使用禁止词汇列表中的任何词语
"""

# ==================== 训练阶段定义 ====================
TRAIN_LEVEL = {
    1: {"name": "观察阶段", "desc": "允许模糊、整体感觉、通道、节奏、倾向。禁止结构辩论、精确确认、摆动定义。"},
    2: {"name": "行为细化阶段", "desc": "开始细化行为、具体K线、推进连续性。"},
    3: {"name": "结构验证阶段", "desc": "允许失败突破、摆动确认、Always In转换、结构争议。"},
}

# ==================== AI总结提示词 ====================
AI_SUMMARY_PROMPT = """你是训练总结分析师。根据训练对话记录，分析用户的阅读习惯和训练进展。
输出格式（JSON）：
{
    "observations": ["用户的阅读习惯和特点"],
    "strong_areas": ["用户表现好的方面"],
    "weak_areas": ["用户需要加强的方面"],
    "next_focus": ["下一阶段训练建议"],
    "profile_updates": {
        "忽略背景阅读": 0,
        "忽略控制权细节": 0,
        "喜欢提前预测": 0,
        "描述过于笼统": 0,
        "使用禁止词汇": 0
    }
}
profile_updates中的数值表示本次训练中该问题的出现次数（0-3），0表示未出现。
要求：每条分析具体，引用训练中的实际表现，不要笼统评价，要有可操作性。"""

# ==================== 数据加载函数 ====================
def load_data(symbol, period="30"):
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None
    if df is None or len(df) == 0:
        st.error(f"{symbol} 无数据")
        return None
    df = df.rename(columns={
        "date": "time", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
        "open_interest": "open_interest",
    })
    df = df.reset_index(drop=True)
    return df

# ==================== 图表构建函数 ====================
def build_chart(chart_df, bar):
    end = bar + 1
    start = max(0, end - 60)
    df = chart_df.iloc[start:end].copy().reset_index(drop=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        showlegend=False, increasing_line_color="red", decreasing_line_color="cyan",
    ), row=1, col=1)
    
    colors = ["red" if c >= o else "cyan" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors, showlegend=False, opacity=0.5), row=2, col=1)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        ny = row["low"] if row["close"] >= row["open"] else row["high"]
        fig.add_annotation(x=idx, y=ny, text=str(df.index[idx] + start),
                           showarrow=False, font=dict(size=7, color="#888888"),
                           yshift=-10 if row["close"] >= row["open"] else 10)
    
    fig.add_vline(x=bar-start, line_dash="dash", line_color="orange", line_width=1, opacity=0.6)
    fig.update_layout(xaxis_rangeslider_visible=False, height=400,
                      margin=dict(l=10, r=10, t=5, b=5),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

# ==================== 结构位识别函数 ====================
def identify_structures(all_bars, lookback=40):
    if len(all_bars) < lookback:
        lookback = len(all_bars)
    
    recent_bars = all_bars[-lookback:]
    highs = [b["h"] for b in recent_bars]
    lows = [b["l"] for b in recent_bars]
    highest = max(highs)
    lowest = min(lows)
    
    price_range = highest - lowest
    if price_range > 0:
        bucket_size = price_range / 20
        buckets = {}
        for bar in recent_bars:
            mid = (bar["h"] + bar["l"]) / 2
            bucket_idx = int((mid - lowest) / bucket_size)
            bucket_idx = min(bucket_idx, 19)
            buckets[bucket_idx] = buckets.get(bucket_idx, 0) + 1
        
        sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        dense_zones = []
        for idx, count in sorted_buckets[:3]:
            if count >= 3:
                zone_low = lowest + idx * bucket_size
                zone_high = lowest + (idx + 1) * bucket_size
                dense_zones.append({"low": zone_low, "high": zone_high, "touches": count})
    else:
        dense_zones = []
    
    recent_10 = all_bars[-10:] if len(all_bars) >= 10 else all_bars
    recent_high = max([b["h"] for b in recent_10])
    recent_low = min([b["l"] for b in recent_10])
    current_close = all_bars[-1]["c"]
    
    return {
        "highest_40": highest,
        "lowest_40": lowest,
        "recent_high_10": recent_high,
        "recent_low_10": recent_low,
        "dense_zones": dense_zones,
        "current_price": current_close,
        "price_range_40": price_range,
        "price_position_pct": (current_close - lowest) / price_range * 100 if price_range > 0 else 50,
    }

# ==================== 市场信息生成函数（V21优化版 - 加入Brooks原生术语标签）====================
def _market_msg(chart_df, bar, skill_name):
    last = st.session_state.get("_mm_cache", {})
    if last.get("bar") == bar and last.get("skill") == skill_name:
        return last["data"]
    
    start = max(0, bar - 40)
    all_bars = []
    
    for i in range(start, bar + 1):
        row = chart_df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        total_range = h - l
        body_ratio = body / total_range if total_range > 0 else 0
        volume = float(row["volume"]) if "volume" in chart_df.columns else 0
        
        if body > 0:
            if c >= o:
                close_position = (c - o) / body
            else:
                close_position = (o - c) / body
        else:
            close_position = 0.5
        
        if total_range > 0:
            upper_shadow_pct = (h - max(o, c)) / total_range * 100
            lower_shadow_pct = (min(o, c) - l) / total_range * 100
        else:
            upper_shadow_pct = 0
            lower_shadow_pct = 0
        
        all_bars.append({
            "i": i, "o": o, "h": h, "l": l, "c": c,
            "body_ratio": body_ratio, "close_position": close_position,
            "upper_shadow_pct": upper_shadow_pct,
            "lower_shadow_pct": lower_shadow_pct,
            "volume": volume, "total_range": total_range
        })
    
    recent_vols = [b["volume"] for b in all_bars[-20:]]
    avg_volume = sum(recent_vols) / len(recent_vols) if recent_vols else 1
    recent_ranges = [b["total_range"] for b in all_bars[-20:]]
    avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 1
    
    structures = identify_structures(all_bars, lookback=40)
    
    lines = []
    
    # 第一部分：结构位信息
    lines.append("【关键结构位 - 系统自动识别】")
    lines.append(f"  40根K线区间: {structures['lowest_40']:.0f} ~ {structures['highest_40']:.0f}")
    lines.append(f"  最近10根K线高点: {structures['recent_high_10']:.0f}")
    lines.append(f"  最近10根K线低点: {structures['recent_low_10']:.0f}")
    lines.append(f"  当前价格: {structures['current_price']:.0f}")
    lines.append(f"  当前在40根区间的位置: {structures['price_position_pct']:.0f}%")
    
    if structures["dense_zones"]:
        lines.append("  密集成交区:")
        for i, zone in enumerate(structures["dense_zones"][:2]):
            lines.append(f"    区域{i+1}: {zone['low']:.0f}~{zone['high']:.0f} (触及{zone['touches']}次)")
    
    lines.append("")
    lines.append("【K线语义化数据 - Al Brooks术语标签】")
    recent = all_bars[-15:] if len(all_bars) >= 15 else all_bars
    
    for idx, k in enumerate(recent):
        vol_ratio = k["volume"] / avg_volume if avg_volume > 0 else 1
        range_ratio = k["total_range"] / avg_range if avg_range > 0 else 1
        
        # ===== V21新增：Al Brooks原生术语定义 =====
        # 1. 趋势K线 vs 震荡K线
        if k["body_ratio"] >= 0.50:
            bar_type = "趋势阳线(TrendBar)" if k["c"] >= k["o"] else "趋势阴线(TrendBar)"
        else:
            bar_type = "震荡K线(TradingRange)"
        
        # 2. 收盘紧迫度 (Urgency)
        urgency = "普通"
        if k["c"] >= k["o"] and k["lower_shadow_pct"] < 10 and k["upper_shadow_pct"] < 10:
            urgency = "极强多头紧迫(买方完全控制)"
        elif k["c"] < k["o"] and k["upper_shadow_pct"] < 10 and k["lower_shadow_pct"] < 10:
            urgency = "极强空头紧迫(卖方完全控制)"
        elif k["close_position"] >= 0.85:
            urgency = "收在强高位"
        elif k["close_position"] <= 0.15:
            urgency = "收在强低位"
        
        # 3. 重叠度计算
        overlap_ratio = 0.0
        overlap_desc = ""
        if idx > 0:
            prev = recent[idx - 1]
            overlap = min(k["h"], prev["h"]) - max(k["l"], prev["l"])
            if overlap > 0 and k["total_range"] > 0:
                overlap_ratio = overlap / k["total_range"]
                if overlap_ratio >= 0.7:
                    overlap_desc = "高度重叠"
                elif overlap_ratio >= 0.4:
                    overlap_desc = "中度重叠"
                else:
                    overlap_desc = "低度重叠"
            elif overlap <= 0:
                gap = min(k["l"], prev["l"]) - max(k["h"], prev["h"])
                if gap > 0:
                    overlap_ratio = -gap / k["total_range"] if k["total_range"] > 0 else 0
                    overlap_desc = f"跳空{gap:.1f}点"
        
        # 组装语义化输出
        lines.append(
            f"  K{k['i']}: {bar_type} | 紧迫度={urgency} | "
            f"实体={k['body_ratio']*100:.0f}% | "
            f"波幅比={range_ratio:.2f} | 量比={vol_ratio:.2f} | "
            f"重叠={overlap_desc if overlap_desc else f'{overlap_ratio:.0%}'}"
        )
        
        # 附加原始数据供参考
        lines.append(
            f"      └─ 详: O={k['o']:.0f} H={k['h']:.0f} L={k['l']:.0f} C={k['c']:.0f} | "
            f"上影={k['upper_shadow_pct']:.0f}% 下影={k['lower_shadow_pct']:.0f}%"
        )
    
    result = "\n".join(lines)
    st.session_state["_mm_cache"] = {"bar": bar, "skill": skill_name, "data": result}
    return result

# ==================== AI调用函数 ====================
def _gpt(messages):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    if not api_key:
        st.error("API 密钥未配置！请在 Streamlit Cloud 后台设置 OPENAI_API_KEY")
        return "【配置提示】请先配置 DeepSeek API 密钥后再开始训练。"
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return f"【API 错误】{e}"

# ==================== 构建技能约束文本 ====================
def _build_skill_constraints_text(skill_id):
    sk = SKILL_CONSTRAINTS[skill_id]
    std = SKILL_QUALITY_STANDARDS[skill_id]
    forbidden = "、".join(sk["forbidden"])
    reject_keywords = "、".join(std["reject_if_only"])
    must_list = "\n  - ".join(std["must_contain"])
    
    return f"""{sk['desc']}

【必须包含的观察维度】（缺一不可）：
  - {must_list}

【结论性判断】（视为无效回答）：
  {reject_keywords}

允许使用的分析维度：{sk['allowed']}
绝对禁止使用的词汇：{forbidden}
如果用户的回答中缺少任何必须维度，在点评时必须指出。
如果用户只给出结论性判断，必须要求其补充具体观察。"""

# ==================== 构建用户阅读画像文本 ====================
def _build_reading_profile_text():
    rp = st.session_state.get("reading_profile", {})
    if not rp:
        return "暂无训练数据，请在训练过程中积累。"
    total = sum(rp.values())
    if total == 0:
        return "暂无训练数据。"
    items = []
    for key in sorted(rp, key=rp.get, reverse=True):
        pct = rp[key] / total * 100
        items.append(f"  - {key}: 出现{rp[key]}次 ({pct:.0f}%)")
    return "用户历史训练中暴露的薄弱点（出现次数越多越需要关注）：\n" + "\n".join(items)

# ==================== 教练提问函数 ====================
def ask_coach(chart_df, bar, skill_name, skill_id, dialogue, level=1, is_second_round=False):
    lv = TRAIN_LEVEL.get(level, TRAIN_LEVEL[1])
    constraints = _build_skill_constraints_text(skill_id)
    profile_text = _build_reading_profile_text()
    
    std = SKILL_QUALITY_STANDARDS[skill_id]
    reject_keywords = "、".join(std["reject_if_only"])
    
    sp = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill_name, 
        level_name=lv["name"], 
        level_desc=lv["desc"],
        skill_question=SKILL_CONSTRAINTS[skill_id]["question"],
        skill_constraints=constraints,
        quality_standards=f"必须包含维度: {', '.join(std['must_contain'])}",
        reject_keywords=reject_keywords,
        reading_profile_text=profile_text
    )
    sp += "\n\n" + _market_msg(chart_df, bar, skill_name)
    
    if is_second_round:
        sp += "\n\n【这是第2轮】你必须：1) 点评用户的回答  2) 亮出你自己的判断（引用具体K线编号和结构位）"
    
    msgs = [{"role": "system", "content": sp}]
    for m in dialogue[-10:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    
    return _gpt(msgs)

# ==================== 训练总结函数 ====================
def ask_summary(dialogue, observations, skill_id=None):
    if not dialogue and not observations:
        return json.dumps({"observations":["暂无训练数据"], "strong_areas":[], "weak_areas":[], "next_focus":[]})
    
    ot = "\n".join(f"- {o['text']}" for o in observations[-20:])
    dt = "\n".join("{}: {}".format("用户" if m["role"]=="user" else "教练", m["content"]) for m in dialogue[-40:])
    rp_text = "当前用户阅读画像：\n"
    rp = st.session_state.get("reading_profile", {})
    if rp:
        for k, v in rp.items():
            rp_text += f"  {k}: {v}\n"
    else:
        rp_text += "  暂无数据\n"
    
    return _gpt([{"role":"system","content":AI_SUMMARY_PROMPT},
                  {"role":"user","content":f"【观察】\n{ot}\n\n【对话】\n{dt}\n\n{rp_text}"}])

# ==================== 更新用户画像 ====================
def update_reading_profile(summary_json):
    try:
        data = json.loads(summary_json)
        updates = data.get("profile_updates", {})
        if not updates:
            return
        rp = st.session_state.get("reading_profile", {})
        for key, val in updates.items():
            if val > 0:
                rp[key] = rp.get(key, 0) + val
        st.session_state["reading_profile"] = rp
    except (json.JSONDecodeError, AttributeError):
        pass

# ==================== 技能列表 ====================
SKILLS = [
    {"id": 1, "name": "背景阅读"},
    {"id": 2, "name": "控制权识别"},
    {"id": 3, "name": "推进质量"},
    {"id": 4, "name": "回调vs转换"},
    {"id": 5, "name": "市场接受"},
]

# ==================== 数据加载辅助函数 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_all_contracts():
    def _fetch_one(ex):
        try:
            result = ak.match_main_contract(symbol=ex)
            return str(result).split(",")
        except Exception:
            return []
    mc = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, ex): ex for ex in ["shfe", "dce", "czce", "cffex", "gfex"]}
        for future in as_completed(futures):
            for c in future.result():
                c = c.strip()
                if len(c)<3: continue
                code = "".join(ch for ch in c[:4] if ch.isalpha()).upper()
                if code in SYMBOL_NAMES and code not in mc:
                    mc[code] = c
    return mc

def _random_bar(df, min_bar=60):
    return random.randint(min_bar, len(df) - 1)

def _do_load(sym_code, sym_main, period="30"):
    with st.spinner("加载中..."):
        df = load_data(sym_main, period=period)
        if df is not None:
            start_bar = _random_bar(df)
            st.session_state["chart_df"] = df
            st.session_state["current_bar"] = start_bar
            st.session_state["coach_dialogue"] = []
            st.session_state["observations"] = []
            st.session_state["training_summary"] = ""
            st.session_state["skill_round"] = 0
            st.session_state["send_counter"] = 0
            st.session_state["_mm_cache"] = {}
            st.session_state["symbol_code"] = sym_code
            st.session_state["symbol_main"] = sym_main
            st.session_state["symbol_name"] = SYMBOL_NAMES.get(sym_code, sym_code)
            st.session_state["data_period"] = period
            st.success(f"已加载 {SYMBOL_NAMES.get(sym_code, sym_code)} ({sym_main}) | {len(df)}根K线 | 周期{period}分钟")
            time.sleep(0.3)
            st.rerun()

def next_bar_callback():
    if st.session_state.get("chart_df") is not None:
        df = st.session_state["chart_df"]
        current_bar = st.session_state.get("current_bar", 0)
        if current_bar < len(df) - 1:
            st.session_state["current_bar"] = current_bar + 1
            st.session_state["_mm_cache"] = {}
            st.session_state["skill_round"] = 0
            st.session_state["coach_dialogue"] = []

# ==================== 发送用户回答的处理函数（V21优化版）====================
def _send(text, chart_df, bar, skill):
    s = st.session_state
    dlg = s["coach_dialogue"]
    dlg.append({"role": "user", "content": text})
    s["observations"].append({
        "skill_id": skill["id"], "bar": bar, "text": text,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    with st.spinner("教练思考中..."):
        resp = ask_coach(
            chart_df, bar, skill["name"], skill["id"], dlg,
            level=s.get("train_level", 1), is_second_round=(s["skill_round"] >= 1)
        )
    s["skill_round"] += 1
    if s["skill_round"] >= 2:
        resp += "\n\n---\n本项技能训练结束，可切换下一项继续。"
        summary = ask_summary(dlg, s["observations"], skill_id=skill["id"])
        update_reading_profile(summary)
        s["training_summary"] = summary
    dlg.append({"role": "assistant", "content": resp})
    s["coach_dialogue"] = dlg
    s["send_counter"] = s.get("send_counter", 0) + 1
    st.rerun()

# ==================== 结构化输入组件（V21新增核心功能）====================
def render_structured_input(skill_id, current_round):
    """
    根据当前技能，渲染结构化输入表单（分段式填空）
    """
    std = SKILL_QUALITY_STANDARDS[skill_id]
    dimensions = std["dimensions"]
    
    st.markdown("---")
    st.markdown(f"✍️ **请根据以下维度观察图表（第 {current_round+1}/2 轮）：**")
    st.caption("💡 填写每个维度的观察，系统会自动组合成完整分析发送给教练")
    
    # 动态构建表单
    with st.form(key=f"skill_form_{skill_id}_{current_round}"):
        user_inputs = {}
        cols = st.columns(1)
        
        for idx, dim in enumerate(dimensions):
            # 为每个维度生成带说明的输入框
            label = f"**{idx+1}. {dim['label']}**"
            placeholder = dim["placeholder"]
            val = st.text_input(
                label=label,
                placeholder=placeholder,
                key=f"structured_input_{skill_id}_{idx}_{current_round}",
                label_visibility="visible"
            )
            user_inputs[dim["key"]] = val
        
        submit_button = st.form_submit_button(label="📤 提交本次观察", use_container_width=True, type="primary")
        
        if submit_button:
            # 检查是否有空格未填
            empty_fields = [dim["label"] for dim in dimensions if not user_inputs.get(dim["key"], "").strip()]
            if empty_fields:
                st.warning(f"⚠️ 请填写以下维度: {', '.join(empty_fields)}")
                return None
            else:
                # 潜移默化：自动拼接成一段逻辑严密的PA分析文本
                combined_prompt = "我的观察如下：\n"
                for idx, dim in enumerate(dimensions):
                    combined_prompt += f"{idx+1}. 【{dim['label']}】: {user_inputs[dim['key']]}\n"
                return combined_prompt
    
    return None

# ==================== 主界面函数 ====================
def main():
    st.set_page_config(page_title="Al Brooks 结构训练器 V21", layout="wide")
    
    # CSS样式
    st.markdown("""
    <style>
     section[data-testid="stSidebar"] { width: 280px !important; min-width: 240px !important; max-width: 320px !important; }
     section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
     .stMarkdown, .stCaption { margin-bottom: 0.2rem !important; }
     div[data-testid="stExpander"] { margin-bottom: 0.2rem !important; }
     div[data-testid="stExpander"] details { border: none !important; }
     div[data-testid="stExpander"] summary { padding: 0.2rem 0.5rem !important; font-size: 0.85rem !important; }
     div[data-testid="stExpander"] div[data-testid="element-container"] { padding: 0 !important; }
     .stButton button { font-size: 0.75rem !important; padding: 0.15rem 0.3rem !important; min-height: 0 !important; line-height: 1.4 !important; }
     div[data-testid="stSelectbox"] { margin-bottom: 0.2rem !important; }
     div[data-testid="stSelectbox"] > div { min-height: 1.8rem !important; }
     hr { margin: 0.4rem 0 !important; }
     div[data-testid="stInfo"] { padding: 0.3rem !important; font-size: 0.8rem !important; }
     div[data-testid="stSuccess"] { padding: 0.3rem !important; font-size: 0.8rem !important; }
     .stCaption { font-size: 0.75rem !important; }
     /* 结构化输入框样式 */
     div[data-testid="stForm"] label { font-size: 0.85rem !important; font-weight: 500 !important; }
     div[data-testid="stForm"] input { font-size: 0.85rem !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化session_state变量
    for k, v in {
        "chart_df": None,
        "current_bar": 40,
        "coach_dialogue": [],
        "send_counter": 0,
        "training_summary": "",
        "skill_round": 0,
        "train_level": 1,
        "observations": [],
        "symbol_code": "",
        "symbol_main": "",
        "symbol_name": "",
        "_mm_cache": {},
        "reading_profile": {},
        "data_period": "30",
        "train_mode": 1,  # 当前技能ID
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.markdown("**品种选择**")
        
        period_map = {"15分钟": "15", "30分钟": "30", "60分钟": "60", "日线": "day"}
        period_label = st.selectbox(
            "周期",
            options=list(period_map.keys()),
            index=1,
            key="period_selector",
            label_visibility="collapsed"
        )
        selected_period = period_map[period_label]

        with st.expander("数据源信息", expanded=False):
            st.caption(
                f"来源: 新浪财经(akshare)<br>"
                f"周期: {st.session_state['data_period']}分钟<br>"
                f"合约: 主力连续"
            )

        if st.session_state.get("chart_df") is not None:
            df = st.session_state["chart_df"]
            st.caption(f"K线数: {len(df)}根")

        # 品种分类按钮
        exchanges = {
            "金融": ["IF","IH","IC","IM","TS","TF","T","TL"],
            "有色": ["CU","AL","ZN","PB","NI","SN","AU","AG","BC"],
            "黑色": ["RB","HC","SS","I","J","JM"],
            "化工": ["MA","TA","PP","L","V","EG","EB","PG","SA","UR","SF","SM","PF"],
            "农产品": ["A","B","M","Y","P","C","CS","JD","CF","SR","RM","OI","FG","AP","CJ","PK","LH"],
            "能源": ["SC","FU","BU","LU","NR","RU"],
        }
        
        for cat, codes in exchanges.items():
            with st.expander(cat, expanded=(cat == "金融")):
                cols = st.columns(4)
                for idx, code in enumerate(codes):
                    if cols[idx % 4].button(code, key=f"sym_{code}", use_container_width=True):
                        _do_load(code, f"{code}0", period=selected_period)

        # 操作按钮组
        if st.session_state.get("chart_df") is not None:
            df = st.session_state["chart_df"]
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("🎲 随机", use_container_width=True):
                    new_bar = _random_bar(df)
                    st.session_state["current_bar"] = new_bar
                    st.session_state["_mm_cache"] = {}
                    st.session_state["skill_round"] = 0
                    st.session_state["coach_dialogue"] = []
                    st.rerun()
                    
            with col_b:
                sym_code = st.session_state.get("symbol_code", "")
                if sym_code:
                    if st.button("🔄 重载", use_container_width=True):
                        _do_load(sym_code, f"{sym_code}0", period=selected_period)
            
            with col_c:
                if st.button("⏭️ Next", use_container_width=True):
                    current = st.session_state["current_bar"]
                    max_bar = len(df) - 1
                    if current < max_bar:
                        st.session_state["current_bar"] = current + 1
                        st.session_state["_mm_cache"] = {}
                        st.session_state["skill_round"] = 0
                        st.session_state["coach_dialogue"] = []
                        st.rerun()
                    else:
                        st.toast("已是最后一根K线", icon="ℹ️")

        # 训练阶段选择
        level = st.selectbox(
            "阶段",
            options=[1, 2, 3],
            format_func=lambda x: f"阶段{x}: {TRAIN_LEVEL[x]['name']}",
            index=st.session_state.get("train_level", 1) - 1,
            key="train_level_sel",
            label_visibility="collapsed"
        )
        st.session_state["train_level"] = level

        # K线位置滑块
        if st.session_state.get("chart_df") is not None:
            df = st.session_state["chart_df"]
            bar = st.slider(
                "K线", 
                41, 
                len(df) - 1, 
                value=st.session_state["current_bar"], 
                key="bar_slider", 
                label_visibility="collapsed"
            )
            if bar != st.session_state["current_bar"]:
                st.session_state["current_bar"] = bar
                st.session_state["_mm_cache"] = {}
                st.session_state["skill_round"] = 0
                st.session_state["coach_dialogue"] = []
                st.rerun()

        # 阅读画像显示
        rp = st.session_state.get("reading_profile", {})
        if rp:
            st.markdown("**阅读画像**")
            total = sum(rp.values())
            for key in sorted(rp, key=rp.get, reverse=True):
                pct = rp[key] / total * 100 if total > 0 else 0
                bw = max(pct, 4)
                st.markdown(
                    f"<div style='margin-bottom:3px;font-size:11px;'>"
                    f"<span style='font-size:11px;'>{key}</span>"
                    f"<div style='background:#f0f0f0;border-radius:3px;height:12px;'>"
                    f"<div style='background:#e74c3c;width:{bw}%;height:12px;border-radius:3px;text-align:right;"
                    f"font-size:9px;color:white;padding-right:3px;line-height:12px;'>{rp[key]}</div></div></div>",
                    unsafe_allow_html=True
                )
        
        if st.button("重置画像", use_container_width=True):
            st.session_state["reading_profile"] = {}
            st.rerun()

    # ==================== 主界面 ====================
    df = st.session_state.get("chart_df")
    if df is None:
        st.info("请从左侧选择品种开始训练")
        return

    bar = st.session_state["current_bar"]
    
    # 技能选择按钮组
    with st.container():
        current_skill_id = st.session_state.get("train_mode", 1)
        cols = st.columns(5)
        for idx, sk in enumerate(SKILLS):
            is_active = (sk["id"] == current_skill_id)
            if cols[idx].button(
                sk["name"], 
                type="primary" if is_active else "secondary", 
                use_container_width=True, 
                key=f"skill_{sk['id']}"
            ):
                st.session_state["train_mode"] = sk["id"]
                st.session_state["coach_dialogue"] = []
                st.session_state["skill_round"] = 0
                st.session_state["send_counter"] = 0
                st.rerun()
        
        active_skill = next(sk for sk in SKILLS if sk["id"] == current_skill_id)
        st.caption(
            f"当前技能: {active_skill['name']} | "
            f"阶段: {TRAIN_LEVEL[st.session_state['train_level']]['name']} | "
            f"第{st.session_state['skill_round']+1}/2 轮"
        )

    # K线图表显示
    with st.container():
        st.plotly_chart(build_chart(df, bar), use_container_width=True)

    # 对话界面
    with st.container():
        st.markdown("### 教练")
        
        # 显示历史对话
        for m in st.session_state["coach_dialogue"][-10:]:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.markdown(f"**{'🧑 你' if m['role']=='user' else '🤖 教练'}**")
                st.markdown(m["content"])

    # ==================== V21核心优化：结构化输入 ====================
    s = st.session_state
    can_input = s.get("skill_round", 0) < 2 and s.get("chart_df") is not None

    if can_input:
        current_skill_id = s.get("train_mode", 1)
        
        # 使用结构化输入表单
        combined_prompt = render_structured_input(current_skill_id, s["skill_round"])
        
        if combined_prompt:
            _send(combined_prompt, df, active_skill, active_skill)
    else:
        if s.get("skill_round", 0) >= 2:
            st.info("✅ 本项技能训练结束，点击上方技能按钮切换下一项继续训练。")

if __name__ == "__main__":
    main()
