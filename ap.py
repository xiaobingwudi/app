"""
Al Brooks 结构训练器 V22 - 终极体验版

核心改进（基于深度点评）：
1. 结构化智能表单：下拉选项 + 快捷词，降低认知负载
2. 修复AI数据流：确保语义化标签真正喂给大模型
3. 铁血判分机制：AI必须依据客观数据严格审核
4. 画像实时喂回 + 针对性刁难
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
        "forbidden": ["买", "卖", "做多", "做空", "进场", "止损", "目标位", "开仓", "平仓", "做单", "看涨", "看跌", "预测"],
        "allowed": "趋势方向、高低点序列(HH/HL或LH/LL)、通道斜率、震荡区间边界、结构位",
        "desc": "只允许描述市场结构，禁止任何交易决策类词汇。"
    },
    2: {
        "name": "控制权识别",
        "question": "现在谁在控制市场？",
        "forbidden": ["趋势", "方向", "预测", "目标位", "进场", "止损", "做多", "做空", "看涨", "看跌"],
        "allowed": "最近3-5根K线谁在主导、推进方实体质量、对手方有无有效反击",
        "desc": "只关注最近几根K线的力量对比，禁止谈论大趋势和预测。"
    },
    3: {
        "name": "推进质量",
        "question": "最近推进的质量如何？",
        "forbidden": ["趋势", "方向", "多空", "预测", "进场", "止损", "目标", "看涨", "看跌"],
        "allowed": "K线实体大小、重叠程度、影线长度、收盘位置、动能变化",
        "desc": "只描述K线本身的质量特征，禁止判断方向。"
    },
    4: {
        "name": "回调vs转换",
        "question": "这是正常回调还是控制权转换？",
        "forbidden": ["预测", "目标位", "进场", "止损", "开仓", "看涨", "看跌"],
        "allowed": "回调K线数量、回调实体强弱、对手方连续性、有无跟进",
        "desc": "区分回调与转换，禁止谈论后续走势预测。"
    },
    5: {
        "name": "市场接受",
        "question": "市场是否接受了新价格？",
        "forbidden": ["进场", "止损", "目标位", "预测", "做多", "做空", "看涨", "看跌"],
        "allowed": "突破后停留几根、有无推回、有无继续推进、成交量确认",
        "desc": "只关注价格突破后的市场反应，禁止交易建议。"
    }
}

# ==================== 技能表单定义（V22核心：结构化下拉选项）====================
SKILL_FORM_DEFINITIONS = {
    1: {  # 背景阅读
        "fields": [
            {
                "label": "高低点序列",
                "type": "select",
                "options": ["请选择...", "持续抬高的多头序列 (HH/HL)", "持续降低的空头序列 (LH/LL)", "没有明显序列（横盘震荡）", "序列正在发生转换"],
                "placeholder": "观察最近10-15根K线的高点和低点序列"
            },
            {
                "label": "区间/通道边界",
                "type": "text",
                "placeholder": "当前结构的高点区间和低点区间分别在哪个价格区域？(例: 高点3950-3960, 低点3900-3910)"
            },
            {
                "label": "当前价格位置",
                "type": "select",
                "options": ["请选择...", "接近结构上沿 (阻力区)", "接近结构下沿 (支撑区)", "处于结构中间位置", "正在突破关键边界"],
                "placeholder": "当前价格在结构中的相对位置"
            }
        ]
    },
    2: {  # 控制权识别
        "fields": [
            {
                "label": "最近3-5根K线推进方",
                "type": "select",
                "options": ["请选择...", "买方(多头)在主动推进", "卖方(空头)在主动推进", "双方交替控制(无明确推进方)", "推进方正在衰竭"],
                "placeholder": "谁在主导最近几根K线的运动方向"
            },
            {
                "label": "推进方实体质量",
                "type": "select",
                "options": ["请选择...", "强实体(占波幅>60%)", "中等实体(占波幅30-60%)", "弱实体(占波幅<30%)", "十字星/Doji(无方向)"],
                "placeholder": "推进方的K线实体强度"
            },
            {
                "label": "对手方反击情况",
                "type": "select",
                "options": ["请选择...", "无有效反击", "有弱反击(小实体/长影线)", "有强反击(大实体反包)", "反击后立即被反制"],
                "placeholder": "对手方是否有反击动作"
            },
            {
                "label": "反击是否被跟进",
                "type": "select",
                "options": ["请选择...", "反击得到后续跟进", "反击被立即反包", "反击后市场陷入僵持"],
                "placeholder": "反击发生后后续K线的反应"
            }
        ]
    },
    3: {  # 推进质量
        "fields": [
            {
                "label": "实体大小变化",
                "type": "select",
                "options": ["请选择...", "实体持续放大(加速)", "实体持续缩小(减速)", "实体大小交替(不稳定)", "实体突然放大(爆发)"],
                "placeholder": "连续K线的实体大小变化趋势"
            },
            {
                "label": "K线重叠程度",
                "type": "select",
                "options": ["请选择...", "高度重叠(>70%)", "中度重叠(40-70%)", "低度重叠(10-40%)", "跳空(<10%或负值)"],
                "placeholder": "K线之间的重叠比例"
            },
            {
                "label": "影线特征",
                "type": "select",
                "options": ["请选择...", "无明显影线(光头光脚)", "有长上影线(抛压)", "有长下影线(买盘支撑)", "上下影线都很长(多空激烈)"],
                "placeholder": "影线长度和位置特征"
            },
            {
                "label": "收盘位置",
                "type": "select",
                "options": ["请选择...", "收在实体高位(>85%)", "收在实体中位", "收在实体低位(<15%)", "收在实体极值位置"],
                "placeholder": "收盘价在K线实体中的位置"
            }
        ]
    },
    4: {  # 回调vs转换
        "fields": [
            {
                "label": "回调/反向持续时间",
                "type": "select",
                "options": ["请选择...", "1-2根K线(短暂回调)", "3-5根K线(正常回调)", "6根以上(深度回调)", "持续反向运动"],
                "placeholder": "回调或反向运动持续了几根K线"
            },
            {
                "label": "回调K线实体强弱",
                "type": "select",
                "options": ["请选择...", "弱回调(小实体/十字星)", "中等回调", "强反向(大实体趋势K线)", "回调K线呈现衰竭特征"],
                "placeholder": "回调K线的实体强度"
            },
            {
                "label": "是否触及关键结构位",
                "type": "select",
                "options": ["请选择...", "未触及关键结构位", "触及但未突破", "已突破关键结构位", "正在结构位附近挣扎"],
                "placeholder": "是否触及前高/前低/密集区"
            },
            {
                "label": "原方向方是否反击",
                "type": "select",
                "options": ["请选择...", "立即强反击", "弱反击(无效)", "无反击动作", "反击后失败"],
                "placeholder": "原主导方是否有反击动作"
            }
        ]
    },
    5: {  # 市场接受
        "fields": [
            {
                "label": "突破后停留时间",
                "type": "select",
                "options": ["请选择...", "1根K线(快速测试)", "2-3根K线(正常停留)", "3根以上(有效停留)", "未形成有效突破"],
                "placeholder": "突破后在结构外停留了几根K线"
            },
            {
                "label": "是否被推回",
                "type": "select",
                "options": ["请选择...", "未被推回(有效突破)", "部分推回(回踩确认)", "完全推回(假突破)", "推回后立即反弹"],
                "placeholder": "价格是否被推回原结构内"
            },
            {
                "label": "是否继续推进",
                "type": "select",
                "options": ["请选择...", "持续向突破方向推进", "推进停滞/横盘", "反向运动开始", "无法判断(需更多K线)"],
                "placeholder": "突破后是否继续向同方向推进"
            },
            {
                "label": "突破后K线实体特征",
                "type": "select",
                "options": ["请选择...", "大实体跟进(确认)", "小实体/十字星(犹豫)", "反向吞没(失败)", "成交量显著放大"],
                "placeholder": "突破后K线的实体和成交量特征"
            }
        ]
    }
}

# ==================== 训练阶段定义 ====================
TRAIN_LEVEL = {
    1: {"name": "观察阶段", "desc": "允许模糊、整体感觉、通道、节奏、倾向。"},
    2: {"name": "行为细化阶段", "desc": "开始细化行为、具体K线、推进连续性。"},
    3: {"name": "结构验证阶段", "desc": "允许失败突破、摆动确认、Always In转换、结构争议。"},
}

# ==================== AI系统提示词模板（V22铁血版）====================
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为训练教练，铁血、严厉、绝不妥协。

【你的双重职责】
1. 分析市场 - 基于注入的【K线客观统计数据】作为唯一真理
2. 训练用户 - 严格审核用户观察，指出错误，逼迫用户掌握正确方法

【核心原则 - 铁血判分机制】
1. 你收到的【K线客观统计数据】是算法严格计算的，这是最高真理。用户的陈述必须与此数据完全吻合。
2. 不要为了礼貌而夸奖用户。如果用户描述与客观数据不符，或包含禁止词汇，直接拒绝让其通过。
3. 绝对禁止给出任何交易建议（做多/做空/进场/止损等）。

【Al Brooks 核心解盘心法】
1. 80%原则：80%的突破尝试会失败并反转；80%的趋势反转尝试会失败并变成顺势回调。
2. 惯性原理：当前K线行为大概率延续前几根K线的惯性。
3. 重叠度判断：K线高度重叠代表震荡区间，此时任何单根K线的突破都不可信。

当前用户正在训练：{skill_name}
训练阶段：{level_name}

当前技能的核心问题：{skill_question}

【语言约束 - 违反直接不合格】
绝对禁止词汇：{forbidden_words}
用户回答中若出现上述任何词汇，直接判定本轮不合格，要求重新作答。

【第1轮：严审期】
1. 对照【K线客观统计数据】，逐一核对用户观察。
2. 检查用户选择是否与数据矛盾（如数据显示是震荡K线，用户却选"强实体"）。
3. 如不合格：明确指出错误，引用客观数据，要求重新观察。
4. 如合格：用Brooks风格反问一个深层问题（如："你注意到影线变长了吗？"），引导第2轮。

【第2轮：亮答案】
1. 对用户回答简短点评
2. 亮出你自己基于客观数据的完整判断（可引用具体K线编号）

【用户历史薄弱点 - 针对性训练】
{reading_profile}
如果用户薄弱点包含"喜欢提前预测"，在第一轮必须严厉质疑。
如果包含"忽略背景阅读"，要求补充背景观察。

回答简短，不超过200字。
"""

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

# ==================== 核心函数：生成K线语义化标签（V22优化版）====================
def generate_semantic_kline_labels(chart_df, bar):
    """生成K线的Brooks原生术语标签，供AI使用"""
    start = max(0, bar - 40)
    all_bars = []
    
    for i in range(start, bar + 1):
        row = chart_df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        total_range = h - l
        body_ratio = body / total_range if total_range > 0 else 0
        volume = float(row["volume"]) if "volume" in chart_df.columns else 0
        
        if total_range > 0:
            upper_shadow_pct = (h - max(o, c)) / total_range * 100
            lower_shadow_pct = (min(o, c) - l) / total_range * 100
        else:
            upper_shadow_pct = 0
            lower_shadow_pct = 0
        
        all_bars.append({
            "i": i, "o": o, "h": h, "l": l, "c": c,
            "body_ratio": body_ratio,
            "upper_shadow_pct": upper_shadow_pct,
            "lower_shadow_pct": lower_shadow_pct,
            "volume": volume, "total_range": total_range,
            "is_bullish": c >= o
        })
    
    recent_vols = [b["volume"] for b in all_bars[-20:]]
    avg_volume = sum(recent_vols) / len(recent_vols) if recent_vols else 1
    recent_ranges = [b["total_range"] for b in all_bars[-20:]]
    avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 1
    
    structures = identify_structures(all_bars, lookback=40)
    
    lines = []
    lines.append("=" * 50)
    lines.append("【K线客观统计数据 - Al Brooks语义化标签】")
    lines.append("=" * 50)
    
    # 结构位信息
    lines.append(f"40根区间: {structures['lowest_40']:.0f} ~ {structures['highest_40']:.0f}")
    lines.append(f"当前价格位置: {structures['price_position_pct']:.0f}% (0%=最低,100%=最高)")
    if structures["dense_zones"]:
        lines.append(f"密集区: {structures['dense_zones'][0]['low']:.0f}~{structures['dense_zones'][0]['high']:.0f}")
    
    lines.append("")
    lines.append("最近15根K线详细数据:")
    lines.append("")
    
    recent = all_bars[-15:] if len(all_bars) >= 15 else all_bars
    
    for idx, k in enumerate(recent):
        vol_ratio = k["volume"] / avg_volume if avg_volume > 0 else 1
        range_ratio = k["total_range"] / avg_range if avg_range > 0 else 1
        
        # Brooks原生术语判断
        if k["body_ratio"] >= 0.50:
            bar_type = "趋势K线(TrendBar)" 
        else:
            bar_type = "震荡K线(TradingRangeBar)"
        
        # 紧迫度
        if k["body_ratio"] >= 0.70 and k["upper_shadow_pct"] < 10 and k["lower_shadow_pct"] < 10:
            urgency = "极强紧迫"
        elif k["body_ratio"] >= 0.50:
            urgency = "正常推进"
        elif k["body_ratio"] >= 0.30:
            urgency = "弱推进"
        else:
            urgency = "无方向"
        
        # 重叠度
        overlap_desc = "N/A"
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
                overlap_desc = "跳空"
        
        direction = "阳线" if k["is_bullish"] else "阴线"
        
        lines.append(
            f"K{k['i']}: {direction} {bar_type} | 紧迫度={urgency} | "
            f"实体占比={k['body_ratio']*100:.0f}% | 重叠={overlap_desc} | "
            f"上影={k['upper_shadow_pct']:.0f}% 下影={k['lower_shadow_pct']:.0f}% | "
            f"量比={vol_ratio:.2f} | 波幅比={range_ratio:.2f}"
        )
    
    return "\n".join(lines)

# ==================== AI调用函数 ====================
def call_ai(system_prompt, user_content):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【配置提示】请先配置 DeepSeek API 密钥。"
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ], 
            temperature=0.2, 
            max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【API错误】{e}"

# ==================== 构建用户画像文本 ====================
def build_profile_text():
    rp = st.session_state.get("reading_profile", {})
    if not rp:
        return "暂无训练数据"
    total = sum(rp.values())
    if total == 0:
        return "暂无训练数据"
    items = [f"{k}: {v}次" for k, v in sorted(rp.items(), key=lambda x: x[1], reverse=True)]
    return "用户历史薄弱点: " + " | ".join(items)

# ==================== 发送回答并获取AI反馈 ====================
def send_and_get_coach_response(user_observations, df, bar, skill):
    """核心函数：将用户观察+语义化标签一起发给AI"""
    
    # 1. 生成语义化标签（这是AI判断的客观依据）
    semantic_labels = generate_semantic_kline_labels(df, bar)
    
    # 2. 构建画像
    profile = build_profile_text()
    
    # 3. 获取技能约束
    constraints = SKILL_CONSTRAINTS[skill["id"]]
    forbidden_words = "、".join(constraints["forbidden"])
    
    # 4. 构建系统提示词
    level = TRAIN_LEVEL.get(st.session_state.get("train_level", 1), TRAIN_LEVEL[1])
    system_prompt = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill["name"],
        level_name=level["name"],
        skill_question=constraints["question"],
        forbidden_words=forbidden_words,
        reading_profile=profile
    )
    
    # 5. 构建用户消息（包含客观数据 + 用户观察）
    user_content = f"""【K线客观统计数据 - 请以此为准审核用户】
{semantic_labels}

【用户的观察陈述】
{user_observations}

请按照你的职责审核并回复。"""
    
    # 6. 调用AI
    with st.spinner("AI教练正在审核你的观察..."):
        response = call_ai(system_prompt, user_content)
    
    return response

# ==================== 渲染结构化表单（V22核心）====================
def render_skill_form(skill_id, current_round):
    """渲染技能的结构化输入表单"""
    form_def = SKILL_FORM_DEFINITIONS.get(skill_id)
    if not form_def:
        return None
    
    st.markdown("---")
    st.markdown(f"✍️ **第{current_round+1}/2轮 - 请完成以下观察（点击提交后将由AI审核）**")
    st.caption("💡 每个选项都来自Al Brooks价格行为学的核心概念，选择最符合当前图表的描述")
    
    user_inputs = {}
    
    for idx, field in enumerate(form_def["fields"]):
        if field["type"] == "select":
            user_inputs[field["label"]] = st.selectbox(
                label=f"**{field['label']}**",
                options=field["options"],
                key=f"select_{skill_id}_{idx}_{current_round}",
                help=field.get("placeholder", "")
            )
        else:  # text type
            user_inputs[field["label"]] = st.text_input(
                label=f"**{field['label']}**",
                placeholder=field["placeholder"],
                key=f"text_{skill_id}_{idx}_{current_round}"
            )
    
    # 提交按钮
    if st.button("📤 提交观察", type="primary", use_container_width=True, key=f"submit_{skill_id}_{current_round}"):
        # 验证
        has_empty = False
        for label, value in user_inputs.items():
            if not value or value == "请选择...":
                has_empty = True
                st.error(f"⚠️ 请完成「{label}」的选择/填写")
                break
        
        if not has_empty:
            # 组装用户观察报告
            report = "【我的观察】\n"
            for label, value in user_inputs.items():
                report += f"- {label}: {value}\n"
            return report
    
    return None

# ==================== 更新画像 ====================
def update_profile_from_response(response, current_skill_id):
    """根据AI回复中的错误信息更新用户画像"""
    rp = st.session_state.get("reading_profile", {})
    
    # 检查是否包含禁止词汇的提示
    if "禁止词汇" in response or "不合格" in response:
        if "使用禁止词汇" not in rp:
            rp["使用禁止词汇"] = 0
        rp["使用禁止词汇"] += 1
    
    # 检查是否提示背景阅读问题
    if "背景" in response and ("忽略" in response or "遗漏" in response):
        if "忽略背景阅读" not in rp:
            rp["忽略背景阅读"] = 0
        rp["忽略背景阅读"] += 1
    
    # 检查是否提示预测问题
    if "预测" in response or ("提前" in response and ("判断" in response or "结论" in response)):
        if "喜欢提前预测" not in rp:
            rp["喜欢提前预测"] = 0
        rp["喜欢提前预测"] += 1
    
    st.session_state["reading_profile"] = rp

# ==================== 数据加载辅助 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_contracts():
    mc = {}
    for ex in ["shfe", "dce", "czce", "cffex"]:
        try:
            result = ak.match_main_contract(symbol=ex)
            codes = str(result).split(",")
            for c in codes:
                c = c.strip()
                if len(c) < 3:
                    continue
                code = "".join(ch for ch in c[:4] if ch.isalpha()).upper()
                if code in SYMBOL_NAMES and code not in mc:
                    mc[code] = c
        except:
            pass
    return mc

def random_bar(df, min_bar=60):
    return random.randint(min_bar, len(df) - 1)

def load_symbol(sym_code, sym_main, period="30"):
    with st.spinner("加载数据中..."):
        df = load_data(sym_main, period=period)
        if df is not None:
            st.session_state["chart_df"] = df
            st.session_state["current_bar"] = random_bar(df)
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.session_state["symbol_code"] = sym_code
            st.session_state["symbol_main"] = sym_main
            st.session_state["symbol_name"] = SYMBOL_NAMES.get(sym_code, sym_code)
            st.session_state["data_period"] = period
            st.success(f"已加载 {st.session_state['symbol_name']} | {len(df)}根K线")
            time.sleep(0.3)
            st.rerun()

# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 结构训练器 V22", layout="wide")
    
    # 初始化session_state
    defaults = {
        "chart_df": None, "current_bar": 40, "coach_dialogue": [],
        "skill_round": 0, "train_level": 1, "observations": [],
        "symbol_code": "", "symbol_main": "", "symbol_name": "",
        "reading_profile": {}, "data_period": "30", "train_mode": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 品种选择")
        
        period_map = {"15分钟": "15", "30分钟": "30", "60分钟": "60", "日线": "day"}
        period_label = st.selectbox("周期", list(period_map.keys()), index=1)
        selected_period = period_map[period_label]
        
        exchanges = {
            "金融": ["IF", "IH", "IC", "IM"],
            "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
            "黑色": ["RB", "HC", "I", "J", "JM"],
            "化工": ["MA", "TA", "PP", "L", "V", "SA", "UR"],
            "农产品": ["A", "B", "M", "Y", "P", "C", "CS", "JD", "CF", "SR", "RM", "OI", "FG", "AP"],
        }
        
        for cat, codes in exchanges.items():
            with st.expander(cat, expanded=(cat == "金融")):
                cols = st.columns(4)
                for idx, code in enumerate(codes):
                    if cols[idx % 4].button(code, key=f"sym_{code}", use_container_width=True):
                        load_symbol(code, f"{code}0", period=selected_period)
        
        if st.session_state.get("chart_df") is not None:
            df = st.session_state["chart_df"]
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🎲 随机K线", use_container_width=True):
                    st.session_state["current_bar"] = random_bar(df)
                    st.session_state["coach_dialogue"] = []
                    st.session_state["skill_round"] = 0
                    st.rerun()
            with col_b:
                if st.button("⏭️ 下一根", use_container_width=True):
                    if st.session_state["current_bar"] < len(df) - 1:
                        st.session_state["current_bar"] += 1
                        st.session_state["coach_dialogue"] = []
                        st.session_state["skill_round"] = 0
                        st.rerun()
        
        st.divider()
        
        st.selectbox("训练阶段", [1, 2, 3], format_func=lambda x: f"阶段{x}: {TRAIN_LEVEL[x]['name']}", 
                     index=st.session_state["train_level"]-1, key="train_level")
        
        if st.session_state.get("reading_profile"):
            st.markdown("### 📈 阅读画像")
            for k, v in st.session_state["reading_profile"].items():
                st.progress(min(v/10, 1.0), text=f"{k}: {v}次")
            if st.button("重置画像"):
                st.session_state["reading_profile"] = {}
                st.rerun()
    
    # 主区域
    df = st.session_state.get("chart_df")
    if df is None:
        st.info("👈 请从左侧选择品种开始训练")
        return
    
    bar = st.session_state["current_bar"]
    
    # 技能选择
    skills = [
        {"id": 1, "name": "背景阅读"},
        {"id": 2, "name": "控制权识别"},
        {"id": 3, "name": "推进质量"},
        {"id": 4, "name": "回调vs转换"},
        {"id": 5, "name": "市场接受"},
    ]
    
    cols = st.columns(5)
    for idx, sk in enumerate(skills):
        is_active = sk["id"] == st.session_state["train_mode"]
        if cols[idx].button(sk["name"], type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state["train_mode"] = sk["id"]
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.rerun()
    
    active_skill = next(s for s in skills if s["id"] == st.session_state["train_mode"])
    st.caption(f"当前: {active_skill['name']} | {TRAIN_LEVEL[st.session_state['train_level']]['name']} | 第{st.session_state['skill_round']+1}/2轮")
    
    # 图表
    st.plotly_chart(build_chart(df, bar), use_container_width=True)
    
    # 对话区域
    st.markdown("### 🤖 教练对话")
    for m in st.session_state["coach_dialogue"][-8:]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
    
    # ==================== V22核心：结构化表单输入 ====================
    if st.session_state["skill_round"] < 2:
        report = render_skill_form(st.session_state["train_mode"], st.session_state["skill_round"])
        if report:
            # 发送到AI并获取回复
            response = send_and_get_coach_response(report, df, bar, active_skill)
            
            # 保存对话
            st.session_state["coach_dialogue"].append({"role": "user", "content": report})
            st.session_state["coach_dialogue"].append({"role": "assistant", "content": response})
            
            # 更新画像
            update_profile_from_response(response, st.session_state["train_mode"])
            
            st.session_state["skill_round"] += 1
            st.rerun()
    else:
        st.info("✅ 本轮技能训练完成！点击上方其他技能按钮继续训练，或点击「下一根」K线开始新的分析。")
    
    # 调试信息（开发时可取消注释）
    # with st.expander("调试信息"):
    #     st.json({
    #         "current_bar": bar,
    #         "skill_round": st.session_state["skill_round"],
    #         "train_mode": st.session_state["train_mode"],
    #         "reading_profile": st.session_state.get("reading_profile", {})
    #     })

if __name__ == "__main__":
    main()
