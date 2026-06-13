"""
Al Brooks 结构训练器 V25 - 生产终极版

三位一体框架：
- 时间惯性：连续趋势棒计数 + 80%原则
- 空间磁铁：ATR距离测量 + 盈亏比感知
- 微观信号：H1/H2/L1/L2二浪 + Follow-through质量

用户体验：
- 快捷词按钮 + 智能占位提示
- AI防幻觉约束 + 交易员方程灌输
- 结构化下拉选项，潜意识建立PA心法
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
    1: {"name": "背景阅读", "question": "当前市场背景是什么？",
        "forbidden": ["买", "卖", "做多", "做空", "进场", "止损", "目标位", "开仓", "平仓", "做单", "看涨", "看跌", "预测"]},
    2: {"name": "控制权识别", "question": "现在谁在控制市场？",
        "forbidden": ["趋势", "方向", "预测", "目标位", "进场", "止损", "做多", "做空", "看涨", "看跌"]},
    3: {"name": "推进质量", "question": "最近推进的质量如何？",
        "forbidden": ["趋势", "方向", "多空", "预测", "进场", "止损", "目标", "看涨", "看跌"]},
    4: {"name": "回调vs转换", "question": "这是正常回调还是控制权转换？",
        "forbidden": ["预测", "目标位", "进场", "止损", "开仓", "看涨", "看跌"]},
    5: {"name": "市场接受", "question": "市场是否接受了新价格？",
        "forbidden": ["进场", "止损", "目标位", "预测", "做多", "做空", "看涨", "看跌"]},
}

# ==================== 快捷词库 ====================
QUICK_WORDS = {
    1: ["实体大", "实体小", "光头光脚", "长上影", "长下影", "十字星", "跳空", "重叠多"],
    2: ["买盘主导", "卖盘主导", "无控制权", "测试失败", "反击无力", "跟进强", "跟进弱"],
    3: ["实体放大", "实体缩小", "高度重叠", "跳空缺口", "收高位", "收低位", "影线长"],
    4: ["H1回调", "H2回调", "L1反弹", "L2反弹", "深回调", "触及磁铁", "立即反击"],
    5: ["好跟随", "坏跟随", "假突破", "缺口未补", "停留3根", "推回测试", "空间充足"],
}

# ==================== 技能表单定义（V25 - 交易员方程文案）====================
SKILL_QUALITY_STANDARDS = {
    1: {
        "name": "背景阅读",
        "fields": [
            {"label": "高低点序列", "type": "select", 
             "options": ["请选择...", "HH/HL - 多头序列(上升趋势)", "LH/LL - 空头序列(下降趋势)", "无明显序列 - 震荡区间(TR)", "序列正在转换 - 突破前高/前低"]},
            {"label": "区间/通道边界", "type": "text", 
             "placeholder": "例: 高点3950-3960, 低点3900-3910"},
            {"label": "当前价格位置", "type": "select",
             "options": ["请选择...", "接近上沿(阻力区)", "接近下沿(支撑区)", "中间位置", "正在突破边界"]}
        ]
    },
    2: {
        "name": "控制权识别",
        "fields": [
            {"label": "最近3-5根K线推进方", "type": "select",
             "options": ["请选择...", "多头趋势棒群(强紧迫买盘)", "空头趋势棒群(强紧迫卖盘)", "阴阳交错重叠(无控制权-TR)", "有突破尝试但留长影(被拒绝)"]},
            {"label": "推进方实体质量", "type": "select",
             "options": ["请选择...", "强实体(>60%波幅)", "中等实体(30-60%)", "弱实体/十字星(<30%)", "实体持续放大(加速)"]},
            {"label": "对手方反击情况", "type": "select",
             "options": ["请选择...", "无有效反击", "弱反击(小实体/长影线)", "强反击(大实体反包)", "反击立即被反制"]},
            {"label": "反击是否被跟进", "type": "select",
             "options": ["请选择...", "反击得到后续跟进", "反击被立即反包", "反击后陷入僵持"]}
        ]
    },
    3: {
        "name": "推进质量",
        "fields": [
            {"label": "实体大小变化", "type": "select",
             "options": ["请选择...", "持续放大(加速推进)", "持续缩小(减速衰竭)", "大小交替(不稳定)", "突然放大(爆发)"]},
            {"label": "K线重叠程度", "type": "select",
             "options": ["请选择...", "高度重叠(>70% - 震荡TR)", "中度重叠(40-70%)", "低度重叠(10-40%)", "跳空(缺口Gap)"]},
            {"label": "影线特征", "type": "select",
             "options": ["请选择...", "无明显影线(光头光脚)", "长上影线(抛压/拒绝)", "长下影线(买盘支撑)", "上下影线都长(激烈博弈)"]},
            {"label": "收盘位置", "type": "select",
             "options": ["请选择...", "收在实体高位(>85% - 紧迫)", "收在实体中位", "收在实体低位(<15%)", "收在极值位置"]}
        ]
    },
    4: {
        "name": "回调vs转换",
        "fields": [
            {"label": "回调/反向持续时间", "type": "select",
             "options": ["请选择...", "1-2根(短暂回调)", "3-5根(正常回调)", "6根以上(深度回调)", "持续反向运动"]},
            {"label": "回调类型(二浪心法)", "type": "select",
             "options": ["请选择...", "H1/L1 - 激进第一波入场点(胜率较低)", "H2/L2 - 高胜率双底/双顶微观架构(交易员方程最佳解)", "第三次以上(衰竭前兆)", "深幅回调触及磁铁(20EMA/前密集区)"]},
            {"label": "是否触及关键结构位", "type": "select",
             "options": ["请选择...", "未触及结构位", "触及但未突破", "已突破结构位", "在结构位挣扎"]},
            {"label": "原方向方是否反击", "type": "select",
             "options": ["请选择...", "立即强反击", "弱反击(无效)", "无反击动作", "反击后失败"]}
        ]
    },
    5: {
        "name": "市场接受",
        "fields": [
            {"label": "突破后停留时间", "type": "select",
             "options": ["请选择...", "1根(快速测试)", "2-3根(正常停留)", "3根以上(有效停留)", "未形成有效突破"]},
            {"label": "是否被推回", "type": "select",
             "options": ["请选择...", "未被推回(有效突破)", "部分推回(回踩确认)", "完全推回(假突破)", "推回后立即反弹"]},
            {"label": "跟随棒质量(Follow-through)", "type": "select",
             "options": ["请选择...", "Good Follow-through - 大实体跟进，机构资金买入铁证", "Bad Follow-through - 十字星/小实体，缺乏紧迫感，大概率震荡", "反向吞没(Failed Breakout) - 80%概率假突破反转", "缺口(Gap)未被回补 - 强势确认"]},
            {"label": "磁铁空间检查", "type": "select",
             "options": ["请选择...", "距下一磁铁空间充足(>2倍ATR)", "距磁铁空间不足(<1倍ATR) - 盈亏比极差", "已到达测量目标(MM)", "无明确磁铁目标"]}
        ]
    }
}

# ==================== 训练阶段 ====================
TRAIN_LEVEL = {
    1: {"name": "观察阶段", "desc": "允许模糊、整体感觉、通道、节奏、倾向。"},
    2: {"name": "行为细化阶段", "desc": "开始细化行为、具体K线、推进连续性。"},
    3: {"name": "结构验证阶段", "desc": "允许失败突破、摆动确认、Always In转换、结构争议。"},
}

# ==================== AI系统提示词（V25防幻觉版）====================
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为训练教练，铁血、严厉、绝不妥协。

【核心原则 - 铁血判分机制】
1. 你收到的【K线客观统计数据】是算法严格计算的最高真理，包含：宏观惯性环境、ATR磁铁距离、连续趋势棒计数。
2. 如果用户的选择与客观数据完全吻合，你必须首要肯定其观察的准确性。
3. 你的"反问"必须是建设性延伸，绝对不允许为了反问而强行否定用户正确的选择。
4. 绝对禁止给出交易建议（做多/做空/进场/止损等）。

【Al Brooks 三位一体铁律】
1. 时间惯性：市场已连续运行{trend_bar_count}根趋势棒。80%原则：第一次反向尝试大概率失败。
2. 空间磁铁：价格距上方磁铁{distance_to_high:.1f}倍ATR，距下方磁铁{distance_to_low:.1f}倍ATR。
3. 微观信号：第二次尝试(H2/L2)是高胜率节点，交易员方程最佳解。

【禁止的传统分析杂音】
严格拒绝以下词汇：头肩顶、旗形、楔形、圆弧底、金叉死叉、超买超卖、RSI、MACD。
Al Brooks价格行为学只关注：K线实体、影线、重叠度、收盘位置、结构位、二浪尝试。

当前技能：{skill_name}
阶段：{level_name}
禁止词汇：{forbidden_words}

【第1轮：严审期】
- 对照客观数据逐一审核用户观察
- 检查用户是否使用了传统技术分析杂音（头肩顶/金叉等），直接指正
- 如合格：简短肯定后，反问一个建设性的深层问题
- 如不合格：指出具体错误，引用数据，要求重新观察

【第2轮：亮答案】
- 点评用户回答
- 亮出你自己基于客观数据的完整判断（引用具体K线编号）

【用户历史薄弱点】
{reading_profile}
针对薄弱点进行针对性刁难。

回答简短，不超过200字。
"""

# ==================== 数据加载 ====================
def load_data(symbol, period="30"):
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None
    if df is None or len(df) == 0:
        st.error(f"{symbol} 无数据")
        return None
    df = df.rename(columns={"date": "time", "open": "open", "high": "high",
                            "low": "low", "close": "close", "volume": "volume"})
    df = df.reset_index(drop=True)
    return df

# ==================== 图表构建 ====================
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

# ==================== 核心：识别结构位 ====================
def find_structures(df, bar, lookback=40):
    start = max(0, bar - lookback)
    subset = df.iloc[start:bar+1]
    
    highs = []
    lows = []
    
    for i in range(5, len(subset)-5):
        if subset['high'].iloc[i] > subset['high'].iloc[i-5:i].max() and \
           subset['high'].iloc[i] > subset['high'].iloc[i+1:i+6].max():
            highs.append(subset['high'].iloc[i])
        if subset['low'].iloc[i] < subset['low'].iloc[i-5:i].min() and \
           subset['low'].iloc[i] < subset['low'].iloc[i+1:i+6].min():
            lows.append(subset['low'].iloc[i])
    
    return {"highs": highs[-3:] if highs else [], "lows": lows[-3:] if lows else []}

# ==================== 核心：生成K线语义化标签 ====================
def generate_semantic_labels(df, bar):
    """V25终极版：惯性计数 + ATR磁铁距离 + 二浪识别"""
    
    # 计算ATR
    ranges = df['high'].iloc[max(0, bar-20):bar+1] - df['low'].iloc[max(0, bar-20):bar+1]
    atr = ranges.mean() if len(ranges) > 0 else 1
    
    # 宏观惯性：统计连续趋势棒数量
    lookback = min(30, bar)
    trend_bars = 0
    recent_close = df['close'].iloc[bar]
    for i in range(bar - lookback, bar + 1):
        if i <= 0:
            continue
        o, c = df['open'].iloc[i], df['close'].iloc[i]
        prev_c = df['close'].iloc[i-1]
        total_range = df['high'].iloc[i] - df['low'].iloc[i]
        is_trend_bar = abs(c - o) / (total_range + 0.001) > 0.5
        
        if is_trend_bar and ((c > prev_c and recent_close > prev_c) or (c < prev_c and recent_close < prev_c)):
            trend_bars += 1
        else:
            if i > bar - 5:
                trend_bars = 0
    
    # 结构位
    structures = find_structures(df, bar)
    current_close = df['close'].iloc[bar]
    
    # 磁铁距离
    distance_to_high = 999
    distance_to_low = 999
    if structures["highs"]:
        nearest_high = min(structures["highs"], key=lambda x: abs(x - current_close))
        if nearest_high > current_close:
            distance_to_high = (nearest_high - current_close) / atr
    if structures["lows"]:
        nearest_low = min(structures["lows"], key=lambda x: abs(x - current_close))
        if nearest_low < current_close:
            distance_to_low = (current_close - nearest_low) / atr
    
    # 近期K线分析
    start_idx = max(0, bar - 15)
    recent_df = df.iloc[start_idx:bar+1]
    avg_volume = df['volume'].iloc[max(0, bar-20):bar+1].mean()
    
    lines = []
    lines.append("=" * 55)
    lines.append("【K线客观统计数据 - Al Brooks三位一体框架】")
    lines.append("=" * 55)
    lines.append(f"📊 宏观惯性: 连续趋势棒 {trend_bars} 根")
    if trend_bars >= 10:
        lines.append(f"   ⚡ 80%原则生效: 第一次反向尝试视为回调，不是反转")
    lines.append(f"🧲 磁铁感知: 距上磁铁 {distance_to_high:.1f}倍ATR | 距下磁铁 {distance_to_low:.1f}倍ATR")
    if distance_to_high < 1.0 and distance_to_high > 0:
        lines.append(f"   ⚠️ 上升空间不足1倍ATR，追多盈亏比极差！")
    if distance_to_low < 1.0 and distance_to_low > 0:
        lines.append(f"   ⚠️ 下降空间不足1倍ATR，追空盈亏比极差！")
    lines.append("")
    lines.append("近期K线 Bar-by-Bar:")
    lines.append("")
    
    for idx in range(len(recent_df)):
        row = recent_df.iloc[idx]
        actual_idx = start_idx + idx
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = abs(c - o)
        total_range = h - l
        body_ratio = body / total_range if total_range > 0 else 0
        volume_ratio = row['volume'] / avg_volume if avg_volume > 0 else 1
        
        bar_type = "TrendBar" if body_ratio >= 0.50 else "TRBar"
        
        if body_ratio >= 0.70:
            urgency = "极强紧迫"
        elif body_ratio >= 0.50:
            urgency = "正常"
        else:
            urgency = "弱/无方向"
        
        # 重叠度
        overlap = "首根"
        if idx > 0:
            prev = recent_df.iloc[idx-1]
            o_lap = min(h, prev['high']) - max(l, prev['low'])
            if o_lap > 0 and total_range > 0:
                o_ratio = o_lap / total_range
                overlap = "高度重叠" if o_ratio >= 0.7 else "中度重叠" if o_ratio >= 0.4 else "低度重叠"
            elif o_lap <= 0:
                overlap = "跳空Gap"
        
        direction = "阳" if c >= o else "阴"
        
        lines.append(
            f"K{actual_idx}: {direction}{bar_type} | {urgency} | "
            f"实体{body_ratio*100:.0f}% | {overlap} | "
            f"量比{volume_ratio:.2f}"
        )
    
    # 保存到session供提示词使用
    st.session_state["_trend_bar_count"] = trend_bars
    st.session_state["_distance_to_high"] = distance_to_high
    st.session_state["_distance_to_low"] = distance_to_low
    
    return "\n".join(lines)

# ==================== AI调用 ====================
def call_ai(system_prompt, messages_history, user_report):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【错误】请配置 DeepSeek API 密钥"
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(messages_history)
        messages.append({"role": "user", "content": user_report})
        
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【API错误】{e}"

# ==================== 画像系统 ====================
def build_profile():
    rp = st.session_state.get("reading_profile", {})
    if not rp:
        return "暂无"
    return " | ".join([f"{k}:{v}次" for k, v in sorted(rp.items(), key=lambda x: x[1], reverse=True)])

def update_profile(response, skill_id):
    rp = st.session_state.get("reading_profile", {})
    if "禁止词汇" in response or "不合格" in response:
        rp["使用禁止词汇"] = rp.get("使用禁止词汇", 0) + 1
    if "背景" in response and ("忽略" in response or "遗漏" in response):
        rp["忽略背景阅读"] = rp.get("忽略背景阅读", 0) + 1
    if "预测" in response or "磁铁" in response:
        rp["喜欢提前预测"] = rp.get("喜欢提前预测", 0) + 1
    if "二浪" in response or "H2" in response or "L2" in response:
        rp["忽略二浪结构"] = rp.get("忽略二浪结构", 0) + 1
    st.session_state["reading_profile"] = rp

# ==================== 核心：发送并获取AI反馈 ====================
def process_submission(user_report, df, bar, skill, current_round):
    semantic_labels = generate_semantic_labels(df, bar)
    
    trend_count = st.session_state.get("_trend_bar_count", 0)
    dist_high = st.session_state.get("_distance_to_high", 999)
    dist_low = st.session_state.get("_distance_to_low", 999)
    
    forbidden = "、".join(SKILL_CONSTRAINTS[skill["id"]]["forbidden"])
    level = TRAIN_LEVEL[st.session_state.get("train_level", 1)]
    
    system_prompt = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill["name"],
        level_name=level["name"],
        forbidden_words=forbidden,
        reading_profile=build_profile(),
        trend_bar_count=trend_count,
        distance_to_high=dist_high,
        distance_to_low=dist_low
    )
    
    full_user_message = f"""{semantic_labels}

【用户的观察报告】
{user_report}

请根据客观数据审核用户的观察。"""
    
    history = st.session_state.get("coach_dialogue", [])[:]
    
    with st.spinner("AI教练正在审核..."):
        response = call_ai(system_prompt, history, full_user_message)
    
    st.session_state["coach_dialogue"].append({"role": "user", "content": user_report})
    st.session_state["coach_dialogue"].append({"role": "assistant", "content": response})
    
    update_profile(response, skill["id"])
    st.session_state["skill_round"] = current_round + 1

# ==================== 渲染表单（V25 - 快捷词版）====================
def render_form(skill_id, current_round, df, bar, active_skill):
    std = SKILL_QUALITY_STANDARDS[skill_id]
    fields = std["fields"]
    quick_words = QUICK_WORDS.get(skill_id, ["实体大", "实体小", "影线长", "重叠多"])
    
    st.markdown("---")
    st.markdown(f"✍️ **第{current_round+1}/2轮 - 请完成以下观察**")
    st.caption("💡 每个选项都来自Al Brooks价格行为学的核心概念")
    
    user_inputs = {}
    
    for idx, field in enumerate(fields):
        unique_key = f"form_{skill_id}_{current_round}_{idx}"
        
        if field["type"] == "select":
            user_inputs[field["label"]] = st.selectbox(
                f"**{field['label']}**",
                field["options"],
                key=unique_key
            )
        else:
            # 文本框 + 快捷词按钮
            col1, col2 = st.columns([3, 1])
            with col1:
                placeholder_map = {
                    1: "例: 高点3950-3960, 低点3900-3910",
                    2: "例: 连续3根阴线，实体占波幅>60%，收于低位",
                    3: "例: 阳线实体持续放大，无上影线，收盘创新高",
                    4: "例: 回调3根小阴线，未破前低，原方向反包",
                    5: "例: 突破后停留2根，跟随棒实体大，空间充足",
                }
                placeholder = placeholder_map.get(skill_id, "请进行纯客观描述，禁止预测性词汇")
                
                user_inputs[field["label"]] = st.text_input(
                    f"**{field['label']}**",
                    placeholder=placeholder,
                    key=unique_key
                )
            with col2:
                st.markdown("---")
                st.markdown("**快捷词**")
                quick_cols = st.columns(2)
                for i, word in enumerate(quick_words[:4]):
                    if quick_cols[i % 2].button(word, key=f"quick_{skill_id}_{idx}_{i}", use_container_width=True):
                        current_val = user_inputs.get(field["label"], "")
                        if current_val:
                            new_val = current_val + f" {word}"
                        else:
                            new_val = word
                        # 更新session state中的值
                        st.session_state[unique_key] = new_val
                        st.rerun()
    
    if st.button("📤 提交观察", type="primary", use_container_width=True, key=f"submit_{skill_id}_{current_round}"):
        has_empty = False
        for label, value in user_inputs.items():
            if not value or value == "请选择...":
                has_empty = True
                st.error(f"⚠️ 请完成「{label}」的选择/填写")
                break
        
        if not has_empty:
            report = f"【第{current_round+1}轮观察 - {active_skill['name']}】\n"
            for label, value in user_inputs.items():
                report += f"- {label}: {value}\n"
            
            process_submission(report, df, bar, active_skill, current_round)
            st.rerun()

# ==================== 辅助函数 ====================
@st.cache_data(ttl=3600)
def fetch_contracts():
    mc = {}
    for ex in ["shfe", "dce", "czce", "cffex"]:
        try:
            result = ak.match_main_contract(symbol=ex)
            for c in str(result).split(","):
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

def load_symbol(code, main, period="30"):
    with st.spinner("加载中..."):
        df = load_data(main, period=period)
        if df is not None:
            st.session_state["chart_df"] = df
            st.session_state["current_bar"] = random_bar(df)
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.session_state["symbol_code"] = code
            st.session_state["symbol_name"] = SYMBOL_NAMES.get(code, code)
            st.session_state["data_period"] = period
            st.success(f"已加载 {st.session_state['symbol_name']} | {len(df)}根K线")
            time.sleep(0.3)
            st.rerun()

# ==================== 主函数 ====================
def main():
    st.set_page_config(page_title="Al Brooks 结构训练器 V25 - 生产终极版", layout="wide")
    
    defaults = {
        "chart_df": None, "current_bar": 40, "coach_dialogue": [],
        "skill_round": 0, "train_level": 1, "train_mode": 1,
        "reading_profile": {}, "data_period": "30",
        "_trend_bar_count": 0, "_distance_to_high": 999, "_distance_to_low": 999,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 品种选择")
        
        period_map = {"15分钟": "15", "30分钟": "30", "60分钟": "60", "日线": "day"}
        period = st.selectbox("周期", list(period_map.keys()), index=1)
        selected_period = period_map[period]
        
        exchanges = {
            "金融": ["IF", "IH", "IC", "IM"],
            "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
            "黑色": ["RB", "HC", "I", "J", "JM"],
            "化工": ["MA", "TA", "PP", "L", "V", "SA", "UR"],
            "农产品": ["A", "M", "Y", "P", "C", "CF", "SR", "RM", "OI", "FG", "AP"],
        }
        
        for cat, codes in exchanges.items():
            with st.expander(cat, expanded=(cat == "金融")):
                cols = st.columns(4)
                for i, code in enumerate(codes):
                    if cols[i % 4].button(code, key=f"sym_{code}"):
                        load_symbol(code, f"{code}0", period=selected_period)
        
        if st.session_state.get("chart_df"):
            df = st.session_state["chart_df"]
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎲 随机K线", use_container_width=True):
                    st.session_state["current_bar"] = random_bar(df)
                    st.session_state["coach_dialogue"] = []
                    st.session_state["skill_round"] = 0
                    st.rerun()
            with col2:
                if st.button("⏭️ 下一根", use_container_width=True):
                    if st.session_state["current_bar"] < len(df) - 1:
                        st.session_state["current_bar"] += 1
                        st.session_state["coach_dialogue"] = []
                        st.session_state["skill_round"] = 0
                        st.rerun()
        
        st.divider()
        
        st.selectbox("训练阶段", [1,2,3], 
                     format_func=lambda x: f"阶段{x}: {TRAIN_LEVEL[x]['name']}",
                     index=st.session_state["train_level"]-1, key="train_level")
        
        if st.session_state.get("reading_profile"):
            st.markdown("### 📈 阅读画像")
            for k, v in st.session_state["reading_profile"].items():
                st.progress(min(v/10, 1.0), text=f"{k}: {v}次")
            if st.button("重置画像"):
                st.session_state["reading_profile"] = {}
                st.rerun()
        
        st.divider()
        st.caption("💡 Al Brooks 三位一体框架")
        st.caption("• 时间惯性: 连续趋势棒计数")
        st.caption("• 空间磁铁: ATR距离测量")
        st.caption("• 微观信号: H1/H2/L1/L2二浪")
        st.caption("• 交易员方程: 胜率×盈亏比")
    
    # 主区域
    df = st.session_state.get("chart_df")
    if df is None:
        st.info("👈 请从左侧选择品种开始训练")
        return
    
    bar = st.session_state["current_bar"]
    
    # 技能选择
    skills = [{"id": i, "name": SKILL_CONSTRAINTS[i]["name"]} for i in range(1, 6)]
    cols = st.columns(5)
    for i, sk in enumerate(skills):
        active = sk["id"] == st.session_state["train_mode"]
        if cols[i].button(sk["name"], type="primary" if active else "secondary", use_container_width=True):
            st.session_state["train_mode"] = sk["id"]
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.rerun()
    
    active_skill = next(s for s in skills if s["id"] == st.session_state["train_mode"])
    st.caption(f"🎯 {active_skill['name']} | {TRAIN_LEVEL[st.session_state['train_level']]['name']} | 第{st.session_state['skill_round']+1}/2轮")
    
    # 图表
    st.plotly_chart(build_chart(df, bar), use_container_width=True)
    
    # 对话区域
    st.markdown("### 🤖 铁血教练")
    for m in st.session_state["coach_dialogue"][-10:]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
    
    # 表单
    if st.session_state["skill_round"] < 2:
        render_form(st.session_state["train_mode"], st.session_state["skill_round"], df, bar, active_skill)
    else:
        st.success("✅ 本轮训练完成！点击「下一根」K线或切换技能继续")
        st.info("💡 小提示：H2/L2是交易员方程的最佳解，关注第二次尝试")

if __name__ == "__main__":
    main()
