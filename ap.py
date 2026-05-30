"""
Al Brooks 结构训练器 V19
================================
核心优化：
  1. 修复真实数据加载问题
  2. 使用正确的 akshare 接口
  3. 适配 Streamlit Cloud Secrets
  4. 增加更强的错误处理和重试
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
from datetime import datetime
import time
import json
import random

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="Al Brooks 结构训练器", layout="wide")

# ── 品种分类定义（使用标准合约代码）───────────────────────
# 期货品种代码映射（akshare 需要的格式）
PRODUCT_CATEGORIES = {
    "金融": {"code": "IF", "name": "沪深300股指", "exchange": "cffex"},
    "金融": {"code": "IH", "name": "上证50股指", "exchange": "cffex"},
    "金融": {"code": "IC", "name": "中证500股指", "exchange": "cffex"},
    "金融": {"code": "IM", "name": "中证1000股指", "exchange": "cffex"},
    "金融": {"code": "T", "name": "10年期国债", "exchange": "cffex"},
    "有色": {"code": "CU", "name": "沪铜", "exchange": "shfe"},
    "有色": {"code": "AL", "name": "沪铝", "exchange": "shfe"},
    "有色": {"code": "ZN", "name": "沪锌", "exchange": "shfe"},
    "有色": {"code": "PB", "name": "沪铅", "exchange": "shfe"},
    "有色": {"code": "NI", "name": "沪镍", "exchange": "shfe"},
    "有色": {"code": "SN", "name": "沪锡", "exchange": "shfe"},
    "有色": {"code": "AU", "name": "黄金", "exchange": "shfe"},
    "有色": {"code": "AG", "name": "白银", "exchange": "shfe"},
    "黑色": {"code": "RB", "name": "螺纹钢", "exchange": "shfe"},
    "黑色": {"code": "HC", "name": "热轧卷板", "exchange": "shfe"},
    "黑色": {"code": "I", "name": "铁矿石", "exchange": "dce"},
    "黑色": {"code": "JM", "name": "焦煤", "exchange": "dce"},
    "黑色": {"code": "J", "name": "焦炭", "exchange": "dce"},
    "化工": {"code": "V", "name": "PVC", "exchange": "dce"},
    "化工": {"code": "PP", "name": "聚丙烯", "exchange": "dce"},
    "化工": {"code": "L", "name": "聚乙烯", "exchange": "dce"},
    "化工": {"code": "TA", "name": "PTA", "exchange": "czce"},
    "化工": {"code": "MA", "name": "甲醇", "exchange": "czce"},
    "化工": {"code": "RU", "name": "橡胶", "exchange": "shfe"},
    "化工": {"code": "BU", "name": "沥青", "exchange": "shfe"},
    "化工": {"code": "FU", "name": "燃料油", "exchange": "shfe"},
    "化工": {"code": "EG", "name": "乙二醇", "exchange": "dce"},
    "化工": {"code": "EB", "name": "苯乙烯", "exchange": "dce"},
    "化工": {"code": "PG", "name": "液化气", "exchange": "dce"},
    "化工": {"code": "SA", "name": "纯碱", "exchange": "czce"},
    "化工": {"code": "UR", "name": "尿素", "exchange": "czce"},
    "化工": {"code": "PF", "name": "短纤", "exchange": "czce"},
    "农产品": {"code": "M", "name": "豆粕", "exchange": "dce"},
    "农产品": {"code": "Y", "name": "豆油", "exchange": "dce"},
    "农产品": {"code": "P", "name": "棕榈油", "exchange": "dce"},
    "农产品": {"code": "A", "name": "豆一", "exchange": "dce"},
    "农产品": {"code": "B", "name": "豆二", "exchange": "dce"},
    "农产品": {"code": "C", "name": "玉米", "exchange": "dce"},
    "农产品": {"code": "CS", "name": "玉米淀粉", "exchange": "dce"},
    "农产品": {"code": "JD", "name": "鸡蛋", "exchange": "dce"},
    "农产品": {"code": "AP", "name": "苹果", "exchange": "czce"},
    "农产品": {"code": "CF", "name": "棉花", "exchange": "czce"},
    "农产品": {"code": "SR", "name": "白糖", "exchange": "czce"},
    "农产品": {"code": "OI", "name": "菜油", "exchange": "czce"},
    "农产品": {"code": "RM", "name": "菜粕", "exchange": "czce"},
    "农产品": {"code": "LH", "name": "生猪", "exchange": "dce"},
    "能源": {"code": "SC", "name": "原油", "exchange": "ine"},
    "能源": {"code": "LU", "name": "低硫燃油", "exchange": "ine"},
    "能源": {"code": "NR", "name": "20号胶", "exchange": "ine"},
}

# 重新组织分类显示
DISPLAY_CATEGORIES = {
    "金融": ["IF", "IH", "IC", "IM", "T"],
    "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
    "黑色": ["RB", "HC", "I", "JM", "J"],
    "化工": ["V", "PP", "L", "TA", "MA", "RU", "BU", "FU", "EG", "EB", "PG", "SA", "UR", "PF"],
    "农产品": ["M", "Y", "P", "A", "B", "C", "CS", "JD", "AP", "CF", "SR", "OI", "RM", "LH"],
    "能源": ["SC", "LU", "NR"],
}
DEFAULT_EXPANDED = ["金融"]

# ── 5个技能定义 ─────────────────────────────────────────
SKILLS = [
    {"id": 1, "name": "背景阅读",   "question": "当前市场背景是什么？"},
    {"id": 2, "name": "控制权识别", "question": "现在谁在控制市场？"},
    {"id": 3, "name": "推进质量",   "question": "最近推进的质量如何？"},
    {"id": 4, "name": "回调vs转换", "question": "这是正常回调还是控制权转换？"},
    {"id": 5, "name": "市场接受",   "question": "市场是否接受了新价格？"},
]

# ── AI配置（从 Streamlit Secrets 读取）────────────────
def get_ai_config():
    """安全获取AI配置"""
    try:
        if "ai" in st.secrets:
            return {
                "base_url": st.secrets["ai"].get("base_url", "https://api.deepseek.com/v1"),
                "api_key": st.secrets["ai"].get("api_key", ""),
                "model": st.secrets["ai"].get("model", "deepseek-chat"),
            }
        else:
            return {
                "base_url": st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                "api_key": st.secrets.get("OPENAI_API_KEY", ""),
                "model": st.secrets.get("OPENAI_MODEL", "deepseek-chat"),
            }
    except Exception:
        return {
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
        }

AI_CONFIG = get_ai_config()

# ── 训练阶段配置 ─────────────────────────────────────────
LEVEL_CONFIG = {
    "level1": {
        "name": "观察阶段",
        "desc": "识别市场结构特征（趋势/震荡/通道/双重顶底）",
        "n_bars": 40,
    },
    "level2": {
        "name": "行为细化阶段",
        "desc": "分析K线行为细节（影线/实体/嵌套/突破）",
        "n_bars": 30,
    },
    "level3": {
        "name": "结构验证阶段",
        "desc": "验证结构预期与多时间框架一致性",
        "n_bars": 60,
    },
}

# ── AI Prompt 模板 ──────────────────────────────────────
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为交易教练。

【核心职责】
1. 训练师：根据训练阶段和当前技能目的，提供结构化的分析指导
2. 点评师：对用户的分析给出专业点评，指出对错与遗漏

当前训练阶段：{level_name} - {level_desc}
当前技能：{skill_name}
核心提问：{skill_question}

【5个技能的核心分析维度】
1. 背景阅读 → 趋势方向、震荡区间、关键支撑阻力、高低点序列(HH/HL或LH/LL)
2. 控制权识别 → 趋势线的角度和持续性、突破K线的力度（实体大小/影线长度）、连续同向K线的数量
3. 推进质量 → 推进波的幅度、K线重叠程度、影线长度、动能是否衰减
4. 回调vs转换 → 回调的时间/幅度特征、转换的确认信号、对手方是否连续出现
5. 市场接受 → 价格对新区域的停留时间、重叠K线的数量、测试关键价位后的反应

【对话流程 - 严格执行】
- 第1轮：围绕技能核心提问进行引导，描述当前市场结构，提出有针对性的问题。不要直接给答案。
- 第2轮：基于用户回答给出结构化点评：1)肯定正确的部分 2)指出遗漏或偏差 3)给出清晰的判断结论

【回答风格】
- 简洁、专业、直击要点，不超过150字
- 使用具体的K线编号作为依据
- 基于实际结构分析，不泛泛而谈"""


def _market_msg(kline_df: pd.DataFrame, n_bars: int = 40) -> str:
    """将K线数据转化为自然语言的市场描述"""
    if kline_df is None or kline_df.empty:
        return "暂无数据"

    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return "数据不足"

    # 统一列名
    if "open" in df.columns:
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    lines = [f"【当前K线】第{n}号K线", "", "【最近行情描述】"]

    start = max(0, n - 15)
    for i in range(start, n):
        body = abs(c[i] - o[i])
        total_range = h[i] - l[i]
        body_ratio = body / total_range if total_range > 0 else 0
        direction = "阳" if c[i] >= o[i] else "阴"

        if body_ratio >= 0.7:
            k_type = "大阳线" if direction == "阳" else "大阴线"
        elif body_ratio >= 0.4:
            k_type = "中阳线" if direction == "阳" else "中阴线"
        elif body_ratio >= 0.1:
            k_type = "小阳线" if direction == "阳" else "小阴线"
        else:
            k_type = "十字星"

        change_desc = ""
        if i > start:
            price_change = c[i] - c[i-1]
            if abs(price_change) > total_range * 0.8 and total_range > 0:
                change_desc = f"，相比前一根{'大涨' if price_change > 0 else '大跌'}了{abs(price_change):.1f}"
            elif price_change > 0:
                change_desc = f"，比前一根涨了{price_change:.1f}"
            elif price_change < 0:
                change_desc = f"，比前一根跌了{abs(price_change):.1f}"

        lines.append(f"  K{i+1}: {k_type}，开{o[i]:.0f} 收{c[i]:.0f} 高{h[i]:.0f} 低{l[i]:.0f}{change_desc}")

    lines.extend(["", "【整体市场感知】"])

    if n >= 10:
        yang_count = sum(1 for j in range(n-10, n) if c[j] >= o[j])
        yin_count = 10 - yang_count

        if yang_count >= 7:
            bias = "近期明显偏多，阳线占主导"
        elif yin_count >= 7:
            bias = "近期明显偏空，阴线占主导"
        else:
            bias = "近期多空平衡"
        lines.append(f"  • {bias}（最近10根中{yang_count}阳{yin_count}阴）")

        total_change = c[-1] - c[0]
        if total_change > 0:
            lines.append(f"  • 整体向上，累计上涨{total_change:.1f}")
        elif total_change < 0:
            lines.append(f"  • 整体向下，累计下跌{abs(total_change):.1f}")

    return "\n".join(lines)


def ask_coach(
    skill_name: str,
    skill_question: str,
    market_msg: str,
    level_name: str,
    level_desc: str,
    user_input: str = "",
    is_second_round: bool = False,
) -> str:
    """调用AI教练"""
    if not AI_CONFIG["api_key"]:
        return "⚠️ API Key 未配置，请在 Streamlit Secrets 中设置"

    client = OpenAI(
        base_url=AI_CONFIG["base_url"],
        api_key=AI_CONFIG["api_key"],
    )

    system_prompt = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        skill_question=skill_question,
        level_name=level_name,
        level_desc=level_desc,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"当前市场状况：\n{market_msg}"},
    ]

    if "chat_history" in st.session_state:
        for m in st.session_state.chat_history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})

    if is_second_round:
        messages.append({
            "role": "user",
            "content": f"【第2轮】用户回答：{user_input}\n\n请按以下结构点评：\n1. 肯定正确的部分\n2. 指出遗漏或偏差\n3. 给出清晰的判断结论"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"【第1轮】当前技能：「{skill_name}」，核心提问：「{skill_question}」。\n请描述当前市场结构，并提出引导性问题。"
        })

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=AI_CONFIG["model"],
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return f"[AI调用失败] {str(e)}"
    
    return "[AI调用失败] 未知错误"


# ── 真实数据获取（使用 akshare 正确接口）────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_data(symbol: str, period: str = "30"):
    """
    获取真实期货K线数据
    period: 1, 5, 15, 30, 60 分钟
    """
    for attempt in range(3):
        try:
            # 方法1: 使用 futures_zh_minute_sina
            df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
            
            if df is not None and not df.empty:
                # 处理返回的数据
                if isinstance(df, pd.DataFrame):
                    # 重命名列
                    rename_dict = {}
                    for old_name in df.columns:
                        if 'date' in old_name.lower() or 'datetime' in old_name.lower():
                            rename_dict[old_name] = 'datetime'
                        elif 'open' in old_name.lower():
                            rename_dict[old_name] = 'open'
                        elif 'high' in old_name.lower():
                            rename_dict[old_name] = 'high'
                        elif 'low' in old_name.lower():
                            rename_dict[old_name] = 'low'
                        elif 'close' in old_name.lower():
                            rename_dict[old_name] = 'close'
                        elif 'volume' in old_name.lower():
                            rename_dict[old_name] = 'volume'
                    
                    df = df.rename(columns=rename_dict)
                    
                    # 确保有必要的列
                    required_cols = ['datetime', 'open', 'high', 'low', 'close']
                    if all(col in df.columns for col in required_cols):
                        # 转换数据类型
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.dropna(subset=required_cols)
                        df = df.sort_values('datetime').reset_index(drop=True)
                        return df
            
            # 方法2: 使用期货主力合约行情
            if attempt == 1:
                main_contract = ak.match_main_contract(symbol=symbol)
                if main_contract:
                    df = ak.futures_zh_minute_sina(symbol=main_contract, period=period)
                    if df is not None and not df.empty:
                        return df
            
            # 方法3: 尝试更短周期
            if attempt == 2:
                period_alt = "15" if period == "30" else period
                df = ak.futures_zh_minute_sina(symbol=symbol, period=period_alt)
                if df is not None and not df.empty:
                    return df
                        
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
    
    return None


def get_main_contract(symbol: str) -> str:
    """获取主力合约代码"""
    try:
        main = ak.match_main_contract(symbol=symbol)
        if main:
            return main
    except Exception:
        pass
    return symbol


# ── 图表绘制 ────────────────────────────────────────────
def plot_kline(kline_df: pd.DataFrame, n_bars: int = 40):
    """绘制K线图"""
    if kline_df is None or kline_df.empty:
        return go.Figure()

    df = kline_df.tail(n_bars).copy()
    if len(df) < 5:
        return go.Figure()

    # 统一列名
    if "open" in df.columns:
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )

    if "Volume" in df.columns:
        colors = ["#ef5350" if row["Close"] >= row["Open"] else "#26a69a" for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df["datetime"], y=df["Volume"], name="成交量", marker_color=colors, opacity=0.5),
            row=2, col=1,
        )

    # K线编号
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 5 == 0:
            fig.add_annotation(
                x=row["datetime"],
                y=row["High"],
                text=str(len(df) - idx),
                showarrow=False,
                yshift=8,
                font_size=8,
                font_color="#666",
                row=1, col=1,
            )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(matches="x", row=2, col=1)
    return fig


# ── Session State 初始化 ──────────────────────────────
defaults = {
    "current_skill": None,
    "skill_round": 1,
    "chat_history": [],
    "data_loaded": False,
    "kline_data": None,
    "current_symbol": "RB",
    "train_level": "level1",
    "load_error": None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 品种选择")

    current_symbol = st.session_state.current_symbol

    for cat_name, symbols in DISPLAY_CATEGORIES.items():
        expanded = cat_name in DEFAULT_EXPANDED
        with st.expander(cat_name, expanded=expanded):
            cols = st.columns(min(4, len(symbols)))
            for i, sym in enumerate(symbols):
                col = cols[i % len(cols)]
                if col.button(sym, key=f"sym_{cat_name}_{sym}", type="primary" if sym == current_symbol else "secondary", use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.session_state.data_loaded = False
                    st.session_state.kline_data = None
                    st.session_state.chat_history = []
                    st.session_state.skill_round = 1
                    st.session_state.load_error = None
                    st.rerun()

    st.markdown("---")

    # 训练阶段
    st.markdown("**训练阶段**")
    level_options = list(LEVEL_CONFIG.keys())
    level_labels = [f"阶段{i+1}: {LEVEL_CONFIG[l]['name']}" for i, l in enumerate(level_options)]
    selected_level_label = st.selectbox("", level_labels, label_visibility="collapsed")
    train_level = level_options[level_labels.index(selected_level_label)]
    st.session_state.train_level = train_level

    level_info = LEVEL_CONFIG[train_level]
    n_bars = level_info["n_bars"]
    level_name = level_info["name"]
    level_desc = level_info["desc"]

    st.markdown("---")

    # 技能选择
    st.markdown("**选择技能**")
    skill_labels = [f"{s['name']}" for s in SKILLS]
    selected_skill_name = st.radio(
        "",
        skill_labels,
        index=None,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # 数据状态
    if st.session_state.kline_data is not None:
        st.caption(f"📊 {len(st.session_state.kline_data)} 根K线")
    elif st.session_state.load_error:
        st.error(st.session_state.load_error)
    else:
        st.caption("⚡ 等待加载数据")


# ═══════════════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════════════

# 技能切换处理
if selected_skill_name is not None:
    skill_obj = next(s for s in SKILLS if s["name"] == selected_skill_name)
    if st.session_state.current_skill is None or st.session_state.current_skill["id"] != skill_obj["id"]:
        st.session_state.current_skill = skill_obj
        st.session_state.skill_round = 1
        st.session_state.chat_history = []
        st.rerun()

# 状态栏
if st.session_state.current_skill:
    skill = st.session_state.current_skill
    round_label = "第1轮(引导)" if st.session_state.skill_round == 1 else "第2轮(点评)"
    st.caption(
        f"📊 {st.session_state.current_symbol} | "
        f"🎯 {skill['name']} — {skill['question']} | "
        f"📖 {level_name} | {round_label}"
    )

# 数据加载
if not st.session_state.data_loaded:
    with st.spinner(f"加载 {st.session_state.current_symbol} 真实数据..."):
        df = fetch_realtime_data(st.session_state.current_symbol, period="30")
        
        if df is not None and len(df) > 0:
            st.session_state.kline_data = df
            st.session_state.data_loaded = True
            st.session_state.load_error = None
            st.rerun()
        else:
            error_msg = f"无法加载 {st.session_state.current_symbol} 数据。请检查网络或稍后重试。"
            st.session_state.load_error = error_msg
            st.error(error_msg)
            
            # 显示可能的解决方案
            with st.expander("查看解决方案"):
                st.markdown("""
                1. **检查网络连接**：确保 Streamlit Cloud 可以访问外网
                2. **更换品种**：尝试其他品种（如 RB、M、Y 等主流品种）
                3. **稍后重试**：数据源可能有临时问题
                4. **本地测试**：先在本地运行确认代码正常
                """)

# 图表区
if st.session_state.data_loaded and st.session_state.kline_data is not None:
    fig = plot_kline(st.session_state.kline_data, n_bars)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.markdown("---")

# 对话区
if st.session_state.current_skill is None:
    st.info("👈 从左侧选择品种、阶段和技能开始训练")
elif not st.session_state.data_loaded:
    st.warning("⚠️ 请等待数据加载完成，或切换其他品种")
else:
    skill = st.session_state.current_skill

    # 显示历史对话
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 第1轮自动发送引导
    is_round2 = st.session_state.skill_round == 2
    has_guide = any("第1轮" in m.get("content", "") for m in st.session_state.chat_history)

    if not is_round2 and not has_guide and st.session_state.data_loaded:
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    level_name=level_name,
                    level_desc=level_desc,
                    is_second_round=False,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 用户输入
    prompt_text = "你的观察是？" if not is_round2 else "你的回答是？(第2轮)"
    user_input = st.chat_input(prompt_text)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                market_msg = _market_msg(st.session_state.kline_data, n_bars)
                reply = ask_coach(
                    skill_name=skill["name"],
                    skill_question=skill["question"],
                    market_msg=market_msg,
                    level_name=level_name,
                    level_desc=level_desc,
                    user_input=user_input,
                    is_second_round=is_round2,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if is_round2:
            st.session_state.skill_round = 1
            st.success("✅ 本轮训练完成，可切换其他技能继续训练")

# ── 样式 ────────────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp { margin: 0; padding: 0; }
    .block-container { padding: 0.5rem 1.5rem 0.5rem 1.5rem !important; }
    section[data-testid="stSidebar"] > div { padding: 0.5rem !important; }
    hr { margin: 6px 0 !important; }
    .stPlotlyChart { margin: 0 !important; }
    .stAlert { margin: 10px 0; }
</style>
""",
    unsafe_allow_html=True,
)
