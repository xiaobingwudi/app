"""
Al Brooks 日内机会寻找训练器 V6.0
核心目标：训练用户在不知道未来的情况下，持续更新对市场控制权的理解

两种模式：
1. 读盘模式（90%）：逐段开放K线，每段判断控制权，新证据出现后修正
2. 故事模式（10%）：完整80根K线，自由讲述，AI随机挑战
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
import random
import time
import re

# ==================== 配置 ====================
SYMBOL_NAMES = {
    "IF": "沪深300股指", "IH": "上证50股指", "IC": "中证500股指", "IM": "中证1000股指",
    "CU": "沪铜", "AL": "沪铝", "ZN": "沪锌", "PB": "沪铅", "NI": "沪镍", "SN": "沪锡",
    "AU": "黄金", "AG": "白银", "RB": "螺纹钢", "HC": "热轧卷板", "SS": "不锈钢",
    "I": "铁矿石", "J": "焦炭", "JM": "焦煤",
    "MA": "甲醇", "TA": "PTA", "SA": "纯碱", "FG": "玻璃",
    "A": "豆一", "M": "豆粕", "Y": "豆油", "P": "棕榈油", "C": "玉米",
    "CF": "棉花", "SR": "白糖", "AP": "苹果",
    "SC": "原油", "FU": "燃料油",
}

EXCHANGES = {
    "股指": ["IF", "IH", "IC", "IM"],
    "黑色": ["RB", "I", "J", "JM"],
    "化工": ["MA", "TA", "SA", "FG"],
    "农产品": ["A", "M", "Y", "P", "C", "CF", "SR"],
    "有色": ["CU", "AL", "ZN", "AU", "AG"],
    "能源": ["SC", "FU"],
}

# 读盘模式分段节点（从K20开始，每10-20根开放一段）
READING_SEGMENTS = [20, 35, 50, 65, 80]

# AI挑战问题池（随机抽取，防止模板化）
CHALLENGE_QUESTIONS = {
    "evidence": [
        "哪根K线最支持你的观点？",
        "如果删掉这3根K线，你还会这么看吗？",
        "你看到的证据是背景信号还是入场信号？",
        "哪一段的成交量最支持你的判断？"
    ],
    "control": [
        "谁真正掌控市场？多头还是空头？",
        "控制权在什么时候发生了变化？",
        "有没有你误判控制权的地方？",
        "当前是趋势市还是震荡市？"
    ],
    "counter": [
        "什么情况会推翻你的故事？",
        "哪个证据与你的判断矛盾？",
        "有没有忽略失败一方的信号？",
        "如果最不利的情况发生，你准备怎么办？"
    ],
    "trade": [
        "如果必须做一笔，只做哪一笔？",
        "哪笔看起来最好但实际最危险？",
        "什么时候应该放弃等待？",
        "当前风险回报比大约是多少？"
    ]
}


# ==================== AI 提示词 ====================
READING_MODE_SYSTEM = """你是 Al Brooks 价格行为教练。

【当前模式】读盘模式
用户正在逐段看到K线，他不知道后面会发生什么。
你的任务是：挑战用户对当前市场控制权的判断。

【输出格式】
- 不要说"对"或"错"
- 只问一个问题（从以下类别中随机选择）
- 控制在80字以内

【问题池】
证据类：{evidence_q}
控制权类：{control_q}
反证类：{counter_q}
"""

STORY_MODE_SYSTEM = """你是 Al Brooks 价格行为教练。

【当前模式】故事模式
用户已经看完了完整的80根K线。
他正在自由讲述今天的故事。
你的任务是：随机选一个问题挑战他。

【输出格式】
- 不要说"对"或"错"
- 只问一个问题
- 控制在100字以内

【问题池】
{questions}
"""


def get_random_question(category=None):
    """随机获取一个问题"""
    if category and category in CHALLENGE_QUESTIONS:
        return random.choice(CHALLENGE_QUESTIONS[category])
    all_questions = []
    for q_list in CHALLENGE_QUESTIONS.values():
        all_questions.extend(q_list)
    return random.choice(all_questions)


def call_reading_mode_challenge(user_story, segment_start, segment_end):
    """读盘模式：AI挑战"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【提示】请配置API密钥"
    
    # 随机选择一个问题类别
    categories = list(CHALLENGE_QUESTIONS.keys())
    chosen_cat = random.choice(categories)
    chosen_q = random.choice(CHALLENGE_QUESTIONS[chosen_cat])
    
    system = READING_MODE_SYSTEM.format(
        evidence_q=random.choice(CHALLENGE_QUESTIONS["evidence"]),
        control_q=random.choice(CHALLENGE_QUESTIONS["control"]),
        counter_q=random.choice(CHALLENGE_QUESTIONS["counter"])
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【K线范围】K{segment_start} ~ K{segment_end}\n【用户判断】{user_story}"}
    ]
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.5, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI: 你注意到控制权变化了吗？ (API错误: {str(e)[:50]})"


def call_story_mode_challenge(user_story):
    """故事模式：AI随机挑战"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【提示】请配置API密钥"
    
    all_questions = []
    for q_list in CHALLENGE_QUESTIONS.values():
        all_questions.extend(q_list)
    random_q = random.choice(all_questions)
    
    system = STORY_MODE_SYSTEM.format(questions="\n".join(all_questions))
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【用户的故事】{user_story}\n\n请用这个问题挑战他：{random_q}"}
    ]
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.5, max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI: {random_q} (API错误: {str(e)[:50]})"


# ==================== 数据加载 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, period="30"):
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None


# ==================== 图表绘制 ====================
def build_chart(df, max_bar, show_labels=True):
    """绘制K线图"""
    start = max(0, len(df) - max_bar)
    plot_df = df.iloc[start:].copy().reset_index(drop=True)
    n_bars = len(plot_df)
    bar_numbers = list(range(1, n_bars + 1))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.8, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        showlegend=False,
        increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    vol_colors = ["#ef5350" if c >= o else "#26a69a"
                  for o, c in zip(plot_df["open"], plot_df["close"])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["volume"],
        marker_color=vol_colors, showlegend=False, opacity=0.5
    ), row=2, col=1)

    # 只在需要时显示编号
    if show_labels:
        for idx, bar_num in enumerate(bar_numbers):
            if bar_num % 10 == 0:  # 每10根显示一次
                row_data = plot_df.iloc[idx]
                if row_data["close"] >= row_data["open"]:
                    y_pos = row_data["low"]
                    y_shift = -22
                else:
                    y_pos = row_data["high"]
                    y_shift = 22
                fig.add_annotation(
                    x=idx, y=y_pos,
                    text=f"K{bar_num}",
                    showarrow=False,
                    font=dict(size=9, color="#666666"),
                    yshift=y_shift,
                    row=1, col=1
                )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    
    return fig


def build_partial_chart(df, max_bar, current_end):
    """绘制部分K线图（读盘模式）"""
    start = max(0, len(df) - max_bar)
    plot_df = df.iloc[start:start+current_end].copy().reset_index(drop=True)
    n_bars = len(plot_df)
    bar_numbers = list(range(1, n_bars + 1))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.8, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        showlegend=False,
        increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    vol_colors = ["#ef5350" if c >= o else "#26a69a"
                  for o, c in zip(plot_df["open"], plot_df["close"])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["volume"],
        marker_color=vol_colors, showlegend=False, opacity=0.5
    ), row=2, col=1)

    # 标记分段边界
    for seg in READING_SEGMENTS:
        if seg <= current_end:
            fig.add_vline(
                x=seg - 1, line_dash="dot",
                line_color="#f9c74f", line_width=1.5, opacity=0.6
            )
            fig.add_annotation(
                x=seg - 1, y=plot_df.iloc[min(seg-1, n_bars-1)]["high"],
                text=f"K{seg}", showarrow=False,
                font=dict(size=8, color="#f9c74f"),
                yshift=15
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    
    return fig


# ==================== Session State ====================
def init_state():
    defaults = {
        "df": None,
        "symbol": None,
        "max_bar": 80,
        "mode": "reading",  # "reading" or "story"
        "current_segment_idx": 0,
        "segment_judgments": [],
        "conversations": [],
        "session_complete": False,
        "practice_count": 0,
        "correction_count": 0,  # 记录观点修正次数
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.current_segment_idx = 0
    st.session_state.segment_judgments = []
    st.session_state.conversations = []
    st.session_state.session_complete = False
    st.session_state.correction_count = 0


def load_new_symbol(code, period_value):
    with st.spinner(f"加载 {code} {period_value}分钟..."):
        df = load_data(f"{code}0", period=period_value)
        if df is None or len(df) < 100:
            st.error(f"{code} 数据加载失败")
            return False
        
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.max_bar = 80
        reset_session()
        return True


def check_correction(previous_judgment, current_judgment):
    """检查用户是否修正了观点"""
    if not previous_judgment or not current_judgment:
        return False
    
    # 简单判断：控制权描述是否发生变化
    control_keywords = ["多头", "空头", "震荡", "买方", "卖方", "均衡"]
    prev_control = None
    curr_control = None
    
    for kw in control_keywords:
        if kw in previous_judgment:
            prev_control = kw
        if kw in current_judgment:
            curr_control = kw
    
    return prev_control != curr_control and prev_control is not None and curr_control is not None


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 训练器 V6.0", layout="wide")
    
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 8px 12px;
    }
    .mode-reading {
        background-color: #e3f2fd;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    .mode-story {
        background-color: #e8f5e9;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 训练器 V6.0")
        st.markdown("---")
        
        # 模式选择
        mode = st.radio(
            "训练模式",
            options=["读盘模式（推荐90%）", "故事模式（10%）"],
            index=0,
            help="读盘模式：逐段开放K线，训练实时判断能力。故事模式：完整80根，训练整体归纳能力。"
        )
        st.session_state.mode = "reading" if "读盘" in mode else "story"
        
        st.markdown("---")
        
        if st.session_state.mode == "reading":
            st.markdown("**当前进度**")
            current = st.session_state.current_segment_idx
            total = len(READING_SEGMENTS)
            st.progress(current / total if total > 0 else 0)
            if current < total:
                st.caption(f"K1 → K{READING_SEGMENTS[current]}")
            else:
                st.caption("已完成")
            
            st.markdown("---")
            st.markdown("**修正次数**")
            st.metric("观点修正", st.session_state.correction_count)
        else:
            st.markdown("**故事模式**")
            st.caption("自由讲述今天的故事")
            st.caption("AI会随机提问挑战")
        
        st.markdown("---")
        st.metric("完成复盘次数", st.session_state.practice_count)
        
        st.markdown("---")
        period_map = {"15分钟": "15", "30分钟": "30", "60分钟": "60"}
        period = st.selectbox("周期", list(period_map.keys()), index=1)
        period_value = period_map[period]

        st.markdown("**选择品种**")
        for cat, codes in EXCHANGES.items():
            with st.expander(cat):
                cols = st.columns(2)
                for idx, code in enumerate(codes):
                    name = SYMBOL_NAMES.get(code, code)
                    if cols[idx % 2].button(f"{code}\n{name}", key=f"btn_{code}", use_container_width=True):
                        if load_new_symbol(code, period_value):
                            st.rerun()
        
        st.markdown("---")
        if st.session_state.df is not None:
            if st.button("🔄 重置", use_container_width=True):
                reset_session()
                st.rerun()

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 训练器 V6.0</h3>
        <p><strong>训练目标：在不知道未来的情况下，持续更新对市场控制权的理解。</strong></p>
        
        <h4>📖 读盘模式（推荐90%训练时间）</h4>
        <ul>
            <li>K线逐段开放（K20 → K35 → K50 → K65 → K80）</li>
            <li>每段结束后，判断当前谁控制市场</li>
            <li>AI随机提问挑战你的判断</li>
            <li>新证据出现时，修正之前的观点</li>
            <li><strong>系统记录修正次数</strong>（这是核心训练指标）</li>
        </ul>
        
        <h4>📖 故事模式（10%训练时间）</h4>
        <ul>
            <li>一次性看完80根K线</li>
            <li>自由讲述今天的故事</li>
            <li>AI随机提问挑战</li>
            <li>训练整体结构归纳能力</li>
        </ul>
        
        <p style="color:#6c757d;">💡 读盘模式训练的是"实时更新"，故事模式训练的是"整体归纳"。两者结合才是完整的Brooks训练。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    max_bar = st.session_state.max_bar

    # ========== 读盘模式 ==========
    if st.session_state.mode == "reading":
        current_idx = st.session_state.current_segment_idx
        total_segments = len(READING_SEGMENTS)
        
        if current_idx >= total_segments:
            # 训练完成
            st.success(f"🎉 完成一次读盘训练！观点修正次数：{st.session_state.correction_count}")
            
            # 显示修正记录
            if st.session_state.segment_judgments:
                st.markdown("### 📝 观点演变记录")
                for i, j in enumerate(st.session_state.segment_judgments):
                    seg_end = READING_SEGMENTS[i]
                    st.caption(f"**K{seg_end}时**：{j[:100]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 继续训练", type="primary"):
                    st.session_state.practice_count += 1
                    reset_session()
                    st.rerun()
            with col2:
                if st.button("🎲 换一个品种"):
                    all_codes = []
                    for codes in EXCHANGES.values():
                        all_codes.extend(codes)
                    new_code = random.choice(all_codes)
                    load_new_symbol(new_code, "30")
                    st.rerun()
            return
        
        # 当前段
        current_end = READING_SEGMENTS[current_idx]
        prev_end = READING_SEGMENTS[current_idx - 1] if current_idx > 0 else 0
        
        # 显示图表（只显示到当前段）
        st.plotly_chart(build_partial_chart(df, max_bar, current_end), use_container_width=True)
        
        st.markdown(f"### 🎯 K{prev_end+1 if prev_end > 0 else 1} → K{current_end}")
        st.markdown("**请判断：当前谁控制市场？**")
        st.caption("（多头/空头/震荡？控制权是否发生了变化？）")
        
        # 对话区域
        conv = st.session_state.conversations
        for msg in conv:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        user_input = st.chat_input("例如：目前空头控制，K12-K18连续下跌，反弹无力...")
        
        if user_input:
            conv.append({"role": "user", "content": user_input})
            
            # 检查是否修正了观点
            if len(st.session_state.segment_judgments) > 0:
                if check_correction(st.session_state.segment_judgments[-1], user_input):
                    st.session_state.correction_count += 1
                    st.toast(f"✅ 观点修正！总修正次数：{st.session_state.correction_count}")
            
            # 保存当前判断
            if current_idx == len(st.session_state.segment_judgments):
                st.session_state.segment_judgments.append(user_input)
            
            with st.spinner("AI思考中..."):
                ai_response = call_reading_mode_challenge(user_input, prev_end+1, current_end)
            
            conv.append({"role": "assistant", "content": ai_response})
            
            # 进入下一段
            st.session_state.current_segment_idx += 1
            st.rerun()
    
    # ========== 故事模式 ==========
    else:
        # 显示完整图表
        show_labels = st.checkbox("显示K线编号", value=False)
        st.plotly_chart(build_chart(df, max_bar, show_labels), use_container_width=True)
        
        st.markdown("### 📖 请讲述今天的故事")
        st.caption("自由讲述，不限格式。AI会随机提问挑战。")
        
        # 对话区域
        conv = st.session_state.conversations
        for msg in conv:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        if st.session_state.session_complete:
            st.success("🎉 故事已充分讨论！")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 继续训练", type="primary"):
                    st.session_state.practice_count += 1
                    reset_session()
                    st.rerun()
            with col2:
                if st.button("🎲 换一个品种"):
                    all_codes = []
                    for codes in EXCHANGES.values():
                        all_codes.extend(codes)
                    new_code = random.choice(all_codes)
                    load_new_symbol(new_code, "30")
                    st.rerun()
            return
        
        user_input = st.chat_input("例如：开盘后空头控制，K15-K25形成下降通道，K28出现双底...")
        
        if user_input:
            conv.append({"role": "user", "content": user_input})
            
            with st.spinner("AI思考中..."):
                ai_response = call_story_mode_challenge(user_input)
            
            conv.append({"role": "assistant", "content": ai_response})
            
            # 简单判断：如果有3轮以上对话，视为完成
            if len([m for m in conv if m["role"] == "user"]) >= 3:
                st.session_state.session_complete = True
            
            st.rerun()


if __name__ == "__main__":
    main()
