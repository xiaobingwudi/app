"""
Al Brooks 日内机会寻找训练器 V7.0
核心目标：训练用户不断重复四句话——
1. 我观察到了什么？
2. 我如何解释这些证据？
3. 我的置信度是多少？
4. 什么新证据会让我改变看法？

三个阶段：
1. 事件驱动读盘（95%）：事件触发开放，记录观察、解释、置信度、反证条件
2. 观点演化复盘（自动）：生成观点演化图
3. Brooks总结（5%）：最后才问"今天是什么样的一天"
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
from datetime import datetime

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

# 事件类型定义（触发暂停的关键事件）
KEY_EVENTS = [
    "连续3根同向K线",
    "大K线（实体>80%波幅）",
    "突破前高/前低",
    "失败突破",
    "双顶/双底形成",
    "趋势线被刺穿",
    "外包/内包K线",
    "成交量异常放大",
    "窄幅横盘（连续3根小K线）",
    "反转K线（长影线）"
]

# ==================== AI 提示词 ====================
OBSERVATION_SYSTEM = """你是 Al Brooks 价格行为教练。

【当前模式】事件驱动读盘
用户刚刚看到一段新的K线，触发了关键事件。

你的任务是：引导用户完成四步思考，不要替他下结论。

【输出格式】
请依次问这四个问题（每问一行）：
1. 你观察到了什么？（请引用具体K线特征）
2. 你如何解释这些证据？（市场在告诉你什么？）
3. 你现在的置信度是多少？（0-100%）
4. 什么新证据会让你改变这个看法？

【事件信息】
触发事件：{event}
K线范围：K{start} ~ K{end}
"""

CHALLENGE_SYSTEM = """你是 Al Brooks 价格行为教练。

用户已经给出了他的观察和解释。你的任务是：挑战他的思考，而不是判断对错。

【输出格式】
请从以下角度选1-2个提问：
- 有没有相反的证据？
- 这个解释还有其他可能性吗？
- 如果市场现在反转，你会怎么调整？
- 你的置信度基于哪些具体K线？

控制在80字以内。

【用户当前回答】
{user_answer}
"""

SUMMARY_SYSTEM = """你是 Al Brooks 价格行为教练。

用户完成了整个读盘训练。请帮他总结今天的故事。

【输出格式】
用3-5句话总结：
1. 今天故事的主要演变过程
2. 用户观点修正的关键节点
3. 最终的市场理解

【观点演化记录】
{evolution_log}
"""


def call_observation_guide(event, start, end):
    """引导用户完成四步思考"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "请回答：\n1. 你观察到了什么？\n2. 你如何解释？\n3. 置信度？\n4. 什么会让你改变看法？"
    
    system = OBSERVATION_SYSTEM.format(event=event, start=start, end=end)
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"请思考这四个问题：\n1. 观察到了什么？\n2. 如何解释？\n3. 置信度？\n4. 什么会让你改变看法？\n(API: {str(e)[:50]})"


def call_challenge(user_answer):
    """挑战用户的思考"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "有没有相反的证据？还有其他可能性吗？"
    
    system = CHALLENGE_SYSTEM.format(user_answer=user_answer[:500])
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return "有没有相反的证据？还有其他解释吗？"


def call_summary(evolution_log):
    """生成最终总结"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key or not evolution_log:
        return "完成训练后，系统将自动生成今天的故事总结。"
    
    system = SUMMARY_SYSTEM.format(evolution_log=evolution_log[:1500])
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=300
        )
        return resp.choices[0].message.content
    except Exception as e:
        return "今天的故事总结将在这里显示。"


# ==================== 事件检测 ====================
def detect_events(df, start_idx, end_idx):
    """
    检测K线范围内是否发生关键事件
    返回：触发的事件名称，如果没有则返回None
    """
    if end_idx - start_idx < 3:
        return None
    
    sub = df.iloc[start_idx:end_idx+1]
    n = len(sub)
    
    if n < 3:
        return None
    
    # 1. 连续3根同向K线
    last_3 = sub.iloc[-3:]
    bull_count = sum(1 for _, row in last_3.iterrows() if row["close"] >= row["open"])
    if bull_count == 3:
        return "连续3根多头K线（强势上涨）"
    if bull_count == 0:
        return "连续3根空头K线（强势下跌）"
    
    # 2. 大K线（实体>80%波幅）
    last = sub.iloc[-1]
    body = abs(last["close"] - last["open"])
    total = last["high"] - last["low"]
    if total > 0 and body / total > 0.8:
        direction = "阳线" if last["close"] >= last["open"] else "阴线"
        return f"大{direction}线（实体占比{body/total*100:.0f}%）"
    
    # 3. 突破前高/前低
    if n >= 5:
        recent_high = sub.iloc[:-1]["high"].max()
        recent_low = sub.iloc[:-1]["low"].min()
        if last["high"] > recent_high:
            return f"突破近期高点（K{start_idx+end_idx} > 前{min(5, n-1)}根高点）"
        if last["low"] < recent_low:
            return f"跌破近期低点（K{start_idx+end_idx} < 前{min(5, n-1)}根低点）"
    
    # 4. 内包/外包
    if n >= 2:
        prev = sub.iloc[-2]
        curr = sub.iloc[-1]
        if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
            return f"内包线（K{start_idx+end_idx}完全在K{start_idx+end_idx-1}内部）"
        if curr["high"] > prev["high"] and curr["low"] < prev["low"]:
            return f"外包线（K{start_idx+end_idx}完全包含K{start_idx+end_idx-1}）"
    
    # 5. 窄幅横盘
    if n >= 3:
        last_3_range = sub.iloc[-3:]["high"].max() - sub.iloc[-3:]["low"].min()
        avg_range = sub.iloc[:]["high"].max() - sub.iloc[:]["low"].min() if n > 3 else last_3_range
        if avg_range > 0 and last_3_range / avg_range < 0.3:
            return "连续3根小K线（窄幅横盘）"
    
    return None


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
def build_chart(df, max_bar, current_end=None):
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

    # 每10根显示一次编号
    for idx, bar_num in enumerate(bar_numbers):
        if bar_num % 10 == 0:
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

    # 标记当前进度
    if current_end and current_end <= n_bars:
        fig.add_vline(
            x=current_end - 1, line_dash="dash",
            line_color="#ff9800", line_width=2, opacity=0.8
        )
        fig.add_annotation(
            x=current_end - 1, y=plot_df.iloc[min(current_end-1, n_bars-1)]["high"],
            text=f"← 当前位置", showarrow=False,
            font=dict(size=10, color="#ff9800"),
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
        "current_position": 20,  # 当前K线位置（从20开始）
        "event_log": [],  # 事件日志
        "observation_log": [],  # 用户观察日志（观点演化）
        "conversations": [],  # 当前对话
        "session_complete": False,
        "practice_count": 0,
        "waiting_for_observation": False,
        "current_event": None,
        "current_event_start": 0,
        "current_event_end": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.current_position = 20
    st.session_state.event_log = []
    st.session_state.observation_log = []
    st.session_state.conversations = []
    st.session_state.session_complete = False
    st.session_state.waiting_for_observation = False
    st.session_state.current_event = None
    st.session_state.current_event_start = 0
    st.session_state.current_event_end = 0


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


def advance_to_next_event(df, current_pos, max_bar):
    """推进到下一个事件位置"""
    for new_pos in range(current_pos + 1, max_bar + 1):
        event = detect_events(df, max(0, new_pos - 10), new_pos)
        if event:
            return new_pos, event
    return max_bar, None


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 训练器 V7.0", layout="wide")
    
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
    .observation-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 训练器 V7.0")
        st.markdown("---")
        st.markdown("**核心理念**")
        st.caption("不断重复四句话：")
        st.caption("① 我观察到了什么？")
        st.caption("② 我如何解释这些证据？")
        st.caption("③ 我的置信度是多少？")
        st.caption("④ 什么会让我改变看法？")
        
        st.markdown("---")
        st.metric("完成复盘次数", st.session_state.practice_count)
        
        if st.session_state.observation_log:
            st.markdown("---")
            st.markdown("**观点演化**")
            for log in st.session_state.observation_log[-3:]:
                st.caption(f"K{log['end']}: {log.get('interpretation', '')[:50]}...")
        
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
            if st.button("🔄 重置训练", use_container_width=True):
                reset_session()
                st.rerun()

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 训练器 V7.0</h3>
        <p><strong>训练目标：不断重复四句话——</strong></p>
        <ol>
            <li><strong>我观察到了什么？</strong>（描述具体的K线特征）</li>
            <li><strong>我如何解释这些证据？</strong>（市场在告诉你什么）</li>
            <li><strong>我的置信度是多少？</strong>（0-100%）</li>
            <li><strong>什么新证据会让我改变看法？</strong>（反证条件）</li>
        </ol>
        
        <h4>📖 训练流程</h4>
        <ul>
            <li>K线自动推进，遇到关键事件时暂停</li>
            <li>记录你的观察、解释、置信度、反证条件</li>
            <li>AI挑战你的思考（不是判断对错）</li>
            <li>完成后自动生成观点演化图</li>
        </ul>
        
        <p style="color:#6c757d;">💡 训练的是"如何思考"，而不是"答案是什么"。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    max_bar = st.session_state.max_bar

    # ========== 训练完成 ==========
    if st.session_state.session_complete:
        st.success("🎉 完成一次读盘训练！")
        
        # 生成观点演化图
        if st.session_state.observation_log:
            st.markdown("### 📈 观点演化记录")
            for log in st.session_state.observation_log:
                with st.expander(f"📍 K{log['end']} - {log.get('event', '事件')}", expanded=False):
                    st.markdown(f"**观察：** {log.get('observation', '')}")
                    st.markdown(f"**解释：** {log.get('interpretation', '')}")
                    st.markdown(f"**置信度：** {log.get('confidence', '')}%")
                    st.markdown(f"**反证条件：** {log.get('counter', '')}")
        
        # 生成最终总结
        with st.spinner("生成总结..."):
            evolution_text = "\n".join([
                f"K{log['end']}: {log.get('interpretation', '')} (置信度{log.get('confidence', '')}%)"
                for log in st.session_state.observation_log
            ])
            summary = call_summary(evolution_text)
            st.markdown("### 📖 今天的故事")
            st.info(summary)
        
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

    # ========== 事件驱动读盘 ==========
    current_pos = st.session_state.current_position
    
    # 显示图表
    st.plotly_chart(build_chart(df, max_bar, current_pos), use_container_width=True)
    
    # 检查是否需要触发新事件
    if not st.session_state.waiting_for_observation:
        # 推进到下一个事件
        new_pos, event = advance_to_next_event(df, current_pos, max_bar)
        
        if new_pos >= max_bar or event is None:
            # 训练完成
            st.session_state.session_complete = True
            st.rerun()
        else:
            # 触发事件，等待用户观察
            st.session_state.waiting_for_observation = True
            st.session_state.current_event = event
            st.session_state.current_event_start = current_pos
            st.session_state.current_event_end = new_pos
            st.session_state.current_position = new_pos
            st.rerun()
    
    # ========== 等待用户观察 ==========
    else:
        event = st.session_state.current_event
        event_start = st.session_state.current_event_start
        event_end = st.session_state.current_event_end
        
        st.markdown(f"### 🚨 关键事件触发")
        st.markdown(f"**K{event_start+1 if event_start > 0 else 1} → K{event_end}**")
        st.markdown(f"**事件：** {event}")
        
        # 显示已有的对话
        conv = st.session_state.conversations
        for msg in conv:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # 引导用户完成四步思考
        if len(conv) == 0:
            # 第一次：引导用户回答四个问题
            guide = call_observation_guide(event, event_start+1, event_end)
            with st.chat_message("assistant"):
                st.markdown(guide)
            conv.append({"role": "assistant", "content": guide})
        
        # 等待用户输入
        user_input = st.chat_input("请回答上述问题...")
        
        if user_input:
            conv.append({"role": "user", "content": user_input})
            
            # 尝试解析用户的四个回答
            observation = user_input[:200] if "观察" in user_input else user_input[:100]
            interpretation = ""
            confidence = ""
            counter = ""
            
            # 简单解析
            if "观察" in user_input:
                parts = user_input.split("观察")
                observation = "观察" + parts[1][:150] if len(parts) > 1 else user_input[:150]
            if "解释" in user_input:
                parts = user_input.split("解释")
                interpretation = parts[1][:150] if len(parts) > 1 else ""
            if "置信度" in user_input:
                import re
                match = re.search(r'置信度[：:]\s*(\d+)', user_input)
                if match:
                    confidence = match.group(1)
            if "改变" in user_input or "反证" in user_input:
                counter = user_input[-150:]
            
            # 记录到观点演化日志
            st.session_state.observation_log.append({
                "end": event_end,
                "event": event,
                "observation": observation,
                "interpretation": interpretation,
                "confidence": confidence,
                "counter": counter
            })
            
            # AI挑战用户的思考
            with st.spinner("AI思考中..."):
                challenge = call_challenge(user_input)
            
            conv.append({"role": "assistant", "content": challenge})
            
            # 继续下一段
            st.session_state.waiting_for_observation = False
            st.session_state.conversations = []
            st.rerun()


if __name__ == "__main__":
    main()
