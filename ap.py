"""
Al Brooks 日内机会寻找训练器 V8.0
核心目标：在信息不完整的情况下，根据不断出现的新证据，持续更新自己对市场的理解

训练流程：
1. 逐K/每N根推进（用户可调节：5/10/20根）
2. 每段结束后，记录：我看到、我认为、置信度、什么会改变看法
3. 系统自动记录观点演化时间轴（重点记录修正）
4. 训练结束后，用户自己总结，AI只挑战不下结论
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

# 离散置信度选项
CONFIDENCE_OPTIONS = {
    "30%": "低置信（几乎不知道，两边都合理）",
    "50%": "中置信（两边都行，轻微倾向）",
    "70%": "高置信（有一定把握，但可能错）",
    "90%": "极高置信（市场给了非常强证据）"
}

# 推进步长选项
STEP_OPTIONS = [5, 10, 20]

# ==================== AI 提示词 ====================
CHALLENGE_SYSTEM = """你是 Al Brooks 价格行为教练。

【你的角色】
你不是老师，你是陪练。
你的任务不是判断对错，而是挑战用户的思考过程。

【用户刚刚完成了第{step_num}段的分析】
他看到：{observation}
他认为：{interpretation}
置信度：{confidence}
反证条件：{counter}

【输出格式】
请从以下角度选1-2个提问，不要下结论，不要替用户分析：
- 有没有相反的证据？
- 这个解释还有其他可能性吗？
- 你的置信度主要基于哪几根K线？
- 如果市场现在反转，你会怎么调整？

控制在80字以内。
"""

SUMMARY_CHALLENGE_SYSTEM = """你是 Al Brooks 价格行为教练。

用户刚刚完成了整个读盘训练，正在自己总结今天的故事。

【用户的总结】
{user_summary}

【观点演化记录】
{evolution_log}

【输出格式】
请只做两件事：
1. 追问1-2个帮助用户发现遗漏的问题
2. 不要替用户总结，不要给出标准答案

控制在100字以内。
"""


def call_challenge(observation, interpretation, confidence, counter, step_num):
    """挑战用户的思考"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "有没有相反的证据？还有其他可能性吗？"
    
    system = CHALLENGE_SYSTEM.format(
        step_num=step_num,
        observation=observation[:200],
        interpretation=interpretation[:200],
        confidence=confidence,
        counter=counter[:200]
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"有没有相反的证据？(API: {str(e)[:50]})"


def call_summary_challenge(user_summary, evolution_log):
    """挑战用户的总结"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "有没有遗漏的关键转折点？"
    
    system = SUMMARY_CHALLENGE_SYSTEM.format(
        user_summary=user_summary[:500],
        evolution_log=evolution_log[:1000]
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return "有没有遗漏的关键证据？"


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

    # 标记观点修正点
    if st.session_state.observation_log:
        for log in st.session_state.observation_log:
            if log.get("corrected", False):
                end_pos = log["end"]
                if end_pos <= n_bars:
                    fig.add_annotation(
                        x=end_pos - 1, y=plot_df.iloc[min(end_pos-1, n_bars-1)]["low"],
                        text="✏️ 修正", showarrow=False,
                        font=dict(size=10, color="#4caf50"),
                        yshift=-20
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
        "step_size": 10,  # 每段推进的K线数量
        "current_position": 10,  # 当前K线位置（从10开始）
        "observation_log": [],  # 观点演化日志
        "session_complete": False,
        "practice_count": 0,
        "current_step_num": 0,
        "waiting_for_observation": False,
        "temp_observation": "",
        "temp_interpretation": "",
        "temp_confidence": "50%",
        "temp_counter": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.current_position = st.session_state.step_size
    st.session_state.observation_log = []
    st.session_state.session_complete = False
    st.session_state.current_step_num = 0
    st.session_state.waiting_for_observation = False
    st.session_state.temp_observation = ""
    st.session_state.temp_interpretation = ""
    st.session_state.temp_confidence = "50%"
    st.session_state.temp_counter = ""


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


def check_correction(prev_log, current_observation, current_interpretation):
    """检查是否发生了观点修正"""
    if not prev_log:
        return False
    
    # 简单判断：置信度变化超过30% 或 方向性描述改变
    prev_conf = int(prev_log.get("confidence", "50%").replace("%", ""))
    current_conf = int(current_confidence.replace("%", "")) if isinstance(current_confidence, str) else 50
    
    if abs(current_conf - prev_conf) >= 30:
        return True
    
    # 检查方向性关键词变化
    prev_text = prev_log.get("interpretation", "")
    current_text = current_interpretation
    
    direction_keywords = ["多头", "空头", "震荡", "买方", "卖方", "上涨", "下跌"]
    prev_dir = None
    current_dir = None
    
    for kw in direction_keywords:
        if kw in prev_text:
            prev_dir = kw
        if kw in current_text:
            current_dir = kw
    
    return prev_dir != current_dir and prev_dir is not None and current_dir is not None


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 训练器 V8.0", layout="wide")
    
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
    .correction-badge {
        background-color: #4caf50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 训练器 V8.0")
        st.markdown("---")
        st.markdown("**核心理念**")
        st.caption("在信息不完整的情况下，根据不断出现的新证据，持续更新自己对市场的理解")
        
        st.markdown("---")
        st.markdown("**训练参数**")
        
        # 推进步长选择
        step_size = st.select_slider(
            "每段推进K线数",
            options=STEP_OPTIONS,
            value=st.session_state.step_size,
            help="较小的步长训练更细致，较大的步长训练更快"
        )
        if step_size != st.session_state.step_size:
            st.session_state.step_size = step_size
            reset_session()
            st.rerun()
        
        st.markdown("---")
        st.metric("完成复盘次数", st.session_state.practice_count)
        
        if st.session_state.observation_log:
            st.markdown("---")
            st.markdown("**📈 观点演化**")
            for log in st.session_state.observation_log[-5:]:
                conf = log.get("confidence", "?")
                corrected = " ✏️" if log.get("corrected") else ""
                st.caption(f"K{log['end']}: {conf}{corrected}")
        
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
        <h3>Al Brooks 训练器 V8.0</h3>
        <p><strong>训练目标：在信息不完整的情况下，不断更新对市场的理解。</strong></p>
        
        <h4>📖 训练流程</h4>
        <ol>
            <li>设置每段推进的K线数量（5/10/20根）</li>
            <li>每段结束后，记录四个核心问题：
                <ul>
                    <li><strong>我看到：</strong>具体的K线特征</li>
                    <li><strong>我认为：</strong>市场在告诉我什么</li>
                    <li><strong>置信度：</strong>30%/50%/70%/90%</li>
                    <li><strong>什么会改变看法：</strong>反证条件</li>
                </ul>
            </li>
            <li>AI挑战你的思考（不是判断对错）</li>
            <li>系统自动记录观点演化，标记修正点</li>
            <li>全部结束后，自己总结今天的故事</li>
        </ol>
        
        <p style="color:#6c757d;">💡 核心能力：知道什么时候自己可能是错的，并毫不犹豫地修正。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    max_bar = st.session_state.max_bar
    step_size = st.session_state.step_size
    current_pos = st.session_state.current_position

    # ========== 训练完成 ==========
    if st.session_state.session_complete:
        st.success("🎉 完成一次读盘训练！")
        
        # 显示观点演化时间轴
        if st.session_state.observation_log:
            st.markdown("### 📈 观点演化时间轴")
            
            for log in st.session_state.observation_log:
                corrected = " ✏️ **（修正点）**" if log.get("corrected") else ""
                with st.expander(f"📍 K{log['end']} - 置信度 {log.get('confidence', '?')}{corrected}", expanded=False):
                    st.markdown(f"**我看到：** {log.get('observation', '')}")
                    st.markdown(f"**我认为：** {log.get('interpretation', '')}")
                    st.markdown(f"**反证条件：** {log.get('counter', '')}")
                    if log.get("corrected"):
                        st.caption("✓ 此处发生了观点修正")
            
            # 修正统计
            correction_count = sum(1 for log in st.session_state.observation_log if log.get("corrected"))
            st.info(f"📊 本次训练共发生 **{correction_count}** 次观点修正")
        
        # ========== 阶段3：用户自己总结 ==========
        st.markdown("---")
        st.markdown("### 📝 请总结今天的故事")
        st.markdown("请回答以下问题：")
        st.markdown("1. **今天市场试图做什么？**")
        st.markdown("2. **它成功了吗？**")
        st.markdown("3. **最大的意外是什么？**")
        st.markdown("4. **你什么时候改变了看法？为什么？**")
        st.markdown("5. **下次遇到类似情况，你会注意什么？**")
        
        # 显示用户总结历史
        if "user_summary" in st.session_state and st.session_state.user_summary:
            with st.expander("你的总结", expanded=False):
                st.markdown(st.session_state.user_summary)
        
        # 总结输入
        user_summary = st.text_area("你的总结", height=200, placeholder="请用自然语言描述今天的故事...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("提交总结", type="primary"):
                if user_summary:
                    st.session_state.user_summary = user_summary
                    
                    # 生成观点演化文本
                    evolution_text = "\n".join([
                        f"K{log['end']}: {log.get('interpretation', '')} (置信度{log.get('confidence', '')}%)"
                        for log in st.session_state.observation_log
                    ])
                    
                    with st.spinner("AI分析中..."):
                        challenge = call_summary_challenge(user_summary, evolution_text)
                    
                    st.session_state.summary_challenge = challenge
                    st.rerun()
        
        # 显示AI挑战
        if "summary_challenge" in st.session_state:
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 AI挑战**\n\n{st.session_state.summary_challenge}")
        
        # 继续训练按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 继续训练", type="secondary"):
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

    # ========== 阶段1：逐段推进 ==========
    # 显示图表
    st.plotly_chart(build_chart(df, max_bar, current_pos), use_container_width=True)
    
    # 显示进度
    progress = current_pos / max_bar
    st.progress(progress, text=f"进度：K{current_pos} / K{max_bar}")
    
    # 计算当前段范围
    prev_pos = current_pos - step_size
    if prev_pos < 0:
        prev_pos = 0
    
    # ========== 等待用户观察 ==========
    if st.session_state.waiting_for_observation:
        st.markdown(f"### 🔍 K{prev_pos+1 if prev_pos > 0 else 1} → K{current_pos}")
        st.markdown("请记录你的观察：")
        
        # 观察输入
        observation = st.text_area(
            "① 我看到了什么？",
            value=st.session_state.temp_observation,
            placeholder="例如：K10-K15连续下跌，K16出现一根大阳线但后续没有跟进...",
            height=80,
            key="obs_input"
        )
        
        # 解释输入
        interpretation = st.text_area(
            "② 我认为市场在告诉我什么？",
            value=st.session_state.temp_interpretation,
            placeholder="例如：空头仍然控制，但多头开始尝试反扑...",
            height=80,
            key="int_input"
        )
        
        # 置信度选择
        confidence = st.radio(
            "③ 我的置信度",
            options=list(CONFIDENCE_OPTIONS.keys()),
            format_func=lambda x: f"{x} - {CONFIDENCE_OPTIONS[x]}",
            horizontal=True,
            index=list(CONFIDENCE_OPTIONS.keys()).index(st.session_state.temp_confidence)
        )
        
        # 反证条件输入
        counter = st.text_area(
            "④ 什么会让我改变看法？",
            value=st.session_state.temp_counter,
            placeholder="例如：如果K17收盘价突破K15的低点，我会重新评估空头控制...",
            height=80,
            key="cnt_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 提交", type="primary"):
                if observation and interpretation:
                    # 检查是否修正
                    prev_log = st.session_state.observation_log[-1] if st.session_state.observation_log else None
                    corrected = check_correction(prev_log, observation, interpretation)
                    
                    # 记录到日志
                    st.session_state.observation_log.append({
                        "end": current_pos,
                        "observation": observation,
                        "interpretation": interpretation,
                        "confidence": confidence,
                        "counter": counter,
                        "corrected": corrected
                    })
                    
                    # AI挑战
                    with st.spinner("AI思考中..."):
                        challenge = call_challenge(observation, interpretation, confidence, counter, st.session_state.current_step_num + 1)
                    
                    st.session_state.last_challenge = challenge
                    st.session_state.waiting_for_observation = False
                    st.session_state.current_step_num += 1
                    
                    # 清空临时变量
                    st.session_state.temp_observation = ""
                    st.session_state.temp_interpretation = ""
                    st.session_state.temp_confidence = "50%"
                    st.session_state.temp_counter = ""
                    
                    st.rerun()
                else:
                    st.warning("请至少填写"观察"和"我认为"")
        
        with col2:
            if st.button("↩️ 返回修改"):
                # 保存临时值
                st.session_state.temp_observation = observation
                st.session_state.temp_interpretation = interpretation
                st.session_state.temp_confidence = confidence
                st.session_state.temp_counter = counter
                st.rerun()
        
        # 显示AI挑战
        if "last_challenge" in st.session_state:
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 AI挑战**\n\n{st.session_state.last_challenge}")
    
    # ========== 推进到下一段 ==========
    else:
        if current_pos >= max_bar:
            st.session_state.session_complete = True
            st.rerun()
        else:
            next_pos = min(current_pos + step_size, max_bar)
            st.info(f"📌 当前分析到 K{current_pos}，点击继续查看下一段")
            
            if st.button("➡️ 继续下一段", type="primary"):
                st.session_state.current_position = next_pos
                st.session_state.waiting_for_observation = True
                st.rerun()


if __name__ == "__main__":
    main()
