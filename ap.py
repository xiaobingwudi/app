"""
Al Brooks 结构训练器 V19
========================
基于 V18 布局，只改交互逻辑：
1. chart_df 加载后固定，AI 点评不改变图表
2. 每个技能最多 2 轮输入，完成后输入框禁用
3. 点击"下一根"或"随机跳转"才移动图表
4. 修复数据加载空 DataFrame 错误
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from datetime import datetime, timedelta
import random
import json

# ====== 页面配置 ======
st.set_page_config(page_title="Al Brooks 结构训练器 V19", layout="wide")
st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 280px !important; }
    section[data-testid="stSidebar"] > div { padding: 0.3rem 0.6rem !important; }
    section[data-testid="stSidebar"] .stButton button {
        padding: 0.1rem 0.3rem !important; font-size: 0.75rem !important;
        min-height: 0 !important; height: 28px !important; line-height: 1 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stNumberInput {
        margin-bottom: 0.2rem !important;
    }
    section[data-testid="stSidebar"] label { font-size: 0.75rem !important; margin-bottom: 0 !important; }
    section[data-testid="stSidebar"] .row-widget { margin: 0 !important; }
    section[data-testid="stSidebar"] hr { margin: 0.3rem 0 !important; }
    .stPlotlyChart { width: 100%; }
    div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

# ====== Session State 初始化 ======
defaults = {
    "chart_df": None,
    "display_start": 0,
    "display_count": 60,
    "current_skill": None,
    "skill_rounds": {},
    "coach_dialogue": [],
    "reading_profile": {
        "经常忽略背景": 0,
        "经常忽略控制权": 0,
        "喜欢提前预测": 0,
        "容易贴标签": 0,
        "忽略细节行为": 0,
    },
    "data_source": {},
    "total_bars": 0,
    "reload_data_flag": False,
    "random_jump_flag": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====== 品种与周期 ======
SYMBOLS = {
    "螺纹": "rb", "铁矿": "i", "沪铜": "cu", "原油": "sc",
    "豆粕": "m", "甲醇": "MA", "PTA": "TA", "棕榈": "p",
}
PERIODS = {"15分钟": "15", "30分钟": "30", "60分钟": "60", "日线": "D"}
EXCHANGES = {
    "rb": "shfe", "i": "dce", "cu": "shfe", "sc": "shfe",
    "m": "dce", "MA": "czce", "TA": "czce", "p": "dce",
}
EXCHANGE_NAMES = {
    "shfe": "上海期货交易所", "dce": "大连商品交易所", "czce": "郑州商品交易所",
}

SKILL_NAMES = {
    "skill_1": "背景阅读",
    "skill_2": "控制权识别",
    "skill_3": "推进质量",
    "skill_4": "回调vs转换",
    "skill_5": "市场接受度",
}

SKILL_PROMPTS = {
    "skill_1": """## 技能：背景阅读
你的任务是分析当前市场背景。关注：
- 趋势方向（上升/下降/震荡）
- 高低点序列（HH/HL/LH/LL）
- 通道/节奏（陡峭/平缓/无）

【禁止词汇】
买、卖、做多、做空、进场、止损、开仓、平仓

【允许讨论】
趋势、区间、高低点、通道、节奏、动能、延续

请对用户的观察进行点评和补充。""",

    "skill_2": """## 技能：控制权识别
你的任务是分析最近3-5根K线，判断谁在控制市场。

【禁止词汇】
趋势、结构、预测、未来、方向、做多、做空、买、卖

【允许讨论】
最近3-5根K线、推动、压制、买方力量、卖方力量、谁在主导、突破、失败

请对用户的观察进行点评和补充。""",

    "skill_3": """## 技能：推进质量
你的任务是分析K线的推进质量。关注：
- 实体大小（BodyRatio）
- 收盘位置（CloseLocation）
- 影线长度
- 与前一根重叠程度（OverlapRatio）
- 成交量确认（VolRatio）

【禁止词汇】
趋势、方向、买卖、进场、止损、做多、做空

【允许讨论】
实体、重叠、影线、收盘位置、成交量、跟进、失败、突破质量

请对用户的观察进行点评和补充。""",

    "skill_4": """## 技能：回调vs转换
你的任务是判断当前走势是回调还是转换。

【禁止词汇】
大趋势、方向、买卖、进场、止损

【允许讨论】
回调深度、反弹高度、重叠程度、突破确认、失败信号、跟随、反转

请对用户的观察进行点评和补充。""",

    "skill_5": """## 技能：市场接受度
你的任务是分析市场对价格行为的接受程度。

【禁止词汇】
趋势、买卖、做多、做空、进场、止损、开仓

【允许讨论】
接受、拒绝、测试、突破失败、跟随、跟进力量、拒绝信号、失败突破

请对用户的观察进行点评和补充。""",
}

MAX_ROUNDS_PER_SKILL = 2

# ====== 数据加载 ======
def load_data(symbol_key, period_key):
    """加载品种数据"""
    try:
        code = SYMBOLS[symbol_key]
        period = PERIODS[period_key]
        exchange = EXCHANGES[code]
        
        if period == "D":
            raw = ak.futures_zh_daily_sina(symbol=f"{exchange}{code}")
        else:
            raw = ak.futures_zh_minute_sina(symbol=f"{exchange}{code}", period=period)
        
        if raw is None or raw.empty:
            return None
        
        df = raw.copy()
        
        # 列名映射：逐列判断，避免列数不匹配错误
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl in ("date", "日期", "交易日", "day", "datetime", "时间"):
                col_map[c] = "time"
            elif cl in ("open", "开盘", "o"):
                col_map[c] = "o"
            elif cl in ("high", "最高", "h"):
                col_map[c] = "h"
            elif cl in ("low", "最低", "l"):
                col_map[c] = "l"
            elif cl in ("close", "收盘", "c"):
                col_map[c] = "c"
            elif cl in ("volume", "成交量", "vol", "v", "持仓量"):
                col_map[c] = "v"
            else:
                col_map[c] = c  # 保持原名
        
        df = df.rename(columns=col_map)
        
        # 检查必要列
        for col in ["o", "h", "l", "c"]:
            if col not in df.columns:
                return None
        
        if "time" not in df.columns:
            df["time"] = range(len(df))
        if "v" not in df.columns:
            df["v"] = 0
        
        # 数值化
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        df = df.sort_values("time").reset_index(drop=True)
        
        if len(df) < 10:
            return None
        
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None

def compute_bar_stats(df, idx):
    if idx < 0 or idx >= len(df):
        return ""
    bar = df.iloc[idx]
    o, h, l, c, v = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"]), float(bar["v"])
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return f"K{idx}: O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} Vol={v:.0f}"
    
    body_ratio = body / rng
    close_location = ((c - l) / rng) * 100 if c >= o else ((l - c) / rng) * 100
    upper_shadow = (h - max(o, c)) / rng
    lower_shadow = (min(o, c) - l) / rng
    
    overlap_ratio = 0.0
    if idx > 0:
        prev = df.iloc[idx - 1]
        overlap = max(0, min(h, float(prev["h"])) - max(l, float(prev["l"])))
        overlap_ratio = overlap / rng if rng > 0 else 0
    
    vol_ratio = 1.0
    if idx >= 5:
        vols = [float(df.iloc[i]["v"]) for i in range(idx - 5, idx)]
        avg_v = sum(vols) / len(vols)
        vol_ratio = v / avg_v if avg_v > 0 else 1.0
    
    return (
        f"K{idx}: O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} "
        f"BodyRatio={body_ratio:.0%} CloseLocation={close_location:.0f}% "
        f"UpperShadow={upper_shadow:.0%} LowerShadow={lower_shadow:.0%} "
        f"Vol={v:.0f} VolRatio={vol_ratio:.2f} OverlapRatio={overlap_ratio:.2f}"
    )

def prepare_market_msg(df, display_start, display_count):
    start = display_start
    end = min(display_start + display_count, len(df))
    return "\n".join([compute_bar_stats(df, i) for i in range(start, end)])

# ====== AI 教练 ======
def ask_coach(skill_key, user_input, market_data_str, reading_profile):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return "错误：API Key 未配置。请在 Streamlit Cloud 后台设置 OPENAI_API_KEY。"
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    profile_str = "\n".join([f"- {k}: {v}次" for k, v in reading_profile.items() if v > 0])
    profile_hint = f"\n\n【用户阅读画像（薄弱点追踪）】\n{profile_str}\n请根据画像针对性地训练用户。" if profile_str else ""
    
    system_prompt = f"""你是Al Brooks价格行为训练教练。你的任务是引导用户观察市场行为，而不是直接给答案。

{SKILL_PROMPTS.get(skill_key, '')}

【规则】
1. 先点评用户的观察（是否正确、遗漏了什么）
2. 再补充你的分析，但不要提前告诉用户结论
3. 用提问引导用户自己发现关键特征
4. 每轮点评控制在200字以内
5. 不使用"大阳线""放量""缩量"等传统标签
6. 使用客观数据（BodyRatio、CloseLocation等）
{profile_hint}

【当前市场数据（系统自动生成，非用户发言）】
{market_data_str}

请基于以上数据对用户的观察进行点评。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        reply = resp.choices[0].message.content
        try:
            updates = json.loads(reply)
            if isinstance(updates, dict) and "profile_updates" in updates:
                for k, v in updates["profile_updates"].items():
                    if k in st.session_state.reading_profile:
                        st.session_state.reading_profile[k] += v
        except:
            pass
        return reply
    except Exception as e:
        return f"AI调用失败: {str(e)}"

# ====== 绘图 ======
def plot_chart(df, display_start, display_count):
    end = min(display_start + display_count, len(df))
    plot_df = df.iloc[display_start:end].reset_index(drop=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    
    fig.add_trace(
        go.Candlestick(x=list(range(len(plot_df))), open=plot_df["o"], high=plot_df["h"],
                       low=plot_df["l"], close=plot_df["c"],
                       increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False),
        row=1, col=1,
    )
    
    vol_colors = ["#26a69a" if row["c"] >= row["o"] else "#ef5350" for _, row in plot_df.iterrows()]
    fig.add_trace(
        go.Bar(x=list(range(len(plot_df))), y=plot_df["v"],
               marker_color=vol_colors, opacity=0.4, showlegend=False),
        row=2, col=1,
    )
    
    for i in range(len(plot_df)):
        fig.add_annotation(x=i, y=float(plot_df.iloc[i]["h"]), text=str(display_start + i),
                           showarrow=False, yshift=8, font=dict(size=7, color="#888888"), row=1, col=1)
    
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(row=2, col=1, title_text="K线序号")
    fig.update_yaxes(row=1, col=1, title_text="价格")
    fig.update_yaxes(row=2, col=1, title_text="成交量")
    return fig

def reset_skill_states():
    st.session_state.current_skill = None
    st.session_state.skill_rounds = {}
    st.session_state.coach_dialogue = []

# ====== 侧边栏（与V18一致） ======
with st.sidebar:
    st.markdown("### ⚙️ 控制面板")
    
    col1, col2 = st.columns(2)
    with col1:
        sel_symbol = st.selectbox("品种", list(SYMBOLS.keys()), key="sel_symbol")
    with col2:
        sel_period = st.selectbox("周期", list(PERIODS.keys()), key="sel_period")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 随机跳转", use_container_width=True):
            st.session_state.random_jump_flag = True
    with col_b:
        if st.button("📥 重载数据", use_container_width=True):
            st.session_state.reload_data_flag = True
    
    st.markdown("---")
    
    col_c, col_d = st.columns(2)
    with col_c:
        num_bars = st.number_input("显示K线数", min_value=10, max_value=200, value=60, step=5, key="num_bars_input")
    with col_d:
        start_offset = st.number_input("起始偏移", min_value=0, max_value=1000, value=0, step=1, key="start_offset")
    
    if st.button("⏩ 下一根", use_container_width=True):
        if st.session_state.chart_df is not None:
            total = len(st.session_state.chart_df)
            new_start = st.session_state.display_start + 1
            if new_start + st.session_state.display_count <= total:
                st.session_state.display_start = new_start
                reset_skill_states()
    
    st.markdown("---")
    
    st.markdown("### 🎯 训练技能")
    cols = st.columns(2)
    skill_keys = list(SKILL_NAMES.keys())
    for i, sk in enumerate(skill_keys):
        with cols[i % 2]:
            rounds_used = st.session_state.skill_rounds.get(sk, 0)
            label = f"{SKILL_NAMES[sk]} ({rounds_used}/{MAX_ROUNDS_PER_SKILL})"
            is_active = st.session_state.current_skill == sk
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"skill_{sk}"):
                if st.session_state.chart_df is not None:
                    st.session_state.current_skill = sk
                    if sk not in st.session_state.skill_rounds:
                        st.session_state.skill_rounds[sk] = 0
                    st.session_state.coach_dialogue = []
    
    st.markdown("---")
    
    if st.session_state.data_source:
        with st.expander("📊 数据源", expanded=False):
            ds = st.session_state.data_source
            st.caption(f"品种: {ds.get('symbol', '-')}")
            st.caption(f"周期: {ds.get('period', '-')}")
            st.caption(f"交易所: {ds.get('exchange', '-')}")
            st.caption(f"总K线数: {ds.get('total_bars', 0)}")
            st.caption(f"时间范围: {ds.get('time_range', '-')}")

# ====== 主区域 ======
st.title("Al Brooks 价格行为结构训练器 V19")

# ====== 数据加载/切换 ======
df = st.session_state.chart_df

if df is None:
    # 首次加载
    with st.spinner("正在加载数据..."):
        df = load_data(sel_symbol, sel_period)
        if df is not None:
            st.session_state.chart_df = df
            st.session_state.total_bars = len(df)
            st.session_state.display_count = num_bars
            st.session_state.display_start = min(start_offset, max(0, len(df) - num_bars))
            exchange_name = EXCHANGE_NAMES.get(EXCHANGES.get(SYMBOLS.get(sel_symbol, ""), ""), "")
            tr = f"{df.iloc[0]['time']} ~ {df.iloc[-1]['time']}" if "time" in df.columns else "-"
            st.session_state.data_source = {
                "symbol": sel_symbol, "period": sel_period,
                "exchange": exchange_name, "total_bars": len(df), "time_range": tr,
            }
            reset_skill_states()
            st.rerun()
        else:
            st.error("数据加载失败，请尝试其他品种或周期")

elif st.session_state.reload_data_flag:
    # 重载数据（重新从接口拉取）
    st.session_state.reload_data_flag = False
    with st.spinner("正在重载数据..."):
        df = load_data(sel_symbol, sel_period)
        if df is not None:
            st.session_state.chart_df = df
            st.session_state.total_bars = len(df)
            st.session_state.display_count = num_bars
            st.session_state.display_start = 0
            exchange_name = EXCHANGE_NAMES.get(EXCHANGES.get(SYMBOLS.get(sel_symbol, ""), ""), "")
            tr = f"{df.iloc[0]['time']} ~ {df.iloc[-1]['time']}" if "time" in df.columns else "-"
            st.session_state.data_source = {
                "symbol": sel_symbol, "period": sel_period,
                "exchange": exchange_name, "total_bars": len(df), "time_range": tr,
            }
            reset_skill_states()
            st.rerun()

elif st.session_state.random_jump_flag:
    # 随机跳转（同一数据内随机位置）
    st.session_state.random_jump_flag = False
    total = len(st.session_state.chart_df)
    max_start = max(0, total - st.session_state.display_count - 10)
    if max_start > 0:
        st.session_state.display_start = random.randint(0, max_start)
        reset_skill_states()
        st.rerun()

# ====== 显示图表 ======
if st.session_state.chart_df is not None:
    df = st.session_state.chart_df
    display_start = st.session_state.display_start
    display_count = st.session_state.display_count
    display_end = min(display_start + display_count, len(df))
    
    fig = plot_chart(df, display_start, display_count)
    st.plotly_chart(fig, use_container_width=True, key="main_chart")
    
    st.caption(f"显示 K{display_start} ~ K{display_end - 1} / 共 {len(df)} 根K线 | {sel_period} | {sel_symbol}")
    
    st.markdown("---")
    
    # ====== 技能训练区域 ======
    if st.session_state.current_skill is None:
        st.info("👈 请先在左侧选择一个训练技能")
    else:
        sk = st.session_state.current_skill
        skill_name = SKILL_NAMES[sk]
        rounds_used = st.session_state.skill_rounds.get(sk, 0)
        rounds_left = MAX_ROUNDS_PER_SKILL - rounds_used
        
        st.markdown(f"### 🎯 当前技能：{skill_name}")
        
        # 显示对话历史
        for msg in st.session_state.coach_dialogue:
            if msg["role"] == "user":
                st.markdown(f"**🧑 你**\n{msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**🤖 教练**\n{msg['content']}")
        
        # 输入区域（最多2轮）
        if rounds_left > 0:
            with st.form(key=f"input_form_{sk}", clear_on_submit=True):
                user_text = st.text_area(
                    f"你的观察（还可输入 {rounds_left} 轮）",
                    placeholder="描述你看到的K线行为...",
                    height=80,
                    key=f"input_area_{sk}_{rounds_used}",
                )
                submitted = st.form_submit_button("提交", use_container_width=True)
                
                if submitted and user_text.strip():
                    with st.spinner("AI教练正在分析..."):
                        market_data = prepare_market_msg(df, display_start, display_count)
                        reply = ask_coach(sk, user_text, market_data, st.session_state.reading_profile)
                        st.session_state.coach_dialogue.append({"role": "user", "content": user_text})
                        st.session_state.coach_dialogue.append({"role": "assistant", "content": reply})
                        st.session_state.skill_rounds[sk] = rounds_used + 1
                        st.rerun()
        else:
            st.warning(f"✅ {skill_name} 已完成 {MAX_ROUNDS_PER_SKILL} 轮训练，请选择下一个技能。")
            
            all_done = all(st.session_state.skill_rounds.get(sk, 0) >= MAX_ROUNDS_PER_SKILL for sk in skill_keys)
            if all_done:
                st.success("🎉 所有5个技能训练完成！点击「下一根」或「随机跳转」开始新的训练。")
        
        # 阅读画像
        with st.expander("📊 你的阅读画像", expanded=False):
            for k, v in st.session_state.reading_profile.items():
                if v > 0:
                    st.markdown(f"{k}: {'█' * min(v, 30)} ({v}次)")
                else:
                    st.markdown(f"{k}: 暂无记录")

st.markdown("---")
st.caption("数据来源: 新浪财经 (akshare) | AI: DeepSeek | Al Brooks 价格行为训练系统 V19")
