import json
import random
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# =========================================================
# 1. 页面配置 (专业交易暗黑风)
# =========================================================
st.set_page_config(
    page_title="Al Brooks 市场感知训练系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. CSS 美化样式
# =========================================================
st.markdown("""
<style>
/* 全局背景与字体 */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #0e1117;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 容器优化 */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* 标题美化 */
h1 {
    font-size: 32px !important;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}
h3 {
    color: #aebac9;
    border-left: 4px solid #00c3ff;
    padding-left: 12px;
    margin-top: 1.5rem;
}

/* 指标卡片 (Metrics) */
div[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    color: #58a6ff;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #8b949e;
}

/* 控件美化 */
.stSelectbox > div > div, .stRadio > div {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}
.stButton>button {
    width: 100%;
    border-radius: 6px;
    border: 1px solid #238636;
    background-color: #238636;
    color: white;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: #2ea043;
    box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
}

/* 文本域 */
.stTextArea textarea {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

/* 输入框 */
.stTextInput > div > div > input {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Al Brooks 市场感知训练系统")
st.markdown("---")

# =========================================================

# API 配置 - 安全版
# =========================================================
# 1. 从 st.secrets 中安全地读取 API Key
# 这里的 "OPENAI_API_KEY" 是你在 Streamlit 后台设置的变量名
api_key = st.secrets["OPENAI_API_KEY"]

# 2. 其他配置可以保持不变，或者也用 secrets 管理
BASE_URL = "https://api.videocaptioner.cn/v1"
MODEL_NAME = "gpt-5.4-nano"

# 3. 将读取到的 api_key 传给 OpenAI 客户端
client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)
# =========================================================
# 4. 数据加载与初始化
# =========================================================
DATA_DIR = Path("data")
if not DATA_DIR.exists():
    st.error("未找到 data 文件夹，请在同级目录下创建 data 文件夹并放入 CSV 文件。")
    st.stop()

csv_files = list(DATA_DIR.glob("*.csv"))
if len(csv_files) == 0:
    st.error("data 文件夹中没有 CSV 文件。")
    st.stop()

# Session State 初始化
if "selected_file" not in st.session_state:
    st.session_state.selected_file = random.choice(csv_files)
if "logs" not in st.session_state:
    st.session_state.logs = []
if "random_start" not in st.session_state:
    st.session_state.random_start = random.randint(100, 500)
if "current_index" not in st.session_state:
    st.session_state.current_index = st.session_state.random_start

# 加载数据
file_path = st.session_state.selected_file
df = pd.read_csv(file_path)
required_columns = ["Open", "High", "Low", "Close"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"CSV 缺少必要字段: {col}")
        st.stop()

df = df.reset_index(drop=True)
current_index = st.session_state.current_index

# =========================================================
# 5. 数据切片与特征计算
# =========================================================
WINDOW_SIZE = 80
start_index = max(0, current_index - WINDOW_SIZE)
visible_df = df.iloc[start_index:current_index]

# 当前 K 线数据
if len(visible_df) == 0:
    st.error("数据不足，请重新加载。")
    st.stop()

current_bar = visible_df.iloc[-1]
current_open = float(current_bar["Open"])
current_high = float(current_bar["High"])
current_low = float(current_bar["Low"])
current_close = float(current_bar["Close"])

# =========================================================
# 6. 顶部控制栏
# =========================================================
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
with col_nav1:
    if st.button("🎲 随机新图表"):
        st.session_state.selected_file = random.choice(csv_files)
        st.session_state.random_start = random.randint(100, len(df) - 50)
        st.session_state.current_index = st.session_state.random_start
        st.session_state.logs = []
        st.rerun()

with col_nav2:
    if st.button("⏭️ 下一根 K 线"):
        if current_index < len(df):
            st.session_state.current_index += 1
            st.rerun()

with col_nav3:
    st.markdown(f"**文件:** `{file_path.name}`  |  **进度:** {current_index} / {len(df)}")

# =========================================================
# 7. K 线图绘制 (Plotly 暗黑风)
# =========================================================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=list(range(len(visible_df))),
    open=visible_df["Open"],
    high=visible_df["High"],
    low=visible_df["Low"],
    close=visible_df["Close"],
    name="K线",
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
    hoverinfo='none'
))

fig.update_layout(
    height=500,
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor='#0e1117',
    plot_bgcolor='#161b22',
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(gridcolor='#30363d', color='#8b949e'),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# 8. 用户感知训练 (核心交互区 - 填空题形式)
# =========================================================
st.markdown("### 🧠 你的市场感知 (Market Reading)")
st.caption("请基于图表背景进行判断，不要只看最后一根 K 线。请用文字详细描述你的观察和分析。")

# 多空主导权判断
st.markdown("**1. 多空主导权分析**")
user_control = st.text_area(
    "谁掌握主动权？请详细描述你的观察依据（例如：多头主导、空头主导、势均力敌）",
    placeholder="请描述：当前市场是由多头、空头主导，还是势均力敌？你的判断依据是什么？",
    height=100
)

# 市场结构判断
st.markdown("**2. 市场结构分析**")
user_structure = st.text_area(
    "当前处于什么市场结构？请详细描述你的观察依据",
    placeholder="请描述：当前市场处于上升趋势、下降趋势、交易区间，还是突破/反转阶段？你的判断依据是什么？",
    height=100
)

# 关键价位识别
st.markdown("**3. 关键价位识别**")
user_key_levels = st.text_input(
    "你认为当前最关键的多空分界价位是多少？",
    placeholder="请填写一个具体的价格数值"
)

# 推动力量评估
st.markdown("**4. 推动力量评估**")
user_momentum = st.text_area(
    "当前的推动力量如何？请详细描述你的观察依据",
    placeholder="请描述：当前市场的推动力量是强、中等、弱？你的判断依据是什么？",
    height=100
)

# 综合观察笔记
st.markdown("**5. 综合观察笔记**")
user_notes = st.text_area(
    "📝 综合观察笔记 (支持 Markdown)",
    value="请详细描述你对当前市场状态的完整分析，包括：\n- 市场背景与结构\n- 多空力量对比\n- 关键价位\n- 你的分析逻辑",
    height=150
)


# =========================================================
# 10. AI 分析与反馈 (Context-Aware)
# =========================================================
def analyze_with_gpt(candles_data, user_input):
    system_prompt = """
    你是一位极度严谨的 Al Brooks 价格行为交易导师。请基于当前的K线图表，严格评估学生对"市场状态、多空主导权、结构完整性"的感知是否准确。

    【核心原则】
    1. 严禁预测：绝对不要根据未来的K线走势来倒推学生的对错，只基于当前可见的历史形态进行评判。
    2. 严格标准：请用 Al Brooks 的专业视角（如趋势强度、信号K线质量、高低点排列等）来审视学生的判断。如果学生忽略了明显的长影线、K线重叠或大实体推进等关键细节，请给予严厉且专业的指正。

    请以 JSON 格式返回：
    {
        "score": 0-100的整数,
        "market_read_correction": "你对当前市场状态的修正与专业解读",
        "coach_feedback": "具体的导师点评，指出其思维盲区",
        "focus_next": "下一根K线需要重点观察的关键价位"
    }
    """

    user_prompt = f"""
    市场数据（最近 80 根 K 线，包含 Open, High, Low, Close）：
    {json.dumps(candles_data, indent=2)}

    学生的判断：
    - 多空主导权: {user_input['control']}
    - 市场结构: {user_input['structure']}
    - 关键价位: {user_input['key_levels']}
    - 推动力量: {user_input['momentum']}
    - 综合笔记: {user_input['notes']}

    请基于这 80 根 K 线的大背景，评估学生的判断。
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# =========================================================
# 11. 执行分析
# =========================================================
if st.button("🔍 提交判断并获取 AI 导师反馈", type="primary"):
    with st.spinner("AI 正在结合大背景分析你的感知..."):
        # 准备数据
        context_candles = []
        for _, row in visible_df.iterrows():
            context_candles.append({
                "o": round(row["Open"], 2),
                "h": round(row["High"], 2),
                "l": round(row["Low"], 2),
                "c": round(row["Close"], 2)
            })

        user_input = {
            "control": user_control,
            "structure": user_structure,
            "key_levels": user_key_levels,
            "momentum": user_momentum,
            "notes": user_notes
        }

        try:
            ai_result = analyze_with_gpt(context_candles, user_input)

            # 展示结果
            st.markdown("---")
            st.markdown("### 🏆 导师评估报告")

            res_col1, res_col2 = st.columns([1, 3])
            with res_col1:
                score = ai_result.get("score", 0)
                st.metric("感知准确度评分", score)

            with res_col2:
                st.info(f"**市场背景修正：** {ai_result.get('market_read_correction')}")

            st.markdown(f"**💬 导师点评：** {ai_result.get('coach_feedback')}")
            st.warning(f"**👀 下一步关注：** {ai_result.get('focus_next')}")

            # 记录日志
            st.session_state.logs.append({
                "index": current_index,
                "score": score,
                "feedback": ai_result.get("coach_feedback")
            })

        except Exception as e:
            st.error(f"AI 分析出错: {str(e)}")

# =========================================================
# 12. 历史记录
# =========================================================
if st.session_state.logs:
    st.markdown("---")
    st.markdown("### 📜 训练记录")
    log_df = pd.DataFrame(st.session_state.logs)
    st.dataframe(log_df, hide_index=True, height=200)
(AI生成)
