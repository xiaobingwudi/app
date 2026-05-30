"""
Al Brooks 结构训练器 V18
================================
核心改进：
  1. AI_SYSTEM_PROMPT 重写 - 双重职责+两轮流程+技能核心维度
  2. _market_msg 重写 - 从JSON数字改为自然语言文字描述
  3. ask_coach 加 is_second_round 参数
  4. 对话记忆扩大到最近10轮
  5. 第2轮强制点评+亮判断
  6. 布局严格保持V17风格
  7. 适配 DeepSeek API
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

# ============================================================
# 品种映射
# ============================================================
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

# ============================================================
# Prompt 模板
# ============================================================
AI_SYSTEM_PROMPT_TEMPLATE = """你是 Al Brooks 价格行为训练教练。

【你的双重职责】
1. 分析市场 - 你和其他教练一样，能看到K线数据，对当前行情有自己的判断
2. 训练用户 - 通过提问和反馈，帮助用户提升观察能力

【核心原则】
你的判断是"参考答案"，不是"标准答案"。
价格行为分析没有唯一正确答案，你的作用是：
- 检查用户是否遗漏了重要维度
- 提供另一个角度的观察
- 帮助用户建立系统的观察框架

当前用户正在训练：{skill_name}
训练阶段：{level_name}
{level_desc}

当前技能的核心问题：{skill_question}

【技能的核心观察维度】

技能1 背景阅读：趋势方向、高低点序列(HH/HL或LH/LL)、通道斜率、震荡区间边界
技能2 控制权识别：最近3-5根谁在主导、推进方K线实体质量、对手方有无有效反击
技能3 推进质量：K线实体大小、K线重叠程度、影线长度、收盘位置、动能是否衰减
技能4 回调vs转换：回调K线数量、回调K线实体强弱、对手方是否连续出现、有无跟进
技能5 市场接受：突破后停留几根、有无立刻推回、有无继续朝突破方向推进

【训练流程 - 严格执行】

第1轮（用户首次作答）：
- 检查用户的回答是否触及该技能的核心观察维度
- 如果到位：说"好的，我明白了"，然后直接进入第2轮流程
- 如果不到位：给一次提示，只指向用户遗漏的具体维度，**绝对不给答案**

第2轮（用户二次作答）：
无论用户答得如何，执行以下两步：
1. 对用户的回答给出简短点评（肯定到位的部分，指出仍可补充的维度）
2. 亮出你自己的判断（基于你看到的K线数据，说清楚你的观察依据，引用具体K线编号）
然后结束，不再追问。

【约束】
- 提示只给一次，第2轮必须亮出自己判断
- 优先关注当前技能，用户问到其他领域可简短回应
- 回答简短，不列大纲，不超过150字
"""

TRAIN_LEVEL = {
    1: {"name": "观察阶段", "desc": "允许模糊、整体感觉、通道、节奏、倾向。禁止结构辩论、精确确认、摆动定义。"},
    2: {"name": "行为细化阶段", "desc": "开始细化行为、具体K线、推进连续性。"},
    3: {"name": "结构验证阶段", "desc": "允许失败突破、摆动确认、Always In转换、结构争议。"},
}

AI_SUMMARY_PROMPT = """你是训练总结分析师。根据训练对话记录，分析用户的阅读习惯和训练进展。
输出格式（JSON）：
{
    "observations": ["用户的阅读习惯和特点"],
    "strong_areas": ["用户表现好的方面"],
    "weak_areas": ["用户需要加强的方面"],
    "next_focus": ["下一阶段训练建议"]
}
要求：每条分析具体，引用训练中的实际表现，不要笼统评价，要有可操作性。"""

# ============================================================
# 数据加载
# ============================================================
def load_data(symbol, period="30", seed=None):
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
    
    # 处理可能存在的 open_interest 列
    if "open_interest" in df.columns:
        df = df.drop(columns=["open_interest"])
    
    df = df.dropna().reset_index(drop=True)
    
    # 随机截取一段
    if seed is not None and len(df) > 300:
        rng = random.Random(seed)
        start = rng.randint(0, len(df) - 300)
        df = df.iloc[start:start+300].reset_index(drop=True)
    
    return df

# ============================================================
# 图表构建
# ============================================================
def build_chart(chart_df, bar):
    end = bar + 1
    start = max(0, end - 60)
    df = chart_df.iloc[start:end].copy().reset_index(drop=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.8, 0.2])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        showlegend=False,
        increasing_line_color="red", decreasing_line_color="cyan",
    ), row=1, col=1)

    colors = ["red" if c >= o else "cyan" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=colors,
        showlegend=False, opacity=0.5,
    ), row=2, col=1)

    for idx in range(len(df)):
        if idx % 5 == 0:
            row = df.iloc[idx]
            ny = row["low"] if row["close"] >= row["open"] else row["high"]
            fig.add_annotation(
                x=idx, y=ny,
                text=str(df.index[idx] + start),
                showarrow=False, font=dict(size=9, color="gray"),
                yshift=-10 if row["close"] >= row["open"] else 10,
            )

    current_local = bar - start
    fig.add_vline(x=current_local, line_dash="dash",
                  line_color="orange", line_width=1, opacity=0.6)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

# ============================================================
# 市场数据消息（自然语言版）
# ============================================================
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
        direction = "阳" if c >= o else "阴"

        if body_ratio >= 0.7:
            k_type = "大阳线" if direction == "阳" else "大阴线"
        elif body_ratio >= 0.4:
            k_type = "中阳线" if direction == "阳" else "中阴线"
        elif body_ratio >= 0.1:
            k_type = "小阳线" if direction == "阳" else "小阴线"
        else:
            k_type = "十字星"

        all_bars.append({
            "i": i, "o": o, "h": h, "l": l, "c": c,
            "direction": direction, "type": k_type,
            "body": body, "body_ratio": body_ratio, "total_range": total_range,
        })

    lines = [f"【当前K线】第{bar}号K线", "", "【最近行情描述】"]
    recent = all_bars[-15:] if len(all_bars) >= 15 else all_bars

    for idx, k in enumerate(recent):
        change_desc = ""
        if idx > 0:
            prev = recent[idx - 1]
            price_change = k["c"] - prev["c"]
            if abs(price_change) > k["total_range"] * 0.5 and k["total_range"] > 0:
                change_desc = f"，相比前一根{'大涨' if price_change > 0 else '大跌'}了{abs(price_change):.1f}"
            elif price_change > 0:
                change_desc = f"，比前一根涨了{price_change:.1f}"
            elif price_change < 0:
                change_desc = f"，比前一根跌了{abs(price_change):.1f}"
            else:
                change_desc = "，与前一根收盘持平"

        upper_wick = k["h"] - max(k["o"], k["c"])
        lower_wick = min(k["o"], k["c"]) - k["l"]
        wick_parts = []
        if k["body"] > 0:
            if upper_wick > k["body"] * 2:
                wick_parts.append("上影线很长")
            if lower_wick > k["body"] * 2:
                wick_parts.append("下影线很长")
        wick_text = f"，{','.join(wick_parts)}" if wick_parts else ""

        lines.append(f"  K{k['i']}: {k['type']}，开{k['o']:.0f} 收{k['c']:.0f} 高{k['h']:.0f} 低{k['l']:.0f}{wick_text}{change_desc}")

    lines.extend(["", "【整体市场感知】"])

    if len(all_bars) >= 10:
        last_10 = all_bars[-10:]
        yang_count = sum(1 for k in last_10 if k["direction"] == "阳")
        yin_count = 10 - yang_count

        if yang_count >= 7:
            bias = "近期明显偏多，阳线占主导"
        elif yin_count >= 7:
            bias = "近期明显偏空，阴线占主导"
        elif yang_count >= 6:
            bias = "近期略偏多"
        elif yin_count >= 6:
            bias = "近期略偏空"
        else:
            bias = "近期多空平衡，无明显偏向"
        lines.append(f"  * {bias}（最近10根中{yang_count}阳{yin_count}阴）")

        tc = all_bars[-1]["c"] - all_bars[0]["c"]
        if tc > 0:
            lines.append(f"  * 整体向上，累计上涨{tc:.1f}")
        elif tc < 0:
            lines.append(f"  * 整体向下，累计下跌{abs(tc):.1f}")
        else:
            lines.append("  * 整体持平")
        lines.append(f"  * 最近10根平均波幅{sum(k['total_range'] for k in last_10) / 10:.1f}")

        cons, max_c = 1, 1
        for i in range(1, len(last_10)):
            if last_10[i]["direction"] == last_10[i-1]["direction"]:
                cons += 1
                max_c = max(max_c, cons)
            else:
                cons = 1
        if max_c >= 4:
            lines.append(f"  * 出现连续{max_c}根{last_10[-1]['direction']}线，趋势有延续性")

    lines.extend(["", f"【当前训练技能】{skill_name}"])
    result = "\n".join(lines)
    st.session_state["_mm_cache"] = {"bar": bar, "skill": skill_name, "data": result}
    return result

# ============================================================
# GPT 调用（DeepSeek API）
# ============================================================
def _gpt(messages):
    # 从 Streamlit Secrets 读取 DeepSeek API Key
    api_key = st.secrets["OPENAI_API_KEY"]
    
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
    )
    
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            content = resp.choices[0].message.content.strip()
            # 清理可能的markdown代码块标记
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            return content
        except Exception as e:
            if attempt < 2 and "429" in str(e):
                time.sleep(2 ** (attempt + 1))
                continue
            return f"AI调用失败: {e}"


def ask_coach(chart_df, bar, skill_name, skill_question, dialogue, level=1, is_second_round=False):
    lv = TRAIN_LEVEL.get(level, TRAIN_LEVEL[1])
    system_prompt = AI_SYSTEM_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        level_name=lv["name"],
        level_desc=lv["desc"],
        skill_question=skill_question,
    )
    if is_second_round:
        system_prompt += "\n\n【这是第2轮】你必须：1) 点评用户的回答  2) 亮出你自己的判断（引用具体K线编号）"

    msgs = [{"role": "system", "content": system_prompt}]
    msgs.append({"role": "user", "content": _market_msg(chart_df, bar, skill_name)})
    for m in dialogue[-10:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    return _gpt(msgs)


def ask_summary(dialogue, observations):
    if not dialogue and not observations:
        return json.dumps({"observations": ["暂无训练数据"], "strong_areas": [], "weak_areas": [], "next_focus": []})
    ot = "\n".join(f"- {o['text']}" for o in observations[-20:])
    dt = "\n".join("{}: {}".format("用户" if m["role"] == "user" else "教练", m["content"]) for m in dialogue[-40:])
    return _gpt([
        {"role": "system", "content": AI_SUMMARY_PROMPT},
        {"role": "user", "content": f"【观察】\n{ot}\n\n【对话】\n{dt}"},
    ])

# ============================================================
# 技能定义（含提问模板）
# ============================================================
SKILLS = [
    {"id": 1, "name": "背景阅读",   "question": "当前市场背景是什么？",       "desc": "识别当前趋势环境、通道、区间"},
    {"id": 2, "name": "控制权识别", "question": "现在谁在控制市场？",         "desc": "判断多空谁在主导当前走势"},
    {"id": 3, "name": "推进质量",   "question": "最近推进的质量如何？",       "desc": "评估K线实体、重叠、影线反映的动能"},
    {"id": 4, "name": "回调vs转换", "question": "这是正常回调还是控制权转换？","desc": "区分回调与趋势转换的关键特征"},
    {"id": 5, "name": "市场接受",   "question": "市场是否接受了新价格？",     "desc": "突破后市场是否接受新价格区域"},
]

# ============================================================
# 合约数据加载
# ============================================================
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
                if len(c) < 3:
                    continue
                code = "".join(ch for ch in c[:4] if ch.isalpha()).upper()
                if code in SYMBOL_NAMES and code not in mc:
                    mc[code] = c
    return mc


def _load_all_main_contracts(mc):
    result = _fetch_all_contracts()
    mc.update(result)


def _do_load(sym_code, sym_main):
    with st.spinner("加载中..."):
        try:
            seed = random.randint(0, 999999)
            df = load_data(sym_main, seed=seed)
            if df is not None and len(df) > 0:
                st.session_state["chart_df"] = df
                st.session_state["current_bar"] = min(40, len(df) - 1)
                st.session_state["coach_dialogue"] = []
                st.session_state["observations"] = []
                st.session_state["training_summary"] = ""
                st.session_state["skill_round"] = 0
                st.session_state["send_counter"] = 0
                st.session_state["_mm_cache"] = {}
                st.session_state["symbol_code"] = sym_code
                st.session_state["symbol_main"] = sym_main
                st.session_state["symbol_name"] = SYMBOL_NAMES.get(sym_code, sym_code)
                st.success(f"已加载 {SYMBOL_NAMES.get(sym_code, sym_code)} ({sym_main})")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"加载失败：{sym_main} 无数据")
        except Exception as e:
            st.error(f"加载出错: {str(e)}")

# ============================================================
# 发送观察
# ============================================================
def _send(text, chart_df, bar, skill):
    s = st.session_state
    dlg = s["coach_dialogue"]
    dlg.append({"role": "user", "content": text})
    s["observations"].append({
        "skill_id": s.get("train_mode", 1), "bar": bar,
        "text": text, "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    with st.spinner("教练思考中..."):
        resp = ask_coach(
            chart_df, bar, skill["name"], skill["question"], dlg,
            level=s.get("train_level", 1),
            is_second_round=(s["skill_round"] >= 1),
        )

    s["skill_round"] += 1
    if s["skill_round"] >= 2:
        resp += "\n\n---\n本项技能训练结束，可切换下一项继续。"

    dlg.append({"role": "assistant", "content": resp})
    s["coach_dialogue"] = dlg
    s["send_counter"] = s.get("send_counter", 0) + 1
    st.rerun()

# ============================================================
# 主界面
# ============================================================
def main():
    st.set_page_config(page_title="Al Brooks 结构训练器", layout="wide")
    st.markdown("""
    <style>
        .stApp {background:#fafafa}
        .block-container {padding:1rem 2rem}
        .stButton>button {font-size:13px; padding:2px 12px; min-height:28px; height:28px; line-height:1}
        .stRadio>label {font-size:13px}
        .stTextInput>div>input {font-size:13px}
        .stSelectbox>div>div>select {font-size:13px}
        .css-1d391kg {padding-top:0.5rem}
        div[data-testid="stSidebar"] {width:320px}
    </style>
    """, unsafe_allow_html=True)

    # 初始化 session_state
    defaults = {
        "chart_df": None, "current_bar": 40,
        "coach_dialogue": [], "send_counter": 0,
        "training_summary": "", "skill_round": 0,
        "train_level": 1, "train_mode": 1,
        "observations": [],
        "symbol_code": "", "symbol_main": "", "symbol_name": "",
        "_mm_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown("### 品种选择")

        exchanges = {
            "金融": ["IF", "IH", "IC", "IM", "TS", "TF", "T", "TL"],
            "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG", "BC"],
            "黑色": ["RB", "HC", "SS", "I", "J", "JM"],
            "化工": ["MA", "TA", "PP", "L", "V", "EG", "EB", "PG", "SA", "UR", "SF", "SM", "PF"],
            "农产品": ["A", "B", "M", "Y", "P", "C", "CS", "JD", "CF", "SR", "RM", "OI", "FG", "AP", "CJ", "PK", "LH"],
            "能源": ["SC", "FU", "BU", "LU", "NR", "RU"],
        }

        mc = {}
        _load_all_main_contracts(mc)

        for cat, codes in exchanges.items():
            with st.expander(cat, expanded=(cat == "金融")):
                cols = st.columns(3)
                for idx, code in enumerate(codes):
                    if code in mc:
                        if cols[idx % 3].button(code, key=f"sym_{code}", use_container_width=True):
                            _do_load(code, mc[code])

        st.divider()

        level = st.selectbox(
            "训练阶段", options=[1, 2, 3],
            format_func=lambda x: f"阶段{x}: {TRAIN_LEVEL[x]['name']}",
            index=st.session_state.get("train_level", 1) - 1,
            key="train_level_sel",
        )
        st.session_state["train_level"] = level

        st.divider()
        st.caption(f"当前: {st.session_state.get('symbol_name', '未选择')}")

        if st.session_state.get("chart_df") is not None:
            df = st.session_state["chart_df"]
            st.caption(f"数据: {len(df)} 根K线 | Bar: {st.session_state['current_bar']}")

            max_bar = len(df) - 1
            bar = st.slider("K线位置", 41, max_bar,
                            value=st.session_state["current_bar"],
                            key="bar_slider")
            st.session_state["current_bar"] = bar

            if st.button("生成总结", use_container_width=True):
                with st.spinner("分析中..."):
                    summary = ask_summary(
                        st.session_state["coach_dialogue"],
                        st.session_state["observations"],
                    )
                    st.session_state["training_summary"] = summary
                    st.rerun()

            if st.session_state.get("training_summary"):
                with st.expander("训练总结", expanded=True):
                    st.text(st.session_state["training_summary"])

    # ---- 主区域 ----
    df = st.session_state.get("chart_df")
    if df is None:
        st.info("请从左侧选择品种开始训练")
        return

    bar = st.session_state["current_bar"]

    # 技能按钮 - 分两行显示
    current_skill_id = st.session_state.get("train_mode", 1)
    
    # 第一行：技能1-3
    cols1 = st.columns(3)
    for idx, sk in enumerate(SKILLS[:3]):
        is_active = (sk["id"] == current_skill_id)
        if cols1[idx].button(
            f"技能{sk['id']}: {sk['name']}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            key=f"skill_{sk['id']}",
        ):
            st.session_state["train_mode"] = sk["id"]
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.session_state["send_counter"] = 0
            st.rerun()

    # 第二行：技能4-5
    cols2 = st.columns(3)
    for idx, sk in enumerate(SKILLS[3:]):
        is_active = (sk["id"] == current_skill_id)
        if cols2[idx].button(
            f"技能{sk['id']}: {sk['name']}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            key=f"skill_{sk['id']}",
        ):
            st.session_state["train_mode"] = sk["id"]
            st.session_state["coach_dialogue"] = []
            st.session_state["skill_round"] = 0
            st.session_state["send_counter"] = 0
            st.rerun()

    active_skill = next(sk for sk in SKILLS if sk["id"] == current_skill_id)
    st.caption(f"当前技能: {active_skill['name']} - {active_skill['desc']} | "
               f"阶段: {TRAIN_LEVEL[st.session_state['train_level']]['name']} | "
               f"第{st.session_state['skill_round'] + 1}/2 轮")

    # K线图
    fig = build_chart(df, bar)
    st.plotly_chart(fig, use_container_width=True)

    # ---- 教练对话 ----
    st.markdown("### 教练")
    dialogue = st.session_state["coach_dialogue"]
    for m in dialogue[-10:]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(f"**{'🧑 你' if m['role'] == 'user' else '🤖 教练'}**")
            st.markdown(m["content"])

    s = st.session_state
    can_input = s.get("skill_round", 0) < 2 and s.get("chart_df") is not None

    if can_input:
        prompt = st.chat_input("分享你对当前行情的观察...")
    else:
        if s.get("skill_round", 0) >= 2:
            st.info("本项技能训练结束，点击上方技能按钮切换下一项继续训练。")
        prompt = None

    if prompt:
        _send(prompt, df, bar, active_skill)


if __name__ == "__main__":
    main()
