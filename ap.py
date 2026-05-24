# Al Brooks 读盘训练器 V16
# =========================================================
# 认知训练工程 — 不是软件工程
# 用户 = 训练者 | GPT = 教练 | 软件 = 训练场
#
# V16 核心变更：
# - 教练连续对话链（多轮追问直到回到K线行为）
# - 删除Swing/Leg检测、规则化能力画像、行为1/2/3
# - "判断"→"观察"（消除结论诱导）
# - 布局：左80%图表 + 右20%对话
# - 训练结束GPT总结（替代规则检测）
# - 时间轴简化为纯文本记录
# =========================================================

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak
from openai import OpenAI

# =========================================================
# 常量
# =========================================================
CHUNK_SIZE = 300

SKILLS = {
    1: {"name": "背景阅读",   "question": "当前市场背景是什么？"},
    2: {"name": "控制权识别", "question": "现在谁在控制市场？"},
    3: {"name": "推进质量",   "question": "最近推进的质量如何？"},
    4: {"name": "回调vs转换", "question": "这是正常回调还是控制权转换？"},
    5: {"name": "市场接受",   "question": "市场是否接受了新价格？"},
}

AI_SYSTEM_PROMPT = """
你是 Al Brooks 价格行为训练教练。

你不是交易员。
你不是分析师。
你不是预测模型。

你唯一职责：

帮助用户训练以下5项核心能力：

1. 背景阅读
2. 控制权识别
3. 推进质量判断
4. 区分正常回调与真正转换
5. 理解市场是否接受新价格

--------------------------------------------------

【核心原则】

真正获得能力的人只能是用户。

你永远不能替用户：
- 观察
- 推理
- 下结论
- 判断市场

你只能：
- 引导
- 追问
- 纠偏
- 强迫用户回到具体K线行为

--------------------------------------------------

【你必须严格禁止】

禁止：

- 告诉用户市场方向
- 告诉用户趋势/区间/反转
- 告诉用户谁控制市场
- 告诉用户用户是否正确
- 给交易建议
- 给买卖建议
- 预测后续走势
- 替用户总结市场结论

--------------------------------------------------

【最重要规则】

当用户使用以下抽象词：

- 转强
- 转弱
- 趋势
- 反转
- 突破
- 控制
- 接受
- 拒绝
- 多头
- 空头
- 强势
- 弱势

你绝对不能围绕这些词讨论。

你必须：

强制用户重新回到：

具体K线行为。

例如：

- 哪几根K线？
- 行为从哪里开始变化？
- 后续有没有跟进？
- 对手有没有回应？
- 重叠有没有增加？
- 收盘位置有没有变化？
- 推进是否持续？
- 行为是否连续？

--------------------------------------------------

【你的真正职责】

你只能做5件事：

1. 强迫用户引用具体K线

例如：
- 从哪几根开始？
- 哪一段？
- 哪里的行为发生变化？

--------------------------------------------------

2. 强迫用户描述行为

只能讨论：
- 实体变化
- 收盘位置变化
- 高低点变化
- 重叠变化
- 跟进行为
- 对手回应
- 推进连续性

禁止讨论抽象市场定义。

--------------------------------------------------

3. 强迫用户提供依据

用户每一个观点，
都必须要求：

"依据是什么？"

--------------------------------------------------

4. 强迫用户面对矛盾

例如：

用户说：
"这里开始转强"

你必须追问：

- 为什么后续没有持续跟进？
- 为什么价格仍然频繁重叠？
- 为什么对手仍然持续回应？

--------------------------------------------------

5. 强迫用户观察连续性

你必须不断提醒用户：

不要只看：
- 单根K线
- 单个形态
- 单次突破

而要观察：

- 行为是否持续
- 跟进是否衰减
- 对手是否回应
- 市场是否真正接受价格

--------------------------------------------------

【你的回答风格】

- 简短
- 直接
- 一次只推进一步
- 不长篇解释
- 不分析市场
- 不总结市场
- 不教学式讲解

--------------------------------------------------

【最关键规则】

如果你发现：

用户开始：
- 下定义
- 猜趋势
- 猜反转
- 猜方向

你必须立即：

把用户拉回：

"具体发生了什么行为？"

这是你的最高优先级。
"""

# =========================================================
# 数据类
# =========================================================
@dataclass
class Observation:
    skill_id: int
    bar: int
    text: str
    timestamp: str

@dataclass
class TimelineEvent:
    bar: int
    text: str
    timestamp: str

# =========================================================
# 数据加载
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_raw(symbol):
    for _ in range(3):
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol, period="15")
            df = df.rename(columns={
                "datetime": "datetime", "open": "open",
                "high": "high", "low": "low", "close": "close"})
            df = df.reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df.reset_index(drop=True)
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


def load_data(symbol, seed=None):
    raw = _fetch_raw(symbol)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    n = len(raw)
    if n <= CHUNK_SIZE:
        return raw.reset_index(drop=True)
    rng = random.Random(seed)
    start = rng.randint(0, n - CHUNK_SIZE)
    return raw.iloc[start:start + CHUNK_SIZE].reset_index(drop=True)

# =========================================================
# 图表（极简：只有K线）
# =========================================================
def build_chart(chart_df, bar):
    fig = go.Figure()
    vis = chart_df.iloc[:bar + 1]
    if len(vis) == 0:
        return fig
    fig.add_trace(go.Candlestick(
        x=vis.index, open=vis["open"], high=vis["high"],
        low=vis["low"], close=vis["close"],
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71"))
    cur = chart_df.iloc[bar]
    fig.update_layout(
        annotations=[dict(
            x=bar, y=cur["high"], text="#{}".format(bar),
            showarrow=True, arrowhead=0, arrowcolor="#aaa",
            font=dict(size=8, color="#aaa"), ax=0, ay=25)],
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        template="plotly_white",
    )
    return fig

# =========================================================
# GPT 教练
# =========================================================
def _build_market_msg(chart_df, bar, skill_name):
    start = max(0, bar - 30)
    recent = []
    for i in range(start, bar + 1):
        r = chart_df.iloc[i]
        recent.append({
            "bar": i,
            "open": round(float(r["open"]), 1),
            "high": round(float(r["high"]), 1),
            "low": round(float(r["low"]), 1),
            "close": round(float(r["close"]), 1),
        })
    return json.dumps({
        "current_bar": bar,
        "total_bars": len(chart_df),
        "skill": skill_name,
        "market": recent,
    }, ensure_ascii=False)


def ask_coach(chart_df, bar, skill_name, dialogue, extra=None):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")

    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": _build_market_msg(chart_df, bar, skill_name)},
    ]
    for msg in dialogue:
        messages.append({"role": msg["role"], "content": msg["content"]})
    if extra:
        messages.append({"role": "user", "content": extra})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=messages,
                temperature=0.4,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2 and "429" in str(e):
                time.sleep(2 ** (attempt + 1))
                continue
            return "AI调用失败: {}".format(e)


def ask_summary(chart_df, observations, dialogue):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")

    obs_text = "\n".join(
        "[K{}] {}".format(o.bar, o.text) for o in observations)
    dlg_text = "\n".join(
        "{}: {}".format(
            "用户" if m["role"] == "user" else "教练", m["content"])
        for m in dialogue[-20:])

    prompt = (
        "以下是用户本次训练的全部观察和教练对话。\n\n"
        "【观察记录】\n{}\n\n"
        "【教练对话】\n{}\n\n"
        "请分析用户的读盘能力，输出：\n"
        "1. 用户长期问题（具体到行为层面）\n"
        "2. 习惯性错误（引用对话中的实际表现）\n"
        "3. 下一阶段训练重点\n\n"
        "基于对话中的实际表现分析，不要泛泛而谈。"
    ).format(obs_text, dlg_text)

    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "总结生成失败: {}".format(e)


def ask_memory_test(chart_df, bar, observations):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")

    market = _build_market_msg(chart_df, bar, "")
    obs_text = "\n".join(
        "[K{}] {}".format(o.bar, o.text) for o in observations[-10:])

    prompt = (
        "这是一次延迟记忆训练。\n\n"
        "当前盘面：\n{}\n\n"
        "用户的观察记录：\n{}\n\n"
        "请根据用户的观察记录，出1-2个记忆测试问题：\n"
        "测试用户是否记得之前观察到的具体行为变化。\n"
        "只问问题，不给出答案。\n"
        "问题要具体到K线行为，不要问抽象概念。"
    ).format(market, obs_text)

    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "AI调用失败: {}".format(e)


def ask_contradiction(chart_df, bar, skill_name, dialogue):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.videocaptioner.cn/v1")

    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT},
        {"role": "user", "content": _build_market_msg(chart_df, bar, skill_name)},
    ]
    for msg in dialogue:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": (
        "请找出用户观察中的矛盾之处。\n"
        "用户之前说了一些观察，现在盘面已经变化。\n"
        "指出用户观察与实际K线行为之间的矛盾。\n"
        "不要直接告诉答案，用提问的方式让用户自己发现矛盾。"
    )})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=messages,
                temperature=0.4,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2 and "429" in str(e):
                time.sleep(2 ** (attempt + 1))
                continue
            return "AI调用失败: {}".format(e)

# =========================================================
# 主程序
# =========================================================
def main():
    for key, default in [
        ("data_loaded", False), ("observations", []),
        ("train_mode", 1), ("timeline", []),
        ("replay_mode", "复盘模式"), ("coach_dialogue", []),
        ("send_counter", 0), ("training_summary", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- 侧栏 ----
    with st.sidebar:
        st.title("读盘训练器 V16")
        st.caption("认知训练工程")

        symbol = st.text_input("合约代码", value="rb2510", key="sym")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("加载", key="load", use_container_width=True):
                _do_load(symbol)
        with c2:
            if st.button("换一段", key="rand", use_container_width=True):
                _do_load(symbol)

        if st.session_state.get("data_loaded"):
            st.markdown("---")
            st.session_state["replay_mode"] = st.radio(
                "Replay", ["复盘模式", "严格模式"], key="rm_radio")

            st.markdown("---")
            st.markdown("**训练目标**")
            for sid in range(1, 6):
                name = SKILLS[sid]["name"]
                active = st.session_state.get("train_mode") == sid
                prefix = "▶ " if active else "  "
                if st.button(
                    "{}{}. {}".format(prefix, sid, name),
                    key="mode_{}".format(sid),
                    use_container_width=True,
                ):
                    st.session_state["train_mode"] = sid
                    st.rerun()

            st.markdown("---")
            if st.button(
                "结束训练 → 总结", key="end_train",
                use_container_width=True, type="primary"):
                _do_summary()
                st.stop()

            obs_n = len(st.session_state.get("observations", []))
            dlg_n = len(st.session_state.get("coach_dialogue", [])) // 2
            st.caption("观察: {}次 | 对话: {}轮".format(obs_n, dlg_n))

    # ---- 欢迎页 ----
    if not st.session_state.get("data_loaded"):
        st.markdown("# Al Brooks 读盘训练器")
        st.markdown("")
        for sid in range(1, 6):
            s = SKILLS[sid]
            st.markdown("**{}. {}** — {}".format(
                sid, s["name"], s["question"]))
        st.markdown("")
        st.markdown("> 你看图。你观察。教练只提问，不给答案。")
        st.markdown("")
        st.markdown("**训练架构：**")
        st.markdown("- 用户 = 真正训练者")
        st.markdown("- GPT = 教练（与你看同一个盘面）")
        st.markdown("- 软件 = 训练场")
        return

    # ---- 训练总结页 ----
    if st.session_state.get("training_summary"):
        st.markdown("## 训练总结")
        st.markdown(st.session_state["training_summary"])
        if st.button("继续训练", key="resume"):
            st.session_state["training_summary"] = ""
            st.rerun()
        return

    # ---- 主布局 ----
    chart_df = st.session_state["chart_df"]
    bar = st.session_state.get("current_bar", 0)
    if bar >= len(chart_df):
        bar = len(chart_df) - 1
        st.session_state["current_bar"] = bar

    mode = st.session_state.get("train_mode", 1)
    skill = SKILLS[mode]
    strict = st.session_state.get("replay_mode") == "严格模式"

    col_left, col_right = st.columns([4, 1])

    # ===== 左列：图表 + 导航 =====
    with col_left:
        chart = build_chart(chart_df, bar)
        st.plotly_chart(chart, use_container_width=True)

        cur = chart_df.iloc[bar]
        chg = cur["close"] - cur["open"]
        sign = "+" if chg >= 0 else ""
        st.caption(
            "#{}  O:{:.0f}  H:{:.0f}  L:{:.0f}  C:{:.0f}  {}{:.0f}".format(
                bar, cur["open"], cur["high"], cur["low"],
                cur["close"], sign, chg))

        # Slider
        if strict:
            new_bar = bar
        else:
            new_bar = st.slider(
                "K线", 0, len(chart_df) - 1, bar, key="bar_slider")
        if not strict and new_bar != bar:
            st.session_state["current_bar"] = new_bar
            st.rerun()

        # 按钮行
        bc = st.columns(6)
        btns = [
            (" -5 ", "b_p5",  not strict),
            (" -1 ", "b_p1",  not strict),
            (" +1 ", "b_n1",  True),
            (" +5 ", "b_n5",  not strict),
            ("+15 ", "b_n15", not strict),
            ("末尾", "b_end",  not strict),
        ]
        for i, (label, key, enabled) in enumerate(btns):
            with bc[i]:
                if enabled:
                    if label.strip() == "-5":
                        step = -5
                    elif label.strip() == "-1":
                        step = -1
                    elif label.strip() == "+1":
                        step = 1
                    elif label.strip() == "+5":
                        step = 5
                    elif label.strip() == "+15":
                        step = 15
                    else:
                        step = None
                    if st.button(label, key=key, use_container_width=True):
                        if step is not None:
                            st.session_state["current_bar"] = max(
                                0, min(len(chart_df) - 1, bar + step))
                        else:
                            st.session_state["current_bar"] = len(chart_df) - 1
                        st.rerun()

    # ===== 右列：对话区 =====
    with col_right:
        st.markdown("**{}**".format(skill["name"]))
        st.caption(skill["question"])
        st.markdown("---")

        # 对话历史
        dialogue = st.session_state["coach_dialogue"]
        for msg in dialogue:
            if msg["role"] == "user":
                st.markdown("**你：** {}".format(msg["content"]))
            else:
                st.markdown("**教练：** {}".format(msg["content"]))

        # 输入区
        cnt = st.session_state.get("send_counter", 0)
        obs_text = st.text_area(
            "你观察到了什么？", height=80,
            key="obs_{}".format(cnt), label_visibility="visible")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("发送", key="send_obs", use_container_width=True):
                if obs_text.strip():
                    _send_observation(obs_text.strip(), chart_df, bar, skill)
        with c2:
            if st.button("新一轮", key="new_round", use_container_width=True):
                st.session_state["coach_dialogue"] = []
                st.rerun()

        # 特殊教练交互
        st.markdown("---")
        if st.button("记忆训练", key="btn_memory", use_container_width=True):
            observations = st.session_state.get("observations", [])
            if len(observations) < 3:
                st.warning("至少观察3次后可用")
            else:
                with st.spinner("出题中..."):
                    q = ask_memory_test(chart_df, bar, observations)
                st.session_state["coach_dialogue"].append(
                    {"role": "assistant", "content": "[记忆测试] " + q})
                st.rerun()

        if st.button("找矛盾", key="btn_contra", use_container_width=True):
            if len(dialogue) < 4:
                st.warning("至少对话2轮后可用")
            else:
                with st.spinner("分析中..."):
                    q = ask_contradiction(
                        chart_df, bar, skill["name"], dialogue)
                st.session_state["coach_dialogue"].append(
                    {"role": "assistant", "content": q})
                st.rerun()

    # ===== 底部：时间轴 =====
    tl = st.session_state.get("timeline", [])
    if tl or True:
        with st.expander("行为变化记录 ({})".format(len(tl))):
            for ev in tl[-10:]:
                st.caption("[K{}] {}".format(ev.bar, ev.text))
            tc = st.columns([5, 1])
            with tc[0]:
                tl_input = st.text_input(
                    "记录", key="tl_input",
                    placeholder="描述这里的行为变化...")
            with tc[1]:
                if st.button("记", key="tl_add"):
                    if tl_input.strip():
                        tl.append(TimelineEvent(
                            bar=bar, text=tl_input.strip(),
                            timestamp=datetime.now().strftime("%H:%M:%S")))
                        st.session_state["timeline"] = tl
                        st.rerun()
            if tl and st.button("清空", key="tl_clear"):
                st.session_state["timeline"] = []
                st.rerun()


def _send_observation(text, chart_df, bar, skill):
    session = st.session_state
    dialogue = session["coach_dialogue"]
    mode = session.get("train_mode", 1)

    dialogue.append({"role": "user", "content": text})

    session["observations"].append(Observation(
        skill_id=mode, bar=bar, text=text,
        timestamp=datetime.now().strftime("%H:%M:%S")))

    with st.spinner("教练思考中..."):
        response = ask_coach(chart_df, bar, skill["name"], dialogue)

    dialogue.append({"role": "assistant", "content": response})
    session["coach_dialogue"] = dialogue
    session["send_counter"] = session.get("send_counter", 0) + 1
    st.rerun()


def _do_load(symbol):
    with st.spinner("加载中..."):
        seed = random.randint(0, 999999)
        df = load_data(symbol, seed=seed)
        if df is not None and len(df) > 0:
            st.session_state.update({
                "chart_df": df,
                "current_bar": min(40, len(df) - 1),
                "data_loaded": True,
                "observations": [],
                "timeline": [],
                "train_mode": 1,
                "coach_dialogue": [],
                "training_summary": "",
                "send_counter": 0,
            })
            st.success("{}根K线".format(len(df)))
        else:
            st.error("加载失败")


def _do_summary():
    session = st.session_state
    chart_df = session.get("chart_df")
    observations = session.get("observations", [])
    dialogue = session.get("coach_dialogue", [])
    if not observations:
        st.warning("还没有观察记录")
        return
    with st.spinner("生成训练总结..."):
        summary = ask_summary(chart_df, observations, dialogue)
    session["training_summary"] = summary


if __name__ == "__main__":
    main()
