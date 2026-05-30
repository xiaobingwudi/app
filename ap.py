# =========================================================
# 数据加载（修复版）
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_raw(symbol):
    """获取原始K线数据"""
    try:
        # 直接使用传入的合约代码（已经是主力合约格式）
        df = ak.futures_zh_minute_sina(symbol=symbol, period="30")
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 重命名列
        df = df.rename(columns={
            "datetime": "datetime",
            "open": "open",
            "high": "high", 
            "low": "low",
            "close": "close"
        })
        
        df = df.reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df.reset_index(drop=True)
        
    except Exception as e:
        st.warning(f"数据获取失败 {symbol}: {str(e)[:100]}")
        return pd.DataFrame()


def load_data(symbol, seed=None):
    """加载数据并随机截取一段"""
    raw = _fetch_raw(symbol)
    
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    
    n = len(raw)
    if n <= CHUNK_SIZE:
        return raw.reset_index(drop=True)
    
    rng = random.Random(seed)
    start = rng.randint(0, n - CHUNK_SIZE)
    return raw.iloc[start:start + CHUNK_SIZE].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_main_contracts():
    """获取所有品种的主力合约代码"""
    import akshare as ak
    
    main_contracts = {}
    
    # 需要获取主力合约的品种列表
    symbols_to_fetch = list(SYMBOL_NAMES.keys())
    
    # 按交易所分组
    exchange_map = {
        "CU": "shfe", "AL": "shfe", "ZN": "shfe", "PB": "shfe", "NI": "shfe", "SN": "shfe",
        "AU": "shfe", "AG": "shfe", "RB": "shfe", "HC": "shfe", "SS": "shfe", "BU": "shfe",
        "RU": "shfe", "FU": "shfe", "SC": "shfe", "SP": "shfe",
        "I": "dce", "J": "dce", "JM": "dce", "A": "dce", "B": "dce", "M": "dce",
        "Y": "dce", "P": "dce", "C": "dce", "L": "dce", "PP": "dce", "V": "dce",
        "EG": "dce", "EB": "dce", "PG": "dce", "JD": "dce", "RR": "dce", "LH": "dce",
        "CF": "czce", "SR": "czce", "TA": "czce", "MA": "czce", "FG": "czce", "SA": "czce",
        "OI": "czce", "RM": "czce", "AP": "czce", "CJ": "czce", "UR": "czce", "PF": "czce",
        "PK": "czce", "SH": "czce", "PX": "czce",
        "IF": "cffex", "IC": "cffex", "IM": "cffex", "IH": "cffex", "T": "cffex", "TF": "cffex", "TS": "cffex",
        "SI": "gfex", "LC": "gfex"
    }
    
    # 按交易所分别获取主力合约
    for code in symbols_to_fetch:
        exchange = exchange_map.get(code, "shfe")
        try:
            # 方法1: 直接获取单个品种的主力合约
            main = ak.match_main_contract(symbol=code)
            if main and isinstance(main, str) and len(main) > 2:
                main_contracts[code] = main
                continue
        except Exception:
            pass
        
        try:
            # 方法2: 通过交易所获取所有主力合约
            result = ak.match_main_contract(symbol=exchange)
            if result:
                contracts = str(result).split(",")
                for c in contracts:
                    c = c.strip()
                    if len(c) < 3:
                        continue
                    # 提取品种代码（去掉数字）
                    raw_code = "".join(ch for ch in c[:4] if ch.isalpha()).upper()
                    if raw_code == code and code not in main_contracts:
                        main_contracts[code] = c
                        break
        except Exception:
            pass
        
        # 方法3: 如果都失败，使用默认格式（品种+主力后缀）
        if code not in main_contracts:
            # 尝试常见的后缀格式
            for suffix in ["888", "000", "0", "99"]:
                try:
                    test_symbol = f"{code}{suffix}"
                    df = ak.futures_zh_minute_sina(symbol=test_symbol, period="30")
                    if df is not None and not df.empty:
                        main_contracts[code] = test_symbol
                        break
                except Exception:
                    continue
        
        # 避免请求过快
        time.sleep(0.1)
    
    return main_contracts


def _load_all_main_contracts(mc):
    """加载所有主力合约到session_state"""
    result = get_main_contracts()
    mc.update(result)
    
    
def _do_load(sym_code, sym_main):
    """加载品种数据"""
    with st.spinner(f"加载 {SYMBOL_NAMES.get(sym_code, sym_code)} 数据..."):
        seed = random.randint(0, 999999)
        df = load_data(sym_main, seed=seed)
        
        if df is not None and len(df) > 0:
            sw = detect_swings(df)
            st.session_state.update({
                "chart_df": df,
                "swings": sw,
                "current_bar": min(40, len(df) - 1),
                "data_loaded": True,
                "observations": [],
                "timeline": [],
                "train_mode": 1,
                "coach_dialogue": [],
                "training_summary": "",
                "skill_round": 0,
                "send_counter": 0,
            })
            st.success(f"已加载 {SYMBOL_NAMES.get(sym_code, sym_code)} ({sym_main})，共 {len(df)} 根K线")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(f"加载失败：{sym_main}")
