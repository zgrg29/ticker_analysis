
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# ============================================================
# 1. 基础配置
# ============================================================

st.set_page_config(
    page_title="多因子投资信号分析 V2",
    page_icon="📈",
    layout="wide",
)

WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "relative": 0.15,
    "volume": 0.10,
    "risk": 0.20,
    "regime": 0.10,
}

DIMENSIONS = list(WEIGHTS.keys())

LABELS = {
    "中文": {
        "title": "📈 多因子投资信号分析 V2",
        "subtitle": "股票与 ETF · 六维度评分 · 风险管理 · 历史回测",
        "trend": "趋势",
        "momentum": "动量",
        "relative": "相对强弱",
        "volume": "量价确认",
        "risk": "风险质量",
        "regime": "市场环境",
        "buy": "偏向买入",
        "hold": "持有 / 观察",
        "reduce": "偏向减仓",
        "avoid": "风险规避",
        "price": "价格",
        "score": "综合评分",
        "strategy": "信号策略",
        "benchmark": "买入并持有",
        "settings": "分析参数",
        "ticker": "股票 / ETF 代码",
        "reference": "相对强弱基准",
        "start": "开始日期",
        "end": "结束日期",
        "cost": "单边交易成本（基点）",
        "threshold": "买入评分门槛",
        "exit": "退出评分门槛",
        "overview": "当前综合判断",
        "details": "六维度评分",
        "history": "价格与历史信号",
        "backtest": "历史回测",
        "indicators": "指标明细",
        "returns": "累计收益",
        "cagr": "年化收益",
        "volatility": "年化波动率",
        "drawdown": "最大回撤",
        "sharpe": "夏普比率",
        "exposure": "持仓比例",
        "trades": "交易次数",
        "date": "数据日期",
        "warning": "仅供投资研究，不构成投资建议。",
    },
    "日本語": {
        "title": "📈 マルチファクター投資シグナル V2",
        "subtitle": "株式・ETF · 6分野評価 · リスク管理 · バックテスト",
        "trend": "トレンド",
        "momentum": "モメンタム",
        "relative": "相対強度",
        "volume": "出来高・価格",
        "risk": "リスク品質",
        "regime": "市場環境",
        "buy": "買い優勢",
        "hold": "保有・様子見",
        "reduce": "縮小優勢",
        "avoid": "リスク回避",
        "price": "価格",
        "score": "総合スコア",
        "strategy": "シグナル戦略",
        "benchmark": "バイ・アンド・ホールド",
        "settings": "分析設定",
        "ticker": "銘柄コード",
        "reference": "相対強度ベンチマーク",
        "start": "開始日",
        "end": "終了日",
        "cost": "片道取引コスト（bp）",
        "threshold": "買いスコア閾値",
        "exit": "退出スコア閾値",
        "overview": "現在の総合判断",
        "details": "6分野スコア",
        "history": "価格と過去シグナル",
        "backtest": "バックテスト",
        "indicators": "指標詳細",
        "returns": "累積リターン",
        "cagr": "年率リターン",
        "volatility": "年率ボラティリティ",
        "drawdown": "最大ドローダウン",
        "sharpe": "シャープレシオ",
        "exposure": "保有比率",
        "trades": "取引回数",
        "date": "データ日付",
        "warning": "投資研究目的のみ。投資助言ではありません。",
    },
}


# ============================================================
# 2. 数据获取
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(symbol, start, end):
    """
    获取复权 OHLCV 数据。
    end 在 yfinance 中为不包含的日期。
    """
    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[data["Close"] > 0]
    data = data.sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data[~data.index.duplicated(keep="last")]


def safe_signal(condition, valid):
    """
    True = +1, False = -1, 数据无效 = NaN。
    """
    result = pd.Series(
        np.where(condition, 1.0, -1.0),
        index=condition.index,
    )
    return result.where(valid)


def mean_signals(signals):
    """
    每个维度内部的信号平均值，范围 -1 至 +1。
    缺失信号不参与计算。
    """
    return pd.concat(signals, axis=1).mean(axis=1)


# ============================================================
# 3. 技术指标计算
# ============================================================

def calculate_indicators(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ---------- 均线 ----------
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # ---------- MACD ----------
    ema12 = close.ewm(
        span=12, adjust=False, min_periods=26
    ).mean()
    ema26 = close.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ---------- RSI (Wilder) ----------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    df["RSI"] = rsi

    # ---------- 中期 / 长期收益 ----------
    df["Return63"] = close.pct_change(63, fill_method=None)
    df["Return126"] = close.pct_change(126, fill_method=None)
    df["Return252"] = close.pct_change(252, fill_method=None)

    # ---------- 12-1 动量 ----------
    # 排除最近约一个月，减少短期反转干扰。
    df["Momentum12_1"] = (
        close.shift(21) / close.shift(252) - 1
    )

    # ---------- ATR ----------
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    df["ATR_Pct"] = df["ATR14"] / close

    # ---------- 波动率 ----------
    daily_returns = close.pct_change(fill_method=None)

    df["Daily_Return"] = daily_returns
    df["Volatility63"] = (
        daily_returns.rolling(63).std() * np.sqrt(252)
    )

    # ---------- 过去 252 日回撤 ----------
    rolling_high = close.rolling(252).max()
    df["Drawdown252"] = close / rolling_high - 1

    # ---------- OBV ----------
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume.fillna(0)).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # ---------- 成交量 ----------
    df["Volume_MA20"] = volume.rolling(20).mean()

    # ---------- 突破 ----------
    # 使用前一日之前的历史区间，避免当前价格参与阈值计算。
    df["High55"] = high.shift(1).rolling(55).max()
    df["Low55"] = low.shift(1).rolling(55).min()

    return df


# ============================================================
# 4. 六维度信号
# ============================================================

def calculate_scores(asset, reference, market):
    df = asset.copy()

    close = df["Close"]

    # ---------- 1. 趋势：25% ----------
    # 长期价格位置与均线方向。
    trend_signals = [
        safe_signal(
            close > df["SMA200"],
            df["SMA200"].notna(),
        ),
        safe_signal(
            df["SMA50"] > df["SMA200"],
            df["SMA50"].notna() & df["SMA200"].notna(),
        ),
        safe_signal(
            df["SMA200"] > df["SMA200"].shift(20),
            df["SMA200"].shift(20).notna(),
        ),
    ]

    df["trend"] = mean_signals(trend_signals)

    # ---------- 2. 动量：20% ----------
    # 采用不同时间尺度，避免 Momentum 和 ROC 重复。
    momentum_signals = [
        safe_signal(
            df["Momentum12_1"] > 0,
            df["Momentum12_1"].notna(),
        ),
        safe_signal(
            df["MACD_Hist"] > 0,
            df["MACD_Hist"].notna(),
        ),
        safe_signal(
            df["RSI"] > 50,
            df["RSI"].notna(),
        ),
    ]

    df["momentum"] = mean_signals(momentum_signals)

    # ---------- 3. 相对强弱：15% ----------
    # 比较标的和基准的相同周期收益。
    reference_close = reference["Close"].reindex(df.index)

    ref_return63 = reference_close.pct_change(
        63, fill_method=None
    )
    ref_return126 = reference_close.pct_change(
        126, fill_method=None
    )

    df["Relative63"] = df["Return63"] - ref_return63
    df["Relative126"] = df["Return126"] - ref_return126

    relative_signals = [
        safe_signal(
            df["Relative63"] > 0,
            df["Relative63"].notna(),
        ),
        safe_signal(
            df["Relative126"] > 0,
            df["Relative126"].notna(),
        ),
    ]

    df["relative"] = mean_signals(relative_signals)

    # ---------- 4. 量价：10% ----------
    # 成交量本身不代表上涨，必须结合价格方向。
    volume_valid = (
        df["Volume_MA20"].notna()
        & (df["Volume_MA20"] > 0)
        & (df["Volume"] > 0)
    )

    volume_ratio = df["Volume"] / df["Volume_MA20"]
    price_change = close.diff()

    # 放量上涨 +1，放量下跌 -1，其他情况 0。
    volume_confirmation = pd.Series(
        0.0, index=df.index
    )

    volume_confirmation = volume_confirmation.mask(
        (volume_ratio > 1.2) & (price_change > 0),
        1.0,
    )
    volume_confirmation = volume_confirmation.mask(
        (volume_ratio > 1.2) & (price_change < 0),
        -1.0,
    )
    volume_confirmation = volume_confirmation.where(volume_valid)

    obv_signal = safe_signal(
        df["OBV"] > df["OBV_MA20"],
        df["OBV_MA20"].notna() & volume_valid,
    )

    df["volume"] = mean_signals(
        [volume_confirmation, obv_signal]
    )

    # ---------- 5. 风险质量：20% ----------
    # 风险不是方向预测，而是风险条件是否适宜。
    # 以下阈值为初始研究假设，并非通用最优值。
    volatility = df["Volatility63"]
    atr_pct = df["ATR_Pct"]
    drawdown = df["Drawdown252"]

    risk_signals = [
        safe_signal(
            volatility < 0.35,
            volatility.notna(),
        ),
        safe_signal(
            atr_pct < 0.04,
            atr_pct.notna(),
        ),
        safe_signal(
            drawdown > -0.25,
            drawdown.notna(),
        ),
    ]

    df["risk"] = mean_signals(risk_signals)

    # ---------- 6. 市场环境：10% ----------
    market_close = market["Close"].reindex(df.index)
    market_sma200 = market_close.rolling(200).mean()
    market_return63 = market_close.pct_change(
        63, fill_method=None
    )

    regime_signals = [
        safe_signal(
            market_close > market_sma200,
            market_sma200.notna(),
        ),
        safe_signal(
            market_return63 > 0,
            market_return63.notna(),
        ),
    ]

    df["regime"] = mean_signals(regime_signals)

    # ---------- 综合评分 ----------
    # 只有六个维度全部有效才发布综合信号。
    valid = df[DIMENSIONS].notna().all(axis=1)

    weighted = sum(
        df[key] * weight
        for key, weight in WEIGHTS.items()
    )

    df["Score"] = (weighted * 100).where(valid)
    df["Score"] = df["Score"].clip(-100, 100)

    # ---------- 风险保护 ----------
    df["Risk_Block"] = (
        (df["risk"] < -0.5)
        | (df["regime"] < -0.5)
    )

    return df


# ============================================================
# 5. 交易信号与仓位
# ============================================================

def generate_positions(
    df,
    entry_threshold=35,
    exit_threshold=0,
):
    """
    长仓 / 现金策略。

    1 = 满仓
    0 = 现金

    使用滞后阈值，避免在单一门槛附近反复交易。
    """
    result = df.copy()

    positions = []
    position = 0

    for _, row in result.iterrows():
        score = row["Score"]

        if pd.isna(score):
            position = 0

        elif bool(row["Risk_Block"]):
            position = 0

        elif position == 0 and score >= entry_threshold:
            position = 1

        elif position == 1 and score <= exit_threshold:
            position = 0

        positions.append(position)

    result["Target_Position"] = positions

    # 显示给用户的信号是收盘后的目标仓位。
    # 实际交易必须等下一交易日。
    return result


def current_recommendation(df, labels):
    latest = df.iloc[-1]

    if pd.isna(latest["Score"]):
        return "数据不足", "无法生成可靠信号"

    score = latest["Score"]

    if bool(latest["Risk_Block"]):
        return labels["avoid"], "风险或市场环境触发保护条件"

    if score >= 35:
        return labels["buy"], "综合评分较高"

    if score >= 0:
        return labels["hold"], "信号尚未形成明显优势"

    if score >= -35:
        return labels["reduce"], "综合信号偏弱"

    return labels["avoid"], "多个维度出现负面信号"


# ============================================================
# 6. 无未来函数的开盘执行回测
# ============================================================

def run_backtest(df, cost_bps=10):
    """
    t-1 收盘信号决定 t 日开盘持仓。

    使用开盘成交和收盘估值：
      昨收 -> 今开：原持仓承担隔夜收益
      今开 -> 今收：新持仓承担日内收益

    成本按开盘交易时的仓位变化计入。
    该模型假设可以按开盘价成交，未建模滑点、
    市场冲击、税务、分红现金流与融资成本。
    """
    result = df.copy()

    open_price = result["Open"]
    close_price = result["Close"]

    overnight_return = (
        open_price / close_price.shift(1) - 1
    )
    intraday_return = close_price / open_price - 1

    # 前一日收盘决定今天开盘的新仓位。
    new_position = (
        result["Target_Position"]
        .shift(1)
        .fillna(0)
        .astype(float)
    )

    # 今天开盘之前持有的仓位。
    old_position = new_position.shift(1).fillna(0)

    turnover = (new_position - old_position).abs()

    transaction_cost = turnover * cost_bps / 10000

    # 分段复利，成本在开盘扣除。
    overnight_factor = 1 + old_position * overnight_return
    intraday_factor = 1 + new_position * intraday_return

    strategy_factor = (
        overnight_factor
        * (1 - transaction_cost)
        * intraday_factor
    )

    strategy_return = strategy_factor - 1

    # 首日没有可用的前收盘价，不纳入回测。
    strategy_return.iloc[0] = 0.0

    result["Executed_Position"] = new_position
    result["Turnover"] = turnover
    result["Strategy_Return"] = strategy_return

    result["Strategy_Equity"] = (
        (1 + strategy_return).cumprod()
    )

    # 买入持有基准，从第一个可用收盘价开始。
    buy_hold_return = close_price.pct_change(
        fill_method=None
    ).fillna(0)

    result["BuyHold_Return"] = buy_hold_return
    result["BuyHold_Equity"] = (
        (1 + buy_hold_return).cumprod()
    )

    return result


# ============================================================
# 7. 绩效统计
# ============================================================

def performance_metrics(
    returns,
    equity,
    positions=None,
    turnover=None,
):
    returns = returns.dropna()

    if returns.empty or equity.empty:
        return {}

    n = len(returns)
    years = n / 252

    final_equity = float(equity.iloc[-1])
    total_return = final_equity - 1

    cagr = (
        final_equity ** (1 / years) - 1
        if years > 0 and final_equity > 0
        else np.nan
    )

    annual_vol = returns.std() * np.sqrt(252)

    # 无风险利率暂设为 0。
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    result = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": annual_vol,
        "max_drawdown": drawdown.min(),
        "sharpe": sharpe,
    }

    if positions is not None:
        result["exposure"] = positions.mean()

    if turnover is not None:
        result["trades"] = int((turnover > 0).sum())

    return result


# ============================================================
# 8. 界面
# ============================================================

language = st.sidebar.selectbox(
    "语言 / Language",
    ["中文", "日本語"],
)

t = LABELS[language]

st.title(t["title"])
st.caption(t["subtitle"])

st.sidebar.header(t["settings"])

ticker = st.sidebar.text_input(
    t["ticker"],
    value="QQQ",
).strip().upper()

reference_symbol = st.sidebar.text_input(
    t["reference"],
    value="SPY",
).strip().upper()

market_symbol = "SPY"

today = dt.date.today()

start_date = st.sidebar.date_input(
    t["start"],
    value=today - dt.timedelta(days=365 * 3),
)

end_date = st.sidebar.date_input(
    t["end"],
    value=today,
)

cost_bps = st.sidebar.slider(
    t["cost"],
    min_value=0,
    max_value=100,
    value=10,
    step=1,
)

entry_threshold = st.sidebar.slider(
    t["threshold"],
    min_value=0,
    max_value=80,
    value=35,
    step=5,
)

exit_threshold = st.sidebar.slider(
    t["exit"],
    min_value=-80,
    max_value=20,
    value=0,
    step=5,
)

if not ticker or not reference_symbol:
    st.error("请输入有效的标的代码。")
    st.stop()

if start_date >= end_date:
    st.error("开始日期必须早于结束日期。")
    st.stop()

if exit_threshold >= entry_threshold:
    st.error("退出门槛必须低于买入门槛。")
    st.stop()

# 为 252 日指标和长期均线预留足够预热期。
fetch_start = start_date - dt.timedelta(days=650)

# 包含用户选择的结束日期。
fetch_end = end_date + dt.timedelta(days=1)

with st.spinner("正在下载行情并计算多因子评分..."):
    try:
        asset_data = download_prices(
            ticker, fetch_start, fetch_end
        )

        reference_data = download_prices(
            reference_symbol, fetch_start, fetch_end
        )

        market_data = download_prices(
            market_symbol, fetch_start, fetch_end
        )

    except Exception as exc:
        st.error(f"行情下载失败：{exc}")
        st.stop()

if (
    asset_data.empty
    or reference_data.empty
    or market_data.empty
):
    st.error(
        "行情数据不足或下载失败。请检查代码、日期和网络。"
    )
    st.stop()

# 对齐日期。不同交易所的休市日期可能不同。
common_dates = (
    asset_data.index
    .intersection(reference_data.index)
    .intersection(market_data.index)
)

asset_data = asset_data.loc[common_dates]
reference_data = reference_data.loc[common_dates]
market_data = market_data.loc[common_dates]

if len(asset_data) < 300:
    st.error(
        "可用交易日不足 300 天，无法可靠计算长期指标。"
    )
    st.stop()

asset = calculate_indicators(asset_data)

asset = calculate_scores(
    asset,
    reference_data,
    market_data,
)

asset = generate_positions(
    asset,
    entry_threshold=entry_threshold,
    exit_threshold=exit_threshold,
)

# 先在完整预热数据上计算指标，再裁剪展示区间。
df = asset.loc[
    pd.Timestamp(start_date):pd.Timestamp(end_date)
].copy()

if df.empty:
    st.error("所选区间内没有数据。")
    st.stop()

valid_scores = df["Score"].dropna()

if valid_scores.empty:
    st.error(
        "当前日期范围没有完整的六维度评分，请扩大时间范围。"
    )
    st.stop()

latest = df.iloc[-1]

recommendation, reason = current_recommendation(df, t)

# ============================================================
# 9. 当前综合判断
# ============================================================

st.subheader(t["overview"])

st.caption(
    f"{t['date']}：{latest.name.strftime('%Y-%m-%d')}"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        t["price"],
        f"{latest['Close']:,.2f}",
    )

with c2:
    st.metric(
        t["score"],
        (
            f"{latest['Score']:+.1f} / 100"
            if pd.notna(latest["Score"])
            else "N/A"
        ),
    )

with c3:
    st.metric(
        "Signal",
        recommendation,
    )

with c4:
    st.metric(
        "Target Position",
        f"{latest['Target_Position']:.0%}",
    )

st.info(reason)

st.caption(
    "目标仓位是策略在最新收盘后计算的状态，"
    "不代表已经成交，也不代表个性化投资建议。"
)

# ============================================================
# 10. 六维度评分
# ============================================================

st.subheader(t["details"])

score_rows = []

for key, weight in WEIGHTS.items():
    value = latest[key]

    score_rows.append({
        "维度 / Factor": t[key],
        "权重 / Weight": f"{weight:.0%}",
        "原始评分 / Raw": (
            round(float(value), 3)
            if pd.notna(value)
            else np.nan
        ),
        "加权贡献 / Contribution": (
            round(float(value * weight * 100), 2)
            if pd.notna(value)
            else np.nan
        ),
    })

score_table = pd.DataFrame(score_rows)

st.dataframe(
    score_table,
    use_container_width=True,
    hide_index=True,
)

fig_factor = go.Figure()

fig_factor.add_trace(
    go.Bar(
        x=[t[k] for k in DIMENSIONS],
        y=[
            latest[k] * WEIGHTS[k] * 100
            if pd.notna(latest[k])
            else 0
            for k in DIMENSIONS
        ],
        marker_color=[
            "#28a745"
            if pd.notna(latest[k]) and latest[k] >= 0
            else "#dc3545"
            for k in DIMENSIONS
        ],
    )
)

fig_factor.update_layout(
    title="Weighted Factor Contributions",
    yaxis_title="Points",
    template="plotly_white",
    height=350,
)

st.plotly_chart(fig_factor, use_container_width=True)

# ============================================================
# 11. 价格与历史信号
# ============================================================

st.subheader(t["history"])

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        name=t["price"],
        line=dict(color="#0071e3", width=2),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA50"],
        name="SMA50",
        line=dict(color="#ff9500", width=1),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA200"],
        name="SMA200",
        line=dict(color="#8e44ad", width=1),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Score"],
        name=t["score"],
        fill="tozeroy",
        line=dict(color="#34c759", width=2),
    ),
    row=2,
    col=1,
)

fig.add_hline(
    y=entry_threshold,
    line_dash="dash",
    line_color="green",
    row=2,
    col=1,
)

fig.add_hline(
    y=exit_threshold,
    line_dash="dash",
    line_color="red",
    row=2,
    col=1,
)

fig.update_layout(
    height=650,
    template="plotly_white",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 12. 历史回测
# ============================================================

st.subheader(t["backtest"])

# 回测必须从所选区间开始，且第一天无前收盘，
# 因此将首日作为初始估值点。
backtest_df = run_backtest(
    df,
    cost_bps=cost_bps,
)

strategy_metrics = performance_metrics(
    backtest_df["Strategy_Return"],
    backtest_df["Strategy_Equity"],
    backtest_df["Executed_Position"],
    backtest_df["Turnover"],
)

hold_metrics = performance_metrics(
    backtest_df["BuyHold_Return"],
    backtest_df["BuyHold_Equity"],
)

comparison = pd.DataFrame(
    {
        t["strategy"]: {
            t["returns"]: strategy_metrics["total_return"],
            t["cagr"]: strategy_metrics["cagr"],
            t["volatility"]: strategy_metrics["volatility"],
            t["drawdown"]: strategy_metrics["max_drawdown"],
            t["sharpe"]: strategy_metrics["sharpe"],
        },
        t["benchmark"]: {
            t["returns"]: hold_metrics["total_return"],
            t["cagr"]: hold_metrics["cagr"],
            t["volatility"]: hold_metrics["volatility"],
            t["drawdown"]: hold_metrics["max_drawdown"],
            t["sharpe"]: hold_metrics["sharpe"],
        },
    }
)

st.dataframe(
    comparison.style.format("{:.2%}", subset=[
        t["returns"],
        t["cagr"],
        t["volatility"],
        t["drawdown"],
    ]).format("{:.2f}", subset=[t["sharpe"]]),
    use_container_width=True,
)

m1, m2 = st.columns(2)

with m1:
    st.metric(
        t["exposure"],
        f"{strategy_metrics['exposure']:.1%}",
    )

with m2:
    st.metric(
        t["trades"],
        str(strategy_metrics["trades"]),
    )

fig_equity = go.Figure()

fig_equity.add_trace(
    go.Scatter(
        x=backtest_df.index,
        y=backtest_df["Strategy_Equity"],
        name=t["strategy"],
        line=dict(color="#0071e3", width=2),
    )
)

fig_equity.add_trace(
    go.Scatter(
        x=backtest_df.index,
        y=backtest_df["BuyHold_Equity"],
        name=t["benchmark"],
        line=dict(color="#ff9500", width=2),
    )
)

fig_equity.update_layout(
    title="Strategy vs Buy & Hold",
    yaxis_title="Growth of 1 Unit",
    template="plotly_white",
    height=450,
    hovermode="x unified",
)

st.plotly_chart(fig_equity, use_container_width=True)

# ============================================================
# 13. 指标明细
# ============================================================

with st.expander(t["indicators"]):
    indicator_columns = [
        "Close",
        "SMA20",
        "SMA50",
        "SMA200",
        "RSI",
        "MACD_Hist",
        "Momentum12_1",
        "Relative63",
        "Relative126",
        "ATR_Pct",
        "Volatility63",
        "Drawdown252",
        "Score",
        "Target_Position",
    ]

    st.dataframe(
        df[indicator_columns].tail(100).round(4),
        use_container_width=True,
    )

    csv = df.to_csv().encode("utf-8-sig")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{ticker}_factor_analysis.csv",
        mime="text/csv",
    )

# ============================================================
# 14. 风险提示
# ============================================================

st.warning(t["warning"])

st.caption(
    "本系统采用公开行情数据及研究性规则。"
    "指标之间仍可能存在相关性，"
    "评分权重和风险阈值未经样本外验证。"
    "历史回测不代表未来表现。"
    "不包含完整的交易税费、真实滑点、"
    "市场冲击和个人投资约束。"
)
