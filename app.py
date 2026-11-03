
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# ============================================================
# 1. 基础配置
# ============================================================

st.set_page_config(
    page_title="多因子投资分析 V3",
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

I18N = {
    "中文": {
        "title": "多因子投资分析 V3",
        "subtitle": "六维雷达图 · 信号解释 · 历史回测",
        "trend": "趋势",
        "momentum": "动量",
        "relative": "相对强弱",
        "volume": "量价确认",
        "risk": "风险质量",
        "regime": "市场环境",
        "settings": "参数设置",
        "ticker": "标的代码",
        "reference": "比较基准",
        "start": "开始日期",
        "end": "结束日期",
        "cost": "单边成本（基点）",
        "entry": "入场阈值",
        "exit": "退出阈值",
        "overview": "最新信号",
        "radar": "六维度雷达图",
        "history": "历史价格与评分",
        "backtest": "策略回测",
        "detail": "指标明细",
        "buy": "偏向买入",
        "hold": "持有 / 观察",
        "reduce": "偏向减仓",
        "avoid": "风险规避",
        "unavailable": "数据不足",
        "price": "收盘价",
        "score": "综合评分",
        "position": "下一开盘目标仓位",
        "strategy": "策略",
        "baseline": "买入持有",
        "total": "累计收益",
        "annual": "年化收益",
        "vol": "年化波动",
        "drawdown": "最大回撤",
        "sharpe": "夏普比率",
        "exposure": "平均仓位",
        "trades": "交易次数",
        "buy_mark": "买入",
        "sell_mark": "卖出",
    },
    "日本語": {
        "title": "マルチファクター投資分析 V3",
        "subtitle": "6軸レーダー · シグナル解説 · バックテスト",
        "trend": "トレンド",
        "momentum": "モメンタム",
        "relative": "相対強度",
        "volume": "出来高と価格",
        "risk": "リスク品質",
        "regime": "市場環境",
        "settings": "設定",
        "ticker": "銘柄コード",
        "reference": "比較ベンチマーク",
        "start": "開始日",
        "end": "終了日",
        "cost": "片道コスト（bp）",
        "entry": "エントリー閾値",
        "exit": "退出閾値",
        "overview": "最新シグナル",
        "radar": "6軸レーダーチャート",
        "history": "価格とスコア",
        "backtest": "バックテスト",
        "detail": "指標詳細",
        "buy": "買い優勢",
        "hold": "保有・様子見",
        "reduce": "縮小優勢",
        "avoid": "リスク回避",
        "unavailable": "データ不足",
        "price": "終値",
        "score": "総合スコア",
        "position": "翌営業日寄付き目標保有率",
        "strategy": "戦略",
        "baseline": "バイ＆ホールド",
        "total": "累積収益",
        "annual": "年率収益",
        "vol": "年率変動率",
        "drawdown": "最大下落率",
        "sharpe": "シャープ比率",
        "exposure": "平均保有率",
        "trades": "取引回数",
        "buy_mark": "買い",
        "sell_mark": "売り",
    },
}


# ============================================================
# 2. Yahoo Finance 数据下载
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download(symbol, start, end):
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

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if any(c not in data.columns for c in required):
        return pd.DataFrame()

    data = data[required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    data = data[
        (data["Open"] > 0)
        & (data["Close"] > 0)
    ].sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data[
        ~data.index.duplicated(keep="last")
    ]


# ============================================================
# 3. 信号工具函数
# ============================================================

def signal(condition, valid):
    """
    看涨：+1
    看跌：-1
    无效数据：NaN
    """
    return pd.Series(
        np.where(condition, 1.0, -1.0),
        index=condition.index,
    ).where(valid)


def average(*values):
    """
    对同一维度的多个信号取平均。
    """
    return pd.concat(
        values,
        axis=1,
    ).mean(axis=1)


# ============================================================
# 4. 技术指标
# ============================================================

def indicators(data):
    d = data.copy()

    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    # 长期趋势
    d["SMA50"] = c.rolling(50).mean()
    d["SMA200"] = c.rolling(200).mean()

    # MACD
    e12 = c.ewm(
        span=12,
        adjust=False,
        min_periods=26,
    ).mean()

    e26 = c.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    macd = e12 - e26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    d["MACD_Hist"] = macd - macd_signal

    # RSI
    diff = c.diff()

    up = diff.clip(lower=0).ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    down = (-diff.clip(upper=0)).ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rsi = 100 - 100 / (
        1 + up / down.replace(0, np.nan)
    )

    d["RSI"] = (
        rsi
        .mask((down == 0) & (up > 0), 100)
        .mask((up == 0) & (down > 0), 0)
        .mask((up == 0) & (down == 0), 50)
    )

    # 收益率
    d["Ret63"] = c.pct_change(
        63,
        fill_method=None,
    )

    d["Ret126"] = c.pct_change(
        126,
        fill_method=None,
    )

    # 12-1 动量
    d["Mom12_1"] = (
        c.shift(21) / c.shift(252) - 1
    )

    # ATR
    prev = c.shift(1)

    true_range = pd.concat(
        [
            h - l,
            (h - prev).abs(),
            (l - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    d["ATR_Pct"] = atr / c

    # 年化波动率
    daily_return = c.pct_change(
        fill_method=None
    )

    d["Vol63"] = (
        daily_return.rolling(63).std()
        * np.sqrt(252)
    )

    # 252 日回撤
    d["DD252"] = (
        c / c.rolling(252).max() - 1
    )

    # 成交量
    d["VolumeMA20"] = v.rolling(20).mean()

    # OBV
    direction = np.sign(
        c.diff()
    ).fillna(0)

    d["OBV"] = (
        direction * v.fillna(0)
    ).cumsum()

    d["OBVMA20"] = (
        d["OBV"].rolling(20).mean()
    )

    return d


# ============================================================
# 5. 六维度评分
# ============================================================

def score_factors(asset, reference, market):
    d = indicators(asset)

    c = d["Close"]

    # 1. 趋势 25%
    d["trend"] = average(
        signal(
            c > d["SMA200"],
            d["SMA200"].notna(),
        ),
        signal(
            d["SMA50"] > d["SMA200"],
            d["SMA200"].notna(),
        ),
        signal(
            d["SMA200"] > d["SMA200"].shift(20),
            d["SMA200"].shift(20).notna(),
        ),
    )

    # 2. 动量 20%
    d["momentum"] = average(
        signal(
            d["Mom12_1"] > 0,
            d["Mom12_1"].notna(),
        ),
        signal(
            d["MACD_Hist"] > 0,
            d["MACD_Hist"].notna(),
        ),
        signal(
            d["RSI"] > 50,
            d["RSI"].notna(),
        ),
    )

    # 3. 相对强弱 15%
    rc = reference["Close"].reindex(d.index)

    r63 = rc.pct_change(
        63,
        fill_method=None,
    )

    r126 = rc.pct_change(
        126,
        fill_method=None,
    )

    d["Relative63"] = d["Ret63"] - r63
    d["Relative126"] = d["Ret126"] - r126

    d["relative"] = average(
        signal(
            d["Relative63"] > 0,
            d["Relative63"].notna(),
        ),
        signal(
            d["Relative126"] > 0,
            d["Relative126"].notna(),
        ),
    )

    # 4. 量价确认 10%
    valid_volume = (
        (d["Volume"] > 0)
        & (d["VolumeMA20"] > 0)
        & d["VolumeMA20"].notna()
    )

    ratio = (
        d["Volume"] / d["VolumeMA20"]
    )

    volume_confirmation = pd.Series(
        0.0,
        index=d.index,
    )

    volume_confirmation = (
        volume_confirmation
        .mask(
            (ratio > 1.2) & (c.diff() > 0),
            1.0,
        )
        .mask(
            (ratio > 1.2) & (c.diff() < 0),
            -1.0,
        )
        .where(valid_volume)
    )

    obv_signal = signal(
        d["OBV"] > d["OBVMA20"],
        valid_volume & d["OBVMA20"].notna(),
    )

    d["volume"] = average(
        volume_confirmation,
        obv_signal,
    )

    # 5. 风险质量 20%
    # 风险评分不等于未来涨跌预测。
    d["risk"] = average(
        signal(
            d["Vol63"] < 0.35,
            d["Vol63"].notna(),
        ),
        signal(
            d["ATR_Pct"] < 0.04,
            d["ATR_Pct"].notna(),
        ),
        signal(
            d["DD252"] > -0.25,
            d["DD252"].notna(),
        ),
    )

    # 6. 市场环境 10%
    mc = market["Close"].reindex(d.index)

    msma = mc.rolling(200).mean()

    mret = mc.pct_change(
        63,
        fill_method=None,
    )

    d["regime"] = average(
        signal(
            mc > msma,
            msma.notna(),
        ),
        signal(
            mret > 0,
            mret.notna(),
        ),
    )

    # 综合评分
    valid = d[
        list(WEIGHTS)
    ].notna().all(axis=1)

    weighted = sum(
        d[k] * w
        for k, w in WEIGHTS.items()
    )

    d["Score"] = (
        weighted * 100
    ).where(valid)

    # 风险保护
    d["Risk_Block"] = (
        (d["risk"] < -0.5)
        | (d["regime"] < -0.5)
    )

    return d


# ============================================================
# 6. 持仓信号
# ============================================================

def positions(d, entry, exit_level):
    d = d.copy()

    state = 0
    target = []

    for score, blocked in zip(
        d["Score"],
        d["Risk_Block"],
    ):
        if pd.isna(score) or bool(blocked):
            state = 0

        elif state == 0 and score >= entry:
            state = 1

        elif state == 1 and score <= exit_level:
            state = 0

        target.append(state)

    d["Target_Position"] = target

    return d


# ============================================================
# 7. 历史回测
# ============================================================

def backtest(d, cost_bps):
    """
    初始为空仓。
    昨日收盘信号于今日开盘执行。
    """

    x = d.copy()

    old_close = x["Close"].shift(1)

    new = (
        x["Target_Position"]
        .shift(1)
        .fillna(0)
        .astype(float)
    )

    old = new.shift(1).fillna(0)

    turnover = (new - old).abs()

    overnight = (
        x["Open"] / old_close - 1
    )

    intraday = (
        x["Close"] / x["Open"] - 1
    )

    transaction_cost = (
        turnover * cost_bps / 10000
    )

    strategy = (
        (1 + old * overnight)
        * (1 - transaction_cost)
        * (1 + new * intraday)
        - 1
    )

    strategy.iloc[0] = 0.0

    hold = (
        x["Close"]
        .pct_change(fill_method=None)
        .fillna(0)
    )

    x["Executed_Position"] = new
    x["Turnover"] = turnover
    x["Strategy_Return"] = strategy
    x["Hold_Return"] = hold

    x["Strategy_Equity"] = (
        (1 + strategy).cumprod()
    )

    x["Hold_Equity"] = (
        (1 + hold).cumprod()
    )

    x["Trade_Change"] = new - old

    return x


# ============================================================
# 8. 绩效统计
# ============================================================

def metrics(ret, equity):
    ret = ret.dropna()

    n = len(ret)

    if n < 2:
        return None

    std = ret.std()

    final = float(equity.iloc[-1])

    dd = (
        equity / equity.cummax() - 1
    )

    return {
        "total": final - 1,
        "annual": (
            final ** (252 / n) - 1
            if final > 0
            else np.nan
        ),
        "vol": std * np.sqrt(252),
        "drawdown": float(dd.min()),
        "sharpe": (
            ret.mean() / std * np.sqrt(252)
            if std > 0
            else np.nan
        ),
    }


# ============================================================
# 9. 当前建议
# ============================================================

def recommendation(row, t, entry, exit_level):
    score = row["Score"]

    if pd.isna(score):
        return t["unavailable"]

    if bool(row["Risk_Block"]):
        return t["avoid"]

    if score >= entry:
        return t["buy"]

    if score <= exit_level:
        return t["reduce"]

    return t["hold"]


# ============================================================
# 10. Streamlit 参数
# ============================================================

lang = st.sidebar.selectbox(
    "语言 / Language",
    list(I18N),
)

t = I18N[lang]

st.title(t["title"])
st.caption(t["subtitle"])

st.sidebar.header(t["settings"])

ticker = st.sidebar.text_input(
    t["ticker"],
    "QQQ",
).strip().upper()

reference_symbol = st.sidebar.text_input(
    t["reference"],
    "SPY",
).strip().upper()

today = dt.date.today()

start = st.sidebar.date_input(
    t["start"],
    today - dt.timedelta(days=365 * 3),
)

end = st.sidebar.date_input(
    t["end"],
    today,
)

cost = st.sidebar.slider(
    t["cost"],
    0,
    100,
    10,
)

entry = st.sidebar.slider(
    t["entry"],
    0,
    80,
    35,
    5,
)

exit_level = st.sidebar.slider(
    t["exit"],
    -80,
    20,
    0,
    5,
)

if (
    not ticker
    or not reference_symbol
    or start >= end
    or exit_level >= entry
):
    st.error(
        "请检查代码、日期和阈值 / "
        "銘柄・日付・閾値を確認してください"
    )
    st.stop()


# ============================================================
# 11. 下载并处理行情
# ============================================================

with st.spinner("Loading market data..."):
    try:
        fetch_start = (
            start - dt.timedelta(days=850)
        )

        fetch_end = (
            end + dt.timedelta(days=1)
        )

        asset = download(
            ticker,
            fetch_start,
            fetch_end,
        )

        reference = download(
            reference_symbol,
            fetch_start,
            fetch_end,
        )

        market = download(
            "SPY",
            fetch_start,
            fetch_end,
        )

    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

if any(
    x.empty
    for x in [asset, reference, market]
):
    st.error(
        "行情下载失败 / データ取得失敗"
    )
    st.stop()

# 对齐共同交易日
common = (
    asset.index
    .intersection(reference.index)
    .intersection(market.index)
)

asset = asset.loc[common]
reference = reference.loc[common]
market = market.loc[common]

if len(asset) < 300:
    st.error(
        "共同交易日不足300天 / "
        "共通取引日が300日未満です"
    )
    st.stop()

full = positions(
    score_factors(
        asset,
        reference,
        market,
    ),
    entry,
    exit_level,
)

d = full.loc[
    pd.Timestamp(start):pd.Timestamp(end)
].copy()

if (
    len(d) < 2
    or d["Score"].notna().sum() < 2
):
    st.error(
        "分析区间数据不足 / "
        "分析期間のデータが不足しています"
    )
    st.stop()

latest = d.iloc[-1]


# ============================================================
# 12. 当前信号
# ============================================================

st.subheader(t["overview"])

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    t["price"],
    f"{latest['Close']:,.2f}",
)

c2.metric(
    t["score"],
    (
        f"{latest['Score']:+.1f}/100"
        if pd.notna(latest["Score"])
        else "N/A"
    ),
)

c3.metric(
    "Signal",
    recommendation(
        latest,
        t,
        entry,
        exit_level,
    ),
)

c4.metric(
    t["position"],
    f"{latest['Target_Position']:.0%}",
)

st.caption(
    f"{latest.name:%Y-%m-%d} · "
    "收盘后信号，下一交易日开盘执行 / "
    "終値後シグナル、翌営業日寄付きで執行"
)


# ============================================================
# 13. 六维雷达图
# ============================================================

st.subheader(t["radar"])

labels = [
    t[k]
    for k in WEIGHTS
]

# -1 -> 0分
#  0 -> 50分
# +1 -> 100分

values = [
    float((latest[k] + 1) * 50)
    if pd.notna(latest[k])
    else None
    for k in WEIGHTS
]

radar = go.Figure()

radar.add_trace(
    go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=t["score"],
        line=dict(
            color="#0071e3",
            width=3,
        ),
        fillcolor="rgba(0,113,227,0.20)",
        marker=dict(size=7),
    )
)

radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
        ),
    ),
    showlegend=False,
    height=450,
    margin=dict(
        l=35,
        r=35,
        t=35,
        b=35,
    ),
)

st.plotly_chart(
    radar,
    use_container_width=True,
)

st.caption(
    "雷达图展示各维度标准化分数（0–100），"
    "不包含权重。综合评分仍按照 "
    "25/20/15/10/20/10 加权。"
)

# 六维度详细评分
rows = []

for k, w in WEIGHTS.items():
    v = latest[k]

    rows.append({
        "Factor": t[k],
        "Weight": f"{w:.0%}",
        "Score (0-100)": (
            round((v + 1) * 50, 1)
            if pd.notna(v)
            else "N/A"
        ),
        "Contribution": (
            round(v * w * 100, 2)
            if pd.notna(v)
            else "N/A"
        ),
    })

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 14. 回测数据准备
# ============================================================

# 回测从用户选择的日期开始空仓。
# 不继承预热期仓位。
backtest_input = positions(
    d.drop(columns=["Target_Position"]),
    entry,
    exit_level,
)

b = backtest(
    backtest_input,
    cost,
)


# ============================================================
# 15. 历史价格与买卖信号
# ============================================================

st.subheader(t["history"])

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
)

# 价格和均线
for name, color in [
    ("Close", "#0071e3"),
    ("SMA50", "#ff9500"),
    ("SMA200", "#8e44ad"),
]:
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d[name],
            name=name,
            line=dict(
                color=color,
                width=2 if name == "Close" else 1,
            ),
        ),
        row=1,
        col=1,
    )

# 买入卖出标记
buys = b[
    b["Trade_Change"] > 0
]

sells = b[
    b["Trade_Change"] < 0
]

for part, label, symbol, color in [
    (
        buys,
        t["buy_mark"],
        "triangle-up",
        "#28a745",
    ),
    (
        sells,
        t["sell_mark"],
        "triangle-down",
        "#dc3545",
    ),
]:
    fig.add_trace(
        go.Scatter(
            x=part.index,
            y=part["Open"],
            mode="markers",
            name=label,
            marker=dict(
                symbol=symbol,
                size=11,
                color=color,
            ),
        ),
        row=1,
        col=1,
    )

# 历史综合评分
fig.add_trace(
    go.Scatter(
        x=d.index,
        y=d["Score"],
        name=t["score"],
        line=dict(
            color="#34c759",
            width=2,
        ),
    ),
    row=2,
    col=1,
)

# 入场与退出门槛
fig.add_hline(
    y=entry,
    line_dash="dash",
    line_color="green",
    row=2,
    col=1,
)

fig.add_hline(
    y=exit_level,
    line_dash="dash",
    line_color="red",
    row=2,
    col=1,
)

fig.update_layout(
    height=650,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# 16. 历史回测
# ============================================================

st.subheader(t["backtest"])

sm = metrics(
    b["Strategy_Return"],
    b["Strategy_Equity"],
)

hm = metrics(
    b["Hold_Return"],
    b["Hold_Equity"],
)

if sm and hm:

    # 修复 KeyError：
    # 不再使用 Styler.format(subset=...)。
    # 直接格式化数值后创建展示表格。

    records = []

    for key in [
        "total",
        "annual",
        "vol",
        "drawdown",
        "sharpe",
    ]:
        if key == "sharpe":
            def fmt(x):
                return (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "N/A"
                )
        else:
            def fmt(x):
                return (
                    f"{x:.2%}"
                    if pd.notna(x)
                    else "N/A"
                )

        records.append({
            "Metric": t[key],
            t["strategy"]: fmt(sm[key]),
            t["baseline"]: fmt(hm[key]),
        })

    comparison = pd.DataFrame(records)

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    m1, m2 = st.columns(2)

    m1.metric(
        t["exposure"],
        f"{b['Executed_Position'].mean():.1%}",
    )

    m2.metric(
        t["trades"],
        str(
            int(
                (b["Turnover"] > 0).sum()
            )
        ),
    )

    # 净值曲线
    equity = go.Figure()

    equity.add_trace(
        go.Scatter(
            x=b.index,
            y=b["Strategy_Equity"],
            name=t["strategy"],
            line=dict(
                color="#0071e3",
                width=2,
            ),
        )
    )

    equity.add_trace(
        go.Scatter(
            x=b.index,
            y=b["Hold_Equity"],
            name=t["baseline"],
            line=dict(
                color="#ff9500",
                width=2,
            ),
        )
    )

    equity.update_layout(
        template="plotly_white",
        height=430,
        hovermode="x unified",
        yaxis_title="Growth of 1",
    )

    st.plotly_chart(
        equity,
        use_container_width=True,
    )


# ============================================================
# 17. 指标明细与 CSV
# ============================================================

with st.expander(t["detail"]):

    columns = [
        "Close",
        "SMA50",
        "SMA200",
        "RSI",
        "MACD_Hist",
        "Mom12_1",
        "Relative63",
        "Relative126",
        "ATR_Pct",
        "Vol63",
        "DD252",
        *WEIGHTS,
        "Score",
        "Target_Position",
    ]

    st.dataframe(
        d[columns].tail(150).round(4),
        use_container_width=True,
    )

    st.download_button(
        label="Download CSV",
        data=d.to_csv().encode("utf-8-sig"),
        file_name=f"{ticker}_factor_v3.csv",
        mime="text/csv",
    )


# ============================================================
# 18. 风险提示
# ============================================================

st.warning(
    "研究工具，不构成投资建议。"
    "权重和阈值尚未经过样本外验证，"
    "历史回测不代表未来表现。"
    "回测不包含真实滑点、税务、市场冲击和现金利息。"
    " / "
    "研究用ツールであり、投資助言ではありません。"
)
