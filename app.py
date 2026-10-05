import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# 页面配置
st.set_page_config(
    page_title="多指标量化看板与历史回测", page_icon="📈", layout="wide"
)

st.title("📊 金融标的技术指标量化看板与历史信号分析")
st.markdown(
    "通过计算 **12个经典技术指标** 的多空状态（牛市/上涨信号为 +1，熊市/下跌信号为 -1），"
    "实时评估当前市场情绪，并可回溯历史指标得分与价格的走势关系。"
)

# ----------------- 侧边栏配置 -----------------
st.sidebar.header("参数配置")
default_ticker = "QQQ"
ticker = (
    st.sidebar.text_input("输入标的代码 (Ticker)", value=default_ticker)
    .upper()
    .strip()
)

# 默认半年前到今天
default_start = datetime.date.today() - datetime.timedelta(days=180)
default_end = datetime.date.today()

start_date = st.sidebar.date_input("开始日期", value=default_start)
end_date = st.sidebar.date_input("结束日期", value=default_end)

if start_date >= end_date:
  st.sidebar.error("错误：开始日期必须早于结束日期！")
  st.stop()


# ----------------- 数据获取与指标计算函数 -----------------
@st.cache_data(ttl=3600)
def load_and_calculate_data(ticker_symbol, start, end):
  # 额外多取一些历史数据用于均线计算（如200日均线）
  fetch_start = pd.to_datetime(start) - pd.Timedelta(days=300)
  df = yf.download(
      ticker_symbol, start=fetch_start, end=end, progress=False, auto_adjust=True
  )

  if df.empty or len(df) < 50:
    return None

  # 适配多级表头
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
  df.dropna(inplace=True)

  # --- 计算 12 个技术指标 ---
  df["SMA20"] = df["Close"].rolling(window=20).mean()
  df["Ind_SMA20"] = np.where(df["Close"] > df["SMA20"], 1, -1)

  df["SMA50"] = df["Close"].rolling(window=50).mean()
  df["Ind_SMA50"] = np.where(df["Close"] > df["SMA50"], 1, -1)

  df["SMA200"] = df["Close"].rolling(window=200).mean()
  df["Ind_SMA200"] = np.where(df["Close"] > df["SMA200"], 1, -1)

  df["Ind_SMA_Cross"] = np.where(df["SMA20"] > df["SMA50"], 1, -1)

  df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
  df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
  df["Ind_EMA"] = np.where(df["EMA12"] > df["EMA26"], 1, -1)

  df["MACD_Line"] = df["EMA12"] - df["EMA26"]
  df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]
  df["Ind_MACD_Hist"] = np.where(df["MACD_Hist"] > 0, 1, -1)

  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))
  df["Ind_RSI"] = np.where(df["RSI"] > 50, 1, -1)

  df["BB_Middle"] = df["Close"].rolling(window=20).mean()
  df["Ind_BB"] = np.where(df["Close"] > df["BB_Middle"], 1, -1)

  df["Mom"] = df["Close"].diff(10)
  df["Ind_Mom"] = np.where(df["Mom"] > 0, 1, -1)

  df["ROC"] = df["Close"].pct_change(10) * 100
  df["Ind_ROC"] = np.where(df["ROC"] > 0, 1, -1)

  df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()
  df["Ind_Vol"] = np.where(df["Volume"] > df["Vol_SMA20"], 1, -1)

  df["High20"] = df["High"].rolling(window=20).max()
  df["Low20"] = df["Low"].rolling(window=20).min()
  df["Ind_Channel"] = np.where(
      df["Close"] > (df["High20"] + df["Low20"]) / 2, 1, -1
  )

  ind_cols = [
      ("收盘价 > SMA20", "Ind_SMA20"),
      ("收盘价 > SMA50", "Ind_SMA50"),
      ("收盘价 > SMA200", "Ind_SMA200"),
      ("SMA20 > SMA50 (均线多头)", "Ind_SMA_Cross"),
      ("EMA12 > EMA26", "Ind_EMA"),
      ("MACD 柱状图 > 0", "Ind_MACD_Hist"),
      ("RSI(14) > 50", "Ind_RSI"),
      ("收盘价 > 布林带中轨", "Ind_BB"),
      ("10日动量 Momentum > 0", "Ind_Mom"),
      ("10日变化率 ROC > 0", "Ind_ROC"),
      ("成交量 > 20日均量", "Ind_Vol"),
      ("价格处于20日通道中上轨", "Ind_Channel"),
  ]

  score_df = pd.DataFrame(index=df.index)
  for name, col in ind_cols:
    score_df[name] = df[col]

  df["Net_Score"] = score_df.sum(axis=1)
  df["Total_Indicators"] = len(ind_cols)

  # 裁剪用户选择的日期区间
  df = df.loc[pd.to_datetime(start) : pd.to_datetime(end)]
  score_df = score_df.loc[pd.to_datetime(start) : pd.to_datetime(end)]

  return df, score_df, ind_cols


# 加载数据
with st.spinner(f"正在获取 {ticker} 数据并计算技术指标..."):
  result = load_and_calculate_data(ticker, start_date, end_date)

if result is None:
  st.error(
      f"未能获取到标的 **{ticker}** 的有效数据，请检查代码拼写或尝试其他标的。"
  )
  st.stop()

df, score_df, ind_cols = result

if df.empty:
  st.warning("所选日期范围内没有足够的数据，请调整开始和结束日期。")
  st.stop()


# ----------------- 1. 历史趋势双图展示（置顶） -----------------
st.subheader(f"📈 {ticker} 历史指标净得分与价格走势")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        f"{ticker} 资产价格走势（背景红绿分界）",
        "多空净指标得分历史（上涨指标数 - 下跌指标数）",
    ),
)

# 动态计算连续的红/绿区间，用于在价格子图背景中渲染色块
df["Color_State"] = np.where(df["Net_Score"] >= 0, "Green", "Red")
df["Block"] = (df["Color_State"] != df["Color_State"].shift()).cumsum()

for _, group in df.groupby("Block"):
  start_t = group.index[0]
  end_t = group.index[-1]
  state = group["Color_State"].iloc[0]

  # 修复：防止单日区间宽度为0无法渲染，统一向后延伸一天确保色块能正常显示
  end_t = end_t + pd.Timedelta(days=1)

  fill_color = (
      "rgba(40, 167, 69, 0.2)" if state == "Green" else "rgba(220, 53, 69, 0.2)"
  )
  fig.add_vrect(
      x0=start_t,
      x1=end_t,
      fillcolor=fill_color,
      opacity=1,
      layer="below",
      line_width=0,
      row=1,
      col=1,
  )

# 子图 1：价格折线图
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="收盘价 (Close)",
        line=dict(color="#1f77b4", width=2),
    ),
    row=1,
    col=1,
)

# 子图 2：净得分柱状图
bar_colors = ["#28a745" if val >= 0 else "#dc3545" for val in df["Net_Score"]]
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Net_Score"],
        name="净得分 (Score)",
        marker_color=bar_colors,
    ),
    row=2,
    col=1,
)

fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
)

fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
fig.update_yaxes(title_text="净得分", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ----------------- 2. 最新指标状态展示（放下方） -----------------
st.subheader(f"📌 {ticker} 最新技术指标状态面板")
latest_date = df.index[-1].strftime("%Y-%m-%d")
st.caption(f"数据截止日期：{latest_date}")

latest_scores = score_df.iloc[-1]

cols = st.columns(4)
for i, (name, col_key) in enumerate(ind_cols):
  val = latest_scores[name]
  with cols[i % 4]:
    if val > 0:
      st.markdown(
          f"""
                <div style="padding: 10px; border-radius: 6px; background-color: rgba(0, 255, 0, 0.08); border-left: 5px solid #28a745; margin-bottom: 10px;">
                    <strong style="font-size: 14px;">{name}</strong><br>
                    <span style="color: #28a745; font-weight: bold; font-size: 16px;">🟢 上涨信号 (+1)</span>
                </div>
                """,
          unsafe_allow_html=True,
      )
    else:
      st.markdown(
          f"""
                <div style="padding: 10px; border-radius: 6px; background-color: rgba(255, 0, 0, 0.08); border-left: 5px solid #dc3545; margin-bottom: 10px;">
                    <strong style="font-size: 14px;">{name}</strong><br>
                    <span style="color: #dc3545; font-weight: bold; font-size: 16px;">🔴 下跌信号 (-1)</span>
                </div>
                """,
          unsafe_allow_html=True,
      )

total_net = latest_scores.sum()
total_count = len(ind_cols)
bullish_count = int((total_net + total_count) / 2)
bearish_count = total_count - bullish_count

st.info(
    f"💡 **综合多空盘点**：在全部 **{total_count}** 个指标中，当前共有 **{bullish_count}** 个看涨指标、"
    f"**{bearish_count}** 个看跌指标。净得分为 **{int(total_net)}**（满分 +{total_count} / 最低 -{total_count}）。"
)
