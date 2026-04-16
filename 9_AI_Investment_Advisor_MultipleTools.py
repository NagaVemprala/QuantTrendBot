import os
import json
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Optional

import yfinance as yf
import pandas as pd
import talib as ta
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# --- 1. Schemas & Context ---

@dataclass
class UserInvestmentContext:
    risk_tolerance: str
    preferred_sectors: List[str]


@dataclass
class InvestmentSummary:
    symbol: str
    current_price: float
    strategy_signal: str  # "BUY", "SELL", or "WAIT"
    technical_analysis: str
    recommendation: str


def print_investment_summary(data: dict):
    """Pretty-print investment summary to the console."""
    symbol = data.get("symbol", "N/A")
    current_price = data.get("current_price", "N/A")
    strategy_signal = data.get("strategy_signal", "N/A")
    technical_analysis = data.get("technical_analysis", "N/A")
    recommendation = data.get("recommendation", "N/A")

    print("\n" + "═" * 90)
    print(f"{'INVESTMENT SUMMARY':^90}")
    print("═" * 90)
    print(f"{'Symbol':25}: {symbol}")
    print(f"{'Current Price':25}: {current_price}")
    print(f"{'Overall Signal':25}: {strategy_signal}")
    print("─" * 90)
    print("TECHNICAL ANALYSIS")
    print("─" * 90)
    print(textwrap.fill(str(technical_analysis), width=88))
    print("─" * 90)
    print("RECOMMENDATION")
    print("─" * 90)
    print(textwrap.fill(str(recommendation), width=88))
    print("═" * 90 + "\n")


def _format_dict_pretty(title: str, data: dict) -> str:
    lines = [title]
    for k, v in data.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


# --- Helper to get clean price/volume series ---
def _get_clean_series(df: pd.DataFrame, column: str):
    """Ensure we have a 1D float64 numpy array for TA-Lib, handling yfinance MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    actual_col = None
    for c in df.columns:
        if str(c).lower() == column.lower():
            actual_col = c
            break

    if actual_col is None:
        raise KeyError(f"Column '{column}' not found in DataFrame. Available: {df.columns.tolist()}")

    return df[actual_col].values.flatten().astype(float)


def _get_latest_price(df: pd.DataFrame) -> float:
    close = _get_clean_series(df, "Close")
    return float(close[-1])


# --- 2. Technical Tools ---

@tool("fetch_stock_data", description="Fetches historical stock data from Yahoo Finance.")
def fetch_stock_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


@tool("analyze_macd_strategy", description="Calculates MACD and returns latest signal and values.")
def analyze_macd_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        close = _get_clean_series(df, 'Close')

        if len(close) < 35:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for MACD calculation."}

        macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        if len(macd) < 2 or pd.isna(macd[-1]) or pd.isna(signal[-1]) or pd.isna(macd[-2]) or pd.isna(signal[-2]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_macd, last_signal = float(macd[-1]), float(signal[-1])
        prev_macd, prev_signal = float(macd[-2]), float(signal[-2])

        if last_macd > last_signal and prev_macd <= prev_signal:
            trade_signal = "BUY"
            label = "BULLISH CROSSOVER"
        elif last_macd < last_signal and prev_macd >= prev_signal:
            trade_signal = "SELL"
            label = "BEARISH CROSSOVER"
        else:
            trade_signal = "NEUTRAL"
            label = "NO CROSSOVER"

        return {
            "signal": trade_signal,
            "label": label,
            "macd": last_macd,
            "signal_line": last_signal,
            "histogram": float(hist[-1]) if not pd.isna(hist[-1]) else None,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_rsi_vo_strategy", description="Analyzes RSI and Volume Oscillator using TA-Lib.")
def analyze_rsi_vo_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        close = _get_clean_series(df, 'Close')
        volume = _get_clean_series(df, 'Volume')

        if len(close) < 30 or len(volume) < 30:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for RSI/VO calculation."}

        rsi = ta.RSI(close, timeperiod=14)
        short_vol = ta.SMA(volume, timeperiod=5)
        long_vol = ta.SMA(volume, timeperiod=20)
        vol_osc = (short_vol - long_vol) / (long_vol + 1e-8)

        if len(rsi) < 2 or pd.isna(rsi[-1]) or pd.isna(vol_osc[-1]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready yet."}

        last_rsi = float(rsi[-1])
        prev_rsi = float(rsi[-2]) if not pd.isna(rsi[-2]) else last_rsi
        last_vo = float(vol_osc[-1])

        if prev_rsi < 30 and last_rsi >= 30 and last_vo > 0:
            trade_signal = "BUY"
            label = "MOMENTUM RECOVERY"
        elif prev_rsi >= 70 and last_rsi < 70:
            trade_signal = "SELL"
            label = "OVERBOUGHT EXIT"
        else:
            trade_signal = "NEUTRAL"
            label = "NO ACTION"

        return {
            "signal": trade_signal,
            "label": label,
            "rsi": last_rsi,
            "volume_oscillator": last_vo,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_bollinger_band_strategy", description="Analyzes Bollinger Bands for mean reversion and breakout context.")
def analyze_bollinger_band_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        close = _get_clean_series(df, "Close")

        if len(close) < 25:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for Bollinger Bands calculation."}

        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        if pd.isna(upper[-1]) or pd.isna(middle[-1]) or pd.isna(lower[-1]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_close = float(close[-1])
        last_upper = float(upper[-1])
        last_middle = float(middle[-1])
        last_lower = float(lower[-1])

        bandwidth = (last_upper - last_lower) / (last_middle + 1e-8)
        position = (last_close - last_lower) / ((last_upper - last_lower) + 1e-8)

        if last_close < last_lower:
            trade_signal = "BUY"
            label = "PRICE BELOW LOWER BAND"
        elif last_close > last_upper:
            trade_signal = "SELL"
            label = "PRICE ABOVE UPPER BAND"
        else:
            trade_signal = "NEUTRAL"
            label = "PRICE WITHIN BANDS"

        return {
            "signal": trade_signal,
            "label": label,
            "close": last_close,
            "upper_band": last_upper,
            "middle_band": last_middle,
            "lower_band": last_lower,
            "bandwidth": float(bandwidth),
            "band_position": float(position),
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_sma_crossover_strategy", description="Analyzes 50-day and 200-day SMA crossover for trend direction.")
def analyze_sma_crossover_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        close = _get_clean_series(df, "Close")

        if len(close) < 210:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for 50/200 SMA crossover calculation."}

        sma50 = ta.SMA(close, timeperiod=50)
        sma200 = ta.SMA(close, timeperiod=200)

        if pd.isna(sma50[-1]) or pd.isna(sma200[-1]) or pd.isna(sma50[-2]) or pd.isna(sma200[-2]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_sma50 = float(sma50[-1])
        last_sma200 = float(sma200[-1])
        prev_sma50 = float(sma50[-2])
        prev_sma200 = float(sma200[-2])
        last_close = float(close[-1])

        if last_sma50 > last_sma200 and prev_sma50 <= prev_sma200:
            trade_signal = "BUY"
            label = "GOLDEN CROSS"
        elif last_sma50 < last_sma200 and prev_sma50 >= prev_sma200:
            trade_signal = "SELL"
            label = "DEATH CROSS"
        elif last_sma50 > last_sma200:
            trade_signal = "BUY"
            label = "UPTREND CONFIRMED"
        elif last_sma50 < last_sma200:
            trade_signal = "SELL"
            label = "DOWNTREND CONFIRMED"
        else:
            trade_signal = "NEUTRAL"
            label = "TREND UNCLEAR"

        return {
            "signal": trade_signal,
            "label": label,
            "close": last_close,
            "sma_50": last_sma50,
            "sma_200": last_sma200,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_adx_dmi_strategy", description="Analyzes ADX with +DI and -DI to assess trend strength and direction.")
def analyze_adx_dmi_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        high = _get_clean_series(df, "High")
        low = _get_clean_series(df, "Low")
        close = _get_clean_series(df, "Close")

        if len(close) < 30:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for ADX/DMI calculation."}

        adx = ta.ADX(high, low, close, timeperiod=14)
        plus_di = ta.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = ta.MINUS_DI(high, low, close, timeperiod=14)

        if pd.isna(adx[-1]) or pd.isna(plus_di[-1]) or pd.isna(minus_di[-1]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_adx = float(adx[-1])
        last_plus = float(plus_di[-1])
        last_minus = float(minus_di[-1])

        if last_adx >= 25 and last_plus > last_minus:
            trade_signal = "BUY"
            label = "STRONG UPTREND"
        elif last_adx >= 25 and last_minus > last_plus:
            trade_signal = "SELL"
            label = "STRONG DOWNTREND"
        else:
            trade_signal = "NEUTRAL"
            label = "WEAK OR SIDEWAYS TREND"

        return {
            "signal": trade_signal,
            "label": label,
            "adx": last_adx,
            "+di": last_plus,
            "-di": last_minus,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_stochastic_strategy", description="Analyzes Stochastic Oscillator for overbought/oversold reversals.")
def analyze_stochastic_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        high = _get_clean_series(df, "High")
        low = _get_clean_series(df, "Low")
        close = _get_clean_series(df, "Close")

        if len(close) < 20:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for Stochastic calculation."}

        slowk, slowd = ta.STOCH(
            high, low, close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )

        if pd.isna(slowk[-1]) or pd.isna(slowd[-1]) or pd.isna(slowk[-2]) or pd.isna(slowd[-2]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_k = float(slowk[-1])
        last_d = float(slowd[-1])
        prev_k = float(slowk[-2])
        prev_d = float(slowd[-2])

        if prev_k <= prev_d and last_k > last_d and last_k < 25:
            trade_signal = "BUY"
            label = "BULLISH OVERSOLD CROSS"
        elif prev_k >= prev_d and last_k < last_d and last_k > 75:
            trade_signal = "SELL"
            label = "BEARISH OVERBOUGHT CROSS"
        else:
            trade_signal = "NEUTRAL"
            label = "NO ACTION"

        return {
            "signal": trade_signal,
            "label": label,
            "%K": last_k,
            "%D": last_d,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


@tool("analyze_obv_strategy", description="Analyzes On-Balance Volume trend confirmation.")
def analyze_obv_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
        close = _get_clean_series(df, "Close")
        volume = _get_clean_series(df, "Volume")

        if len(close) < 25 or len(volume) < 25:
            return {"signal": "NEUTRAL", "reason": "Insufficient data for OBV calculation."}

        obv = ta.OBV(close, volume)
        obv_ma = ta.SMA(obv, timeperiod=10)

        if pd.isna(obv[-1]) or pd.isna(obv_ma[-1]) or pd.isna(obv[-2]) or pd.isna(obv_ma[-2]):
            return {"signal": "NEUTRAL", "reason": "Indicator data not ready."}

        last_obv = float(obv[-1])
        last_obv_ma = float(obv_ma[-1])
        prev_obv = float(obv[-2])
        prev_obv_ma = float(obv_ma[-2])

        if last_obv > last_obv_ma and prev_obv <= prev_obv_ma:
            trade_signal = "BUY"
            label = "BULLISH VOLUME CONFIRMATION"
        elif last_obv < last_obv_ma and prev_obv >= prev_obv_ma:
            trade_signal = "SELL"
            label = "BEARISH VOLUME CONFIRMATION"
        else:
            trade_signal = "NEUTRAL"
            label = "VOLUME TREND STABLE"

        return {
            "signal": trade_signal,
            "label": label,
            "obv": last_obv,
            "obv_ma_10": last_obv_ma,
        }
    except Exception as e:
        return {"signal": "ERROR", "reason": str(e)}


# --- 3. Agent Configuration ---
checkpointerMemory = InMemorySaver()

my_model = init_chat_model(
    "gpt-5-nano",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

trading_agent = create_agent(
    model=my_model,
    tools=[
        fetch_stock_data,
        analyze_macd_strategy,
        analyze_rsi_vo_strategy,
        analyze_bollinger_band_strategy,
        analyze_sma_crossover_strategy,
        analyze_adx_dmi_strategy,
        analyze_stochastic_strategy,
        analyze_obv_strategy,
    ],
    system_prompt=(
        "You are a Quantitative Investment Advisor. Your goal is to summarize technical indicators "
        "and provide a clear investment recommendation (Invest, Wait, or Sell). "
        "Use 'analyze_macd_strategy' for trend following and 'analyze_rsi_vo_strategy' for momentum. "
        "You may also use 'analyze_bollinger_band_strategy', 'analyze_sma_crossover_strategy', "
        "'analyze_adx_dmi_strategy', 'analyze_stochastic_strategy', and 'analyze_obv_strategy' "
        "to improve signal quality. "
        "Preserve MACD and RSI/VO in your reasoning, and use additional tools as confirmation layers. "
        "Always provide a detailed technical summary referencing indicator values, signal conflicts, "
        "trend strength, momentum, and volume confirmation. "
        "Return a structured response matching the InvestmentSummary schema. "
        "For strategy_signal, use BUY, SELL, or WAIT only. "
        "Make the technical_analysis and recommendation elaborate, polished, and neatly organized."
    ),
    checkpointer=checkpointerMemory,
    response_format=InvestmentSummary,
)

# --- 4. Execution ---
config_settings = {'configurable': {'thread_id': 'investment_chat_01'}}
query = "Should I invest in GE? Use the data starting from 2025-01-01."

response = trading_agent.invoke(
    {
        "messages": [{"role": "user", "content": query}]
    },
    config=config_settings
)

# --- 5. Output Handling ---
last_message = response["messages"][-1]

if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
    print("\n[INFO] Agent is still processing tool calls.\n")
else:
    content = last_message.content

    try:
        data = json.loads(content)
        print_investment_summary(data)

    except (json.JSONDecodeError, TypeError):
        structured = response.get("structured_response")

        if structured:
            if hasattr(structured, "__dataclass_fields__"):
                print_investment_summary(asdict(structured))
            elif isinstance(structured, dict):
                print_investment_summary(structured)
            else:
                print("\n" + "═" * 90)
                print(f"{'INVESTMENT SUMMARY':^90}")
                print("═" * 90)
                print(textwrap.fill(str(structured), width=88))
                print("═" * 90 + "\n")
        else:
            print("\n" + "═" * 90)
            print(f"{'INVESTMENT SUMMARY':^90}")
            print("═" * 90)
            print(textwrap.fill(str(content), width=88))
            print("═" * 90 + "\n")