import os
import json
import yfinance as yf
import pandas as pd
import talib as ta
from dataclasses import dataclass, asdict
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
# -----------------
# These are the imports specific to formatting output 
import textwrap
# -----------------
from dotenv import load_dotenv

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

    print("\n" + "=" * 60)
    print(f"INVESTMENT SUMMARY: {symbol}")
    print("=" * 60)
    print(f"{'Current Price:':20} {current_price}")
    print(f"{'Signal:':20} {strategy_signal}")
    print(f"{'Technical Analysis:':20} {textwrap.fill(str(technical_analysis), width=55, subsequent_indent=' ' * 22)}")
    print(f"{'Recommendation:':20} {textwrap.fill(str(recommendation), width=55, subsequent_indent=' ' * 22)}")
    print("=" * 60 + "\n")

# --- Helper to get clean price/volume series ---
def _get_clean_series(df: pd.DataFrame, column: str):
    """Ensure we have a 1D float64 numpy array for TA-Lib, handling yfinance MultiIndex."""
    
    # 1. Handle MultiIndex columns by flattening them
    # yfinance often returns columns like [('Close', 'NVDA'), ('Open', 'NVDA')]
    if isinstance(df.columns, pd.MultiIndex):
        # We take the first level (the metric name) and discard the ticker name
        df.columns = df.columns.get_level_values(0)
    
    # 2. Case-insensitive check (sometimes it's 'close' vs 'Close')
    actual_col = None
    for c in df.columns:
        if str(c).lower() == column.lower():
            actual_col = c
            break
            
    if actual_col is None:
        raise KeyError(f"Column '{column}' not found in DataFrame. Available: {df.columns.tolist()}")

    # 3. Extract, flatten, and convert to float64
    return df[actual_col].values.flatten().astype(float)


# --- 2. Technical Tools ---

@tool('fetch_stock_data', description='Fetches historical stock data from Yahoo Finance.')
def fetch_stock_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    # Standardize
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

@tool("analyze_macd_strategy", description="Calculates MACD and returns latest signal and values.")
def analyze_macd_strategy(symbol: str, start_date: str) -> dict:
    try:
        df = yf.download(symbol, start=start_date, progress=False)
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
        df = yf.download(symbol, start=start_date, progress=False)
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


# --- 3. Agent Configuration ---

checkpointerMemory = InMemorySaver()
my_model = init_chat_model("gpt-5-nano", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))

trading_agent = create_agent(
    model=my_model,
    tools=[fetch_stock_data, analyze_macd_strategy, analyze_rsi_vo_strategy],
    system_prompt=(
        "You are a Quantitative Investment Advisor. Your goal is to summarize technical indicators "
        "and provide a clear investment recommendation (Invest, Wait, or Sell). "
        "Use 'analyze_macd_strategy' for trend following and 'analyze_rsi_vo_strategy' for momentum. "
        "Always provide a summary of the technical values and a final decision."
    ),
    checkpointer=checkpointerMemory,
    response_format=InvestmentSummary,
)

# --- 4. Execution ---

config_settings = {'configurable': {'thread_id': 'investment_chat_01'}}
query = "Should I invest in LLY? Use the data starting from 2025-01-01."

response = trading_agent.invoke({
    "messages": [{"role": "user", "content": query}]
}, config=config_settings)

# Output
# Extract the last message
last_message = response["messages"][-1]

if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
    print("\n[INFO] Agent is still processing tool calls.\n")
else:
    content = last_message.content

    try:
        # Case 1: JSON string output
        data = json.loads(content)
        print_investment_summary(data)

    except (json.JSONDecodeError, TypeError):
        # Case 2: If response content is not JSON, try structured response
        structured = response.get("structured_response")

        if structured:
            if hasattr(structured, "__dataclass_fields__"):
                print_investment_summary(asdict(structured))
            elif isinstance(structured, dict):
                print_investment_summary(structured)
            else:
                print("\n--- Investment Summary ---")
                print(structured)
                print()
        else:
            print("\n--- Investment Summary ---")
            print(textwrap.fill(str(content), width=80))
            print()