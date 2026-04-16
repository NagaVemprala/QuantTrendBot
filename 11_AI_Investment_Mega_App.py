import os
import json
import textwrap
import streamlit as st
import yfinance as yf
import pandas as pd
import talib as ta
from dataclasses import dataclass, asdict
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# --- Page Config ---
st.set_page_config(page_title="AI Quant Advisor", layout="wide")

# --- 1. Schemas & Indicators ---

@dataclass
class InvestmentSummary:
    symbol: str
    current_price: float
    strategy_signal: str 
    technical_analysis: str
    recommendation: str

def _get_clean_series(df: pd.DataFrame, column: str):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    actual_col = next((c for c in df.columns if str(c).lower() == column.lower()), None)
    if actual_col is None:
        raise KeyError(f"Column '{column}' not found.")
    return df[actual_col].values.flatten().astype(float)

# --- 2. Technical Tools (Tools remain identical to your source) ---
# [Keep your tool definitions here: analyze_macd_strategy, analyze_rsi_vo_strategy, etc.]
# I am omitting the full tool bodies for brevity, but they should be pasted here exactly as you have them.

@tool("fetch_stock_data")
def fetch_stock_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical stock data (Open, High, Low, Close, Volume) using yfinance."""
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# analyze_macd_strategy, analyze_rsi_vo_strategy, 
# analyze_bollinger_band_strategy, analyze_sma_crossover_strategy, 
# analyze_adx_dmi_strategy, analyze_stochastic_strategy, analyze_obv_strategy as tools...

@tool("analyze_macd_strategy", description="Calculates MACD and returns latest signal and values.")
def analyze_macd_strategy(symbol: str, start_date: str) -> dict:
    """
    Analyzes MACD for the given symbol and returns the latest signal and indicator values.
     - BUY if MACD crosses above signal line (bullish crossover)
     - SELL if MACD crosses below signal line (bearish crossover)
     - NEUTRAL if no crossover
     """
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
    """
    Analyzes RSI and Volume Oscillator for the given symbol and returns the latest signal and indicator values.
        - BUY if RSI crosses above 30 with positive volume oscillator (momentum recovery)
        - SELL if RSI crosses below 70 (overbought exit)
            - NEUTRAL otherwise
    """
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
    """
    Analyzes Bollinger Bands for the given symbol and returns the latest signal and indicator values.
     - BUY if price is below lower band (potential mean reversion)
     - SELL if price is above upper band (potential breakout)
     - NEUTRAL if price is within bands
     """
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
    """
    Analyzes 50-day and 200-day SMA crossover for the given symbol and returns the latest signal and indicator values.
     - BUY if 50-day SMA crosses above 200-day SMA (golden cross)
     - SELL if 50-day SMA crosses below 200-day SMA (death cross)
     - NEUTRAL if no crossover
    """
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
    """
    Analyzes ADX with +DI and -DI for the given symbol and returns the latest signal and indicator values.
     - BUY if ADX >= 25 and +DI > -DI (strong uptrend)
     - SELL if ADX >= 25 and -DI > +DI (strong downtrend
     - NEUTRAL if ADX < 25 (weak or no trend)
     """
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
    """
    Analyzes Stochastic Oscillator for the given symbol and returns the latest signal and indicator values.
        - BUY if %K crosses above %D below 25 (oversold bullish reversal)
        - SELL if %K crosses below %D above 75 (overbought bearish reversal)
        - NEUTRAL otherwise
    """
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
    """
    Analyzes On-Balance Volume trend confirmation for the given symbol and returns the latest signal and indicator values.
        - BUY if OBV crosses above its 10-day SMA (bullish volume confirmation)
        - SELL if OBV crosses below its 10-day SMA (bearish volume confirmation)
        - NEUTRAL if no crossover
    """
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


# --- 3. Streamlit UI Layout ---

st.title("📈 AI Quantitative Investment Advisor")
st.markdown("""
This agent uses **LangGraph** and **TA-Lib** to perform multi-layer technical analysis.
It combines Trend, Momentum, Volatility, and Volume indicators to reach a consensus.
""")

# Initialize the session state for the query if it doesn't exist
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

with st.sidebar:
    st.header("Settings")
    # api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    # api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.secrets["OPENAI_API_KEY"]
    model_choice = st.selectbox("Model", ["gpt-4o", "gpt-5-nano"])
    start_date = st.date_input("Analysis Start Date", value=pd.to_datetime("2025-01-01"))
    
    st.divider()
    st.markdown("### Recommended Queries")
    q1 = "Analyze NVDA using MACD and RSI for momentum signals."
    q2 = "Provide a trend confirmation for GE using SMA Crossover and ADX."
    q3 = "Is TSLA oversold? Check Bollinger Bands and Stochastics."
    
    if st.button("Example 1: Momentum"): 
        st.session_state.user_query = q1
    if st.button("Example 2: Trend"): 
        st.session_state.user_query = q2
    if st.button("Example 3: Volatility"): 
        st.session_state.user_query = q3

# The text_input is now linked to the session state 'user_query'
user_query = st.text_input(
    "Enter your investment query:", 
    key="user_query" # This links the input to st.session_state.user_query
)
# --- 4. Agent Initialization ---

if api_key:
    my_model = init_chat_model(model_choice, temperature=0.1, api_key=api_key)
    
    # Checkpointer for conversation memory
    if "memory" not in st.session_state:
        st.session_state.memory = InMemorySaver()

    streamlit_system_prompt = st.secrets["SYSTEM_PROMPT"]
    trading_agent = create_agent(
        model=my_model,
        tools=[
            fetch_stock_data, analyze_macd_strategy, analyze_rsi_vo_strategy,
            analyze_bollinger_band_strategy, analyze_sma_crossover_strategy,
            analyze_adx_dmi_strategy, analyze_stochastic_strategy, analyze_obv_strategy,
        ],
        system_prompt=streamlit_system_prompt,
        checkpointer=st.session_state.memory,
        response_format=InvestmentSummary,
    )

    # --- 5. Main Chat Interface ---

    user_query = st.text_input("Enter your investment query:", key="query_input", 
                               value=st.session_state.get('query', ''))

    if st.button("Run Analysis"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing market data and indicators..."):
                config = {'configurable': {'thread_id': 'streamlit_user'}}
                try:
                    response = trading_agent.invoke(
                        {"messages": [{"role": "user", "content": f"{user_query} starting from {start_date}"}]},
                        config=config
                    )
                    
                    # Handling Structured Output
                    last_msg = response["messages"][-1]
                    
                    # Logic to extract content
                    try:
                        data = json.loads(last_msg.content)
                    except:
                        data = response.get("structured_response")
                        if hasattr(data, "__dataclass_fields__"):
                            data = asdict(data)

                    # --- Display Results ---
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Symbol", data.get("symbol"))
                    with col2:
                        st.metric("Price", f"${data.get('current_price'):.2f}")
                    with col3:
                        signal = data.get("strategy_signal")
                        color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
                        st.markdown(f"### Signal: :{color}[{signal}]")

                    st.subheader("Technical Analysis")
                    st.info(data.get("technical_analysis"))

                    st.subheader("Final Recommendation")
                    st.success(data.get("recommendation"))

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
else:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")