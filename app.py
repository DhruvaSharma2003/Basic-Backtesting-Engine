import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from web3 import Web3
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings("ignore")

# --- Secrets ---
DEFAULT_INFURA = st.secrets["INFURA_URL"]
DEFAULT_PRIVATE_KEY = st.secrets["PRIVATE_KEY"]

# --- Constants ---
CRYPTO_OPTIONS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Tether (USDT)": "tether",
    "XRP (XRP)": "ripple",
    "BNB (BNB)": "binancecoin",
    "Solana (SOL)": "solana",
    "USD Coin (USDC)": "usd-coin",
    "TRON (TRX)": "tron",
    "Dogecoin (DOGE)": "dogecoin",
    "Cardano (ADA)": "cardano",
}

# --- Functions ---
def fetch_ohlc_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=30"
    response = requests.get(url)
    if response.status_code == 200:
        raw = response.json()
        if len(raw) == 0:
            raise Exception("Empty OHLC response from API")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        raise Exception("Failed to fetch OHLC data")

def apply_strategy(df, strategy, **kwargs):
    df = df.copy()
    df['signal'] = 0
    df['buy_signal'] = np.nan
    df['sell_signal'] = np.nan

    if strategy == "Simple Moving Average (SMA)":
        short_window = kwargs.get("short_window", 5)
        long_window = kwargs.get("long_window", 15)
        df['SMA_short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=long_window).mean()
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
        df.loc[df['SMA_short'] <= df['SMA_long'], 'signal'] = -1

    elif strategy == "Bollinger Bands":
        period = kwargs.get("period", 20)
        std_dev = kwargs.get("std_dev", 2.0)
        df['MA'] = df['close'].rolling(window=period).mean()
        df['BB_up'] = df['MA'] + std_dev * df['close'].rolling(window=period).std()
        df['BB_down'] = df['MA'] - std_dev * df['close'].rolling(window=period).std()
        df.loc[df['close'] < df['BB_down'], 'signal'] = 1
        df.loc[df['close'] > df['BB_up'], 'signal'] = -1

    df['buy_signal'] = np.where((df['signal'] == 1) & (df['signal'].shift(1) != 1), df['close'], np.nan)
    df['sell_signal'] = np.where((df['signal'] == -1) & (df['signal'].shift(1) != -1), df['close'], np.nan)

    return df

def forecast_prices(df, days_to_display, model_type='ARIMA'):
    y = df['close'] if 'close' in df.columns else df['price']
    y.index = pd.to_datetime(y.index)

    if len(y) < 20:
        st.warning("Not enough data points to forecast.")
        return None, None

    model = None
    try:
        if model_type == 'AR':
            model = AutoReg(y, lags=5).fit()
        elif model_type == 'IMA':
            model = ARIMA(y, order=(0, 1, 1)).fit()
        else:
            model = ARIMA(y, order=(5, 1, 0)).fit()

        total_days = 7
        forecast = model.forecast(steps=total_days)
        forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=total_days)
        forecast_series = pd.Series(forecast.values, index=forecast_index)

        selected_forecast = forecast_series.iloc[:days_to_display]

        st.subheader("Forecasted Values")
        st.dataframe(selected_forecast.rename("Forecast Price"))

        return y, selected_forecast

    except Exception as e:
        st.error(f"Forecasting Error: {str(e)}")
        return None, None

def simulate_trade(account, w3):
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        tx = {
            'nonce': nonce,
            'to': account.address,
            'value': 0,
            'gas': 21000,
            'gasPrice': w3.to_wei('10', 'gwei'),
            'chainId': 11155111
        }
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return w3.to_hex(tx_hash)
    except Exception as e:
        st.error(f"Transaction Failed: {str(e)}")
        return None

def load_data(file):
    data = pd.read_csv(file, parse_dates=['timestamp'])
    if 'price' not in data.columns:
        raise KeyError("Column 'price' not found in uploaded file. Make sure your CSV contains 'timestamp' and 'price' columns.")
    data.set_index('timestamp', inplace=True)
    return data

def backtest(data, initial_capital, fee, max_trades, batch_size):
    capital = initial_capital
    positions = []
    portfolio = []
    buy_signals = []
    sell_signals = []

    for index, row in data.iterrows():
        if row['signal'] == 1 and len(positions) < max_trades:
            trade_amount = capital * batch_size
            if trade_amount > 0:
                position = {
                    'amount': trade_amount / row['price'],
                    'entry_price': row['price']
                }
                positions.append(position)
                capital -= trade_amount
                capital *= (1 - fee)
                buy_signals.append((index, row['price']))
            else:
                buy_signals.append((index, None))
        else:
            buy_signals.append((index, None))

        if row['signal'] == -1 and positions:
            sell_price = row['price']
            for position in positions:
                capital += position['amount'] * sell_price
                capital *= (1 - fee)
            positions = []
            sell_signals.append((index, sell_price))
        else:
            sell_signals.append((index, None))

        portfolio_value = capital + sum(p['amount'] * row['price'] for p in positions)
        portfolio.append(portfolio_value)

    data['portfolio'] = portfolio
    data['buy_signal'] = [x[1] for x in buy_signals]
    data['sell_signal'] = [x[1] for x in sell_signals]
    return data

def evaluate_performance(data, initial_capital):
    final_value = data['portfolio'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    max_drawdown = (data['portfolio'] / data['portfolio'].cummax() - 1).min() * 100
    daily_returns = data['portfolio'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0
    return total_return, max_drawdown, sharpe_ratio

# --- Streamlit App ---
st.title("\ud83d\udd0d Crypto Backtesting Engine + Web3 Execution")

mode = st.radio("Select Mode", ["Live Data", "Historical Data Upload"])

strategy_choice = st.sidebar.selectbox("Select Strategy", ["Simple Moving Average (SMA)", "Bollinger Bands"])

if strategy_choice == "Simple Moving Average (SMA)":
    short_window = st.sidebar.number_input("Short SMA", min_value=1, value=5)
    long_window = st.sidebar.number_input("Long SMA", min_value=1, value=15)
    strategy_params = {"short_window": short_window, "long_window": long_window}
elif strategy_choice == "Bollinger Bands":
    bb_period = st.sidebar.number_input("Period", min_value=1, value=20)
    bb_std_dev = st.sidebar.number_input("Std Dev Multiplier", min_value=0.1, value=2.0)
    strategy_params = {"period": bb_period, "std_dev": bb_std_dev}

forecast_days = st.sidebar.slider("Days to Forecast", 1, 7, 1)
model_choice = st.sidebar.selectbox("Forecasting Model", ["AR", "IMA", "ARIMA"])

if mode == "Historical Data Upload":
    with st.sidebar:
        st.markdown("<div style='border:2px solid #1f77b4; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
        initial_capital = st.number_input("Initial Capital", min_value=1, value=10000)
        fee = st.number_input("Transaction Fee (%)", min_value=0.0, value=0.1) / 100
        max_trades = st.number_input("Max Trades", min_value=1, value=1)
        batch_size = st.slider("Batch Size (%)", min_value=1, max_value=100, value=20) / 100
        st.markdown("</div>", unsafe_allow_html=True)

if mode == "Live Data":
    coin_name = st.selectbox("Choose Crypto", list(CRYPTO_OPTIONS.keys()))
    coin_id = CRYPTO_OPTIONS[coin_name]

    try:
        df = fetch_ohlc_data(coin_id)
        df = apply_strategy(df, strategy_choice, **strategy_params)
        actual_series, forecast_series = forecast_prices(df, forecast_days, model_type=model_choice)

        fig = go.Figure(data=[
            go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'),
            go.Scatter(x=df.index, y=df['buy_signal'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')),
            go.Scatter(x=df.index, y=df['sell_signal'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')),
        ])
        if strategy_choice == "Simple Moving Average (SMA)":
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', name='SMA Short'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines', name='SMA Long'))
        elif strategy_choice == "Bollinger Bands":
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_up'], mode='lines', name='Upper Band', line=dict(color='blue', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_down'], mode='lines', name='Lower Band', line=dict(color='blue', dash='dot')))

        if forecast_series is not None:
            fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines+markers', name='Forecast', line=dict(dash='dot', color='orange')))

        fig.update_layout(title=f"Candlestick Chart + {strategy_choice} + Forecast for {coin_name}", xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        if df['signal'].iloc[-1] == 1:
            if st.button("\ud83d\ude80 BUY Signal Detected: Execute Trade"):
                w3 = Web3(Web3.HTTPProvider(DEFAULT_INFURA))
                account = w3.eth.account.from_key(DEFAULT_PRIVATE_KEY)
                tx_hash = simulate_trade(account, w3)
                if tx_hash:
                    st.success(f"TX Hash: {tx_hash}")
                    st.markdown(f"[View on Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})")
        else:
            st.info("No Buy Signal at the latest point.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

elif mode == "Historical Data Upload":
    uploaded_file = st.file_uploader("Upload Historical CSV (timestamp, price)", type="csv")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            df = apply_strategy(df, strategy_choice, **strategy_params)
            df = backtest(df, initial_capital, fee, max_trades, batch_size)
            actual_series, forecast_series = forecast_prices(df, forecast_days, model_type=model_choice)

            st.subheader("Strategy Results on Uploaded Data")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['buy_signal'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)))
            fig.add_trace(go.Scatter(x=df.index, y=df['sell_signal'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))
            if strategy_choice == "Simple Moving Average (SMA)":
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', name='SMA Short'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines', name='SMA Long'))
            elif strategy_choice == "Bollinger Bands":
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_up'], mode='lines', name='Upper Band', line=dict(color='blue', dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_down'], mode='lines', name='Lower Band', line=dict(color='blue', dash='dot')))

            if forecast_series is not None:
                fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines+markers', name='Forecast', line=dict(dash='dot', color='orange')))

            fig.update_layout(title="Backtest + Forecast with Buy/Sell Signals", xaxis_title='Time', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

            total_return, max_drawdown, sharpe_ratio = evaluate_performance(df, initial_capital)
            st.metric("Total Return (%)", f"{total_return:.2f}")
            st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        except Exception as e:
            st.error(f"File Error: {str(e)}")
