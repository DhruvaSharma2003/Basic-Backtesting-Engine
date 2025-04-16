import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
from web3 import Web3
import os

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
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        raise Exception("Failed to fetch OHLC data")

def apply_strategy(df, short_window, long_window):
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df.loc[df['SMA_short'] <= df['SMA_long'], 'signal'] = -1
    return df

def plot_candlestick_with_sma(df):
    apds = [
        mpf.make_addplot(df['SMA_short'], color='blue'),
        mpf.make_addplot(df['SMA_long'], color='orange')
    ]
    mpf.plot(df, type='candle', style='charles', addplot=apds, volume=False, mav=(None), returnfig=True)
    st.pyplot(mpf.plot(df, type='candle', style='charles', addplot=apds, returnfig=True)[0])

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
    data.set_index('timestamp', inplace=True)
    return data

def evaluate_performance(data, initial_capital):
    final_value = data['portfolio'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    max_drawdown = (data['portfolio'] / data['portfolio'].cummax() - 1).min() * 100
    daily_returns = data['portfolio'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return total_return, max_drawdown, sharpe_ratio

# --- Streamlit App ---
st.title("🔍 Crypto Backtesting Engine + Web3 Execution")

mode = st.radio("Select Mode", ["Live Data", "Historical Data Upload"])

short_window = st.sidebar.number_input("Short SMA", min_value=1, value=5)
long_window = st.sidebar.number_input("Long SMA", min_value=1, value=15)
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1, value=10000)
fee = st.sidebar.number_input("Transaction Fee (%)", min_value=0.0, value=0.1) / 100
max_trades = st.sidebar.number_input("Max Trades", min_value=1, value=1)
batch_size = st.sidebar.slider("Batch Size (%)", min_value=1, max_value=100, value=20) / 100

if mode == "Live Data":
    coin_name = st.selectbox("Choose Crypto", list(CRYPTO_OPTIONS.keys()))
    coin_id = CRYPTO_OPTIONS[coin_name]

    try:
        df = fetch_ohlc_data(coin_id)
        df = apply_strategy(df, short_window, long_window)
        st.subheader(f"Candlestick Chart with SMA for {coin_name}")
        plot_candlestick_with_sma(df)

        # Web3 trade trigger
        if df['signal'].iloc[-1] == 1:
            if st.button("🚀 BUY Signal Detected: Execute Trade"):
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
        df = load_data(uploaded_file)
        df = apply_strategy(df, short_window, long_window)
        # Evaluate & Plot
        df['portfolio'] = initial_capital
        st.subheader("Strategy Results on Uploaded Data")
        st.line_chart(df[['price', 'SMA_short', 'SMA_long']])
        total_return, max_drawdown, sharpe_ratio = evaluate_performance(df, initial_capital)
        st.metric("Total Return (%)", f"{total_return:.2f}")
        st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

