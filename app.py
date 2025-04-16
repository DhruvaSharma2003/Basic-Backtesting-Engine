import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from web3 import Web3
import json
import datetime
import requests
import time

# Load secrets
INFURA_URL = st.secrets["INFURA_URL"]
PRIVATE_KEY = st.secrets["PRIVATE_KEY"]

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

# Function to get live price from Binance REST API
def get_live_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    return float(response.json()['price'])

# Apply moving average strategy
def apply_strategy(data, short_window, long_window):
    data['SMA_short'] = data['price'].rolling(window=short_window).mean()
    data['SMA_long'] = data['price'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
    data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1
    return data

# Simulate trade on-chain
def simulate_trade(account, w3):
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        tx = {
            'nonce': nonce,
            'to': account.address,
            'value': 0,
            'gas': 21000,
            'gasPrice': w3.to_wei('10', 'gwei'),
            'chainId': 11155111  # Sepolia
        }
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return w3.to_hex(tx_hash)
    except Exception as e:
        st.error(f"Transaction Failed: {str(e)}")
        return None

# Streamlit UI
st.title("📡 Live Backtesting Engine + Web3 Trade Execution")

st.sidebar.header("Parameters")
symbol = st.sidebar.text_input("Crypto Symbol (e.g., BTCUSDT)", value="BTCUSDT")
short_sma = st.sidebar.number_input("Short SMA", value=5)
long_sma = st.sidebar.number_input("Long SMA", value=15)
max_rows = st.sidebar.slider("History Size", 50, 500, 100)
trade_button = st.sidebar.button("Execute Trade if Buy Signal")

# Session state
if "live_prices" not in st.session_state:
    st.session_state.live_prices = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# Fetch live price and update state
price = get_live_price(symbol)
st.session_state.live_prices.append(price)
st.session_state.timestamps.append(datetime.datetime.now())

# Keep only last N points
if len(st.session_state.live_prices) > max_rows:
    st.session_state.live_prices = st.session_state.live_prices[-max_rows:]
    st.session_state.timestamps = st.session_state.timestamps[-max_rows:]

# Create dataframe and apply strategy
df = pd.DataFrame({
    "timestamp": st.session_state.timestamps,
    "price": st.session_state.live_prices
})
df.set_index("timestamp", inplace=True)
df = apply_strategy(df, short_sma, long_sma)

# Display charts
st.write("### 📈 Live Price Chart")
st.line_chart(df["price"])

st.write("### 🧠 Strategy Signals")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["price"], label="Price", alpha=0.7)
ax.plot(df.index, df["SMA_short"], label="SMA Short", linestyle="--")
ax.plot(df.index, df["SMA_long"], label="SMA Long", linestyle="--")

buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]
ax.scatter(buy_signals.index, buy_signals["price"], marker="^", color="green", label="Buy", s=100)
ax.scatter(sell_signals.index, sell_signals["price"], marker="v", color="red", label="Sell", s=100)
ax.legend()
st.pyplot(fig)

# Simulate trade if button is pressed and buy signal detected
if trade_button and df["signal"].iloc[-1] == 1:
    tx_hash = simulate_trade(account, w3)
    if tx_hash:
        st.success(f"✅ Trade executed! TX Hash: {tx_hash}")
        st.markdown(f"[🔗 View on Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})")
else:
    st.info("ℹ️ Trade not executed. No buy signal or button not clicked.")
