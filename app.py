import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from web3 import Web3

# --- SECRETS ---
DEFAULT_INFURA = st.secrets["INFURA_URL"]
DEFAULT_PRIVATE_KEY = st.secrets["PRIVATE_KEY"]

# --- LIVE PRICE FETCHING ---
def get_live_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    return float(response.json()['price'])

# --- STRATEGY ---
def apply_strategy(data, short_window, long_window):
    data['SMA_short'] = data['price'].rolling(window=short_window).mean()
    data['SMA_long'] = data['price'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
    data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1
    return data

# --- BACKTESTING ---
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
                position = {'amount': trade_amount / row['price'], 'entry_price': row['price']}
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

# --- PERFORMANCE ---
def evaluate_performance(data, initial_capital):
    final_value = data['portfolio'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    max_drawdown = (data['portfolio'] / data['portfolio'].cummax() - 1).min() * 100
    daily_returns = data['portfolio'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return total_return, max_drawdown, sharpe_ratio

# --- PLOTTING ---
def plot_price_signals(data):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['price'], label="Price", alpha=0.7)
    ax.plot(data.index, data['SMA_short'], label="SMA Short", linestyle="--")
    ax.plot(data.index, data['SMA_long'], label="SMA Long", linestyle="--")
    ax.scatter(data[data['buy_signal'].notnull()].index, data['buy_signal'], label="Buy", color="green", marker="^", s=100)
    ax.scatter(data[data['sell_signal'].notnull()].index, data['sell_signal'], label="Sell", color="red", marker="v", s=100)
    ax.set_title("Price and Signals")
    ax.legend()
    st.pyplot(fig)

# --- WEB3 SIMULATION ---
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

# --- STREAMLIT APP ---
st.title("📈 Live Backtesting Engine + Web3 Trade Execution")

st.sidebar.header("Parameters")
short_window = st.sidebar.number_input("Short SMA", 1, 50, 5)
long_window = st.sidebar.number_input("Long SMA", 1, 100, 15)
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1, value=10000)
fee = st.sidebar.number_input("Transaction Fee (%)", 0.0, 5.0, 0.1) / 100
max_trades = st.sidebar.number_input("Max Trades", 1, 10, 1)
batch_size = st.sidebar.slider("Batch Size (%)", 1, 100, 20) / 100

# --- LIVE DATA BUFFER ---
if 'live_prices' not in st.session_state:
    st.session_state.live_prices = []
    st.session_state.timestamps = []

price = get_live_price("BTCUSDT")
st.session_state.live_prices.append(price)
st.session_state.timestamps.append(datetime.now())

if len(st.session_state.live_prices) > 50:
    st.session_state.live_prices = st.session_state.live_prices[-50:]
    st.session_state.timestamps = st.session_state.timestamps[-50:]

df = pd.DataFrame({
    'timestamp': st.session_state.timestamps,
    'price': st.session_state.live_prices
})
df.set_index('timestamp', inplace=True)

# --- STRATEGY + BACKTEST + VISUALS ---
df = apply_strategy(df, short_window, long_window)
df = backtest(df, initial_capital, fee, max_trades, batch_size)
total_return, max_drawdown, sharpe_ratio = evaluate_performance(df, initial_capital)

st.metric("Total Return", f"{total_return:.2f}%")
st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

plot_price_signals(df)

# --- WEB3 ---
st.subheader("🔗 Web3 Simulation")
try:
    w3 = Web3(Web3.HTTPProvider(DEFAULT_INFURA))
    account = w3.eth.account.from_key(DEFAULT_PRIVATE_KEY)
    st.success(f"Connected: {account.address}")

    if df['signal'].iloc[-1] == 1:
        if st.button("🚀 Execute BUY Trade"):
            tx_hash = simulate_trade(account, w3)
            if tx_hash:
                st.success(f"TX Sent: {tx_hash}")
                st.markdown(f"[🔍 View TX](https://sepolia.etherscan.io/tx/{tx_hash})")
    else:
        st.info("No active BUY signal detected.")
except Exception as e:
    st.error(f"Web3 Connection Error: {str(e)}")
