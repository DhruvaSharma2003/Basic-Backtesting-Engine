import streamlit as st
import pandas as pd
import numpy as np
import web3
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from web3 import Web3
import os

# --- Secrets ---
INFURA_URL = st.secrets["INFURA_URL"]
PRIVATE_KEY = st.secrets["PRIVATE_KEY"]

# --- FUNCTIONS ---

def load_data(file):
    data = pd.read_csv(file, parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def apply_strategy(data, short_window, long_window):
    data['SMA_short'] = data['price'].rolling(window=short_window).mean()
    data['SMA_long'] = data['price'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
    data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1
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
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return total_return, max_drawdown, sharpe_ratio

def plot_price_signals(data, start_date=None, end_date=None):
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['price'], label="Price", alpha=0.7)
    ax.plot(data.index, data['SMA_short'], label="SMA Short", linestyle="--")
    ax.plot(data.index, data['SMA_long'], label="SMA Long", linestyle="--")

    buy_signals = data[data['buy_signal'].notnull()]
    ax.scatter(buy_signals.index, buy_signals['buy_signal'], label="Buy Signal", color="green", marker="^", s=100)

    sell_signals = data[data['sell_signal'].notnull()]
    ax.scatter(sell_signals.index, sell_signals['sell_signal'], label="Sell Signal", color="red", marker="v", s=100)

    ax.set_title("Price and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_portfolio(data, start_date=None, end_date=None):
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['portfolio'], label="Portfolio Value", color="blue")
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

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

# --- STREAMLIT APP ---

st.title("📈 Crypto Backtesting Engine + Web3 Integration")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("**Loaded Data:**")
    st.write(data.head())

    st.sidebar.header("Strategy Parameters")
    short_window = st.sidebar.number_input("Short Moving Average Window", min_value=1, value=20)
    long_window = st.sidebar.number_input("Long Moving Average Window", min_value=1, value=50)
    initial_capital = st.sidebar.number_input("Initial Capital (USD)", min_value=1, value=10000)
    fee = st.sidebar.number_input("Transaction Fee (%)", min_value=0.0, value=0.1) / 100
    max_trades = st.sidebar.number_input("Max Trades Open at a Time", min_value=1, value=1)
    batch_size = st.sidebar.slider("Batch Size per Trade (%)", min_value=1, max_value=100, value=20) / 100

    start_date = st.sidebar.date_input("Start Date", value=data.index.min().date())
    end_date = st.sidebar.date_input("End Date", value=data.index.max().date())

    data = apply_strategy(data, short_window, long_window)
    data = backtest(data, initial_capital, fee, max_trades, batch_size)
    total_return, max_drawdown, sharpe_ratio = evaluate_performance(data, initial_capital)

    st.write("### 📊 Performance Metrics")
    st.write(f"**Total Return:** {total_return:.2f}%")
    st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

    st.write("### 📈 Price and Signals")
    plot_price_signals(data, start_date=start_date, end_date=end_date)

    st.write("### 💼 Portfolio Performance")
    plot_portfolio(data, start_date=start_date, end_date=end_date)

    # --- WEB3 INTEGRATION ---
    st.subheader("🔗 Web3 Trade Simulation (Testnet)")
    infura_url = st.text_input("Infura URL", type="password", value=DEFAULT_INFURA or "")
    private_key = st.text_input("Private Key", type="password", value=DEFAULT_PRIVATE_KEY or "")

    if infura_url and private_key:
        try:
            w3 = Web3(Web3.HTTPProvider(infura_url))
            account = w3.eth.account.from_key(private_key)
            st.success(f"✅ Connected to Sepolia as: {account.address}")

            if data['signal'].iloc[-1] == 1:
                if st.button("📤 BUY Signal Detected: Simulate Trade"):
                    tx_hash = simulate_trade(account, w3)
                    if tx_hash:
                        st.success(f"✅ TX sent: {tx_hash}")
                        st.markdown(f"[🔍 View on Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})")
            else:
                st.info("No active BUY signal at the latest point.")
        except Exception as e:
            st.error(f"Web3 Connection Error: {str(e)}")
