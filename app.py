import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# --- FUNCTIONS ---

def load_data(file):
    """Load the uploaded CSV file."""
    data = pd.read_csv(file, parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def apply_strategy(data, short_window, long_window):
    """Add SMA and generate buy/sell signals."""
    data['SMA_short'] = data['price'].rolling(window=short_window).mean()
    data['SMA_long'] = data['price'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1  # Buy
    data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1  # Sell
    return data

def backtest(data, initial_capital, fee):
    """Backtesting logic."""
    capital = initial_capital
    position = 0
    portfolio = []

    for index, row in data.iterrows():
        if row['signal'] == 1 and position == 0:  # Buy
            position = capital / row['price']
            capital = 0
            position *= (1 - fee)

        elif row['signal'] == -1 and position > 0:  # Sell
            capital = position * row['price']
            capital *= (1 - fee)
            position = 0

        portfolio_value = capital + position * row['price']
        portfolio.append(portfolio_value)

    data['portfolio'] = portfolio
    return data

def evaluate_performance(data, initial_capital):
    """Evaluate performance metrics."""
    final_value = data['portfolio'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    max_drawdown = (data['portfolio'] / data['portfolio'].cummax() - 1).min() * 100
    daily_returns = data['portfolio'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return total_return, max_drawdown, sharpe_ratio

# --- STREAMLIT APP ---

st.title("Crypto Backtesting Engine")

# Upload data
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = load_data(uploaded_file)
    st.write("**Loaded Data:**")
    st.write(data.head())

    # Strategy parameters
    st.sidebar.header("Strategy Parameters")
    short_window = st.sidebar.number_input("Short Moving Average Window", min_value=1, value=20)
    long_window = st.sidebar.number_input("Long Moving Average Window", min_value=1, value=50)
    initial_capital = st.sidebar.number_input("Initial Capital (USD)", min_value=1, value=10000)
    fee = st.sidebar.number_input("Transaction Fee (%)", min_value=0.0, value=0.1) / 100

    # Apply strategy
    data = apply_strategy(data, short_window, long_window)

    # Backtest strategy
    data = backtest(data, initial_capital, fee)

    # Evaluate performance
    total_return, max_drawdown, sharpe_ratio = evaluate_performance(data, initial_capital)

    # Display metrics
    st.write("### Performance Metrics")
    st.write(f"**Total Return:** {total_return:.2f}%")
    st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

    # Plot price and signals
    st.write("### Price and Signals")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['price'], label="Price", alpha=0.5)
    ax.plot(data.index, data['SMA_short'], label=f"SMA {short_window}")
    ax.plot(data.index, data['SMA_long'], label=f"SMA {long_window}")
    ax.scatter(data.index[data['signal'] == 1], data['price'][data['signal'] == 1], label="Buy Signal", color="green", marker="^")
    ax.scatter(data.index[data['signal'] == -1], data['price'][data['signal'] == -1], label="Sell Signal", color="red", marker="v")
    ax.set_title("Price and Signals")
    ax.legend()
    st.pyplot(fig)

    # Plot portfolio performance
    st.write("### Portfolio Performance")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['portfolio'], label="Portfolio Value", color="blue")
    ax.set_title("Portfolio Value Over Time")
    ax.legend()
    st.pyplot(fig)
