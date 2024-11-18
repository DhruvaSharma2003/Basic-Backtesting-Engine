import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

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

def plot_price_signals(data, start_date=None, end_date=None):
    """Plot price with buy/sell signals and SMAs."""
    # Filter data for the selected date range
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['price'], label="Price", alpha=0.7, linewidth=1)
    ax.plot(data.index, data['SMA_short'], label=f"SMA Short", linestyle='--', linewidth=1)
    ax.plot(data.index, data['SMA_long'], label=f"SMA Long", linestyle='--', linewidth=1)
    ax.scatter(data.index[data['signal'] == 1], data['price'][data['signal'] == 1], label="Buy Signal", color="green", marker="^", alpha=1)
    ax.scatter(data.index[data['signal'] == -1], data['price'][data['signal'] == -1], label="Sell Signal", color="red", marker="v", alpha=1)

    ax.set_title("Price and Signals", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    date_format = DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_portfolio(data, start_date=None, end_date=None):
    """Plot portfolio value."""
    # Filter data for the selected date range
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['portfolio'], label="Portfolio Value", color="blue", linewidth=1.5)
    ax.set_title("Portfolio Value Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    date_format = DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    st.pyplot(fig)

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

    # Date range selection
    st.sidebar.header("Visualization Date Range")
    start_date = st.sidebar.date_input("Start Date", value=data.index.min().date())
    end_date = st.sidebar.date_input("End Date", value=data.index.max().date())

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
    plot_price_signals(data, start_date=start_date, end_date=end_date)

    # Plot portfolio performance
    st.write("### Portfolio Performance")
    plot_portfolio(data, start_date=start_date, end_date=end_date)
