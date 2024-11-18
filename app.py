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

def backtest(data, initial_capital, fee, max_trades, batch_size):
    """
    Backtesting logic with max trades limit and batch size.

    Parameters:
    - data (pd.DataFrame): Historical price data with signals.
    - initial_capital (float): Starting capital in USD.
    - fee (float): Transaction fee as a fraction (e.g., 0.001 for 0.1%).
    - max_trades (int): Maximum number of open trades at any time.
    - batch_size (float): Percentage of capital to use per trade (0 to 1).

    Returns:
    - pd.DataFrame: Data with portfolio value and trading points.
    """
    capital = initial_capital
    position = 0  # Current open trades
    portfolio = []  # Portfolio value tracker
    buy_signals = []  # Track buy points for visualization
    sell_signals = []  # Track sell points for visualization

    for index, row in data.iterrows():
        # If Buy Signal and within max trades limit
        if row['signal'] == 1 and position < max_trades:
            trade_amount = capital * batch_size  # Use only a portion of capital
            if trade_amount > 0:
                position += trade_amount / row['price']  # Add to position
                capital -= trade_amount  # Deduct capital used
                position *= (1 - fee)  # Apply transaction fee
                buy_signals.append((index, row['price']))  # Record buy signal
            else:
                buy_signals.append((index, None))  # No buy due to no capital
        else:
            buy_signals.append((index, None))  # No buy

        # If Sell Signal and position is open
        if row['signal'] == -1 and position > 0:
            capital += position * row['price']  # Sell all holdings
            capital *= (1 - fee)  # Apply transaction fee
            position = 0  # Reset position
            sell_signals.append((index, row['price']))  # Record sell signal
        else:
            sell_signals.append((index, None))  # No sell

        # Track portfolio value
        portfolio_value = capital + position * row['price']
        portfolio.append(portfolio_value)

    # Add buy/sell points and portfolio value to the data
    data['portfolio'] = portfolio
    data['buy_signal'] = [x[1] for x in buy_signals]  # Extract buy prices
    data['sell_signal'] = [x[1] for x in sell_signals]  # Extract sell prices
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
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['price'], label="Price", alpha=0.7, linewidth=1.5)
    ax.plot(data.index, data['SMA_short'], label=f"SMA Short", linestyle="--", alpha=0.7)
    ax.plot(data.index, data['SMA_long'], label=f"SMA Long", linestyle="--", alpha=0.7)

    # Plot buy signals
    buy_signals = data[data['buy_signal'].notnull()]
    ax.scatter(buy_signals.index, buy_signals['buy_signal'], label="Buy Signal", color="green", marker="^", s=100)

    # Plot sell signals
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
    """Plot portfolio value."""
    if start_date and end_date:
        data = data.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['portfolio'], label="Portfolio Value", color="blue", linewidth=1.5)
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- STREAMLIT APP ---

st.title("Crypto Backtesting Engine with Trade Constraints")

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

    st.write("### Performance Metrics")
    st.write(f"**Total Return:** {total_return:.2f}%")
    st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

    st.write("### Price and Signals")
    plot_price_signals(data, start_date=start_date, end_date=end_date)

    st.write("### Portfolio Performance")
    plot_portfolio(data, start_date=start_date, end_date=end_date)
