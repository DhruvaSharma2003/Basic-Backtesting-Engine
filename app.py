import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from web3 import Web3
import os
import pytz
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error
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

    price_col = 'close' if 'close' in df.columns else 'price'

    if strategy == "Simple Moving Average (SMA)":
        short_window = kwargs.get("short_window", 5)
        long_window = kwargs.get("long_window", 15)
        df['SMA_short'] = df[price_col].rolling(window=short_window).mean()
        df['SMA_long'] = df[price_col].rolling(window=long_window).mean()
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
        df.loc[df['SMA_short'] <= df['SMA_long'], 'signal'] = -1

    elif strategy == "Bollinger Bands":
        period = kwargs.get("period", 20)
        std_dev = kwargs.get("std_dev", 2.0)
        df['MA'] = df[price_col].rolling(window=period).mean()
        df['BB_up'] = df['MA'] + std_dev * df[price_col].rolling(window=period).std()
        df['BB_down'] = df['MA'] - std_dev * df[price_col].rolling(window=period).std()
        df.loc[df[price_col] < df['BB_down'], 'signal'] = 1
        df.loc[df[price_col] > df['BB_up'], 'signal'] = -1

    df['buy_signal'] = np.where((df['signal'] == 1) & (df['signal'].shift(1) != 1), df[price_col], np.nan)
    df['sell_signal'] = np.where((df['signal'] == -1) & (df['signal'].shift(1) != -1), df[price_col], np.nan)

    return df

def forecast_prices(df, days_to_display):
    y = df['close'] if 'close' in df.columns else df['price']
    y.index = pd.to_datetime(y.index)

    if len(y) < 20:
        st.warning("Not enough data points to forecast.")
        return None, None

    model_outputs = evaluate_models(y, days_to_display)
    if not model_outputs:
        st.error("Forecasting failed for all models.")
        return y, None

    sorted_models = sorted(model_outputs.items(), key=lambda x: x[1][1])  # sort by RMSE%

    st.sidebar.markdown("---")
    st.sidebar.subheader("📉 Forecast Model Comparison")
    for name, (forecast, rmse) in sorted_models:
        st.sidebar.write(f"{name}: {rmse:.2f}% error")

    best_model = sorted_models[0][0]
    selected_model = st.sidebar.selectbox("Select Forecasting Model", [m[0] for m in sorted_models], index=0)

    selected_forecast = model_outputs[selected_model][0]
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days_to_display)
    forecast_series = pd.Series(selected_forecast.values, index=forecast_index)

    st.subheader(f"{selected_model} Forecast")
    st.dataframe(forecast_series.rename("Forecast Price"))

    return y, forecast_series

def simulate_trade(account, w3):
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        tx = {
            'nonce': nonce,
            'to': account.address,  # or change to '0x000000000000000000000000000000000000dead'
            'value': 0,
            'gas': 21000,
            'gasPrice': w3.to_wei('20', 'gwei'),
            'chainId': 11155111
        }
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = w3.to_hex(tx_hash)

        # ✅ Log the transaction in session state
        st.session_state.trade_log.append({
            "Tx Hash": tx_hash_hex,
            "Status": "Success",
            "Time": datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S IST"),
            "ETH Sent": 0.0,
            "To": tx['to']
        })

        return tx_hash_hex

    except Exception as e:
        st.session_state.trade_log.append({
            "Tx Hash": "N/A",
            "Status": f"Failed ({str(e)[:20]}...)",
            "Time":  datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S IST"),
            "ETH Sent": 0.0,
            "To": "N/A"
        })
        st.error(f"Transaction Failed: {e}")
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

    if 'backtest_log' not in st.session_state:
        st.session_state.backtest_log = []

    for index, row in data.iterrows():
        action_taken = None
        price = row['price']

        if row['signal'] == 1 and len(positions) < max_trades:
            trade_amount = capital * batch_size
            if trade_amount > 0:
                position = {
                    'amount': trade_amount / price,
                    'entry_price': price
                }
                positions.append(position)
                capital -= trade_amount
                capital *= (1 - fee)
                buy_signals.append((index, price))
                action_taken = "BUY"
            else:
                buy_signals.append((index, None))
        else:
            buy_signals.append((index, None))

        if row['signal'] == -1 and positions:
            sell_price = price
            for position in positions:
                capital += position['amount'] * sell_price
                capital *= (1 - fee)
            positions = []
            sell_signals.append((index, sell_price))
            action_taken = "SELL"
        else:
            sell_signals.append((index, None))

        portfolio_value = capital + sum(p['amount'] * price for p in positions)
        portfolio.append(portfolio_value)

        # ✅ Store to log if any action taken
        if action_taken:
            last_value = st.session_state.backtest_log[-1]["Portfolio Value"] if st.session_state.backtest_log else initial_capital
            pnl = round(portfolio_value - last_value, 2)

            st.session_state.backtest_log.append({
                "Time": index,
                "Action": action_taken,
                "Price": price,
                "Portfolio Value": round(portfolio_value, 2),
                "P&L": pnl
            })

    data['portfolio'] = portfolio
    data['buy_signal'] = [x[1] for x in buy_signals]
    data['sell_signal'] = [x[1] for x in sell_signals]
    return data

def evaluate_models(y, forecast_days):
    results = {}
    test_size = 5
    train, test = y[:-test_size], y[-test_size:]

    def compute_rmse(actual, predicted):
        actual = actual[:len(predicted)]
        predicted = predicted[:len(actual)]
        return np.sqrt(mean_squared_error(actual, predicted)) / np.mean(actual) * 100

    # AR Model
    try:
        model_ar = AutoReg(train, lags=5).fit()
        forecast = model_ar.forecast(steps=forecast_days)
        rmse = compute_rmse(test, forecast)
        results["AR"] = (forecast, rmse)
    except:
        pass

    # IMA Model
    try:
        model_ima = ARIMA(train, order=(0, 1, 1)).fit()
        forecast = model_ima.forecast(steps=forecast_days)
        rmse = compute_rmse(test, forecast)
        results["IMA"] = (forecast, rmse)
    except:
        pass

    # ARIMA Model
    try:
        model_arima = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model_arima.forecast(steps=forecast_days)
        rmse = compute_rmse(test, forecast)
        results["ARIMA"] = (forecast, rmse)
    except:
        pass

    # SARIMA Model
    try:
        model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7)).fit(disp=False)
        forecast = model_sarima.forecast(steps=forecast_days)
        rmse = compute_rmse(test, forecast)
        results["SARIMA"] = (forecast, rmse)
    except:
        pass

    # Exponential Smoothing
    try:
        model_ets = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
        forecast = model_ets.forecast(steps=forecast_days)
        rmse = compute_rmse(test, forecast)
        results["ETS"] = (forecast, rmse)
    except:
        pass

    # Prophet
    try:
        df_prophet = train.reset_index()
        df_prophet.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)[['ds', 'yhat']].set_index('ds').iloc[-forecast_days:]['yhat']
        rmse = compute_rmse(test, forecast)
        results["Prophet"] = (forecast, rmse)
    except:
        pass

    return results

def evaluate_performance(df, initial_capital):
    portfolio = df['portfolio']
    returns = portfolio.pct_change().dropna()

    total_return = (portfolio.iloc[-1] - initial_capital) / initial_capital * 100
    max_drawdown = ((portfolio.cummax() - portfolio) / portfolio.cummax()).max() * 100
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    return total_return, max_drawdown, sharpe_ratio

# --- Streamlit App ---
st.title("📊 Crypto Backtesting Engine + Web3 Execution")

if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

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

if mode == "Historical Data Upload":
    with st.sidebar:
        st.markdown("<div style='border:2px solid #1f77b4; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
        initial_capital = st.number_input("Initial Capital", min_value=1, value=10000)
        fee = st.number_input("Transaction Fee (%)", min_value=0.0, value=0.1) / 100
        max_trades = st.number_input("Max Trades", min_value=1, value=1)
        batch_size = st.slider("Batch Size (%)", min_value=1, max_value=100, value=20) / 100
        st.markdown("</div>", unsafe_allow_html=True)

    # Clear backtest log when sidebar inputs change
    if "prev_params" not in st.session_state:
        st.session_state.prev_params = {
            "capital": initial_capital,
            "fee": fee,
            "trades": max_trades,
            "batch": batch_size
        }

    params_changed = (
        st.session_state.prev_params["capital"] != initial_capital or
        st.session_state.prev_params["fee"] != fee or
        st.session_state.prev_params["trades"] != max_trades or
        st.session_state.prev_params["batch"] != batch_size
    )

    if params_changed:
        st.session_state.backtest_log = []
        st.session_state.prev_params = {
            "capital": initial_capital,
            "fee": fee,
            "trades": max_trades,
            "batch": batch_size
        }

if mode == "Live Data":
    coin_name = st.selectbox("Choose Crypto", list(CRYPTO_OPTIONS.keys()))
    coin_id = CRYPTO_OPTIONS[coin_name]

    try:
        df = fetch_ohlc_data(coin_id)
        df = apply_strategy(df, strategy_choice, **strategy_params)
        actual_series, forecast_series = forecast_prices(df, forecast_days)

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

        # BUY Button
        if df['signal'].iloc[-1] == 1:
            if st.button("BUY Signal Detected: Execute Trade"):
                w3 = Web3(Web3.HTTPProvider(DEFAULT_INFURA))
                account = w3.eth.account.from_key(DEFAULT_PRIVATE_KEY)
                tx_hash = simulate_trade(account, w3)
                if tx_hash:
                    st.success(f"TX Hash: {tx_hash}")
                    st.markdown(f"[View on Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})")
        else:
            st.info("No Buy Signal at the latest point.")

        # Trade Execution Log Panel
        if st.session_state.trade_log:
            st.subheader("🧾 Trade Execution Log")
            df_log = pd.DataFrame(st.session_state.trade_log)
            df_log_display = df_log[::-1]  # latest first
            st.dataframe(df_log_display, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧹 Clear Log"):
                    st.session_state.trade_log = []
                    st.rerun()

            with col2:
                csv_data = df_log_display.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Log", data=csv_data, file_name="trade_log.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")


elif mode == "Historical Data Upload":
    uploaded_file = st.file_uploader("Upload Historical CSV (timestamp, price)", type="csv")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            df = apply_strategy(df, strategy_choice, **strategy_params)
            df = backtest(df, initial_capital, fee, max_trades, batch_size)
            actual_series, forecast_series = forecast_prices(df, forecast_days)

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

            # Optional: Show Backtest Trade Log
            if 'backtest_log' in st.session_state and st.session_state.backtest_log:
                st.subheader("📜 Backtest Trade Log")
                df_backtest_log = pd.DataFrame(st.session_state.backtest_log)
                df_backtest_log_display = df_backtest_log[::-1]
                
                def highlight_pnl(row):
                    color = "green" if row["P&L"] > 0 else ("red" if row["P&L"] < 0 else "gray")
                    return ['background-color: {}'.format(color) if col == "P&L" else '' for col in row.index]

                st.dataframe(df_backtest_log_display.style.apply(highlight_pnl, axis=1), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🧹 Clear Backtest Log"):
                        st.session_state.backtest_log = []
                        st.rerun()

                with col2:
                    csv_bt = df_backtest_log_display.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download Backtest Log", data=csv_bt, file_name="backtest_log.csv", mime="text/csv")

        except Exception as e:
            st.error(f"File Error: {str(e)}")
