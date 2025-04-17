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
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        raise Exception("Failed to fetch OHLC data")

def apply_strategy(df, short_window, long_window, price_col='close'):
    df['SMA_short'] = df[price_col].rolling(window=short_window).mean()
    df['SMA_long'] = df[price_col].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df.loc[df['SMA_short'] <= df['SMA_long'], 'signal'] = -1
    df['buy_signal'] = np.where((df['signal'] == 1) & (df['signal'].shift(1) != 1), df[price_col], np.nan)
    df['sell_signal'] = np.where((df['signal'] == -1) & (df['signal'].shift(1) != -1), df[price_col], np.nan)
    return df

def forecast_prices(df, days, model_type='ARIMA'):
    y = df['close'] if 'close' in df.columns else df['price']
    y.index = pd.to_datetime(y.index)

    if len(y) < 20:
        st.warning("Not enough data points to forecast.")
        return None, None

    model = None
    if model_type == 'AR':
        model = AutoReg(y, lags=5).fit()
    elif model_type == 'IMA':
        model = ARIMA(y, order=(0, 1, 1)).fit()
    else:  # ARIMA
        model = ARIMA(y, order=(5, 1, 0)).fit()

    forecast = model.forecast(steps=days)
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast_series = pd.Series(forecast, index=forecast_index)

    return y, forecast_series

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
    except AttributeError:
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        return w3.to_hex(tx_hash)
    except Exception as e:
            st.error(f"File Error: {str(e)}")
