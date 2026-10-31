import streamlit as st
import feedparser
import urllib.parse
import re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from streamlit_autorefresh import st_autorefresh

# প্রতি ৫ মিনিট (৩০০,০০০ মিলি-সেকেন্ড) পর পর পেজ নিজে নিজেই রিফ্রেশ হবে
st_autorefresh(interval=300 * 1000, key="nifty_desk_autorefresh")
st.set_page_config(page_title="Nifty 50 Pro Desk", layout="wide", page_icon="⚡")

ist_tz = ZoneInfo("Asia/Kolkata")
now_ist = datetime.now(ist_tz)

st.title("⚡ NIFTY 50 REAL-TIME TRADING DESK")
st.caption(f"Last Updated: {now_ist.strftime('%d-%b-%Y | %I:%M %p IST')}")

# ==========================================
# 0. AI FinBERT Model Loader
# ==========================================
@st.cache_resource(show_spinner="AI FinBERT model loading...")
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_finbert()

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    return probs[1] - probs[2]

def get_scalar(val):
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return float(val.values.flatten()[0])
    return float(val)

# ==========================================
# 1. Macro Bias & Global Metrics (Score: -1 to +1)
# ==========================================
st.subheader("🌐 1. Global Markets & Macro Screening")

@st.cache_data(ttl=300)
def get_macro_data():
    tickers = ["DX-Y.NYB", "BZ=F", "^DJI", "^IXIC", "^INDIAVIX"]
    data = yf.download(tickers, period="5d", interval="1d", auto_adjust=False, progress=False)
    close_data = data['Close']

    dxy_close = close_data['DX-Y.NYB'].dropna()
    crude_close = close_data['BZ=F'].dropna()
    dji_close = close_data['^DJI'].dropna()
    nasdaq_close = close_data['^IXIC'].dropna()
    vix_close = close_data['^INDIAVIX'].dropna()

    dxy_pct = ((get_scalar(dxy_close.iloc[-1]) - get_scalar(dxy_close.iloc[-2])) / get_scalar(dxy_close.iloc[-2])) * 100
    crude_pct = ((get_scalar(crude_close.iloc[-1]) - get_scalar(crude_close.iloc[-2])) / get_scalar(crude_close.iloc[-2])) * 100
    dji_pct = ((get_scalar(dji_close.iloc[-1]) - get_scalar(dji_close.iloc[-2])) / get_scalar(dji_close.iloc[-2])) * 100
    vix_val = get_scalar(vix_close.iloc[-1])

    macro_score = 0
    if dxy_pct < -0.15 and crude_pct < -0.75:
        macro_score = 1
    elif dxy_pct > 0.15 and crude_pct > 0.75:
        macro_score = -1

    return {
        "score": macro_score,
        "dxy_val": get_scalar(dxy_close.iloc[-1]),
        "dxy_pct": dxy_pct,
        "crude_val": get_scalar(crude_close.iloc[-1]),
        "crude_pct": crude_pct,
        "dji_val": get_scalar(dji_close.iloc[-1]),
        "dji_pct": dji_pct,
        "vix": vix_val
    }

macro = get_macro_data()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dollar Index (DXY)", f"{macro['dxy_val']:.2f}", f"{macro['dxy_pct']:+.2f}%")
c2.metric("Brent Crude", f"${macro['crude_val']:.2f}", f"{macro['crude_pct']:+.2f}%")
c3.metric("Dow Jones", f"{macro['dji_val']:.2f}", f"{macro['dji_pct']:+.2f}%")
c4.metric("India VIX", f"{macro['vix']:.2f}")

# ==========================================
# 2. Key Movers: Indices & Heavyweights Live Prices
# ==========================================
st.subheader("🏛️ 2. Live Indices & Heavyweights Alignment")

@st.cache_data(ttl=60)
def get_live_market_tickers():
    symbols = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS"
    }

    results = {}
    data = yf.download(list(symbols.values()), period="1d", interval="5m", auto_adjust=False, progress=False)
    close_data = data['Close']

    for name, sym in symbols.items():
        try:
            s_close = close_data[sym].dropna()
            if len(s_close) >= 2:
                curr_p = get_scalar(s_close.iloc[-1])
                first_p = get_scalar(s_close.iloc[0])
                chg_pct = ((curr_p - first_p) / first_p) * 100
                results[name] = {"price": curr_p, "change": chg_pct}
            else:
                results[name] = {"price": 0.0, "change": 0.0}
        except Exception:
            results[name] = {"price": 0.0, "change": 0.0}

    # Heavyweight alignment score
    rel_chg = results["Reliance Industries"]["change"]
    hdfc_chg = results["HDFC Bank"]["change"]

    hw_score = 0
    if rel_chg > 0.3 and hdfc_chg > 0.3:
        hw_score = 1
    elif rel_chg < -0.3 and hdfc_chg < -0.3:
        hw_score = -1

    return results, hw_score

live_data, hw_score = get_live_market_tickers()

# ৪টি কলামে পাশাপাশি ডিসপ্লে
col_nifty, col_bank, col_rel, col_hdfc = st.columns(4)

col_nifty.metric(
    "NIFTY 50",
    f"₹{live_data['Nifty 50']['price']:,.2f}",
    f"{live_data['Nifty 50']['change']:+.2f}% (Today)"
)

col_bank.metric(
    "BANK NIFTY",
    f"₹{live_data['Bank Nifty']['price']:,.2f}",
    f"{live_data['Bank Nifty']['change']:+.2f}% (Today)"
)

col_rel.metric(
    "Reliance Industries",
    f"₹{live_data['Reliance Industries']['price']:,.2f}",
    f"{live_data['Reliance Industries']['change']:+.2f}% (Intraday)"
)

col_hdfc.metric(
    "HDFC Bank",
    f"₹{live_data['HDFC Bank']['price']:,.2f}",
    f"{live_data['HDFC Bank']['change']:+.2f}% (Intraday)"
)

# Alignment Status
if hw_score == 1:
    st.success("🟢 **Heavyweights Status:** Reliance ও HDFC Bank উভয়ই ঊর্ধ্বমুখী (Bullish Alignment)")
elif hw_score == -1:
    st.error("🔴 **Heavyweights Status:** Reliance ও HDFC Bank উভয়ই নিম্নমুখী (Bearish Pressure)")
else:
    st.info("⚪ **Heavyweights Status:** Mixed / Sideways Movement")

# ==========================================
# 3. Nifty Technicals (Tech Score: -1 to +1)
# ==========================================
@st.cache_data(ttl=120)
def get_nifty_technicals():
    nifty = yf.download("^NSEI", period="5d", interval="15m", auto_adjust=False, progress=False)
    daily_nifty = yf.download("^NSEI", period="5d", interval="1d", auto_adjust=False, progress=False)

    pdh = get_scalar(daily_nifty['High'].iloc[-2])
    pdl = get_scalar(daily_nifty['Low'].iloc[-2])

    close = nifty['Close'].squeeze()
    high = nifty['High'].squeeze()
    low = nifty['Low'].squeeze()

    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = get_scalar(tr.rolling(14).mean().iloc[-1])

    c_price = get_scalar(close.iloc[-1])
    c_ema20 = get_scalar(ema_20.iloc[-1])
    c_ema50 = get_scalar(ema_50.iloc[-1])

    tech_score = 0
    if c_price > pdh and c_price > c_ema20 > c_ema50:
        tech_score = 1
    elif c_price < pdl and c_price < c_ema20 < c_ema50:
        tech_score = -1

    return {
        "price": round(c_price, 2),
        "pdh": round(pdh, 2),
        "pdl": round(pdl, 2),
        "atr": round(atr, 2),
        "score": tech_score
    }

tech = get_nifty_technicals()

# ==========================================
# 4. Total Confluence Engine (+2 / -2 Logic)
# ==========================================
st.subheader("🎯 3. Multi-Factor Confluence Signal (+2 / -2)")
