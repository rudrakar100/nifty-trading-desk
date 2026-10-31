import streamlit as st
import feedparser
import urllib.parse
import re
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from streamlit_autorefresh import st_autorefresh

# প্রতি ৫ মিনিট (৩০০,০০০ মিলি-সেকেন্ড) পর পর পেজ নিজে নিজেই রিফ্রেশ হবে
st.set_page_config(page_title="Nifty 50 Pro Desk", layout="wide", page_icon="⚡")
st_autorefresh(interval=300 * 1000, key="nifty_desk_autorefresh")

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
# PRE-MARKET LEVEL ENGINE (EOD -> NEXT SESSION)
# ==========================================
import requests

st.subheader("🧭 6. Next-Day Important Levels (EOD Preparation)")
st.caption("এই অংশটি সম্পূর্ণ হওয়া daily/15m/5m candles এবং সর্বশেষ উপলব্ধ NIFTY option-chain snapshot থেকে সম্ভাব্য zones তৈরি করে।")

@st.cache_data(ttl=900, show_spinner=False)
def get_nifty_level_data():
    daily = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=False, progress=False)
    h1 = yf.download("^NSEI", period="60d", interval="1h", auto_adjust=False, progress=False)
    m15 = yf.download("^NSEI", period="60d", interval="15m", auto_adjust=False, progress=False)
    m5 = yf.download("^NSEI", period="5d", interval="5m", auto_adjust=False, progress=False)
    return daily, h1, m15, m5

def clean_ohlc(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    out = df[[c for c in needed if c in df.columns]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna()

def detect_fvg(df, limit=8):
    """Three-candle imbalance zones. Uses completed candles supplied to this function."""
    d = clean_ohlc(df)
    zones = []
    if len(d) < 3:
        return zones
    for i in range(2, len(d)):
        a, c = d.iloc[i-2], d.iloc[i]
        if float(a["High"]) < float(c["Low"]):
            zones.append({"Type": "Bullish FVG", "Low": float(a["High"]),
                          "High": float(c["Low"]), "Time": str(d.index[i])})
        elif float(a["Low"]) > float(c["High"]):
            zones.append({"Type": "Bearish FVG", "Low": float(c["High"]),
                          "High": float(a["Low"]), "Time": str(d.index[i])})
    return zones[-limit:][::-1]

def detect_order_blocks(df, limit=6):
    """
    Heuristic OB: last opposite candle before a >=1.2 ATR displacement
    that closes beyond the previous candle's local 5-bar extreme.
    This is a rule-based approximation, not institutional order-flow data.
    """
