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

# Streamlit page config must be the first Streamlit command
st.set_page_config(page_title="Nifty 50 Pro Desk", layout="wide", page_icon="⚡")

# প্রতি ৫ মিনিট (৩০০,০০০ মিলি-সেকেন্ড) পর পর পেজ নিজে নিজেই রিফ্রেশ হবে
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
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_finbert()
    FINBERT_READY = True
except Exception as e:
    tokenizer, model = None, None
    FINBERT_READY = False
    st.warning(f"FinBERT model load হয়নি। News sentiment আপাতত neutral থাকবে। Details: {e}")

def get_sentiment(text):
    if not FINBERT_READY:
        return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        return float(probs[1] - probs[2])
    except Exception:
        return 0.0

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
