import streamlit as st
import feedparser
import urllib.parse
import re
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from streamlit_autorefresh import st_autorefresh

# প্রতি ৫ মিনিট (৩০০,০০০ মিলি-সেকেন্ড) পর পর পেজ নিজে নিজেই ডেটা রিফ্রেশ করবে
st_autorefresh(interval=300 * 1000, key="nifty_desk_autorefresh")
st.set_page_config(page_title="Nifty 50 Pro Desk", layout="wide", page_icon="⚡")

ist_tz = ZoneInfo("Asia/Kolkata")
now_ist = datetime.now(ist_tz)

st.title("⚡ NIFTY 50 REAL-TIME TRADING DESK")
st.caption(f"Last Updated: {now_ist.strftime('%d-%b-%Y | %I:%M %p IST')}")

# AI Model Caching
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

# 1. Global Markets & VIX
st.subheader("📊 1. Global Markets & Macro Metrics")
tickers = {
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "Crude Oil": "CL=F",
    "India VIX": "^INDIAVIX"
}

cols = st.columns(4)
col_idx = 0

for name, sym in tickers.items():
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="2d")
        if len(hist) >= 2:
            prev = hist['Close'].iloc[-2]
            curr = hist['Close'].iloc[-1]
            chg = ((curr - prev) / prev) * 100
            with cols[col_idx]:
                st.metric(label=name, value=f"{curr:.2f}", delta=f"{chg:+.2f}%")
            col_idx += 1
    except Exception:
        pass

# 2. News & Sentiment
st.subheader("📰 2. Nifty Triggers & News")
nifty_queries = [
    '("GIFT Nifty" OR "SGX Nifty") when:1d',
    '("pre-market" OR "morning cues" OR "opening bell") (Nifty OR Sensex) when:1d',
    '("FII" AND "DII" AND "inflow" OR "outflow") when:1d'
]
trade_triggers = ["gift nifty", "sgx", "fii", "dii", "pre-market", "gap up", "gap down", "cues", "crude"]

curated_items = []
predicted_points = []

for q in nifty_queries:
    encoded = urllib.parse.quote(q)
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en")
    for entry in feed.entries[:8]:
        title = entry.title
        if any(trig in title.lower() for trig in trade_triggers):
            pub_utc = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
            pub_ist = pub_utc.astimezone(ist_tz)
            
            score = get_sentiment(title)
            
            match = re.search(r'(\d+)\s*(?:pts|points)', title, re.IGNORECASE)
            if match and "gift nifty" in title.lower():
                pts = int(match.group(1))
                if score < 0:
                    pts = -pts
                predicted_points.append(pts)

            curated_items.append({
                "datetime_obj": pub_ist,
                "Time (IST)": pub_ist.strftime("%d-%b %I:%M %p"),
                "Headline": title,
                "FinBERT Score": round(score, 3),
                "Action Bias": "🟢 BUY" if score > 0.25 else ("🔴 SELL" if score < -0.25 else "⚪ NEUTRAL")
            })

df = pd.DataFrame(curated_items)
if not df.empty:
    df_sorted = df.drop_duplicates(subset=['Headline']).sort_values(by="datetime_obj", ascending=True)
    st.dataframe(df_sorted[['Time (IST)', 'Headline', 'FinBERT Score', 'Action Bias']], use_container_width=True)
else:
    st.info("No fresh trigger news found.")

# 3. Opening Prediction
st.subheader("🎯 3. Nifty Opening Outlook")
if predicted_points:
    avg_gap = sum(predicted_points) / len(predicted_points)
    if avg_gap >= 70:
        st.success(f"🚀 **STRONG GAP-UP EXPECTED:** Approx **{avg_gap:+.0f} points** | Plan: Buy on Dips")
    elif avg_gap <= -70:
        st.error(f"🩸 **STRONG GAP-DOWN EXPECTED:** Approx **{avg_gap:+.0f} points** | Plan: Sell on Rise")
    else:
        st.info(f"⚪ **FLAT / MILD GAP:** Approx **{avg_gap:+.0f} points**")
else:
    st.write("No specific point headline found.")
