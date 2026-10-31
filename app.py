        "pdl": round(pdl, 2),
        "atr": round(atr, 2),
        "score": tech_score
    }

tech = get_nifty_technicals()

# ==========================================
# 4. Total Confluence Engine (+2 / -2 Logic)
# ==========================================
st.subheader("🎯 3. Multi-Factor Confluence Signal (+2 / -2)")

total_confluence = macro['score'] + hw_score + tech['score']

t1, t2, t3, t4 = st.columns(4)
t1.metric("Nifty Spot", f"₹{tech['price']}")
t2.metric("Previous Day High (PDH)", f"₹{tech['pdh']}")
t3.metric("Previous Day Low (PDL)", f"₹{tech['pdl']}")
t4.metric("Confluence Score", f"{total_confluence} / 3", help="Macro (+1/-1) + Heavyweight (+1/-1) + Technical (+1/-1)")

if total_confluence >= 2:
    sl = round(tech['price'] - (1.5 * tech['atr']), 2)
    risk_unit = tech['price'] - sl
    tgt = round(tech['price'] + (2.0 * risk_unit), 2)
    st.success(
        f"🟢 **HIGH CONFLUENCE LONG SETUP (SCORE: {total_confluence})**\n\n"
        f"• **Entry Price:** ₹{tech['price']}\n\n"
        f"• **Stop Loss:** ₹{sl} (~{round(1.5 * tech['atr'], 1)} pts)\n\n"
        f"• **Target (1:2):** ₹{tgt}\n\n"
        f"• **Reasons:** Breakout above PDH + Weak DXY/Crude + Reliance & HDFC Bullish"
    )
elif total_confluence <= -2:
    sl = round(tech['price'] + (1.5 * tech['atr']), 2)
    risk_unit = sl - tech['price']
    tgt = round(tech['price'] - (2.0 * risk_unit), 2)
    st.error(
        f"🔴 **HIGH CONFLUENCE SHORT SETUP (SCORE: {total_confluence})**\n\n"
        f"• **Short Entry:** ₹{tech['price']}\n\n"
        f"• **Stop Loss:** ₹{sl} (~{round(1.5 * tech['atr'], 1)} pts)\n\n"
        f"• **Target (1:2):** ₹{tgt}\n\n"
        f"• **Reasons:** Breakdown below PDL + Strong DXY/Crude + Reliance & HDFC Bearish"
    )
else:
    st.info(
        f"⚖️ **Market Neutral / Choppy Range (Score: {total_confluence})**\n\n"
        f"ম্যাক্রো, টেকনিক্যাল ও হেভিওয়েট একে অপরের সাথে পুরোপুরি একমুখী নয়। সাইডওয়েজ মার্কেটে ফেক সিগন্যাল এড়াতে অপেক্ষা করুন।"
    )

# ==========================================
# 5. News & Sentiment Analysis
# ==========================================
st.subheader("📰 4. Nifty Triggers & News")
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

# ==========================================
# 6. Opening Prediction
# ==========================================
st.subheader("🎯 5. Nifty Opening Outlook")
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


# ==========================================
# 4. AI Market Prediction & Research Engine
# ==========================================
st.subheader("🧠 4. Dedicated AI Market Research & Prediction Desk")

nifty_queries = [
    '("GIFT Nifty" OR "SGX Nifty") when:1d',
    '("pre-market" OR "morning cues" OR "opening bell") (Nifty OR Sensex) when:1d',
    '("FII" AND "DII" AND "inflow" OR "outflow") when:1d',
    '("Reliance" OR "HDFC Bank" OR "TCS" OR "Infosys" OR "ICICI Bank") ("Q1" OR "Q2" OR "Q3" OR "Q4" OR "results" OR "net profit" OR "revenue") when:1d',
    '("RBI" OR "US Fed" OR "inflation" OR "budget" OR "rate cut") (Nifty OR market) when:1d'
]

trade_triggers = [
    "gift nifty", "sgx", "fii", "dii", "pre-market", "gap up", "gap down", "cues", "crude",
    "reliance", "hdfc", "tcs", "infosys", "icici", "result", "results", "profit", "earnings",
    "rbi", "fed", "inflation"
]

curated_items = []
predicted_points = []
all_sentiment_scores = []
earnings_news = []

for q in nifty_queries:
    encoded = urllib.parse.quote(q)
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en")
    for entry in feed.entries[:8]:
        title = entry.title
        lower_title = title.lower()
        if any(trig in lower_title for trig in trade_triggers):
            pub_utc = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
            pub_ist = pub_utc.astimezone(ist_tz)
            
            score = get_sentiment(title)
            all_sentiment_scores.append(score)
            
            # ক্যাটাগরি নির্ধারণ
            if any(k in lower_title for k in ["result", "results", "profit", "earnings", "revenue"]):
                category = "📊 Earnings/Corporate"
                earnings_news.append((title, score))
            elif any(k in lower_title for k in ["fii", "dii", "inflow", "outflow"]):
                category = "💰 Institutional (FII/DII)"
            elif any(k in lower_title for k in ["gift nifty", "sgx", "pre-market", "cues"]):
                category = "🌍 Global & Open Cues"
            else:
                category = "🗞️ General Macro"

            # পয়েন্ট প্রেডিকশন
            match = re.search(r'(\d+)\s*(?:pts|points)', title, re.IGNORECASE)
            if match and "gift nifty" in lower_title:
                pts = int(match.group(1))
                if score < 0:
                    pts = -pts
                predicted_points.append(pts)

            curated_items.append({
                "datetime_obj": pub_ist,
                "Time (IST)": pub_ist.strftime("%d-%b %I:%M %p"),
                "Category": category,
                "Headline": title,
                "FinBERT Score": round(score, 3),
                "Action Bias": "🟢 BUY" if score > 0.20 else ("🔴 SELL" if score < -0.20 else "⚪ NEUTRAL")
            })

# --- AI সামগ্রিক গবেষণা ও দিনের মার্কেট ডিরেকশন কার্ড ---
avg_sentiment = np.mean(all_sentiment_scores) if all_sentiment_scores else 0.0
avg_gap = (sum(predicted_points) / len(predicted_points)) if predicted_points else 0.0

col_ai1, col_ai2 = st.columns([1.2, 2])

with col_ai1:
    st.markdown("##### 📌 **AI Market Verdict**")
    if avg_sentiment > 0.15:
        st.success(f"🟢 **BULLISH BIAS (Score: {avg_sentiment:+.2f})**\n\nমার্কেট সংবাদের সামগ্রিক ভিত্তি পজিটিভ। ডিপ পেলে কেনার সুযোগ খুঁজুন।")
    elif avg_sentiment < -0.15:
        st.error(f"🔴 **BEARISH BIAS (Score: {avg_sentiment:+.2f})**\n\nখারাপ খবর বা সেলিং প্রেশারের পাল্লা ভারী। পুলব্যাকে শর্ট সেটআপ খুঁজুন।")
    else:
        st.warning(f"⚖️ **RANGE-BOUND / CHOP (Score: {avg_sentiment:+.2f})**\n\nমিশ্র সংবাদ। ট্রেন্ড পরিষ্কার না হওয়া পর্যন্ত কনফার্মেশনের অপেক্ষা করুন।")

with col_ai2:
