import os
import requests
import feedparser
import urllib.parse

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SENT_LOG_FILE = "sent_news.txt"

# আগে পাঠানো খবর ট্র্যাক রাখা যাতে একই নোটিফিকেশন বারবার না আসে
sent_headlines = set()
if os.path.exists(SENT_LOG_FILE):
    with open(SENT_LOG_FILE, "r", encoding="utf-8") as f:
        sent_headlines = set(line.strip() for line in f if line.strip())

queries = [
    '("GIFT Nifty" OR "SGX Nifty") when:1d',
    '("pre-market" OR "morning cues" OR "opening bell") (Nifty OR Sensex) when:1d',
    '("FII" AND "DII" AND "inflow" OR "outflow") when:1d'
]

trade_triggers = ["gift nifty", "sgx", "fii", "dii", "pre-market", "gap up", "gap down", "cues", "crude"]

new_alerts = []

for q in queries:
    encoded_query = urllib.parse.quote(q)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries[:5]:
        title = entry.title
        title_lower = title.lower()
        
        if any(trig in title_lower for trig in trade_triggers):
            if title not in sent_headlines:
                new_alerts.append(title)
                sent_headlines.add(title)

# নতুন খবর পেলে টেলিগ্রামে পাঠানো
if new_alerts and BOT_TOKEN and CHAT_ID:
    for news in new_alerts[:3]:  # একবারে সর্বোচ্চ ৩টি টপ নিউজ
        msg = f"⚡ *NIFTY DESK ALERT*\n\n📰 {news}\n\n🔗 [Open Dashboard](https://rudrakar100-nifty-trading-desk-app.streamlit.app)"
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload)
    
    # লগ আপডেট রাখা
    with open(SENT_LOG_FILE, "w", encoding="utf-8") as f:
        for item in list(sent_headlines)[-50:]:
            f.write(f"{item}\n")
    print(f"Sent {len(new_alerts)} alerts successfully.")
else:
    print("No new alerts.")
