import streamlit as st
st.set_page_config(page_title="Investment Advisor", layout="wide")
st.title("📊 Investment Advisor Dashboard")

import pandas as pd
from collections import OrderedDict
from invest_advisor_roberta import analyze_asset
from invest_advisor_roberta import fetch_google_news_articles, fetch_reddit_articles



# Ordered asset config
assets = OrderedDict({
    'SPY': 'SPY OR S&P500',
    'BTC': 'Bitcoin OR BTC',
    'Gold': 'Gold OR XAU'
})

# === Sidebar: Live Tuning Panel ===
with st.sidebar:
    st.header("⚙️ Live Tuning Panel")
    
    st.subheader("📢 Subreddit Selection")
    st.caption("Customize subreddits used to analyze sentiment for each asset.")
    
    default_subs = {
        "SPY": "investing,stocks,StockMarket,wallstreetbets",
        "BTC": "cryptocurrency,bitcoin,CryptoMarkets",
        "Gold": "Gold,WallStreetSilver,Economics"
    }
    subreddit_inputs = {}
    for name in assets:
        subreddit_inputs[name] = st.text_input(f"{name}", value=default_subs[name])

    st.markdown("---")

    st.subheader("🧠 Sentiment Bias")
    st.caption("Slide left for **News** influence, right for **Reddit** influence.")

    default_weights = {
        "SPY": 0.5,
        "BTC": 0.5,
        "Gold": 0.5
    }
    reddit_weights = {}
    for name in assets:
        reddit_weights[name] = st.slider(f"{name}", 0.0, 1.0, default_weights[name])


# === Build dynamic mappings ===
weights = {
    name: {
        "reddit": reddit_weights[name],
        "news": round(1 - reddit_weights[name], 2)
    }
    for name in assets
}

subreddits_map = {
    name: [s.strip() for s in subreddit_inputs[name].split(",")]
    for name in assets
}

# === Main Dashboard ===
for name, query in assets.items():
    result = analyze_asset(name, query, weights=weights, subreddits=subreddits_map)

    advice_color = {
        "STRONG BUY": "green",
        "BUY": "lightgreen",
        "HOLD": "orange",
        "SELL": "red",
        "STRONG SELL": "darkred"
    }.get(result["Advice"], "gray")

    with st.container():
        st.subheader(f"🪙 {result['Asset']}")
        st.caption(f"📊 Reddit weight: {result['Reddit Weight']} | News weight: {result['News Weight']}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Price", f"${round(result['Price'], 2)}")
            st.metric("Trend", f"{result['Price Trend']:+.2f}")

        with col2:
            st.metric("RSI", f"{result['RSI']:.2f}")
            st.metric("MACD", f"{result['MACD']:.2f}")
            st.metric("MACD Signal", f"{result['MACD Signal']:.2f}")

        with col3:
            st.metric("Reddit Sentiment", f"{result['Reddit Sentiment']:.2f}")
            st.metric("News Sentiment", f"{result['News Sentiment']:.2f}")
            st.metric("Total Sentiment", f"{result['Total Sentiment']:.2f}")

        st.markdown(
            f"### 🧠 Advice: <span style='color:{advice_color};'>{result['Advice']}</span>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # 👇 Sentiment audit breakdown
        news_articles = fetch_google_news_articles(query)
        reddit_articles = fetch_reddit_articles(query, subreddits=subreddits_map[name])
        df_sentiment = pd.DataFrame(news_articles + reddit_articles)

        # 🔧 Round scores for clean UI
        df_sentiment['Score'] = df_sentiment['Score'].round(3)

        # (Optional) Sort by Score or Source
        # df_sentiment = df_sentiment.sort_values(by='Score', ascending=False)

        with st.expander(f"🧠 Sentiment Audit for {name}"):
            st.dataframe(
                df_sentiment[['Source', 'Title', 'Score', 'Label']],
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.2f"),
                    "Title": st.column_config.TextColumn(),
                    "Label": st.column_config.TextColumn()
                }
            )
