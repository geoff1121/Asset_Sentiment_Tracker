import yfinance as yf
import requests
import feedparser
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from urllib.parse import quote_plus
import pandas as pd
import ta
from env import REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT
from collections import OrderedDict
import streamlit as st 

######### cardiffnlp/twitter-roberta-base-sentiment pre-trained model #########
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

subreddits_map = {
    "BTC": ['cryptocurrency', 'bitcoin', 'CryptoMarkets'],
    "Gold": ['Gold', 'WallStreetSilver', 'Economics'],
    "SPY": ['investing', 'stocks', 'StockMarket', 'wallstreetbets']
}


def analyze_transformer_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()
    score = probs[2] - probs[0]  # Positive - Negative
    label = "Positive" if score > 0.3 else "Negative" if score < -0.3 else "Neutral"
    return round(score, 3), label


reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# ----- Market Data -----
@st.cache_data(ttl=300)
def get_price(asset):
    #print(f"⏱️ Fetching fresh data for {asset}")
    if asset == 'BTC':
        response = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                                params={"ids": "bitcoin", "vs_currencies": "usd"}).json()
        return response["bitcoin"]["usd"]
    elif asset == 'SPY':
        return yf.Ticker("SPY").history(period="1d")['Close'].iloc[-1]
    elif asset == 'Gold':
        return yf.Ticker("GLD").history(period="1d")['Close'].iloc[-1]
    return None

# ----- News -----
@st.cache_data(ttl=900)
def fetch_google_news_sentiment(query, limit=10):
    encoded_query = quote_plus(query)  # fixes the InvalidURL error
    url = f"https://news.google.com/rss/search?q={encoded_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    headlines = [entry.title for entry in feed.entries[:limit]]

    # Extract only the score from the (score, label) tuple
    scores = [analyze_transformer_sentiment(h)[0] for h in headlines]
    avg = sum(scores) / len(scores) if scores else 0
    return round(avg, 3)


# ----- Reddit -----
@st.cache_data(ttl=900)
def fetch_reddit_sentiment(query, asset, subreddits=None):
    if subreddits is None:
        default_map = {
            "BTC": ['cryptocurrency', 'bitcoin', 'CryptoMarkets'],
            "Gold": ['Gold', 'WallStreetSilver', 'Economics'],
            "SPY": ['investing', 'stocks', 'StockMarket', 'wallstreetbets']
        }
        subreddits = default_map.get(asset, ['investing'])

    posts = []
    for sub in subreddits:
        try:
            results = reddit.subreddit(sub).search(query, sort='new', limit=10)
            posts.extend([post.title for post in results])
        except Exception:
            continue

    scores = [analyze_transformer_sentiment(p)[0] for p in posts]
    avg = sum(scores) / len(scores) if scores else 0
    return round(avg, 3)




# ------Price Trend Function-----
import pandas as pd
import ta

@st.cache_data(ttl=600)
def get_price_and_indicators(asset, days=60):
    if asset == 'BTC':
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
        data = requests.get(url, params=params).json()
        prices = [p[1] for p in data['prices']]
        df = pd.DataFrame(prices, columns=['Close'])
    else:
        ticker = "SPY" if asset == 'SPY' else "GLD"
        df = yf.Ticker(ticker).history(period=f"{days}d")[['Close']]

    df.dropna(inplace=True)

    # RSI (14-period)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    latest = df.iloc[-1]
    trend = round(latest['Close'] - df['Close'].iloc[-4:-1].mean(), 2)

    return {
        'trend': trend,
        'RSI': round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else 0,
        'MACD': round(latest['MACD'], 2) if not pd.isna(latest['MACD']) else 0,
        'MACD_signal': round(latest['MACD_signal'], 2) if not pd.isna(latest['MACD_signal']) else 0,
    }

@st.cache_data(ttl=900)  # cache for 15 minutes
def fetch_google_news_articles(query, limit=10):
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries[:limit]:
        title = entry.title
        score, label = analyze_transformer_sentiment(title)
        articles.append({
            "Source": "News",
            "Asset": query,
            "Title": title,
            "Score": score,
            "Label": label
        })

    return articles

@st.cache_data(ttl=900)
def fetch_reddit_articles(query, subreddits, limit=10):
    articles = []

    for sub in subreddits:
        try:
            results = reddit.subreddit(sub).search(query, sort='new', limit=limit)
            for post in results:
                score, label = analyze_transformer_sentiment(post.title)
                articles.append({
                    "Source": f"Reddit ({sub})",
                    "Asset": query,
                    "Title": post.title,
                    "Score": score,
                    "Label": label
                })
        except Exception:
            continue
    return articles





# ----- Advisor Logic -----
def generate_advice(sentiment, trend, rsi, macd, macd_signal):
    advice = "HOLD"

    if sentiment > 0.3 and trend > 0:
        if macd > macd_signal and rsi < 70:
            advice = "STRONG BUY"
        else:
            advice = "BUY"
    elif sentiment < -0.3 and trend < 0:
        if macd < macd_signal and rsi > 30:
            advice = "STRONG SELL"
        else:
            advice = "SELL"
    
    return advice



def analyze_asset(name, query, weights=None, subreddits=None):
    # Use fallback weights if not passed
    if weights is None:
        weights = {
            "BTC": {"reddit": 0.7, "news": 0.3},
            "Gold": {"reddit": 0.4, "news": 0.6},
            "SPY": {"reddit": 0.3, "news": 0.7}
        }

    # Use fallback subreddit map if not passed
    if subreddits is None:
        subreddits = {
            "BTC": ['cryptocurrency', 'bitcoin', 'CryptoMarkets'],
            "Gold": ['Gold', 'WallStreetSilver', 'Economics'],
            "SPY": ['investing', 'stocks', 'StockMarket']
        }

    price = get_price(name)
    indicators = get_price_and_indicators(name)

    reddit_score = fetch_reddit_sentiment(query, name, subreddits=subreddits.get(name, []))
    news_score = fetch_google_news_sentiment(query)

    r_weight = weights.get(name, {}).get("reddit", 0.5)
    n_weight = weights.get(name, {}).get("news", 0.5)

    total_sentiment = round((reddit_score * r_weight + news_score * n_weight), 3)

    advice = generate_advice(
        total_sentiment,
        indicators['trend'],
        indicators['RSI'],
        indicators['MACD'],
        indicators['MACD_signal']
    )

    return {
        "Asset": name,
        "Price": price,
        "Price Trend": indicators['trend'],
        "RSI": indicators['RSI'],
        "MACD": indicators['MACD'],
        "MACD Signal": indicators['MACD_signal'],
        "Reddit Sentiment": reddit_score,
        "News Sentiment": news_score,
        "Total Sentiment": total_sentiment,
        "Reddit Weight": r_weight,
        "News Weight": n_weight,
        "Advice": advice
    }



# ----- Main -----
def main():
    
    assets = OrderedDict({
        'SPY': 'SPY OR S&P500',
        'BTC': 'Bitcoin OR BTC',
        'Gold': 'Gold OR XAU'  
    })

    for asset, query in assets.items():
        result = analyze_asset(asset, query)

        print(f"\n--- {result['Asset']} ---")
        print(f"Price: ${round(result['Price'], 2)}")
        print(f"Price Trend: {result['Price Trend']}")
        print(f"RSI: {result['RSI']}")
        print(f"MACD: {result['MACD']}")
        print(f"MACD Signal: {result['MACD Signal']}")
        print(f"Reddit Sentiment: {result['Reddit Sentiment']}")
        print(f"News Sentiment: {result['News Sentiment']}")
        print(f"→ Total Sentiment: {result['Total Sentiment']}")
        print(f"📈 Investment Advice: {result['Advice']}")



