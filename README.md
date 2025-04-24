Asset Sentiment Tracker

Real-time sentiment analysis for financial assets from news and Reddit

Overview

Asset Sentiment Tracker is a Python-based command-line tool that analyzes real-time sentiment for major financial assets such as SPY, Bitcoin (BTC), and Gold.

It pulls and processes:

- News headlines for each asset
- Reddit discussions and investor posts
- Asset price data
- Sentiment scores using NLP

This tool helps you monitor the market mood and identify trends across news and subreddits.

Features

- Live sentiment breakdown from both News and Reddit
- Displays real-time prices for selected assets
- Combines multiple sentiment scores into a clear, readable summary
- Easily extendable to support more assets or data sources

Getting Started

1. Clone the repo:
git clone https://github.com/yourusername/asset-sentiment-tracker.git
cd asset-sentiment-tracker

2. Set up the virtual environment:
python3.11 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

3. Run the app:
python "invest app.py"

4. or use streamlit for a prettier interface:
streamlit run streamlit_app.py   



Requirements:

snscrape
pandas
textblob
requests
certifi
streamlit

Example Output:

Asset   Price (USD)
SPY     505.28
BTC     83,521.00
Gold    279.72

SPY News Sentiment: -0.26
SPY Reddit Sentiment: +0.02

BTC News Sentiment: +0.00
BTC Reddit Sentiment: +0.04

Gold News Sentiment: -0.05
Gold Reddit Sentiment: +0.02


⚠ Known Limitations:

- Twitter sentiment is temporarily disabled due to scraping limitations.
- Sentiment engine is basic (TextBlob); advanced models can be integrated.

Roadmap:

- Integrate Twitter/X via official API
- Support more assets (e.g. ETH, TSLA, NVDA)
- Add interactive charts and visual sentiment trends
