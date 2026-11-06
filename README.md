# stockdashboard

Streamlit stock market analytics terminal: major indices, sector performance, ETF charts with technical indicators, fundamentals, and NewsAPI headlines with TextBlob sentiment.

## Features

- Market overview for S&P 500, Dow Jones, Nasdaq, and Russell 2000 (Yahoo Finance via yfinance)
- Market breadth widgets: S&P 500 RSI, VIX, 10Y Treasury yield
- Sector performance treemap and rankings (SPDR sector ETFs)
- ETF analysis for popular tickers (SPY, QQQ, DIA, IWM, VTI, VOO, VEA, VWO, AGG, GLD)
- Candlestick charts with EMA/SMA, Bollinger Bands, volume, and RSI (ta + Plotly)
- Fund metrics: market cap / assets, P/E, forward P/E, dividend yield, beta, 52-week change
- Market and ticker-related news via NewsAPI, with TextBlob sentiment labeling

## Requirements

See `requirements.txt` (pinned versions include streamlit, pandas, numpy, yfinance, plotly, textblob, newsapi-python, ta, and related libs).

## Setup

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain a [NewsAPI](https://newsapi.org/) key. The app currently passes an API key into `MarketNewsAggregator` in `stockdashboard.py`; move that key into Streamlit secrets or an environment variable before sharing or deploying.

## Usage

```bash
streamlit run stockdashboard.py
```

The UI opens in wide layout with market overview, sector performance, ETF analysis, and news sections.

## Notes

- Price and volume data come from Yahoo Finance (`yfinance`).
- Do not commit live NewsAPI keys; rotate any key that was previously hardcoded in source.
