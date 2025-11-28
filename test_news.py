import yfinance as yf
try:
    ticker = yf.Ticker("AAPL")
    news = ticker.news
    print(f"News found: {len(news)}")
    if news:
        print(news[0])
except Exception as e:
    print(f"Error: {e}")
