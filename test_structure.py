import yfinance as yf
import json

# Test básico
ticker = yf.Ticker("AAPL")
news = ticker.news

if news:
    print("Estructura primer item:")
    print(json.dumps(news[0], indent=2))
