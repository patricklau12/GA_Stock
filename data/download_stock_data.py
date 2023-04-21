import yfinance as yf
yf.download('AMZN').to_csv('data/AMZN.csv')
yf.download('TSLA').to_csv('data/TSLA.csv')
yf.download('AAPL').to_csv('data/AAPL.csv')
yf.download('GOOG').to_csv('data/GOOG.csv')
yf.download('AMD').to_csv('data/AMD.csv')
yf.download('NVDA').to_csv('data/NVDA.csv')
yf.download('MSFT').to_csv('data/MSFT.csv')