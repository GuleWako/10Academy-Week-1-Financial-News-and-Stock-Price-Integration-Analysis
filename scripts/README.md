### loadHistoricalData(ticker): 
Loads historical stock data for a given ticker symbol from a CSV file.
### get_sentiment(text):
Analyzes the sentiment of a text string using TextBlob and returns a polarity score.
### numberOfArticlesWithSentimentAnalysis(news_data): 
Analyzes the sentiment distribution of news articles in the provided data and creates a bar chart.
### getSentimentAnalysisOfPublisher(news_data, target_publisher): 
Analyzes the sentiment of news articles from a specific publisher and creates a bar chart.
### checkMissingValueOfHistoricalDataset(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla): 
Checks for missing values in each historical stock data set and displays a summary.
### calculateDescriptiveStatisticsOfHistoricalData(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla): 
Calculates descriptive statistics (mean, standard deviation, etc.) for the closing prices of each stock and displays them in a table.
### analysisClosingPriceWithDate(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda): 
Creates time series plots for the closing prices of multiple stocks.
### calculateTechnicalIndicator(stock_data): 
Calculates technical indicators (SMA, RSI, EMA, MACD) for a given stock data set and adds them as new columns.
### technicalIndicatorsVsClosingPrice(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,ticker): 
Creates time series plots comparing the closing prices of multiple stocks with a chosen technical indicator.
### closingPriceRelativeStrengthIndex(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda): 
Creates time series plots for closing prices alongside the RSI indicator for multiple stocks.
### closingPriceMovingAverageConvergenceDivergence(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda): 
Creates time series plots for closing prices alongside the MACD indicator for multiple stocks.
### calculatePortfolioWeightAndPerformance(): 
Calculates optimal portfolio weights and ana