import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from textblob import TextBlob
import talib as tl

def loadHistoricalData(ticker):
    stock_data=pd.read_csv(f'../docs/yfinance_data/{ticker}_historical_data.csv')
    return stock_data


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def numberOfArticlesWithSentimentAnalysis(news_data):
    sentiment_counts = news_data['sentiment_score_word'].value_counts().sort_index()

    # Define colors for each category
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

    # Create the bar plot with specified colors
    sentiment_counts.plot(kind="bar", figsize=(10, 4), title='Sentiment Analysis',
                        xlabel='Sentiment categories', ylabel='Number of Published Articles',
                        color=[colors[category] for category in sentiment_counts.index])

    plt.show()


def getSentimentAnalysisOfPublisher(news_data, target_publisher):
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    # Filter data for the target publisher
    publisher_data = news_data[news_data['publisher'] == target_publisher]
    sentiment_counts = publisher_data['sentiment_score_word'].value_counts().sort_index()

    sentiment_counts.plot(kind="bar", figsize=(10, 4), title=f'Sentiment Analysis of {target_publisher}',
                      xlabel='Sentiment categories', ylabel='Number of Published Articles',
                      color=[colors[category] for category in sentiment_counts.index])


def checkMissingValueOfHistoricalDataset(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla):
    combined_df = pd.concat([stock_data_aapl.isnull().sum(),
                            stock_data_goog.isnull().sum(),
                            stock_data_amzn.isnull().sum(),
                            stock_data_msft.isnull().sum(),
                            stock_data_meta.isnull().sum(),
                            stock_data_nvda.isnull().sum(),
                            stock_data_tsla.isnull().sum()],
                            axis=0)
    combined_df.head()

def calculateDescriptiveStatisticsOfHistoricalData(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,stock_data_tsla):
    aapl_stats = stock_data_aapl['Close'].describe().to_frame('AAPL')
    goog_stats = stock_data_goog['Close'].describe().to_frame('GOOG')
    amzn_stats = stock_data_amzn['Close'].describe().to_frame('AMZN')
    msft_stats = stock_data_msft['Close'].describe().to_frame('MSFT')
    meta_stats = stock_data_meta['Close'].describe().to_frame('META')
    nvda_stats = stock_data_nvda['Close'].describe().to_frame('NVDA')
    tsla_stats = stock_data_tsla['Close'].describe().to_frame('TSLA')
    combined_stats = pd.concat([aapl_stats, goog_stats,amzn_stats,msft_stats,meta_stats,nvda_stats,tsla_stats], axis=1)
    return combined_stats

def analysisClosingPriceWithDate(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda):
    # Create subplots for side-by-side display
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Adjust figsize as needed

    axs[0,0].plot(stock_data_aapl['Date'], stock_data_aapl['Close'], label='Close',color='green')
    axs[0,0].set_title('AAPL')
    axs[0,0].legend()

    axs[0,1].plot(stock_data_amzn['Date'], stock_data_amzn['Close'], label='AMZN')
    axs[0,1].set_title('AMZN')
    axs[0,1].legend()


    axs[0,2].plot(stock_data_goog['Date'], stock_data_goog['Close'], label='Close',color='yellow')
    axs[0,2].set_title('GOOG')
    axs[0,2].legend()


    axs[1,0].plot(stock_data_nvda['Date'], stock_data_nvda['Close'], label='Close',color='brown')
    axs[1,0].set_title('NVDA')
    axs[1,0].legend()
    axs[1,0].set_xlabel('Date')


    axs[1,1].plot(stock_data_msft['Date'], stock_data_msft['Close'], label='Close',color='purple')
    axs[1,1].set_title('MSFT')
    axs[1,1].legend()
    axs[1,1].set_xlabel('Date')

    axs[1,2].plot(stock_data_meta['Date'], stock_data_meta['Close'], label='Close',color='orange')
    axs[1,2].set_title('META')
    axs[1,2].legend()
    axs[1,2].set_xlabel('Date')

    plt.show()

def calculateTechnicalIndicator(stock_data):
    stock_data['SMA'] = tl.SMA(stock_data['Close'], timeperiod=20)
    stock_data['RSI'] = tl.RSI(stock_data['Close'], timeperiod=14)
    stock_data['EMA'] = tl.EMA(stock_data['Close'], timeperiod=20)

    macd_signal, macd, _ = tl.MACD(stock_data['Close'])
    stock_data['MACD'] =macd
    stock_data['MACD_Signal']=macd_signal

def technicalIndicatorsVsClosingPrice(stock_data_aapl,stock_data_amzn,stock_data_goog,stock_data_meta,stock_data_msft,stock_data_nvda,ticker):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Adjust figsize as needed

    axs[0,0].plot(stock_data_aapl['Date'], stock_data_aapl['Close'], label='Closing price',color='green')
    axs[0,0].plot(stock_data_aapl['Date'], stock_data_aapl[ticker], label=ticker,color='red')
    axs[0,0].set_title('AAPL')
    axs[0,0].legend()

    axs[0,1].plot(stock_data_amzn['Date'], stock_data_amzn['Close'], label='Closing price')
    axs[0,1].plot(stock_data_amzn['Date'], stock_data_amzn[ticker], label=ticker,color='red')
    axs[0,1].set_title('AMZN')
    axs[0,1].legend()


    axs[0,2].plot(stock_data_goog['Date'], stock_data_goog['Close'], label='Closing price',color='yellow')
    axs[0,2].plot(stock_data_goog['Date'], stock_data_goog[ticker], label=ticker,color='red')
    axs[0,2].set_title('GOOG')
    axs[0,2].legend()


    axs[1,0].plot(stock_data_nvda['Date'], stock_data_nvda['Close'], label='Closing price',color='blue')
    axs[1,0].plot(stock_data_nvda['Date'], stock_data_nvda[ticker], label=ticker,color='red')
    axs[1,0].set_title('NVDA')
    axs[1,0].legend()
    axs[1,0].set_xlabel('Date')


    axs[1,1].plot(stock_data_msft['Date'], stock_data_msft['Close'], label='Closing price',color='purple')
    axs[1,1].plot(stock_data_msft['Date'], stock_data_msft[ticker], label=ticker,color='red')
    axs[1,1].set_title('MSFT')
    axs[1,1].legend()
    axs[1,1].set_xlabel('Date')

    axs[1,2].plot(stock_data_meta['Date'], stock_data_meta['Close'], label='Closing price',color='pink')
    axs[1,2].plot(stock_data_meta['Date'], stock_data_meta[ticker], label=ticker,color='red')
    axs[1,2].set_title('META')
    axs[1,2].legend()
    axs[1,2].set_xlabel('Date')

    plt.show()

