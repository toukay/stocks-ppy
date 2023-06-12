import pandas as pd
from pandas import DataFrame
import numpy as np
import json
from argparse import ArgumentParser

from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


def main(tickers: list, file_paths: list, start_date: str, end_date: str):
    dataframes = []
    if file_paths:
        for fp, ticker in zip(file_paths, tickers):
            df = pd.read_csv(fp)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            dataframes.append(df)
    
    
    if (len(file_paths) < len(tickers)):
        tickers = tickers[len(file_paths):]
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        df = pdr.get_data_yahoo(tickers, start_date, end_date)
        if len(tickers) == 1:
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.columns = pd.MultiIndex.from_product([df.columns, tickers])
        dataframes.append(df)

    data = pd.concat(dataframes, axis=1)
    data.sort_index(axis=1, level=0, inplace=True)

    line_plot(data)


def line_plot(data: DataFrame):
    stacked_df = data.stack().reset_index()
    stacked_df = stacked_df.rename(columns={"level_1": "Company"})
    stacked_df['Date'] = pd.to_datetime(stacked_df['Date'])
    stacked_df = stacked_df[['Date', 'Close', 'Company']]
    plt.figure(figsize=(8,8))
    sns.lineplot(data=stacked_df, x='Date', y='Close', hue='Company')
    plt.title('Stock Price')
    plt.ylabel('USD')
    plt.show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--tickers', nargs='*', default=['AAPL'], help='Ticker')
    parser.add_argument('-fp', '--file-paths', nargs='*', default=[], help='File path')
    parser.add_argument('-sd', '--start-date', type=str, default='2022-01-01', help='Start date')
    parser.add_argument('-ed', '--end-date', type=str, default='2023-01-01', help='End date')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.tickers, args.file_paths, args.start_date, args.end_date)