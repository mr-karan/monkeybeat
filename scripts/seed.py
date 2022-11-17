#!/usr/bin/python3
import argparse
import logging
from datetime import date
from os import getcwd, path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
)

INSTRUMENTS_FILE = "ind_nifty500list.csv"
TICKER_DATA_OUTPUT_FILE = "ticker.csv"
INDEX_DATA_OUTPUT_FILE = "index.csv"
SUFFIX = ".NS"
NIFTY500_SYMBOL = "^CRSLDX"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--from",
    type=date.fromisoformat,
    help="Start date to fetch the data for",
    dest="start",
    required=True,
)
parser.add_argument(
    "--to",
    type=date.fromisoformat,
    help="End date to fetch the data for",
    dest="end",
    required=True,
)
parser.add_argument(
    "--output",
    type=str,
    help="Directory to store the files",
    dest="dir",
    default=getcwd(),
    required=False,
)
args = parser.parse_args()


def main():
    # Load the list of instruments to extract the data for.
    df = pd.read_csv(INSTRUMENTS_FILE)
    # Load the symbols column as a list and add the suffix to each stock.
    ticker_list = df["Symbol"].to_list()

    # Download data for stocks.
    download(ticker_list, False, TICKER_DATA_OUTPUT_FILE)

    # Download data for indices.
    download([NIFTY500_SYMBOL], True, INDEX_DATA_OUTPUT_FILE)


def download(symbols: str, is_index: bool, filename: str):
    # Append data of all stocks to this list.
    df_list = list()

    # For each ticker, fetch the data.
    for ticker in symbols:
        # Add a suffix `.NS` for the stocks.
        if not is_index:
            ticker = ticker + SUFFIX

        data = yf.download(
            ticker,
            start=args.start,
            end=args.end,
            group_by="Ticker",
            interval="1d",
            auto_adjust=True,
        )
        data["ticker"] = cleanup_name(ticker)
        data["segment"] = "EQ"
        if is_index:
            data["segment"] = "INDEX"
        df_list.append(data)

    # Combine all dataframes into a single dataframe.
    df = pd.concat(df_list)

    # Save to csv.
    df.to_csv(path.join(args.dir, filename))


def cleanup_name(ticker: str) -> str:
    ticker = ticker.removeprefix("^")
    ticker = ticker.removesuffix(".NS")
    return ticker


if __name__ == "__main__":
    main()
