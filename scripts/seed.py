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
    print(args.start, args.end)
    # Load the list of instruments to extract the data for.
    df = pd.read_csv(INSTRUMENTS_FILE)

    # Load the symbols column as a list and add the suffix to each stock.
    ticker_list = [s + SUFFIX for s in df["Symbol"].to_list()]
    # Append NIFTY500 index to the list as well since we need to compute the returns for the index also.
    ticker_list.append(NIFTY500_SYMBOL)

    # Append data of all stocks to this list.
    df_list = list()

    # For each ticker, fetch the data.
    for ticker in ticker_list:
        data = yf.download(
            ticker,
            start=args.start,
            end=args.end,
            group_by="Ticker",
            interval="1d",
            auto_adjust=True,
        )
        # Remove the suffix if it has.
        data["ticker"] = ticker.split(".NS")[0]
        df_list.append(data)

    # combine all dataframes into a single dataframe
    df = pd.concat(df_list)

    # save to csv
    df.to_csv(path.join(args.dir, TICKER_DATA_OUTPUT_FILE))


if __name__ == "__main__":
    main()
