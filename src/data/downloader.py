# src/data/downloader.py

from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import List, Optional
from ..config import (SYMBOLS, INTERVAL, START_DATE, END_DATE,
                      BASE_URL, RAW_DATA_PATH, PROCESSED_DATA_PATH)


class CryptoDataDownloader:
    def __init__(self, base_url: str = BASE_URL):
        self.client = Client(base_url=base_url)
        self.ensure_directories()

    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist."""
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    def download_symbol_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Download data for a single symbol."""
        all_klines = []
        current_date = start_date

        while current_date < end_date:
            next_date = min(current_date + timedelta(minutes=1000), end_date)
            try:
                klines = self.client.klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(current_date.timestamp() * 1000),
                    endTime=int(next_date.timestamp() * 1000),
                    limit=1000
                )
                all_klines.extend(klines)
            except Exception as e:
                print(f"Error downloading {symbol} data for" +
                      f"period {current_date}: {e}")
                continue

            current_date = next_date

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_volume',
                           'taker_buy_quote_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df['number_of_trades'] = df['number_of_trades'].astype(int)

        df['symbol'] = symbol
        return df

    def download_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = INTERVAL,
        start_date: str = START_DATE,
        end_date: str = END_DATE
    ) -> None:
        """Download data for all specified symbols and save to parquet."""
        symbols = symbols or SYMBOLS
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        for symbol in symbols:
            print(f"Downloading data for {symbol}...")
            df = self.download_symbol_data(symbol, interval, start_dt, end_dt)

            # Save to parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                os.path.join(PROCESSED_DATA_PATH, f"{symbol}_{interval}_\
                             {start_date}_{end_date}.parquet")
            )
            print(f"Saved {symbol} data to parquet")


if __name__ == "__main__":
    downloader = CryptoDataDownloader()
    downloader.download_all_symbols()
