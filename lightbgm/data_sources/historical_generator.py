"""Historical dataset generator for trading indices."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import yfinance as yf
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for historical data generation."""
    symbols: List[str]
    timeframes: List[str]
    start_date: datetime
    end_date: datetime
    data_source: str = "yfinance"  # "yfinance" or "mt5"


class HistoricalDataGenerator:
    """
    Generates historical dataset for ML training.
    
    Processes 3-5 years of data for US30, US100, US500 across multiple timeframes
    to create training dataset for setup detection and scoring.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_cache = {}
        
    def generate_dataset(self, output_path: str) -> pd.DataFrame:
        """
        Generate complete historical dataset.
        
        Args:
            output_path: Path to save the generated dataset
            
        Returns:
            DataFrame with historical price/volume data
        """
        all_data = []
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                print(f"Processing {symbol} - {timeframe}")
                
                # Fetch data
                data = self._fetch_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    # Add metadata
                    data['symbol'] = symbol
                    data['timeframe'] = timeframe
                    data['timestamp'] = data.index
                    
                    # Reset index to make timestamp a column
                    data = data.reset_index()
                    
                    all_data.append(data)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save to parquet for efficient storage
            output_file = Path(output_path) / "historical_dataset.parquet"
            combined_data.to_parquet(output_file)
            
            print(f"Dataset saved to {output_file}")
            print(f"Total records: {len(combined_data)}")
            
            return combined_data
        else:
            raise ValueError("No data could be fetched for any symbol/timeframe")
    
    def _fetch_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from configured source."""
        if self.config.data_source == "yfinance":
            return self._fetch_yfinance_data(symbol, timeframe)
        elif self.config.data_source == "mt5":
            return self._fetch_mt5_data(symbol, timeframe)
        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")
    
    def _fetch_yfinance_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        try:
            # Map symbols to Yahoo Finance tickers
            symbol_map = {
                "US30": "^DJI",
                "US100": "^NDX", 
                "US500": "^GSPC"
            }
            
            ticker = symbol_map.get(symbol, symbol)
            
            # Map timeframes to yfinance intervals
            interval_map = {
                "D1": "1d",
                "H4": "1h",  # Note: yfinance doesn't have 4h, using 1h
                "H1": "1h",
                "M15": "15m"
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            # Fetch data
            data = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                print(f"No data received for {symbol} ({ticker}) - {timeframe}")
                return None
                
            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            # For 4H timeframe, resample 1H data
            if timeframe == "H4" and interval == "1h":
                data = self._resample_to_4h(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching {symbol} - {timeframe}: {e}")
            return None
    
    def _fetch_mt5_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from MetaTrader 5."""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                print("Failed to initialize MT5")
                return None
            
            # Map timeframes to MT5 constants
            timeframe_map = {
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                print(f"Unsupported timeframe: {timeframe}")
                return None
            
            # Fetch data
            rates = mt5.copy_rates_range(
                symbol,
                mt5_timeframe,
                self.config.start_date,
                self.config.end_date
            )
            
            if rates is None or len(rates) == 0:
                print(f"No data received for {symbol} - {timeframe}")
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame(rates)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            data = data.set_index('time')
            
            # Rename columns to match standard format
            data = data.rename(columns={
                'tick_volume': 'volume'
            })
            
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching MT5 data for {symbol} - {timeframe}: {e}")
            return None
        finally:
            mt5.shutdown()
    
    def _resample_to_4h(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H data to 4H timeframe."""
        resampled = data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled


def create_sample_dataset():
    """Create a sample dataset for testing."""
    config = DataConfig(
        symbols=["US30", "US100", "US500"],
        timeframes=["D1", "H4", "H1", "M15"],
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2024, 1, 1),
        data_source="yfinance"
    )
    
    generator = HistoricalDataGenerator(config)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return generator.generate_dataset(str(data_dir))


if __name__ == "__main__":
    # Generate sample dataset
    dataset = create_sample_dataset()
    print("Dataset generation complete!")