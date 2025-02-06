#!/usr/bin/env python
import sys
import asyncio
import logging
import time
import random
import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Union

from fredapi import Fred
import pandas as pd
from datetime import datetime
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# On Windows, use the Selector event loop to avoid "Event loop is closed" warnings.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import external fetch functions.
from backend.fetch_fred import fetch_fred_data
from backend.fetch_openbb import fetch_data as fetch_openbb

# Add OpenBB import with proper error handling
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    logging.warning("OpenBB SDK not available. Install with 'pip install openbb'")
    OPENBB_AVAILABLE = False
    obb = None

class DataFetcher:
    """
    Main data fetcher class that coordinates fetching market data from FRED and OpenBB.
    Data is cached locally in an SQLite database.
    
    The class first attempts to fetch data from FRED (if an API key is set) and then falls back
    to OpenBB if FRED returns no data.
    """
    # Mapping for FRED series.
    fred_series_mapping = {
        'Inflation': 'CPIAUCSL',
        'Growth': 'GDPC1',
        'Unemployment': 'UNRATE'
    }

    # OpenBB mappings for two providers: ECONDB and IMF.
    openbb_mappings = {
        'econdb': {
            'Inflation': {'symbol': 'CPI', 'country': 'united_states'},
            'Growth': {'symbol': 'GDP', 'country': 'united_states'},
            'Unemployment': {'symbol': 'URATE', 'country': 'united_states'}
        },
        'imf': {
            'Inflation': {'symbol': 'PCPIPCH', 'country': 'US'},
            'Growth': {'symbol': 'NGDP_RPCH', 'country': 'US'},
            'Unemployment': {'symbol': 'LUR', 'country': 'US'}
        }
    }

    # Alternate indicator codes for IMF.
    imf_alternate = {
        "PCPIPCH": "PCPI_CH",   # Alternate for inflation
        "NGDP_RPCH": "NGDP_CH"    # Alternate for growth
    }

    def __init__(self, db_path: str = 'market_data.db', 
                 start_date: str = "1980-01-01", end_date: str = "2023-01-01"):
        """Initialize the DataFetcher with proper path handling"""
        # Convert db_path to absolute path if not in memory
        if db_path != ':memory:':
            db_path = str(Path(db_path).absolute())
        
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.default_country_econdb = "united_states"
        self.default_country_imf = "us"
        self.max_retries = 3
        
        # Ensure FRED API key is available
        if not os.getenv('FRED_API_KEY'):
            logging.warning("FRED_API_KEY not found in environment variables")
        
        # Initialize database
        self._init_db()
        
        # Log initialization
        logging.info(f"DataFetcher initialized with db: {db_path}")

    def _init_db(self):
        """Initialize the SQLite cache database if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp INTEGER,
                    expiry INTEGER
                )
            """)

    @staticmethod
    def print_available_indicators(provider: str):
        """Print available indicators for the given provider."""
        if not OPENBB_AVAILABLE:
            print(f"OpenBB SDK not available. Cannot fetch indicators for {provider}")
            return
            
        try:
            available = obb.economy.available_indicators(provider=provider)
            print(f"\nAvailable indicators for provider '{provider}':")
            print(available)
        except Exception as e:
            print(f"Error retrieving available indicators for provider '{provider}': {e}")

    @staticmethod
    def clean_imf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame that contains IMF data where cells may be tuples (key, value).
        Extracts the value from each tuple.
        """
        cleaned_rows = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, tuple) and len(val) == 2:
                    row_dict[col] = val[1]
                else:
                    row_dict[col] = val
            cleaned_rows.append(row_dict)
        return pd.DataFrame(cleaned_rows)

    def _attempt_fetch_openbb(self, provider: str, symbol: str, country: str) -> pd.DataFrame:
        """
        Attempt to fetch data from OpenBB using the given parameters.
        Implements exponential backoff with jitter.
        """
        for attempt in range(1, self.max_retries + 1):
            logging.info(f"[{provider.upper()}] Attempt {attempt}/{self.max_retries} for {symbol} using country {country}")
            try:
                data = fetch_openbb(
                    provider,
                    symbol,
                    country,
                    self.start_date,
                    self.end_date
                )
                if hasattr(data, "results") and data.results:
                    df = pd.DataFrame(data.results)
                    # Clean tuple values if necessary (common with IMF data)
                    if df.shape[1] >= 4 and isinstance(df.iloc[0, 0], tuple):
                        df = self.clean_imf_dataframe(df)
                    logging.info(f"[{provider.upper()}] Successfully fetched data for {symbol}")
                    return df
                else:
                    logging.error(f"[{provider.upper()}] Attempt {attempt} returned no time series data for {symbol}")
            except Exception as e:
                err_msg = str(e)
                logging.error(f"[{provider.upper()}] Attempt {attempt} failed for {symbol}: {err_msg}")
                # If rate-limiting is detected, wait extra time.
                if "rate-limiting" in err_msg.lower():
                    extra_wait = 10
                    logging.info(f"[{provider.upper()}] Detected rate-limiting; waiting extra {extra_wait} seconds.")
                    time.sleep(extra_wait)
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logging.info(f"[{provider.upper()}] Waiting {sleep_time:.2f} seconds before next attempt.")
            time.sleep(sleep_time)
        return pd.DataFrame()

    def fetch_data_from_openbb(self, provider: str, symbol: str, country: str) -> pd.DataFrame:
        """
        Fetch data from OpenBB for the specified provider, symbol, and country.
        If provider is IMF and the primary symbol fails, try an alternate indicator code.
        """
        if not OPENBB_AVAILABLE:
            logging.warning("OpenBB SDK not available. Skipping OpenBB data fetch.")
            return pd.DataFrame()

        df = self._attempt_fetch_openbb(provider, symbol, country)
        # For IMF, if no data is returned, try an alternate indicator code if available.
        if provider.lower() == "imf" and df.empty and symbol in self.imf_alternate:
            alt_symbol = self.imf_alternate[symbol]
            logging.info(f"[IMF] No data for {symbol}. Trying alternate indicator code: {alt_symbol}")
            df = self._attempt_fetch_openbb(provider, alt_symbol, country)
        if df.empty:
            logging.error(f"[{provider.upper()}] Failed to fetch data for {symbol}. Verify the indicator code, country, and date range.")
        return df

    def display_fetch_summary(self, data: pd.DataFrame, start_date: str, end_date: str, indicators: List[str]) -> None:
        """
        Display a summary of successfully fetched data.
        
        Args:
            data: The fetched DataFrame
            start_date: Start date of the data
            end_date: End date of the data
            indicators: List of requested indicators
        """
        if data.empty:
            logging.info("\n❌ No data was fetched.")
            return

        logging.info("\n✅ Data Fetch Summary:")
        logging.info("=" * 50)
        logging.info(f"📅 Date Range: {start_date} to {end_date}")
        logging.info(f"📊 Total Records: {len(data)}")
        
        # Show which indicators were successfully fetched
        fetched_indicators = data['indicator'].unique() if 'indicator' in data.columns else []
        logging.info("\nIndicators Status:")
        for indicator in indicators:
            status = "✅" if indicator in fetched_indicators else "❌"
            logging.info(f"{status} {indicator}")
        
        # Show data source used
        source = "FRED" if os.getenv('FRED_API_KEY') and not data.empty else "OpenBB"
        logging.info(f"\n🔍 Data Source: {source}")
        
        # Show cache status
        cache_key = f"market_data_{start_date}_{end_date}_{'-'.join(sorted(indicators))}"
        cached = self._get_cached_data(cache_key) is not None
        logging.info(f"💾 Cached: {'Yes' if cached else 'No'}")
        
        logging.info("=" * 50)

    def calculate_surprise(self, actual: float, expected: float) -> float:
        """
        Calculate the surprise factor between actual and expected values.
        
        Args:
            actual: Actual value from FRED
            expected: Expected value from ALFRED
            
        Returns:
            float: Surprise value (actual - expected)
        """
        if pd.isna(actual) or pd.isna(expected):
            return 0.0
        return actual - expected

    def fetch_market_data(self, start_date: str, end_date: str, indicators: List[str]) -> pd.DataFrame:
        """Main method to fetch market data with surprise calculations"""
        # Update date range.
        self.start_date = start_date
        self.end_date = end_date

        # Create a unique cache key.
        cache_key = f"market_data_{start_date}_{end_date}_{'-'.join(sorted(indicators))}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and not cached_data.empty:
            logging.info("Using cached market data")
            self.display_fetch_summary(cached_data, start_date, end_date, indicators)
            return cached_data

        # Try FRED first if FRED_API_KEY is set.
        fred_data = pd.DataFrame()
        if os.getenv('FRED_API_KEY'):
            try:
                fred_data = fetch_fred_data(start_date, end_date, indicators)
                if not fred_data.empty:
                    # Add surprise calculation
                    for indicator in indicators:
                        mask = fred_data['indicator'] == indicator
                        expected = self.fetch_alfred_vintage(indicator, start_date, end_date)
                        if not expected.empty:
                            fred_data.loc[mask, 'expected'] = expected['value']
                            fred_data.loc[mask, 'surprise'] = fred_data.loc[mask].apply(
                                lambda x: self.calculate_surprise(x['value'], x['expected']), 
                                axis=1
                            )
                    
                    self._cache_data(cache_key, fred_data)
                    self.display_fetch_summary(fred_data, start_date, end_date, indicators)
                    return fred_data
            except Exception as e:
                logging.error(f"FRED fetch failed: {e}")

        # Only try OpenBB if it's available
        if OPENBB_AVAILABLE:
            all_data = []
            for indicator in indicators:
                # Try providers in order: ECONDB then IMF.
                for provider in ["econdb", "imf"]:
                    mapping = self.openbb_mappings.get(provider, {}).get(indicator)
                    if mapping:
                        df = self.fetch_data_from_openbb(provider, mapping['symbol'], mapping['country'])
                        if not df.empty:
                            df['indicator'] = indicator
                            all_data.append(df)
                            break  # Stop after successful fetch for this indicator

            if all_data:
                openbb_data = pd.concat(all_data, ignore_index=True)
                self._cache_data(cache_key, openbb_data)
                self.display_fetch_summary(openbb_data, start_date, end_date, indicators)
                return openbb_data

        # If we have FRED data but no OpenBB data, return FRED data
        if not fred_data.empty:
            self.display_fetch_summary(fred_data, start_date, end_date, indicators)
            return fred_data

        # If all data sources fail, raise an exception.
        raise Exception("Failed to fetch data from all sources")

    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data from the SQLite database if it is not expired."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data, expiry FROM market_data_cache WHERE key = ?", (cache_key,))
                result = cursor.fetchone()
                if result and result[1] > time.time():
                    return pd.read_json(result[0])
        except Exception as e:
            logging.error(f"Cache read error: {e}")
        return None

    def _cache_data(self, cache_key: str, data: pd.DataFrame, expiry_hours: int = 24):
        """Cache the data in the SQLite database with an expiration time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                expiry = int(time.time() + expiry_hours * 3600)
                conn.execute(
                    "INSERT OR REPLACE INTO market_data_cache (key, data, timestamp, expiry) VALUES (?, ?, ?, ?)",
                    (cache_key, data.to_json(), int(time.time()), expiry)
                )
        except Exception as e:
            logging.error(f"Cache write error: {e}")

    def load_from_excel(self, file_path: Union[str, Path], data_type: str = 'yields', **kwargs) -> pd.DataFrame:
        """Load data from an Excel file. Extend this method as needed."""
        # Placeholder for Excel loading logic.
        return pd.DataFrame()

    def fetch_alfred_vintage(self, indicator: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical vintage data from ALFRED for consensus estimates.
        
        Args:
            indicator: Economic indicator name
            start_date: Start date string
            end_date: End date string
            
        Returns:
            DataFrame with expected values
        """
        try:
            # Get FRED series ID for the indicator
            series_id = self.fred_series_mapping.get(indicator)
            if not series_id:
                logging.warning(f"No FRED series mapping for indicator: {indicator}")
                return pd.DataFrame()

            # Initialize FRED API
            fred = Fred(api_key=os.getenv('FRED_API_KEY'))
            
            # Get vintage dates for this series
            vintage_dates = fred.get_vintage_dates(series_id, 
                                                 start_date=pd.to_datetime(start_date),
                                                 end_date=pd.to_datetime(end_date))
            
            if not vintage_dates:
                return pd.DataFrame()

            # Get the first vintage for each release date
            vintage_data = []
            for date in vintage_dates:
                try:
                    # Get the vintage data from the day before (consensus estimate)
                    prev_day = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    vintage = fred.get_series_vintage(series_id, prev_day)
                    if not vintage.empty:
                        df = pd.DataFrame({
                            'date': vintage.index,
                            'value': vintage.values,
                            'indicator': indicator
                        })
                        vintage_data.append(df)
                except Exception as e:
                    logging.error(f"Error fetching vintage for {date}: {str(e)}")
                    continue

            if vintage_data:
                return pd.concat(vintage_data, ignore_index=True)
            return pd.DataFrame()
        
        except Exception as e:
            logging.error(f"Error fetching ALFRED vintage: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Optional: Print available indicators for troubleshooting.
    DataFetcher.print_available_indicators("econdb")
    DataFetcher.print_available_indicators("imf")
    
    # Test the DataFetcher with sample indicators.
    fetcher = DataFetcher()
    start_date = "2000-01-01"
    end_date = "2023-01-01"
    indicators = ['Inflation', 'Growth', 'Unemployment']
    
    try:
        market_data = fetcher.fetch_market_data(start_date, end_date, indicators)
        if not market_data.empty:
            print("\nFetched Market Data:")
            print(market_data.head())
        else:
            print("No data was returned.")
    except Exception as e:
        print(f"Error: {e}")
