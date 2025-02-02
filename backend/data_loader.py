"""
Data loader module for fetching economic data from FRED and other sources.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import os
from fredapi import Fred
from dotenv import load_dotenv
import logging
import time
from openbb import obb

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass

class FREDLoader:
    """Class for loading data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED API client
        
        Args:
            api_key: FRED API key (optional, defaults to env var)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise DataLoadError("FRED API key not found")
            
        try:
            self.fred = Fred(api_key=self.api_key)
        except Exception as e:
            raise DataLoadError(f"Failed to initialize FRED client: {str(e)}")
    
    def get_indicator_data(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                    **kwargs
                )
                return pd.DataFrame({series_id: series})
            except Exception as e:
                if attempt == max_retries - 1:
                    raise DataLoadError(f"Failed to fetch {series_id}: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple series from FRED
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            **kwargs: Additional arguments for fred.get_series
            
        Returns:
            pandas.DataFrame: Combined time series data
        """
        all_series = {}
        
        for series_id in series_ids:
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                    **kwargs
                )
                all_series[series_id] = series
                
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {str(e)}")
                continue
        
        if not all_series:
            raise DataLoadError("Failed to fetch any series")
            
        return pd.DataFrame(all_series)
    
    def get_vintage_data(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        vintage_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch vintage (historical) data for a series
        
        Args:
            series_id: FRED series ID
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            vintage_dates: List of vintage dates to fetch
            
        Returns:
            pandas.DataFrame: Vintage data for different release dates
        """
        try:
            vintages = {}
            
            if vintage_dates is None:
                # Get all available vintage dates
                info = self.fred.get_series_info(series_id)
                vintage_dates = pd.date_range(
                    start=info['realtime_start'],
                    end=info['realtime_end']
                ).strftime('%Y-%m-%d').tolist()
            
            for vintage_date in vintage_dates:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                    realtime_start=vintage_date,
                    realtime_end=vintage_date
                )
                vintages[vintage_date] = series
            
            return pd.DataFrame(vintages)
            
        except Exception as e:
            raise DataLoadError(f"Failed to fetch vintage data for {series_id}: {str(e)}")

def fetch_economic_indicators(
    indicators: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_vintage: bool = False
) -> pd.DataFrame:
    """
    Fetch economic indicator data from FRED
    
    Args:
        indicators: List of indicator series IDs
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        include_vintage: Whether to include vintage data
        
    Returns:
        pandas.DataFrame: Economic indicator data
    """
    try:
        loader = FREDLoader()
        
        if indicators is None:
            indicators = ['GDP', 'UNRATE', 'CPIAUCSL']  # Default indicators
            
        data = loader.get_multiple_series(indicators, start_date, end_date)
        
        if include_vintage:
            for indicator in indicators:
                vintage = loader.get_vintage_data(indicator, start_date, end_date)
                data[f"{indicator}_vintage"] = vintage.iloc[:, -1]  # Latest vintage
        
        return data
        
    except Exception as e:
        raise DataLoadError(f"Failed to fetch economic indicators: {str(e)}")

def fetch_treasury_yields(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    maturities: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch treasury yield data from FRED
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        maturities: List of yield maturities to fetch
        
    Returns:
        pandas.DataFrame: Treasury yield data
    """
    try:
        loader = FREDLoader()
        
        if maturities is None:
            maturities = ['DGS2', 'DGS5', 'DGS10', 'DGS30']  # Default maturities
            
        data = loader.get_multiple_series(maturities, start_date, end_date)
        
        # Rename columns to more readable format
        rename_map = {
            'DGS2': '2Y',
            'DGS5': '5Y',
            'DGS10': '10Y',
            'DGS30': '30Y'
        }
        data = data.rename(columns=rename_map)
        
        return data
        
    except Exception as e:
        raise DataLoadError(f"Failed to fetch treasury yields: {str(e)}")

def load_economic_calendar(start_date: str, end_date: str, countries: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load economic calendar data from OpenBB.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        countries: Optional list of country codes (e.g., ['US', 'EU'])
    
    Returns:
        DataFrame with economic events and consensus values
    """
    try:
        # Get economic calendar data
        calendar_df = obb.economy.calendar(
            start_date=start_date,
            end_date=end_date,
            countries=countries or ['US']  # Default to US if no countries specified
        )
        
        # Clean and process the data
        if not calendar_df.empty:
            # Standardize column names
            calendar_df = calendar_df.rename(columns={
                'date': 'date',
                'time': 'time',
                'country': 'country',
                'event': 'event',
                'actual': 'actual',
                'consensus': 'consensus',
                'previous': 'previous',
                'impact': 'impact'
            })
            
            # Convert date and time columns
            calendar_df['datetime'] = pd.to_datetime(
                calendar_df['date'].astype(str) + ' ' + calendar_df['time'].astype(str)
            )
            
            # Calculate surprise (actual vs consensus)
            calendar_df['surprise'] = pd.to_numeric(calendar_df['actual'], errors='coerce') - \
                                    pd.to_numeric(calendar_df['consensus'], errors='coerce')
            
            # Add normalized surprise
            calendar_df['surprise_std'] = calendar_df.groupby('event')['surprise'].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
            return calendar_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading economic calendar: {str(e)}")
        return pd.DataFrame()

def prepare_excel_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Excel Add-in export compatibility.
    
    Args:
        df: DataFrame with economic calendar data
    
    Returns:
        DataFrame formatted for Excel export
    """
    if df.empty:
        return df
        
    # Create Excel-friendly format
    excel_df = df.copy()
    
    # Format datetime for Excel
    excel_df['date'] = pd.to_datetime(excel_df['datetime']).dt.strftime('%Y-%m-%d')
    excel_df['time'] = pd.to_datetime(excel_df['datetime']).dt.strftime('%H:%M:%S')
    
    # Ensure numeric columns are properly formatted
    numeric_cols = ['actual', 'consensus', 'previous', 'surprise', 'surprise_std']
    for col in numeric_cols:
        if col in excel_df.columns:
            excel_df[col] = pd.to_numeric(excel_df[col], errors='coerce')
    
    # Add metadata for Excel Add-in
    excel_df.attrs['source'] = 'OpenBB Economic Calendar'
    excel_df.attrs['last_updated'] = datetime.now().isoformat()
    
    return excel_df

def combine_fred_and_calendar_data(
    fred_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    event_mapping: dict
) -> pd.DataFrame:
    """
    Combine FRED historical data with OpenBB calendar data.
    
    Args:
        fred_df: DataFrame with FRED historical data
        calendar_df: DataFrame with OpenBB calendar data
        event_mapping: Dictionary mapping FRED series to calendar events
    
    Returns:
        Combined DataFrame with historical and forward-looking data
    """
    if fred_df.empty or calendar_df.empty:
        return fred_df
    
    # Create a copy to avoid modifying original data
    combined_df = fred_df.copy()
    
    # Process calendar data for each FRED series
    for fred_series, event_name in event_mapping.items():
        if fred_series in combined_df.columns:
            # Get relevant calendar events
            events = calendar_df[calendar_df['event'] == event_name].copy()
            
            if not events.empty:
                # Add consensus and surprise data
                events = events.sort_values('datetime')
                latest_date = combined_df.index.max()
                
                # Add forward-looking consensus values
                future_events = events[events['datetime'] > latest_date]
                for _, event in future_events.iterrows():
                    if pd.notnull(event['consensus']):
                        combined_df.loc[event['datetime'], f"{fred_series}_consensus"] = event['consensus']
                        combined_df.loc[event['datetime'], f"{fred_series}_surprise"] = event['surprise_std']
    
    return combined_df

# Example event mapping
FRED_TO_CALENDAR_MAPPING = {
    'CPIAUCSL': 'CPI m/m',
    'GDPC1': 'GDP Growth Rate QoQ',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'INDPRO': 'Industrial Production m/m'
} 