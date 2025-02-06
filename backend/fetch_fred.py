#!/usr/bin/env python
import logging
import os
import sys
import pandas as pd
from fredapi import Fred
import time
from typing import List, Optional

def fetch_fred_data(start_date: str, end_date: str, indicators: list) -> pd.DataFrame:
    """Fetch data from FRED API with retries"""
    if not os.getenv('FRED_API_KEY'):
        raise ValueError("FRED API key not found in environment variables")
    
    max_retries = 3
    retry_delay = 2
    
    try:
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        data_frames = []
        
        # Standard series mapping
        series_mapping = {
            'Inflation': 'CPIAUCSL',
            'Growth': 'GDPC1',
            'Unemployment': 'UNRATE'
        }
        
        for indicator in indicators:
            series_id = series_mapping.get(indicator)
            if not series_id:
                logging.warning(f"No FRED series mapping for indicator: {indicator}")
                continue
            
            # Try with retries
            for attempt in range(max_retries):
                try:
                    logging.info(f"Fetching {indicator} (series: {series_id}) from FRED - attempt {attempt + 1}")
                    
                    if attempt > 0:
                        time.sleep(retry_delay * attempt)
                    
                    data = fred.get_series(
                        series_id,
                        observation_start=pd.to_datetime(start_date),
                        observation_end=pd.to_datetime(end_date)
                    )
                    
                    if isinstance(data, pd.Series) and not data.empty:
                        df = pd.DataFrame({
                            'date': data.index,
                            'value': data.values,
                            'indicator': indicator
                        })
                        data_frames.append(df)
                        break  # Success, move to next indicator
                        
                except Exception as e:
                    logging.error(f"FRED fetch attempt {attempt + 1} failed for {indicator}: {str(e)}")
                    if attempt == max_retries - 1:
                        break
                    continue

        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"FRED API error: {str(e)}")
        return pd.DataFrame()

def main(start_date: str, end_date: str, indicators: list) -> pd.DataFrame:
    return fetch_fred_data(start_date, end_date, indicators)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test the fetcher
    start_date = "2000-01-01"
    end_date = "2023-01-01"
    indicators = ['Inflation', 'Growth', 'Unemployment']
    
    df = main(start_date, end_date, indicators)
    if not df.empty:
        print("\nFRED Data:")
        print(df)
    else:
        print("No data was returned")
