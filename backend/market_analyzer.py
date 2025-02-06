import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
from .data_fetcher import DataFetcher
from backend.data_processor import DataProcessError

# Add OpenBB check
try:
    from fetch_openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

# Add project root to Python path when running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(os.path.dirname(current_dir))

class MarketDataAnalyzer:
    """Analyzes market movements around economic releases"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.market_data = None
        self.economic_releases = None
    
    def load_market_data(self, 
                        start_date: str,
                        end_date: str,
                        instrument: str = 'ZN=F') -> 'pd.DataFrame':
        """Load intraday market data"""
        try:
            import pandas as pd
            
            cache_key = f"market_data_{instrument}_{start_date}_{end_date}"
            cached_data = self.data_fetcher.db.get_cached_data(cache_key)
            
            if cached_data is not None:
                self.market_data = cached_data
                return cached_data
            
            if OPENBB_AVAILABLE:
                # Updated OpenBB API call
                market_data = obb.market.historical(
                    symbol=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1m"
                )
                
                if not market_data.empty:
                    self.market_data = market_data
                    self.data_fetcher.db.batch_insert(
                        market_data.reset_index().to_dict('records'),
                        cache_key
                    )
                    return market_data
            
            return pd.DataFrame()
            
        except Exception as e:
            raise DataProcessError(f"Error loading market data: {str(e)}")
    
    async def load_economic_releases(self, 
                                   start_date: str,
                                   end_date: str) -> pd.DataFrame:
        """Load economic calendar data asynchronously"""
        try:
            cache_key = f"economic_releases_{start_date}_{end_date}"
            cached_data = self.data_fetcher.db.get_cached_data(cache_key)
            
            if cached_data is not None:
                self.economic_releases = cached_data
                return cached_data
            
            if OPENBB_AVAILABLE:
                # OpenBB calendar is not async
                calendar_data = obb.economy.calendar(
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not calendar_data.empty:
                    self.economic_releases = calendar_data
                    self.data_fetcher.db.batch_insert(
                        calendar_data.reset_index().to_dict('records'),
                        cache_key
                    )
                    return calendar_data
            
            return pd.DataFrame()
            
        except Exception as e:
            raise DataProcessError(f"Error loading economic releases: {str(e)}")
    
    def analyze_surprise_impact(self,
                              window_minutes: int = 30,
                              threshold_std: float = 2.0) -> pd.DataFrame:
        """Analyze market impact around economic releases"""
        if self.market_data is None or self.economic_releases is None:
            raise DataProcessError("Market data and economic releases must be loaded first")
            
        results = []
        
        # Get market data at daily frequency for analysis
        market_data_daily = self.market_data.resample('D').last()
        
        for _, release in self.economic_releases.iterrows():
            release_date = pd.to_datetime(release['datetime']).date()
            
            # Get market data window around release
            window_start = release_date - timedelta(days=window_minutes)
            window_end = release_date + timedelta(days=window_minutes)
            
            window_data = market_data_daily.loc[window_start:window_end].copy()
            
            if not window_data.empty:
                # Calculate returns
                window_data['return'] = window_data['Close'].pct_change()
                
                # Calculate impact metrics
                results.append({
                    'date': release_date,
                    'indicator': release['event'],
                    'actual': release.get('actual', None),
                    'forecast': release.get('forecast', None),
                    'surprise': release.get('surprise', None),
                    'pre_return': window_data['return'].iloc[:-1].mean(),
                    'post_return': window_data['return'].iloc[1:].mean(),
                    'volatility': window_data['return'].std()
                })
        
        return pd.DataFrame(results)

    def analyze_economic_impact(self, df: pd.DataFrame, indicators: List[str], window_days: int = 30) -> pd.DataFrame:
        """Analyze the impact of economic releases on market movements"""
        if df.empty:
            return pd.DataFrame()
            
        results = []
        for indicator in indicators:
            indicator_data = df[df['indicator'] == indicator].copy()
            if not indicator_data.empty:
                # Sort by date
                indicator_data = indicator_data.sort_values('date')
                
                # Calculate returns and volatility
                indicator_data['value_change'] = indicator_data['value'].pct_change()
                
                # Calculate rolling metrics
                indicator_data['volatility'] = indicator_data['value_change'].rolling(window_days).std()
                indicator_data['pre_trend'] = indicator_data['value_change'].rolling(window_days).mean().shift(1)
                indicator_data['post_trend'] = indicator_data['value_change'].rolling(window_days).mean()
                
                # Store results
                results.append({
                    'indicator': indicator,
                    'avg_impact': indicator_data['value_change'].mean(),
                    'volatility': indicator_data['volatility'].mean(),
                    'pre_trend': indicator_data['pre_trend'].mean(),
                    'post_trend': indicator_data['post_trend'].mean(),
                    'observations': len(indicator_data)
                })
        
        return pd.DataFrame(results) 