"""
FRED (Federal Reserve Economic Data) API configuration and utilities.
Provides organized access to economic indicators and implements retry logic for API calls.
"""

import os
import time
from functools import wraps
from typing import Any, Callable, Dict
import fredapi

# Initialize FRED API client
fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))

# Economic indicator series mappings
ECONOMIC_INDICATORS: Dict[str, str] = {
    # Output and Income
    "GDP": "GDPC1",                    # Real Gross Domestic Product
    "GDI": "A261RL1Q225SBEA",         # Real Gross Domestic Income
    "INDUSTRIAL_PRODUCTION": "INDPRO",  # Industrial Production Index
    
    # Labor Market
    "UNEMPLOYMENT": "UNRATE",          # Unemployment Rate
    "NONFARM_PAYROLL": "PAYEMS",       # Total Nonfarm Payrolls
    "LABOR_FORCE_PART": "CIVPART",     # Labor Force Participation Rate
    
    # Prices and Inflation
    "CPI": "CPIAUCSL",                # Consumer Price Index
    "CORE_CPI": "CPILFESL",           # Core Consumer Price Index
    "PCE": "PCEPI",                   # Personal Consumption Expenditures Price Index
    "CORE_PCE": "PCEPILFE",           # Core PCE Price Index
    
    # Interest Rates
    "SOFR": "SOFR",                   # Secured Overnight Financing Rate
    "EFFR": "DFF",                    # Effective Federal Funds Rate
}

# Treasury yield series
TREASURY_YIELDS: Dict[str, str] = {
    "T_BILL_1M": "DGS1MO",            # 1-Month Treasury Constant Maturity Rate
    "T_BILL_3M": "DGS3MO",            # 3-Month Treasury Constant Maturity Rate
    "T_BILL_6M": "DGS6MO",            # 6-Month Treasury Constant Maturity Rate
    "T_BILL_1Y": "DGS1",              # 1-Year Treasury Constant Maturity Rate
    "T_NOTE_2Y": "DGS2",              # 2-Year Treasury Constant Maturity Rate
    "T_NOTE_5Y": "DGS5",              # 5-Year Treasury Constant Maturity Rate
    "T_NOTE_10Y": "DGS10",            # 10-Year Treasury Constant Maturity Rate
    "T_BOND_30Y": "DGS30",            # 30-Year Treasury Constant Maturity Rate
}

def retry_on_failure(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that implements retry logic for API calls.
    
    Args:
        max_attempts (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        backoff_factor (float): Multiplicative factor for delay between retries
        exceptions (tuple): Tuple of exceptions to catch and retry
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise last_exception
                    
                    time.sleep(delay)
                    delay *= backoff_factor
                    
            raise last_exception
        return wrapper
    return decorator

@retry_on_failure()
def get_series(series_id: str, **kwargs: Any) -> Any:
    """
    Wrapper for FRED API get_series with retry logic.
    
    Args:
        series_id (str): FRED series ID
        **kwargs: Additional arguments to pass to fred.get_series
        
    Returns:
        pandas.Series: The requested FRED data series
    """
    return fred.get_series(series_id, **kwargs) 