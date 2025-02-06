import sys
from pathlib import Path
import warnings
import aiohttp
import streamlit as st
import logging

from backend.market_analyzer import MarketDataAnalyzer

# Add project root to Python path first
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pytest
import pytest

# Add at the top with other imports
from backend.data_fetcher import DataFetcher

# Suppress all DeprecationWarnings from openbb_core
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"openbb_core\..*"
)

# Suppress Streamlit warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    # Import dependencies inside fixture
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(
        start='2024-01-01 09:30:00',
        end='2024-01-01 16:00:00',
        freq='1min'
    )
    return pd.DataFrame({
        'price': np.random.normal(100, 1, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

@pytest.fixture
def sample_releases():
    """Create sample economic releases"""
    import pandas as pd
    return pd.DataFrame({
        'datetime': ['2024-01-01 10:00:00'],
        'event': ['NFP'],
        'surprise': [0.5]
    })

@pytest.fixture(scope="session")
def data_fetcher():
    """Create a DataFetcher instance for testing"""
    return DataFetcher(db_path=':memory:')  # Use in-memory SQLite for testing

@pytest.fixture(scope="session")
def market_analyzer(data_fetcher):
    """Create a MarketDataAnalyzer instance for testing"""
    return MarketDataAnalyzer(data_fetcher)

# Add Streamlit test configuration
@pytest.fixture(autouse=True)
def streamlit_test_env():
    """Configure Streamlit for testing"""
    # Create a mock session state
    if not hasattr(st, 'session_state'):
        setattr(st, 'session_state', {}) 