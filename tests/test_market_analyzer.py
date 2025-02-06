import pytest
from backend.market_analyzer import MarketDataAnalyzer
from backend.data_fetcher import CachedDataFetcher
import pandas as pd
from unittest.mock import Mock, patch

# Add OpenBB availability check
try:
    from fetch_openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    import numpy as np
    dates = pd.date_range(start='2024-01-01 09:30:00', 
                         end='2024-01-01 16:00:00',
                         freq='1min')
    
    return pd.DataFrame({
        'price': np.random.normal(100, 1, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

@pytest.fixture
def sample_releases():
    """Create sample economic releases"""
    return pd.DataFrame({
        'datetime': ['2024-01-01 10:00:00'],
        'event': ['NFP'],
        'surprise': [0.5]
    })

@pytest.fixture
def mock_openbb():
    """Mock OpenBB API responses"""
    mock_data = pd.DataFrame({
        'price': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
    
    with patch('backend.market_analyzer.obb') as mock_obb:
        # Set up mock responses
        mock_obb.market.historical = Mock(return_value=mock_data)
        mock_obb.economy.calendar = Mock(return_value=pd.DataFrame({
            'datetime': ['2024-01-01 10:00:00'],
            'event': ['NFP']
        }))
        yield mock_obb

@pytest.fixture
def market_analyzer(sample_market_data, sample_releases):
    """Create MarketDataAnalyzer instance with sample data"""
    data_fetcher = CachedDataFetcher()
    analyzer = MarketDataAnalyzer(data_fetcher)
    analyzer.market_data = sample_market_data
    analyzer.economic_releases = sample_releases
    return analyzer

def test_analyze_surprise_impact(market_analyzer):
    """Test surprise impact analysis"""
    results = market_analyzer.analyze_surprise_impact(
        window_minutes=15,
        threshold_std=2.0
    )
    
    assert isinstance(results, pd.DataFrame)
    assert all(col in results.columns for col in [
        'datetime', 'event', 'surprise', 
        'max_move', 'min_move', 'significant_moves'
    ])

@pytest.mark.asyncio
async def test_load_economic_releases(market_analyzer, mock_openbb):
    """Test loading economic calendar data"""
    start_date = '2024-01-01'
    end_date = '2024-01-02'

    # Await the coroutine
    releases = await market_analyzer.load_economic_releases(
        start_date=start_date,
        end_date=end_date
    )

    assert isinstance(releases, pd.DataFrame)
    if not releases.empty:
        assert 'datetime' in releases.columns
        assert 'event' in releases.columns

@pytest.mark.skipif(not OPENBB_AVAILABLE, reason="OpenBB not available")
def test_load_market_data_with_openbb(market_analyzer, mock_openbb):
    """Test loading market data using OpenBB"""
    start_date = '2024-01-01'
    end_date = '2024-01-02'
    
    market_data = market_analyzer.load_market_data(
        start_date=start_date,
        end_date=end_date,
        instrument='ZN=F'
    )
    
    assert isinstance(market_data, pd.DataFrame)
    if not market_data.empty:
        assert 'price' in market_data.columns
        assert 'volume' in market_data.columns 