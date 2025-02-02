import pytest
import pandas as pd
from backend.db_manager import DatabaseManager
from backend.data_fetcher import CachedDataFetcher
import os
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def db_manager():
    db = DatabaseManager(db_path=":memory:")  # Use in-memory database for testing
    yield db
    db.close()

@pytest.fixture
def data_fetcher():
    fetcher = CachedDataFetcher(batch_size=10)
    yield fetcher
    fetcher.cleanup()

def test_cache_insertion(db_manager):
    test_data = [
        {"id": 1, "value": "test1"},
        {"id": 2, "value": "test2"}
    ]
    
    db_manager.batch_insert(test_data, "test_key")
    cached_data = db_manager.get_cached_data("test_key")
    
    assert isinstance(cached_data, pd.DataFrame)
    assert len(cached_data) == 2
    assert list(cached_data.columns) == ["id", "value"]

async def test_batch_processing(data_fetcher):
    class MockAPIClient:
        async def get_data(self, limit, offset):
            if offset >= 30:  # Return 3 batches of data
                return []
            return [{"id": i, "value": f"test{i}"} for i in range(offset, offset + limit)]
    
    result = await data_fetcher.fetch_and_cache_data(MockAPIClient(), "test_batch")
    assert len(result) == 30 

def create_test_excel():
    """Create test Excel file with sample data"""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # Yield curve data
    yields_data = {
        'Date': dates,
        '2Y': np.random.normal(4, 0.5, len(dates)),
        '5Y': np.random.normal(4.2, 0.5, len(dates)),
        '10Y': np.random.normal(4.4, 0.5, len(dates))
    }
    
    # Economic indicators
    indicators_data = {
        'Date': dates,
        'NFP': np.random.normal(200, 50, len(dates)),
        'CPI MoM': np.random.normal(0.2, 0.1, len(dates)),
        'GDP QoQ': np.random.normal(2.5, 0.5, len(dates))
    }
    
    # Create Excel file
    with pd.ExcelWriter('test_data.xlsx') as writer:
        pd.DataFrame(yields_data).to_excel(writer, sheet_name='Yields', index=False)
        pd.DataFrame(indicators_data).to_excel(writer, sheet_name='Indicators', index=False)
    
    return 'test_data.xlsx'

def test_excel_yield_loader(data_fetcher):
    test_file = create_test_excel()
    
    try:
        # Test loading yield curves
        yields_df = data_fetcher.load_from_excel(
            test_file,
            data_type='yields',
            sheet_name='Yields',
            maturity_cols=['2Y', '5Y', '10Y']
        )
        
        assert isinstance(yields_df, pd.DataFrame)
        assert all(col in yields_df.columns for col in ['2Y', '5Y', '10Y'])
        assert yields_df.index.dtype == 'datetime64[ns]'
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def test_excel_indicators_loader(data_fetcher):
    test_file = create_test_excel()
    
    try:
        # Test loading indicators
        indicators_df = data_fetcher.load_from_excel(
            test_file,
            data_type='indicators',
            sheet_name='Indicators',
            indicator_cols={
                'NFP': 'NFP',
                'CPI': 'CPI MoM',
                'GDP': 'GDP QoQ'
            }
        )
        
        assert isinstance(indicators_df, pd.DataFrame)
        assert all(col in indicators_df.columns for col in ['NFP', 'CPI', 'GDP'])
        assert indicators_df.index.dtype == 'datetime64[ns]'
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file) 