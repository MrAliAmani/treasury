import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from unittest.mock import Mock, patch
import tempfile

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.data_processor import DataProcessor
from backend.data_loader import FREDLoader, fetch_economic_indicators, fetch_treasury_yields, DataLoadError

# Test data fixtures
@pytest.fixture
def sample_fred_data():
    """Sample FRED API response data"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', '2024-01-10'),
        'value': np.random.normal(2, 0.5, 10)
    })

@pytest.fixture
def sample_yield_data():
    """Sample yield curve data"""
    dates = pd.date_range('2024-01-01', periods=10)
    return pd.DataFrame({
        'date': dates,
        'maturity': ['2Y', '5Y', '10Y', '2Y', '5Y', '10Y', '2Y', '5Y', '10Y', '2Y'],
        'yield': np.random.normal(3, 0.3, 10)
    })

@pytest.fixture
def mock_fred_api():
    """Mock FRED API client"""
    with patch('fredapi.Fred') as mock_fred:
        mock_fred.return_value.get_series.return_value = pd.Series(
            np.random.normal(2, 0.5, 10),
            index=pd.date_range('2024-01-01', '2024-01-10')
        )
        yield mock_fred

@pytest.fixture
def data_processor():
    """DataProcessor instance with mocked dependencies"""
    return DataProcessor()

class TestDataMerging:
    """Test data merging functionality"""
    
    def test_merge_fred_yield_data(self, data_processor, sample_fred_data, sample_yield_data):
        """Test merging FRED data with yield curve data"""
        merged_df = data_processor.merge_datasets(sample_fred_data, sample_yield_data)
        
        assert isinstance(merged_df, pd.DataFrame)
        assert 'date' in merged_df.columns
        assert 'value' in merged_df.columns
        assert 'yield' in merged_df.columns
        assert len(merged_df) > 0
        
    def test_merge_with_missing_dates(self, data_processor):
        """Test merging behavior with missing dates"""
        df1 = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-05'),
            'value': np.random.normal(2, 0.5, 5)
        })
        df2 = pd.DataFrame({
            'date': pd.date_range('2024-01-03', '2024-01-07'),
            'yield': np.random.normal(3, 0.3, 5)
        })
        
        merged_df = data_processor.merge_datasets(df1, df2)
        
        assert len(merged_df) == 3  # Only overlapping dates
        assert not merged_df['value'].isna().any()
        assert not merged_df['yield'].isna().any()

class TestFREDAPIHandling:
    """Test FRED API error handling and retries"""
    
    @pytest.fixture
    def mock_fred(self, mocker):
        """Mock FRED API with proper patching"""
        # Mock the entire Fred class
        mock = mocker.patch('backend.data_loader.Fred', autospec=True)
        mock_instance = mock.return_value
        return mock_instance

    @pytest.mark.unit
    def test_fred_api_retry_on_error(self, mock_fred):
        """Test API retry mechanism on temporary failures"""
        # Setup mock responses
        mock_fred.get_series.side_effect = [
            ValueError("API Error"),
            pd.Series([1, 2, 3], index=pd.date_range('2024-01-01', periods=3))
        ]
        
        with patch.dict('os.environ', {'FRED_API_KEY': 'dummy_key'}):
            loader = FREDLoader()
            data = loader.get_indicator_data("GDP", "2024-01-01", "2024-01-03")
            
            assert isinstance(data, pd.DataFrame)
            assert mock_fred.get_series.call_count == 2

    @pytest.mark.unit
    def test_fred_api_invalid_series(self, mock_fred):
        """Test handling of invalid series IDs"""
        mock_fred.get_series.side_effect = ValueError("Invalid series ID")
        
        with patch.dict('os.environ', {'FRED_API_KEY': 'dummy_key'}):
            loader = FREDLoader()
            with pytest.raises(DataLoadError) as exc_info:
                loader.get_indicator_data("INVALID_SERIES", "2024-01-01", "2024-01-05")
            
            assert "Invalid series ID" in str(exc_info.value)

class TestCacheFunctionality:
    """Test data caching functionality"""
    
    def test_cache_save_load(self, data_processor, sample_fred_data):
        """Test saving and loading data from cache"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "test_cache.pkl")
            
            # Save to cache
            data_processor.save_to_cache(sample_fred_data, cache_file)
            
            # Load from cache
            loaded_data = data_processor.load_from_cache(cache_file)
            
            assert isinstance(loaded_data, pd.DataFrame)
            pd.testing.assert_frame_equal(loaded_data, sample_fred_data)
    
    def test_cache_expiry(self, data_processor, sample_fred_data):
        """Test cache expiry functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "test_cache.pkl")
            
            # Save to cache
            data_processor.save_to_cache(sample_fred_data, cache_file)
            
            # Modify file timestamp to simulate old cache
            old_time = datetime.now() - timedelta(days=2)
            os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))
            
            # Check if cache is expired
            assert data_processor.is_cache_expired(cache_file, max_age_hours=24)

class TestReportGeneration:
    """Test report generation functionality"""
    
    @pytest.mark.unit
    def test_excel_report_generation(self, data_processor, sample_fred_data):
        """Test Excel report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "report.xlsx")
            
            sheets = {
                'Raw Data': sample_fred_data,
                'Summary': sample_fred_data.describe()
            }
            
            data_processor.generate_excel_report(
                data=sample_fred_data,
                output_path=output_file,
                sheets=sheets
            )
            
            # Verify file exists and can be read
            assert os.path.exists(output_file)
            df_dict = pd.read_excel(output_file, sheet_name=None)
            assert set(df_dict.keys()) == set(sheets.keys())
    
    def test_pdf_report_generation(self, data_processor, sample_fred_data):
        """Test PDF report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "report.pdf")
            
            data_processor.generate_pdf_report(
                data=sample_fred_data,
                output_path=output_file,
                title="Test Report",
                sections=[
                    {'name': 'Data Overview', 'content': sample_fred_data.head()},
                    {'name': 'Statistics', 'content': sample_fred_data.describe()}
                ]
            )
            
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0

    def test_report_with_invalid_data(self, data_processor):
        """Test report generation with invalid data"""
        with pytest.raises(ValueError):
            data_processor.generate_excel_report(
                data=None,
                output_path="invalid.xlsx",
                sheets={}
            ) 