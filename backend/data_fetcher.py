from backend.db_manager import DatabaseManager
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from .config.cache_config import CACHE_CONFIG

# Change the OpenBB import to be optional
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

class ExcelDataLoader:
    """Handles loading data from Excel files"""
    
    @staticmethod
    def load_yield_curves(file_path: Union[str, Path], 
                         sheet_name: Optional[str] = None,
                         date_col: str = 'Date',
                         maturity_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load yield curve data from Excel
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name containing yield data
            date_col: Name of date column
            maturity_cols: List of maturity column names
            
        Returns:
            DataFrame with yield curve data
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Validate required columns
            if date_col not in df.columns:
                raise ValueError(f"Date column '{date_col}' not found")
                
            if maturity_cols:
                missing_cols = [col for col in maturity_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing maturity columns: {missing_cols}")
            else:
                # Try to automatically detect yield columns
                maturity_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['1y', '2y', '5y', '10y', '30y', 'year']
                )]
            
            # Set date as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            return df[maturity_cols]
            
        except Exception as e:
            raise ValueError(f"Error loading yield curve data: {str(e)}")
    
    @staticmethod
    def load_economic_data(file_path: Union[str, Path],
                          sheet_name: Optional[str] = None,
                          date_col: str = 'Date',
                          indicator_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Load economic indicator data from Excel
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name containing indicator data
            date_col: Name of date column
            indicator_cols: Dict mapping indicator names to column names
            
        Returns:
            DataFrame with economic indicator data
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Validate date column
            if date_col not in df.columns:
                raise ValueError(f"Date column '{date_col}' not found")
            
            # Set up indicator columns
            if indicator_cols is None:
                # Try to automatically detect indicator columns
                possible_indicators = {
                    'NFP': ['nfp', 'payroll', 'employment'],
                    'CPI': ['cpi', 'inflation'],
                    'GDP': ['gdp', 'growth'],
                    'UNEMPLOYMENT': ['unemployment', 'jobless']
                }
                
                indicator_cols = {}
                for indicator, keywords in possible_indicators.items():
                    matching_cols = [col for col in df.columns if any(
                        kw in col.lower() for kw in keywords
                    )]
                    if matching_cols:
                        indicator_cols[indicator] = matching_cols[0]
            
            # Validate indicator columns
            missing_cols = [col for col in indicator_cols.values() if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing indicator columns: {missing_cols}")
            
            # Set date as index and select indicators
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Rename columns to standard names
            df = df[list(indicator_cols.values())]
            df.columns = list(indicator_cols.keys())
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading economic data: {str(e)}")

class CachedDataFetcher:
    def __init__(self, batch_size: int = CACHE_CONFIG['BATCH_SIZE']):
        self.db = DatabaseManager()
        self.batch_size = batch_size
        self.excel_loader = ExcelDataLoader()
    
    async def fetch_and_cache_data(self, api_client, cache_key: str) -> pd.DataFrame:
        """Fetch data with batch processing and cache results"""
        # Check cache first
        cached_data = self.db.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch new data in batches
        all_data = []
        
        async def fetch_batch(offset: int) -> List[Dict[str, Any]]:
            return await api_client.get_data(
                limit=self.batch_size,
                offset=offset
            )
        
        # Use ThreadPoolExecutor for parallel batch processing
        with ThreadPoolExecutor() as executor:
            offset = 0
            while True:
                batch = await fetch_batch(offset)
                if not batch:
                    break
                    
                all_data.extend(batch)
                offset += self.batch_size
                
                # Add small delay to prevent API rate limiting
                time.sleep(0.1)
        
        # Cache the fetched data
        self.db.batch_insert(all_data, cache_key)
        
        return pd.DataFrame(all_data)
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage"""
        # Downcast numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to categories when beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def cleanup(self):
        """Clean up resources"""
        self.db.clear_expired_cache()
        self.db.close()
    
    async def fetch_indicators(self, 
                             indicators: List[str],
                             start_date: str,
                             end_date: str,
                             source: str = 'excel') -> pd.DataFrame:
        """
        Fetch economic indicators from OpenBB or Excel
        
        Args:
            indicators: List of indicator names (e.g., ['NFP', 'CPI'])
            start_date: Start date string
            end_date: End date string
            source: Data source ('openbb' or 'excel')
            
        Returns:
            DataFrame with indicator data
        """
        cache_key = f"indicators_{','.join(indicators)}_{start_date}_{end_date}"
        
        # Check cache first
        cached_data = self.db.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        if source == 'openbb':
            if not OPENBB_AVAILABLE:
                print("OpenBB not available. Please install with 'pip install openbb'")
                return pd.DataFrame()
            
            # Use OpenBB SDK to fetch data
            all_data = []
            for indicator in indicators:
                try:
                    data = await obb.economy.historical(
                        symbol=indicator,
                        start_date=start_date,
                        end_date=end_date
                    )
                    all_data.append(data)
                except Exception as e:
                    print(f"Error fetching {indicator}: {str(e)}")
                    continue
            
            if all_data:
                combined_data = pd.concat(all_data, axis=1)
                self.db.batch_insert(combined_data.reset_index().to_dict('records'), cache_key)
                return combined_data
                
        return pd.DataFrame()
    
    def load_from_excel(self,
                       file_path: Union[str, Path],
                       data_type: str = 'yields',
                       **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Args:
            file_path: Path to Excel file
            data_type: Type of data to load ('yields' or 'indicators')
            **kwargs: Additional arguments for loader methods
            
        Returns:
            DataFrame with loaded data
        """
        if data_type == 'yields':
            return self.excel_loader.load_yield_curves(file_path, **kwargs)
        elif data_type == 'indicators':
            return self.excel_loader.load_economic_data(file_path, **kwargs)
        else:
            raise ValueError(f"Invalid data type: {data_type}") 