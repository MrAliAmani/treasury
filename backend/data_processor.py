"""
Data processor module for analyzing economic surprises and yield changes.
Handles calculation of surprise values, time series alignment, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging
from datetime import datetime, timedelta
import os
import pickle
from fpdf import FPDF
import sys

# Add project root to Python path when running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(os.path.dirname(current_dir))

# Use absolute imports
from backend.data_loader import (
    fetch_economic_indicators,
    fetch_treasury_yields,
    DataLoadError
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessError(Exception):
    """Custom exception for data processing errors"""
    pass

def calculate_surprise(
    actual: float,
    estimate: float,
    std_dev: float
) -> float:
    """
    Calculate standardized surprise value.
    
    Args:
        actual: Actual value
        estimate: Estimated/forecast value
        std_dev: Standard deviation of historical surprises
        
    Returns:
        float: Standardized surprise value
    """
    if std_dev == 0:
        return 0.0
    return (actual - estimate) / std_dev

def calculate_yield_changes(
    yields_df: pd.DataFrame,
    window: str = '1D'
) -> pd.DataFrame:
    """
    Calculate yield changes over specified window.
    
    Args:
        yields_df: DataFrame of treasury yields
        window: Time window for changes (e.g., '1D' for daily)
        
    Returns:
        pandas.DataFrame: Yield changes
    """
    try:
        # Calculate changes
        changes = yields_df.diff(periods=1)
        
        # Add rolling statistics
        for col in changes.columns:
            changes[f"{col}_MA5"] = changes[col].rolling(window=5).mean()
            changes[f"{col}_MA20"] = changes[col].rolling(window=20).mean()
            changes[f"{col}_STD20"] = changes[col].rolling(window=20).std()
            
        return changes
        
    except Exception as e:
        raise DataProcessError(f"Error calculating yield changes: {str(e)}")

def detect_outliers(
    series: pd.Series,
    z_threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Args:
        series: Time series data
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        pandas.Series: Boolean mask of outliers
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    return pd.Series(
        z_scores > z_threshold,
        index=series.dropna().index
    )

class EconomicSurpriseAnalyzer:
    """Analyzes economic surprises and their relationship with yield changes."""
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.surprises_df = None
        self.yields_df = None
        self.changes_df = None
        
    def prepare_data(
        self,
        indicators: Optional[List[str]] = None,
        maturities: Optional[List[str]] = None
    ) -> None:
        """
        Fetch and prepare economic and yield data.
        
        Args:
            indicators: List of economic indicators to analyze
            maturities: List of treasury maturities to analyze
        """
        try:
            # Fetch economic data with vintage (estimates)
            self.surprises_df = fetch_economic_indicators(
                indicators=indicators,
                start_date=self.start_date,
                end_date=self.end_date,
                include_vintage=True
            )
            
            # Fetch treasury yields
            self.yields_df = fetch_treasury_yields(
                start_date=self.start_date,
                end_date=self.end_date,
                maturities=maturities
            )
            
            # Calculate yield changes
            self.changes_df = calculate_yield_changes(self.yields_df)
            
        except Exception as e:
            raise DataProcessError(f"Error preparing data: {str(e)}")
    
    def calculate_surprises(self) -> pd.DataFrame:
        """
        Calculate standardized surprises for economic indicators.
        
        Returns:
            pandas.DataFrame: Surprise values and related statistics
        """
        if self.surprises_df is None:
            raise DataProcessError("Data not prepared. Call prepare_data first.")
            
        results = {}
        
        for indicator in self.surprises_df.columns:
            if '_' not in indicator:  # Skip vintage columns
                continue
                
            base_indicator = indicator.split('_')[0]
            vintage_data = self.surprises_df[indicator]
            actual_data = self.surprises_df[base_indicator]
            
            # Calculate surprise statistics
            diff = actual_data - vintage_data
            std_dev = diff.std()
            
            surprises = diff.apply(
                lambda x: calculate_surprise(x, 0, std_dev)
            )
            
            results[f"{base_indicator}_surprise"] = surprises
            results[f"{base_indicator}_std"] = pd.Series(
                std_dev,
                index=surprises.index
            )
            
        return pd.DataFrame(results)
    
    def analyze_yield_response(
        self,
        surprise_df: pd.DataFrame,
        window: str = '1D'
    ) -> pd.DataFrame:
        """
        Analyze yield changes in response to economic surprises.
        
        Args:
            surprise_df: DataFrame of economic surprises
            window: Time window for analysis
            
        Returns:
            pandas.DataFrame: Analysis results
        """
        if self.changes_df is None:
            raise DataProcessError("Yield changes not calculated.")
            
        results = {}
        
        # Align dates and calculate correlations
        aligned_data = pd.merge(
            surprise_df,
            self.changes_df,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Detect outliers in yield changes
        for col in self.changes_df.columns:
            if not col.endswith(('MA5', 'MA20', 'STD20')):
                outliers = detect_outliers(self.changes_df[col])
                results[f"{col}_outliers"] = outliers
        
        # Add rolling correlations
        for surprise_col in surprise_df.columns:
            if surprise_col.endswith('_surprise'):
                for yield_col in self.changes_df.columns:
                    if not yield_col.endswith(('MA5', 'MA20', 'STD20')):
                        corr = aligned_data[surprise_col].rolling(20).corr(
                            aligned_data[yield_col]
                        )
                        results[f"{surprise_col}_{yield_col}_corr"] = corr
        
        return pd.DataFrame(results)

def create_analysis_summary(
    analyzer: EconomicSurpriseAnalyzer,
    indicators: List[str],
    maturities: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive analysis summary.
    
    Args:
        analyzer: Initialized EconomicSurpriseAnalyzer
        indicators: List of indicators to analyze
        maturities: List of maturities to analyze
        
    Returns:
        Dict[str, pd.DataFrame]: Analysis results
    """
    try:
        analyzer.prepare_data(indicators, maturities)
        surprises = analyzer.calculate_surprises()
        response = analyzer.analyze_yield_response(surprises)
        
        return {
            'surprises': surprises,
            'yield_changes': analyzer.changes_df,
            'response_analysis': response
        }
        
    except Exception as e:
        raise DataProcessError(f"Error creating analysis summary: {str(e)}")

class DataProcessor:
    """Class for processing and managing financial data"""
    
    def __init__(self):
        """Initialize DataProcessor"""
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two datasets on date column
        
        Args:
            df1: First DataFrame with 'date' column
            df2: Second DataFrame with 'date' column
            
        Returns:
            Merged DataFrame
        """
        if 'date' not in df1.columns or 'date' not in df2.columns:
            raise ValueError("Both DataFrames must have a 'date' column")
            
        return pd.merge(df1, df2, on='date', how='inner')

    def save_to_cache(self, data: pd.DataFrame, cache_file: str) -> None:
        """
        Save data to cache file
        
        Args:
            data: DataFrame to cache
            cache_file: Path to cache file
        """
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_file: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache file
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Cached DataFrame or None if file doesn't exist
        """
        if not os.path.exists(cache_file):
            return None
            
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def is_cache_expired(self, cache_file: str, max_age_hours: int = 24) -> bool:
        """
        Check if cache file is expired
        
        Args:
            cache_file: Path to cache file
            max_age_hours: Maximum age in hours before cache is considered expired
            
        Returns:
            True if cache is expired, False otherwise
        """
        if not os.path.exists(cache_file):
            return True
            
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age = datetime.now() - file_time
        
        return age.total_seconds() > (max_age_hours * 3600)

    def generate_excel_report(self, data: pd.DataFrame, output_path: str,
                            sheets: Dict[str, pd.DataFrame]) -> None:
        """
        Generate Excel report with multiple sheets
        
        Args:
            data: Main DataFrame
            output_path: Path to save Excel file
            sheets: Dict of sheet names and DataFrames
        """
        if data is None or not isinstance(sheets, dict):
            raise ValueError("Invalid data or sheets parameter")
            
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def generate_pdf_report(self, data: pd.DataFrame, output_path: str,
                          title: str, sections: List[Dict[str, Union[str, pd.DataFrame]]]) -> None:
        """
        Generate PDF report with multiple sections
        
        Args:
            data: Main DataFrame
            output_path: Path to save PDF file
            title: Report title
            sections: List of dicts with 'name' and 'content' keys
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        
        # Add sections
        pdf.set_font('Arial', size=12)
        for section in sections:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, section['name'], ln=True)
            pdf.ln(5)
            
            pdf.set_font('Arial', size=10)
            content = section['content']
            if isinstance(content, pd.DataFrame):
                content = content.to_string()
            pdf.multi_cell(0, 5, str(content))
            pdf.ln(10)
        
        pdf.output(output_path)

    def process_indicator_data(self, data: pd.DataFrame,
                             indicators: List[str]) -> pd.DataFrame:
        """
        Process indicator data with basic calculations
        
        Args:
            data: DataFrame with indicator data
            indicators: List of indicator names to process
            
        Returns:
            Processed DataFrame
        """
        if not all(ind in data.columns for ind in indicators):
            raise ValueError("Not all indicators found in data")
            
        result = data.copy()
        
        # Calculate basic statistics
        for indicator in indicators:
            result[f'{indicator}_ma_5'] = result[indicator].rolling(window=5).mean()
            result[f'{indicator}_std'] = result[indicator].rolling(window=20).std()
            result[f'{indicator}_zscore'] = (
                (result[indicator] - result[indicator].mean()) / result[indicator].std()
            )
            
        return result.dropna()

    def calculate_correlations(self, data: pd.DataFrame,
                             indicators: List[str]) -> pd.DataFrame:
        """
        Calculate correlation matrix between indicators
        
        Args:
            data: DataFrame with indicator data
            indicators: List of indicator names
            
        Returns:
            Correlation matrix DataFrame
        """
        if not all(ind in data.columns for ind in indicators):
            raise ValueError("Not all indicators found in data")
            
        return data[indicators].corr() 