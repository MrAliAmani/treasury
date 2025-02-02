"""
Analysis module for advanced statistical analysis of economic surprises and yield curve dynamics.
Implements regression analysis, yield curve processing, and event studies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import logging
from datetime import datetime, timedelta

from backend.data_processor import (
    EconomicSurpriseAnalyzer,
    calculate_surprise,
    detect_outliers
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionAnalysis:
    """Handles time-series regression analysis of surprise impacts."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 252  # One trading year
    ):
        self.data = data
        self.window_size = window_size
        
    def run_rolling_regression(
        self,
        y_col: str,
        x_cols: List[str],
        control_vars: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run rolling window regression with controls.
        
        Args:
            y_col: Dependent variable column
            x_cols: List of independent variable columns
            control_vars: Optional control variables
            
        Returns:
            pandas.DataFrame: Rolling regression results
        """
        try:
            # Prepare variables
            y = self.data[y_col]
            X = self.data[x_cols]
            
            if control_vars:
                X = pd.concat([X, self.data[control_vars]], axis=1)
            
            # Add constant
            X = sm.add_constant(X)
            
            # Run rolling regression
            rols = RollingOLS(
                y,
                X,
                window=self.window_size,
                min_nobs=self.window_size//2
            )
            
            rolling_results = rols.fit()
            
            return pd.DataFrame({
                'params': rolling_results.params,
                't_stats': rolling_results.tvalues,
                'r2': rolling_results.rsquared
            })
            
        except Exception as e:
            logger.error(f"Error in rolling regression: {str(e)}")
            raise

class YieldCurveAnalysis:
    """Analyzes yield curve dynamics and regimes."""
    
    def __init__(self, yields_df: pd.DataFrame):
        self.yields_df = yields_df
        self.curve_metrics = None
        
    def calculate_curve_metrics(self) -> pd.DataFrame:
        """
        Calculate yield curve metrics including steepness and curvature.
        
        Returns:
            pandas.DataFrame: Yield curve metrics
        """
        try:
            metrics = {}
            
            # Steepness measures
            metrics['2s10s_spread'] = (
                self.yields_df['T_NOTE_10Y'] - self.yields_df['T_NOTE_2Y']
            )
            metrics['3m10y_spread'] = (
                self.yields_df['T_NOTE_10Y'] - self.yields_df['T_BILL_3M']
            )
            
            # Curvature (butterfly)
            metrics['belly_richness'] = (
                2 * self.yields_df['T_NOTE_5Y'] -
                self.yields_df['T_NOTE_2Y'] -
                self.yields_df['T_NOTE_10Y']
            )
            
            self.curve_metrics = pd.DataFrame(metrics)
            return self.curve_metrics
            
        except Exception as e:
            logger.error(f"Error calculating curve metrics: {str(e)}")
            raise
            
    def identify_curve_regime(
        self,
        lookback_window: int = 20
    ) -> pd.DataFrame:
        """
        Identify yield curve regime based on level and slope changes.
        
        Args:
            lookback_window: Window for calculating changes
            
        Returns:
            pandas.DataFrame: Curve regime classifications
        """
        if self.curve_metrics is None:
            self.calculate_curve_metrics()
            
        try:
            # Calculate changes
            level_chg = self.yields_df['T_NOTE_10Y'].diff(lookback_window)
            slope_chg = self.curve_metrics['2s10s_spread'].diff(lookback_window)
            
            # Classify regimes
            regimes = pd.DataFrame(index=self.yields_df.index)
            
            regimes['regime'] = np.where(
                (level_chg > 0) & (slope_chg > 0),
                'bull_steepener',
                np.where(
                    (level_chg < 0) & (slope_chg < 0),
                    'bear_flattener',
                    np.where(
                        (level_chg > 0) & (slope_chg < 0),
                        'bear_steepener',
                        'bull_flattener'
                    )
                )
            )
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error identifying curve regime: {str(e)}")
            raise

class EventStudyAnalysis:
    """Analyzes yield behavior around economic events."""
    
    def __init__(
        self,
        yields_df: pd.DataFrame,
        events_df: pd.DataFrame,
        window_size: int = 5
    ):
        self.yields_df = yields_df
        self.events_df = events_df
        self.window_size = window_size
        
    def calculate_abnormal_returns(
        self,
        benchmark_col: str = 'T_NOTE_10Y'
    ) -> pd.DataFrame:
        """
        Calculate abnormal returns around events.
        
        Args:
            benchmark_col: Benchmark yield for normal return calculation
            
        Returns:
            pandas.DataFrame: Abnormal returns around events
        """
        try:
            results = []
            
            for event_date in self.events_df.index:
                window_start = event_date - timedelta(days=self.window_size)
                window_end = event_date + timedelta(days=self.window_size)
                
                # Get window data
                window_data = self.yields_df.loc[window_start:window_end].copy()
                
                # Calculate expected returns (simple moving average)
                expected = window_data[benchmark_col].rolling(
                    window=self.window_size
                ).mean()
                
                # Calculate abnormal returns
                abnormal = window_data[benchmark_col] - expected
                
                results.append({
                    'event_date': event_date,
                    'abnormal_returns': abnormal,
                    'cumulative_ar': abnormal.cumsum()
                })
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error calculating abnormal returns: {str(e)}")
            raise

def calculate_impact_score(
    surprise_data: pd.DataFrame,
    yield_changes: pd.DataFrame,
    lookback_window: int = 252
) -> pd.DataFrame:
    """
    Calculate historical impact scores for economic surprises.
    
    Args:
        surprise_data: Economic surprise data
        yield_changes: Yield curve changes
        lookback_window: Window for historical comparison
        
    Returns:
        pandas.DataFrame: Impact scores
    """
    try:
        impact_scores = pd.DataFrame(index=surprise_data.index)
        
        for col in surprise_data.columns:
            if not col.endswith('_surprise'):
                continue
                
            # Calculate rolling standard deviation of surprises
            rolling_std = surprise_data[col].rolling(
                window=lookback_window
            ).std()
            
            # Calculate standardized impact
            for yield_col in yield_changes.columns:
                if not yield_col.endswith(('MA5', 'MA20', 'STD20')):
                    impact = (
                        surprise_data[col] *
                        yield_changes[yield_col] /
                        rolling_std
                    )
                    
                    impact_scores[f"{col}_{yield_col}_impact"] = impact
                    
        return impact_scores
        
    except Exception as e:
        logger.error(f"Error calculating impact scores: {str(e)}")
        raise 