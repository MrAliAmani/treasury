import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest
import sys
import os
import tempfile
from unittest.mock import patch, Mock
import json
import time

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.visualizations import FinancialVisualizer
from frontend.app import export_to_excel, export_to_pdf, load_sample_data

@pytest.fixture
def visualizer():
    """Fixture to create FinancialVisualizer instance"""
    return FinancialVisualizer()

@pytest.fixture
def sample_df():
    """Fixture to create sample DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', '2024-01-10'),
        'indicator': ['Inflation', 'Growth'] * 5,
        'surprise': np.random.normal(0, 1, 10),
        'yield_change': np.random.normal(0, 0.5, 10)
    })

class TestVisualization:
    """Test visualization functionality"""
    
    @pytest.mark.unit
    def test_create_surprise_scatter(self, visualizer, sample_df):
        """Test surprise scatter plot creation"""
        fig = visualizer.create_surprise_scatter(sample_df)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have at least one trace
        assert fig.layout.title.text == "Economic Surprises vs Yield Changes"
        
    @pytest.mark.unit
    def test_create_yield_curve_animation(self, visualizer):
        """Test yield curve animation creation"""
        maturities = [1, 2, 5, 10, 30]
        pre_curves = [np.random.normal(2, 0.5, len(maturities)) for _ in range(3)]
        post_curves = [np.random.normal(2.5, 0.5, len(maturities)) for _ in range(3)]
        
        fig = visualizer.create_yield_curve_animation(maturities, pre_curves, post_curves)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Should have pre and post curves
        assert fig.layout.title.text == "Dynamic Yield Curve Comparison"
        
    @pytest.mark.unit
    def test_create_impact_heatmap(self, visualizer):
        """Test impact heatmap creation"""
        indicators = ["Inflation", "Growth"]
        maturities = ["2Y", "5Y", "10Y"]
        impact_matrix = np.random.normal(0, 1, (len(indicators), len(maturities)))
        
        fig = visualizer.create_impact_heatmap(impact_matrix, indicators, maturities)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Should have one heatmap trace
        assert fig.layout.title.text == "Heatmap of Surprise Impacts"

class TestApp:
    """Test Streamlit app functionality"""
    
    @pytest.mark.unit
    def test_load_sample_data(self):
        """Test sample data generation"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=10)
        indicators = ["Inflation", "Growth"]
        
        df = load_sample_data(start_date, end_date, indicators)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11  # 10 days + end date
        assert all(col in df.columns for col in ['date', 'indicator', 'surprise', 'yield_change'])
        assert all(ind in indicators for ind in df['indicator'].unique())
        
    @pytest.mark.unit
    def test_export_to_excel(self, sample_df):
        """Test Excel export functionality"""
        excel_data = export_to_excel(sample_df)
        
        assert isinstance(excel_data, bytes)
        assert len(excel_data) > 0
        
    @pytest.mark.unit
    def test_export_to_pdf(self, sample_df, mocker):
        """Test PDF export functionality"""
        # Mock date_range and selected_indicators since they're from st.sidebar
        mocker.patch('frontend.app.date_range', return_value=('2024-01-01', '2024-01-10'))
        mocker.patch('frontend.app.selected_indicators', return_value=['Inflation', 'Growth'])
        
        figs = {
            'scatter': go.Figure(),
            'curve': go.Figure(),
            'heatmap': go.Figure()
        }
        
        pdf_data = export_to_pdf(figs, sample_df)
        
        assert isinstance(pdf_data, bytes)
        assert len(pdf_data) > 0

@pytest.mark.integration
class TestStreamlitIntegration:
    """Test Streamlit app integration"""
    
    def test_app_loads(self):
        """Test that the Streamlit app loads without errors"""
        at = AppTest.from_file("frontend/app.py")
        at.run()
        
        # Check main components are present - using more flexible title matching
        assert any("Dashboard" in str(elem.value) 
                  for elem in at.title), "Dashboard title not found"
        assert at.sidebar, "Sidebar not found"
        
    def test_sidebar_components(self):
        """Test sidebar components are present and functional"""
        at = AppTest.from_file("frontend/app.py")
        at.run()
        
        # Check date selector
        assert at.sidebar.date_input
        
        # Check indicator selector
        assert at.sidebar.multiselect
        
        # Check refresh button by label
        assert any("🔄 Refresh Data" in str(button.label) 
                  for button in at.sidebar.button) 