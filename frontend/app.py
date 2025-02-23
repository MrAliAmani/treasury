import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path BEFORE imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now import from backend
from backend.data_loader import fetch_treasury_yields

# Must be the first Streamlit command
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

import time
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import logging
import os
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO
import io
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# Use absolute imports
from backend.data_fetcher import DataFetcher
from backend.market_analyzer import MarketDataAnalyzer
from backend.visualizations import FinancialVisualizer

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Initialize data fetcher with proper configuration
@st.cache_resource
def get_data_fetcher():
    """Create or get cached data fetcher instance"""
    db_path = str(project_root / 'market_data.db')
    return DataFetcher(db_path=db_path)

# Initialize data fetcher
data_fetcher = get_data_fetcher()

def load_market_data(start_date, end_date, indicators):
    """Load market data with proper error handling"""
    try:
        df = data_fetcher.fetch_market_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            indicators=indicators
        )
        
        if df is None or df.empty:
            st.error("No data returned from data fetcher")
            return pd.DataFrame()
            
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate additional metrics
        df['yield_change'] = df.groupby('indicator')['value'].diff()
        
        # If expected values aren't present, use shifted values as proxy
        if 'expected' not in df.columns:
            df['expected'] = df.groupby('indicator')['value'].shift(1)
            
        # Calculate surprise
        df['surprise'] = df['value'] - df['expected']
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Initialize visualizer
visualizer = FinancialVisualizer()

# Initialize empty figures
global scatter_fig, curve_fig, heatmap_fig  # Declare all globals at once
scatter_fig = go.Figure()
curve_fig = go.Figure()
heatmap_fig = go.Figure()

# Initialize session state variables
if "data_refreshed" not in st.session_state:
    st.session_state["data_refreshed"] = False

if "show_error" not in st.session_state:
    st.session_state["show_error"] = False
    st.session_state["error_message"] = ""

if "show_date_warning" not in st.session_state:
    st.session_state["show_date_warning"] = False

if "excel_data" not in st.session_state:
    st.session_state["excel_data"] = None

if "pdf_data" not in st.session_state:
    st.session_state["pdf_data"] = None

# Custom CSS for tooltips
st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip:hover::before {
        content: attr(data-tooltip);
        position: absolute;
        background: #f0f2f6;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 14px;
        white-space: nowrap;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize market analyzer
market_analyzer = MarketDataAnalyzer(data_fetcher)

def process_uploaded_excel(uploaded_file, data_type: str) -> Tuple[pd.DataFrame, str]:
    """Process uploaded Excel file"""
    try:
        if data_type == 'yields':
            df = data_fetcher.load_from_excel(
                uploaded_file,
                data_type='yields',
                sheet_name='Yields'
            )
            message = "✅ Yield curve data loaded successfully"
        else:
            df = data_fetcher.load_from_excel(
                uploaded_file,
                data_type='indicators',
                sheet_name='Indicators'
            )
            message = "✅ Economic indicators loaded successfully"
        return df, message
    except Exception as e:
        return pd.DataFrame(), f"❌ Error loading data: {str(e)}"

# ----------------------------
# Sidebar: Filters and Controls
# ----------------------------
st.sidebar.header("Filters and Controls")

def get_default_dates() -> Tuple[date, date]:
    """Get default date range (past year until today)"""
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=get_default_dates(),  # Use default dates
    min_value=date(1980, 1, 1),
    max_value=date.today(),  # Limit max date to today
    help="Choose the date range for analysis"
)

# Convert to tuple if single date is selected
if isinstance(date_range, date):
    date_range = (date_range, date_range)

# Indicator multi-select with proper data-testid
indicators = ["Inflation", "Growth", "Unemployment"]
selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    options=indicators,
    default=["Inflation", "Growth"],
    help="Choose which economic indicators to analyze",
    key="indicator_select"  # Add a key for testing
)

# Real-time data refresh button
if st.sidebar.button("🔄 Refresh Data", help="Click to fetch latest data", key="refresh_button"):
    try:
        with st.spinner("Fetching market data..."):
            st.session_state["show_error"] = False
            st.session_state["error_message"] = ""
            st.session_state["data_refreshed"] = False
            
            # Force refresh by clearing cache for this date range
            cache_key = f"market_data_{date_range[0]}_{date_range[1]}_{'-'.join(sorted(selected_indicators))}"
            data_fetcher.clear_cache(cache_key)
            
            # Fetch fresh data
            df = load_market_data(date_range[0], date_range[1], selected_indicators)
            st.session_state["data_refreshed"] = True
            st.success("Market data refreshed successfully!")
            
    except Exception as e:
        st.session_state["show_error"] = True
        st.session_state["error_message"] = str(e)
        st.session_state["data_refreshed"] = False
        # st.error(f"Error fetching market data: {str(e)}")
        df = pd.DataFrame()

# Update date validation with proper alert
if date_range[0] > date.today():
    st.session_state["show_date_warning"] = True
    st.warning("Selected dates cannot be in the future", icon="⚠️")

# Add to the sidebar
with st.sidebar:
    st.header("Data Input")
    
    # File upload section
    st.subheader("Upload Data")
    
    # Yield curve data upload
    yield_file = st.file_uploader(
        "Upload Yield Curve Data (Excel)",
        type=['xlsx', 'xls'],
        help="Excel file with yield curve data",
        key="yield_upload"
    )
    
    # Economic indicators upload
    indicators_file = st.file_uploader(
        "Upload Economic Indicators (Excel)",
        type=['xlsx', 'xls'],
        help="Excel file with economic indicators",
        key="indicators_upload"
    )
    
    # Process uploaded files
    if yield_file:
        yield_data, yield_message = process_uploaded_excel(yield_file, 'yields')
        if not yield_data.empty:
            st.session_state['yield_data'] = yield_data
        st.info(yield_message)
        
    if indicators_file:
        indicator_data, indicator_message = process_uploaded_excel(indicators_file, 'indicators')
        if not indicator_data.empty:
            st.session_state['indicator_data'] = indicator_data
        st.info(indicator_message)

def export_to_excel(df):
    """Export DataFrame to Excel with proper formatting"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Convert DataFrame to Excel
        df.to_excel(writer, index=False, sheet_name='Analysis')
        
        # Get the worksheet to adjust column widths
        worksheet = writer.sheets['Analysis']
        
        # Auto-adjust columns width based on content
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),  # Length of longest value
                len(str(col))  # Length of column name
            ) + 2  # Add some padding
            
            # Convert column index to letter (A, B, C, etc)
            col_letter = chr(65 + idx)  # A=65 in ASCII
            worksheet.column_dimensions[col_letter].width = max_length
            
            # Special formatting for date column
            if col == 'date':
                for cell in worksheet[f'{col_letter}2:{col_letter}{len(df)+1}']:
                    cell[0].number_format = 'YYYY-MM-DD'
    
    return output.getvalue()

def create_impact_figure():
    """Create impact heatmap using real market data"""
    try:
        if df.empty:
            return go.Figure()
            
        # Create pivot table for heatmap
        pivot_df = df.pivot_table(
            values='value',  # Use 'value' instead of 'surprise'
            index='indicator',
            columns='date',
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap figure
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Date", y="Indicator", color="Value"),
            title="Economic Impact Analysis"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Indicator",
            coloraxis_colorbar_title="Value"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating impact figure: {str(e)}")
        return go.Figure()

def export_to_pdf(figs, df):
    """Export figures and data to PDF"""
    try:
        # For testing purposes
        if hasattr(export_to_pdf, '_mock_data'):
            return export_to_pdf._mock_data
            
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Financial Analysis Report', ln=True, align='C')
        pdf.ln(10)
        
        # Add summary statistics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Summary Statistics', ln=True)
        pdf.set_font('Courier', size=10)
        stats = df.describe().round(2).to_string()
        pdf.multi_cell(0, 5, stats)
        pdf.ln(10)
        
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        raise

def create_pdf_download_button(figs, df, key_prefix=""):
    """Create a consistent PDF download button"""
    if st.button(f"📥 Download PDF Report", key=f"{key_prefix}_pdf_btn"):
        try:
            pdf_data = export_to_pdf(figs, df)
            st.session_state["pdf_data"] = pdf_data
            st.download_button(
                label="Download PDF File",
                data=pdf_data,
                file_name="financial_analysis.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_pdf_download"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

# ----------------------------
# Main Layout
# ----------------------------
st.title("Financial Analysis Dashboard")

# Create tabs
tab_overview, tab_analysis, tab_export, tab_yield_analysis, tab_missing_data = st.tabs([
    "Overview", "Detailed Analysis", "Export", "Yield Analysis", "Missing Data"
])

# Load Data
df = load_market_data(date_range[0], date_range[1], selected_indicators)

# Add auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Check if it's time to refresh (every 5 minutes)
if time.time() - st.session_state.last_refresh > 300:  # 5 minutes
    try:
        with st.spinner("Refreshing data..."):
            df = load_market_data(date_range[0], date_range[1], selected_indicators)
            st.session_state.last_refresh = time.time()
    except Exception as e:
        st.error(f"Auto-refresh failed: {str(e)}")

# ----------------------------
# Overview Tab
# ----------------------------
with tab_overview:
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Value",
            value=f"{df['value'].mean():.2f}" if 'value' in df.columns else "0.00",
            help="Mean actual value across all indicators"
        )
    
    with col2:
        st.metric(
            label="Average Expected",
            value=f"{df['expected'].mean():.2f}" if 'expected' in df.columns else "0.00",
            help="Mean expected value across all indicators"
        )
    
    with col3:
        st.metric(
            label="Average Surprise",
            value=f"{df['surprise'].mean():.2f}" if 'surprise' in df.columns else "0.00",
            help="Mean surprise value (actual - expected)"
        )
    
    # Surprise scatter plot
    st.subheader("Surprise vs Yield Change")
    scatter_fig = visualizer.create_surprise_scatter(df)
    st.plotly_chart(scatter_fig, use_container_width=True, key="overview_scatter")

    # Display uploaded data
    if 'yield_data' in st.session_state or 'indicator_data' in st.session_state:
        st.subheader("Uploaded Data Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'yield_data' in st.session_state:
                st.write("Yield Curve Data")
                st.dataframe(
                    st.session_state['yield_data'].head(),
                    use_container_width=True
                )
                
        with col2:
            if 'indicator_data' in st.session_state:
                st.write("Economic Indicators")
                st.dataframe(
                    st.session_state['indicator_data'].head(),
                    use_container_width=True
                )

# ----------------------------
# Analysis Tab
# ----------------------------
with tab_analysis:
    st.header("Detailed Analysis")
    
    if not df.empty:
        # Add date selector for detailed analysis
        available_dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        selected_date = st.selectbox(
            "Select Date for Detailed Analysis",
            options=available_dates,
            index=0
        )
        
        # Show detailed view for selected date
        detail_fig = visualizer.create_surprise_detail_view(df, selected_date)
        if detail_fig:
            st.plotly_chart(detail_fig, use_container_width=True)
        
        # Show intraday analysis
        st.subheader("Surprise Impact Analysis")
        intraday_fig = visualizer.create_intraday_surprise_analysis(df)
        if intraday_fig:
            st.plotly_chart(intraday_fig, use_container_width=True)
        
        # Time Series Analysis
        st.subheader("Time Series Analysis")
        time_series_fig = visualizer.create_time_series(df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
        
        # Surprise vs Yield Change Analysis
        st.subheader("Surprise vs Yield Change Analysis")
        scatter_fig = visualizer.create_surprise_scatter(df)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Impact Analysis
        st.subheader("Economic Impact Analysis")
        heatmap_fig = visualizer.create_impact_heatmap(df)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        summary_stats = df.groupby('indicator').agg({
            'value': ['mean', 'std', 'min', 'max'],
            'yield_change': ['mean', 'std']
        }).round(3)
        st.dataframe(summary_stats)
    else:
        st.warning("No data available for analysis. Please load or refresh the data.")

# ----------------------------
# Export Tab
# ----------------------------
with tab_export:
    st.header("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Single button for Excel export
        if st.button("📥 Download Excel", key="excel_btn"):
            excel_data = export_to_excel(df)
            st.session_state["excel_data"] = excel_data
            st.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name="financial_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
    
    with col2:
        create_pdf_download_button(
            figs={
                'scatter': scatter_fig,
                'curve': curve_fig,
                'heatmap': heatmap_fig
            },
            df=df,
            key_prefix="export"
        )
    
    # Preview data table
    st.subheader("Data Preview")
    st.dataframe(df)

# Add footer
st.markdown("---")
st.markdown(
    """
    """,
    unsafe_allow_html=True
)

# At the bottom of the file, store figures in session state
st.session_state.scatter_fig = scatter_fig
st.session_state.curve_fig = curve_fig
st.session_state.heatmap_fig = heatmap_fig 

# Add after the existing tab implementations
with tab_yield_analysis:
    st.header("Yield Curves Analysis")
    
    if not df.empty:
        # Get yield data
        try:
            yield_data = fetch_treasury_yields(
                start_date=date_range[0].strftime('%Y-%m-%d'),
                end_date=date_range[1].strftime('%Y-%m-%d'),
                maturities=['DGS2', 'DGS5', 'DGS10']
            )
            
            # Create yield curve plot
            yield_fig = visualizer.create_yield_curve_indicator_plot(yield_data, df)
            if yield_fig:
                st.plotly_chart(yield_fig, use_container_width=True)
            else:
                st.warning("No yield curve data available for the selected period.")
                
        except Exception as e:
            st.error(f"Error loading yield curve data: {str(e)}")
    else:
        st.warning("No indicator data available. Please load data first.")

with tab_missing_data:
    st.header("Missing Data Analysis")
    
    if not df.empty:
        # Create missing data table
        missing_data_fig = visualizer.create_missing_data_table(df)
        if missing_data_fig:
            st.plotly_chart(missing_data_fig, use_container_width=True)
        else:
            st.success("No significant data gaps found in the selected period.")
            
        # Add download button for missing data report
        if st.button("📥 Download Missing Data Report"):
            # Convert missing periods to DataFrame for download
            missing_df = pd.DataFrame(missing_data_fig.data[0].cells.values).T
            missing_df.columns = missing_data_fig.data[0].header.values
            
            # Create Excel buffer
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                missing_df.to_excel(writer, sheet_name='Missing Data', index=False)
            
            # Offer download
            st.download_button(
                label="Download Excel File",
                data=buffer.getvalue(),
                file_name="missing_data_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No data available for missing data analysis.")