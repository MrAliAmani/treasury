import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
from fpdf import FPDF
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import from backend
from backend.data_fetcher import CachedDataFetcher
from typing import Tuple

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.visualizations import FinancialVisualizer

# ----------------------------
# Helper Functions
# ----------------------------
def load_sample_data(start_date, end_date, indicators):
    """Generate sample data for demonstration"""
    dates = pd.date_range(start=start_date, end=end_date)
    data = {
        'date': dates,
        'indicator': np.random.choice(indicators, size=len(dates)),
        'surprise': np.random.normal(0, 1, size=len(dates)),
        'yield_change': np.random.normal(0, 0.5, size=len(dates))
    }
    return pd.DataFrame(data)

# Configure page
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize visualizer
visualizer = FinancialVisualizer()

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

# Initialize data fetcher
data_fetcher = CachedDataFetcher()

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

# Date range selector with proper data-testid
today = datetime.date.today()
default_start = today - datetime.timedelta(days=30)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start, today),
    help="Choose the start and end dates for your analysis",
    key="date_range_input"  # Add a key for testing
)

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
    # Initialize state
    st.session_state["show_error"] = False
    st.session_state["error_message"] = ""
    st.session_state["data_refreshed"] = False
    
    try:
        with st.spinner("Refreshing data..."):
            df = load_sample_data(date_range[0], date_range[1], selected_indicators)
            st.session_state["data_refreshed"] = True
            st.success("Data refreshed successfully!")
            
    except Exception as e:
        # Set error state
        st.session_state["show_error"] = True
        st.session_state["error_message"] = str(e)
        st.session_state["data_refreshed"] = False
        st.error(f"Error refreshing data: {str(e)}")
        df = pd.DataFrame()  # Empty DataFrame instead of None

# Update date validation with proper alert
if date_range[0] > datetime.date.today():
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
    """Export DataFrame to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def export_to_pdf(figs, df):
    """Export figures and data to PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Financial Analysis Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 10, f'Summary Statistics', ln=True)
    stats = df.describe().to_string()
    pdf.multi_cell(0, 5, stats)
    return pdf.output(dest='S').encode('latin-1')

# ----------------------------
# Load Data
# ----------------------------
df = load_sample_data(date_range[0], date_range[1], selected_indicators)

# ----------------------------
# Main Layout
# ----------------------------
st.title("Financial Analysis Dashboard")

# Create tabs
tab_overview, tab_analysis, tab_export = st.tabs([
    "Overview", "Detailed Analysis", "Export"
])

# ----------------------------
# Overview Tab
# ----------------------------
with tab_overview:
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Surprise",
            value=f"{df['surprise'].mean():.2f}",
            help="Mean surprise value across all indicators"
        )
    
    with col2:
        st.metric(
            label="Average Yield Change",
            value=f"{df['yield_change'].mean():.2f}%",
            help="Mean yield change across all maturities"
        )
    
    with col3:
        st.metric(
            label="Data Points",
            value=len(df),
            help="Total number of observations"
        )
    
    # Surprise scatter plot
    st.subheader("Surprise vs Yield Change")
    scatter_fig = visualizer.create_surprise_scatter(df)
    st.plotly_chart(scatter_fig, use_container_width=True)

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
    
    # Sample yield curve data for demonstration
    maturities = [1, 2, 5, 10, 30]
    pre_curves = [np.random.normal(2, 0.5, len(maturities)) for _ in range(10)]
    post_curves = [np.random.normal(2.5, 0.5, len(maturities)) for _ in range(10)]
    
    # Yield curve animation
    st.subheader("Yield Curve Evolution")
    curve_fig = visualizer.create_yield_curve_animation(
        maturities, pre_curves, post_curves
    )
    st.plotly_chart(curve_fig, use_container_width=True)
    
    # Impact heatmap
    st.subheader("Impact Analysis")
    impact_matrix = np.random.normal(0, 1, (len(selected_indicators), len(maturities)))
    heatmap_fig = visualizer.create_impact_heatmap(
        impact_matrix,
        selected_indicators,
        [f"{m}Y" for m in maturities]
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

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
        # Single button for PDF export
        if st.button("📥 Download PDF Report", key="pdf_btn"):
            figs = {
                'scatter': scatter_fig,
                'curve': curve_fig,
                'heatmap': heatmap_fig
            }
            pdf_data = export_to_pdf(figs, df)
            st.session_state["pdf_data"] = pdf_data
            st.download_button(
                label="Download PDF File",
                data=pdf_data,
                file_name="financial_analysis.pdf",
                mime="application/pdf",
                key="pdf_download"
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

# Store figures in session state
st.session_state.scatter_fig = scatter_fig
st.session_state.curve_fig = curve_fig
st.session_state.heatmap_fig = heatmap_fig 