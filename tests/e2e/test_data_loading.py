import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import subprocess
import signal
import requests
import platform
import psutil
from pathlib import Path
from typing import Generator
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the components we want to test
from backend.data_fetcher import DataFetcher
from backend.fetch_fred import fetch_fred_data
from backend.fetch_openbb import fetch_data as fetch_openbb

# Constants
STREAMLIT_PORT = 8501
WAIT_TIMEOUT = 20
MAX_RETRIES = 30
RETRY_INTERVAL = 1

@pytest.fixture
def data_fetcher():
    """Create a DataFetcher instance for testing"""
    fetcher = DataFetcher(db_path=':memory:')  # Use in-memory SQLite for testing
    return fetcher

@pytest.mark.e2e
def test_fred_data_fetching(data_fetcher):
    """Test FRED data fetching functionality"""
    # Set up test parameters
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    indicators = ['Inflation', 'Growth']
    
    try:
        # Fetch data
        data = data_fetcher.fetch_market_data(start_date, end_date, indicators)
        
        # Verify the data
        assert not data.empty, "Data should not be empty"
        assert 'date' in data.columns, "Data should have a date column"
        assert 'indicator' in data.columns, "Data should have an indicator column"
        assert 'value' in data.columns, "Data should have a value column"
        assert set(data['indicator'].unique()) == set(indicators), "All requested indicators should be present"
        
    except Exception as e:
        pytest.fail(f"Data fetching failed: {str(e)}")

@pytest.mark.e2e
def test_data_caching(data_fetcher):
    """Test data caching functionality"""
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    indicators = ['Inflation']
    
    # First fetch - should get from source
    data1 = data_fetcher.fetch_market_data(start_date, end_date, indicators)
    
    # Second fetch - should get from cache
    data2 = data_fetcher.fetch_market_data(start_date, end_date, indicators)
    
    # Verify both datasets are identical
    pd.testing.assert_frame_equal(data1, data2)

@pytest.mark.e2e
def test_streamlit_integration(streamlit_server, driver):
    """Test data fetching integration with Streamlit UI"""
    try:
        # Wait for initial load
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".stApp"))
        )
        
        # Test export functionality
        test_export_functionality(driver)
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"test_streamlit_error_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Test failed. Screenshot saved to {screenshot_path}")
        print(f"Error: {str(e)}")
        raise

def test_export_functionality(driver):
    """Test the export functionality in the Streamlit app"""
    try:
        # Switch to Export tab
        tabs = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((
                By.CSS_SELECTOR,
                "[role='tab'], .streamlit-tabs button"
            ))
        )
        export_tab = next(tab for tab in tabs if "Export" in tab.text)
        driver.execute_script("arguments[0].click();", export_tab)
        
        # Test Excel export
        excel_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'Excel')]"
            ))
        )
        driver.execute_script("arguments[0].click();", excel_button)
        
        # Verify Excel download button
        download_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "a[download], a[href*='download']"
            ))
        )
        assert download_button.is_displayed()
        
        # Test PDF export
        pdf_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'PDF')]"
            ))
        )
        driver.execute_script("arguments[0].click();", pdf_button)
        
        # Verify PDF download button
        pdf_download = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "a[download], a[href*='download']"
            ))
        )
        assert pdf_download.is_displayed()
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"test_export_error_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Test failed. Screenshot saved to {screenshot_path}")
        print(f"Error: {str(e)}")
        raise

@pytest.mark.e2e
def test_error_handling(data_fetcher):
    """Test error handling in data fetching"""
    # Test with invalid dates
    with pytest.raises(Exception):
        data_fetcher.fetch_market_data(
            start_date="invalid_date",
            end_date="invalid_date",
            indicators=['Inflation']
        )
    
    # Test with invalid indicators
    with pytest.raises(Exception):
        data_fetcher.fetch_market_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
            indicators=['InvalidIndicator']
        )

def kill_process_on_port(port):
    """Kill any process running on the specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.net_connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                    proc.kill()
                    time.sleep(1)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

@pytest.fixture(scope="session")
def streamlit_server() -> Generator:
    """Start Streamlit server for testing"""
    # Kill any existing process on the port
    kill_process_on_port(STREAMLIT_PORT)
    
    # Start the Streamlit server
    process = None
    try:
        cmd = [
            "streamlit", "run", 
            str(project_root / "frontend" / "app.py"),
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--server.address", "localhost"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < MAX_RETRIES * RETRY_INTERVAL:
            try:
                response = requests.get(f"http://localhost:{STREAMLIT_PORT}/_stcore/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(RETRY_INTERVAL)
        
        yield process
        
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

@pytest.fixture(scope="function")
def driver(streamlit_server):
    """Setup WebDriver for testing"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(WAIT_TIMEOUT)
    
    try:
        driver.get(f"http://localhost:{STREAMLIT_PORT}")
        yield driver
    finally:
        driver.quit()

def print_element_info(driver, css_selector):
    """Debug helper to print information about elements"""
    print(f"\nLooking for elements matching: {css_selector}")
    elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
    print(f"Found {len(elements)} elements")
    for i, element in enumerate(elements):
        print(f"\nElement {i + 1}:")
        print(f"Text: {element.text}")
        print(f"Tag: {element.tag_name}")
        print(f"Attributes: {element.get_attribute('outerHTML')}")
        print(f"Visible: {element.is_displayed()}")

@pytest.mark.e2e
def test_data_refresh(driver):
    """Test data refresh functionality"""
    time.sleep(10)  # Wait for app to load
    
    try:
        # Find and click refresh button using a more general selector
        refresh_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'Refresh')]"
            ))
        )
        driver.execute_script("arguments[0].click();", refresh_button)
        
        # Verify metrics are displayed using a more general selector
        metrics = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((
                By.CSS_SELECTOR,
                "div[class*='stMetric']"
            ))
        )
        assert len(metrics) >= 3  # Average Surprise, Average Yield Change, Data Points
        
        # Verify plot is displayed using a more general selector
        plot = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div[class*='stPlot'], div[class*='plotly']"
            ))
        )
        assert plot.is_displayed()
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"test_data_refresh_error_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Test failed. Screenshot saved to {screenshot_path}")
        print(f"Error: {str(e)}")
        raise

@pytest.mark.e2e
def test_data_filtering(driver):
    """Test data filtering functionality"""
    time.sleep(10)  # Wait for app to load
    
    try:
        # Find sidebar using a more general selector
        sidebar = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "[data-testid='stSidebarNav'], .streamlit-sidebar"
            ))
        )
        
        # Set date range using a more general selector
        date_input = WebDriverWait(sidebar, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "input[type='text']"
            ))
        )
        
        # Set dates using JavaScript
        today = datetime.now()
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        driver.execute_script("""
            arguments[0].focus();
            arguments[0].value = arguments[1];
            arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
            arguments[0].dispatchEvent(new Event('blur', { bubbles: true }));
        """, date_input, f"{start_date} – {end_date}")
        
        # Select indicators using a more general selector
        multiselect = WebDriverWait(sidebar, 20).until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "[aria-label*='Select options'], .streamlit-multiselect"
            ))
        )
        driver.execute_script("arguments[0].click();", multiselect)
        
        # Wait for options and select Inflation and Growth
        for indicator in ["Inflation", "Growth"]:
            option = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//div[contains(@class, 'streamlit-selectbox')]//div[text()='{indicator}']"
                ))
            )
            driver.execute_script("arguments[0].click();", option)
        
        # Click outside to close dropdown
        driver.execute_script("arguments[0].click();", sidebar)
        
        # Verify filtered data is displayed
        plot = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div[class*='stPlot'], div[class*='plotly']"
            ))
        )
        assert plot.is_displayed()
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"test_data_filtering_error_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Test failed. Screenshot saved to {screenshot_path}")
        print(f"Error: {str(e)}")
        raise

@pytest.mark.e2e
def test_data_export(driver):
    """Test data export functionality"""
    time.sleep(10)  # Wait for app to load
    
    try:
        # Switch to Export tab using a more general selector
        tabs = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((
                By.CSS_SELECTOR,
                "[role='tab'], .streamlit-tabs button"
            ))
        )
        export_tab = next(tab for tab in tabs if "Export" in tab.text)
        driver.execute_script("arguments[0].click();", export_tab)
        
        # Wait for export tab content to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "[role='tabpanel'], .streamlit-tabcontent"
            ))
        )
        
        # Click Excel export button using a more general selector
        excel_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'Excel')]"
            ))
        )
        driver.execute_script("arguments[0].click();", excel_button)
        
        # Verify download button appears
        download_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "a[download], a[href*='download']"
            ))
        )
        assert download_button.is_displayed()
        
        # Click PDF export button
        pdf_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'PDF')]"
            ))
        )
        driver.execute_script("arguments[0].click();", pdf_button)
        
        # Verify PDF download button appears
        pdf_download = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "a[download], a[href*='download']"
            ))
        )
        assert pdf_download.is_displayed()
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"test_data_export_error_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Test failed. Screenshot saved to {screenshot_path}")
        print(f"Error: {str(e)}")
        raise 