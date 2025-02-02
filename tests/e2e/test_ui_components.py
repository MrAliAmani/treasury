import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import os
import time
from datetime import datetime, timedelta
import subprocess
import signal
import requests
from typing import Generator
import platform
import psutil

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Constants
STREAMLIT_PORT = 8501
WAIT_TIMEOUT = 20  # Increased timeout for slower systems
MAX_RETRIES = 30
RETRY_INTERVAL = 1

def kill_process_on_port(port):
    """Kill any process running on the specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Use net_connections() instead of deprecated connections()
            connections = proc.net_connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                    proc.kill()
                    time.sleep(1)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def get_streamlit_process():
    """Get Streamlit process info using process name instead of connections"""
    for proc in psutil.process_iter(['pid', 'name']):
        if 'streamlit' in proc.info['name'].lower():
            return proc
    return None

@pytest.fixture(scope="session")
def streamlit_server() -> Generator:
    """Start Streamlit server for testing"""
    # Kill any existing process on the port
    kill_process_on_port(STREAMLIT_PORT)
    
    # Determine platform-specific settings
    is_windows = platform.system() == 'Windows'
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if is_windows else 0
    
    # Start Streamlit server with improved error handling
    try:
        process = subprocess.Popen(
            ["streamlit", "run", "frontend/app.py", 
             "--server.port", str(STREAMLIT_PORT),
             "--server.headless", "true",
             "--server.address", "localhost"],
            creationflags=creation_flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Use text mode for output
        )
        
        # Wait for server to start
        start_time = time.time()
        server_ready = False
        
        while time.time() - start_time < MAX_RETRIES * RETRY_INTERVAL:
            try:
                response = requests.get(f"http://localhost:{STREAMLIT_PORT}/_stcore/health")
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                # Check if process has failed
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    raise Exception(
                        f"Streamlit server failed to start.\nStdout: {stdout}\nStderr: {stderr}"
                    )
                time.sleep(RETRY_INTERVAL)
        
        if not server_ready:
            raise Exception("Streamlit server failed to respond in time")
        
        yield process
        
        # Cleanup
        if is_windows:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            
    except Exception as e:
        # Ensure process is cleaned up on error
        if 'process' in locals():
            process.kill()
            process.wait()
        raise e

@pytest.fixture(scope="function")
def driver(streamlit_server):
    """Setup WebDriver for testing"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1920, 1080)
    driver.implicitly_wait(WAIT_TIMEOUT)
    
    # Connect to app with retry logic
    connected = False
    for _ in range(MAX_RETRIES):
        try:
            driver.get(f"http://localhost:{STREAMLIT_PORT}")
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            connected = True
            break
        except Exception:
            time.sleep(RETRY_INTERVAL)
    
    if not connected:
        driver.quit()
        raise Exception("Failed to connect to Streamlit app")
    
    # Additional wait for app to load
    time.sleep(2)
    yield driver
    driver.quit()

@pytest.fixture(scope="function")
def mobile_driver(streamlit_server):
    """Setup mobile-emulating Chrome WebDriver"""
    mobile_emulation = {
        "deviceMetrics": { "width": 375, "height": 812, "pixelRatio": 3.0 },
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    }
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
    chrome_options.add_argument('--headless')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f"http://localhost:{STREAMLIT_PORT}")
    
    # Wait for app to load
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    yield driver
    driver.quit()

class TestDatePicker:
    """Test date picker functionality"""
    
    @pytest.mark.e2e
    def test_date_validation(self, driver):
        """Test date input validation"""
        time.sleep(10)  # Increased wait time for app load
        
        try:
            # Wait for date input with more general selector
            date_container = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".streamlit-dateinput, [data-baseweb='datepicker']"
                ))
            )
            
            # Find input with more general selector
            date_input = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.element_to_be_clickable((
                    By.CSS_SELECTOR,
                    "input[type='text']"
                ))
            )
            assert date_input is not None, "Date input not found"
            
            # Set future date using JavaScript
            future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            driver.execute_script("""
                arguments[0].focus();
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                arguments[0].dispatchEvent(new Event('blur', { bubbles: true }));
            """, date_input, future_date)
            
            time.sleep(2)  # Wait for validation to trigger
            
            # Wait for warning message with more general selector
            warning = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".streamlit-alert, [role='alert']"
                ))
            )
            assert warning.is_displayed(), "Warning message not displayed"
            warning_text = warning.text.lower()
            assert "future" in warning_text, f"Unexpected warning text: {warning_text}"
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"test_date_validation_error_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            raise e

    @pytest.mark.e2e
    def test_date_range_persistence(self, driver):
        """Test date range persistence"""
        # Wait for date input container
        date_container = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stDateInput']"))
        )
        
        # Wait for date picker to be fully loaded
        time.sleep(2)
        
        # Find all input elements within the date container
        date_inputs = driver.execute_script("""
            return Array.from(arguments[0].querySelectorAll('input[type="text"]'));
        """, date_container)
        
        assert len(date_inputs) > 0, "No date inputs found"
        
        # Set dates
        start_date = "2025-01-02"
        end_date = "2025-02-01"
        
        if len(date_inputs) >= 2:
            driver.execute_script("""
                arguments[0].value = arguments[2];
                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                arguments[1].value = arguments[3];
                arguments[1].dispatchEvent(new Event('change', { bubbles: true }));
            """, date_inputs[0], date_inputs[1], start_date, end_date)
        else:
            # Handle single input case
            driver.execute_script("""
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
            """, date_inputs[0], f"{start_date} – {end_date}")

class TestMultiSelect:
    """Test multi-select functionality"""
    
    @pytest.mark.e2e
    def test_indicator_selection(self, driver):
        """Test indicator multi-select functionality"""
        time.sleep(10)  # Increased wait time for app load
        
        try:
            # Find multiselect with more general selector
            multiselect = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".streamlit-multiselect, [data-baseweb='select']"
                ))
            )
            
            # Ensure element is in view
            driver.execute_script("arguments[0].scrollIntoView(true);", multiselect)
            time.sleep(1)
            
            # Click using JavaScript
            driver.execute_script("arguments[0].click();", multiselect)
            time.sleep(2)
            
            # Look for options with more general selector
            options = WebDriverWait(driver, WAIT_TIMEOUT).until(
                lambda d: d.find_elements(
                    By.CSS_SELECTOR,
                    "[role='option'], .streamlit-selectbox li"
                )
            )
            
            assert len(options) > 0, "No options found in multiselect after waiting"
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            driver.save_screenshot(f"test_multiselect_error_{timestamp}.png")
            raise e

    @pytest.mark.e2e
    def test_mobile_layout(self, driver):
        """Test mobile layout adjustments"""
        time.sleep(10)  # Increased wait time for app load
        
        try:
            # Set mobile viewport
            driver.set_window_size(375, 812)
            time.sleep(2)
            
            # Look for menu button with more general selector
            menu_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    ".streamlit-menu, [aria-label='Menu']"
                ))
            )
            
            # Ensure menu button is visible
            assert menu_button.is_displayed(), "Menu button not visible"
            
            # Click menu using JavaScript
            driver.execute_script("arguments[0].click();", menu_button)
            time.sleep(2)
            
            # Check sidebar with more general selector
            sidebar = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.visibility_of_element_located((
                    By.CSS_SELECTOR,
                    ".streamlit-sidebar, [data-testid='stSidebarNav']"
                ))
            )
            assert sidebar.is_displayed(), "Sidebar not displayed after clicking menu"
            
        except Exception as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            driver.save_screenshot(f"test_mobile_layout_error_{timestamp}.png")
            raise e

class TestChartRendering:
    """Test chart rendering functionality"""
    
    @pytest.mark.e2e
    def test_plotly_chart_render(self, driver):
        """Test if Plotly charts render correctly"""
        # Wait for Plotly charts to load
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CLASS_NAME, "js-plotly-plot"))
        )
        
        # Verify chart elements
        charts = driver.find_elements(By.CLASS_NAME, "js-plotly-plot")
        assert len(charts) > 0
        
        # Test chart interactivity
        chart = charts[0]
        hover_element = chart.find_element(By.CLASS_NAME, "plotly")
        
        # Simulate hover interaction
        webdriver.ActionChains(driver).move_to_element(hover_element).perform()
        
        # Verify tooltip appears
        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CLASS_NAME, "hoverlayer"))
            )
            tooltip_visible = True
        except TimeoutException:
            tooltip_visible = False
            
        assert tooltip_visible

@pytest.mark.e2e
def test_error_boundary(driver):
    """Test error boundary handling"""
    time.sleep(10)  # Increased wait time for app load
    
    try:
        # Find refresh button with more general selector
        refresh_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(., 'Refresh')]"
            ))
        )
        
        # Click refresh button
        driver.execute_script("arguments[0].click();", refresh_button)
        time.sleep(2)
        
        # Force an error in the app
        driver.execute_script("""
            window.streamlit.setComponentValue('error_test', new Error('Test Error'));
        """)
        time.sleep(2)
        
        # Look for error message with more general selector
        error_element = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                ".streamlit-alert, [role='alert']"
            ))
        )
        
        assert error_element.is_displayed(), "Error message not displayed"
        error_text = error_element.text.lower()
        assert "error" in error_text, f"Unexpected error text: {error_text}"
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        driver.save_screenshot(f"test_error_boundary_error_{timestamp}.png")
        raise e

# Setup and teardown
@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for tests"""
    # Setup
    yield
    # Teardown - clean up any screenshots
    for file in os.listdir():
        if file.endswith("_error.png"):
            try:
                os.remove(file)
            except:
                pass 