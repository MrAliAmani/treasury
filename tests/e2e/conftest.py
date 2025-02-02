import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import subprocess
import signal
import requests
import platform
import psutil
from typing import Generator

# Constants
STREAMLIT_PORT = 8501
WAIT_TIMEOUT = 20
MAX_RETRIES = 30
RETRY_INTERVAL = 1

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
    
    # Determine platform-specific settings
    is_windows = platform.system() == 'Windows'
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if is_windows else 0
    
    try:
        process = subprocess.Popen(
            ["streamlit", "run", "frontend/app.py", 
             "--server.port", str(STREAMLIT_PORT),
             "--server.headless", "true",
             "--server.address", "localhost"],
            creationflags=creation_flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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