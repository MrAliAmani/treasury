# Add to existing cache_config.py
CACHE_CONFIG = {
    # Database configuration
    'DB_PATH': "cache/dashboard.db",
    
    # Cache duration in hours for different data types
    'CACHE_DURATION': {
        "dashboard_metrics": 24,
        "user_data": 12,
        "system_stats": 1
    },
    
    # Batch processing configuration
    'BATCH_SIZE': 100,
    'MAX_WORKERS': 4,
    
    # Rate limiting
    'RATE_LIMIT_DELAY': 0.1  # seconds between batch requests
} 