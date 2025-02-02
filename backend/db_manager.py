import sqlite3
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "cache/dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = None
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize database and create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create cache table with timestamp for data freshness
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE,
                data_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                expiry_time DATETIME
            )
        """)
        self.conn.commit()
    
    def batch_insert(self, data: List[Dict[str, Any]], cache_key: str, expiry_hours: int = 24):
        """Batch insert data with pandas for better performance"""
        df = pd.DataFrame(data)
        
        # Convert to JSON string for storage
        data_json = df.to_json(orient='records')
        
        # Using parameterized query to prevent SQL injection
        self.conn.execute("""
            INSERT OR REPLACE INTO data_cache (cache_key, data_json, expiry_time)
            VALUES (?, ?, datetime('now', '+' || ? || ' hours'))
        """, (cache_key, data_json, expiry_hours))
        
        self.conn.commit()
    
    def get_cached_data(self, cache_key: str) -> pd.DataFrame:
        """Retrieve cached data with optimized query"""
        query = """
            SELECT data_json 
            FROM data_cache 
            WHERE cache_key = ? 
            AND datetime('now') < expiry_time
        """
        
        result = self.conn.execute(query, (cache_key,)).fetchone()
        
        if result:
            # Convert JSON string back to DataFrame
            return pd.read_json(result[0])
        return None
    
    def clear_expired_cache(self):
        """Remove expired cache entries"""
        self.conn.execute("""
            DELETE FROM data_cache 
            WHERE datetime('now') > expiry_time
        """)
        self.conn.commit()
    
    def close(self):
        if self.conn:
            self.conn.close() 