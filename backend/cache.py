import sqlite3
import json
import time
from typing import Any, Optional
import logging

class SQLiteCache:
    def __init__(self, db_path: str):
        """Initialize SQLite cache"""
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Create cache table if not exists"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expire_at INTEGER
                )
            """)
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT value, expire_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cur.fetchone()
                
                if row is None:
                    return None
                    
                value, expire_at = row
                
                # Check expiration
                if expire_at and time.time() > expire_at:
                    self.delete(key)
                    return None
                    
                return json.loads(value)
                
        except Exception as e:
            logging.error(f"Cache get failed: {str(e)}")
            return None
            
    def set(self, key: str, value: Any, expire_in: Optional[int] = None):
        """Set cache value with optional expiration"""
        try:
            expire_at = time.time() + expire_in if expire_in else None
            value_json = json.dumps(value)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire_at) VALUES (?, ?, ?)",
                    (key, value_json, expire_at)
                )
                
        except Exception as e:
            logging.error(f"Cache set failed: {str(e)}")
            
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except Exception as e:
            logging.error(f"Cache delete failed: {str(e)}")
            
    def clear(self):
        """Clear all cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
        except Exception as e:
            logging.error(f"Cache clear failed: {str(e)}") 