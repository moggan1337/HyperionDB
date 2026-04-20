"""
Storage Module for HyperionDB
=============================

Provides storage engine and buffer pool management.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import os
import json
import threading
import hashlib


@dataclass
class Page:
    """
    Represents a database page.
    
    Pages are the basic unit of storage and I/O.
    """
    page_id: int
    table_name: str
    data: bytes
    is_dirty: bool = False
    pin_count: int = 0
    last_accessed: float = 0.0
    
    # Page metadata
    page_type: str = "data"  # data, index, overflow
    free_space: int = 0
    record_count: int = 0
    
    @property
    def size(self) -> int:
        return len(self.data)
    
    def pin(self):
        """Pin the page in memory."""
        self.pin_count += 1
    
    def unpin(self):
        """Unpin the page."""
        self.pin_count = max(0, self.pin_count - 1)
    
    def mark_dirty(self):
        """Mark page as modified."""
        self.is_dirty = True
    
    def to_bytes(self) -> bytes:
        """Serialize page to bytes."""
        header = {
            "page_id": self.page_id,
            "table_name": self.table_name,
            "page_type": self.page_type,
            "free_space": self.free_space,
            "record_count": self.record_count
        }
        header_bytes = json.dumps(header).encode()
        return header_bytes + b"|" + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Page':
        """Deserialize page from bytes."""
        header_end = data.index(b"|")
        header = json.loads(data[:header_end].decode())
        page_data = data[header_end + 1:]
        
        return cls(
            page_id=header["page_id"],
            table_name=header["table_name"],
            data=page_data,
            page_type=header.get("page_type", "data"),
            free_space=header.get("free_space", 0),
            record_count=header.get("record_count", 0)
        )


class BufferPool:
    """
    Buffer pool for caching database pages in memory.
    
    Implements LRU-K replacement policy with learned page importance.
    """
    
    def __init__(self, pool_size: int = 10000):
        """
        Initialize buffer pool.
        
        Args:
            pool_size: Maximum number of pages in pool
        """
        self.pool_size = pool_size
        self.pages: Dict[int, Page] = {}
        self.access_history: Dict[int, List[float]] = {}  # For LRU-K
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.write_count = 0
        self.read_count = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Page importance model (learned)
        self.page_importance: Dict[int, float] = {}
        
        # Eviction candidates cache
        self._eviction_candidates: Optional[List[int]] = None
    
    @property
    def size(self) -> int:
        return len(self.pages)
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_page(self, page_id: int) -> Optional[Page]:
        """Get a page from the buffer pool."""
        with self.lock:
            if page_id in self.pages:
                page = self.pages[page_id]
                page.last_accessed = self._current_time()
                self.hit_count += 1
                
                # Record access for LRU-K
                if page_id not in self.access_history:
                    self.access_history[page_id] = []
                self.access_history[page_id].append(page.last_accessed)
                
                # Keep only last K accesses
                if len(self.access_history[page_id]) > 10:
                    self.access_history[page_id] = self.access_history[page_id][-10:]
                
                return page
            
            self.miss_count += 1
            return None
    
    def add_page(self, page: Page) -> bool:
        """
        Add a page to the buffer pool.
        
        Returns True if page was added, False if pool is full and eviction occurred.
        """
        with self.lock:
            # If page already exists, just update
            if page.page_id in self.pages:
                self.pages[page.page_id] = page
                return True
            
            # Need to evict if pool is full
            if len(self.pages) >= self.pool_size:
                evicted = self._evict_page()
                if not evicted:
                    return False
            
            page.last_accessed = self._current_time()
            self.pages[page.page_id] = page
            self.read_count += 1
            
            return True
    
    def remove_page(self, page_id: int) -> bool:
        """Remove a page from the pool."""
        with self.lock:
            if page_id in self.pages:
                del self.pages[page_id]
                if page_id in self.access_history:
                    del self.access_history[page_id]
                if page_id in self.page_importance:
                    del self.page_importance[page_id]
                return True
            return False
    
    def _evict_page(self) -> bool:
        """
        Evict a page using LRU-K with learned importance.
        
        Returns True if a page was evicted.
        """
        if not self.pages:
            return False
        
        candidates = []
        
        for page_id, page in self.pages.items():
            # Don't evict pinned pages
            if page.pin_count > 0:
                continue
            
            # Calculate eviction score
            score = self._calculate_eviction_score(page_id, page)
            candidates.append((page_id, score))
        
        if not candidates:
            return False
        
        # Sort by score (higher = better candidate)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict the worst candidate
        evict_id, _ = candidates[0]
        
        # Flush if dirty
        evicted_page = self.pages[evict_id]
        if evicted_page.is_dirty:
            self._flush_page(evicted_page)
            self.write_count += 1
        
        self.remove_page(evict_id)
        return True
    
    def _calculate_eviction_score(self, page_id: int, page: Page) -> float:
        """
        Calculate eviction score using LRU-K and learned importance.
        
        Higher score = better candidate for eviction.
        """
        # Recency score (0-1, higher = less recently used)
        if page_id in self.access_history and self.access_history[page_id]:
            recency = self._current_time() - max(self.access_history[page_id])
            recency_score = min(recency / 3600, 1.0)  # Cap at 1 hour
        else:
            recency_score = 1.0  # Never accessed
        
        # Frequency score (0-1, higher = less frequently used)
        if page_id in self.access_history:
            frequency = len(self.access_history[page_id])
            frequency_score = 1.0 / (1.0 + frequency * 0.1)
        else:
            frequency_score = 1.0
        
        # Importance score (0-1, higher = less important)
        importance = self.page_importance.get(page_id, 0.5)
        importance_score = 1.0 - importance
        
        # Dirty penalty
        dirty_penalty = 0.5 if page.is_dirty else 0.0
        
        # Combined score
        score = (
            recency_score * 0.3 +
            frequency_score * 0.3 +
            importance_score * 0.3 +
            dirty_penalty
        )
        
        return score
    
    def update_page_importance(self, page_id: int, importance: float):
        """Update learned importance score for a page."""
        with self.lock:
            self.page_importance[page_id] = importance
    
    def resize(self, new_size: int):
        """Resize the buffer pool."""
        with self.lock:
            self.pool_size = new_size
            
            # Evict until under limit
            while len(self.pages) > new_size:
                if not self._evict_page():
                    break
    
    def _flush_page(self, page: Page):
        """Flush a dirty page to disk."""
        # In a real implementation, this would write to disk
        page.is_dirty = False
    
    def flush_all(self):
        """Flush all dirty pages to disk."""
        with self.lock:
            for page in self.pages.values():
                if page.is_dirty:
                    self._flush_page(page)
                    self.write_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer pool statistics."""
        with self.lock:
            return {
                "pool_size": self.pool_size,
                "pages_cached": len(self.pages),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": self.hit_rate,
                "read_count": self.read_count,
                "write_count": self.write_count
            }
    
    def _current_time(self) -> float:
        return __import__('time').time()


class StorageEngine:
    """
    Storage engine for HyperionDB.
    
    Manages persistent storage of pages and files.
    """
    
    def __init__(self, db_path: str, page_size: int = 8192):
        """
        Initialize storage engine.
        
        Args:
            db_path: Base path for database files
            page_size: Size of each page in bytes
        """
        self.db_path = db_path
        self.page_size = page_size
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Page file mapping
        self.page_files: Dict[str, str] = {}
        
        # Buffer pool
        self.buffer_pool = BufferPool()
        
        # Statistics
        self.io_count = 0
        self.bytes_read = 0
        self.bytes_written = 0
    
    def get_page(self, table_name: str, page_id: int) -> Optional[Page]:
        """
        Read a page from storage.
        
        Args:
            table_name: Name of the table
            page_id: Page ID
            
        Returns:
            Page object or None if not found
        """
        # Check buffer pool first
        pool_key = self._get_pool_key(table_name, page_id)
        cached = self.buffer_pool.get_page(pool_key)
        
        if cached:
            return cached
        
        # Read from disk
        page = self._read_page_from_disk(table_name, page_id)
        
        if page:
            self.buffer_pool.add_page(page)
        
        return page
    
    def write_page(self, page: Page) -> bool:
        """
        Write a page to storage.
        
        Args:
            page: Page to write
            
        Returns:
            True if successful
        """
        # Update buffer pool
        self.buffer_pool.add_page(page)
        page.mark_dirty()
        
        # In real implementation, would write to disk
        self.io_count += 1
        self.bytes_written += page.size
        
        return True
    
    def allocate_page(self, table_name: str) -> int:
        """
        Allocate a new page for a table.
        
        Returns:
            New page ID
        """
        # Get next page ID for table
        if table_name not in self.page_files:
            self.page_files[table_name] = f"{self.db_path}/{table_name}.db"
        
        # In real implementation, this would track free pages
        page_id = len(self.page_files) * 1000 + self.io_count
        
        return page_id
    
    def _read_page_from_disk(self, table_name: str, page_id: int) -> Optional[Page]:
        """Read a page from disk."""
        file_path = self._get_table_file(table_name)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                offset = page_id * self.page_size
                f.seek(offset)
                data = f.read(self.page_size)
                
                if data:
                    self.io_count += 1
                    self.bytes_read += len(data)
                    return Page.from_bytes(data)
        except Exception:
            pass
        
        return None
    
    def _get_table_file(self, table_name: str) -> str:
        """Get the file path for a table."""
        if table_name not in self.page_files:
            self.page_files[table_name] = f"{self.db_path}/{table_name}.db"
        return self.page_files[table_name]
    
    def _get_pool_key(self, table_name: str, page_id: int) -> int:
        """Get buffer pool key for a page."""
        return hash(f"{table_name}:{page_id}")
    
    def checkpoint(self):
        """Perform a checkpoint - flush all dirty pages."""
        self.buffer_pool.flush_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "io_count": self.io_count,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "buffer_pool_stats": self.buffer_pool.get_stats()
        }
