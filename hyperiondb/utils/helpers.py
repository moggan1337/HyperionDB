"""
Helper Utilities for HyperionDB
===============================

Common utility functions and decorators.
"""

import uuid
import time
import logging
from functools import wraps
from typing import Any, Callable


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{uid}" if prefix else uid


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def setup_logging(level: int = logging.INFO, 
                  format_string: str = None) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
