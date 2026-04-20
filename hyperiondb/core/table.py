"""
Table Module for HyperionDB
===========================

Provides table data structure and operations.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import json
import numpy as np


@dataclass
class Column:
    """Represents a table column."""
    name: str
    data_type: str
    nullable: bool = True
    default_value: Any = None
    is_indexed: bool = False
    is_primary_key: bool = False
    
    def validate(self, value: Any) -> bool:
        """Validate a value for this column."""
        if value is None:
            return self.nullable
        
        if self.data_type == "int":
            return isinstance(value, int)
        elif self.data_type == "float":
            return isinstance(value, (int, float))
        elif self.data_type == "str":
            return isinstance(value, str)
        elif self.data_type == "bool":
            return isinstance(value, bool)
        
        return True


@dataclass
class Row:
    """Represents a table row with a unique row ID."""
    _rid: int
    _values: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self._values[key]
    
    def __setitem__(self, key: str, value: Any):
        self._values[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._values.copy()
    
    def __iter__(self):
        return iter(self._values)
    
    def __repr__(self):
        return f"Row(rid={self._rid}, {self._values})"


class Table:
    """
    HyperionDB Table with learned index support.
    
    A table stores rows and maintains various indexes including
    learned indexes for optimized query execution.
    """
    
    def __init__(self, name: str, columns: List[Tuple[str, str]], 
                 primary_key: Optional[str] = None):
        """
        Initialize a table.
        
        Args:
            name: Table name
            columns: List of (column_name, column_type) tuples
            primary_key: Primary key column name
        """
        self.name = name
        self.columns: Dict[str, Column] = {}
        self.primary_key = primary_key
        self.row_count = 0
        self._next_rid = 0
        
        # Initialize columns
        for col_name, col_type in columns:
            is_pk = (col_name == primary_key)
            self.columns[col_name] = Column(
                name=col_name,
                data_type=col_type,
                nullable=True,
                is_primary_key=is_pk
            )
        
        # Storage
        self.rows: List[Row] = []
        self.indexes: Dict[str, Any] = {}  # column_name -> index
        
        # Statistics for query optimization
        self.stats = TableStatistics()
        
        # Concurrency control
        self.lock = threading.RLock()
        
        # Column statistics cache
        self._column_min: Dict[str, Any] = {}
        self._column_max: Dict[str, Any] = {}
        self._column_null_count: Dict[str, int] = defaultdict(int)
        self._column_nunique: Dict[str, int] = {}
    
    def add_index(self, column: str, index: Any):
        """Add an index for a column."""
        if column not in self.columns:
            raise ValueError(f"Column '{column}' does not exist")
        self.indexes[column] = index
        self.columns[column].is_indexed = True
    
    def insert(self, values: Dict[str, Any]) -> int:
        """
        Insert a row into the table.
        
        Args:
            values: Dictionary of column -> value
            
        Returns:
            Row ID of inserted row
        """
        with self.lock:
            # Validate columns
            for col_name in values:
                if col_name not in self.columns:
                    raise ValueError(f"Column '{col_name}' does not exist")
                
                if not self.columns[col_name].validate(values[col_name]):
                    raise ValueError(f"Invalid value for column '{col_name}'")
            
            # Create row
            rid = self._next_rid
            self._next_rid += 1
            
            row = Row(_rid=rid, _values=values.copy())
            self.rows.append(row)
            self.row_count += 1
            
            # Update statistics
            self._update_stats(values)
            
            return rid
    
    def _update_stats(self, values: Dict[str, Any]):
        """Update column statistics after insert."""
        for col_name, value in values.items():
            if value is None:
                self._column_null_count[col_name] += 1
            else:
                # Update min/max
                if col_name not in self._column_min:
                    self._column_min[col_name] = value
                    self._column_max[col_name] = value
                else:
                    if value < self._column_min[col_name]:
                        self._column_min[col_name] = value
                    if value > self._column_max[col_name]:
                        self._column_max[col_name] = value
        
        # Update nunique (approximate)
        unique_values = set()
        for row in self.rows[-100:]:  # Sample recent rows
            for col, val in row._values.items():
                unique_values.add(val)
        self._column_nunique = {col: len(unique_values) for col in self.columns}
    
    def scan(self, conditions: Optional[Dict] = None,
             columns: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan the table with optional filtering.
        
        Args:
            conditions: WHERE conditions as {column: value} or {column: (op, value)}
            columns: Columns to select
            
        Returns:
            List of matching rows as dictionaries
        """
        with self.lock:
            results = []
            
            for row in self.rows:
                if self._matches_conditions(row, conditions):
                    if columns:
                        filtered = {k: row.get(k) for k in columns if k in row._values}
                        results.append(filtered)
                    else:
                        results.append(row.to_dict())
            
            return results
    
    def _matches_conditions(self, row: Row, 
                           conditions: Optional[Dict]) -> bool:
        """Check if a row matches the conditions."""
        if not conditions:
            return True
        
        for col, cond in conditions.items():
            value = row.get(col)
            
            if isinstance(cond, tuple):
                op, val = cond
                if op == "=":
                    if value != val:
                        return False
                elif op == ">":
                    if value is None or value <= val:
                        return False
                elif op == ">=":
                    if value is None or value < val:
                        return False
                elif op == "<":
                    if value is None or value >= val:
                        return False
                elif op == "<=":
                    if value is None or value > val:
                        return False
                elif op == "!=":
                    if value == val:
                        return False
                elif op == "IN":
                    if value not in val:
                        return False
                elif op == "BETWEEN":
                    if value is None or value < val[0] or value > val[1]:
                        return False
            else:
                if value != cond:
                    return False
        
        return True
    
    def update(self, values: Dict[str, Any],
               conditions: Optional[Dict] = None) -> int:
        """
        Update rows matching conditions.
        
        Args:
            values: Dictionary of column -> new_value
            conditions: WHERE conditions
            
        Returns:
            Number of rows updated
        """
        with self.lock:
            count = 0
            
            for row in self.rows:
                if self._matches_conditions(row, conditions):
                    for col, val in values.items():
                        if col in row._values:
                            row[col] = val
                    count += 1
            
            return count
    
    def delete(self, conditions: Optional[Dict] = None) -> int:
        """
        Delete rows matching conditions.
        
        Args:
            conditions: WHERE conditions
            
        Returns:
            Number of rows deleted
        """
        with self.lock:
            if not conditions:
                count = len(self.rows)
                self.rows.clear()
                self.row_count = 0
                self._next_rid = 0
                return count
            
            original_count = len(self.rows)
            self.rows = [r for r in self.rows if not self._matches_conditions(r, conditions)]
            self.row_count = len(self.rows)
            
            return original_count - self.row_count
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get statistics for a column."""
        if column not in self.columns:
            raise ValueError(f"Column '{column}' does not exist")
        
        values = [row.get(column) for row in self.rows if row.get(column) is not None]
        
        if not values:
            return {
                "count": 0,
                "null_count": self._column_null_count.get(column, 0),
                "nunique": 0,
                "min": None,
                "max": None,
                "avg": None
            }
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        return {
            "count": len(values),
            "null_count": self._column_null_count.get(column, 0),
            "nunique": len(set(values)),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "avg": sum(numeric_values) / len(numeric_values) if numeric_values else None
        }
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Get overall table statistics."""
        return {
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "size_bytes": sum(len(str(r)) for r in self.rows),
            "column_stats": {
                col: self.get_column_stats(col) for col in self.columns
            },
            "indexed_columns": list(self.indexes.keys())
        }
    
    def to_json(self) -> str:
        """Serialize table to JSON."""
        data = {
            "name": self.name,
            "columns": [(c.name, c.data_type) for c in self.columns.values()],
            "primary_key": self.primary_key,
            "rows": [r.to_dict() for r in self.rows]
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Table':
        """Deserialize table from JSON."""
        data = json.loads(json_str)
        table = cls(data["name"], data["columns"], data.get("primary_key"))
        for row_data in data["rows"]:
            # Remove _rid as it will be regenerated
            values = {k: v for k, v in row_data.items() if not k.startswith("_")}
            table.insert(values)
        return table
    
    def __len__(self):
        return self.row_count
    
    def __repr__(self):
        return f"Table({self.name}, rows={self.row_count}, cols={len(self.columns)})"


@dataclass
class TableStatistics:
    """Statistics for query optimization."""
    row_count: int = 0
    page_count: int = 0
    avg_row_size: float = 0.0
    total_size_bytes: int = 0
    last_updated: float = 0.0
    
    # Cardinality estimates
    column_cardinalities: Dict[str, float] = field(default_factory=dict)
    
    # Histogram data
    column_histograms: Dict[str, List[Tuple[Any, int]]] = field(default_factory=dict)
    
    # Sample data for learning
    sample_rows: List[Dict] = field(default_factory=list)
    
    def update(self, row_count: int, avg_row_size: float):
        """Update statistics."""
        self.row_count = row_count
        self.avg_row_size = avg_row_size
        self.last_updated = 0  # Would use time.time()
