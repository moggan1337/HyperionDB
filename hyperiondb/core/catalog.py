"""
Catalog Module for HyperionDB
=============================

Provides database catalog for metadata management.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os


@dataclass
class TableMetadata:
    """Metadata for a table."""
    name: str
    columns: List[Dict[str, str]]  # [{name, type, nullable, ...}]
    primary_key: Optional[str] = None
    row_count: int = 0
    size_bytes: int = 0
    indexes: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "primary_key": self.primary_key,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "indexes": self.indexes,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TableMetadata':
        return cls(**data)


@dataclass 
class IndexMetadata:
    """Metadata for an index."""
    name: str
    table_name: str
    column_name: str
    index_type: str  # btree, hash, learned
    is_unique: bool = False
    size_bytes: int = 0
    cardinality: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "index_type": self.index_type,
            "is_unique": self.is_unique,
            "size_bytes": self.size_bytes,
            "cardinality": self.cardinality
        }


class Catalog:
    """
    Database catalog for managing metadata.
    
    The catalog stores information about tables, columns, indexes,
    and other database objects.
    """
    
    def __init__(self):
        """Initialize empty catalog."""
        self.tables: Dict[str, TableMetadata] = {}
        self.indexes: Dict[str, IndexMetadata] = {}
        
        # Statistics
        self.table_count = 0
        self.index_count = 0
        self.total_size = 0
    
    def add_table(self, table) -> TableMetadata:
        """
        Add a table to the catalog.
        
        Args:
            table: Table object
            
        Returns:
            Created TableMetadata
        """
        import time
        
        metadata = TableMetadata(
            name=table.name,
            columns=[
                {"name": col.name, "type": col.data_type, "nullable": col.nullable}
                for col in table.columns.values()
            ],
            primary_key=table.primary_key,
            row_count=table.row_count,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.tables[table.name] = metadata
        self.table_count += 1
        
        return metadata
    
    def remove_table(self, table_name: str):
        """Remove a table from the catalog."""
        if table_name in self.tables:
            del self.tables[table_name]
            self.table_count -= 1
            
            # Remove associated indexes
            to_remove = [
                idx_name for idx_name, idx in self.indexes.items()
                if idx.table_name == table_name
            ]
            for idx_name in to_remove:
                del self.indexes[idx_name]
                self.index_count -= 1
    
    def add_index(self, metadata: IndexMetadata):
        """Add an index to the catalog."""
        self.indexes[metadata.name] = metadata
        self.index_count += 1
        
        # Update table metadata
        if metadata.table_name in self.tables:
            self.tables[metadata.table_name].indexes.append(metadata.name)
    
    def get_table(self, table_name: str) -> Optional[TableMetadata]:
        """Get table metadata."""
        return self.tables.get(table_name)
    
    def get_index(self, index_name: str) -> Optional[IndexMetadata]:
        """Get index metadata."""
        return self.indexes.get(index_name)
    
    def get_indexes_for_table(self, table_name: str) -> List[IndexMetadata]:
        """Get all indexes for a table."""
        return [
            idx for idx in self.indexes.values()
            if idx.table_name == table_name
        ]
    
    def update_table_stats(self, table_name: str, row_count: int, size_bytes: int):
        """Update table statistics."""
        if table_name in self.tables:
            import time
            self.tables[table_name].row_count = row_count
            self.tables[table_name].size_bytes = size_bytes
            self.tables[table_name].updated_at = time.time()
    
    def list_tables(self) -> List[str]:
        """List all table names."""
        return list(self.tables.keys())
    
    def list_indexes(self) -> List[str]:
        """List all index names."""
        return list(self.indexes.keys())
    
    def save(self, path: str):
        """Save catalog to disk."""
        data = {
            "tables": {name: meta.to_dict() for name, meta in self.tables.items()},
            "indexes": {name: meta.to_dict() for name, meta in self.indexes.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load catalog from disk."""
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.tables = {
            name: TableMetadata.from_dict(meta)
            for name, meta in data.get("tables", {}).items()
        }
        
        self.indexes = {
            name: IndexMetadata(**meta)
            for name, meta in data.get("indexes", {}).items()
        }
        
        self.table_count = len(self.tables)
        self.index_count = len(self.indexes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return {
            "table_count": self.table_count,
            "index_count": self.index_count,
            "total_size": self.total_size
        }
