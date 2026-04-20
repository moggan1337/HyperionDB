"""
Query Module for HyperionDB
===========================

Provides query representation and plan generation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class PlanNodeType(Enum):
    """Types of plan nodes in a query plan."""
    TABLE_SCAN = "table_scan"
    INDEX_SCAN = "index_scan"
    LEARNED_INDEX_SCAN = "learned_index_scan"
    FILTER = "filter"
    PROJECT = "project"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    UNION = "union"


class JoinType(Enum):
    """Types of joins."""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"


class PhysicalOperator(Enum):
    """Physical operators for execution."""
    SEQ_SCAN = "seq_scan"
    INDEX_SCAN = "index_scan"
    LEARNED_INDEX_SCAN = "learned_index_scan"
    NESTED_LOOP = "nested_loop"
    HASH_JOIN = "hash_join"
    SORT_MERGE_JOIN = "sort_merge_join"
    HASH_AGG = "hash_agg"
    SORT_AGG = "sort_agg"
    SORT = "sort"
    LIMIT = "limit"


@dataclass
class PlanNode:
    """
    Represents a node in a query execution plan.
    
    Each node represents an operation that transforms data.
    """
    node_id: int
    node_type: PlanNodeType
    physical_operator: PhysicalOperator
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    conditions: Optional[Dict] = None
    children: List['PlanNode'] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    
    # Cost estimates
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    estimated_cpu_cost: float = 0.0
    estimated_io_cost: float = 0.0
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'PlanNode'):
        """Add a child node."""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "physical_operator": self.physical_operator.value,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "conditions": self.conditions,
            "estimated_cost": self.estimated_cost,
            "estimated_rows": self.estimated_rows,
            "children": [c.to_dict() for c in self.children]
        }
    
    def __repr__(self):
        return f"PlanNode({self.node_type.value}, cost={self.estimated_cost:.2f})"


@dataclass
class QueryPlan:
    """
    Represents a complete query execution plan.
    
    A query plan is a tree of PlanNodes that describes how to execute
    a query efficiently.
    """
    plan_id: str
    root: Optional[PlanNode] = None
    tables: List[str] = field(default_factory=list)
    selected_columns: List[str] = field(default_factory=list)
    conditions: Optional[Dict] = field(default_factory=None)
    join_order: List[str] = field(default_factory=list)
    
    # Optimization metadata
    optimization_time: float = 0.0
    total_estimated_cost: float = 0.0
    confidence_score: float = 0.0
    
    # Learning metadata
    q_value: float = 0.0  # RL Q-value
    visit_count: int = 0
    reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "root": self.root.to_dict() if self.root else None,
            "tables": self.tables,
            "selected_columns": self.selected_columns,
            "join_order": self.join_order,
            "optimization_time": self.optimization_time,
            "total_estimated_cost": self.total_estimated_cost,
            "confidence_score": self.confidence_score
        }
    
    def __repr__(self):
        return f"QueryPlan(tables={self.tables}, cost={self.total_estimated_cost:.2f})"


class Query:
    """
    Represents a database query.
    
    Queries are parsed and converted to QueryPlans by the optimizer.
    """
    
    def __init__(self, query_type: str):
        """
        Initialize a query.
        
        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
        """
        self.query_type = query_type
        self.tables: List[str] = []
        self.columns: List[str] = []
        self.conditions: Dict[str, Any] = {}
        self.join_conditions: List[Dict] = []
        self.order_by: List[Tuple[str, str]] = []
        self.group_by: List[str] = []
        self.having: Optional[Dict] = None
        self.limit: Optional[int] = None
        self.offset: Optional[int] = None
        
        # Values for INSERT/UPDATE
        self.values: Optional[Dict[str, Any]] = None
    
    @classmethod
    def select(cls, tables: List[str], columns: Optional[List[str]] = None,
              conditions: Optional[Dict] = None) -> 'Query':
        """Create a SELECT query."""
        q = cls("SELECT")
        q.tables = tables
        q.columns = columns or []
        q.conditions = conditions or {}
        return q
    
    @classmethod
    def insert(cls, table: str, values: Dict[str, Any]) -> 'Query':
        """Create an INSERT query."""
        q = cls("INSERT")
        q.tables = [table]
        q.values = values
        return q
    
    @classmethod
    def update(cls, table: str, values: Dict[str, Any],
               conditions: Optional[Dict] = None) -> 'Query':
        """Create an UPDATE query."""
        q = cls("UPDATE")
        q.tables = [table]
        q.values = values
        q.conditions = conditions or {}
        return q
    
    @classmethod
    def delete(cls, table: str, 
               conditions: Optional[Dict] = None) -> 'Query':
        """Create a DELETE query."""
        q = cls("DELETE")
        q.tables = [table]
        q.conditions = conditions or {}
        return q
    
    def set_join_condition(self, condition: Dict):
        """Add a join condition."""
        self.join_conditions.append(condition)
    
    def set_order_by(self, column: str, direction: str = "ASC"):
        """Set ORDER BY clause."""
        self.order_by.append((column, direction))
    
    def set_group_by(self, columns: List[str]):
        """Set GROUP BY clause."""
        self.group_by = columns
    
    def set_limit(self, limit: int, offset: Optional[int] = None):
        """Set LIMIT clause."""
        self.limit = limit
        self.offset = offset
    
    def __repr__(self):
        return f"Query({self.query_type}, tables={self.tables})"
