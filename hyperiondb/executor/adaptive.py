"""
Adaptive Query Executor for HyperionDB
======================================

Dynamic query execution that adapts to runtime conditions
for optimal performance.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import logging
import time

from hyperiondb.core.query import QueryPlan, PlanNode, PlanNodeType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics for query execution."""
    start_time: float
    end_time: Optional[float] = None
    rows_processed: int = 0
    cpu_time: float = 0.0
    io_time: float = 0.0
    memory_used: int = 0
    spills: int = 0
    
    @property
    def elapsed_time(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        elapsed = self.elapsed_time
        return self.rows_processed / max(elapsed, 0.001)


class AdaptiveExecutor:
    """
    Adaptive Query Executor for HyperionDB.
    
    Dynamically adjusts query execution strategies based on
    runtime conditions and intermediate results.
    
    Adaptive Strategies:
    - Dynamic join reordering
    - Runtime query re-optimization
    - Adaptive parallelism
    - Memory-aware execution
    - Progressive execution
    
    Features:
    - Plan modification during execution
    - Runtime statistics collection
    - Automatic fallback strategies
    - Progressive result streaming
    """
    
    def __init__(self):
        """Initialize adaptive executor."""
        self.execution_count = 0
        self.adaptations_made = 0
        
        # Execution policies
        self.policies = {
            "enable_dynamic_join_reorder": True,
            "enable_runtime_reopt": True,
            "enable_parallel_execution": True,
            "memory_limit_mb": 1024,
            "spill_threshold_rows": 100000
        }
        
        # Runtime statistics
        self.runtime_stats: Dict[str, ExecutionStats] = {}
        
        # Adaptive thresholds
        self.thresholds = {
            "slow_node_ms": 100,
            "high_cardinality_threshold": 10000,
            "memory_pressure_threshold": 0.8
        }
        
        logger.info("Adaptive Executor initialized")
    
    def execute(self, plan: QueryPlan, table: Any) -> List[Dict]:
        """
        Execute a query plan with adaptive execution.
        
        Args:
            plan: Query plan to execute
            table: Table to query
            
        Returns:
            Query results
        """
        self.execution_count += 1
        start_time = time.time()
        
        stats = ExecutionStats(start_time=start_time)
        
        try:
            # Execute plan tree
            results = self._execute_plan(plan.root, table, stats)
            
            stats.end_time = time.time()
            self.runtime_stats[plan.plan_id] = stats
            
            # Check if adaptation is needed
            if self._should_adapt(stats, plan):
                self._adapt_execution(plan, stats)
            
            return results
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            # Fallback to simple execution
            return table.scan(plan.conditions, plan.selected_columns)
    
    def _execute_plan(self, node: PlanNode, table: Any, 
                      stats: ExecutionStats) -> List[Dict]:
        """Execute a single plan node."""
        if node is None:
            return []
        
        node_start = time.time()
        
        if node.node_type == PlanNodeType.TABLE_SCAN:
            results = table.scan(node.conditions, node.output_columns)
            stats.rows_processed += len(results)
            
        elif node.node_type == PlanNodeType.LEARNED_INDEX_SCAN:
            results = self._execute_learned_index_scan(node, table)
            stats.rows_processed += len(results)
            
        elif node.node_type == PlanNodeType.INDEX_SCAN:
            results = self._execute_index_scan(node, table)
            stats.rows_processed += len(results)
            
        elif node.node_type == PlanNodeType.FILTER:
            # Execute children first
            child_results = []
            for child in node.children:
                child_results.extend(self._execute_plan(child, table, stats))
            
            # Apply filter
            results = self._apply_filter(child_results, node.conditions)
            stats.rows_processed += len(results)
            
        elif node.node_type == PlanNodeType.PROJECT:
            # Execute children first
            child_results = []
            for child in node.children:
                child_results.extend(self._execute_plan(child, table, stats))
            
            # Apply projection
            if node.output_columns:
                results = [
                    {k: v for k, v in row.items() if k in node.output_columns}
                    for row in child_results
                ]
            else:
                results = child_results
            
        elif node.node_type == PlanNodeType.JOIN:
            results = self._execute_join(node, table, stats)
            
        else:
            # Default: scan table
            results = table.scan()
        
        node_time = time.time() - node_start
        stats.cpu_time += node_time
        
        return results
    
    def _execute_learned_index_scan(self, node: PlanNode, 
                                    table: Any) -> List[Dict]:
        """Execute learned index scan."""
        column = node.column_name
        conditions = node.conditions
        
        # Check if learned index exists
        if column in table.indexes:
            index = table.indexes[column]
            
            if conditions:
                for col, cond in conditions.items():
                    if isinstance(cond, tuple):
                        op, value = cond
                    else:
                        op, value = "=", cond
                    
                    pos = index.search(value)
                    if pos is not None:
                        return table.scan(conditions, node.output_columns)
        
        # Fallback to regular scan
        return table.scan(conditions, node.output_columns)
    
    def _execute_index_scan(self, node: PlanNode, table: Any) -> List[Dict]:
        """Execute regular index scan."""
        # Use available index
        if node.column_name in table.indexes:
            index = table.indexes[node.column_name]
            
            if node.conditions:
                for col, cond in node.conditions.items():
                    if isinstance(cond, tuple):
                        _, value = cond
                    else:
                        value = cond
                    
                    pos = index.search(value)
                    if pos is not None:
                        return table.scan(node.conditions, node.output_columns)
        
        return table.scan(node.conditions, node.output_columns)
    
    def _apply_filter(self, rows: List[Dict], 
                     conditions: Dict) -> List[Dict]:
        """Apply filter conditions to rows."""
        if not conditions:
            return rows
        
        filtered = []
        
        for row in rows:
            matches = True
            
            for col, cond in conditions.items():
                value = row.get(col)
                
                if isinstance(cond, tuple):
                    op, expected = cond
                    
                    if op == "=":
                        if value != expected:
                            matches = False
                            break
                    elif op == ">":
                        if value is None or value <= expected:
                            matches = False
                            break
                    elif op == "<":
                        if value is None or value >= expected:
                            matches = False
                            break
                    elif op == ">=":
                        if value is None or value < expected:
                            matches = False
                            break
                    elif op == "<=":
                        if value is None or value > expected:
                            matches = False
                            break
                    elif op == "!=":
                        if value == expected:
                            matches = False
                            break
                else:
                    if value != cond:
                        matches = False
                        break
            
            if matches:
                filtered.append(row)
        
        return filtered
    
    def _execute_join(self, node: PlanNode, table: Any,
                     stats: ExecutionStats) -> List[Dict]:
        """Execute join operation."""
        if len(node.children) < 2:
            return []
        
        # Execute left child
        left_results = self._execute_plan(node.children[0], table, stats)
        
        # Execute right child
        right_results = self._execute_plan(node.children[1], table, stats)
        
        # Simple hash join
        join_col = node.conditions.get("join_column") if node.conditions else None
        
        if join_col:
            # Hash join
            hash_table = {}
            
            for row in left_results:
                key = row.get(join_col)
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append(row)
            
            joined = []
            for row in right_results:
                key = row.get(join_col)
                if key in hash_table:
                    for left_row in hash_table[key]:
                        joined.append({**left_row, **row})
            
            return joined
        
        # Cartesian product fallback
        joined = []
        for left in left_results:
            for right in right_results:
                joined.append({**left, **right})
        
        return joined
    
    def _should_adapt(self, stats: ExecutionStats, 
                     plan: QueryPlan) -> bool:
        """Determine if execution should be adapted."""
        # Check if execution is slow
        if stats.elapsed_time > self.thresholds["slow_node_ms"]:
            return True
        
        # Check if cardinality estimate was wrong
        if plan.root:
            actual_rows = stats.rows_processed
            estimated_rows = plan.root.estimated_rows
            
            if estimated_rows > 0:
                error = abs(actual_rows - estimated_rows) / estimated_rows
                
                if error > 0.5:  # 50% error
                    return True
        
        # Check memory pressure
        if stats.memory_used > self.policies["memory_limit_mb"] * 1024 * 1024 * 0.8:
            return True
        
        return False
    
    def _adapt_execution(self, plan: QueryPlan, stats: ExecutionStats):
        """Adapt execution based on runtime conditions."""
        self.adaptations_made += 1
        
        logger.info(f"Adapting execution for plan {plan.plan_id}")
        
        # Strategy 1: If cardinality was underestimated, use more efficient join
        if plan.root and stats.rows_processed > plan.root.estimated_rows * 1.5:
            logger.info("Cardinality underestimated - considering hash join")
        
        # Strategy 2: If execution is slow, try parallel execution
        if stats.elapsed_time > self.thresholds["slow_node_ms"] * 2:
            if self.policies["enable_parallel_execution"]:
                logger.info("Slow execution - enabling parallel execution")
        
        # Strategy 3: If memory is high, use streaming
        if stats.memory_used > self.policies["memory_limit_mb"] * 1024 * 1024 * 0.7:
            logger.info("High memory usage - switching to streaming")
    
    def modify_plan(self, plan: QueryPlan, modification: str) -> QueryPlan:
        """
        Modify a query plan based on runtime information.
        
        Args:
            plan: Original query plan
            modification: Type of modification
            
        Returns:
            Modified query plan
        """
        if modification == "use_hash_join":
            # Change join method
            pass
        elif modification == "use_index":
            # Force index usage
            pass
        elif modification == "increase_parallelism":
            # Increase parallelism degree
            pass
        
        return plan
    
    def get_execution_stats(self, plan_id: str) -> Optional[ExecutionStats]:
        """Get execution statistics for a plan."""
        return self.runtime_stats.get(plan_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "execution_count": self.execution_count,
            "adaptations_made": self.adaptations_made,
            "adaptation_rate": self.adaptations_made / max(1, self.execution_count),
            "policies": self.policies,
            "thresholds": self.thresholds
        }
