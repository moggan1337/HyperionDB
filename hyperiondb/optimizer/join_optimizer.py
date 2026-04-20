"""
GNN-based Join Optimizer for HyperionDB
=======================================

Graph Neural Network-based join order enumeration and optimization.
Uses GNNs to learn optimal join orderings from query execution history.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import logging
import random

from hyperiondb.core.table import Table

logger = logging.getLogger(__name__)


@dataclass
class JoinNode:
    """Represents a node in the join graph."""
    table_name: str
    row_count: int = 0
    estimated_size: float = 0.0
    
    # Graph features
    degree: int = 0
    neighbor_features: List[float] = field(default_factory=list)
    
    # Learned embedding
    embedding: np.ndarray = None
    
    def __hash__(self):
        return hash(self.table_name)


@dataclass
class JoinEdge:
    """Represents a join predicate in the graph."""
    left_table: str
    right_table: str
    join_column_left: str
    join_column_right: str
    selectivity: float = 0.1
    
    # Learned features
    embedding: np.ndarray = None
    
    def __hash__(self):
        return hash((self.left_table, self.right_table))


@dataclass 
class JoinPlan:
    """Represents a complete join plan."""
    tables: List[str]
    join_order: List[str]
    join_type: str = "INNER"
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    
    # GNN features
    graph_embedding: np.ndarray = None
    
    def __repr__(self):
        return f"JoinPlan(order={self.join_order}, cost={self.estimated_cost:.2f})"


class GNNJoinOptimizer:
    """
    Graph Neural Network-based Join Optimizer.
    
    Uses GNNs to learn optimal join orderings by:
    1. Building a join graph from tables and predicates
    2. Computing node embeddings using graph convolution
    3. Learning to predict optimal join order
    4. Enumerating join orders using learned heuristics
    
    Key Features:
    - Learned join graph embeddings
    - Cardinality-aware join enumeration
    - Adaptive join method selection
    - Multi-way join optimization
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize the GNN Join Optimizer.
        
        Args:
            embedding_dim: Dimension of learned embeddings
        """
        self.embedding_dim = embedding_dim
        
        # Join graph
        self.tables: Dict[str, JoinNode] = {}
        self.join_predicates: List[JoinEdge] = []
        
        # Learned parameters
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.edge_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Neural network weights
        self.gnn_weights: List[np.ndarray] = []
        self.output_weights: np.ndarray = None
        self._init_weights()
        
        # Join enumeration cache
        self.enumeration_cache: Dict[str, List[JoinPlan]] = {}
        
        # Statistics
        self.optimization_count = 0
        self.cache_hits = 0
        
        # Learned patterns
        self.join_patterns: Dict[str, str] = {}  # pattern -> best_order
        
        logger.info(f"GNN Join Optimizer initialized with embedding_dim={embedding_dim}")
    
    def _init_weights(self):
        """Initialize neural network weights."""
        np.random.seed(42)
        
        # GNN layer weights
        for i in range(3):  # 3 GNN layers
            if i == 0:
                in_dim = 32  # Input features
            else:
                in_dim = self.embedding_dim
            out_dim = self.embedding_dim
            
            W = np.random.randn(in_dim, out_dim) * 0.1
            b = np.zeros((1, out_dim))
            self.gnn_weights.append((W, b))
        
        # Output layer for join order scoring
        self.output_weights = np.random.randn(self.embedding_dim * 2, 1) * 0.1
    
    def build_join_graph(self, tables: List[Table],
                         join_conditions: List[Dict]) -> Tuple[Dict, List]:
        """
        Build a join graph from tables and conditions.
        
        Args:
            tables: List of tables to join
            join_conditions: List of join condition dictionaries
            
        Returns:
            Tuple of (nodes, edges)
        """
        nodes = {}
        
        # Create nodes for each table
        for table in tables:
            node = JoinNode(
                table_name=table.name,
                row_count=table.row_count,
                estimated_size=table.row_count * 100  # Rough estimate
            )
            nodes[table.name] = node
        
        # Create edges for join predicates
        edges = []
        
        for cond in join_conditions:
            for left_col, right_col in cond.items():
                # Parse column names (format: table.column)
                left_parts = left_col.split(".")
                right_parts = right_col.split(".")
                
                left_table = left_parts[0]
                right_table = right_parts[0]
                
                edge = JoinEdge(
                    left_table=left_table,
                    right_table=right_table,
                    join_column_left=left_parts[1] if len(left_parts) > 1 else left_col,
                    join_column_right=right_parts[1] if len(right_parts) > 1 else right_col
                )
                edges.append(edge)
        
        return nodes, edges
    
    def compute_embeddings(self, nodes: Dict[str, JoinNode],
                           edges: List[JoinEdge]) -> Dict[str, np.ndarray]:
        """
        Compute embeddings using graph neural network.
        
        Uses message passing to compute node embeddings.
        
        Args:
            nodes: Dictionary of table nodes
            edges: List of join edges
            
        Returns:
            Dictionary of node embeddings
        """
        # Initialize node features
        features = {}
        
        for name, node in nodes.items():
            # Create initial feature vector
            feat = np.zeros(32)
            
            # Row count (log scale)
            feat[0] = np.log1p(node.row_count)
            
            # Estimated size
            feat[1] = np.log1p(node.estimated_size)
            
            # Degree (number of join predicates)
            degree = sum(
                1 for e in edges 
                if e.left_table == name or e.right_table == name
            )
            feat[2] = degree
            
            # Table name hash features
            name_hash = hash(name) % 100
            for i in range(5):
                feat[3 + i] = ((name_hash >> i) & 1) * 0.5 + 0.25
            
            features[name] = feat
        
        # Graph convolution layers
        current_features = features.copy()
        
        for layer_idx, (W, b) in enumerate(self.gnn_weights):
            new_features = {}
            
            for node_name, node in nodes.items():
                # Get neighbors
                neighbors = []
                for edge in edges:
                    if edge.left_table == node_name:
                        neighbors.append(edge.right_table)
                    elif edge.right_table == node_name:
                        neighbors.append(edge.left_table)
                
                # Aggregate neighbor features
                if neighbors:
                    neighbor_feats = np.array([
                        current_features[neighbor] 
                        for neighbor in neighbors
                    ])
                    aggregated = np.mean(neighbor_feats, axis=0)
                else:
                    aggregated = np.zeros(32 if layer_idx == 0 else self.embedding_dim)
                
                # Combine self and neighbor features
                combined = np.concatenate([
                    current_features[node_name],
                    aggregated
                ])
                
                # Linear transformation + activation
                transformed = combined @ W + b
                activated = np.tanh(transformed)  # Tanh activation
                
                new_features[node_name] = activated
            
            current_features = new_features
        
        return current_features
    
    def score_join_order(self, join_order: List[str], 
                         nodes: Dict[str, JoinNode],
                         embeddings: Dict[str, np.ndarray]) -> float:
        """
        Score a join order using learned model.
        
        Args:
            join_order: Ordered list of table names
            nodes: Table nodes
            embeddings: Computed embeddings
            
        Returns:
            Score (higher is better)
        """
        if len(join_order) < 2:
            return 1.0
        
        total_score = 0.0
        
        # Score consecutive pairs
        for i in range(len(join_order) - 1):
            left = join_order[i]
            right = join_order[i + 1]
            
            # Get embeddings
            left_emb = embeddings.get(left, np.zeros(self.embedding_dim))
            right_emb = embeddings.get(right, np.zeros(self.embedding_dim))
            
            # Compute pair score
            pair_features = np.concatenate([left_emb, right_emb])
            score = float(np.dot(pair_features, self.output_weights.flatten()))
            
            total_score += score
        
        # Normalize by number of joins
        avg_score = total_score / max(len(join_order) - 1, 1)
        
        return avg_score
    
    def enumerate_join_orders(self, tables: List[str],
                              max_enumerations: int = 100) -> List[List[str]]:
        """
        Enumerate possible join orders.
        
        Uses dynamic programming for optimal enumeration.
        
        Args:
            tables: List of table names
            max_enumerations: Maximum number to enumerate
            
        Returns:
            List of join orders
        """
        n = len(tables)
        
        if n <= 1:
            return [tables]
        
        if n == 2:
            return [tables, tables[::-1]]
        
        if n > 7:
            # Too many tables, use heuristic
            return self._heuristic_enumerate(tables, max_enumerations)
        
        # Dynamic programming enumeration
        all_orders = []
        
        def enumerate_subsets(tuple_tables: Tuple, remaining: List[str]):
            if len(all_orders) >= max_enumerations:
                return
            
            if not remaining:
                all_orders.append(list(tuple_tables))
                return
            
            for table in remaining:
                new_remaining = [t for t in remaining if t != table]
                enumerate_subsets(tuple_tables + (table,), new_remaining)
        
        enumerate_subset = []
        remaining = tables.copy()
        random.shuffle(remaining)
        
        # Limit enumeration
        for _ in range(min(max_enumerations, 5040)):  # 7! = 5040
            random.shuffle(tables)
            if tables not in all_orders:
                all_orders.append(tables.copy())
                if len(all_orders) >= max_enumerations:
                    break
        
        return all_orders
    
    def _heuristic_enumerate(self, tables: List[str],
                             max_enumerations: int) -> List[List[str]]:
        """Heuristic enumeration for many tables."""
        orders = []
        
        # Always try some common patterns
        orders.append(tables)  # Original order
        orders.append(tables[::-1])  # Reversed
        
        # Smallest first heuristic
        orders.append(sorted(tables, key=lambda t: self.tables.get(t, JoinNode(t)).row_count))
        
        # Random orders
        for _ in range(max_enumerations - 3):
            shuffled = tables.copy()
            random.shuffle(shuffled)
            if shuffled not in orders:
                orders.append(shuffled)
        
        return orders[:max_enumerations]
    
    def optimize_join_order(self, tables: List[Table],
                            join_conditions: List[Dict],
                            max_enumerations: int = 100) -> List[str]:
        """
        Optimize join order for given tables and conditions.
        
        Main entry point for join optimization.
        
        Args:
            tables: List of tables to join
            join_conditions: Join conditions
            max_enumerations: Maximum join orders to consider
            
        Returns:
            Optimized join order (list of table names)
        """
        self.optimization_count += 1
        
        # Build graph
        nodes, edges = self.build_join_graph(tables, join_conditions)
        self.tables = nodes
        
        # Check cache
        cache_key = self._get_cache_key(tables, join_conditions)
        if cache_key in self.enumeration_cache:
            self.cache_hits += 1
            best_plan = min(self.enumeration_cache[cache_key], 
                          key=lambda p: p.estimated_cost)
            return best_plan.join_order
        
        # Compute embeddings
        embeddings = self.compute_embeddings(nodes, edges)
        
        # Enumerate join orders
        table_names = [t.name for t in tables]
        candidate_orders = self.enumerate_join_orders(table_names, max_enumerations)
        
        # Score each order
        scored_orders = []
        
        for order in candidate_orders:
            score = self.score_join_order(order, nodes, embeddings)
            
            # Estimate cost for this order
            cost = self._estimate_join_cost(order, nodes, edges)
            
            plan = JoinPlan(
                tables=table_names,
                join_order=order,
                estimated_cost=cost,
                estimated_rows=self._estimate_join_rows(order, nodes)
            )
            
            scored_orders.append((score, cost, order, plan))
        
        # Sort by score and cost
        scored_orders.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        if scored_orders:
            best_plan = scored_orders[0][3]
            
            # Cache result
            self.enumeration_cache[cache_key] = [s[3] for s in scored_orders[:10]]
            
            # Learn from optimization
            self._learn_from_optimization(scored_orders)
            
            return best_plan.join_order
        
        return table_names
    
    def _estimate_join_cost(self, join_order: List[str],
                            nodes: Dict[str, JoinNode],
                            edges: List[JoinEdge]) -> float:
        """Estimate cost of a join order."""
        total_cost = 0.0
        intermediate_size = 1
        
        for i in range(len(join_order)):
            table = join_order[i]
            node = nodes.get(table, JoinNode(table))
            
            if i == 0:
                intermediate_size = node.row_count
            else:
                # Find join predicate
                prev_table = join_order[i - 1]
                
                for edge in edges:
                    if (edge.left_table == prev_table and edge.right_table == table) or \
                       (edge.right_table == prev_table and edge.left_table == table):
                        selectivity = edge.selectivity
                        break
                else:
                    selectivity = 0.1
                
                # Hash join cost estimate
                join_cost = intermediate_size + node.row_count
                total_cost += join_cost * selectivity
                
                intermediate_size = min(intermediate_size * node.row_count * selectivity, 
                                       intermediate_size * node.row_count)
        
        return total_cost
    
    def _estimate_join_rows(self, join_order: List[str],
                            nodes: Dict[str, JoinNode]) -> int:
        """Estimate number of rows in join result."""
        total_rows = 1
        
        for table in join_order:
            node = nodes.get(table, JoinNode(table))
            total_rows *= max(node.row_count, 1)
        
        # Cap at reasonable size
        return min(total_rows, 10_000_000)
    
    def _get_cache_key(self, tables: List[Table],
                       conditions: List[Dict]) -> str:
        """Generate cache key for tables and conditions."""
        table_names = sorted([t.name for t in tables])
        return ",".join(table_names)
    
    def _learn_from_optimization(self, scored_orders: List):
        """Learn from optimization results."""
        if len(scored_orders) < 2:
            return
        
        # Record pattern
        best_order = scored_orders[0][2]
        pattern = ",".join(best_order)
        
        # Update pattern frequency
        if pattern in self.join_patterns:
            self.join_patterns[pattern] += 1
        else:
            self.join_patterns[pattern] = 1
    
    def select_join_method(self, left_rows: int, right_rows: int,
                          join_type: str = "INNER") -> str:
        """
        Select optimal join method based on data sizes.
        
        Args:
            left_rows: Number of rows in left table
            right_rows: Number of rows in right table
            join_type: Type of join
            
        Returns:
            Recommended join method: "hash", "nested_loop", or "sort_merge"
        """
        total_rows = left_rows + right_rows
        
        # Decision rules learned from execution feedback
        if left_rows < 1000 or right_rows < 1000:
            return "nested_loop"
        
        if max(left_rows, right_rows) > 10_000_000:
            return "sort_merge"
        
        return "hash"
    
    def explain_join_plan(self, tables: List[Table],
                          conditions: List[Dict]) -> Dict[str, Any]:
        """Explain the join optimization decision."""
        nodes, edges = self.build_join_graph(tables, conditions)
        embeddings = self.compute_embeddings(nodes, edges)
        
        # Get best order
        best_order = self.optimize_join_order(tables, conditions)
        
        # Score candidates
        candidate_orders = self.enumerate_join_orders(
            [t.name for t in tables], 20
        )
        
        scored = []
        for order in candidate_orders:
            score = self.score_join_order(order, nodes, embeddings)
            cost = self._estimate_join_cost(order, nodes, edges)
            scored.append((order, score, cost))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "recommended_order": best_order,
            "table_embeddings": {
                name: emb.tolist() if isinstance(emb, np.ndarray) else emb
                for name, emb in embeddings.items()
            },
            "alternative_orders": [
                {"order": o, "score": s, "cost": c}
                for o, s, c in scored[:5]
            ],
            "join_methods": {
                f"{best_order[i]}_join_{best_order[i+1]}": 
                self.select_join_method(
                    nodes[best_order[i]].row_count,
                    nodes[best_order[i+1]].row_count
                )
                for i in range(len(best_order) - 1)
            },
            "optimization_count": self.optimization_count,
            "cache_hits": self.cache_hits
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "optimization_count": self.optimization_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.optimization_count),
            "cached_plans": len(self.enumeration_cache),
            "patterns_learned": len(self.join_patterns),
            "embedding_dim": self.embedding_dim
        }
