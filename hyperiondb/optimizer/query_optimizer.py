"""
Learned Query Optimizer for HyperionDB
=======================================

Reinforcement Learning-based query optimizer that learns optimal
query execution strategies from past query performance.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import random
import logging
from collections import defaultdict

from hyperiondb.core.query import QueryPlan, PlanNode, PlanNodeType, PhysicalOperator
from hyperiondb.core.table import Table
from hyperiondb.optimizer.cost_model import NeuralCostModel
from hyperiondb.optimizer.cardinality import CardinalityEstimator
from hyperiondb.optimizer.join_optimizer import GNNJoinOptimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """State representation for RL optimization."""
    table_name: str
    available_indexes: List[str]
    estimated_rows: int
    condition_complexity: float
    has_joins: bool
    estimated_selectivity: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        return np.array([
            self.estimated_rows,
            self.condition_complexity,
            1.0 if self.has_joins else 0.0,
            self.estimated_selectivity,
            len(self.available_indexes)
        ], dtype=np.float32)


@dataclass
class OptimizationAction:
    """Action representation for query optimization."""
    action_type: str  # use_index, seq_scan, nested_loop, hash_join, etc.
    target: Optional[str] = None  # Index name, table name, etc.
    
    def __hash__(self):
        return hash((self.action_type, self.target or ""))
    
    def __eq__(self, other):
        return self.action_type == other.action_type and self.target == other.target


@dataclass
class QValue:
    """Q-value for state-action pair with eligibility trace."""
    value: float = 0.0
    count: int = 0
    eligibility: float = 0.0


class LearnedQueryOptimizer:
    """
    Learned Query Optimizer using Reinforcement Learning.
    
    This optimizer uses Q-learning to learn optimal query execution
    strategies by observing the performance of past queries.
    
    Key features:
    - State representation based on query and table statistics
    - Action space for various optimization decisions
    - Q-value table for policy learning
    - Exploration vs exploitation trade-off
    - Experience replay for stable learning
    """
    
    def __init__(self, 
                 cost_model: NeuralCostModel,
                 cardinality_estimator: CardinalityEstimator,
                 join_optimizer: GNNJoinOptimizer,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.1,
                 discount_factor: float = 0.9):
        """
        Initialize the learned query optimizer.
        
        Args:
            cost_model: Neural network for cost estimation
            cardinality_estimator: Deep learning cardinality estimator
            join_optimizer: GNN-based join optimizer
            learning_rate: Learning rate for Q-learning
            exploration_rate: Exploration rate (epsilon)
            discount_factor: Discount factor for future rewards
        """
        self.cost_model = cost_model
        self.cardinality_estimator = cardinality_estimator
        self.join_optimizer = join_optimizer
        
        # RL hyperparameters
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        
        # Q-value table: (state_key, action) -> QValue
        self.q_table: Dict[Tuple[str, str], QValue] = defaultdict(QValue)
        
        # Experience replay buffer
        self.experience_buffer: List[Dict] = []
        self.max_buffer_size = 10000
        
        # Statistics
        self.total_optimizations = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        
        # Action space
        self.action_space = [
            OptimizationAction("seq_scan"),
            OptimizationAction("index_scan", "primary"),
            OptimizationAction("learned_index_scan"),
            OptimizationAction("hash_join"),
            OptimizationAction("nested_loop_join"),
            OptimizationAction("sort_merge_join"),
            OptimizationAction("hash_aggregate"),
            OptimizationAction("sort"),
        ]
        
        # Learned patterns
        self.pattern_cache: Dict[str, str] = {}  # query_pattern -> best_action
        
        logger.info("Learned Query Optimizer initialized with RL")
    
    def optimize(self, table: Table, conditions: Optional[Dict],
                 selected_columns: Optional[List[str]] = None) -> QueryPlan:
        """
        Optimize a query plan for the given table and conditions.
        
        Args:
            table: Table to query
            conditions: WHERE conditions
            selected_columns: Columns to select
            
        Returns:
            Optimized QueryPlan
        """
        self.total_optimizations += 1
        
        # Build state
        state = self._build_state(table, conditions)
        state_key = self._state_to_key(state)
        
        # Select action (explore or exploit)
        action = self._select_action(state, table)
        
        # Generate plan based on action
        plan = self._generate_plan(table, conditions, selected_columns, action)
        
        # Estimate cost
        estimated_cost = self.cost_model.estimate(plan) if self.cost_model else 0.0
        
        # Cache pattern for future use
        pattern = self._get_query_pattern(table.name, conditions)
        self.pattern_cache[pattern] = action.action_type
        
        return plan
    
    def _build_state(self, table: Table, 
                     conditions: Optional[Dict]) -> OptimizationState:
        """Build optimization state from table and conditions."""
        # Estimate cardinality
        est_rows = len(table.rows)
        if conditions and self.cardinality_estimator:
            est_rows = self.cardinality_estimator.estimate(table.name, conditions)
        
        # Calculate condition complexity
        complexity = self._calculate_complexity(conditions)
        
        # Calculate selectivity
        selectivity = est_rows / max(len(table.rows), 1)
        
        # Available indexes
        available_indexes = list(table.indexes.keys())
        
        return OptimizationState(
            table_name=table.name,
            available_indexes=available_indexes,
            estimated_rows=est_rows,
            condition_complexity=complexity,
            has_joins=False,  # Set by caller if needed
            estimated_selectivity=selectivity
        )
    
    def _state_to_key(self, state: OptimizationState) -> str:
        """Convert state to hashable key."""
        indexes_str = ",".join(sorted(state.available_indexes))
        return f"{state.table_name}|{indexes_str}|{state.estimated_rows:.0f}|{state.condition_complexity:.2f}"
    
    def _select_action(self, state: OptimizationState, 
                       table: Table) -> OptimizationAction:
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current optimization state
            table: Table being queried
            
        Returns:
            Selected action
        """
        # Check pattern cache first
        pattern = self._get_query_pattern(state.table_name, state.conditions)
        if pattern in self.pattern_cache:
            cached_action = self.pattern_cache[pattern]
            for action in self.action_space:
                if action.action_type == cached_action:
                    return action
        
        # Epsilon-greedy selection
        if random.random() < self.exploration_rate:
            # Explore: random action
            self.exploration_count += 1
            return random.choice(self.action_space)
        else:
            # Exploit: best known action
            self.exploitation_count += 1
            return self._get_best_action(state)
    
    def _get_best_action(self, state: OptimizationState) -> OptimizationAction:
        """Get the best action based on Q-values."""
        state_key = self._state_to_key(state)
        
        best_action = self.action_space[0]
        best_q_value = float('-inf')
        
        for action in self.action_space:
            q_value = self.q_table[(state_key, action.action_type)].value
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        # If no Q-values learned yet, use heuristic
        if best_q_value == 0:
            return self._heuristic_action(state)
        
        return best_action
    
    def _heuristic_action(self, state: OptimizationState) -> OptimizationAction:
        """
        Fallback heuristic when no Q-values are available.
        
        Uses rules based on statistics:
        - Low selectivity + available index -> use index
        - High selectivity -> seq scan
        - Large tables -> prefer hash joins
        """
        if state.available_indexes and state.estimated_selectivity < 0.1:
            if "learned" in str(state.available_indexes).lower():
                return OptimizationAction("learned_index_scan", state.available_indexes[0])
            return OptimizationAction("index_scan", "primary")
        
        if state.estimated_rows > 100000:
            return OptimizationAction("hash_join")
        
        return OptimizationAction("seq_scan")
    
    def _generate_plan(self, table: Table, conditions: Optional[Dict],
                       selected_columns: Optional[List[str]],
                       action: OptimizationAction) -> QueryPlan:
        """Generate query plan based on selected action."""
        plan_id = f"plan_{self.total_optimizations}"
        plan = QueryPlan(plan_id=plan_id)
        
        plan.tables = [table.name]
        plan.selected_columns = selected_columns or []
        plan.conditions = conditions
        plan.join_order = [table.name]
        
        # Build plan tree based on action
        node_id = 0
        
        if action.action_type == "seq_scan":
            root = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.TABLE_SCAN,
                physical_operator=PhysicalOperator.SEQ_SCAN,
                table_name=table.name,
                estimated_rows=len(table.rows),
                estimated_cost=len(table.rows) * 0.001
            )
        elif action.action_type == "learned_index_scan":
            target_col = action.target or (list(table.indexes.keys())[0] if table.indexes else None)
            root = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.LEARNED_INDEX_SCAN,
                physical_operator=PhysicalOperator.LEARNED_INDEX_SCAN,
                table_name=table.name,
                column_name=target_col,
                conditions=conditions,
                estimated_rows=self.cardinality_estimator.estimate(table.name, conditions) if self.cardinality_estimator else len(table.rows),
                estimated_cost=10.0  # Learned indexes are fast
            )
        elif action.action_type == "index_scan":
            target_col = action.target or list(table.indexes.keys())[0]
            root = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.INDEX_SCAN,
                physical_operator=PhysicalOperator.INDEX_SCAN,
                table_name=table.name,
                column_name=target_col,
                conditions=conditions,
                estimated_rows=len(table.rows) * 0.5,
                estimated_cost=100.0
            )
        else:
            # Default to table scan
            root = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.TABLE_SCAN,
                physical_operator=PhysicalOperator.SEQ_SCAN,
                table_name=table.name,
                estimated_rows=len(table.rows),
                estimated_cost=len(table.rows) * 0.001
            )
        
        plan.root = root
        
        # Add filter node if conditions exist
        if conditions:
            node_id += 1
            filter_node = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.FILTER,
                physical_operator=PhysicalOperator.SEQ_SCAN,
                conditions=conditions,
                estimated_rows=root.estimated_rows * 0.3,
                estimated_cost=50.0
            )
            filter_node.add_child(root)
            plan.root = filter_node
        
        # Add project node if selecting specific columns
        if selected_columns:
            node_id += 1
            project_node = PlanNode(
                node_id=node_id,
                node_type=PlanNodeType.PROJECT,
                physical_operator=PhysicalOperator.SEQ_SCAN,
                output_columns=selected_columns,
                estimated_rows=plan.root.estimated_rows,
                estimated_cost=10.0
            )
            project_node.add_child(plan.root)
            plan.root = project_node
        
        plan.total_estimated_cost = plan.root.estimated_cost
        
        return plan
    
    def update(self, state: OptimizationState, action: OptimizationAction,
               reward: float, next_state: Optional[OptimizationState] = None):
        """
        Update Q-values based on observed reward.
        
        Uses Q-learning update rule with eligibility traces.
        
        Args:
            state: Current state
            action: Action taken
            reward: Observed reward (negative cost)
            next_state: Resulting state
        """
        state_key = self._state_to_key(state)
        q_entry = self.q_table[(state_key, action.action_type)]
        
        # Calculate TD target
        if next_state:
            next_state_key = self._state_to_key(next_state)
            next_best_q = max(
                self.q_table[(next_state_key, a.action_type)].value
                for a in self.action_space
            )
            td_target = reward + self.discount_factor * next_best_q
        else:
            td_target = reward
        
        # Update Q-value with eligibility trace
        td_error = td_target - q_entry.value
        
        # Eligibility trace decay
        q_entry.eligibility = 0.9 * q_entry.eligibility + 1
        
        # Q-learning update
        q_entry.value += self.learning_rate * td_error * q_entry.eligibility
        q_entry.count += 1
        
        # Decay eligibility
        for key, entry in self.q_table.items():
            entry.eligibility *= 0.95
    
    def record_experience(self, state: OptimizationState, action: OptimizationAction,
                          reward: float, next_state: Optional[OptimizationState],
                          done: bool):
        """Record experience for replay buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_experiences(self, batch_size: int = 32):
        """Replay random experiences to update Q-values."""
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = random.sample(self.experience_buffer, batch_size)
        
        for exp in batch:
            self.update(
                exp["state"],
                exp["action"],
                exp["reward"],
                exp["next_state"]
            )
    
    def learn_from_query(self, plan: QueryPlan, actual_latency: float):
        """
        Learn from query execution feedback.
        
        Updates Q-values based on actual query performance.
        
        Args:
            plan: The executed query plan
            actual_latency: Actual execution time in milliseconds
        """
        # Calculate reward (negative latency)
        estimated_cost = plan.total_estimated_cost
        reward = -(actual_latency / max(estimated_cost, 1))
        
        # Update based on error between estimate and actual
        error = abs(actual_latency - estimated_cost) / max(estimated_cost, 1)
        
        if error > 0.5:  # Significant error
            # Penalize and learn
            reward *= 0.5
        
        # Update cost model
        if self.cost_model:
            self.cost_model.update(plan, actual_latency)
        
        logger.debug(f"Learned from query: latency={actual_latency:.2f}ms, reward={reward:.4f}")
    
    def _calculate_complexity(self, conditions: Optional[Dict]) -> float:
        """Calculate condition complexity score."""
        if not conditions:
            return 0.0
        
        complexity = 0.0
        for col, cond in conditions.items():
            complexity += 1.0
            if isinstance(cond, tuple):
                complexity += 0.5  # Operators add complexity
        
        return complexity
    
    def _get_query_pattern(self, table_name: str, 
                           conditions: Optional[Dict]) -> str:
        """Get a pattern string for query caching."""
        if not conditions:
            return f"{table_name}|none"
        
        cond_keys = sorted(conditions.keys())
        return f"{table_name}|{','.join(cond_keys)}"
    
    def explain_plan(self, table: Table, 
                     conditions: Optional[Dict]) -> Dict[str, Any]:
        """Explain the optimization decision for a query."""
        state = self._build_state(table, conditions)
        state_key = self._state_to_key(state)
        
        # Get Q-values for this state
        q_values = {
            action.action_type: self.q_table[(state_key, action.action_type)].value
            for action in self.action_space
        }
        
        best_action = max(q_values.items(), key=lambda x: x[1])
        
        return {
            "state": {
                "table": state.table_name,
                "estimated_rows": state.estimated_rows,
                "complexity": state.condition_complexity,
                "selectivity": state.estimated_selectivity,
                "available_indexes": state.available_indexes
            },
            "q_values": q_values,
            "recommended_action": best_action[0],
            "confidence": abs(best_action[1]) / (sum(abs(v) for v in q_values.values()) + 0.001),
            "exploration_rate": self.exploration_rate,
            "total_optimizations": self.total_optimizations
        }
    
    def get_insights(self) -> List[Dict]:
        """Get optimization insights and recommendations."""
        insights = []
        
        # Most frequently selected actions
        action_counts = defaultdict(int)
        for (_, action_type), q_entry in self.q_table.items():
            if q_entry.count > 0:
                action_counts[action_type] += q_entry.count
        
        if action_counts:
            most_common = max(action_counts.items(), key=lambda x: x[1])
            insights.append({
                "type": "frequent_action",
                "action": most_common[0],
                "count": most_common[1]
            })
        
        # High-value patterns
        high_value_patterns = [
            (pattern, q_val.value) 
            for (pattern, _), q_val in self.q_table.items()
            if q_val.value > 0 and q_val.count > 10
        ]
        if high_value_patterns:
            insights.append({
                "type": "high_value_patterns",
                "patterns": high_value_patterns[:5]
            })
        
        return insights
    
    def save_checkpoint(self, path: str):
        """Save optimizer state to checkpoint."""
        import pickle
        
        checkpoint = {
            "q_table": dict(self.q_table),
            "experience_buffer": self.experience_buffer,
            "pattern_cache": self.pattern_cache,
            "exploration_rate": self.exploration_rate,
            "total_optimizations": self.total_optimizations
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Optimizer checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load optimizer state from checkpoint."""
        import pickle
        
        if not __import__('os').path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.q_table = defaultdict(QValue, checkpoint["q_table"])
        self.experience_buffer = checkpoint["experience_buffer"]
        self.pattern_cache = checkpoint["pattern_cache"]
        self.exploration_rate = checkpoint["exploration_rate"]
        self.total_optimizations = checkpoint["total_optimizations"]
        
        logger.info(f"Optimizer checkpoint loaded from {path}")
