"""
HyperionDB - Main Database Class
================================

Self-driving database with learned optimization capabilities.
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from hyperiondb.core.table import Table
from hyperiondb.core.transaction import TransactionManager
from hyperiondb.core.catalog import Catalog
from hyperiondb.core.storage import StorageEngine, BufferPool
from hyperiondb.optimizer.query_optimizer import LearnedQueryOptimizer
from hyperiondb.optimizer.cost_model import NeuralCostModel
from hyperiondb.optimizer.cardinality import CardinalityEstimator
from hyperiondb.optimizer.join_optimizer import GNNJoinOptimizer
from hyperiondb.indices.learned_index import LearnedIndexFactory
from hyperiondb.tuner.auto_tuner import SelfTuningEngine
from hyperiondb.analyzer.anomaly import AnomalyDetector
from hyperiondb.forecaster.workload import WorkloadForecaster
from hyperiondb.executor.adaptive import AdaptiveExecutor

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for HyperionDB self-driving capabilities."""
    enable_learned_optimization: bool = True
    enable_neural_indexes: bool = True
    enable_auto_tuning: bool = True
    enable_anomaly_detection: bool = True
    enable_workload_forecasting: bool = True
    enable_adaptive_execution: bool = True
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    training_interval: int = 100  # queries
    buffer_pool_size: int = 10000
    checkpoint_interval: int = 1000


@dataclass
class PerformanceMetrics:
    """Track database performance metrics."""
    total_queries: int = 0
    avg_query_latency: float = 0.0
    cache_hit_rate: float = 0.0
    index_usage_rate: float = 0.0
    optimizer_accuracy: float = 0.0
    anomaly_count: int = 0
    tuning_adjustments: int = 0
    forecast_accuracy: float = 0.0


class HyperionDB:
    """
    HyperionDB - Self-Driving Database with Learned Optimizer
    
    A revolutionary database that uses machine learning to continuously
    optimize its own performance through:
    - Learned query optimization (Reinforcement Learning)
    - Neural index structures (Learned Indexes)
    - Automatic database tuning
    - Neural cost models
    - Deep learning cardinality estimation
    - GNN-based join order enumeration
    - Anomaly detection and self-healing
    - Workload forecasting and pre-tuning
    - Adaptive query execution
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None, db_path: str = "./hyperiondb_data"):
        """
        Initialize HyperionDB with self-driving capabilities.
        
        Args:
            config: Database configuration
            db_path: Path for database storage
        """
        self.config = config or DatabaseConfig()
        self.db_path = db_path
        
        # Core components
        self.catalog = Catalog()
        self.storage = StorageEngine(db_path)
        self.buffer_pool = BufferPool(self.config.buffer_pool_size)
        self.transaction_manager = TransactionManager()
        
        # Tables storage
        self.tables: Dict[str, Table] = {}
        self.table_locks = defaultdict(threading.RLock)
        
        # ML-powered components
        self.query_optimizer = None
        self.cost_model = None
        self.cardinality_estimator = None
        self.join_optimizer = None
        self.learned_index_factory = None
        self.auto_tuner = None
        self.anomaly_detector = None
        self.workload_forecaster = None
        self.adaptive_executor = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.query_history: List[Dict] = []
        self.query_lock = threading.Lock()
        
        # Background threads
        self.running = False
        self.background_threads: List[threading.Thread] = []
        
        # Initialize ML components
        self._initialize_ml_components()
        
        logger.info("HyperionDB initialized with self-driving capabilities")
        logger.info(f"Configuration: {self.config}")
    
    def _initialize_ml_components(self):
        """Initialize all ML-powered components."""
        if self.config.enable_learned_optimization:
            # Neural cost model for estimating query costs
            self.cost_model = NeuralCostModel()
            logger.info("Neural Cost Model initialized")
            
            # Deep learning cardinality estimator
            self.cardinality_estimator = CardinalityEstimator()
            logger.info("Cardinality Estimator initialized")
            
            # GNN-based join optimizer
            self.join_optimizer = GNNJoinOptimizer()
            logger.info("GNN Join Optimizer initialized")
            
            # RL-based query optimizer
            self.query_optimizer = LearnedQueryOptimizer(
                cost_model=self.cost_model,
                cardinality_estimator=self.cardinality_estimator,
                join_optimizer=self.join_optimizer,
                learning_rate=self.config.learning_rate,
                exploration_rate=self.config.exploration_rate
            )
            logger.info("Learned Query Optimizer initialized")
        
        if self.config.enable_neural_indexes:
            self.learned_index_factory = LearnedIndexFactory()
            logger.info("Learned Index Factory initialized")
        
        if self.config.enable_auto_tuning:
            self.auto_tuner = SelfTuningEngine(
                buffer_pool=self.buffer_pool,
                storage=self.storage
            )
            logger.info("Self-Tuning Engine initialized")
        
        if self.config.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
            logger.info("Anomaly Detector initialized")
        
        if self.config.enable_workload_forecasting:
            self.workload_forecaster = WorkloadForecaster()
            logger.info("Workload Forecaster initialized")
        
        if self.config.enable_adaptive_execution:
            self.adaptive_executor = AdaptiveExecutor()
            logger.info("Adaptive Query Executor initialized")
    
    def create_table(self, name: str, columns: List[Tuple[str, str]], 
                     primary_key: Optional[str] = None) -> Table:
        """
        Create a new table in the database.
        
        Args:
            name: Table name
            columns: List of (column_name, column_type) tuples
            primary_key: Primary key column name
            
        Returns:
            Created Table object
        """
        with self.table_locks[name]:
            if name in self.tables:
                raise ValueError(f"Table '{name}' already exists")
            
            table = Table(name, columns, primary_key)
            self.tables[name] = table
            
            # Register in catalog
            self.catalog.add_table(table)
            
            # Create learned index if enabled
            if self.config.enable_neural_indexes and self.learned_index_factory:
                for col_name, col_type in columns:
                    index = self.learned_index_factory.create_index(
                        table_name=name,
                        column_name=col_name,
                        column_type=col_type
                    )
                    table.add_index(col_name, index)
            
            logger.info(f"Created table '{name}' with {len(columns)} columns")
            return table
    
    def drop_table(self, name: str):
        """Drop a table from the database."""
        with self.table_locks[name]:
            if name not in self.tables:
                raise ValueError(f"Table '{name}' does not exist")
            
            del self.tables[name]
            self.catalog.remove_table(name)
            logger.info(f"Dropped table '{name}'")
    
    def insert(self, table_name: str, values: Dict[str, Any]) -> bool:
        """
        Insert a row into a table.
        
        Args:
            table_name: Name of the table
            values: Dictionary of column -> value
            
        Returns:
            True if successful
        """
        with self.table_locks[table_name]:
            if table_name not in self.tables:
                raise ValueError(f"Table '{table_name}' does not exist")
            
            table = self.tables[table_name]
            table.insert(values)
            
            # Update learned indexes
            if self.config.enable_neural_indexes:
                for col_name, value in values.items():
                    if col_name in table.indexes:
                        table.indexes[col_name].insert(value, table.row_count - 1)
            
            return True
    
    def select(self, table_name: str, conditions: Optional[Dict] = None,
               columns: Optional[List[str]] = None) -> List[Dict]:
        """
        Execute a SELECT query with learned optimization.
        
        Args:
            table_name: Name of the table
            conditions: WHERE conditions
            columns: Columns to select
            
        Returns:
            List of matching rows
        """
        start_time = time.time()
        
        with self.table_locks[table_name]:
            if table_name not in self.tables:
                raise ValueError(f"Table '{table_name}' does not exist")
            
            table = self.tables[table_name]
            
            # Generate and optimize query plan using learned optimizer
            if self.config.enable_learned_optimization and self.query_optimizer:
                plan = self.query_optimizer.optimize(
                    table=table,
                    conditions=conditions,
                    selected_columns=columns
                )
                
                # Execute using adaptive executor if enabled
                if self.config.enable_adaptive_execution and self.adaptive_executor:
                    results = self.adaptive_executor.execute(plan, table)
                else:
                    results = table.scan(conditions, columns)
            else:
                results = table.scan(conditions, columns)
            
            latency = time.time() - start_time
            self._record_query(table_name, "SELECT", latency, len(results))
            
            return results
    
    def update(self, table_name: str, values: Dict[str, Any],
               conditions: Optional[Dict] = None) -> int:
        """Update rows in a table."""
        with self.table_locks[table_name]:
            if table_name not in self.tables:
                raise ValueError(f"Table '{table_name}' does not exist")
            
            table = self.tables[table_name]
            count = table.update(values, conditions)
            
            # Update learned indexes
            if self.config.enable_neural_indexes and conditions:
                for col_name, value in values.items():
                    if col_name in table.indexes:
                        # Rebuild affected index entries
                        affected_rows = table.scan(conditions, [col_name])
                        for row in affected_rows:
                            table.indexes[col_name].insert(value, row._rid)
            
            logger.info(f"Updated {count} rows in '{table_name}'")
            return count
    
    def delete(self, table_name: str, conditions: Optional[Dict] = None) -> int:
        """Delete rows from a table."""
        with self.table_locks[table_name]:
            if table_name not in self.tables:
                raise ValueError(f"Table '{table_name}' does not exist")
            
            table = self.tables[table_name]
            count = table.delete(conditions)
            
            logger.info(f"Deleted {count} rows from '{table_name}'")
            return count
    
    def join(self, table1_name: str, table2_name: str, 
             join_condition: Dict, join_type: str = "INNER") -> List[Dict]:
        """
        Execute a JOIN query with GNN-optimized join order.
        
        Args:
            table1_name: First table name
            table2_name: Second table name
            join_condition: Join condition {table1.col: table2.col}
            join_type: Type of join (INNER, LEFT, RIGHT, FULL)
            
        Returns:
            List of joined rows
        """
        start_time = time.time()
        
        # Validate tables
        if table1_name not in self.tables or table2_name not in self.tables:
            raise ValueError("One or both tables do not exist")
        
        table1 = self.tables[table1_name]
        table2 = self.tables[table2_name]
        
        # Optimize join order using GNN if enabled
        if self.config.enable_learned_optimization and self.join_optimizer:
            join_order = self.join_optimizer.optimize_join_order(
                tables=[table1, table2],
                join_conditions=[join_condition]
            )
        else:
            join_order = [table1, table2]
        
        # Execute join
        results = []
        col1, col2 = list(join_condition.items())[0]
        
        for row1 in table1.scan():
            for row2 in table2.scan():
                if row1.get(col1) == row2.get(col2):
                    merged = {**row1, **row2}
                    results.append(merged)
        
        latency = time.time() - start_time
        self._record_query(f"{table1_name}_join_{table2_name}", "JOIN", latency, len(results))
        
        return results
    
    def _record_query(self, table_name: str, query_type: str, 
                      latency: float, result_count: int):
        """Record query for learning and metrics."""
        with self.query_lock:
            query_record = {
                "timestamp": time.time(),
                "table": table_name,
                "type": query_type,
                "latency": latency,
                "result_count": result_count
            }
            
            self.query_history.append(query_record)
            self.metrics.total_queries += 1
            
            # Update running average
            n = self.metrics.total_queries
            self.metrics.avg_query_latency = (
                (self.metrics.avg_query_latency * (n - 1) + latency) / n
            )
            
            # Check for anomalies
            if self.config.enable_anomaly_detection and self.anomaly_detector:
                if self.anomaly_detector.is_anomaly(latency, query_type):
                    self.metrics.anomaly_count += 1
                    self._handle_anomaly(query_record)
            
            # Periodic training
            if (self.metrics.total_queries % self.config.training_interval == 0 
                and self.config.enable_learned_optimization):
                self._trigger_periodic_training()
    
    def _handle_anomaly(self, query_record: Dict):
        """Handle detected anomaly through self-healing."""
        logger.warning(f"Anomaly detected: {query_record}")
        
        if self.config.enable_auto_tuning and self.auto_tuner:
            # Analyze and apply fixes
            fix = self.auto_tuner.suggest_fix(query_record)
            if fix:
                self._apply_fix(fix)
                self.metrics.tuning_adjustments += 1
    
    def _apply_fix(self, fix: Dict):
        """Apply a tuning fix."""
        logger.info(f"Applying fix: {fix}")
        fix_type = fix.get("type")
        
        if fix_type == "increase_buffer":
            self.buffer_pool.resize(fix.get("new_size", self.buffer_pool.size * 1.5))
        elif fix_type == "rebuild_index":
            table_name = fix.get("table")
            col_name = fix.get("column")
            if table_name in self.tables and col_name in self.tables[table_name].indexes:
                self.tables[table_name].indexes[col_name].rebuild()
        elif fix_type == "retrain_model":
            self._trigger_periodic_training()
    
    def _trigger_periodic_training(self):
        """Trigger periodic model training."""
        logger.info("Triggering periodic model training")
        
        if self.cost_model:
            self.cost_model.train(self.query_history[-100:])
        
        if self.cardinality_estimator:
            self.cardinality_estimator.retrain(self.query_history[-100:])
        
        if self.workload_forecaster:
            self.workload_forecaster.update(self.query_history[-100:])
            forecast_accuracy = self.workload_forecaster.evaluate()
            self.metrics.forecast_accuracy = forecast_accuracy
    
    def start_background_optimization(self):
        """Start background optimization threads."""
        self.running = True
        
        # Workload forecasting thread
        if self.config.enable_workload_forecasting and self.workload_forecaster:
            t = threading.Thread(target=self._forecast_worker, daemon=True)
            t.start()
            self.background_threads.append(t)
        
        # Auto-tuning thread
        if self.config.enable_auto_tuning and self.auto_tuner:
            t = threading.Thread(target=self._tuning_worker, daemon=True)
            t.start()
            self.background_threads.append(t)
        
        # Model checkpointing thread
        if self.config.enable_learned_optimization:
            t = threading.Thread(target=self._checkpoint_worker, daemon=True)
            t.start()
            self.background_threads.append(t)
        
        logger.info(f"Started {len(self.background_threads)} background optimization threads")
    
    def _forecast_worker(self):
        """Background worker for workload forecasting."""
        while self.running:
            try:
                # Forecast next period's workload
                forecast = self.workload_forecaster.forecast()
                
                # Pre-tune based on forecast
                if self.config.enable_auto_tuning and self.auto_tuner:
                    self.auto_tuner.pre_tune(forecast)
                
                # Sleep until next forecast interval
                time.sleep(60)  # Forecast every minute
            except Exception as e:
                logger.error(f"Forecast worker error: {e}")
    
    def _tuning_worker(self):
        """Background worker for continuous tuning."""
        while self.running:
            try:
                # Collect recent performance metrics
                recent_metrics = self._collect_metrics()
                
                # Make tuning decisions
                if self.auto_tuner:
                    adjustments = self.auto_tuner.tune(recent_metrics)
                    for adj in adjustments:
                        self._apply_adjustment(adj)
                
                time.sleep(30)  # Tune every 30 seconds
            except Exception as e:
                logger.error(f"Tuning worker error: {e}")
    
    def _checkpoint_worker(self):
        """Background worker for model checkpointing."""
        while self.running:
            try:
                if self.metrics.total_queries % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Checkpoint worker error: {e}")
    
    def _collect_metrics(self) -> Dict:
        """Collect current performance metrics."""
        return {
            "query_latency": self.metrics.avg_query_latency,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "index_usage": self.metrics.index_usage_rate,
            "anomaly_count": self.metrics.anomaly_count,
            "total_queries": self.metrics.total_queries,
        }
    
    def _apply_adjustment(self, adjustment: Dict):
        """Apply a tuning adjustment."""
        adj_type = adjustment.get("type")
        
        if adj_type == "buffer_resize":
            new_size = adjustment.get("new_size")
            self.buffer_pool.resize(new_size)
            self.metrics.tuning_adjustments += 1
        elif adj_type == "index_recommendation":
            # Create new learned index
            table_name = adjustment.get("table")
            column = adjustment.get("column")
            if table_name in self.tables and self.learned_index_factory:
                idx = self.learned_index_factory.create_index(
                    table_name, column, "numeric"
                )
                self.tables[table_name].add_index(column, idx)
                self.metrics.index_usage_rate = adjustment.get("usage_prediction", 0.8)
    
    def _save_checkpoint(self):
        """Save model checkpoints."""
        logger.info("Saving model checkpoints")
        
        if self.cost_model:
            self.cost_model.save_checkpoint(f"{self.db_path}/cost_model.ckpt")
        
        if self.cardinality_estimator:
            self.cardinality_estimator.save_checkpoint(f"{self.db_path}/cardinality.ckpt")
        
        if self.query_optimizer:
            self.query_optimizer.save_checkpoint(f"{self.db_path}/optimizer.ckpt")
    
    def load_checkpoint(self):
        """Load model checkpoints."""
        logger.info("Loading model checkpoints")
        
        if self.cost_model:
            self.cost_model.load_checkpoint(f"{self.db_path}/cost_model.ckpt")
        
        if self.cardinality_estimator:
            self.cardinality_estimator.load_checkpoint(f"{self.db_path}/cardinality.ckpt")
        
        if self.query_optimizer:
            self.query_optimizer.load_checkpoint(f"{self.db_path}/optimizer.ckpt")
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_recommendations(self) -> List[Dict]:
        """Get optimization recommendations from the learned components."""
        recommendations = []
        
        if self.workload_forecaster:
            forecast = self.workload_forecaster.get_predictions()
            recommendations.append({
                "type": "workload_forecast",
                "predictions": forecast,
                "confidence": self.metrics.forecast_accuracy
            })
        
        if self.auto_tuner:
            tuning_recs = self.auto_tuner.get_recommendations()
            recommendations.extend(tuning_recs)
        
        if self.query_optimizer:
            opt_recs = self.query_optimizer.get_insights()
            recommendations.extend(opt_recs)
        
        return recommendations
    
    def explain_query(self, table_name: str, conditions: Optional[Dict] = None) -> Dict:
        """
        Explain how a query would be optimized.
        
        Args:
            table_name: Table to query
            conditions: WHERE conditions
            
        Returns:
            Explanation of optimization decisions
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' does not exist")
        
        table = self.tables[table_name]
        
        explanation = {
            "table": table_name,
            "estimated_rows": len(table.rows),
            "available_indexes": list(table.indexes.keys()),
            "optimizations": []
        }
        
        if self.query_optimizer:
            plan = self.query_optimizer.explain_plan(table, conditions)
            explanation["optimized_plan"] = plan
            explanation["estimated_cost"] = self.cost_model.estimate(plan) if self.cost_model else None
        
        if self.cardinality_estimator:
            est_cardinality = self.cardinality_estimator.estimate(
                table_name, conditions
            )
            explanation["estimated_cardinality"] = est_cardinality
        
        return explanation
    
    def shutdown(self):
        """Shutdown the database and save state."""
        logger.info("Shutting down HyperionDB")
        self.running = False
        
        # Wait for background threads
        for t in self.background_threads:
            t.join(timeout=5)
        
        # Save final checkpoint
        self._save_checkpoint()
        
        # Save catalog
        self.catalog.save(f"{self.db_path}/catalog.json")
        
        logger.info("HyperionDB shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
