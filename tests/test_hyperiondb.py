"""
Tests for HyperionDB
====================

Unit tests for the self-driving database components.
"""

import unittest
import time
import numpy as np
import tempfile
import os
import shutil

from hyperiondb.core.database import HyperionDB, DatabaseConfig
from hyperiondb.core.table import Table, Column
from hyperiondb.core.query import Query, QueryPlan, PlanNode, PlanNodeType
from hyperiondb.optimizer.query_optimizer import LearnedQueryOptimizer
from hyperiondb.optimizer.cost_model import NeuralCostModel
from hyperiondb.optimizer.cardinality import CardinalityEstimator
from hyperiondb.optimizer.join_optimizer import GNNJoinOptimizer
from hyperiondb.indices.learned_index import BTreeLearnedIndex, HashLearnedIndex
from hyperiondb.tuner.auto_tuner import SelfTuningEngine
from hyperiondb.analyzer.anomaly import AnomalyDetector
from hyperiondb.forecaster.workload import WorkloadForecaster
from hyperiondb.executor.adaptive import AdaptiveExecutor


class TestHyperionDB(unittest.TestCase):
    """Test cases for HyperionDB core functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.db_path = tempfile.mkdtemp()
        cls.db = HyperionDB(db_path=cls.db_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.db.shutdown()
        shutil.rmtree(cls.db_path, ignore_errors=True)
    
    def test_create_table(self):
        """Test table creation."""
        table = self.db.create_table(
            "users",
            [("id", "int"), ("name", "str"), ("email", "str")],
            primary_key="id"
        )
        self.assertIsNotNone(table)
        self.assertEqual(table.name, "users")
        self.assertEqual(len(table.columns), 3)
    
    def test_insert_and_select(self):
        """Test insert and select operations."""
        self.db.create_table(
            "products",
            [("id", "int"), ("name", "str"), ("price", "float")],
            primary_key="id"
        )
        
        self.db.insert("products", {"id": 1, "name": "Laptop", "price": 999.99})
        self.db.insert("products", {"id": 2, "name": "Mouse", "price": 29.99})
        
        results = self.db.select("products")
        self.assertEqual(len(results), 2)
        
        results = self.db.select("products", conditions={"id": 1})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Laptop")
    
    def test_update_and_delete(self):
        """Test update and delete operations."""
        self.db.create_table(
            "orders",
            [("id", "int"), ("status", "str")],
            primary_key="id"
        )
        
        self.db.insert("orders", {"id": 1, "status": "pending"})
        self.db.insert("orders", {"id": 2, "status": "pending"})
        
        # Update
        count = self.db.update("orders", {"status": "completed"}, {"id": 1})
        self.assertEqual(count, 1)
        
        # Delete
        count = self.db.delete("orders", {"id": 2})
        self.assertEqual(count, 1)
        
        results = self.db.select("orders")
        self.assertEqual(len(results), 1)
    
    def test_join_query(self):
        """Test join query execution."""
        self.db.create_table("customers", [("id", "int"), ("name", "str")])
        self.db.create_table("orders", [("id", "int"), ("customer_id", "int")])
        
        self.db.insert("customers", {"id": 1, "name": "Alice"})
        self.db.insert("customers", {"id": 2, "name": "Bob"})
        self.db.insert("orders", {"id": 1, "customer_id": 1})
        
        results = self.db.join("customers", "orders", {"customers.id": "orders.customer_id"})
        self.assertGreater(len(results), 0)
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked."""
        metrics = self.db.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.total_queries, 0)


class TestLearnedQueryOptimizer(unittest.TestCase):
    """Test cases for the learned query optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cost_model = NeuralCostModel()
        self.cardinality_estimator = CardinalityEstimator()
        self.join_optimizer = GNNJoinOptimizer()
        self.optimizer = LearnedQueryOptimizer(
            cost_model=self.cost_model,
            cardinality_estimator=self.cardinality_estimator,
            join_optimizer=self.join_optimizer
        )
    
    def test_optimization_state(self):
        """Test optimization state building."""
        from hyperiondb.core.table import Table
        
        table = Table("test", [("id", "int"), ("value", "str")])
        for i in range(100):
            table.insert({"id": i, "value": f"val_{i}"})
        
        state = self.optimizer._build_state(table, {"id": 50})
        self.assertEqual(state.table_name, "test")
        self.assertGreater(state.estimated_rows, 0)
    
    def test_q_value_update(self):
        """Test Q-value updates."""
        from hyperiondb.optimizer.query_optimizer import OptimizationState, OptimizationAction
        
        state = OptimizationState(
            table_name="test",
            available_indexes=[],
            estimated_rows=100,
            condition_complexity=1.0,
            has_joins=False,
            estimated_selectivity=0.5
        )
        
        action = OptimizationAction("seq_scan")
        
        # Update Q-value
        self.optimizer.update(state, action, reward=-100.0)
        
        # Check Q-table
        state_key = self.optimizer._state_to_key(state)
        q_value = self.optimizer.q_table[(state_key, "seq_scan")]
        self.assertLess(q_value.value, 0)  # Negative reward should decrease Q-value
    
    def test_experience_replay(self):
        """Test experience replay functionality."""
        from hyperiondb.optimizer.query_optimizer import OptimizationState, OptimizationAction
        
        state = OptimizationState(
            table_name="test",
            available_indexes=[],
            estimated_rows=100,
            condition_complexity=1.0,
            has_joins=False,
            estimated_selectivity=0.5
        )
        
        action = OptimizationAction("seq_scan")
        
        # Record experience
        self.optimizer.record_experience(state, action, -50.0, None, True)
        self.assertGreater(len(self.optimizer.experience_buffer), 0)
        
        # Replay
        self.optimizer.replay_experiences(batch_size=1)


class TestNeuralCostModel(unittest.TestCase):
    """Test cases for the neural cost model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = NeuralCostModel()
    
    def test_feature_extraction(self):
        """Test feature extraction from query plan."""
        plan = QueryPlan(plan_id="test_plan")
        plan.tables = ["test_table"]
        plan.join_order = ["test_table"]
        
        plan.root = PlanNode(
            node_id=1,
            node_type=PlanNodeType.TABLE_SCAN,
            physical_operator=None,
            table_name="test_table",
            estimated_rows=1000,
            estimated_cost=100.0
        )
        
        features = self.model.extract_features(plan)
        self.assertEqual(len(features), 50)  # Feature vector size
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_prediction(self):
        """Test cost prediction."""
        plan = QueryPlan(plan_id="test_plan")
        plan.root = PlanNode(
            node_id=1,
            node_type=PlanNodeType.TABLE_SCAN,
            physical_operator=None,
            estimated_rows=1000,
            estimated_cost=100.0
        )
        
        cost, breakdown = self.model.predict(plan)
        self.assertGreater(cost, 0)
        self.assertIn("cpu_cost", breakdown)
        self.assertIn("io_cost", breakdown)
    
    def test_training(self):
        """Test model training."""
        history = [
            {"features": np.random.randn(50), "actual_cost": 100.0},
            {"features": np.random.randn(50), "actual_cost": 150.0},
            {"features": np.random.randn(50), "actual_cost": 200.0},
        ]
        
        self.model.train(history, epochs=5)
        self.assertGreater(self.model.epoch_count, 0)


class TestCardinalityEstimator(unittest.TestCase):
    """Test cases for the cardinality estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.estimator = CardinalityEstimator()
    
    def test_column_stats_update(self):
        """Test column statistics update."""
        values = list(range(1000))
        self.estimator.update_column_stats("test", "id", values)
        
        stats = self.estimator.column_stats.get("test", {}).get("id")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.n_distinct, 1000)
    
    def test_cardinality_estimation(self):
        """Test cardinality estimation."""
        values = list(range(1000))
        self.estimator.update_column_stats("test", "id", values)
        
        # Estimate with equality predicate
        estimated = self.estimator.estimate("test", {"id": 500})
        self.assertGreater(estimated, 0)
        self.assertLessEqual(estimated, 1000)
    
    def test_learn_from_feedback(self):
        """Test learning from query feedback."""
        values = list(range(1000))
        self.estimator.update_column_stats("test", "id", values)
        
        # Record feedback
        self.estimator.update_from_feedback("test", {"id": 500}, actual_count=100)
        
        self.assertGreater(len(self.estimator.training_samples), 0)


class TestLearnedIndexes(unittest.TestCase):
    """Test cases for learned index structures."""
    
    def test_btree_learned_index(self):
        """Test B-tree learned index."""
        index = BTreeLearnedIndex("test_idx")
        
        # Train on sorted data
        keys = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        index.train(keys, positions)
        self.assertTrue(index.is_trained)
        
        # Search
        pos = index.search(5)
        self.assertIsNotNone(pos)
    
    def test_hash_learned_index(self):
        """Test hash learned index."""
        index = HashLearnedIndex("hash_idx")
        
        keys = np.array([10, 20, 30, 40, 50])
        positions = np.array([0, 1, 2, 3, 4])
        
        index.train(keys, positions)
        self.assertTrue(index.is_trained)
        
        # Search
        pos = index.search(30)
        self.assertIsNotNone(pos)


class TestSelfTuningEngine(unittest.TestCase):
    """Test cases for the self-tuning engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        from hyperiondb.core.storage import BufferPool, StorageEngine
        
        self.buffer_pool = BufferPool(1000)
        self.storage = StorageEngine("./test_storage")
        self.tuner = SelfTuningEngine(self.buffer_pool, self.storage)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.tuner.collect_metrics()
        self.assertIn("timestamp", metrics)
        self.assertIn("buffer_pool_hit_rate", metrics)
    
    def test_tuning_decisions(self):
        """Test tuning decision making."""
        metrics = self.tuner.collect_metrics()
        metrics["buffer_pool_hit_rate"] = 0.7
        metrics["buffer_pool_utilization"] = 0.95
        
        actions = self.tuner.tune(metrics)
        # May or may not have actions depending on thresholds
    
    def test_pre_tuning(self):
        """Test pre-tuning based on forecast."""
        forecast = {
            "predicted_queries": 5000,
            "predicted_type": "write_heavy"
        }
        
        actions = self.tuner.pre_tune(forecast)
        self.assertIsInstance(actions, list)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for the anomaly detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(sensitivity=0.95)
    
    def test_metric_recording(self):
        """Test metric recording."""
        self.detector.record_metric("query_latency", 50.0)
        self.detector.record_metric("query_latency", 55.0)
        
        self.assertIn("query_latency", self.detector.metrics_history)
        self.assertEqual(len(self.detector.metrics_history["query_latency"]), 2)
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Record normal values
        for i in range(100):
            self.detector.record_metric("test_metric", 50.0 + np.random.randn())
        
        # Check for anomaly
        is_anomaly = self.detector.is_anomaly(200.0, "test_metric")
        self.assertTrue(is_anomaly)
    
    def test_baseline_update(self):
        """Test baseline statistics update."""
        values = [50.0 + np.random.randn() for _ in range(100)]
        for v in values:
            self.detector.record_metric("baseline_test", v)
        
        self.assertIn("baseline_test", self.detector.baselines)
        baseline = self.detector.baselines["baseline_test"]
        self.assertIn("mean", baseline)
        self.assertIn("std", baseline)


class TestWorkloadForecaster(unittest.TestCase):
    """Test cases for the workload forecaster."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = WorkloadForecaster(forecast_horizon=60)
    
    def test_update_and_forecast(self):
        """Test updating with data and forecasting."""
        # Add query history
        history = [
            {"timestamp": time.time() - 3600 + i, "type": "SELECT", "latency": 50.0}
            for i in range(100)
        ]
        self.forecaster.update(history)
        
        # Forecast
        forecast = self.forecaster.forecast()
        
        self.assertIn("predicted_queries", forecast)
        self.assertIn("predicted_type", forecast)
        self.assertIn("confidence", forecast)
    
    def test_evaluation(self):
        """Test forecast evaluation."""
        accuracy = self.forecaster.evaluate()
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)


class TestAdaptiveExecutor(unittest.TestCase):
    """Test cases for the adaptive executor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = AdaptiveExecutor()
    
    def test_execution_stats(self):
        """Test execution statistics tracking."""
        stats = self.executor.get_statistics()
        
        self.assertIn("execution_count", stats)
        self.assertIn("adaptations_made", stats)
        self.assertIn("policies", stats)
    
    def test_plan_modification(self):
        """Test plan modification."""
        plan = QueryPlan(plan_id="test_plan")
        
        modified = self.executor.modify_plan(plan, "use_hash_join")
        self.assertIsNotNone(modified)


if __name__ == "__main__":
    unittest.main()
