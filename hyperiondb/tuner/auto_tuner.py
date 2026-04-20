"""
Self-Tuning Engine for HyperionDB
=================================

Automatic database tuning through continuous monitoring and adjustment
of database parameters based on workload characteristics and performance.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TuningKnob:
    """Represents a tunable database parameter."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    adjustment_cost: float = 1.0  # Cost to change this knob
    
    def clamp(self, value: float) -> float:
        """Clamp value to valid range."""
        return max(self.min_value, min(self.max_value, value))


@dataclass
class TuningAction:
    """Represents a tuning action."""
    type: str  # knob_change, index_create, index_drop, etc.
    target: str  # What to change
    new_value: Any
    reason: str
    expected_improvement: float


class SelfTuningEngine:
    """
    Self-Tuning Database Engine.
    
    Automatically tunes database configuration based on workload
    characteristics without requiring manual knob tuning.
    
    Features:
    - Continuous performance monitoring
    - Workload pattern recognition
    - Automated configuration adjustment
    - Index recommendation
    - Resource allocation optimization
    - Self-healing capabilities
    """
    
    def __init__(self, buffer_pool: Any, storage: Any):
        """
        Initialize the self-tuning engine.
        
        Args:
            buffer_pool: Database buffer pool
            storage: Storage engine
        """
        self.buffer_pool = buffer_pool
        self.storage = storage
        
        # Knob definitions (knob-free means these are auto-tuned)
        self.knobs: Dict[str, TuningKnob] = {
            "buffer_pool_size": TuningKnob(
                name="buffer_pool_size",
                current_value=10000,
                min_value=1000,
                max_value=100000,
                default_value=10000
            ),
            "checkpoint_interval": TuningKnob(
                name="checkpoint_interval",
                current_value=300,
                min_value=60,
                max_value=3600,
                default_value=300
            ),
            "sort_buffer_size": TuningKnob(
                name="sort_buffer_size",
                current_value=1024,
                min_value=64,
                max_value=32768,
                default_value=1024
            ),
            "connection_pool_size": TuningKnob(
                name="connection_pool_size",
                current_value=100,
                min_value=10,
                max_value=1000,
                default_value=100
            ),
            "query_timeout": TuningKnob(
                name="query_timeout",
                current_value=30,
                min_value=5,
                max_value=300,
                default_value=30
            )
        }
        
        # Performance metrics history
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Tuning decisions history
        self.tuning_history: List[Dict] = []
        
        # Index recommendations
        self.index_recommendations: List[Dict] = []
        
        # Workload pattern
        self.workload_pattern = "unknown"
        
        # Statistics
        self.tuning_count = 0
        self.last_tuning_time = 0
        self.tuning_interval = 60  # seconds
        
        logger.info("Self-Tuning Engine initialized")
    
    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "timestamp": time.time(),
            "buffer_pool_hit_rate": self.buffer_pool.hit_rate if hasattr(self.buffer_pool, 'hit_rate') else 0.0,
            "buffer_pool_size": len(self.buffer_pool.pages) if hasattr(self.buffer_pool, 'pages') else 0,
            "buffer_pool_utilization": len(self.buffer_pool.pages) / max(1, self.buffer_pool.pool_size) if hasattr(self.buffer_pool, 'pool_size') else 0,
            "io_operations": self.storage.io_count if hasattr(self.storage, 'io_count') else 0,
            "bytes_read": self.storage.bytes_read if hasattr(self.storage, 'bytes_read') else 0,
            "bytes_written": self.storage.bytes_written if hasattr(self.storage, 'bytes_written') else 0,
        }
        
        # Add tuning-specific metrics
        for name, knob in self.knobs.items():
            metrics[f"knob_{name}"] = knob.current_value
        
        return metrics
    
    def analyze_workload(self, metrics: Dict[str, float]) -> str:
        """
        Analyze workload pattern based on metrics.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Workload pattern: "read_heavy", "write_heavy", "mixed", "batch"
        """
        # Analyze based on I/O patterns
        bytes_read = metrics.get("bytes_read", 0)
        bytes_written = metrics.get("bytes_written", 0)
        total_io = bytes_read + bytes_written
        
        if total_io == 0:
            return "mixed"
        
        read_ratio = bytes_read / total_io
        
        if read_ratio > 0.8:
            return "read_heavy"
        elif read_ratio < 0.2:
            return "write_heavy"
        elif metrics.get("batch_processing", 0) > 0.5:
            return "batch"
        else:
            return "mixed"
    
    def tune(self, current_metrics: Dict[str, float]) -> List[TuningAction]:
        """
        Determine tuning actions based on current metrics.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            List of tuning actions to apply
        """
        actions = []
        
        # Store metrics history
        self.metrics_history.append(current_metrics)
        
        # Analyze workload
        self.workload_pattern = self.analyze_workload(current_metrics)
        
        # Tune buffer pool
        buffer_action = self._tune_buffer_pool(current_metrics)
        if buffer_action:
            actions.append(buffer_action)
        
        # Tune checkpoints
        checkpoint_action = self._tune_checkpoint_interval(current_metrics)
        if checkpoint_action:
            actions.append(checkpoint_action)
        
        # Tune connections
        connection_action = self._tune_connection_pool(current_metrics)
        if connection_action:
            actions.append(connection_action)
        
        # Update workload pattern
        self._update_workload_model(current_metrics)
        
        return actions
    
    def _tune_buffer_pool(self, metrics: Dict[str, float]) -> Optional[TuningAction]:
        """Tune buffer pool size based on hit rate."""
        knob = self.knobs["buffer_pool_size"]
        hit_rate = metrics.get("buffer_pool_hit_rate", 0.5)
        utilization = metrics.get("buffer_pool_utilization", 0.5)
        
        new_value = knob.current_value
        
        # If hit rate is low and buffer is well-utilized, increase size
        if hit_rate < 0.8 and utilization > 0.9:
            new_value = knob.clamp(knob.current_value * 1.2)
            reason = f"Low hit rate ({hit_rate:.2f}) and high utilization ({utilization:.2f})"
        # If hit rate is high and utilization is low, decrease size
        elif hit_rate > 0.95 and utilization < 0.5:
            new_value = knob.clamp(knob.current_value * 0.8)
            reason = f"High hit rate ({hit_rate:.2f}) and low utilization ({utilization:.2f})"
        
        if new_value != knob.current_value:
            knob.current_value = new_value
            return TuningAction(
                type="knob_change",
                target="buffer_pool_size",
                new_value=new_value,
                reason=reason,
                expected_improvement=0.05
            )
        
        return None
    
    def _tune_checkpoint_interval(self, metrics: Dict[str, float]) -> Optional[TuningAction]:
        """Tune checkpoint interval based on write patterns."""
        knob = self.knobs["checkpoint_interval"]
        
        # In batch mode, increase checkpoint interval
        if self.workload_pattern == "batch":
            new_value = knob.clamp(knob.current_value * 1.5)
            reason = "Batch workload - reducing checkpoint frequency"
        # In write-heavy mode, decrease interval
        elif self.workload_pattern == "write_heavy":
            new_value = knob.clamp(knob.current_value * 0.7)
            reason = "Write-heavy workload - more frequent checkpoints"
        else:
            new_value = knob.current_value
            reason = None
        
        if new_value != knob.current_value:
            knob.current_value = new_value
            return TuningAction(
                type="knob_change",
                target="checkpoint_interval",
                new_value=new_value,
                reason=reason,
                expected_improvement=0.03
            )
        
        return None
    
    def _tune_connection_pool(self, metrics: Dict[str, float]) -> Optional[TuningAction]:
        """Tune connection pool size."""
        knob = self.knobs["connection_pool_size"]
        
        # Simplified tuning
        active_connections = metrics.get("active_connections", 50)
        
        if active_connections > knob.current_value * 0.9:
            new_value = knob.clamp(knob.current_value * 1.2)
            reason = "High connection utilization"
        elif active_connections < knob.current_value * 0.3:
            new_value = knob.clamp(knob.current_value * 0.8)
            reason = "Low connection utilization"
        else:
            return None
        
        knob.current_value = new_value
        return TuningAction(
            type="knob_change",
            target="connection_pool_size",
            new_value=new_value,
            reason=reason,
            expected_improvement=0.02
        )
    
    def _update_workload_model(self, metrics: Dict[str, float]):
        """Update internal workload model."""
        if len(self.metrics_history) < 10:
            return
        
        # Detect workload changes
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate variance in key metrics
        hit_rates = [m.get("buffer_pool_hit_rate", 0) for m in recent_metrics]
        variance = np.var(hit_rates)
        
        if variance > 0.1:
            self.workload_pattern = "variable"
    
    def pre_tune(self, forecast: Dict) -> List[TuningAction]:
        """
        Pre-tune based on workload forecast.
        
        Args:
            forecast: Predicted workload for next period
            
        Returns:
            List of pre-tuning actions
        """
        actions = []
        
        predicted_load = forecast.get("predicted_queries", 0)
        predicted_type = forecast.get("predicted_type", "mixed")
        
        # Pre-allocate resources based on forecast
        if predicted_load > 10000:
            buffer_action = self._pre_tune_buffer_pool(predicted_load)
            if buffer_action:
                actions.append(buffer_action)
        
        if predicted_type == "write_heavy":
            checkpoint_action = TuningAction(
                type="knob_change",
                target="checkpoint_interval",
                new_value=60,  # More frequent
                reason="Forecasted write-heavy workload",
                expected_improvement=0.05
            )
            self.knobs["checkpoint_interval"].current_value = 60
            actions.append(checkpoint_action)
        
        return actions
    
    def _pre_tune_buffer_pool(self, predicted_load: int) -> Optional[TuningAction]:
        """Pre-allocate buffer pool based on predicted load."""
        knob = self.knobs["buffer_pool_size"]
        
        # Scale buffer pool with predicted load
        scale_factor = min(predicted_load / 10000, 2.0)
        new_value = knob.clamp(knob.default_value * scale_factor)
        
        if new_value != knob.current_value:
            knob.current_value = new_value
            return TuningAction(
                type="knob_change",
                target="buffer_pool_size",
                new_value=new_value,
                reason=f"Pre-tuning for predicted load: {predicted_load}",
                expected_improvement=0.1
            )
        
        return None
    
    def suggest_fix(self, problem: Dict) -> Optional[Dict]:
        """
        Suggest a fix for a detected problem.
        
        Args:
            problem: Problem description
            
        Returns:
            Fix suggestion or None
        """
        problem_type = problem.get("type", "")
        
        if problem_type == "high_latency":
            return {
                "type": "increase_buffer",
                "new_size": int(self.knobs["buffer_pool_size"].current_value * 1.5)
            }
        elif problem_type == "index_scan_slow":
            return {
                "type": "rebuild_index",
                "table": problem.get("table"),
                "column": problem.get("column")
            }
        elif problem_type == "model_inaccurate":
            return {"type": "retrain_model"}
        
        return None
    
    def recommend_indexes(self, query_history: List[Dict]) -> List[Dict]:
        """
        Recommend indexes based on query patterns.
        
        Args:
            query_history: Recent query history
            
        Returns:
            List of index recommendations
        """
        recommendations = []
        
        # Analyze filter predicates
        filter_columns = {}
        
        for query in query_history:
            if query.get("type") == "SELECT":
                conditions = query.get("conditions", {})
                for col in conditions.keys():
                    filter_columns[col] = filter_columns.get(col, 0) + 1
        
        # Recommend indexes for frequently filtered columns
        for col, count in filter_columns.items():
            if count >= 5:  # Threshold
                recommendations.append({
                    "column": col,
                    "frequency": count,
                    "index_type": "learned",
                    "expected_improvement": 0.2
                })
        
        self.index_recommendations = recommendations
        return recommendations
    
    def get_recommendations(self) -> List[Dict]:
        """Get current tuning recommendations."""
        recommendations = []
        
        # Knob recommendations
        for name, knob in self.knobs.items():
            if abs(knob.current_value - knob.default_value) > 0.1 * knob.default_value:
                recommendations.append({
                    "type": "knob",
                    "knob": name,
                    "current": knob.current_value,
                    "default": knob.default_value,
                    "reason": "Deviation from default"
                })
        
        # Index recommendations
        recommendations.extend(self.index_recommendations)
        
        return recommendations
    
    def get_knob_values(self) -> Dict[str, float]:
        """Get current knob values."""
        return {name: knob.current_value for name, knob in self.knobs.items()}
    
    def reset_knobs(self):
        """Reset all knobs to default values."""
        for knob in self.knobs.values():
            knob.current_value = knob.default_value
        logger.info("All knobs reset to default values")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tuning engine statistics."""
        return {
            "tuning_count": self.tuning_count,
            "last_tuning_time": self.last_tuning_time,
            "workload_pattern": self.workload_pattern,
            "knob_values": self.get_knob_values(),
            "recommendations_count": len(self.index_recommendations),
            "metrics_history_size": len(self.metrics_history)
        }
