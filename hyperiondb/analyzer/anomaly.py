"""
Anomaly Detector for HyperionDB
===============================

Machine learning-based anomaly detection for identifying unusual
database behavior and triggering self-healing actions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    timestamp: float
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    metrics: Dict[str, float]
    recommended_action: Optional[str] = None
    resolved: bool = False


class AnomalyDetector:
    """
    Anomaly Detection System for HyperionDB.
    
    Uses statistical and machine learning methods to detect unusual
    database behavior and trigger self-healing actions.
    
    Detection Methods:
    - Statistical process control (SPC)
    - Time series anomaly detection
    - Pattern matching
    - Threshold-based detection
    
    Features:
    - Continuous monitoring
    - Adaptive thresholds
    - Root cause analysis
    - Self-healing integration
    """
    
    def __init__(self, sensitivity: float = 0.95):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1), higher = more sensitive
        """
        self.sensitivity = sensitivity
        
        # Metrics history for each metric type
        self.metrics_history: Dict[str, deque] = {}
        self.max_history = 1000
        
        # Statistical models
        self.baselines: Dict[str, Dict] = {}
        
        # Detected anomalies
        self.anomalies: List[Anomaly] = []
        self.anomaly_count = 0
        
        # Thresholds (learned or configured)
        self.thresholds: Dict[str, Tuple[float, float]] = {
            "query_latency": (0, 1000),  # (min, max) in ms
            "error_rate": (0, 0.05),  # (min, max) as fraction
            "cpu_usage": (0, 0.9),  # (min, max) as fraction
            "memory_usage": (0, 0.9),
            "io_wait": (0, 0.3),
            "cache_hit_rate": (0.5, 1.0),  # min, max
        }
        
        # Pattern detection
        self.patterns: Dict[str, List] = {}
        
        logger.info(f"Anomaly Detector initialized with sensitivity={sensitivity}")
    
    def record_metric(self, metric_name: str, value: float, 
                      timestamp: Optional[float] = None):
        """
        Record a metric value for analysis.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.max_history)
        
        self.metrics_history[metric_name].append({
            "timestamp": timestamp,
            "value": value
        })
        
        # Update baseline periodically
        if len(self.metrics_history[metric_name]) % 100 == 0:
            self._update_baseline(metric_name)
    
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric."""
        if metric_name not in self.metrics_history:
            return
        
        values = [m["value"] for m in self.metrics_history[metric_name]]
        
        if len(values) < 10:
            return
        
        self.baselines[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }
    
    def is_anomaly(self, value: float, metric_type: str = "query_latency") -> bool:
        """
        Check if a value is anomalous.
        
        Args:
            value: Value to check
            metric_type: Type of metric
            
        Returns:
            True if anomalous
        """
        # Check against threshold
        if metric_type in self.thresholds:
            min_val, max_val = self.thresholds[metric_type]
            if value < min_val or value > max_val:
                return True
        
        # Check against statistical baseline
        if metric_type in self.baselines:
            baseline = self.baselines[metric_type]
            
            # Z-score based detection
            z_score = (value - baseline["mean"]) / max(baseline["std"], 0.001)
            
            # Adjust threshold based on sensitivity
            threshold = 3.0 / self.sensitivity
            
            if abs(z_score) > threshold:
                return True
        
        # Check for pattern anomalies
        if self._check_pattern_anomaly(metric_type, value):
            return True
        
        return False
    
    def _check_pattern_anomaly(self, metric_type: str, value: float) -> bool:
        """Check for pattern-based anomalies."""
        if metric_type not in self.metrics_history:
            return False
        
        history = list(self.metrics_history[metric_type])
        if len(history) < 10:
            return False
        
        # Check for sudden changes
        recent_values = [m["value"] for m in history[-5:]]
        
        # Calculate trend
        if len(recent_values) >= 2:
            trend = recent_values[-1] - recent_values[0]
            
            # Detect sudden spikes
            if len(recent_values) >= 3:
                max_val = max(recent_values)
                min_val = min(recent_values)
                
                if max_val / max(min_val, 0.001) > 2.0:  # 2x difference
                    return True
        
        return False
    
    def detect_anomalies(self) -> List[Anomaly]:
        """
        Perform comprehensive anomaly detection on all metrics.
        
        Returns:
            List of detected anomalies
        """
        detected_anomalies = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 5:
                continue
            
            recent = list(history)[-5:]
            
            for entry in recent:
                if self.is_anomaly(entry["value"], metric_name):
                    anomaly = self._create_anomaly(
                        metric_type=metric_name,
                        value=entry["value"],
                        timestamp=entry["timestamp"]
                    )
                    if anomaly:
                        detected_anomalies.append(anomaly)
        
        self.anomalies.extend(detected_anomalies)
        self.anomaly_count += len(detected_anomalies)
        
        return detected_anomalies
    
    def _create_anomaly(self, metric_type: str, value: float,
                        timestamp: float) -> Optional[Anomaly]:
        """Create an anomaly record."""
        # Check if already detected similar anomaly
        for anomaly in self.anomalies[-10:]:
            if (anomaly.anomaly_type == metric_type and 
                timestamp - anomaly.timestamp < 60):  # Within 1 minute
                return None  # Already detected
        
        # Determine severity
        severity = self._determine_severity(metric_type, value)
        
        # Generate description
        description = self._generate_description(metric_type, value)
        
        # Determine recommended action
        action = self._get_recommended_action(metric_type, value)
        
        anomaly = Anomaly(
            anomaly_id=f"anomaly_{self.anomaly_count}_{int(timestamp)}",
            timestamp=timestamp,
            anomaly_type=metric_type,
            severity=severity,
            description=description,
            metrics={metric_type: value},
            recommended_action=action
        )
        
        return anomaly
    
    def _determine_severity(self, metric_type: str, value: float) -> str:
        """Determine anomaly severity."""
        if metric_type not in self.baselines:
            return "medium"
        
        baseline = self.baselines[metric_type]
        z_score = abs((value - baseline["mean"]) / max(baseline["std"], 0.001))
        
        if z_score > 5:
            return "critical"
        elif z_score > 4:
            return "high"
        elif z_score > 3:
            return "medium"
        else:
            return "low"
    
    def _generate_description(self, metric_type: str, value: float) -> str:
        """Generate human-readable anomaly description."""
        if metric_type not in self.baselines:
            return f"Unexpected {metric_type}: {value}"
        
        baseline = self.baselines[metric_type]
        deviation = ((value - baseline["mean"]) / max(baseline["mean"], 0.001)) * 100
        
        direction = "higher" if value > baseline["mean"] else "lower"
        
        return (f"{metric_type} is {abs(deviation):.1f}% {direction} "
                f"(value: {value:.2f}, baseline: {baseline['mean']:.2f})")
    
    def _get_recommended_action(self, metric_type: str, value: float) -> str:
        """Get recommended action for anomaly."""
        actions = {
            "query_latency": "investigate_slow_queries",
            "error_rate": "check_error_logs",
            "cpu_usage": "scale_resources",
            "memory_usage": "increase_buffer_pool",
            "io_wait": "optimize_io",
            "cache_hit_rate": "tune_cache",
        }
        
        return actions.get(metric_type, "investigate")
    
    def analyze_root_cause(self, anomaly: Anomaly) -> Dict[str, Any]:
        """
        Analyze potential root cause of an anomaly.
        
        Args:
            anomaly: The anomaly to analyze
            
        Returns:
            Root cause analysis results
        """
        metric_type = anomaly.anomaly_type
        
        # Correlate with other metrics
        correlations = self._find_correlated_metrics(metric_type)
        
        # Find potential causes
        causes = []
        
        if metric_type == "query_latency":
            if "cpu_usage" in correlations:
                causes.append("High CPU usage causing query delays")
            if "io_wait" in correlations:
                causes.append("I/O bottleneck affecting queries")
            if "cache_hit_rate" in correlations and correlations["cache_hit_rate"] < 0.5:
                causes.append("Low cache hit rate causing disk reads")
        
        elif metric_type == "error_rate":
            causes.append("Check application error logs")
            causes.append("Review recent deployment changes")
            causes.append("Investigate database connection issues")
        
        return {
            "anomaly": anomaly,
            "correlated_metrics": correlations,
            "potential_causes": causes,
            "suggested_investigation": self._get_investigation_plan(anomaly)
        }
    
    def _find_correlated_metrics(self, metric_type: str) -> Dict[str, float]:
        """Find metrics correlated with the given metric."""
        correlations = {}
        
        if metric_type not in self.metrics_history:
            return correlations
        
        target_values = [m["value"] for m in self.metrics_history[metric_type]]
        
        for other_type, history in self.metrics_history.items():
            if other_type == metric_type or len(history) < 10:
                continue
            
            other_values = [m["value"] for m in history]
            
            # Calculate Pearson correlation
            if len(target_values) == len(other_values) and len(target_values) > 0:
                corr = np.corrcoef(target_values, other_values)[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.5:
                    correlations[other_type] = float(corr)
        
        return correlations
    
    def _get_investigation_plan(self, anomaly: Anomaly) -> List[str]:
        """Get investigation plan for anomaly."""
        plans = {
            "query_latency": [
                "1. Check slow query log",
                "2. Review query execution plans",
                "3. Analyze table statistics",
                "4. Check index usage",
                "5. Review recent schema changes"
            ],
            "error_rate": [
                "1. Check database error log",
                "2. Review application logs",
                "3. Verify database connectivity",
                "4. Check for deadlocks",
                "5. Review recent changes"
            ],
            "cpu_usage": [
                "1. Check running queries",
                "2. Review long-running transactions",
                "3. Analyze query plans",
                "4. Consider scaling resources"
            ],
            "memory_usage": [
                "1. Check buffer pool utilization",
                "2. Review memory-intensive queries",
                "3. Analyze table sizes",
                "4. Consider increasing memory"
            ]
        }
        
        return plans.get(anomaly.anomaly_type, ["1. Investigate metric anomaly"])
    
    def get_recent_anomalies(self, count: int = 10) -> List[Anomaly]:
        """Get recent anomalies."""
        return list(self.anomalies)[-count:]
    
    def get_unresolved_anomalies(self) -> List[Anomaly]:
        """Get unresolved anomalies."""
        return [a for a in self.anomalies if not a.resolved]
    
    def resolve_anomaly(self, anomaly_id: str):
        """Mark an anomaly as resolved."""
        for anomaly in self.anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.resolved = True
                logger.info(f"Anomaly {anomaly_id} marked as resolved")
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_anomalies": self.anomaly_count,
            "unresolved": len(self.get_unresolved_anomalies()),
            "metrics_monitored": len(self.metrics_history),
            "baselines_established": len(self.baselines),
            "sensitivity": self.sensitivity
        }
