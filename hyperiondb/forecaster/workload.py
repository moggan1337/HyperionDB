"""
Workload Forecaster for HyperionDB
===================================

Time series forecasting for database workloads to enable
proactive tuning and resource allocation.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import deque
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkloadForecaster:
    """
    Workload Forecasting System for HyperionDB.
    
    Uses time series forecasting to predict future workload patterns
    and enable proactive database tuning.
    
    Forecasting Methods:
    - Moving average
    - Exponential smoothing
    - Linear regression
    - Pattern-based prediction
    
    Features:
    - Multi-metric forecasting
    - Workload pattern detection
    - Trend analysis
    - Seasonal pattern recognition
    - Pre-tuning integration
    """
    
    def __init__(self, forecast_horizon: int = 60):
        """
        Initialize workload forecaster.
        
        Args:
            forecast_horizon: How far ahead to forecast (in seconds)
        """
        self.forecast_horizon = forecast_horizon
        
        # Time series data
        self.query_history: deque = deque(maxlen=10000)
        self.metric_series: Dict[str, deque] = {}
        
        # Forecast models
        self.models: Dict[str, Dict] = {}
        
        # Predictions
        self.current_forecast: Dict[str, Any] = {}
        
        # Statistics
        self.forecast_count = 0
        self.accuracy_history: deque = deque(maxlen=100)
        
        # Pattern detection
        self.patterns: Dict[str, List] = {
            "daily": [],
            "weekly": [],
            "hourly": []
        }
        
        logger.info(f"Workload Forecaster initialized with horizon={forecast_horizon}s")
    
    def update(self, query_history: List[Dict]):
        """
        Update forecaster with new query data.
        
        Args:
            query_history: Recent query records
        """
        for record in query_history:
            self.query_history.append(record)
            
            # Extract timestamp if present
            timestamp = record.get("timestamp", time.time())
            
            # Update metric series
            for metric_name, metric_value in self._extract_metrics(record).items():
                if metric_name not in self.metric_series:
                    self.metric_series[metric_name] = deque(maxlen=5000)
                
                self.metric_series[metric_name].append({
                    "timestamp": timestamp,
                    "value": metric_value
                })
        
        # Update patterns
        self._update_patterns()
    
    def _extract_metrics(self, record: Dict) -> Dict[str, float]:
        """Extract metrics from a query record."""
        metrics = {}
        
        if "latency" in record:
            metrics["query_latency"] = record["latency"]
        
        if "result_count" in record:
            metrics["result_size"] = record["result_count"]
        
        metrics["query_count"] = 1
        
        return metrics
    
    def _update_patterns(self):
        """Update detected workload patterns."""
        if len(self.query_history) < 100:
            return
        
        # Aggregate by time bucket
        current_time = time.time()
        
        # Hourly pattern (24 buckets)
        hour_buckets = {}
        for record in self.query_history:
            ts = record.get("timestamp", current_time)
            hour = int((ts % 86400) / 3600)  # Hour of day
            
            if hour not in hour_buckets:
                hour_buckets[hour] = 0
            hour_buckets[hour] += 1
        
        self.patterns["hourly"] = [
            hour_buckets.get(h, 0) for h in range(24)
        ]
        
        # Daily pattern (7 buckets)
        day_buckets = {}
        for record in self.query_history:
            ts = record.get("timestamp", current_time)
            day = int((ts % (86400 * 7)) / 86400)  # Day of week
            
            if day not in day_buckets:
                day_buckets[day] = 0
            day_buckets[day] += 1
        
        self.patterns["daily"] = [
            day_buckets.get(d, 0) for d in range(7)
        ]
    
    def forecast(self) -> Dict[str, Any]:
        """
        Generate workload forecast.
        
        Returns:
            Forecast for the next period
        """
        self.forecast_count += 1
        
        forecast = {
            "timestamp": time.time(),
            "predicted_queries": self._forecast_query_count(),
            "predicted_type": self._forecast_workload_type(),
            "predicted_latency": self._forecast_latency(),
            "confidence": self._calculate_confidence(),
            "patterns": self.patterns.copy()
        }
        
        self.current_forecast = forecast
        return forecast
    
    def _forecast_query_count(self) -> int:
        """Forecast number of queries in next period."""
        if len(self.query_history) < 10:
            return 100  # Default estimate
        
        # Use recent rate
        recent_window = list(self.query_history)[-100:]
        
        if not recent_window:
            return 100
        
        # Calculate time span
        start_ts = recent_window[0].get("timestamp", time.time() - 100)
        end_ts = recent_window[-1].get("timestamp", time.time())
        time_span = max(end_ts - start_ts, 1)
        
        # Calculate rate
        query_rate = len(recent_window) / time_span
        
        # Apply pattern adjustment
        pattern_factor = self._get_pattern_factor()
        
        # Forecast
        predicted_count = int(query_rate * self.forecast_horizon * pattern_factor)
        
        return max(1, predicted_count)
    
    def _forecast_workload_type(self) -> str:
        """Forecast workload type."""
        if len(self.query_history) < 10:
            return "mixed"
        
        recent = list(self.query_history)[-50:]
        
        # Analyze query types
        select_count = sum(1 for q in recent if q.get("type") == "SELECT")
        write_count = sum(1 for q in recent if q.get("type") in ("INSERT", "UPDATE", "DELETE"))
        
        total = len(recent)
        if total == 0:
            return "mixed"
        
        select_ratio = select_count / total
        write_ratio = write_count / total
        
        if select_ratio > 0.8:
            return "read_heavy"
        elif write_ratio > 0.3:
            return "write_heavy"
        else:
            return "mixed"
    
    def _forecast_latency(self) -> float:
        """Forecast average query latency."""
        if "query_latency" not in self.metric_series:
            return 50.0  # Default estimate
        
        history = list(self.metric_series["query_latency"])
        
        if len(history) < 10:
            return 50.0
        
        # Use exponential moving average
        values = [m["value"] for m in history[-50:]]
        
        # Simple EMA
        alpha = 0.3
        ema = values[0]
        
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        
        # Apply trend adjustment
        if len(values) >= 10:
            recent_trend = np.mean(values[-5:]) - np.mean(values[-10:-5])
            trend_factor = 1 + (recent_trend / max(ema, 1))
        else:
            trend_factor = 1.0
        
        predicted_latency = ema * trend_factor
        
        return max(1.0, predicted_latency)
    
    def _get_pattern_factor(self) -> float:
        """Get workload pattern adjustment factor."""
        current_hour = datetime.now().hour
        
        if len(self.patterns["hourly"]) > current_hour:
            hourly_pattern = self.patterns["hourly"]
            
            # Calculate average activity
            avg_activity = np.mean(hourly_pattern)
            current_activity = hourly_pattern[current_hour]
            
            if avg_activity > 0:
                return current_activity / avg_activity
        
        return 1.0
    
    def _calculate_confidence(self) -> float:
        """Calculate forecast confidence."""
        # Base confidence
        confidence = 0.7
        
        # Increase with more data
        if len(self.query_history) > 1000:
            confidence += 0.1
        elif len(self.query_history) > 100:
            confidence += 0.05
        
        # Decrease with high variance
        if "query_latency" in self.metric_series:
            values = [m["value"] for m in self.metric_series["query_latency"][-50:]]
            if len(values) > 0:
                cv = np.std(values) / max(np.mean(values), 1)  # Coefficient of variation
                if cv > 0.5:
                    confidence -= 0.2
                elif cv < 0.2:
                    confidence += 0.1
        
        # Increase with pattern detection
        if len(self.patterns["hourly"]) > 0:
            hourly_variance = np.var(self.patterns["hourly"]) / max(np.mean(self.patterns["hourly"]), 1)
            if hourly_variance < 0.5:  # Consistent patterns
                confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def evaluate(self) -> float:
        """
        Evaluate forecast accuracy.
        
        Returns:
            Accuracy score (0-1)
        """
        if len(self.accuracy_history) < 5:
            return 0.5
        
        return np.mean(list(self.accuracy_history))
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get current predictions."""
        return self.current_forecast
    
    def predict_resource_needs(self) -> Dict[str, Any]:
        """Predict resource requirements based on forecast."""
        forecast = self.forecast()
        
        # Estimate buffer pool needs
        predicted_queries = forecast["predicted_queries"]
        
        # Simple linear scaling
        buffer_estimate = int(predicted_queries * 10)  # 10 bytes per query estimate
        
        # Estimate memory needs
        memory_estimate = buffer_estimate * 2
        
        # Estimate connections
        connection_estimate = min(100, max(10, predicted_queries // 100))
        
        return {
            "buffer_pool_mb": buffer_estimate // (1024 * 1024),
            "memory_mb": memory_estimate // (1024 * 1024),
            "connections": connection_estimate,
            "io_operations_per_second": int(predicted_queries * 0.5),
            "confidence": forecast["confidence"]
        }
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze workload trends."""
        trends = {}
        
        for metric_name, history in self.metric_series.items():
            if len(history) < 20:
                continue
            
            values = [m["value"] for m in history]
            
            # Calculate trend
            recent = values[-10:]
            older = values[-20:-10] if len(values) >= 20 else values[:10]
            
            if len(older) > 0:
                recent_mean = np.mean(recent)
                older_mean = np.mean(older)
                
                trend = (recent_mean - older_mean) / max(older_mean, 1)
                
                trends[metric_name] = {
                    "current": recent_mean,
                    "previous": older_mean,
                    "change_percent": trend * 100,
                    "direction": "increasing" if trend > 0.05 else "decreasing" if trend < -0.05 else "stable"
                }
        
        return trends
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get forecaster statistics."""
        return {
            "forecast_count": self.forecast_count,
            "queries_stored": len(self.query_history),
            "metrics_tracked": len(self.metric_series),
            "pattern_detected": {
                "hourly": len(self.patterns["hourly"]) > 0,
                "daily": len(self.patterns["daily"]) > 0
            },
            "accuracy": self.evaluate(),
            "confidence": self._calculate_confidence()
        }
