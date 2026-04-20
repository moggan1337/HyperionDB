"""
HyperionDB - Self-Driving Database with Learned Optimizer
=========================================================

A revolutionary database system that leverages machine learning to optimize
its own performance through learned query optimization, neural index structures,
and autonomous tuning.

Components:
- Learned Query Optimizer (RL-based)
- Neural Index Structures (Learned Indexes)
- Automatic Database Tuning
- Neural Cost Models
- Deep Learning Cardinality Estimation
- GNN-based Join Order Enumeration
- Anomaly Detection & Self-Healing
- Workload Forecasting & Pre-tuning
- Adaptive Query Execution
- Knob-free Auto-configuration

Author: HyperionDB Team
License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "HyperionDB Team"

from hyperiondb.core.database import HyperionDB
from hyperiondb.core.table import Table
from hyperiondb.core.query import Query
from hyperiondb.core.transaction import Transaction

__all__ = [
    "HyperionDB",
    "Table", 
    "Query",
    "Transaction",
]
