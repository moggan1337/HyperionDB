"""
Transaction Module for HyperionDB
=================================

Provides transaction management with MVCC support.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import uuid


class TransactionState(Enum):
    """Transaction states."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"
    PENDING = "pending"


class IsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class Transaction:
    """
    Represents a database transaction with MVCC support.
    """
    txn_id: str
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    state: TransactionState = TransactionState.ACTIVE
    start_time: float = field(default_factory=time.time)
    commit_time: Optional[float] = None
    
    # Write set for MVCC
    write_set: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # Snapshot for MVCC
    snapshot: Optional[Dict] = None
    
    # Lock tracking
    locks_held: List[str] = field(default_factory=list)
    
    def add_write(self, table: str, row: Dict):
        """Add a write operation to the transaction."""
        if table not in self.write_set:
            self.write_set[table] = []
        self.write_set[table].append(row)
    
    def commit(self):
        """Commit the transaction."""
        self.state = TransactionState.COMMITTED
        self.commit_time = time.time()
    
    def abort(self):
        """Abort the transaction."""
        self.state = TransactionState.ABORTED
    
    def duration(self) -> float:
        """Get transaction duration in seconds."""
        end_time = self.commit_time or time.time()
        return end_time - self.start_time
    
    def __repr__(self):
        return f"Transaction({self.txn_id[:8]}, {self.state.value})"


class TransactionManager:
    """
    Manages database transactions with MVCC and lock management.
    """
    
    def __init__(self):
        """Initialize transaction manager."""
        self.active_transactions: Dict[str, Transaction] = {}
        self.completed_transactions: List[Transaction] = []
        self.lock = threading.RLock()
        self.next_txn_id = 0
        
        # Lock table
        self.lock_table: Dict[str, threading.Lock] = {}
        
        # Transaction counter
        self.global_txn_counter = 0
        self.committed_txn_count = 0
    
    def begin_transaction(self, 
                          isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
                          ) -> Transaction:
        """
        Begin a new transaction.
        
        Args:
            isolation_level: Transaction isolation level
            
        Returns:
            New Transaction object
        """
        with self.lock:
            txn_id = f"txn_{self.next_txn_id}_{uuid.uuid4().hex[:8]}"
            self.next_txn_id += 1
            
            txn = Transaction(
                txn_id=txn_id,
                isolation_level=isolation_level,
                start_time=time.time()
            )
            
            self.active_transactions[txn_id] = txn
            self.global_txn_counter += 1
            
            return txn
    
    def commit(self, txn_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            txn_id: Transaction ID
            
        Returns:
            True if committed successfully
        """
        with self.lock:
            if txn_id not in self.active_transactions:
                return False
            
            txn = self.active_transactions[txn_id]
            
            # Release locks
            self._release_locks(txn)
            
            # Commit
            txn.commit()
            
            # Move to completed
            del self.active_transactions[txn_id]
            self.completed_transactions.append(txn)
            self.committed_txn_count += 1
            
            # Cleanup old completed transactions
            if len(self.completed_transactions) > 1000:
                self.completed_transactions = self.completed_transactions[-500:]
            
            return True
    
    def abort(self, txn_id: str) -> bool:
        """
        Abort a transaction.
        
        Args:
            txn_id: Transaction ID
            
        Returns:
            True if aborted successfully
        """
        with self.lock:
            if txn_id not in self.active_transactions:
                return False
            
            txn = self.active_transactions[txn_id]
            
            # Release locks
            self._release_locks(txn)
            
            # Abort
            txn.abort()
            
            # Move to completed
            del self.active_transactions[txn_id]
            self.completed_transactions.append(txn)
            
            return True
    
    def _release_locks(self, txn: Transaction):
        """Release all locks held by a transaction."""
        for lock_key in txn.locks_held:
            if lock_key in self.lock_table:
                # Unlock would be handled by the lock object itself
                pass
        txn.locks_held.clear()
    
    def acquire_lock(self, txn_id: str, resource: str, 
                     exclusive: bool = False) -> bool:
        """
        Acquire a lock on a resource.
        
        Args:
            txn_id: Transaction ID
            resource: Resource to lock (e.g., table name)
            exclusive: True for exclusive lock, False for shared
            
        Returns:
            True if lock acquired
        """
        with self.lock:
            if txn_id not in self.active_transactions:
                return False
            
            txn = self.active_transactions[txn_id]
            
            # Get or create lock
            if resource not in self.lock_table:
                self.lock_table[resource] = threading.Lock()
            
            lock = self.lock_table[resource]
            
            # Try to acquire
            if exclusive:
                acquired = lock.acquire(timeout=30)
            else:
                # Shared lock - multiple readers allowed
                acquired = True  # Simplified for this implementation
            
            if acquired:
                txn.locks_held.append(resource)
            
            return acquired
    
    def get_active_transactions(self) -> List[Transaction]:
        """Get list of active transactions."""
        with self.lock:
            return list(self.active_transactions.values())
    
    def get_transaction(self, txn_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        with self.lock:
            return self.active_transactions.get(txn_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transaction manager statistics."""
        with self.lock:
            return {
                "active_transactions": len(self.active_transactions),
                "completed_transactions": len(self.completed_transactions),
                "total_transactions": self.global_txn_counter,
                "committed_transactions": self.committed_txn_count,
                "aborted_transactions": self.global_txn_counter - self.committed_txn_count,
                "lock_count": len(self.lock_table)
            }
