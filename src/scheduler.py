#!/usr/bin/env python3
"""
Task Scheduler for AI DJ System

Provides scheduling capabilities for:
- One-time scheduled tasks
- Recurring/periodic tasks
- Task queues with priorities
- Background task execution
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
import queue  # stdlib
import json
import os

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Task priority levels (lower number = higher priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    scheduled_time: Optional[datetime] = None
    interval: Optional[timedelta] = None  # For recurring tasks
    max_retries: int = 3
    retry_count: int = 0
    result: Any = None
    error: Optional[Exception] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering"""
        if self.priority == other.priority:
            return self.scheduled_time < other.scheduled_time
        return self.priority.value < other.priority.value
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "priority": self.priority.name,
            "status": self.status.name,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "interval": self.interval.total_seconds() if self.interval else None,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": str(self.error) if self.error else None,
        }


class TaskScheduler:
    """Main task scheduler for AI DJ System"""
    
    def __init__(self, max_workers: int = 4, persist_path: Optional[str] = None):
        """
        Initialize the scheduler
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            persist_path: Optional path for persisting scheduled tasks
        """
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AI-DJ-Scheduler")
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._persist_path = persist_path
        self._callbacks: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
        }
        
        # Load persisted tasks if available
        if persist_path and os.path.exists(persist_path):
            self._load_tasks()
    
    def _load_tasks(self):
        """Load tasks from persistence file"""
        try:
            with open(self._persist_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data.get('tasks', []))} tasks from persistence")
        except Exception as e:
            logger.warning(f"Failed to load persisted tasks: {e}")
    
    def _save_tasks(self):
        """Persist tasks to file"""
        if not self._persist_path:
            return
        try:
            data = {
                "tasks": [task.to_dict() for task in self._tasks.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist tasks: {e}")
    
    def _fire_callbacks(self, event: str, task: ScheduledTask):
        """Fire registered callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(task)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for task events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def schedule_once(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        scheduled_time: Optional[datetime] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> ScheduledTask:
        """
        Schedule a one-time task
        
        Args:
            name: Unique task name
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            scheduled_time: When to run the task (None = run immediately)
            priority: Task priority
            
        Returns:
            The created ScheduledTask
        """
        kwargs = kwargs or {}
        task = ScheduledTask(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_time=scheduled_time or datetime.now(),
        )
        
        with self._lock:
            self._tasks[name] = task
            self._task_queue.put(task)
        
        logger.info(f"Scheduled one-time task: {name} for {task.scheduled_time}")
        self._save_tasks()
        return task
    
    def schedule_interval(
        self,
        name: str,
        func: Callable,
        interval: Union[timedelta, float, int],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        start_time: Optional[datetime] = None,
    ) -> ScheduledTask:
        """
        Schedule a recurring task
        
        Args:
            name: Unique task name
            func: Function to execute
            interval: Interval between executions (timedelta or seconds)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority
            start_time: When to start the first execution
            
        Returns:
            The created ScheduledTask
        """
        kwargs = kwargs or {}
        
        if isinstance(interval, (int, float)):
            interval = timedelta(seconds=interval)
        
        task = ScheduledTask(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_time=start_time or datetime.now(),
            interval=interval,
        )
        
        with self._lock:
            self._tasks[name] = task
            self._task_queue.put(task)
        
        logger.info(f"Scheduled recurring task: {name} every {interval}")
        self._save_tasks()
        return task
    
    def schedule_cron(
        self,
        name: str,
        func: Callable,
        cron_expr: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> ScheduledTask:
        """
        Schedule a task using cron-like expression
        
        Args:
            name: Unique task name
            func: Function to execute
            cron_expr: Cron expression (minute hour day month weekday)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority
            
        Returns:
            The created ScheduledTask
            
        Note: This is a simplified cron implementation
        """
        # Parse simplified cron: "minute hour day month weekday"
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")
        
        # For simplicity, we'll calculate next run based on minute/hour
        now = datetime.now()
        minute, hour = int(parts[0]), int(parts[1])
        
        next_run = now.replace(minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(hours=1)
        
        return self.schedule_once(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            scheduled_time=next_run,
            priority=priority,
        )
    
    def cancel(self, name: str) -> bool:
        """
        Cancel a scheduled task
        
        Args:
            name: Task name to cancel
            
        Returns:
            True if task was cancelled, False if not found
        """
        with self._lock:
            if name in self._tasks:
                task = self._tasks[name]
                task.status = TaskStatus.CANCELLED
                logger.info(f"Cancelled task: {name}")
                self._save_tasks()
                return True
        return False
    
    def get_task(self, name: str) -> Optional[ScheduledTask]:
        """Get a task by name"""
        return self._tasks.get(name)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[ScheduledTask]:
        """List all tasks, optionally filtered by status"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks)
    
    def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._fire_callbacks("task_started", task)
        
        try:
            logger.info(f"Executing task: {task.name}")
            
            # Check if function is async
            if asyncio.iscoroutinefunction(task.func):
                # Run in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task.result = loop.run_until_complete(task.func(*task.args, **task.kwargs))
                finally:
                    loop.close()
            else:
                task.result = task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self._fire_callbacks("task_completed", task)
            logger.info(f"Task completed: {task.name}")
            
        except Exception as e:
            task.error = e
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                task.scheduled_time = datetime.now() + timedelta(seconds=2 ** task.retry_count)
                self._task_queue.put(task)
                logger.warning(f"Task {task.name} failed, retry {task.retry_count}/{task.max_retries}")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                self._fire_callbacks("task_failed", task)
                logger.error(f"Task {task.name} failed permanently: {e}")
        
        self._save_tasks()
        return task.result
    
    def _worker_loop(self):
        """Main worker loop"""
        while self._running:
            try:
                # Get task with timeout
                try:
                    task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if we should wait for scheduled time
                wait_time = (task.scheduled_time - datetime.now()).total_seconds()
                if wait_time > 0:
                    # Put back in queue and wait
                    time.sleep(min(wait_time, 1.0))  # Max 1 second sleep
                    self._task_queue.put(task)
                    continue
                
                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    self._task_queue.task_done()
                    continue
                
                # Execute the task
                self._execute_task(task)
                
                # If recurring, reschedule
                if task.interval and task.status == TaskStatus.COMPLETED:
                    task.scheduled_time = datetime.now() + task.interval
                    task.status = TaskStatus.PENDING
                    self._task_queue.put(task)
                
                self._task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Task scheduler started")
    
    def stop(self, wait: bool = True):
        """Stop the scheduler"""
        self._running = False
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Task scheduler stopped")
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submit a task for immediate execution
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future representing the task
        """
        return self._executor.submit(func, *args, **kwargs)
    
    def run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function synchronously (blocking)
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        return self._executor.submit(func, *args, **kwargs).result()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler and executor"""
        self.stop(wait=wait)
        self._executor.shutdown(wait=wait)
        logger.info("Scheduler shutdown complete")


# Decorator for scheduled tasks
def scheduled_task(name: Optional[str] = None, priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to mark a function as a scheduled task"""
    def decorator(func: Callable) -> Callable:
        func._is_scheduled_task = True
        func._task_name = name or func.__name__
        func._task_priority = priority
        return func
    return decorator


# Example usage and test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create scheduler
    scheduler = TaskScheduler(max_workers=2)
    
    # Example task functions
    def example_task(task_id: int):
        """Example of a simple task"""
        logger.info(f"Running task {task_id}")
        time.sleep(1)
        return f"Task {task_id} completed"
    
    async def async_task(task_id: int):
        """Example of an async task"""
        logger.info(f"Running async task {task_id}")
        await asyncio.sleep(1)
        return f"Async task {task_id} completed"
    
    # Schedule some tasks
    scheduler.schedule_once("task1", example_task, (1,))
    scheduler.schedule_once("task2", example_task, (2,), priority=TaskPriority.HIGH)
    scheduler.schedule_interval("recurring_task", example_task, timedelta(seconds=10), (3,))
    
    # Register callbacks
    def on_task_complete(task: ScheduledTask):
        logger.info(f"Callback: {task.name} completed with result: {task.result}")
    
    scheduler.register_callback("task_completed", on_task_complete)
    
    # Start scheduler
    scheduler.start()
    
    # Let it run for a bit
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown()
