#!/usr/bin/env python3
"""
AI DJ Workflow Engine
A robust workflow system for music production pipelines.

Features:
- Define multi-step workflows with dependencies
- Execute pipelines with proper ordering
- Track progress across steps
- Handle errors with retry logic
- Emit events for monitoring
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow_engine')


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class StepResult:
    """Result of a workflow step execution."""
    step_name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    retries: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    func: Callable[[], Any]
    description: str = ""
    dependencies: list = field(default_factory=list)
    retry_count: int = 0
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    required: bool = True
    step_type: str = "generic"  # e.g., "generation", "processing", "analysis"
    
    def __post_init__(self):
        if not callable(self.func):
            raise ValueError(f"Step '{self.name}' func must be callable")


@dataclass 
class Workflow:
    """A complete workflow definition."""
    name: str
    steps: list
    description: str = ""
    on_complete: Optional[Callable[[list], None]] = None
    on_error: Optional[Callable[[StepResult], None]] = None
    allow_partial: bool = False  # Allow workflow to complete with some steps failed
    
    def __post_init__(self):
        self._validate()
        
    def _validate(self):
        """Validate workflow definition."""
        step_names = {s.name for s in self.steps}
        
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )
        
        # Check for circular dependencies
        self._check_circular()
        
    def _check_circular(self):
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(step_name):
            visited.add(step_name)
            rec_stack.add(step_name)
            
            step = next((s for s in self.steps if s.name == step_name), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(step_name)
            return False
        
        for step in self.steps:
            if step.name not in visited:
                if dfs(step.name):
                    raise ValueError(f"Circular dependency detected in workflow '{self.name}'")


class WorkflowEngine:
    """
    Main workflow engine for executing music production pipelines.
    """
    
    def __init__(self, workspace: str = None, state_file: str = None):
        self.workspace = workspace or "/Users/johnpeter/ai-dj-project"
        self.src_dir = os.path.join(self.workspace, "src")
        self.state_file = state_file or os.path.join(self.src_dir, "workflow", "engine_state.json")
        self.workflows: dict[str, Workflow] = {}
        self.current_workflow: Optional[Workflow] = None
        self.step_results: dict[str, StepResult] = {}
        self._lock = threading.Lock()
        
        # Ensure workflow directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow with the engine."""
        with self._lock:
            self.workflows[workflow.name] = workflow
            logger.info(f"Registered workflow: {workflow.name}")
            
    def get_execution_order(self, workflow: Workflow) -> list[WorkflowStep]:
        """
        Get steps in correct execution order based on dependencies.
        Uses topological sort (Kahn's algorithm).
        """
        in_degree = {s.name: 0 for s in workflow.steps}
        graph = {s.name: [] for s in workflow.steps}
        
        for step in workflow.steps:
            in_degree[step.name] = len(step.dependencies)
            for dep in step.dependencies:
                graph[dep].append(step.name)
        
        # Queue of steps with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(
                next(s for s in workflow.steps if s.name == current)
            )
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return execution_order
    
    def can_run_step(self, step: WorkflowStep) -> bool:
        """Check if a step's dependencies are satisfied."""
        for dep in step.dependencies:
            if dep not in self.step_results:
                return False
            if self.step_results[dep].status != StepStatus.COMPLETED:
                return False
        return True
    
    def execute_step(self, step: WorkflowStep, workflow_name: str) -> StepResult:
        """Execute a single workflow step with retry logic."""
        start_time = time.time()
        result = StepResult(step_name=step.name, status=StepStatus.RUNNING)
        
        logger.info(f"[{workflow_name}] Executing step: {step.name}")
        
        attempt = 0
        while attempt <= step.retry_count:
            try:
                if step.timeout:
                    # Run with timeout using thread
                    result.output = self._run_with_timeout(step.func, step.timeout)
                else:
                    result.output = step.func()
                    
                result.status = StepStatus.COMPLETED
                result.retries = attempt
                break
                
            except Exception as e:
                attempt += 1
                result.retries = attempt - 1
                error_msg = str(e)
                
                if attempt <= step.retry_count:
                    result.status = StepStatus.RETRYING
                    logger.warning(f"[{workflow_name}] Step {step.name} failed (attempt {attempt}), retrying in {step.retry_delay}s: {error_msg}")
                    time.sleep(step.retry_delay)
                else:
                    result.status = StepStatus.FAILED if step.required else StepStatus.SKIPPED
                    result.error = error_msg
                    logger.error(f"[{workflow_name}] Step {step.name} failed after {attempt} attempts: {error_msg}")
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run a function with timeout."""
        result = []
        exception = []
        
        def target():
            try:
                result.append(func())
            except Exception as e:
                exception.append(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Step timed out after {timeout}s")
        if exception:
            raise exception[0]
        return result[0] if result else None
    
    def run_workflow(
        self, 
        workflow_name: str, 
        context: dict = None,
        step_callback: Optional[Callable[[StepResult], None]] = None
    ) -> dict:
        """
        Execute a workflow and return results.
        
        Args:
            workflow_name: Name of registered workflow
            context: Optional context passed to all steps
            step_callback: Optional callback after each step
            
        Returns:
            Dict with workflow status and all step results
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
            
        workflow = self.workflows[workflow_name]
        self.current_workflow = workflow
        self.step_results = {}
        
        execution_order = self.get_execution_order(workflow)
        
        logger.info(f"[{workflow_name}] Starting workflow with {len(execution_order)} steps")
        
        for step in execution_order:
            if not self.can_run_step(step):
                result = StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    error="Dependencies not met"
                )
                self.step_results[step.name] = result
                continue
            
            result = self.execute_step(step, workflow_name)
            self.step_results[step.name] = result
            
            # Save state after each step
            self._save_state(workflow_name)
            
            if step_callback:
                step_callback(result)
            
            # Handle error
            if result.status == StepStatus.FAILED and not workflow.allow_partial:
                logger.error(f"[{workflow_name}] Workflow failed at step '{step.name}'")
                break
        
        # Determine overall status
        all_completed = all(
            r.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) 
            for r in self.step_results.values()
        )
        any_failed = any(
            r.status == StepStatus.FAILED 
            for r in self.step_results.values()
        )
        
        if any_failed and not workflow.allow_partial:
            overall_status = WorkflowStatus.FAILED
        elif all_completed:
            overall_status = WorkflowStatus.COMPLETED
        else:
            overall_status = WorkflowStatus.RUNNING
            
        # Run completion callback
        if overall_status == WorkflowStatus.COMPLETED and workflow.on_complete:
            workflow.on_complete(list(self.step_results.values()))
            
        logger.info(f"[{workflow_name}] Workflow {overall_status.value}")
        
        return {
            "workflow": workflow_name,
            "status": overall_status.value,
            "steps": {name: {
                "status": r.status.value,
                "output": str(r.output) if r.output else None,
                "error": r.error,
                "duration_ms": r.duration_ms,
                "retries": r.retries
            } for name, r in self.step_results.items()}
        }
    
    def _save_state(self, workflow_name: str) -> None:
        """Save current workflow state to file."""
        state = {
            "workflow": workflow_name,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        for name, result in self.step_results.items():
            state["steps"][name] = {
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "retries": result.retries,
                "timestamp": result.timestamp
            }
            
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_progress(self) -> dict:
        """Get current workflow progress."""
        if not self.step_results:
            return {"total": 0, "completed": 0, "percent": 0}
            
        total = len(self.step_results)
        completed = sum(1 for r in self.step_results.values() 
                       if r.status == StepStatus.COMPLETED)
        
        return {
            "total": total,
            "completed": completed,
            "percent": int((completed / total) * 100) if total else 0
        }
    
    def get_status(self, workflow_name: str = None) -> dict:
        """Get status of workflow(s)."""
        if workflow_name:
            if workflow_name not in self.workflows:
                return {"error": "Workflow not found"}
            return {
                "name": workflow_name,
                "exists": True,
                "steps": [s.name for s in self.workflows[workflow_name].steps]
            }
        return {
            "registered": list(self.workflows.keys())
        }


# === Pre-defined Music Production Workflows ===

def create_full_production_workflow(engine: WorkflowEngine) -> Workflow:
    """Create the main full production pipeline."""
    
    # Import the generator modules
    sys.path.insert(0, engine.src_dir)
    
    steps = [
        WorkflowStep(
            name="analyze_input",
            func=lambda: _load_module("stem_processor").analyze_track("input.wav"),
            description="Analyze input track for stems and characteristics",
            step_type="analysis"
        ),
        WorkflowStep(
            name="generate_drums",
            func=lambda: _load_module("drum_generator").generate_drums(bpm=120, style="electronic"),
            description="Generate drum track",
            dependencies=["analyze_input"],
            step_type="generation"
        ),
        WorkflowStep(
            name="generate_bass",
            func=lambda: _load_module("bass_generator").generate_bass(),
            description="Generate bass line",
            dependencies=["analyze_input"],
            step_type="generation"
        ),
        WorkflowStep(
            name="generate_melody",
            func=lambda: _load_module("melody_generator").generate_melody(),
            description="Generate melody",
            dependencies=["analyze_input", "generate_bass"],
            step_type="generation"
        ),
        WorkflowStep(
            name="generate_chords",
            func=lambda: _load_module("chord_generator").generate_chords(),
            description="Generate chord progression",
            dependencies=["analyze_input"],
            step_type="generation"
        ),
        WorkflowStep(
            name="process_vocals",
            func=lambda: _load_module("vocal_processor").process_vocals(),
            description="Process any vocals",
            dependencies=["analyze_input"],
            step_type="processing",
            required=False
        ),
        WorkflowStep(
            name="apply_effects",
            func=lambda: _load_module("effects_processor").apply_effects(),
            description="Apply effects processing",
            dependencies=["generate_drums", "generate_bass", "generate_melody", "generate_chords"],
            step_type="processing"
        ),
        WorkflowStep(
            name="mix_stems",
            func=lambda: _load_module("mixing_console").mix_all(),
            description="Mix all stems together",
            dependencies=["apply_effects", "process_vocals"],
            step_type="mixing"
        ),
        WorkflowStep(
            name="master",
            func=lambda: _load_module("auto_master").master_track(),
            description="Master the final track",
            dependencies=["mix_stems"],
            step_type="mastering"
        ),
        WorkflowStep(
            name="export",
            func=lambda: _export_final(),
            description="Export final track",
            dependencies=["master"],
            step_type="output",
            retry_count=2,
            retry_delay=2.0
        )
    ]
    
    return Workflow(
        name="full_production",
        description="Complete music production pipeline from analysis to export",
        steps=steps,
        on_complete=lambda results: logger.info("Production complete!")
    )


def create_quick_mix_workflow(engine: WorkflowEngine) -> Workflow:
    """Create a quick mixing workflow."""
    
    steps = [
        WorkflowStep(
            name="load_stems",
            func=lambda: print("Loading stems..."),
            description="Load existing stems",
            step_type="input"
        ),
        WorkflowStep(
            name="auto_mix",
            func=lambda: _load_module("mixing_console").auto_mix(),
            description="Auto-mix stems",
            dependencies=["load_stems"],
            step_type="mixing"
        ),
        WorkflowStep(
            name="quick_master",
            func=lambda: _load_module("auto_master").quick_master(),
            description="Quick master",
            dependencies=["auto_mix"],
            step_type="mastering"
        )
    ]
    
    return Workflow(
        name="quick_mix",
        description="Quick mix and master existing stems",
        steps=steps,
        allow_partial=True
    )


def create_stem_generation_workflow(engine: WorkflowEngine) -> Workflow:
    """Create a stem generation workflow."""
    
    steps = [
        WorkflowStep(
            name="demucs_extract",
            func=lambda: _load_module("stem_processor").extract_stems_demucs(),
            description="Extract stems using Demucs",
            step_type="processing",
            retry_count=1,
            timeout=300
        ),
        WorkflowStep(
            name="separate_vocals",
            func=lambda: _load_module("stem_processor").separate_vocals(),
            description="Separate vocal track",
            dependencies=["demucs_extract"],
            step_type="processing"
        ),
        WorkflowStep(
            name="enhance_stems",
            func=lambda: _load_module("stem_processor").enhance_all(),
            description="Enhance all stem quality",
            dependencies=["separate_vocals"],
            step_type="processing"
        )
    ]
    
    return Workflow(
        name="stem_generation",
        description="Generate stems from audio file",
        steps=steps
    )


# === Helper Functions ===

def _load_module(module_name: str):
    """Lazy load a module."""
    try:
        return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Module {module_name} not available: {e}")
        return None


def _export_final():
    """Export the final track."""
    # Placeholder for export logic
    logger.info("Exporting final track...")
    return {"file": "final_track.wav", "path": "/Users/johnpeter/ai-dj-project/output/"}


# === CLI Interface ===

def main():
    """CLI for workflow engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Workflow Engine")
    parser.add_argument("command", choices=["list", "run", "status", "progress"],
                       help="Command to execute")
    parser.add_argument("--workflow", "-w", help="Workflow name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    engine = WorkflowEngine()
    
    # Register default workflows
    engine.register_workflow(create_full_production_workflow(engine))
    engine.register_workflow(create_quick_mix_workflow(engine))
    engine.register_workflow(create_stem_generation_workflow(engine))
    
    if args.command == "list":
        status = engine.get_status()
        print("Registered workflows:")
        for wf in status["registered"]:
            print(f"  - {wf}")
            
    elif args.command == "run":
        if not args.workflow:
            print("Error: --workflow required")
            sys.exit(1)
            
        def print_progress(result: StepResult):
            status_icon = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.RUNNING: "🔄",
                StepStatus.RETRYING: "🔁",
                StepStatus.SKIPPED: "⏭️"
            }.get(result.status, "⚪")
            print(f"{status_icon} {result.step_name}: {result.status.value}")
        
        result = engine.run_workflow(args.workflow, step_callback=print_progress)
        print(f"\nWorkflow status: {result['status']}")
        
    elif args.command == "status":
        if args.workflow:
            print(json.dumps(engine.get_status(args.workflow), indent=2))
        else:
            print(json.dumps(engine.get_status(), indent=2))
            
    elif args.command == "progress":
        progress = engine.get_progress()
        print(f"Progress: {progress['completed']}/{progress['total']} ({progress['percent']}%)")


if __name__ == "__main__":
    main()
