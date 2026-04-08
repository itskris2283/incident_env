"""
Scenario definitions with FIXED delayed failure handling.
Hard scenario has truly delayed failures and misleading signals.
"""

from .models import Scenario
from typing import Dict, Any


def get_easy_scenario() -> Scenario:
    """Easy: Single obvious root cause."""
    return Scenario(
        id="easy_single_failure",
        name="Database Connection Exhaustion",
        difficulty="easy",
        description=(
            "ALERT: API and frontend showing errors. "
            "Users report timeouts. Started 5 minutes ago. "
            "Identify root cause and restore service."
        ),
        root_causes=["db"],
        required_remediations=[{"action": "restart", "service": "db"}],
        initial_failures={
            "db": {
                "health": "down",
                "error_rate": 0.90,
                "latency_ms": 5000,
                "cpu_usage": 0.85,
                "memory_usage": 0.75
            }
        },
        time_limit=25,
        deploy_history={
            "db": [
                {"time": -200, "version": "v2.0.0", "change": "Major release"},
                {"time": -30, "version": "v2.0.1", "change": "Increased pool size"},
            ],
        },
        misleading_services=[],
        delayed_failures={},  # No delayed failures
        delayed_failure_configs={},
        optimal_action_count=6
    )


def get_medium_scenario() -> Scenario:
    """Medium: Cascading failure requiring trace-back."""
    return Scenario(
        id="medium_cascade",
        name="Redis Memory Cascade",
        difficulty="medium",
        description=(
            "INCIDENT: Multiple services degraded. "
            "Auth showing intermittent failures. API latency 10x normal. "
            "Frontend 503 errors increasing. Escalated 3 minutes ago. "
            "Find the origin of the cascade."
        ),
        root_causes=["redis"],
        required_remediations=[{"action": "scale", "service": "redis"}],
        initial_failures={
            "redis": {
                "health": "degraded",
                "error_rate": 0.65,
                "latency_ms": 1500,
                "cpu_usage": 0.60,
                "memory_usage": 0.97
            }
        },
        time_limit=35,
        deploy_history={
            "redis": [
                {"time": -300, "version": "v6.2.0", "change": "Version upgrade"},
            ],
            "auth": [
                {"time": -20, "version": "v1.9.0", "change": "Added aggressive caching"},
            ],
        },
        misleading_services=["api"],  # API shows errors but isn't root cause
        delayed_failures={},
        delayed_failure_configs={},
        optimal_action_count=8
    )


def get_hard_scenario() -> Scenario:
    """
    Hard: Multiple simultaneous root causes with delayed failure.
    
    - DB has disk I/O issues (IMMEDIATE)
    - Redis has memory issues (DELAYED - triggers at step 4)
    - API shows errors but is NOT a root cause (misleading)
    
    Agent must:
    1. Investigate to find DB issue first
    2. Fix DB
    3. Notice Redis failure appears later
    4. Investigate and fix Redis
    5. Resolve
    """
    return Scenario(
        id="hard_multi_root",
        name="Multi-Root: DB I/O + Redis Memory (Delayed)",
        difficulty="hard",
        description=(
            "CRITICAL INCIDENT: Severe degradation across services. "
            "Conflicting reports - some requests work, others timeout. "
            "Error rates fluctuating. Multiple teams escalated. "
            "WARNING: This may involve MULTIPLE root causes that appear at DIFFERENT TIMES. "
            "Investigate thoroughly before acting."
        ),
        root_causes=["db", "redis"],  # Both are root causes
        required_remediations=[
            {"action": "restart", "service": "db"},
            {"action": "scale", "service": "redis"}
        ],
        # FIXED: Only DB fails immediately. Redis is NOT in initial_failures.
        initial_failures={
            "db": {
                "health": "degraded",
                "error_rate": 0.45,
                "latency_ms": 1200,
                "cpu_usage": 0.75,
                "memory_usage": 0.55
            }
            # NOTE: Redis is NOT here - it will fail later
        },
        time_limit=50,
        deploy_history={
            "db": [
                {"time": -400, "version": "v2.0.0", "change": "Stable"},
                {"time": -80, "version": "v2.0.2", "change": "Added new index"},
            ],
            "api": [
                {"time": -150, "version": "v3.4.0", "change": "Stable"},
                {"time": -15, "version": "v3.5.0", "change": "Refactored handlers"},
            ],
            "redis": [
                {"time": -500, "version": "v6.2.0", "change": "Stable"},
            ],
            "auth": [
                {"time": -100, "version": "v1.8.5", "change": "Routine update"},
            ],
        },
        misleading_services=["api", "auth"],  # These look bad but aren't root causes
        # FIXED: Redis failure is truly delayed - triggers at step 4
        delayed_failures={
            "redis": 4  # Will trigger at step 4
        },
        # Configuration for when Redis fails
        delayed_failure_configs={
            "redis": {
                "health": "degraded",
                "error_rate": 0.60,
                "latency_ms": 900,
                "cpu_usage": 0.55,
                "memory_usage": 0.94
            }
        },
        optimal_action_count=14  # Need more actions for multi-root
    )


def get_scenario(task_id: str) -> Scenario:
    """Get scenario by ID."""
    scenarios = {
        "easy_single_failure": get_easy_scenario,
        "medium_cascade": get_medium_scenario,
        "hard_multi_root": get_hard_scenario,
    }
    
    if task_id not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(f"Unknown scenario: {task_id}. Available: {available}")
    
    return scenarios[task_id]()


def list_scenarios() -> list:
    """List available scenarios."""
    return [
        {"id": "easy_single_failure", "name": "Database Connection Exhaustion", "difficulty": "easy"},
        {"id": "medium_cascade", "name": "Redis Memory Cascade", "difficulty": "medium"},
        {"id": "hard_multi_root", "name": "Multi-Root: DB I/O + Redis Memory (Delayed)", "difficulty": "hard"},
    ]