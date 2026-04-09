"""
Pydantic models for IncidentCommanderEnv.
UPDATED: Added delayed_failure_configs field.
"""

from typing import Dict, List, Optional, Literal, Any, Set
from pydantic import BaseModel, Field
from enum import Enum


class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ServiceStatus(BaseModel):
    """Current status of a single service."""
    name: str
    health: ServiceHealth = ServiceHealth.HEALTHY
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = Field(default=50, ge=0)
    cpu_usage: float = Field(default=0.3, ge=0.0, le=1.0)
    memory_usage: float = Field(default=0.4, ge=0.0, le=1.0)
    last_deploy_time: int = Field(default=-120)
    last_deploy_version: str = Field(default="v1.0.0")


class Alert(BaseModel):
    """An alert from monitoring system."""
    timestamp: int
    service: str
    severity: AlertSeverity
    message: str


class QueryResult(BaseModel):
    """Result from a query action."""
    query_type: str
    service: str
    data: str
    timestamp: int


class ActionType(str, Enum):
    QUERY_LOGS = "query_logs"
    QUERY_METRICS = "query_metrics"
    CHECK_DEPLOYS = "check_deploys"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_SERVICE = "rollback_service"
    SCALE_SERVICE = "scale_service"
    DECLARE_ROOT_CAUSE = "declare_root_cause"
    RESOLVE_INCIDENT = "resolve_incident"


class Action(BaseModel):
    """Action that the agent can take."""
    action_type: ActionType
    target_service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True


class Observation(BaseModel):
    """What the agent observes at each step."""
    current_time: int = Field(description="Minutes since incident started")
    services: Dict[str, ServiceStatus]
    alerts: List[Alert]
    last_query_result: Optional[QueryResult] = None
    action_history: List[str] = Field(default_factory=list)
    incident_resolved: bool = False
    declared_root_causes: List[str] = Field(default_factory=list)
    task_id: str = ""
    task_description: str = ""


class StepReward(BaseModel):
    """Reward information returned after each step."""
    value: float
    reason: str
    cumulative: float


class StepResult(BaseModel):
    """Complete result of a step() call."""
    observation: Observation
    reward: StepReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    """Final grading result."""
    score: float = Field(gt=0.0, lt=1.0)
    root_cause_score: float = Field(gt=0.0, lt=1.0)
    remediation_score: float = Field(gt=0.0, lt=1.0)
    investigation_score: float = Field(gt=0.0, lt=1.0)
    efficiency_score: float = Field(gt=0.0, lt=1.0)
    penalty_score: float = Field(gt=0.0, lt=1.0)
    breakdown: Dict[str, Any]


class Scenario(BaseModel):
    """Definition of a scenario/task."""
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    root_causes: List[str]
    required_remediations: List[Dict[str, Any]]
    initial_failures: Dict[str, Dict[str, Any]]
    time_limit: int = Field(default=60)
    deploy_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    misleading_services: List[str] = Field(default_factory=list)
    delayed_failures: Dict[str, int] = Field(default_factory=dict)  # service -> step to trigger
    delayed_failure_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # service -> failure config
    optimal_action_count: int = Field(default=8)


class InvestigationState(BaseModel):
    """Tracks what the agent has investigated."""
    queried_logs: Set[str] = Field(default_factory=set)
    queried_metrics: Set[str] = Field(default_factory=set)
    checked_deploys: Set[str] = Field(default_factory=set)
    query_counts: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True