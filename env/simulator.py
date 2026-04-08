"""
Core simulation logic for IncidentCommanderEnv.
FIXED: Proper failure propagation, delayed failures, reward diminishing, penalty consistency.
"""

from typing import Dict, List, Optional, Tuple, Set
from .models import (
    ServiceStatus, ServiceHealth, Alert, AlertSeverity,
    QueryResult, Action, ActionType, Scenario, InvestigationState
)
import hashlib


class ServiceSimulator:
    """Simulates the microservices infrastructure with temporal dynamics."""
    
    DEPENDENCIES = {
        "frontend": ["api"],
        "api": ["auth", "db"],
        "auth": ["db", "redis"],
        "db": [],
        "redis": []
    }
    
    DEPENDENTS = {
        "frontend": [],
        "api": ["frontend"],
        "auth": ["api"],
        "db": ["api", "auth"],
        "redis": ["auth"]
    }
    
    SERVICE_NAMES = ["frontend", "api", "auth", "db", "redis"]
    
    ACTION_TIME_COSTS = {
        ActionType.QUERY_LOGS: 1,
        ActionType.QUERY_METRICS: 1,
        ActionType.CHECK_DEPLOYS: 1,
        ActionType.RESTART_SERVICE: 4,
        ActionType.ROLLBACK_SERVICE: 6,
        ActionType.SCALE_SERVICE: 3,
        ActionType.DECLARE_ROOT_CAUSE: 0,
        ActionType.RESOLVE_INCIDENT: 1,
    }
    
    def __init__(self, scenario: Scenario, seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        self.current_time = 0
        self.step_count = 0  # Track discrete steps for delayed failures
        self.services: Dict[str, ServiceStatus] = {}
        self.alerts: List[Alert] = []
        self.action_history: List[str] = []
        self.declared_root_causes: List[str] = []
        self.incident_resolved = False
        self.cumulative_reward = 0.0
        
        # Tracking for grading
        self.wrong_actions = 0
        self.total_actions = 0
        self.remediation_actions: List[Dict] = []
        
        # Investigation tracking
        self.investigation = InvestigationState()
        
        # Track which root causes have been fixed
        self.fixed_root_causes: Set[str] = set()
        
        # Delayed failure tracking - copy to avoid mutating scenario
        self.pending_delayed_failures: Dict[str, int] = dict(self.scenario.delayed_failures)
        self.triggered_delayed_failures: Set[str] = set()
        
        # Delayed action effects (action results that apply later)
        self.pending_effects: List[Dict] = []
        
        # Track last action for repeat detection
        self.last_action: Optional[str] = None
        self.consecutive_same_action = 0
        
        self._initialize_services()
        self._apply_initial_failures()
        self._propagate_failures()
        self._generate_initial_alerts()
    
    def _initialize_services(self):
        """Initialize all services to healthy state."""
        for name in self.SERVICE_NAMES:
            deploy_history = self.scenario.deploy_history.get(name, [])
            last_deploy = deploy_history[-1] if deploy_history else {"time": -120, "version": "v1.0.0"}
            
            self.services[name] = ServiceStatus(
                name=name,
                health=ServiceHealth.HEALTHY,
                error_rate=0.01,
                latency_ms=50,
                cpu_usage=0.3,
                memory_usage=0.4,
                last_deploy_time=last_deploy.get("time", -120),
                last_deploy_version=last_deploy.get("version", "v1.0.0")
            )
    
    def _apply_initial_failures(self):
        """Apply initial failure conditions (NOT delayed ones)."""
        for service_name, config in self.scenario.initial_failures.items():
            # Skip services that are meant to fail later
            if service_name in self.pending_delayed_failures:
                continue
                
            if service_name in self.services:
                service = self.services[service_name]
                service.health = ServiceHealth(config.get("health", "degraded"))
                service.error_rate = config.get("error_rate", 0.5)
                service.latency_ms = config.get("latency_ms", 500)
                service.cpu_usage = config.get("cpu_usage", 0.7)
                service.memory_usage = config.get("memory_usage", 0.8)
    
    def _compute_cascaded_state(self, service_name: str) -> Optional[Dict]:
        """
        Compute the cascaded state for a single service based on its dependencies.
        Returns None if no change needed, or a dict of attribute updates.
        """
        service = self.services[service_name]
        dependencies = self.DEPENDENCIES.get(service_name, [])
        
        if not dependencies:
            return None
        
        # Find worst dependency state
        worst_health = ServiceHealth.HEALTHY
        max_error_contribution = 0.0
        max_latency_contribution = 0
        
        for dep_name in dependencies:
            dep = self.services[dep_name]
            
            if dep.health == ServiceHealth.DOWN:
                worst_health = ServiceHealth.DOWN
            elif dep.health == ServiceHealth.DEGRADED and worst_health != ServiceHealth.DOWN:
                worst_health = ServiceHealth.DEGRADED
            
            # Error contribution from dependency
            max_error_contribution = max(max_error_contribution, dep.error_rate * 0.7)
            max_latency_contribution = max(max_latency_contribution, dep.latency_ms)
        
        # Determine if this service should change
        updates = {}
        
        if worst_health == ServiceHealth.DOWN and service.health != ServiceHealth.DOWN:
            updates["health"] = ServiceHealth.DOWN
            updates["error_rate"] = min(0.95, service.error_rate + 0.6)
            updates["latency_ms"] = max(service.latency_ms, 3000)
        elif worst_health == ServiceHealth.DEGRADED and service.health == ServiceHealth.HEALTHY:
            updates["health"] = ServiceHealth.DEGRADED
            updates["error_rate"] = min(0.95, max(service.error_rate, max_error_contribution + 0.1))
            updates["latency_ms"] = max(service.latency_ms, max_latency_contribution + 50)
        
        return updates if updates else None
    
    def _propagate_failures(self):
        """
        Propagate failures through dependency graph.
        FIXED: Uses atomic state computation to avoid in-loop mutation bugs.
        """
        for _ in range(10):  # Max iterations to reach steady state
            # Compute all new states FIRST (no mutation)
            pending_updates: Dict[str, Dict] = {}
            
            for service_name in self.SERVICE_NAMES:
                # Skip fixed services
                if service_name in self.fixed_root_causes:
                    continue
                # Skip root causes (they don't auto-heal)
                if service_name in self.scenario.root_causes and service_name not in self.fixed_root_causes:
                    # Root causes only change if explicitly fixed
                    continue
                
                updates = self._compute_cascaded_state(service_name)
                if updates:
                    pending_updates[service_name] = updates
            
            # No more changes - steady state reached
            if not pending_updates:
                break
            
            # Apply all updates atomically
            for service_name, updates in pending_updates.items():
                service = self.services[service_name]
                for attr, value in updates.items():
                    setattr(service, attr, value)
    
    def _trigger_delayed_failure(self, service_name: str):
        """Trigger a delayed failure for a service."""
        if service_name not in self.services:
            return
        if service_name in self.triggered_delayed_failures:
            return
        
        self.triggered_delayed_failures.add(service_name)
        
        # Get failure config from scenario
        failure_config = self.scenario.delayed_failure_configs.get(service_name, {
            "health": "degraded",
            "error_rate": 0.6,
            "latency_ms": 800,
            "cpu_usage": 0.7,
            "memory_usage": 0.92
        })
        
        service = self.services[service_name]
        service.health = ServiceHealth(failure_config.get("health", "degraded"))
        service.error_rate = failure_config.get("error_rate", 0.6)
        service.latency_ms = failure_config.get("latency_ms", 800)
        service.cpu_usage = failure_config.get("cpu_usage", 0.7)
        service.memory_usage = failure_config.get("memory_usage", 0.92)
        
        # Add alert
        self.alerts.append(Alert(
            timestamp=self.current_time,
            service=service_name,
            severity=AlertSeverity.CRITICAL,
            message=f"NEW ALERT: {service_name} has started failing"
        ))
        
        # Propagate this new failure
        self._propagate_failures()
    
    def _check_delayed_failures(self):
        """Check and trigger any delayed failures based on step count."""
        for service_name, trigger_step in list(self.pending_delayed_failures.items()):
            if self.step_count >= trigger_step and service_name not in self.triggered_delayed_failures:
                self._trigger_delayed_failure(service_name)
    
    def _apply_pending_effects(self):
        """Apply any pending delayed effects from previous actions."""
        remaining_effects = []
        for effect in self.pending_effects:
            effect["delay"] -= 1
            if effect["delay"] <= 0:
                # Apply the effect
                self._apply_effect(effect)
            else:
                remaining_effects.append(effect)
        self.pending_effects = remaining_effects
    
    def _apply_effect(self, effect: Dict):
        """Apply a delayed effect."""
        service_name = effect.get("service")
        effect_type = effect.get("type")
        
        if service_name not in self.services:
            return
        
        service = self.services[service_name]
        
        if effect_type == "gradual_recovery":
            # Gradual improvement after restart
            if service.health == ServiceHealth.DEGRADED:
                service.error_rate = max(0.02, service.error_rate * 0.5)
                service.latency_ms = max(60, service.latency_ms - 100)
    
    def _evolve_system(self, time_delta: int):
        """Evolve system state over time with temporal dynamics."""
        self.current_time += time_delta
        self.step_count += 1
        
        # Check for delayed failures
        self._check_delayed_failures()
        
        # Apply pending effects
        self._apply_pending_effects()
        
        # Gradual degradation of unfixed root causes
        for rc in self.scenario.root_causes:
            if rc not in self.fixed_root_causes and rc in self.services:
                if rc in self.triggered_delayed_failures or rc not in self.pending_delayed_failures:
                    # Only degrade if failure has been triggered (or was immediate)
                    svc = self.services[rc]
                    if svc.health != ServiceHealth.DOWN:
                        # Slowly worsen over time
                        svc.error_rate = min(0.95, svc.error_rate + 0.015 * time_delta)
                        svc.latency_ms = min(5000, svc.latency_ms + 20 * time_delta)
                        if svc.error_rate > 0.75:
                            svc.health = ServiceHealth.DOWN
                            self._propagate_failures()
    
    def _generate_initial_alerts(self):
        """Generate initial alerts based on service states."""
        for name, service in self.services.items():
            if service.health == ServiceHealth.DOWN:
                self.alerts.append(Alert(
                    timestamp=0,
                    service=name,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Service {name} is DOWN"
                ))
            elif service.health == ServiceHealth.DEGRADED:
                self.alerts.append(Alert(
                    timestamp=0,
                    service=name,
                    severity=AlertSeverity.WARNING,
                    message=f"Service {name} degraded performance"
                ))
            
            if service.error_rate > 0.3:
                self.alerts.append(Alert(
                    timestamp=0,
                    service=name,
                    severity=AlertSeverity.CRITICAL if service.error_rate > 0.6 else AlertSeverity.WARNING,
                    message=f"High error rate on {name}: {service.error_rate*100:.0f}%"
                ))
    
    def _deterministic_hash(self, *args) -> int:
        """Generate deterministic hash for reproducibility."""
        data = f"{self.seed}:{':'.join(str(a) for a in args)}"
        return int(hashlib.md5(data.encode()).hexdigest()[:8], 16)
    
    def _get_investigation_depth(self, service_name: str) -> int:
        """Get how thoroughly a service has been investigated."""
        depth = 0
        if service_name in self.investigation.queried_logs:
            depth += 1
        if service_name in self.investigation.queried_metrics:
            depth += 1
        if service_name in self.investigation.checked_deploys:
            depth += 1
        return depth
    
    def _has_investigated_any_root_cause(self) -> bool:
        """Check if agent has investigated at least one root cause."""
        for rc in self.scenario.root_causes:
            if self._get_investigation_depth(rc) >= 1:
                return True
        return False
    
    def _get_diminishing_factor(self, current_count: int) -> float:
        """
        Get reward multiplier based on query count.
        FIXED: Takes current count as parameter (before increment).
        """
        if current_count == 0:
            return 1.0
        elif current_count == 1:
            return 0.4
        elif current_count == 2:
            return 0.1
        else:
            return 0.0  # No reward for 3+ queries
    
    def _check_repeated_action(self, action_str: str) -> float:
        """Check for repeated actions and return penalty."""
        if action_str == self.last_action:
            self.consecutive_same_action += 1
            if self.consecutive_same_action >= 2:
                return -0.03 * self.consecutive_same_action
        else:
            self.consecutive_same_action = 0
        self.last_action = action_str
        return 0.0
    
    def _generate_logs(self, service_name: str) -> str:
        """Generate deterministic log entries."""
        service = self.services[service_name]
        logs = []
        t = self.current_time
        
        is_root_cause = service_name in self.scenario.root_causes
        is_misleading = service_name in self.scenario.misleading_services
        is_delayed = service_name in self.pending_delayed_failures
        has_triggered = service_name in self.triggered_delayed_failures
        
        # For delayed failures that haven't triggered yet
        if is_delayed and not has_triggered:
            if is_root_cause:
                # Show subtle warning signs before failure
                logs.extend([
                    f"[{t-2}m] INFO: Service operating normally",
                    f"[{t-1}m] DEBUG: Memory allocation slightly elevated",
                    f"[{t}m] INFO: Health check passed (minor GC activity)",
                ])
            else:
                logs.extend([
                    f"[{t-1}m] INFO: Health check passed",
                    f"[{t}m] INFO: Operating normally",
                ])
        elif is_root_cause:
            logs.extend(self._get_root_cause_logs(service_name))
        elif is_misleading:
            logs.extend(self._get_misleading_logs(service_name))
        elif service.health == ServiceHealth.DOWN:
            deps = self.DEPENDENCIES.get(service_name, [])
            dep_str = deps[0] if deps else "upstream"
            logs.extend([
                f"[{t-2}m] ERROR: Connection to {dep_str} failed",
                f"[{t-1}m] ERROR: Health check failed - dependency unavailable",
                f"[{t}m] WARN: This service is affected by upstream issues",
            ])
        elif service.health == ServiceHealth.DEGRADED:
            deps = self.DEPENDENCIES.get(service_name, [])
            if deps:
                logs.extend([
                    f"[{t-2}m] WARN: Upstream {deps[0]} responding slowly",
                    f"[{t-1}m] WARN: Increased latency from dependencies",
                    f"[{t}m] INFO: Consider investigating upstream services",
                ])
            else:
                logs.append(f"[{t}m] INFO: Service operating normally")
        else:
            logs.extend([
                f"[{t-1}m] INFO: Health check passed",
                f"[{t}m] INFO: Operating normally",
            ])
        
        return "\n".join(logs)
    
    def _get_root_cause_logs(self, service_name: str) -> List[str]:
        """Generate revealing logs for root causes."""
        t = self.current_time
        scenario_id = self.scenario.id
        
        if scenario_id == "easy_single_failure":
            if service_name == "db":
                return [
                    f"[{t-5}m] WARN: Connection pool usage high: 98/100",
                    f"[{t-4}m] ERROR: Unable to acquire connection from pool",
                    f"[{t-3}m] ERROR: Query timeout - no available connections",
                    f"[{t-2}m] CRITICAL: Connection pool exhausted",
                    f"[{t-1}m] ERROR: All connection attempts failing",
                ]
        
        elif scenario_id == "medium_cascade":
            if service_name == "redis":
                return [
                    f"[{t-4}m] WARN: Memory usage at 94%",
                    f"[{t-3}m] ERROR: maxmemory limit reached",
                    f"[{t-2}m] ERROR: OOM - rejecting write commands",
                    f"[{t-1}m] CRITICAL: Redis refusing new connections",
                ]
        
        elif scenario_id == "hard_multi_root":
            if service_name == "db":
                return [
                    f"[{t-5}m] WARN: Disk I/O latency elevated: 150ms avg",
                    f"[{t-4}m] WARN: WAL write delays detected",
                    f"[{t-3}m] ERROR: Replication lag growing: 45 seconds",
                    f"[{t-2}m] ERROR: Query timeouts due to I/O saturation",
                    f"[{t-1}m] CRITICAL: Database performance severely degraded",
                ]
            elif service_name == "redis":
                # Only show severe logs after triggered
                if service_name in self.triggered_delayed_failures:
                    return [
                        f"[{t-3}m] WARN: Memory fragmentation ratio: 1.8",
                        f"[{t-2}m] ERROR: Memory limit reached, evictions started",
                        f"[{t-1}m] CRITICAL: Session cache corrupted, auth failing",
                        f"[{t}m] ERROR: Large key causing memory pressure: session:bulk",
                    ]
                else:
                    return [
                        f"[{t-1}m] INFO: Operating normally",
                        f"[{t}m] DEBUG: Memory usage: 45% (healthy)",
                    ]
        
        return [f"[{t-1}m] ERROR: Service experiencing issues"]
    
    def _get_misleading_logs(self, service_name: str) -> List[str]:
        """Generate misleading logs for non-root-cause services."""
        t = self.current_time
        return [
            f"[{t-3}m] ERROR: Request timeout (check UPSTREAM services)",
            f"[{t-2}m] WARN: High latency detected (DEPENDENCY issue likely)",
            f"[{t-1}m] ERROR: Cascade effect from upstream failure",
            f"[{t}m] INFO: Local health OK - issue is NOT in this service",
        ]
    
    def _generate_metrics(self, service_name: str) -> str:
        """Generate deterministic metrics."""
        service = self.services[service_name]
        is_root_cause = service_name in self.scenario.root_causes
        is_delayed = service_name in self.pending_delayed_failures
        has_triggered = service_name in self.triggered_delayed_failures
        
        metrics = [
            f"=== Metrics: {service_name} ===",
            f"Status: {service.health.value.upper()}",
            f"Error Rate: {service.error_rate*100:.1f}%",
            f"Latency P50: {service.latency_ms}ms",
            f"Latency P99: {service.latency_ms * 3}ms",
            f"CPU: {service.cpu_usage*100:.0f}%",
            f"Memory: {service.memory_usage*100:.0f}%",
        ]
        
        # Add root cause specific metrics
        if is_root_cause and (not is_delayed or has_triggered):
            if service_name == "db":
                metrics.append("Disk I/O Wait: 45% (HIGH)")
                metrics.append("Connection Pool: 100/100 (SATURATED)")
                metrics.append("Replication Lag: 45s (CRITICAL)")
            elif service_name == "redis":
                metrics.append("Memory Fragmentation: 1.8 (HIGH)")
                metrics.append("Evicted Keys: 15420 (ACTIVE EVICTION)")
                metrics.append("Connected Clients: 250/256 (NEAR LIMIT)")
        
        # Show healthy metrics for delayed failures not yet triggered
        if is_delayed and not has_triggered:
            metrics = [
                f"=== Metrics: {service_name} ===",
                f"Status: HEALTHY",
                f"Error Rate: 1.0%",
                f"Latency P50: 50ms",
                f"CPU: 30%",
                f"Memory: 45%",
                f"All metrics nominal.",
            ]
        
        deps = self.DEPENDENCIES.get(service_name, [])
        if deps:
            metrics.append("--- Dependencies ---")
            for d in deps:
                dep_svc = self.services[d]
                metrics.append(f"  {d}: {dep_svc.health.value} ({dep_svc.error_rate*100:.0f}% errors)")
        
        return "\n".join(metrics)
    
    def _generate_deploy_info(self, service_name: str) -> str:
        """Generate deployment history."""
        history = self.scenario.deploy_history.get(service_name, [])
        service = self.services[service_name]
        
        lines = [
            f"=== Deploys: {service_name} ===",
            f"Current: {service.last_deploy_version}",
            f"Deployed: {abs(service.last_deploy_time)}m ago",
        ]
        
        if history:
            lines.append("--- History ---")
            for d in history[-3:]:
                lines.append(f"  [{abs(d.get('time', 0))}m ago] {d.get('version')} - {d.get('change', 'No notes')}")
        
        return "\n".join(lines)
    
    def process_action(self, action: Action) -> Tuple[Optional[QueryResult], float, str]:
        """Process an action with evidence-based reward calculation."""
        self.total_actions += 1
        action_str = f"{action.action_type}({action.target_service or ''})"
        self.action_history.append(action_str)
        
        # Time cost and system evolution
        time_cost = self.ACTION_TIME_COSTS.get(action.action_type, 1)
        self._evolve_system(time_cost)
        
        # Base time penalty (progressive)
        time_penalty = -0.004 * (self.current_time / 10)
        
        # Check for repeated actions
        repeat_penalty = self._check_repeated_action(action_str)
        
        reward = time_penalty + repeat_penalty
        if repeat_penalty < 0:
            self.wrong_actions += 1
        
        query_result = None
        reason = "time_passed"
        
        handlers = {
            ActionType.QUERY_LOGS: self._handle_query_logs,
            ActionType.QUERY_METRICS: self._handle_query_metrics,
            ActionType.CHECK_DEPLOYS: self._handle_check_deploys,
            ActionType.RESTART_SERVICE: self._handle_restart,
            ActionType.ROLLBACK_SERVICE: self._handle_rollback,
            ActionType.SCALE_SERVICE: self._handle_scale,
            ActionType.DECLARE_ROOT_CAUSE: self._handle_declare_root_cause,
            ActionType.RESOLVE_INCIDENT: self._handle_resolve,
        }
        
        handler = handlers.get(action.action_type)
        if handler:
            query_result, action_reward, reason = handler(action.target_service)
            reward += action_reward
        
        self.cumulative_reward += reward
        return query_result, reward, reason
    
    def _handle_query_logs(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle log query with proper diminishing returns."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="logs", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.05, "invalid_service"
        
        # FIXED: Get count BEFORE incrementing for correct diminishing
        current_count = self.investigation.query_counts.get(service, 0)
        diminish = self._get_diminishing_factor(current_count)
        
        # NOW increment the count
        self.investigation.query_counts[service] = current_count + 1
        is_first_query = service not in self.investigation.queried_logs
        self.investigation.queried_logs.add(service)
        
        logs = self._generate_logs(service)
        
        # No reward for excessive queries
        if diminish == 0:
            self.wrong_actions += 1
            return QueryResult(
                query_type="logs", service=service, data=logs, timestamp=self.current_time
            ), -0.04, "excessive_queries"
        
        base_reward = 0.01
        reason = "gathered_logs"
        svc = self.services[service]
        
        # Bonus for investigating unhealthy services
        if svc.health != ServiceHealth.HEALTHY:
            base_reward = 0.03
            reason = "investigated_unhealthy"
        
        # Bigger bonus for investigating actual root causes (first time)
        if service in self.scenario.root_causes and is_first_query:
            # Only if the failure has actually triggered
            is_delayed = service in self.pending_delayed_failures
            has_triggered = service in self.triggered_delayed_failures
            if not is_delayed or has_triggered:
                base_reward = 0.07
                reason = "investigated_root_cause"
        
        return QueryResult(
            query_type="logs", service=service, data=logs, timestamp=self.current_time
        ), base_reward * diminish, reason
    
    def _handle_query_metrics(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle metrics query with proper diminishing returns."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="metrics", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.05, "invalid_service"
        
        # FIXED: Get count BEFORE incrementing
        current_count = self.investigation.query_counts.get(service, 0)
        diminish = self._get_diminishing_factor(current_count)
        
        # NOW increment
        self.investigation.query_counts[service] = current_count + 1
        is_first = service not in self.investigation.queried_metrics
        self.investigation.queried_metrics.add(service)
        
        metrics = self._generate_metrics(service)
        
        if diminish == 0:
            self.wrong_actions += 1
            return QueryResult(
                query_type="metrics", service=service, data=metrics, timestamp=self.current_time
            ), -0.04, "excessive_queries"
        
        base_reward = 0.01
        reason = "gathered_metrics"
        
        if self.services[service].health != ServiceHealth.HEALTHY:
            base_reward = 0.02
        
        if service in self.scenario.root_causes and is_first:
            is_delayed = service in self.pending_delayed_failures
            has_triggered = service in self.triggered_delayed_failures
            if not is_delayed or has_triggered:
                base_reward = 0.05
                reason = "investigated_root_cause"
        
        return QueryResult(
            query_type="metrics", service=service, data=metrics, timestamp=self.current_time
        ), base_reward * diminish, reason
    
    def _handle_check_deploys(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle deploy check with proper first-check reward."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="deploys", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.05, "invalid_service"
        
        # FIXED: Check if first BEFORE adding to set
        is_first_check = service not in self.investigation.checked_deploys
        self.investigation.checked_deploys.add(service)
        
        deploy_info = self._generate_deploy_info(service)
        
        # Reward only for first check
        reward = 0.02 if is_first_check else 0.0
        reason = "checked_deploys" if is_first_check else "duplicate_deploy_check"
        
        return QueryResult(
            query_type="deploys", service=service, data=deploy_info, timestamp=self.current_time
        ), reward, reason
    
    def _handle_restart(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle restart with evidence-based rewards and consistent penalties."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="restart", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.08, "invalid_service"
        
        self.remediation_actions.append({"action": "restart", "service": service, "time": self.current_time})
        
        investigation_depth = self._get_investigation_depth(service)
        investigated_any_rc = self._has_investigated_any_root_cause()
        
        # PENALTY: Acting without any investigation
        if investigation_depth == 0 and not investigated_any_rc:
            self.wrong_actions += 1
            return QueryResult(
                query_type="restart", service=service,
                data="WARNING: Restarting without investigation is unprofessional!",
                timestamp=self.current_time
            ), -0.15, "premature_action"
        
        # Check if this is the correct remediation for a root cause
        is_correct_remediation = any(
            r.get("action") in ["restart", "restart_service"] and r.get("service") == service
            for r in self.scenario.required_remediations
        )
        
        svc = self.services[service]
        
        if is_correct_remediation and service in self.scenario.root_causes:
            # Check if failure has triggered (for delayed failures)
            is_delayed = service in self.pending_delayed_failures
            has_triggered = service in self.triggered_delayed_failures
            
            if is_delayed and not has_triggered:
                # Trying to fix something that hasn't failed yet
                self.wrong_actions += 1
                return QueryResult(
                    query_type="restart", service=service,
                    data=f"Service {service} is healthy. Unnecessary restart.",
                    timestamp=self.current_time
                ), -0.08, "premature_fix"
            
            # Fix the root cause
            svc.health = ServiceHealth.HEALTHY
            svc.error_rate = 0.02
            svc.latency_ms = 55
            svc.cpu_usage = 0.35
            svc.memory_usage = 0.45
            self.fixed_root_causes.add(service)
            self._propagate_failures()
            self._heal_dependents(service)
            
            # Bonus for thorough investigation before fixing
            investigation_bonus = 0.04 if investigation_depth >= 2 else 0.0
            
            return QueryResult(
                query_type="restart", service=service,
                data=f"SUCCESS: {service} restarted and recovered.",
                timestamp=self.current_time
            ), 0.18 + investigation_bonus, "correct_remediation"
        
        elif svc.health != ServiceHealth.HEALTHY:
            # Restarting a symptomatic (but not root cause) service
            self.wrong_actions += 1
            return QueryResult(
                query_type="restart", service=service,
                data=f"Restarted {service} but issues persist. Check upstream dependencies.",
                timestamp=self.current_time
            ), -0.08, "wrong_target"
        
        else:
            # Restarting a healthy service
            self.wrong_actions += 1
            return QueryResult(
                query_type="restart", service=service,
                data=f"WARNING: {service} was healthy. Unnecessary restart caused brief disruption.",
                timestamp=self.current_time
            ), -0.10, "unnecessary_restart"
    
    def _handle_rollback(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle rollback with evidence-based rewards and consistent penalties."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="rollback", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.08, "invalid_service"
        
        self.remediation_actions.append({"action": "rollback", "service": service, "time": self.current_time})
        
        # PENALTY: Rollback without checking deploys first
        if service not in self.investigation.checked_deploys:
            self.wrong_actions += 1
            return QueryResult(
                query_type="rollback", service=service,
                data="ERROR: Rollback attempted without checking deploy history first!",
                timestamp=self.current_time
            ), -0.12, "rollback_without_deploy_check"
        
        history = self.scenario.deploy_history.get(service, [])
        if len(history) < 2:
            self.wrong_actions += 1
            return QueryResult(
                query_type="rollback", service=service,
                data="ERROR: No previous version available for rollback.",
                timestamp=self.current_time
            ), -0.05, "no_rollback_target"
        
        is_correct_remediation = any(
            r.get("action") in ["rollback", "rollback_service"] and r.get("service") == service
            for r in self.scenario.required_remediations
        )
        
        svc = self.services[service]
        prev_version = history[-2].get("version", "previous")
        
        if is_correct_remediation and service in self.scenario.root_causes:
            svc.health = ServiceHealth.HEALTHY
            svc.error_rate = 0.01
            svc.latency_ms = 50
            svc.last_deploy_version = prev_version
            self.fixed_root_causes.add(service)
            self._propagate_failures()
            self._heal_dependents(service)
            
            return QueryResult(
                query_type="rollback", service=service,
                data=f"SUCCESS: Rolled back {service} to {prev_version}. Service recovered.",
                timestamp=self.current_time
            ), 0.18, "correct_rollback"
        else:
            self.wrong_actions += 1
            svc.last_deploy_version = prev_version
            return QueryResult(
                query_type="rollback", service=service,
                data=f"Rolled back {service} to {prev_version} but issues persist.",
                timestamp=self.current_time
            ), -0.08, "wrong_rollback"
    
    def _handle_scale(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle scaling with evidence-based rewards and consistent penalties."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="scale", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.08, "invalid_service"
        
        self.remediation_actions.append({"action": "scale", "service": service, "time": self.current_time})
        
        investigation_depth = self._get_investigation_depth(service)
        if investigation_depth == 0:
            self.wrong_actions += 1
            return QueryResult(
                query_type="scale", service=service,
                data="WARNING: Scaling without investigation!",
                timestamp=self.current_time
            ), -0.10, "premature_scale"
        
        is_correct_remediation = any(
            r.get("action") in ["scale", "scale_service"] and r.get("service") == service
            for r in self.scenario.required_remediations
        )
        
        svc = self.services[service]
        
        if is_correct_remediation and service in self.scenario.root_causes:
            svc.health = ServiceHealth.HEALTHY
            svc.error_rate = 0.02
            svc.latency_ms = 50
            svc.cpu_usage = 0.4
            svc.memory_usage = 0.5
            self.fixed_root_causes.add(service)
            self._propagate_failures()
            self._heal_dependents(service)
            
            return QueryResult(
                query_type="scale", service=service,
                data=f"SUCCESS: Scaled {service} from 3 to 6 replicas. Capacity restored.",
                timestamp=self.current_time
            ), 0.18, "correct_scale"
        else:
            self.wrong_actions += 1
            return QueryResult(
                query_type="scale", service=service,
                data=f"Scaled {service} but underlying issue persists.",
                timestamp=self.current_time
            ), -0.06, "wrong_scale"
    
    def _handle_declare_root_cause(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle root cause declaration with evidence requirements."""
        if service not in self.services:
            self.wrong_actions += 1
            return QueryResult(
                query_type="declare", service=service or "unknown",
                data="ERROR: Unknown service", timestamp=self.current_time
            ), -0.08, "invalid_service"
        
        if service in self.declared_root_causes:
            self.wrong_actions += 1
            return QueryResult(
                query_type="declare", service=service,
                data="Already declared as root cause.",
                timestamp=self.current_time
            ), -0.03, "duplicate_declaration"
        
        self.declared_root_causes.append(service)
        investigation_depth = self._get_investigation_depth(service)
        
        if service in self.scenario.root_causes:
            # Check if this root cause has actually manifested
            is_delayed = service in self.pending_delayed_failures
            has_triggered = service in self.triggered_delayed_failures
            
            if is_delayed and not has_triggered:
                # Declaring something as root cause before it failed
                self.wrong_actions += 1
                return QueryResult(
                    query_type="declare", service=service,
                    data=f"Declared {service} as root cause (currently healthy though...)",
                    timestamp=self.current_time
                ), -0.06, "premature_declaration"
            
            # Correct declaration - bonus for investigation
            investigation_bonus = 0.05 * min(investigation_depth, 2)
            return QueryResult(
                query_type="declare", service=service,
                data=f"Correctly identified {service} as root cause.",
                timestamp=self.current_time
            ), 0.12 + investigation_bonus, "correct_root_cause"
        else:
            # Wrong declaration
            self.wrong_actions += 1
            penalty = -0.12 if investigation_depth == 0 else -0.08
            return QueryResult(
                query_type="declare", service=service,
                data=f"Declared {service} as root cause (this may be incorrect).",
                timestamp=self.current_time
            ), penalty, "incorrect_root_cause"
    
    def _handle_resolve(self, service: str) -> Tuple[QueryResult, float, str]:
        """Handle incident resolution."""
        all_healthy = all(s.health == ServiceHealth.HEALTHY for s in self.services.values())
        
        # For hard mode, check if all delayed failures have either triggered and been fixed, or not triggered yet
        all_root_causes_handled = True
        for rc in self.scenario.root_causes:
            is_delayed = rc in self.pending_delayed_failures
            has_triggered = rc in self.triggered_delayed_failures
            is_fixed = rc in self.fixed_root_causes
            
            if is_delayed and not has_triggered:
                # This failure hasn't happened yet - that's OK for now
                continue
            elif not is_fixed:
                all_root_causes_handled = False
                break
        
        all_declared = all(rc in self.declared_root_causes for rc in self.scenario.root_causes 
                         if rc not in self.pending_delayed_failures or rc in self.triggered_delayed_failures)
        
        if all_healthy and all_root_causes_handled and all_declared:
            self.incident_resolved = True
            return QueryResult(
                query_type="resolve", service="incident",
                data="INCIDENT RESOLVED: All services healthy, root causes documented.",
                timestamp=self.current_time
            ), 0.22, "fully_resolved"
        
        elif all_healthy and all_root_causes_handled:
            self.incident_resolved = True
            return QueryResult(
                query_type="resolve", service="incident",
                data="Resolved but root cause documentation incomplete.",
                timestamp=self.current_time
            ), 0.10, "resolved_incomplete"
        
        elif all_healthy:
            # Services are healthy but we didn't do proper remediation
            self.incident_resolved = True
            return QueryResult(
                query_type="resolve", service="incident",
                data="Services appear healthy but root cause unclear.",
                timestamp=self.current_time
            ), 0.04, "resolved_unclear"
        
        else:
            self.wrong_actions += 1
            unhealthy = [n for n, s in self.services.items() if s.health != ServiceHealth.HEALTHY]
            return QueryResult(
                query_type="resolve", service="incident",
                data=f"CANNOT RESOLVE: Unhealthy services: {unhealthy}",
                timestamp=self.current_time
            ), -0.15, "premature_resolution"
    
    def _heal_dependents(self, healed: str):
        """Heal dependent services after fixing a root cause."""
        for dep_name in self.DEPENDENTS.get(healed, []):
            # Don't auto-heal unfixed root causes
            if dep_name in self.scenario.root_causes and dep_name not in self.fixed_root_causes:
                continue
            
            dep = self.services[dep_name]
            
            # Check if ALL dependencies are now healthy
            all_deps_ok = all(
                self.services[d].health == ServiceHealth.HEALTHY
                for d in self.DEPENDENCIES.get(dep_name, [])
            )
            
            if all_deps_ok and dep.health != ServiceHealth.HEALTHY:
                dep.health = ServiceHealth.HEALTHY
                dep.error_rate = max(0.02, dep.error_rate * 0.2)
                dep.latency_ms = max(55, dep.latency_ms - 300)
                # Recursively heal
                self._heal_dependents(dep_name)
    
    def get_observation(self) -> dict:
        """Get current observation state."""
        return {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "services": {n: s.model_dump() for n, s in self.services.items()},
            "alerts": [a.model_dump() for a in self.alerts[-8:]],
            "action_history": self.action_history[-10:],
            "declared_root_causes": self.declared_root_causes.copy(),
            "incident_resolved": self.incident_resolved,
            "task_id": self.scenario.id,
            "task_description": self.scenario.description,
        }
    
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.incident_resolved or self.current_time >= self.scenario.time_limit