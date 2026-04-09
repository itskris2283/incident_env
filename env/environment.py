"""
Main IncidentCommanderEnv environment.
UPDATED: Pass triggered_delayed_failures to grader.
"""

from typing import Dict, Any, Optional
from .models import Action, ActionType, GradeResult, Scenario
from .simulator import ServiceSimulator
from .scenarios import get_scenario, list_scenarios
from .grader import grade_trajectory


class IncidentCommanderEnv:
    """
    OpenEnv-compatible IT incident response environment.
    """
    
    VALID_SERVICES = ["frontend", "api", "auth", "db", "redis"]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.simulator: Optional[ServiceSimulator] = None
        self.current_scenario: Optional[Scenario] = None
        self.last_query_result = None
        self._started = False
    
    def reset(self, task_id: str = "easy_single_failure") -> Dict[str, Any]:
        """Reset environment with specified task."""
        self.current_scenario = get_scenario(task_id)
        self.simulator = ServiceSimulator(self.current_scenario, seed=self.seed)
        self.last_query_result = None
        self._started = True
        return self._build_observation()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action and return result."""
        if not self._started or self.simulator is None:
            raise RuntimeError("Call reset() first")
        
        parsed = self._parse_action(action)
        query_result, reward, reason = self.simulator.process_action(parsed)
        self.last_query_result = query_result
        
        done = self.simulator.is_done()
        obs = self._build_observation()
        
        if query_result:
            obs["last_query_result"] = query_result.model_dump()
        
        return {
            "observation": obs,
            "reward": {
                "value": round(reward, 4),
                "reason": reason,
                "cumulative": round(self.simulator.cumulative_reward, 4)
            },
            "done": done,
            "info": {
                "time_remaining": self.current_scenario.time_limit - self.simulator.current_time,
                "healthy_services": sum(1 for s in self.simulator.services.values() if s.health.value == "healthy"),
                "total_services": len(self.simulator.services),
                "step_count": self.simulator.step_count
            }
        }
    
    def state(self) -> Dict[str, Any]:
        """Get current observation."""
        if not self._started:
            return {"error": "Call reset() first"}
        obs = self._build_observation()
        if self.last_query_result:
            obs["last_query_result"] = self.last_query_result.model_dump()
        return obs
    
    def grade(self) -> Dict[str, Any]:
        """Grade the trajectory."""
        if self.simulator is None or self.current_scenario is None:
            return {"error": "No trajectory to grade"}
        
        investigation = {
            "queried_logs": self.simulator.investigation.queried_logs,
            "queried_metrics": self.simulator.investigation.queried_metrics,
            "checked_deploys": self.simulator.investigation.checked_deploys,
        }
        
        result = grade_trajectory(
            scenario=self.current_scenario,
            declared_root_causes=self.simulator.declared_root_causes,
            remediation_actions=self.simulator.remediation_actions,
            action_history=self.simulator.action_history,
            final_time=self.simulator.current_time,
            incident_resolved=self.simulator.incident_resolved,
            wrong_actions=self.simulator.wrong_actions,
            investigation=investigation,
            fixed_root_causes=self.simulator.fixed_root_causes,
            triggered_delayed_failures=self.simulator.triggered_delayed_failures
        )
        grade = result.model_dump()

        # Defensive clamp at API boundary for strict-open score validators.
        def _strict_open(value: float) -> float:
            eps = 1e-4
            if value <= 0.0:
                return eps
            if value >= 1.0:
                return 1.0 - eps
            return value

        for key in [
            "score",
            "root_cause_score",
            "remediation_score",
            "investigation_score",
            "efficiency_score",
            "penalty_score",
        ]:
            if key in grade:
                grade[key] = _strict_open(float(grade[key]))

        return grade
    
    def get_tasks(self) -> list:
        """List available tasks."""
        return list_scenarios()
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space spec."""
        return {
            "action_types": [t.value for t in ActionType],
            "services": self.VALID_SERVICES,
            "examples": [
                {"action_type": "query_logs", "target_service": "db"},
                {"action_type": "restart_service", "target_service": "redis"},
                {"action_type": "declare_root_cause", "target_service": "db"},
                {"action_type": "resolve_incident"},
            ]
        }
    
    def _parse_action(self, action: Dict[str, Any]) -> Action:
        """Parse action dictionary."""
        action_type_str = action.get("action_type", "")
        
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            raise ValueError(f"Invalid action_type: {action_type_str}")
        
        target = action.get("target_service")
        if target and target not in self.VALID_SERVICES:
            raise ValueError(f"Invalid service: {target}")
        
        return Action(
            action_type=action_type,
            target_service=target,
            parameters=action.get("parameters", {})
        )
    
    def _build_observation(self) -> Dict[str, Any]:
        """Build observation dict."""
        if self.simulator is None:
            return {}
        return self.simulator.get_observation()


def create_env(seed: int = 42) -> IncidentCommanderEnv:
    """Create environment instance."""
    return IncidentCommanderEnv(seed=seed)