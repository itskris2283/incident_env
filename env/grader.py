"""
Deterministic grading logic for IncidentCommanderEnv.
FIXED: Proper handling of delayed failures in scoring.
"""

from typing import Dict, List, Set
from .models import GradeResult, Scenario


class IncidentGrader:
    """Deterministic grader for incident response trajectories."""
    
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
    
    def grade(
        self,
        declared_root_causes: List[str],
        remediation_actions: List[Dict],
        action_history: List[str],
        final_time: int,
        incident_resolved: bool,
        wrong_actions: int,
        investigation: Dict[str, Set[str]],
        fixed_root_causes: Set[str],
        triggered_delayed_failures: Set[str]
    ) -> GradeResult:
        """
        Compute final grade.
        
        Components:
        - Root cause accuracy (25%)
        - Remediation correctness (25%)
        - Investigation quality (20%)
        - Efficiency (15%)
        - Penalty avoidance (15%)
        """
        
        # Determine which root causes were actually "active" during the episode
        active_root_causes = []
        for rc in self.scenario.root_causes:
            if rc in self.scenario.delayed_failures:
                # Only count if it triggered
                if rc in triggered_delayed_failures:
                    active_root_causes.append(rc)
            else:
                # Immediate failure - always active
                active_root_causes.append(rc)
        
        rc_score = self._score_root_causes(declared_root_causes, active_root_causes)
        rem_score = self._score_remediation(fixed_root_causes, active_root_causes)
        inv_score = self._score_investigation(investigation, active_root_causes)
        eff_score = self._score_efficiency(len(action_history), final_time, incident_resolved)
        pen_score = self._score_penalties(wrong_actions, len(action_history))
        
        # Weighted final score
        final = (
            0.25 * rc_score +
            0.25 * rem_score +
            0.20 * inv_score +
            0.15 * eff_score +
            0.15 * pen_score
        )
        
        # Clamp to strict open interval (0, 1) for submission validators.
        eps = 1e-4
        if final <= 0.0:
            final = eps
        elif final >= 1.0:
            final = 1.0 - eps
        
        return GradeResult(
            score=round(final, 4),
            root_cause_score=round(rc_score, 4),
            remediation_score=round(rem_score, 4),
            investigation_score=round(inv_score, 4),
            efficiency_score=round(eff_score, 4),
            penalty_score=round(pen_score, 4),
            breakdown={
                "declared_root_causes": declared_root_causes,
                "expected_root_causes": self.scenario.root_causes,
                "active_root_causes": active_root_causes,
                "fixed_root_causes": list(fixed_root_causes),
                "total_actions": len(action_history),
                "optimal_actions": self.scenario.optimal_action_count,
                "final_time": final_time,
                "time_limit": self.scenario.time_limit,
                "wrong_actions": wrong_actions,
                "incident_resolved": incident_resolved,
            }
        )
    
    def _score_root_causes(self, declared: List[str], active: List[str]) -> float:
        """F1 score for root cause identification against active root causes."""
        expected = set(active)
        declared_set = set(declared)
        
        if not expected:
            return 1.0 if not declared_set else 0.5
        
        if not declared_set:
            return 0.0
        
        tp = len(expected & declared_set)
        fp = len(declared_set - expected)
        
        precision = tp / len(declared_set) if declared_set else 0.0
        recall = tp / len(expected) if expected else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        # Additional penalty for false positives
        fp_penalty = min(0.2, fp * 0.1)
        
        return max(0.0, f1 - fp_penalty)
    
    def _score_remediation(self, fixed: Set[str], active: List[str]) -> float:
        """Score based on how many active root causes were fixed."""
        expected = set(active)
        
        if not expected:
            return 1.0
        
        fixed_correctly = len(expected & fixed)
        return fixed_correctly / len(expected)
    
    def _score_investigation(self, investigation: Dict[str, Set[str]], active: List[str]) -> float:
        """Score based on investigation coverage of active root causes."""
        root_causes = set(active)
        
        if not root_causes:
            return 1.0
        
        queried_logs = investigation.get("queried_logs", set())
        queried_metrics = investigation.get("queried_metrics", set())
        
        total_points = 0
        max_points = len(root_causes) * 2  # logs + metrics per root cause
        
        for rc in root_causes:
            if rc in queried_logs:
                total_points += 1
            if rc in queried_metrics:
                total_points += 1
        
        return total_points / max_points if max_points > 0 else 0.0
    
    def _score_efficiency(self, total_actions: int, final_time: int, resolved: bool) -> float:
        """Score based on action count and time efficiency."""
        if not resolved:
            return 0.0
        
        optimal = self.scenario.optimal_action_count
        time_limit = self.scenario.time_limit
        
        # Action efficiency
        if total_actions <= optimal:
            action_eff = 1.0
        else:
            over = total_actions - optimal
            action_eff = max(0.0, 1.0 - (over * 0.06))
        
        # Time efficiency
        time_ratio = final_time / time_limit
        time_eff = max(0.0, 1.0 - time_ratio)
        
        return 0.6 * action_eff + 0.4 * time_eff
    
    def _score_penalties(self, wrong_actions: int, total_actions: int) -> float:
        """Score based on avoiding wrong actions."""
        if total_actions == 0:
            return 0.5
        
        wrong_ratio = wrong_actions / total_actions
        return max(0.0, 1.0 - wrong_ratio * 1.5)


def grade_trajectory(
    scenario: Scenario,
    declared_root_causes: List[str],
    remediation_actions: List[Dict],
    action_history: List[str],
    final_time: int,
    incident_resolved: bool,
    wrong_actions: int,
    investigation: Dict[str, Set[str]],
    fixed_root_causes: Set[str],
    triggered_delayed_failures: Set[str] = None
) -> GradeResult:
    """Convenience function to grade a trajectory."""
    if triggered_delayed_failures is None:
        triggered_delayed_failures = set()
    
    grader = IncidentGrader(scenario)
    return grader.grade(
        declared_root_causes=declared_root_causes,
        remediation_actions=remediation_actions,
        action_history=action_history,
        final_time=final_time,
        incident_resolved=incident_resolved,
        wrong_actions=wrong_actions,
        investigation=investigation,
        fixed_root_causes=fixed_root_causes,
        triggered_delayed_failures=triggered_delayed_failures
    )