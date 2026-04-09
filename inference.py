#!/usr/bin/env python3
"""
Inference script for IncidentCommanderEnv.
Uses HuggingFace Inference API.

OUTPUT FORMAT (STRICT):
- [START] task_id
- [STEP] step_num|action|reward|cumulative
- [END] score

NO other output to stdout.
"""

import os
import sys
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from env import IncidentCommanderEnv


# Load environment variables from a local .env file if present.
load_dotenv()


def get_env_var(name: str, required: bool = True) -> Optional[str]:
    """Get environment variable."""
    val = os.environ.get(name)
    if required and not val:
        print(f"ERROR: {name} environment variable required", file=sys.stderr)
        sys.exit(1)
    return val


def create_client():
    """Create HuggingFace inference client."""
    from openai import OpenAI
    
    api_base = get_env_var("API_BASE_URL", required=False) or "https://router.huggingface.co/v1"
    token = get_env_var("HF_TOKEN", required=True)
    
    return OpenAI(base_url=api_base, api_key=token)


def get_system_prompt() -> str:
    """System prompt for the agent."""
    return """You are an expert SRE incident commander. You must diagnose and resolve a production incident.

ARCHITECTURE:
frontend → api → auth → db
                 ↓
                redis

ACTIONS (respond with JSON only):
- {"action_type": "query_logs", "target_service": "SERVICE"}
- {"action_type": "query_metrics", "target_service": "SERVICE"}
- {"action_type": "check_deploys", "target_service": "SERVICE"}
- {"action_type": "restart_service", "target_service": "SERVICE"}
- {"action_type": "rollback_service", "target_service": "SERVICE"}
- {"action_type": "scale_service", "target_service": "SERVICE"}
- {"action_type": "declare_root_cause", "target_service": "SERVICE"}
- {"action_type": "resolve_incident"}

VALID SERVICES: frontend, api, auth, db, redis

STRATEGY:
1. First investigate: query logs/metrics of unhealthy services
2. Trace dependencies to find ROOT CAUSE (deepest failing service)
3. Declare root cause AFTER investigation
4. Apply remediation (restart/rollback/scale)
5. Resolve incident when all healthy

RESPOND WITH JSON ONLY. NO EXPLANATION."""


def format_observation(obs: Dict[str, Any]) -> str:
    """Format observation for LLM."""
    lines = [f"TIME: {obs.get('current_time', 0)} minutes"]
    lines.append(f"TASK: {obs.get('task_description', '')}")
    lines.append("")
    lines.append("SERVICES:")
    
    for name, svc in obs.get("services", {}).items():
        h = svc.get("health", "unknown").upper()
        e = svc.get("error_rate", 0) * 100
        l = svc.get("latency_ms", 0)
        lines.append(f"  {name}: {h} | err={e:.0f}% | lat={l}ms")
    
    alerts = obs.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("ALERTS:")
        for a in alerts[-4:]:
            lines.append(f"  [{a.get('severity')}] {a.get('service')}: {a.get('message')}")
    
    result = obs.get("last_query_result")
    if result:
        lines.append("")
        lines.append(f"LAST QUERY ({result.get('query_type')}: {result.get('service')}):")
        lines.append(result.get("data", "")[:500])
    
    history = obs.get("action_history", [])
    if history:
        lines.append("")
        lines.append(f"ACTIONS TAKEN: {len(history)}")
        for h in history[-5:]:
            lines.append(f"  - {h}")
    
    declared = obs.get("declared_root_causes", [])
    if declared:
        lines.append(f"DECLARED ROOT CAUSES: {declared}")
    
    return "\n".join(lines)


def parse_action(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response to extract action JSON."""
    response = response.strip()
    
    # Remove markdown code blocks
    if "```" in response:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    
    # Find JSON object
    match = re.search(r'\{[^{}]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def run_episode(env: IncidentCommanderEnv, client, model: str, task_id: str, max_steps: int = 20):
    """Run a single episode."""
    def strict_open(value: float) -> float:
        eps = 1e-4
        if value <= 0.0:
            return eps
        if value >= 1.0:
            return 1.0 - eps
        return value

    # [START] output
    print(f"[START] {task_id}")
    
    obs = env.reset(task_id)
    done = False
    step = 0
    
    while not done and step < max_steps:
        step += 1
        obs_text = format_observation(obs)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": obs_text}
                ],
                max_tokens=100,
                temperature=0.1
            )
            llm_output = response.choices[0].message.content
            action = parse_action(llm_output)
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            action = None
        
        if action is None:
            # Fallback action
            action = {"action_type": "query_logs", "target_service": "db"}
            print(f"Parse failed, using fallback", file=sys.stderr)
        
        try:
            result = env.step(action)
        except ValueError as e:
            print(f"Invalid action: {e}", file=sys.stderr)
            action = {"action_type": "query_logs", "target_service": "db"}
            result = env.step(action)
        
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        
        # [STEP] output
        action_str = f"{action.get('action_type')}({action.get('target_service', '')})"
        print(f"[STEP] {step}|{action_str}|{reward['value']:.4f}|{reward['cumulative']:.4f}")
    
    # Grade and [END] output
    grade = env.grade()
    grade["score"] = strict_open(float(grade["score"]))
    print(f"[END] {grade['score']:.4f}")
    
    return grade


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None,
                       choices=["easy_single_failure", "medium_cascade", "hard_multi_root"],
                       help="Run a single task. If omitted, runs all tasks.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--all-tasks", action="store_true",
                       help="Run all tasks.")
    args = parser.parse_args()
    
    # Get model from env or args
    model = args.model or get_env_var("MODEL_NAME", required=False) or "Qwen/Qwen2.5-7B-Instruct"
    
    client = create_client()
    env = IncidentCommanderEnv(seed=42)
    
    print(f"Using model: {model}", file=sys.stderr)
    
    all_tasks = ["easy_single_failure", "medium_cascade", "hard_multi_root"]
    tasks = [args.task] if args.task and not args.all_tasks else all_tasks
    
    results = []
    for task in tasks:
        grade = run_episode(env, client, model, task, args.max_steps)
        results.append({"task": task, "score": grade["score"]})
    
    if len(results) > 1:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"Average: {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()