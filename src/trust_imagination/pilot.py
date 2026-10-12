"""M0: measured expert baseline, rendering and branch reproducibility.

This is simulator infrastructure validation, not a world-model/VLA result.
"""

from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import metaworld
import metaworld.policies
import numpy as np

from .branching import capture, restore, integration_state

TASKS = {
    "reach-v3": "SawyerReachV3Policy",
    "push-v3": "SawyerPushV3Policy",
    "door-open-v3": "SawyerDoorOpenV3Policy",
}


def make_environment(task: str, seed: int):
    benchmark = metaworld.ML1(task, seed=seed)
    env = benchmark.train_classes[task](
        render_mode="rgb_array", width=224, height=224, camera_name="corner2"
    )
    env.set_task(benchmark.train_tasks[seed % len(benchmark.train_tasks)])
    # Scripted expert instrumentation needs goal coordinates. ML1 otherwise
    # zeros these entries; this privileged observation is never a VLA input.
    env._partially_observable = False
    env.__dict__.pop("sawyer_observation_space", None)
    policy = getattr(metaworld.policies, TASKS[task])()
    return env, policy


def rollout_actions(env, actions):
    observations, rewards, outcomes, states = [], [], [], []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())
        rewards.append(float(reward))
        outcomes.append(
            (bool(terminated), bool(truncated), float(info.get("success", 0)))
        )
        states.append(integration_state(env))
    return (
        np.stack(observations),
        np.asarray(rewards),
        np.asarray(outcomes),
        np.stack(states),
        env.render().copy(),
    )


def check_branch(env, obs, policy, steps: int = 12):
    start = capture(env)
    initial_frame = env.render().copy()
    actions = []
    planning_policy = copy.deepcopy(policy)
    for _ in range(steps):
        action = np.asarray(planning_policy.get_action(obs), dtype=np.float64)
        actions.append(action.copy())
        obs, *_ = env.step(action)
    restore(env, start)
    first = rollout_actions(env, actions)
    # Deliberately move away before restoring for the second branch.
    for _ in range(7):
        env.step(np.array([0.8, -0.7, 0.2, -0.3]))
    restore(env, start)
    restored_frame = env.render().copy()
    second = rollout_actions(env, actions)
    errors = {
        name: float(np.max(np.abs(a.astype(float) - b.astype(float))))
        for name, a, b in zip(
            ("observation", "reward", "outcome", "integration", "final_frame"),
            first,
            second,
        )
    }
    errors["initial_frame"] = float(
        np.max(np.abs(initial_frame.astype(float) - restored_frame.astype(float)))
    )
    passed = all(
        errors[name] <= 1e-9
        for name in ("observation", "reward", "outcome", "integration")
    )
    passed = passed and errors["initial_frame"] == 0 and errors["final_frame"] == 0
    restore(env, start)
    return {"passed": passed, "steps": steps, "max_abs_errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {
        "stage": "M0_simulator_validation",
        "expert_privileged_goal_observation": True,
        "packages": {},
        "tasks": [],
    }
    for package in ("metaworld", "mujoco", "numpy"):
        results["packages"][package] = importlib.metadata.version(package)
    started = time.perf_counter()
    try:
        for task in args.tasks:
            for seed in range(args.episodes):
                env, policy = make_environment(task, seed)
                try:
                    obs, _ = env.reset(seed=seed)
                    branch = check_branch(env, obs.copy(), policy)
                    if not branch["passed"]:
                        results["tasks"].append(
                            {"task": task, "seed": seed, "branch": branch}
                        )
                        raise RuntimeError(
                            f"Branch reproducibility failed for {task} seed {seed}: {branch}"
                        )
                    # Reset for a complete scripted-policy baseline after the restoration test.
                    obs, _ = env.reset(seed=seed)
                    policy = getattr(metaworld.policies, TASKS[task])()
                    frames, total_reward, success = [], 0.0, False
                    later_branches = []
                    ep_start = time.perf_counter()
                    for step in range(250):
                        if step in (30, 60):
                            later = check_branch(env, obs.copy(), policy)
                            later_branches.append({"environment_step": step, **later})
                            if not later["passed"]:
                                results["tasks"].append(
                                    {
                                        "task": task,
                                        "seed": seed,
                                        "branch": branch,
                                        "later_branches": later_branches,
                                    }
                                )
                                raise RuntimeError(
                                    f"Later-state branch failed for {task} at step {step}: {later}"
                                )
                        if seed == 0 and step % 3 == 0:
                            frames.append(env.render().copy())
                        obs, reward, terminated, truncated, info = env.step(
                            policy.get_action(obs)
                        )
                        total_reward += float(reward)
                        success = success or bool(info.get("success", False))
                        if success or terminated or truncated:
                            break
                    seconds = time.perf_counter() - ep_start
                    row = {
                        "task": task,
                        "seed": seed,
                        "branch": branch,
                        "expert_ever_success": success,
                        "later_branches": later_branches,
                        "steps": step + 1,
                        "reward_sum": total_reward,
                        "seconds": seconds,
                        "frame_shape": list(env.render().shape),
                    }
                    if frames:
                        imageio.mimsave(
                            args.output / f"{task}_expert.mp4",
                            frames,
                            fps=20,
                            macro_block_size=1,
                        )
                    results["tasks"].append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    env.close()
        results["expert_success_count"] = sum(
            row["expert_ever_success"] for row in results["tasks"]
        )
        if results["expert_success_count"] != len(results["tasks"]):
            raise RuntimeError(
                "Scripted expert failed at least one pilot episode; inspect task outcomes."
            )
        results["status"] = "passed"
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = repr(exc)
        raise
    finally:
        results["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
