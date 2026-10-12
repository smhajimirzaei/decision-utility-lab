"""Full MuJoCo branch restoration for raw Meta-World environments.

Snapshots are privileged offline instrumentation. They are not policy inputs.
This adapter is limited to unwrapped Meta-World tasks and must pass replay
validation on each task/version before any labels are trusted.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

MODEL_FIELDS = (
    "body_pos",
    "body_quat",
    "site_pos",
    "site_quat",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "body_mass",
    "body_inertia",
    "eq_data",
)


def _plain(value: Any) -> bool:
    if value is None or isinstance(
        value, (str, bool, int, float, np.generic, np.ndarray, np.random.Generator)
    ):
        return True
    if isinstance(value, (list, tuple)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _plain(v) for k, v in value.items())
    return False


@dataclass
class Snapshot:
    integration: np.ndarray
    data: Any
    python_fields: dict[str, Any]
    model_fields: dict[str, np.ndarray]
    numpy_random: tuple
    python_random: tuple
    warmstart: np.ndarray


def capture(env) -> Snapshot:
    if env is not env.unwrapped:
        raise ValueError("Use the raw environment; wrapper state is not captured.")
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, spec)
    fields = {
        key: copy.deepcopy(value) for key, value in vars(env).items() if _plain(value)
    }
    return Snapshot(
        state,
        copy.copy(env.data),
        fields,
        {name: getattr(env.model, name).copy() for name in MODEL_FIELDS},
        copy.deepcopy(np.random.get_state()),
        random.getstate(),
        env.data.qacc_warmstart.copy(),
    )


def restore(env, snapshot: Snapshot) -> None:
    for name, value in snapshot.model_fields.items():
        getattr(env.model, name)[:] = value
    for name, value in snapshot.python_fields.items():
        setattr(env, name, copy.deepcopy(value))
    np.random.set_state(copy.deepcopy(snapshot.numpy_random))
    random.setstate(snapshot.python_random)
    # Copy derived buffers as well: Meta-World can modify rendered site positions
    # directly, and a forward pass may change the image at a freshly reset state.
    env.data = copy.copy(snapshot.data)
    renderer = env.mujoco_renderer
    renderer.data = env.data
    for viewer in renderer._viewers.values():
        viewer.data = env.data


def integration_state(env) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, spec)
    return state
