"""
Straight-line joint space planner with collision checking.
"""

from typing import Optional
import numpy as np
import time
from scara_sim.core.collision import CollisionChecker
from scara_sim.core.trajectory import resample_trajectory, TrajectoryConfig


def plan(
    robot,
    scene,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    cfg: Optional[dict] = None,
) -> dict:
    """
    Plan trajectory by linear interpolation in joint space with collision checks.

    Parameters
    ----------
    robot : ScaraRobot
        Robot model.
    scene : Scene
        Scene with obstacles.
    start_q, goal_q : np.ndarray
        Start and goal configurations [q1, q2, q3, q4].
    cfg : Optional[dict]
        Config with {"n_waypoints": int, "resolution": int}.

    Returns
    -------
    dict
        {
            "success": bool,
            "waypoints": list[np.ndarray] or None,
            "times": np.ndarray or None,
            "dt": float,
            "planning_time": float,
            "path_length": float or None,
            "clearance": float or None,
            "meta": dict,
            "nodes_explored": int,
        }
    """
    if cfg is None:
        cfg = {}

    n_waypoints = cfg.get("n_waypoints", 20)
    resolution = cfg.get("resolution", 10)

    start_time = time.time()
    checker = CollisionChecker(robot, scene)

    # Generate linear interpolation
    waypoints = [
        (1 - alpha) * start_q + alpha * goal_q
        for alpha in np.linspace(0, 1, n_waypoints)
    ]

    # Check trajectory
    if checker.check_trajectory(waypoints, resolution):
        planning_time = time.time() - start_time
        return {
            "success": False,
            "waypoints": None,
            "times": None,
            "dt": 0.01,
            "planning_time": planning_time,
            "path_length": None,
            "clearance": None,
            "meta": cfg,
            "nodes_explored": 1,
        }

    # Time-parameterize
    traj_cfg = TrajectoryConfig()
    traj = resample_trajectory(waypoints, traj_cfg)

    # Compute path length
    path_length = 0.0
    for i in range(len(waypoints) - 1):
        path_length += np.linalg.norm(waypoints[i + 1] - waypoints[i])

    # Compute clearance
    clearance = checker.compute_clearance(waypoints, resolution)

    planning_time = time.time() - start_time

    return {
        "success": True,
        "waypoints": waypoints,
        "times": traj["times"],
        "dt": traj_cfg.dt,
        "planning_time": planning_time,
        "path_length": path_length,
        "clearance": clearance,
        "meta": cfg,
        "nodes_explored": 1,
    }
