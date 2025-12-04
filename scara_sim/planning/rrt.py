"""
RRT (Rapidly-exploring Random Tree) planner for SCARA robot.
"""

from typing import Optional
import numpy as np
import time
from scara_sim.core.collision import CollisionChecker
from scara_sim.core.trajectory import resample_trajectory, TrajectoryConfig


class RRTNode:
    """Node in RRT graph."""

    def __init__(self, q: np.ndarray, parent: Optional["RRTNode"] = None):
        self.q = q.copy()
        self.parent = parent
        self.children = []

    def path_to_root(self) -> list[np.ndarray]:
        """Return path from node to root."""
        path = [self.q]
        node = self.parent
        while node is not None:
            path.insert(0, node.q)
            node = node.parent
        return path


def plan(
    robot,
    scene,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    cfg: Optional[dict] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Plan trajectory using RRT algorithm.

    Parameters
    ----------
    robot : ScaraRobot
        Robot model.
    scene : Scene
        Scene with obstacles.
    start_q, goal_q : np.ndarray
        Start and goal configurations [q1, q2, q3, q4].
    cfg : Optional[dict]
        Config with {"step_size": float, "goal_bias": float, "max_nodes": int}.
    seed : Optional[int]
        Random seed for reproducibility.

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

    step_size = cfg.get("step_size", 0.3)
    goal_bias = cfg.get("goal_bias", 0.15)
    max_nodes = cfg.get("max_nodes", 5000)
    goal_tol = cfg.get("goal_tol", 0.1)

    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    checker = CollisionChecker(robot, scene)

    # Validate start and goal
    if checker.check_configuration(start_q):
        planning_time = time.time() - start_time
        return {
            "success": False,
            "waypoints": None,
            "times": None,
            "dt": 0.01,
            "planning_time": planning_time,
            "path_length": None,
            "clearance": None,
            "meta": {**cfg, "seed": seed},
            "nodes_explored": 0,
        }

    if checker.check_configuration(goal_q):
        planning_time = time.time() - start_time
        return {
            "success": False,
            "waypoints": None,
            "times": None,
            "dt": 0.01,
            "planning_time": planning_time,
            "path_length": None,
            "clearance": None,
            "meta": {**cfg, "seed": seed},
            "nodes_explored": 0,
        }

    # Initialize tree
    root = RRTNode(start_q)
    nodes = [root]

    # Get joint limits for sampling
    limits = robot.joint_limits
    limit_list = [
        limits.get(f"q{i+1}", (-np.pi, np.pi))
        for i in range(len(start_q))
    ]

    while len(nodes) < max_nodes:
        # Sample random config or goal
        if np.random.random() < goal_bias:
            q_rand = goal_q.copy()
        else:
            q_rand = np.array(
                [
                    np.random.uniform(limit_list[i][0], limit_list[i][1])
                    for i in range(len(start_q))
                ]
            )

        # Find nearest node
        min_dist = float("inf")
        nearest_node = None

        for node in nodes:
            # Weighted distance in joint space
            dq = q_rand - node.q
            for i in range(len(dq)):
                lo, hi = limit_list[i]
                range_i = max(1.0, hi - lo)
                dq[i] /= range_i

            dist = np.linalg.norm(dq)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # Steer toward random config
        direction = q_rand - nearest_node.q
        dist_raw = np.linalg.norm(direction)

        if dist_raw < 1e-6:
            continue

        direction = direction / dist_raw
        step = min(step_size, dist_raw)
        q_new = nearest_node.q + step * direction

        # Check if within limits
        if not robot.within_limits(q_new):
            continue

        # Check collision on edge
        edge_waypoints = [nearest_node.q, q_new]
        if checker.check_trajectory(edge_waypoints, resolution=5):
            continue

        # Add new node
        new_node = RRTNode(q_new, parent=nearest_node)
        nearest_node.children.append(new_node)
        nodes.append(new_node)

        # Check if goal reached
        goal_dist = np.linalg.norm(q_new - goal_q)
        if goal_dist < goal_tol:
            # Extract path
            waypoints = new_node.path_to_root()

            # Time-parameterize
            traj_cfg = TrajectoryConfig()
            traj = resample_trajectory(waypoints, traj_cfg)

            # Compute metrics
            path_length = 0.0
            for i in range(len(waypoints) - 1):
                path_length += np.linalg.norm(
                    waypoints[i + 1] - waypoints[i]
                )

            clearance = checker.compute_clearance(waypoints, resolution=5)
            planning_time = time.time() - start_time

            return {
                "success": True,
                "waypoints": waypoints,
                "times": traj["times"],
                "dt": traj_cfg.dt,
                "planning_time": planning_time,
                "path_length": path_length,
                "clearance": clearance,
                "meta": {**cfg, "seed": seed},
                "nodes_explored": len(nodes),
            }

    # Failed to find path
    planning_time = time.time() - start_time
    return {
        "success": False,
        "waypoints": None,
        "times": None,
        "dt": 0.01,
        "planning_time": planning_time,
        "path_length": None,
        "clearance": None,
        "meta": {**cfg, "seed": seed},
        "nodes_explored": len(nodes),
    }
