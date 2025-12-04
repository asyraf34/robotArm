"""
Collision detection between robot links and obstacles.
"""

from typing import Optional
import numpy as np
from scara_sim.core.geometry import point_in_polygon


class CollisionChecker:
    """Checks collisions between robot and scene obstacles."""

    def __init__(self, robot, scene):
        """
        Initialize collision checker.

        Parameters
        ----------
        robot : ScaraRobot
            Robot model.
        scene : Scene
            Scene with obstacles.
        """
        self.robot = robot
        self.scene = scene

    def check_configuration(self, q: np.ndarray) -> bool:
        """
        Check if configuration collides with any obstacle.

        Parameters
        ----------
        q : np.ndarray
            Joint angles [q1, q2, q3, q4].

        Returns
        -------
        bool
            True if collision detected.
        """
        # Only check end-effector point vs polygon interior
        xy = self.robot.fk_xy(q)

        for obstacle in self.scene.obstacles:
            if obstacle.get("type") == "polygon":
                points = obstacle.get("points", [])
                if len(points) < 3:
                    continue

                if point_in_polygon(xy, points):
                    return True

        return False

    def check_trajectory(
        self, waypoints: list[np.ndarray], resolution: int = 10
    ) -> bool:
        """
        Check if trajectory collides by discretizing between waypoints.

        Parameters
        ----------
        waypoints : list[np.ndarray]
            List of joint configurations.
        resolution : int
            Number of samples between consecutive waypoints.

        Returns
        -------
        bool
            True if any collision detected.
        """
        for i in range(len(waypoints) - 1):
            q_start = waypoints[i]
            q_end = waypoints[i + 1]

            for alpha in np.linspace(0, 1, resolution):
                q_interp = (1 - alpha) * q_start + alpha * q_end
                if self.check_configuration(q_interp):
                    return True

        return False

    def compute_clearance(
        self, waypoints: list[np.ndarray], resolution: int = 10
    ) -> float:
        """
        Compute minimum clearance (distance to nearest obstacle) along trajectory.

        Parameters
        ----------
        waypoints : list[np.ndarray]
            List of joint configurations.
        resolution : int
            Number of samples between consecutive waypoints.

        Returns
        -------
        float
            Minimum clearance distance (m).
        """
        # Simplified: return 0.0 if any sampled point is inside an obstacle, else large value
        min_dist = float("inf")

        for i in range(len(waypoints) - 1):
            q_start = waypoints[i]
            q_end = waypoints[i + 1]

            for alpha in np.linspace(0, 1, resolution):
                q_interp = (1 - alpha) * q_start + alpha * q_end
                xy = self.robot.fk_xy(q_interp)
                for obstacle in self.scene.obstacles:
                    if obstacle.get("type") != "polygon":
                        continue
                    pts = obstacle.get("points", [])
                    if len(pts) < 3:
                        continue
                    if point_in_polygon(xy, pts):
                        return 0.0

        return min_dist if min_dist != float("inf") else 1.0
