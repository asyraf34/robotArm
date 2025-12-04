"""
SCARA robot kinematic and dynamics model.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ScaraRobot:
    """
    4-DOF SCARA robot: θ1, θ2 (planar), z (vertical), θ4 (wrist).
    MVP operates as 2-DOF planar with z/θ4 hooks for future expansion.
    """

    L1: float  # Length of link 1 (m)
    L2: float  # Length of link 2 (m)
    joint_limits: dict[str, tuple[float, float]]  # {"q1": (min, max), ...}
    link_radius: float = 0.02  # Capsule radius for collision (m)

    def __post_init__(self):
        """Validate parameters."""
        if self.L1 <= 0 or self.L2 <= 0:
            raise ValueError("Link lengths must be positive")
        if len(self.joint_limits) < 2:
            raise ValueError("Need at least q1 and q2 limits")
        # Ensure all limits are (min, max) tuples
        for key, (lo, hi) in self.joint_limits.items():
            if lo >= hi:
                raise ValueError(f"Invalid limits for {key}: {lo} >= {hi}")

    def fk_xy(self, q: np.ndarray) -> np.ndarray:
        """
        Forward kinematics for 2-DOF planar arm.

        Parameters
        ----------
        q : np.ndarray
            Joint angles [q1, q2] in radians.

        Returns
        -------
        np.ndarray
            End-effector position [x, y].
        """
        q1, q2 = q[0], q[1]
        x = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        y = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        return np.array([x, y])

    def ik_xy(
        self, xy: np.ndarray, elbow: str = "up"
    ) -> Optional[np.ndarray]:
        """
        Inverse kinematics for 2-DOF planar arm (closed-form).

        Parameters
        ----------
        xy : np.ndarray
            Target end-effector position [x, y].
        elbow : str
            "up" or "down" elbow configuration.

        Returns
        -------
        Optional[np.ndarray]
            Joint angles [q1, q2] if reachable and within limits, else None.
        """
        x, y = xy[0], xy[1]
        d = x**2 + y**2
        dist = np.sqrt(d)

        # Check reachability
        max_reach = self.L1 + self.L2
        min_reach = abs(self.L1 - self.L2)
        if dist > max_reach or dist < min_reach:
            return None  # Unreachable

        # Law of cosines for q2
        denom = 2 * self.L1 * self.L2
        if denom == 0:
            return None

        cos_q2 = (d - self.L1**2 - self.L2**2) / denom
        # Clamp to handle numerical errors near singularities
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)

        sin_q2_pos = np.sqrt(1.0 - cos_q2**2)
        q2_up = np.arctan2(sin_q2_pos, cos_q2)
        q2_down = np.arctan2(-sin_q2_pos, cos_q2)

        q2 = q2_up if elbow == "up" else q2_down

        # Solve q1
        k1 = self.L1 + self.L2 * np.cos(q2)
        k2 = self.L2 * np.sin(q2)
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        q = np.array([q1, q2, 0.0, 0.0])

        if self.within_limits(q):
            return q
        return None

    def within_limits(self, q: np.ndarray) -> bool:
        """
        Check if joint configuration is within limits.

        Parameters
        ----------
        q : np.ndarray
            Joint angles [q1, q2, q3, q4].

        Returns
        -------
        bool
            True if all joints within limits.
        """
        for i, (key, (lo, hi)) in enumerate(self.joint_limits.items()):
            if not (lo <= q[i] <= hi):
                return False
        return True

    def get_link_capsules(self, q: np.ndarray) -> list[dict]:
        """
        Get capsule representations of links for collision checking.

        Parameters
        ----------
        q : np.ndarray
            Joint angles [q1, q2, q3, q4].

        Returns
        -------
        list[dict]
            List of {"p1": (x, y), "p2": (x, y), "radius": r} for each link.
        """
        capsules = []

        # Base at origin
        base_pos = np.array([0.0, 0.0])

        # Link 1: base -> joint 2
        p1 = base_pos
        p2 = base_pos + np.array(
            [self.L1 * np.cos(q[0]), self.L1 * np.sin(q[0])]
        )
        capsules.append(
            {"p1": tuple(p1), "p2": tuple(p2), "radius": self.link_radius}
        )

        # Link 2: joint 2 -> end effector
        q1_q2 = q[0] + q[1]
        p3 = p2 + np.array(
            [self.L2 * np.cos(q1_q2), self.L2 * np.sin(q1_q2)]
        )
        capsules.append(
            {"p1": tuple(p2), "p2": tuple(p3), "radius": self.link_radius}
        )

        return capsules

    def get_joint_positions(self, q: np.ndarray) -> list[tuple[float, float]]:
        """
        Get positions of all joints for visualization.

        Parameters
        ----------
        q : np.ndarray
            Joint angles [q1, q2, q3, q4].

        Returns
        -------
        list[tuple[float, float]]
            Positions of base, joint 2, and end effector.
        """
        base = (0.0, 0.0)
        j2 = tuple(
            np.array(
                [self.L1 * np.cos(q[0]), self.L1 * np.sin(q[0])]
            )
        )
        q1_q2 = q[0] + q[1]
        ee = tuple(
            np.array(j2)
            + np.array(
                [self.L2 * np.cos(q1_q2), self.L2 * np.sin(q1_q2)]
            )
        )
        return [base, j2, ee]
