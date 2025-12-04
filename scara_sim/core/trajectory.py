"""
Trajectory generation and time-parameterization.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory time-parameterization."""

    dt: float = 0.01  # Time step (s)
    vmax: np.ndarray = None  # Max velocity per joint (rad/s)
    amax: np.ndarray = None  # Max acceleration per joint (rad/s^2)

    def __post_init__(self):
        """Set defaults if not provided."""
        if self.vmax is None:
            self.vmax = np.array([2.0, 2.0, 0.5, 2.0])
        if self.amax is None:
            self.amax = np.array([5.0, 5.0, 2.0, 5.0])


def resample_trajectory(
    waypoints: list[np.ndarray], cfg: Optional[TrajectoryConfig] = None
) -> dict:
    """
    Generate time-parameterized trajectory from waypoints using trapezoidal profile.

    Parameters
    ----------
    waypoints : list[np.ndarray]
        Joint space waypoints.
    cfg : Optional[TrajectoryConfig]
        Trajectory configuration.

    Returns
    -------
    dict
        {"times": np.ndarray, "waypoints": list[np.ndarray], "dt": float}.
    """
    if cfg is None:
        cfg = TrajectoryConfig()

    times = [0.0]
    current_time = 0.0

    for i in range(len(waypoints) - 1):
        q_curr = waypoints[i]
        q_next = waypoints[i + 1]

        # Compute required time for each joint
        max_segment_time = 0.0

        for j in range(len(q_curr)):
            dq = abs(q_next[j] - q_curr[j])
            if dq < 1e-6:
                seg_time = 0.0
            else:
                # Minimum time with accel + const vel + decel
                v_max = cfg.vmax[j]
                a_max = cfg.amax[j]

                # Time to reach vmax: t_accel = vmax / amax
                # Distance during accel: d_accel = 0.5 * amax * t_accel^2
                t_accel = v_max / a_max
                d_accel = 0.5 * a_max * t_accel**2

                if 2 * d_accel >= dq:
                    # Triangle profile (no constant velocity phase)
                    t_accel = np.sqrt(dq / a_max)
                    seg_time = 2 * t_accel
                else:
                    # Trapezoidal profile
                    d_const_vel = dq - 2 * d_accel
                    t_const_vel = d_const_vel / v_max
                    seg_time = 2 * t_accel + t_const_vel

            max_segment_time = max(max_segment_time, seg_time)

        current_time += max_segment_time
        times.append(current_time)

    # Generate intermediate points with fixed dt
    if times[-1] < 1e-6:
        return {
            "times": np.array(times),
            "waypoints": waypoints,
            "dt": cfg.dt,
        }

    n_samples = max(2, int(np.ceil(times[-1] / cfg.dt)))
    interp_times = np.linspace(0, times[-1], n_samples)
    interp_waypoints = []

    for t in interp_times:
        # Find segment
        seg_idx = 0
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                seg_idx = i
                break

        q_start = waypoints[seg_idx]
        q_end = waypoints[seg_idx + 1]
        t_seg_start = times[seg_idx]
        t_seg_end = times[seg_idx + 1]

        if t_seg_end - t_seg_start < 1e-6:
            alpha = 0.0
        else:
            alpha = (t - t_seg_start) / (t_seg_end - t_seg_start)

        q_interp = (1 - alpha) * q_start + alpha * q_end
        interp_waypoints.append(q_interp)

    return {
        "times": interp_times,
        "waypoints": interp_waypoints,
        "dt": cfg.dt,
    }


def check_trajectory_collision(
    waypoints: List[np.ndarray], collision_checker
) -> dict:
    """
    Check collision status for each point in a trajectory.

    Parameters
    ----------
    waypoints : List[np.ndarray]
        Joint space waypoints.
    collision_checker
        CollisionChecker instance.

    Returns
    -------
    dict
        {
            "waypoints": waypoints,
            "collision_status": [bool, ...],  # True if in collision
            "collision_count": int,
            "collision_segments": [(start_idx, end_idx), ...],
        }
    """
    collision_status = []
    for q in waypoints:
        collision_status.append(collision_checker.check_configuration(q))

    # Find collision segments (continuous ranges of colliding waypoints)
    collision_segments = []
    in_collision = False
    start_idx = 0

    for i, is_collision in enumerate(collision_status):
        if is_collision and not in_collision:
            # Start of collision segment
            start_idx = i
            in_collision = True
        elif not is_collision and in_collision:
            # End of collision segment
            collision_segments.append((start_idx, i - 1))
            in_collision = False

    # Handle case where trajectory ends in collision
    if in_collision:
        collision_segments.append((start_idx, len(collision_status) - 1))

    return {
        "waypoints": waypoints,
        "collision_status": collision_status,
        "collision_count": sum(collision_status),
        "collision_segments": collision_segments,
    }
