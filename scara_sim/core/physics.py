"""
Robot physics and dynamics model for realistic simulation.

Defines physical constraints for the SCARA robot:
- Torque limits per joint
- Maximum angular velocities
- Link masses and inertias
- Minimum command delays
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RobotPhysics:
    """
    Physical parameters for realistic SCARA robot simulation.

    Based on typical industrial SCARA robots (Epson T3, ABB IRB1200, etc).
    """

    # ========== JOINT SPECIFICATIONS ==========
    # Maximum torque each joint can produce (Nm)
    # Increased 2x from previous values (1/7.5 of typical SCARA)
    # Original: 50-100 Nm for horizontal joints, 20-50 Nm for vertical
    max_torque_q1: float = 10.67   # Base joint (horizontal) - 2x faster - 1/7.5 of 80
    max_torque_q2: float = 6.67    # Shoulder joint (horizontal) - 2x faster - 1/7.5 of 50
    max_torque_q3: float = 4.0     # Vertical (Z) motion - 2x faster - 1/7.5 of 30
    max_torque_q4: float = 1.33    # Wrist rotation - 2x faster - 1/7.5 of 10

    # Maximum angular velocity (rad/s)
    # Increased 2x from previous values (1/7.5 of typical SCARA)
    # Original: 2-5 rad/s for planar joints, 4-8 rad/s for wrist
    max_vel_q1: float = 0.533      # rad/s - 2x faster - 1/7.5 of 4.0
    max_vel_q2: float = 0.533      # rad/s - 2x faster - 1/7.5 of 4.0
    max_vel_q3: float = 0.4        # rad/s (Z motion) - 2x faster - 1/7.5 of 3.0
    max_vel_q4: float = 0.8        # rad/s (Wrist faster) - 2x faster - 1/7.5 of 6.0

    # Maximum angular acceleration (rad/s²)
    # Increased 2x from previous values (1/7.5 of typical SCARA)
    # Original derived from torque limits and inertia
    max_acc_q1: float = 1.067      # rad/s² - 2x faster - 1/7.5 of 8.0
    max_acc_q2: float = 0.8        # rad/s² - 2x faster - 1/7.5 of 6.0
    max_acc_q3: float = 0.533      # rad/s² - 2x faster - 1/7.5 of 4.0
    max_acc_q4: float = 1.067      # rad/s² - 2x faster - 1/7.5 of 8.0

    # Link masses (kg) - approximate for typical SCARA
    # Kept same to maintain physical realism
    link1_mass: float = 5.0        # kg (heavier, drives workspace)
    link2_mass: float = 3.0        # kg
    ee_mass: float = 0.5           # kg (end-effector tool)

    # ========== TIMING CONSTRAINTS ==========
    # Minimum delay between consecutive waypoint commands (seconds)
    # Accounts for servo response time and control latency
    # Increased for weak robot servo response
    min_command_delay: float = 0.2  # 200 ms minimum between commands (was 100 ms)

    # Servo response time before movement begins (seconds)
    # Weak servo takes longer to respond
    servo_response_time: float = 0.05  # 50 ms servo lag (was 20 ms)

    # ========== CARTESIAN CONSTRAINTS ==========
    # Maximum end-effector speed (m/s)
    # Increased 2x from previous (1/7.5 of typical)
    # Original: 1-2 m/s
    max_ee_speed: float = 0.2       # m/s - 2x faster - 1/7.5 of 1.5

    # Maximum acceleration (m/s²)
    # Increased 2x from previous (1/7.5 of typical)
    max_ee_accel: float = 0.267     # m/s² - 2x faster - 1/7.5 of 2.0

    # Jerk limit (m/s³) - rate of change of acceleration
    max_ee_jerk: float = 5.0        # m/s³

    # ========== SAFETY MARGINS ==========
    # Extra time to add for safety (percentage of minimum)
    safety_margin: float = 1.2      # 20% extra time for safety

    def __post_init__(self):
        """Validate physics parameters are realistic."""
        if self.min_command_delay <= 0:
            raise ValueError("min_command_delay must be positive")
        if self.servo_response_time < 0:
            raise ValueError("servo_response_time cannot be negative")
        if self.max_ee_speed <= 0:
            raise ValueError("max_ee_speed must be positive")

    def get_max_torques(self) -> np.ndarray:
        """Get maximum torque for each joint."""
        return np.array([self.max_torque_q1, self.max_torque_q2,
                        self.max_torque_q3, self.max_torque_q4])

    def get_max_velocities(self) -> np.ndarray:
        """Get maximum angular velocity for each joint."""
        return np.array([self.max_vel_q1, self.max_vel_q2,
                        self.max_vel_q3, self.max_vel_q4])

    def get_max_accelerations(self) -> np.ndarray:
        """Get maximum angular acceleration for each joint."""
        return np.array([self.max_acc_q1, self.max_acc_q2,
                        self.max_acc_q3, self.max_acc_q4])

    def calculate_min_delay(self, current_q: np.ndarray, target_q: np.ndarray,
                           L1: float, L2: float) -> float:
        """
        Calculate minimum physically realizable delay between waypoints.

        Based on maximum joint velocities and accelerations.

        Parameters
        ----------
        current_q : np.ndarray
            Current joint configuration [q1, q2, q3, q4]
        target_q : np.ndarray
            Target joint configuration [q1, q2, q3, q4]
        L1 : float
            Length of link 1 (meters)
        L2 : float
            Length of link 2 (meters)

        Returns
        -------
        float
            Minimum delay in seconds to reach target from current position
        """
        # Joint-space constraint: time needed based on max velocity
        q_diff = np.abs(target_q - current_q)
        max_vels = self.get_max_velocities()

        # Time for each joint to reach target (assuming linear profile)
        joint_times = q_diff / max_vels
        max_joint_time = np.max(joint_times)

        # Add servo response time
        min_delay = max_joint_time + self.servo_response_time

        # Apply safety margin
        min_delay *= self.safety_margin

        # Enforce absolute minimum
        min_delay = max(min_delay, self.min_command_delay)

        return min_delay

    def validate_delay(self, delay: float) -> tuple[bool, str]:
        """
        Validate if a command delay is physically possible.

        Parameters
        ----------
        delay : float
            Requested delay between commands (seconds)

        Returns
        -------
        tuple[bool, str]
            (is_valid, reason_if_invalid)
        """
        if delay < self.min_command_delay:
            reason = (f"Delay {delay:.3f}s too small. "
                     f"Minimum is {self.min_command_delay}s "
                     f"(servo response: {self.servo_response_time}s)")
            return False, reason
        return True, ""

    def enforce_velocity_limits(self, joint_velocities: np.ndarray) -> np.ndarray:
        """
        Clamp joint velocities to maximum values.

        Parameters
        ----------
        joint_velocities : np.ndarray
            Requested joint velocities [q1_dot, q2_dot, q3_dot, q4_dot]

        Returns
        -------
        np.ndarray
            Clamped joint velocities
        """
        max_vels = self.get_max_velocities()
        return np.clip(joint_velocities, -max_vels, max_vels)

    def enforce_torque_limits(self, joint_torques: np.ndarray) -> np.ndarray:
        """
        Clamp joint torques to maximum values.

        Parameters
        ----------
        joint_torques : np.ndarray
            Requested joint torques [tau1, tau2, tau3, tau4]

        Returns
        -------
        np.ndarray
            Clamped joint torques
        """
        max_torques = self.get_max_torques()
        return np.clip(joint_torques, -max_torques, max_torques)

    def get_total_mass(self) -> float:
        """Get total moving mass (all links + payload)."""
        return self.link1_mass + self.link2_mass + self.ee_mass


# Default physics for standard SCARA robot
DEFAULT_PHYSICS = RobotPhysics()


def get_realistic_delay(distance: float, max_speed: float = 1.5,
                       physics: RobotPhysics = DEFAULT_PHYSICS) -> float:
    """
    Calculate realistic delay for traversing a distance.

    Accounts for acceleration/deceleration phases.

    Parameters
    ----------
    distance : float
        Distance to traverse (meters)
    max_speed : float
        Maximum speed (m/s)
    physics : RobotPhysics
        Physics parameters

    Returns
    -------
    float
        Minimum delay (seconds)
    """
    # For acceleration limited motion:
    # distance = 0.5 * accel * time^2 + speed * time (simplified)
    accel = physics.max_ee_accel

    # Time to reach max speed
    t_accel = max_speed / accel
    # Distance covered during acceleration/deceleration
    dist_accel = 0.5 * accel * t_accel**2 * 2  # accel + decel

    if distance <= dist_accel:
        # Only accelerate, don't reach max speed
        delay = 2.0 * np.sqrt(distance / accel)
    else:
        # Reach max speed
        dist_constant = distance - dist_accel
        t_constant = dist_constant / max_speed
        delay = 2.0 * t_accel + t_constant

    # Add safety margin
    delay *= physics.safety_margin

    # Enforce minimum
    delay = max(delay, physics.min_command_delay)

    return delay
