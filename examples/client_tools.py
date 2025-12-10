"""
High-level client tools for SCARA robot control.

This module provides simple, one-liner commands for robot control,
designed for non-developers and quick scripting.

Example usage:
    robot = RobotController()
    robot.move_to([0.4, 0.1])
    robot.position()
    robot.reset()
    robot.disconnect()

Or with context manager:
    with RobotController() as robot:
        robot.move_to([0.4, 0.1])
        robot.position()
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from scara_sim.client import SimulatorClient
from scara_sim.core.physics import DEFAULT_PHYSICS, get_realistic_delay


# ========== UTILITY FUNCTIONS ==========

def load_scene(scene_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a scene JSON file.

    Parameters
    ----------
    scene_path : str
        Path to scene JSON file (relative to project root or absolute)

    Returns
    -------
    Optional[Dict[str, Any]]
        Scene dictionary or None if failed
    """
    try:
        path_obj = Path(scene_path)
        if not path_obj.is_absolute() and not path_obj.exists():
            path_obj = Path(__file__).parent / scene_path

        if not path_obj.exists():
            print(f"[FAIL] Scene file not found: {path_obj}")
            return None

        with open(path_obj, 'r') as f:
            scene = json.load(f)
        return scene
    except Exception as e:
        print(f"[FAIL] Failed to load scene: {e}")
        return None


def list_scenes(scenes_dir: str = "scenes") -> List[str]:
    """
    List all available scene files.

    Parameters
    ----------
    scenes_dir : str
        Directory containing scene files (default: "scenes")

    Returns
    -------
    List[str]
        List of scene filenames
    """
    try:
        scenes_path = Path(scenes_dir)
        if not scenes_path.is_absolute():
            scenes_path = Path(__file__).parent / scenes_dir

        if not scenes_path.exists():
            print(f"[FAIL] Scenes directory not found: {scenes_path}")
            return []

        scene_files = sorted([f.name for f in scenes_path.glob("*.json")])
        return scene_files
    except Exception as e:
        print(f"[FAIL] Failed to list scenes: {e}")
        return []


class RobotController:
    """High-level robot control interface for easy, intuitive usage."""

    def __init__(self, host: str = "localhost", port: int = 8008, verbose: bool = True):
        """
        Initialize robot controller.

        Parameters
        ----------
        host : str
            Server hostname (default: localhost)
        port : int
            Server port (default: 8008)
        verbose : bool
            Print status messages (default: True)
        """
        self.client = SimulatorClient(host, port)
        self.verbose = verbose
        self._connected = False
        self._info = None

    def connect(self) -> bool:
        """
        Connect to simulator server.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        if self.client.connect():
            self._connected = True
            # Get info - with async queueing, we need to wait for the command to complete
            info_response = self.client.get_info()
            if info_response and info_response.get("status") == "ok":
                self._info = info_response
            elif info_response and info_response.get("status") == "queued":
                # Command was queued, wait for it to complete
                command_id = info_response.get("command_id")
                cmd_status = self.client.wait_for_command(command_id, timeout=5.0)
                if cmd_status and cmd_status.get("status") == "completed":
                    response_data = cmd_status.get("response", {})
                    if response_data.get("status") == "ok":
                        self._info = response_data
                    else:
                        print("[FAIL] Failed to get robot info")
                        self._connected = False
                        return False
                else:
                    print("[FAIL] Timeout waiting for robot info")
                    self._connected = False
                    return False
            else:
                print("[FAIL] Failed to get robot info")
                self._connected = False
                return False

            if self._info and self.verbose:
                print(f"[OK] Connected to robot (L1={self._info.get('L1', '?')}m, L2={self._info.get('L2', '?')}m)")
            return True
        else:
            print("[FAIL] Failed to connect. Is the server running? (python run_simulator_server.py)")
            return False

    def disconnect(self):
        """Disconnect from simulator."""
        self.client.disconnect()
        self._connected = False
        if self.verbose:
            print("[OK] Disconnected")

    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    # ========== MOVEMENT COMMANDS ==========

    def move_to(self, xy: List[float], elbow: str = "up", wait: bool = True) -> bool:
        """
        Move end-effector to cartesian position.

        Parameters
        ----------
        xy : List[float]
            Target position [x, y] in meters
        elbow : str
            Elbow configuration: "up" or "down"
        wait : bool
            If True, automatically wait for movement to complete (default: True)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        if elbow is None:
            elbow = "up"

        if elbow is None:
            elbow = "up"

        response = self.client.move_to(xy, elbow=elbow)
        if not response:
            if self.verbose:
                print(f"[FAIL] Failed to move to {xy}: No response")
            return False

        # Handle queued response (with async queueing)
        if response.get("status") == "queued":
            command_id = response.get("command_id")
            if wait:
                # Wait for the command to complete
                cmd_status = self.client.wait_for_command(command_id, timeout=10.0)
                if cmd_status and cmd_status.get("status") == "completed":
                    final_response = cmd_status.get("response", {})
                    if final_response.get("status") == "ok":
                        if self.verbose:
                            msg = f"[OK] Moved to [{xy[0]:.3f}, {xy[1]:.3f}]"
                            if "warning" in final_response:
                                msg += f" (WARNING: {final_response['warning']})"
                            print(msg)
                        # FIX: Use animation duration from server (physics-based) if available
                        animation_duration = final_response.get("animation_duration")
                        if animation_duration:
                            self.wait_for_move(duration=animation_duration)
                        else:
                            # Fallback to default if server doesn't provide it
                            self.wait_for_move()
                        return not final_response.get("collision", False)
                    else:
                        error = final_response.get("message", "Unknown error")
                        if self.verbose:
                            print(f"[FAIL] Failed to move to {xy}: {error}")
                        return False
                else:
                    if self.verbose:
                        print(f"[FAIL] Failed to move to {xy}: Command timeout or error")
                    return False
            else:
                # Command was queued but not waiting
                if self.verbose:
                    print(f"[OK] Move to [{xy[0]:.3f}, {xy[1]:.3f}] queued (command_id: {command_id})")
                return True

        # Handle direct ok response (without queueing - for backward compatibility)
        elif response.get("status") == "ok":
            if self.verbose:
                msg = f"[OK] Moved to [{xy[0]:.3f}, {xy[1]:.3f}]"
                if "warning" in response:
                    msg += f" (WARNING: {response['warning']})"
                print(msg)

            # Automatically wait for movement to complete if requested
            if wait:
                # FIX: Use animation duration from server (physics-based) if available
                animation_duration = response.get("animation_duration")
                if animation_duration:
                    self.wait_for_move(duration=animation_duration)
                else:
                    # Fallback to default if server doesn't provide it
                    self.wait_for_move()

            return not response.get("collision", False)
        else:
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to move to {xy}: {error}")
            return False

    def wait_for_move(self, duration: float = None) -> None:
        """
        Wait for current movement animation to complete.

        The simulator animates robot movements based on physics calculations.
        The server calculates animation duration based on distance, velocity constraints, and acceleration.
        This method waits for that animation to complete before returning.

        Parameters
        ----------
        duration : float, optional
            Animation duration in seconds. If None, uses default (0.5s).
            Typically provided by the server based on physics calculations.
            The client receives animation_duration in the response for each move_to command.
        """
        if duration is None:
            # Default animation duration - used as fallback if server doesn't provide physics-based duration
            # Modern servers calculate this based on physics, so this rarely gets used
            duration = 0.5

        time.sleep(duration)

    def move_joints(self, q: List[float]) -> bool:
        """
        Move to joint configuration.

        Parameters
        ----------
        q : List[float]
            Joint angles [q1, q2, q3, q4] in radians

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client.set_joints(q)
        if response and response.get("status") == "ok":
            if self.verbose:
                msg = f"[OK] Moved to joints {[f'{x:.3f}' for x in q[:2]]}"
                if "warning" in response:
                    msg += f" (WARNING: {response['warning']})"
                print(msg)
            return True
        else:
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to move: {error}")
            return False

    def load_scene(self, scene_path: str) -> bool:
        """
        Load a scene file on the simulator server.

        Parameters
        ----------
        scene_path : str
            Path to scene JSON file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client._send_command({"cmd": "load_scene", "scene_path": scene_path})

        # Handle queued response
        if response and response.get("status") == "queued":
            command_id = response.get("command_id")
            # Loading a scene can occasionally take longer than other
            # commands (especially on first launch), so give it more time
            # before declaring a timeout.
            cmd_status = self.client.wait_for_command(command_id, timeout=30.0)
            if cmd_status and cmd_status.get("status") == "completed":
                final_response = cmd_status.get("response", {})
                if final_response.get("status") == "ok":
                    if self.verbose:
                        print(f"[OK] Loaded scene: {scene_path}")
                    return True
                elif self.verbose:
                    error = final_response.get("message", "Unknown error")
                    print(f"[FAIL] Failed to load scene {scene_path}: {error}")
            else:
                # Provide a clearer message when the server never replies.
                if self.verbose:
                    print(f"[FAIL] Failed to load scene {scene_path}: command timed out")
            return False

        # Handle direct ok response
        elif response and response.get("status") == "ok":
            if self.verbose:
                print(f"[OK] Loaded scene: {scene_path}")
            return True

        # Handle error response
        else:
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to load scene {scene_path}: {error}")
            return False

    def reset(self) -> bool:
        """
        Reset robot to home position [0, 0, 0, 0].

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client.reset()

        # Handle queued response
        if response and response.get("status") == "queued":
            command_id = response.get("command_id")
            cmd_status = self.client.wait_for_command(command_id, timeout=10.0)
            if cmd_status and cmd_status.get("status") == "completed":
                final_response = cmd_status.get("response", {})
                if final_response.get("status") == "ok":
                    if self.verbose:
                        print("[OK] Reset to home position")
                    return True
            if self.verbose:
                print("[FAIL] Failed to reset")
            return False

        # Handle direct ok response
        if response and response.get("status") == "ok":
            if self.verbose:
                print("[OK] Reset to home position")
            return True
        else:
            if self.verbose:
                print("[FAIL] Failed to reset")
            return False

    # ========== MISSION COMMANDS ==========

    def start_mission(self, mission_id: str) -> bool:
        """
        Start a mission on the simulator.

        Parameters
        ----------
        mission_id : str
            ID of mission to start

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client._send_command({
            "cmd": "start_mission",
            "mission_id": mission_id
        })

        # Handle queued response
        if response and response.get("status") == "queued":
            command_id = response.get("command_id")
            cmd_status = self.client.wait_for_command(command_id, timeout=10.0)
            if cmd_status and cmd_status.get("status") == "completed":
                final_response = cmd_status.get("response", {})
                if final_response.get("status") == "ok":
                    if self.verbose:
                        print(f"[OK] Mission {mission_id} started")
                    return True
            if self.verbose:
                print(f"[FAIL] Failed to start mission")
            return False

        # Handle direct ok response
        elif response and response.get("status") == "ok":
            if self.verbose:
                mission_status = response.get("mission_status", "in_progress")
                print(f"[OK] Mission {mission_id} started (status: {mission_status})")
            return True

        # Handle error response
        else:
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to start mission: {error}")
            return False

    def complete_mission(self, mission_id: str) -> bool:
        """
        Complete a mission on the simulator.

        Parameters
        ----------
        mission_id : str
            ID of mission to complete

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client._send_command({
            "cmd": "complete_mission",
            "mission_id": mission_id
        })

        # Handle queued response
        if response and response.get("status") == "queued":
            command_id = response.get("command_id")
            cmd_status = self.client.wait_for_command(command_id, timeout=10.0)
            if cmd_status and cmd_status.get("status") == "completed":
                final_response = cmd_status.get("response", {})
                elapsed_time = final_response.get("elapsed_time", 0.0)
                if final_response.get("status") == "ok":
                    if self.verbose:
                        print(f"[OK] Mission {mission_id} completed ({elapsed_time:.2f}s)")
                    return True
                else:
                    if self.verbose:
                        msg = final_response.get("message", "Unknown error")
                        print(f"[FAIL] Failed to complete mission {mission_id} ({elapsed_time:.2f}s): {msg}")
                    return False
            if self.verbose:
                print(f"[FAIL] Failed to complete mission")
            return False

        # Handle direct ok response
        elif response and response.get("status") == "ok":
            if self.verbose:
                elapsed_time = response.get("elapsed_time", 0.0)
                print(f"[OK] Mission {mission_id} completed ({elapsed_time:.2f}s)")
            return True

        # Handle error response
        else:
            elapsed_time = response.get("elapsed_time", 0.0) if response else 0.0
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to complete mission {mission_id} ({elapsed_time:.2f}s): {error}")
            return False

    def fail_mission(self, mission_id: str, error_message: str = "") -> bool:
        """
        Fail a mission on the simulator due to collision or error.

        Parameters
        ----------
        mission_id : str
            ID of mission to fail
        error_message : str, optional
            Error message describing why mission failed

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client._send_command({
            "cmd": "fail_mission",
            "mission_id": mission_id,
            "error_message": error_message
        })

        # Handle queued response
        if response and response.get("status") == "queued":
            command_id = response.get("command_id")
            cmd_status = self.client.wait_for_command(command_id, timeout=10.0)
            if cmd_status and cmd_status.get("status") == "completed":
                final_response = cmd_status.get("response", {})
                if final_response.get("status") == "ok":
                    if self.verbose:
                        print(f"[OK] Mission {mission_id} failed: {error_message}")
                    return True
            if self.verbose:
                print(f"[FAIL] Failed to fail mission")
            return False

        # Handle direct ok response
        elif response and response.get("status") == "ok":
            if self.verbose:
                print(f"[OK] Mission {mission_id} failed: {error_message}")
            return True

        # Handle error response
        else:
            error = response.get("message", "Unknown error") if response else "No response"
            if self.verbose:
                print(f"[FAIL] Failed to fail mission: {error}")
            return False

    # ========== STATE QUERIES ==========

    def position(self) -> Optional[Tuple[float, float]]:
        """
        Get and print current end-effector position.

        Returns
        -------
        Optional[Tuple[float, float]]
            Current (x, y) position or None if failed
        """
        if not self._check_connected():
            return None

        response = self.client.get_state()
        if response and response.get("status") == "ok":
            xy = response.get("xy")
            if self.verbose:
                print(f"  Position: [{xy[0]:.3f}, {xy[1]:.3f}]")
            return tuple(xy)
        else:
            if self.verbose:
                print("[FAIL] Failed to get position")
            return None

    def joints(self) -> Optional[List[float]]:
        """
        Get and print current joint angles.

        Returns
        -------
        Optional[List[float]]
            Joint angles [q1, q2, q3, q4] or None if failed
        """
        if not self._check_connected():
            return None

        response = self.client.get_state()
        if response and response.get("status") == "ok":
            q = response.get("q")
            if self.verbose:
                print(f"  Joints: {[f'{x:.3f}' for x in q]}")
            return q
        else:
            if self.verbose:
                print("[FAIL] Failed to get joints")
            return None

    def state(self) -> Optional[dict]:
        """
        Get and print complete robot state.

        Returns
        -------
        Optional[dict]
            State dictionary with status, q, xy, or None if failed
        """
        if not self._check_connected():
            return None

        response = self.client.get_state()
        if response and response.get("status") == "ok":
            q = response.get("q")
            xy = response.get("xy")
            if self.verbose:
                print(f"  State: q={[f'{x:.3f}' for x in q[:2]]}, xy=[{xy[0]:.3f}, {xy[1]:.3f}]")
            return response
        else:
            if self.verbose:
                print("[FAIL] Failed to get state")
            return None

    def workspace(self) -> Optional[float]:
        """
        Get and print workspace information.

        Returns
        -------
        Optional[float]
            Maximum reach in meters or None if failed
        """
        if not self._info:
            if self.verbose:
                print("[FAIL] Robot info not available")
            return None

        max_reach = self._info.get("max_reach", 0)
        min_reach = self._info.get("min_reach", 0)
        if self.verbose:
            print(f"  Workspace: {min_reach:.3f}m - {max_reach:.3f}m radius")
        return max_reach

    # ========== COLLISION CHECKING ==========

    def is_reachable(self, xy: List[float], elbow: str = "up", verbose: bool = None) -> bool:
        """
        Check if position is reachable.

        Parameters
        ----------
        xy : List[float]
            Target position [x, y]
        elbow : str
            Elbow configuration: "up" or "down"
        verbose : bool
            Override verbose setting for this call

        Returns
        -------
        bool
            True if reachable, False otherwise
        """
        if not self._check_connected():
            return False

        response = self.client.get_ik(xy, elbow=elbow)
        reachable = response and response.get("status") == "ok"

        v = verbose if verbose is not None else self.verbose
        if v and reachable:
            print(f"  [OK] Position {xy} is reachable")
        elif v:
            print(f"  [FAIL] Position {xy} is unreachable")

        return reachable

    def is_collision_free(self, xy: List[float], elbow: str = "up", verbose: bool = None) -> bool:
        """
        Check if position is collision-free (implies reachable).

        Parameters
        ----------
        xy : List[float]
            Target position [x, y]
        elbow : str
            Elbow configuration: "up" or "down"
        verbose : bool
            Override verbose setting for this call

        Returns
        -------
        bool
            True if collision-free, False otherwise
        """
        if not self.is_reachable(xy, elbow, verbose=False):
            return False

        response = self.client.get_ik(xy, elbow=elbow)
        if response and response.get("status") == "ok":
            q = response.get("q")
            collision_response = self.client.check_collision(q)
            collision_free = collision_response and not collision_response.get("collision", True)

            v = verbose if verbose is not None else self.verbose
            if v:
                if collision_free:
                    print(f"  [OK] Position {xy} is collision-free")
                else:
                    print(f"  [FAIL] Position {xy} has collision")

            return collision_free
        return False

    # ========== COMPLEX MOTIONS ==========

    def execute_waypoints(self, waypoints: List[List[float]], elbow: str = "up", delay: float = None) -> int:
        """
        Execute motion through multiple waypoints.

        The move_to() function automatically waits for each animation to complete before returning.
        Animation timing is calculated server-side based on robot physics.

        Parameters
        ----------
        waypoints : List[List[float]]
            List of [x, y] target positions
        elbow : str
            Elbow configuration: "up" or "down"
        delay : float
            DEPRECATED: Ignored. Animation timing is server-side only.
            Kept for backward compatibility.

        Returns
        -------
        int
            Number of waypoints successfully reached
        """
        if not self._check_connected():
            return 0

        success_count = 0
        for xy in waypoints:
            if self.move_to(xy, elbow=elbow):
                success_count += 1

        if self.verbose:
            print(f"[OK] Executed {success_count}/{len(waypoints)} waypoints")
        return success_count

    def circular_motion(self, radius: float, n_points: int = 16, elbow: str = "up",
                       clockwise: bool = True, delay: float = None,
                       enforce_physics: bool = True) -> int:
        """
        Execute circular motion at given radius.

        Animation timing is calculated server-side based on robot physics.
        Each move completes fully before the next point is targeted.

        Parameters
        ----------
        radius : float
            Circle radius in meters
        n_points : int
            Number of points on circle (default: 16)
        elbow : str
            Elbow configuration: "up" or "down"
        clockwise : bool
            Direction of motion (default: True)
        delay : float
            DEPRECATED: Ignored. Animation timing is server-side only.
        enforce_physics : bool
            DEPRECATED: Ignored. Server handles physics.

        Returns
        -------
        int
            Number of waypoints successfully reached
        """
        if not self._check_connected():
            return 0

        if self.verbose:
            print(f"Executing circular motion (r={radius:.3f}m, {n_points} points)...")

        success_count = 0
        direction = 1 if clockwise else -1

        for i in range(n_points):
            angle = direction * 2 * np.pi * i / n_points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            if self.move_to([x, y], elbow=elbow):
                success_count += 1

        if self.verbose:
            print(f"[OK] Completed {success_count}/{n_points} circle points")
        return success_count

    def line_motion(self, start_xy: List[float], end_xy: List[float], n_points: int = 10,
                   elbow: str = "up", delay: float = None,
                   enforce_physics: bool = True) -> int:
        """
        Execute linear motion between two points.

        Animation timing is calculated server-side based on robot physics.
        Each waypoint completes fully before the next is targeted.

        Parameters
        ----------
        start_xy : List[float]
            Starting position [x, y]
        end_xy : List[float]
            Ending position [x, y]
        n_points : int
            Number of interpolation points (default: 10)
        elbow : str
            Elbow configuration: "up" or "down"
        delay : float
            DEPRECATED: Ignored. Animation timing is server-side only.
        enforce_physics : bool
            DEPRECATED: Ignored. Server handles physics.

        Returns
        -------
        int
            Number of waypoints successfully reached
        """
        if not self._check_connected():
            return 0

        if self.verbose:
            distance = np.linalg.norm(np.array(end_xy) - np.array(start_xy))
            print(f"Executing line motion from {start_xy} to {end_xy} ({distance:.3f}m)...")

        waypoints = []
        for t in np.linspace(0, 1, n_points):
            x = start_xy[0] + t * (end_xy[0] - start_xy[0])
            y = start_xy[1] + t * (end_xy[1] - start_xy[1])
            waypoints.append([x, y])

        return self.execute_waypoints(waypoints, elbow=elbow)

    # ========== KINEMATICS QUERIES ==========

    def fk(self, q: List[float]) -> Optional[Tuple[float, float]]:
        """
        Forward kinematics query (no movement).

        Parameters
        ----------
        q : List[float]
            Joint angles [q1, q2, q3, q4]

        Returns
        -------
        Optional[Tuple[float, float]]
            End-effector position or None if failed
        """
        if not self._check_connected():
            return None

        response = self.client.get_fk(q)
        if response and response.get("status") == "ok":
            xy = response.get("xy")
            if self.verbose:
                print(f"  FK({[f'{x:.3f}' for x in q[:2]]}) = [{xy[0]:.3f}, {xy[1]:.3f}]")
            return tuple(xy)
        else:
            if self.verbose:
                print("[FAIL] FK failed")
            return None

    def ik(self, xy: List[float], elbow: str = "up") -> Optional[List[float]]:
        """
        Inverse kinematics query (no movement).

        Parameters
        ----------
        xy : List[float]
            Target position [x, y]
        elbow : str
            Elbow configuration: "up" or "down"

        Returns
        -------
        Optional[List[float]]
            Joint angles or None if unreachable
        """
        if not self._check_connected():
            return None

        response = self.client.get_ik(xy, elbow=elbow)
        if response and response.get("status") == "ok":
            q = response.get("q")
            if self.verbose:
                print(f"  IK({xy}, {elbow}) = {[f'{x:.3f}' for x in q[:2]]}")
            return q
        else:
            if self.verbose:
                print(f"[FAIL] IK failed: unreachable")
            return None

    # ========== ASYNC HELPERS ==========

    def wait_for_command_async(self, command_id: str, timeout: float = 10.0) -> bool:
        """
        Wait for a previously sent command to complete execution.

        Advanced method for users who need fine-grained control over command execution.
        Most users should use move_to() with wait=True instead.

        Parameters
        ----------
        command_id : str
            The command ID returned from a queued command
        timeout : float
            Maximum time to wait in seconds

        Returns
        -------
        bool
            True if command completed successfully, False on error or timeout
        """
        if not self._check_connected():
            return False

        cmd_status = self.client.wait_for_command(command_id, timeout=timeout)
        if cmd_status:
            status = cmd_status.get("status", "unknown")
            if status == "completed":
                if self.verbose:
                    print(f"[OK] Command {command_id} completed")
                return True
            elif status == "error":
                error = cmd_status.get("error_message", "Unknown error")
                if self.verbose:
                    print(f"[FAIL] Command {command_id} errored: {error}")
                return False

        if self.verbose:
            print(f"[FAIL] Command {command_id} timed out or failed to complete")
        return False

    # ========== CONTEXT MANAGER SUPPORT ==========

    def __enter__(self):
        """Context manager entry: connect to server."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: disconnect from server."""
        self.disconnect()

    # ========== PRIVATE HELPERS ==========

    def _check_connected(self) -> bool:
        """Check if connected, print message if not."""
        if not self._connected:
            print("[FAIL] Not connected to server. Call connect() first.")
            return False
        return True


# ============================================================================
# Convenience factory functions
# ============================================================================

class MissionMonitor:
    """Monitor and track mission progress during autonomous execution."""

    def __init__(self, scene_path: str = "scenes/demo.json", verbose: bool = True):
        """
        Initialize mission monitor from scene file.

        Parameters
        ----------
        scene_path : str
            Path to scene JSON file containing missions
        verbose : bool
            Print status messages (default: True)
        """
        import json
        from pathlib import Path

        self.verbose = verbose
        self.missions = []
        self.mission_states = {}  # {mission_id: {'status': 'pending|in_progress|completed', 'start_time': time, 'elapsed': float}}

        # Event callbacks
        self.on_mission_started = []  # List of callbacks: callback(mission_id, mission_info)
        self.on_mission_completed = []  # List of callbacks: callback(mission_id, mission_info, elapsed_time)
        self.on_mission_event = []  # Generic event callback: callback(event_type, mission_id, data)

        # Load missions from scene file
        try:
            # Resolve path (handle both relative and absolute paths)
            scene_path_obj = Path(scene_path)
            if not scene_path_obj.is_absolute():
                if not scene_path_obj.exists():
                    script_dir = Path(__file__).parent
                    scene_path_obj = script_dir / scene_path

            with open(scene_path_obj, 'r') as f:
                scene_data = json.load(f)

            self.missions = scene_data.get("missions", [])

            # Initialize mission state tracking
            for idx, mission in enumerate(self.missions):
                mission_id = mission.get("id", f"Mission {idx + 1}")
                self.mission_states[mission_id] = {
                    "index": idx,
                    "status": "pending",
                    "start_time": None,
                    "elapsed_time": 0.0,
                    "pickup_target": mission.get("pickup_target"),
                    "delivery_target": mission.get("delivery_target"),
                }

            if self.verbose:
                print(f"[OK] Loaded {len(self.missions)} missions from {scene_path_obj}")
                for mission in self.missions:
                    print(f"     {mission.get('id', 'Unknown')}: {mission.get('pickup_target')} -> {mission.get('delivery_target')}")

        except Exception as e:
            print(f"[FAIL] Failed to load missions: {e}")
            self.missions = []

    def start_mission(self, mission_id_or_index) -> bool:
        """
        Mark a mission as in progress.

        Parameters
        ----------
        mission_id_or_index : str or int
            Mission ID or index

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        mission_key = self._resolve_mission_key(mission_id_or_index)
        if mission_key is None:
            return False

        if mission_key in self.mission_states:
            state = self.mission_states[mission_key]
            if state["status"] == "pending":
                state["status"] = "in_progress"
                state["start_time"] = time.time()
                if self.verbose:
                    print(f"[OK] Started mission: {mission_key}")

                # Trigger event callbacks
                mission_info = {
                    "id": mission_key,
                    "pickup_target": state["pickup_target"],
                    "delivery_target": state["delivery_target"],
                }
                for callback in self.on_mission_started:
                    try:
                        callback(mission_key, mission_info)
                    except Exception as e:
                        print(f"[WARN] Mission started callback error: {e}")

                for callback in self.on_mission_event:
                    try:
                        callback("mission_started", mission_key, mission_info)
                    except Exception as e:
                        print(f"[WARN] Mission event callback error: {e}")

                return True
            else:
                if self.verbose:
                    print(f"[FAIL] Mission {mission_key} is already {state['status']}")
                return False
        return False

    def complete_mission(self, mission_id_or_index) -> bool:
        """
        Mark a mission as completed.

        Parameters
        ----------
        mission_id_or_index : str or int
            Mission ID or index

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        mission_key = self._resolve_mission_key(mission_id_or_index)
        if mission_key is None:
            return False

        if mission_key in self.mission_states:
            state = self.mission_states[mission_key]
            if state["status"] == "in_progress":
                if state["start_time"] is not None:
                    state["elapsed_time"] = time.time() - state["start_time"]
                state["status"] = "completed"
                if self.verbose:
                    print(f"[OK] Completed mission: {mission_key} ({state['elapsed_time']:.2f}s)")

                # Trigger event callbacks
                mission_info = {
                    "id": mission_key,
                    "pickup_target": state["pickup_target"],
                    "delivery_target": state["delivery_target"],
                }
                elapsed_time = state.get("elapsed_time", 0.0)

                for callback in self.on_mission_completed:
                    try:
                        callback(mission_key, mission_info, elapsed_time)
                    except Exception as e:
                        print(f"[WARN] Mission completed callback error: {e}")

                for callback in self.on_mission_event:
                    try:
                        callback("mission_completed", mission_key, {**mission_info, "elapsed_time": elapsed_time})
                    except Exception as e:
                        print(f"[WARN] Mission event callback error: {e}")

                return True
            else:
                if self.verbose:
                    print(f"[FAIL] Mission {mission_key} is not in progress (current: {state['status']})")
                return False
        return False

    def fail_mission(self, mission_id_or_index, error_message: str = "") -> bool:
        """
        Mark a mission as failed.

        Parameters
        ----------
        mission_id_or_index : str or int
            Mission ID or index
        error_message : str, optional
            Error message describing why mission failed

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        mission_key = self._resolve_mission_key(mission_id_or_index)
        if mission_key is None:
            return False

        if mission_key in self.mission_states:
            state = self.mission_states[mission_key]
            if state["status"] == "in_progress":
                if state["start_time"] is not None:
                    state["elapsed_time"] = time.time() - state["start_time"]
                state["status"] = "failed"
                state["error_message"] = error_message
                if self.verbose:
                    print(f"[OK] Failed mission: {mission_key} ({state['elapsed_time']:.2f}s) - {error_message}")

                # Trigger event callbacks
                mission_info = {
                    "id": mission_key,
                    "pickup_target": state["pickup_target"],
                    "delivery_target": state["delivery_target"],
                }
                elapsed_time = state.get("elapsed_time", 0.0)

                # Call generic mission_event callbacks with "mission_failed" event
                for callback in self.on_mission_event:
                    try:
                        callback("mission_failed", mission_key, {**mission_info, "elapsed_time": elapsed_time, "error_message": error_message})
                    except Exception as e:
                        print(f"[WARN] Mission event callback error: {e}")

                return True
            else:
                if self.verbose:
                    print(f"[FAIL] Mission {mission_key} is not in progress (current: {state['status']})")
                return False
        return False

    def get_status(self, mission_id_or_index) -> Optional[dict]:
        """
        Get current status of a mission.

        Parameters
        ----------
        mission_id_or_index : str or int
            Mission ID or index

        Returns
        -------
        dict or None
            Mission status dict with keys: status, elapsed_time, pickup_target, delivery_target
            Returns None if mission not found
        """
        mission_key = self._resolve_mission_key(mission_id_or_index)
        if mission_key is None:
            return None

        if mission_key in self.mission_states:
            state = self.mission_states[mission_key]
            # Update elapsed time for in-progress missions
            if state["status"] == "in_progress" and state["start_time"] is not None:
                elapsed = time.time() - state["start_time"]
            else:
                elapsed = state.get("elapsed_time", 0.0)

            return {
                "mission_id": mission_key,
                "status": state["status"],
                "elapsed_time": elapsed,
                "pickup_target": state["pickup_target"],
                "delivery_target": state["delivery_target"],
            }
        return None

    def list_missions(self) -> List[dict]:
        """
        Get list of all missions with their current status.

        Returns
        -------
        List[dict]
            List of mission status dicts
        """
        results = []
        for mission_key in self.mission_states.keys():
            status = self.get_status(mission_key)
            if status:
                results.append(status)
        return results

    def get_summary(self) -> dict:
        """
        Get summary statistics of all missions.

        Returns
        -------
        dict
            Summary with keys: total, completed, in_progress, pending, total_elapsed_time
        """
        summary = {
            "total": len(self.missions),
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "total_elapsed_time": 0.0,
        }

        for state in self.mission_states.values():
            status = state["status"]
            if status == "completed":
                summary["completed"] += 1
                summary["total_elapsed_time"] += state.get("elapsed_time", 0.0)
            elif status == "in_progress":
                summary["in_progress"] += 1
            else:
                summary["pending"] += 1

        return summary

    def reset_all(self):
        """Reset all missions to pending state."""
        for state in self.mission_states.values():
            state["status"] = "pending"
            state["start_time"] = None
            state["elapsed_time"] = 0.0
        if self.verbose:
            print("[OK] All missions reset to pending")

    def _resolve_mission_key(self, mission_id_or_index) -> Optional[str]:
        """
        Resolve mission ID from either ID string or index number.

        Parameters
        ----------
        mission_id_or_index : str or int
            Mission ID or 0-based index

        Returns
        -------
        str or None
            Mission ID string, or None if not found
        """
        if isinstance(mission_id_or_index, int):
            # Resolve by index
            if 0 <= mission_id_or_index < len(self.missions):
                return self.missions[mission_id_or_index].get("id", f"Mission {mission_id_or_index + 1}")
            else:
                if self.verbose:
                    print(f"[FAIL] Mission index {mission_id_or_index} out of range (0-{len(self.missions)-1})")
                return None
        else:
            # Resolve by ID string
            if mission_id_or_index in self.mission_states:
                return mission_id_or_index
            else:
                if self.verbose:
                    print(f"[FAIL] Mission '{mission_id_or_index}' not found")
                return None

    def reset_all_missions(self) -> None:
        """
        Reset all missions to pending status and clear progress tracking.

        This is called when a client connects to create a fresh progress log.
        """
        for mission_id in self.mission_states:
            state = self.mission_states[mission_id]
            state["status"] = "pending"
            state["start_time"] = None
            state["elapsed_time"] = 0.0

        if self.verbose:
            print(f"[OK] Reset {len(self.mission_states)} missions to pending status")

    def save_state_to_file(self, filepath: str = "/tmp/mission_state.json") -> bool:
        """
        Save mission state to file for inter-process communication.

        Parameters
        ----------
        filepath : str
            Path to save state file (default: /tmp/mission_state.json)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        import json
        from pathlib import Path
        try:
            # Create directory if needed
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Convert mission_states to serializable format
            serializable_states = {}
            for mission_id, state in self.mission_states.items():
                serializable_states[mission_id] = {
                    "status": state["status"],
                    "elapsed_time": state["elapsed_time"],
                    "pickup_target": state["pickup_target"],
                    "delivery_target": state["delivery_target"],
                    "index": state["index"],
                }

            with open(filepath, 'w') as f:
                json.dump(serializable_states, f)

            return True
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to save mission state to {filepath}: {e}")
            return False

    def load_state_from_file(self, filepath: str = "/tmp/mission_state.json") -> bool:
        """
        Load mission state from file for inter-process synchronization.

        Parameters
        ----------
        filepath : str
            Path to load state file (default: /tmp/mission_state.json)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        import json
        from pathlib import Path
        try:
            if not Path(filepath).exists():
                return False

            with open(filepath, 'r') as f:
                serializable_states = json.load(f)

            # Update mission states from file
            for mission_id, state in serializable_states.items():
                if mission_id in self.mission_states:
                    self.mission_states[mission_id].update(state)

            return True
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to load mission state from {filepath}: {e}")
            return False


def connect_robot(host: str = "localhost", port: int = 8008,
                  scene_path: str = "scenes/demo.json", verbose: bool = True) -> RobotController:
    """
    Create and connect to robot in one call.

    Parameters
    ----------
    host : str
        Server hostname
    port : int
        Server port
    scene_path : str
        Path to scene JSON file to load on the server
    verbose : bool
        Print status messages

    Returns
    -------
    RobotController
        Connected robot controller
    """
    robot = RobotController(host, port, verbose=verbose)
    robot.connect()

    # Load scene on server
    if scene_path:
        response = robot.client._send_command({"cmd": "load_scene", "scene_path": scene_path})
        if response and response.get("status") == "ok":
            if verbose:
                print(f"[OK] Loaded scene: {scene_path}")
        elif verbose:
            print(f"[FAIL] Scene loading failed: {response.get('message', 'unknown error') if response else 'no response'}")

    return robot


class MissionExecutor:
    """High-level mission execution interface for autonomous mission completion."""

    def __init__(self, host: str = "localhost", port: int = 8008, scene_path: str = "scenes/demo.json", verbose: bool = True):
        """
        Initialize mission executor.

        Parameters
        ----------
        host : str
            Server hostname (default: localhost)
        port : int
            Server port (default: 8008)
        scene_path : str
            Path to scene JSON file with missions
        verbose : bool
            Print status messages (default: True)
        """
        self.robot = RobotController(host, port, verbose=verbose)
        self.verbose = verbose
        self.scene_path = scene_path
        self.missions = []
        self.target_map = {}

    def connect(self) -> bool:
        """Connect to simulator server."""
        if not self.robot.connect():
            return False

        # Load scene on server so GUI can display it
        if self.scene_path:
            response = self.robot.client._send_command({"cmd": "load_scene", "scene_path": self.scene_path})
            if response and response.get("status") == "ok":
                if self.verbose:
                    print(f"[OK] Loaded scene on server: {self.scene_path}")
            elif self.verbose:
                print(f"[WARN] Failed to load scene on server: {response.get('message', 'unknown error') if response else 'no response'}")

        # Load missions from scene file
        self._load_missions()
        return True

    def _load_missions(self):
        """Load missions from scene file."""
        scene = load_scene(self.scene_path)
        if scene is None:
            if self.verbose:
                print(f"Failed to load missions: Scene file not found")
            return

        self.missions = scene.get("missions", [])
        targets = scene.get("targets", [])
        for target in targets:
            label = target.get("label")
            if label:
                self.target_map[label] = (target.get("x"), target.get("y"))

        if self.verbose:
            print(f"Loaded {len(self.missions)} missions from {self.scene_path}")

    def start_mission(self, mission_id: str) -> bool:
        """Request simulator to start a mission.

        Parameters
        ----------
        mission_id : str
            ID of mission to start

        Returns
        -------
        bool
            True if successful
        """
        try:
            response = self.robot.client._send_command({
                "cmd": "start_mission",
                "mission_id": mission_id
            })
            if response and response.get("status") in ("success", "ok"):
                if self.verbose:
                    print(f"[OK] Mission {mission_id} started on simulator")
                return True
            else:
                if self.verbose:
                    print(f"[FAIL] Failed to start mission: {response}")
                return False
        except Exception as e:
            if self.verbose:
                print(f"[FAIL] Error starting mission: {e}")
            return False

    def complete_mission(self, mission_id: str, elapsed_time: float = 0.0) -> bool:
        """Request simulator to complete a mission.

        Parameters
        ----------
        mission_id : str
            ID of mission to complete
        elapsed_time : float
            Time taken to complete mission (seconds)

        Returns
        -------
        bool
            True if successful
        """
        try:
            response = self.robot.client._send_command({
                "cmd": "complete_mission",
                "mission_id": mission_id,
                "elapsed_time": elapsed_time
            })
            if response and response.get("status") in ("success", "ok"):
                if self.verbose:
                    print(f"[OK] Mission {mission_id} completed (took {elapsed_time:.2f}s)")
                return True
            else:
                if self.verbose:
                    print(f"[FAIL] Failed to complete mission: {response}")
                return False
        except Exception as e:
            if self.verbose:
                print(f"[FAIL] Error completing mission: {e}")
            return False

    def move_to_target(self, target_label: str, elbow: str = "down") -> bool:
        """Move to a named target from the scene.

        Parameters
        ----------
        target_label : str
            Label of target (e.g., "Target A")
        elbow : str
            Elbow configuration: "up" or "down"

        Returns
        -------
        bool
            True if successful
        """
        if target_label not in self.target_map:
            if self.verbose:
                print(f"Target not found: {target_label}")
            return False

        x, y = self.target_map[target_label]
        return self.robot.move_to([x, y], elbow=elbow)

    def reset(self) -> bool:
        """Reset robot to home position."""
        return self.robot.reset()

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current end-effector position."""
        return self.robot.position()

    def disconnect(self):
        """Disconnect from simulator."""
        self.robot.disconnect()

    def load_scene(self, scene_path: str) -> bool:
        """
        Load a different scene (and reload missions).

        Parameters
        ----------
        scene_path : str
            Path to scene JSON file with missions

        Returns
        -------
        bool
            True if successful
        """
        self.scene_path = scene_path
        self._load_missions()
        return len(self.missions) > 0
