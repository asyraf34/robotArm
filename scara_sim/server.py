"""
Simulator server that accepts connections for remote robot control.
"""

import socket
import struct
import json
import threading
import queue
import time
from typing import Optional, Callable
import numpy as np

from scara_sim.core.robot import ScaraRobot
from scara_sim.io.scene import Scene, load_scene
from scara_sim.core.collision import CollisionChecker
from scara_sim.core.physics import DEFAULT_PHYSICS, get_realistic_delay


class CommandEntry:
    """Tracks the lifecycle of a single command execution."""

    def __init__(self, command_id: str, command: dict):
        self.command_id = command_id
        self.command = command
        self.status = "queued"  # queued, executing, completed, error
        self.start_time = None
        self.end_time = None
        self.response = None
        self.error_message = None
        self.lock = threading.Lock()
        self.ready = threading.Event()  # Signal when done

    def mark_executing(self):
        with self.lock:
            self.status = "executing"
            self.start_time = time.time()

    def mark_completed(self, response: dict):
        with self.lock:
            self.status = "completed"
            self.response = response
            self.end_time = time.time()
            self.ready.set()

    def mark_error(self, error_message: str):
        with self.lock:
            self.status = "error"
            self.error_message = error_message
            self.end_time = time.time()
            self.ready.set()

    def get_status(self) -> dict:
        """Get current status snapshot."""
        with self.lock:
            result = {
                "command_id": self.command_id,
                "status": self.status,
                "start_time": self.start_time,
                "end_time": self.end_time,
            }
            if self.status == "completed" and self.response:
                result["response"] = self.response
            elif self.status == "error":
                result["error_message"] = self.error_message
            return result


class SimulatorServer:
    """
    Real-time simulator server for remote robot control.

    Protocol:
    - Commands are JSON messages terminated by newline
    - Responses are JSON messages terminated by newline

    Commands:
    - {"cmd": "get_state"} -> returns current joint state and end-effector position
    - {"cmd": "set_joints", "q": [q1, q2, q3, q4]} -> set joint positions
    - {"cmd": "move_to", "xy": [x, y], "elbow": "up"} -> move to cartesian position via IK
    - {"cmd": "get_fk", "q": [q1, q2, q3, q4]} -> compute forward kinematics
    - {"cmd": "get_ik", "xy": [x, y], "elbow": "up"} -> compute inverse kinematics
    - {"cmd": "check_collision", "q": [q1, q2, q3, q4]} -> check if config collides
    - {"cmd": "get_limits"} -> get joint limits
    - {"cmd": "get_info"} -> get robot parameters
    - {"cmd": "reset"} -> reset to home position
    - {"cmd": "start_mission", "mission_id": str} -> notify server mission starting
    - {"cmd": "complete_mission", "mission_id": str} -> notify server mission complete (elapsed_time calculated server-side)
    - {"cmd": "load_scene", "scene_path": str} -> load a scene from file
    - {"cmd": "get_scene_info"} -> get currently loaded scene information
    - {"cmd": "shutdown"} -> close connection
    """

    def __init__(
        self,
        robot: ScaraRobot,
        scene: Scene,
        host: str = "localhost",
        port: int = 8008,
        update_callback: Optional[Callable] = None,
        mission_callback: Optional[Callable] = None,
        animation_fps: int = 30,
        animation_duration: float = 0.5,
        scene_path: Optional[str] = None,
        viewer_callback: Optional[Callable] = None,
        animation_duration_callback: Optional[Callable] = None,
    ):
        """
        Initialize simulator server.

        Parameters
        ----------
        robot : ScaraRobot
            Robot model.
        scene : Scene
            Scene with obstacles.
        host : str
            Server host address.
        port : int
            Server port.
        update_callback : Optional[Callable]
            Callback function called when robot state changes.
        mission_callback : Optional[Callable]
            Callback function for mission events (start/complete).
        animation_fps : int
            Frames per second for smooth animation (24-60, default 30).
        animation_duration : float
            Duration of smooth animation in seconds (default 0.5).
        scene_path : Optional[str]
            Path to the loaded scene file.
        viewer_callback : Optional[Callable]
            Callback function called when scene is loaded (for viewer updates).
        """
        self.robot = robot
        self.scene = scene
        self.scene_path = scene_path
        self.host = host
        self.port = port
        self.update_callback = update_callback
        self.mission_callback = mission_callback
        self.viewer_callback = viewer_callback
        self.animation_duration_callback = animation_duration_callback

        # Current robot state
        self.current_q = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_lock = threading.Lock()

        # Collision checker
        self.collision_checker = CollisionChecker(robot, scene)

        # Server socket
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.server_thread = None

        # Command queue for thread-safe updates
        self.command_queue = queue.Queue()

        # Smooth animation state
        self.animation_fps = max(24, min(60, animation_fps))
        self.animation_duration = animation_duration
        self.animation_interval = max(1, int(1000.0 / self.animation_fps))

        # Animation state variables
        self.target_q = None  # Target configuration
        self.start_q = None  # Starting configuration for animation
        self.animation_start_time = None  # When animation started
        self.is_animating = False
        self.animation_duration_callback = animation_duration_callback

        # Mission state tracking
        self.mission_states = {}  # Dict[mission_id] -> {"status": "pending|in_progress|completed", "start_time": float, "elapsed_time": float}

        # Command queue and tracking for async execution
        self.command_queue = queue.Queue()  # Queue of CommandEntry objects
        self.command_history = {}  # Dict[command_id] -> CommandEntry (stores completed commands)
        self.command_lock = threading.Lock()  # Lock for command_id_counter and command_history
        self.command_id_counter = 0  # Counter for generating unique command IDs
        self.command_processor_thread = None  # Reference to command processor thread

        # Background animation update thread
        self.animation_thread = None
        # Track active mission elapsed time
        self.active_mission_id: Optional[str] = None
        self._anim_mission_id: Optional[str] = None
        self._anim_duration: Optional[float] = None

    def _get_active_mission_id(self) -> Optional[str]:
        """
        Return the current mission that is in progress or already failed.
        Prefers the explicit active_mission_id if it is still valid.
        """
        if self.active_mission_id:
            mission = self.mission_states.get(self.active_mission_id)
            if mission and mission.get("status") in ("in_progress", "failed"):
                return self.active_mission_id

        for mid, info in self.mission_states.items():
            if info.get("status") in ("in_progress", "failed"):
                self.active_mission_id = mid
                return mid
        return None

    def _add_elapsed_time(self, duration: float, mission_id: Optional[str] = None):
        """Accumulate elapsed time into a mission (in_progress or failed)."""
        if duration <= 0:
            return
        mid = mission_id or self._get_active_mission_id()
        if not mid:
            return
        mission = self.mission_states.get(mid)
        if not mission:
            return
        mission["elapsed_time"] = mission.get("elapsed_time", 0.0) + duration

    def _mark_active_mission_failed(self, reason: str) -> Optional[str]:
        """Mark the currently active mission (if any) as failed with the provided reason."""
        import time as time_module

        active_mission = self._get_active_mission_id()
        if not active_mission:
            return None

        mission_info = self.mission_states.get(active_mission, {})
        # If already failed, avoid duplicate callback/updates
        if mission_info.get("status") == "failed":
            return active_mission

        now = time_module.time()
        mission_info["status"] = "failed"
        mission_info["end_time"] = now
        mission_info["error_message"] = reason
        mission_info.setdefault("start_time", now)
        mission_info.setdefault("elapsed_time", 0.0)
        # Only set a baseline elapsed time if none recorded yet; further motion time is accumulated separately
        if mission_info.get("elapsed_time", 0.0) == 0.0:
            mission_info["elapsed_time"] = max(0.0, now - mission_info.get("start_time", now))
        self.mission_states[active_mission] = mission_info

        # Notify mission callback immediately so GUI can reflect fail state while animation continues
        if self.mission_callback:
            try:
                self.mission_callback("fail", active_mission, reason)
            except Exception:
                pass

        return active_mission

    def start(self):
        """Start the simulator server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set SO_LINGER to force close and release port immediately on shutdown
        # This prevents TIME_WAIT state from holding the port
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                         struct.pack('ii', 1, 0))
        except (AttributeError, NameError):
            # SO_LINGER not available on this platform, that's OK
            pass

        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True

        print(f"Simulator server started on {self.host}:{self.port}")
        print(f"Animation: {self.animation_fps} FPS, {self.animation_duration}s duration")
        print("Waiting for client connection...")

        # Start animation update thread
        try:
            self.animation_thread = threading.Thread(target=self._animation_update_loop, daemon=True, name="AnimationThread")
            self.animation_thread.start()
            print("Animation thread started")
        except Exception as e:
            print(f"ERROR: Failed to start animation thread: {e}")

        # Start command processor thread
        try:
            self.command_processor_thread = threading.Thread(target=self._command_processor_loop, daemon=True, name="CommandProcessorThread")
            self.command_processor_thread.start()
            print("Command processor thread started")
        except Exception as e:
            print(f"ERROR: Failed to start command processor thread: {e}")

        # Start connection handling thread
        try:
            self.server_thread = threading.Thread(target=self._accept_connections, daemon=True, name="ConnectionThread")
            self.server_thread.start()
            print("Connection handler started")
        except Exception as e:
            print(f"ERROR: Failed to start connection handler: {e}")

    def _accept_connections(self):
        """Accept client connections in a separate thread."""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, address = self.server_socket.accept()
                print(f"Client connected from {address}")

                self.client_socket = client_socket
                self._handle_client(client_socket)

            except socket.timeout:
                continue
            except (OSError, socket.error) as e:
                # Server socket was closed or other socket error
                if self.running:
                    print(f"Connection handler error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"Unexpected error in connection handler: {e}")
                break

    def _handle_client(self, client_socket):
        """Handle client connection and commands."""
        buffer = ""

        try:
            while self.running:
                data = client_socket.recv(4096).decode("utf-8")
                if not data:
                    # Connection closed by client
                    break

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            # Parse command
                            command = json.loads(line)
                            cmd = command.get("cmd")

                            # Commands that execute immediately (not queued)
                            IMMEDIATE_COMMANDS = {"query_command_status", "get_queue_status", "get_command_status"}

                            if cmd in IMMEDIATE_COMMANDS:
                                # Execute immediately without queueing
                                if cmd == "query_command_status":
                                    command_id = command.get("command_id")
                                    response = self._cmd_query_command_status(command_id)
                                else:
                                    response = {"status": "error", "message": f"Unknown immediate command: {cmd}"}
                                client_socket.sendall((json.dumps(response) + "\n").encode("utf-8"))
                            else:
                                # Generate unique command ID
                                with self.command_lock:
                                    self.command_id_counter += 1
                                    command_id = f"cmd_{self.command_id_counter}"

                                # Create command entry and enqueue
                                command_entry = CommandEntry(command_id, command)
                                self.command_queue.put(command_entry)

                                # Return immediate response indicating command was queued
                                response = {
                                    "status": "queued",
                                    "command_id": command_id,
                                    "message": "Command queued for execution"
                                }
                                client_socket.sendall((json.dumps(response) + "\n").encode("utf-8"))

                        except json.JSONDecodeError as e:
                            error_response = {
                                "status": "error",
                                "message": f"Invalid JSON: {str(e)}"
                            }
                            try:
                                client_socket.sendall((json.dumps(error_response) + "\n").encode("utf-8"))
                            except (OSError, socket.error):
                                pass
                        except (OSError, socket.error) as e:
                            print(f"Error sending response: {e}")
                            break

        except (OSError, socket.error) as e:
            print(f"Client connection error: {e}")
        except Exception as e:
            print(f"Unexpected error handling client: {e}")
        finally:
            # Properly close the client socket
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except (OSError, socket.error):
                pass
            try:
                client_socket.close()
            except (OSError, socket.error):
                pass

            self.client_socket = None
            print("Client disconnected")

    def _animation_update_loop(self):
        """Background thread that handles smooth animation of robot state."""
        print("[AnimationThread] Started successfully")
        animation_started = False  # Track if we've already notified GUI of animation duration
        while self.running:
            try:
                if self.is_animating and self.animation_start_time is not None:
                    # Notify GUI of animation duration when animation first starts
                    if not animation_started and self.animation_duration_callback:
                        # Use the actual animation duration calculated for this specific move
                        actual_duration = getattr(self, '_current_animation_duration', self.animation_duration)
                        self.animation_duration_callback(actual_duration)
                        animation_started = True

                    # Calculate elapsed time since animation started
                    elapsed = time.time() - self.animation_start_time

                    # Safely access animation state under lock
                    should_update = False
                    with self.state_lock:
                        if not self.is_animating or self.target_q is None or self.start_q is None:
                            # Animation was cancelled or state is invalid
                            should_update = False
                        else:
                            should_update = True
                            # Use actual animation duration for this move
                            actual_duration = getattr(self, '_current_animation_duration', self.animation_duration)

                    if elapsed >= actual_duration:
                        # Animation complete - snap to target
                        self.current_q = self.target_q.copy()
                        self.is_animating = False
                        self.target_q = None
                        self.start_q = None
                        self.animation_start_time = None
                        self._anim_mission_id = None
                        self._anim_duration = None
                        animation_started = False  # Reset for next animation
                    else:
                        # Interpolate based on elapsed time
                        progress = elapsed / actual_duration
                        # Cosine easing for smooth acceleration/deceleration
                        ease = (1.0 - np.cos(np.pi * progress)) / 2.0

                        # Interpolate each joint
                        interpolated_q = self.start_q + ease * (self.target_q - self.start_q)
                        self.current_q = interpolated_q.copy()

                    # Notify callback (outside of lock to avoid deadlock)
                    if should_update and self.update_callback:
                        self.update_callback(self.current_q)

                # Sleep for animation frame interval
                time.sleep(self.animation_interval / 1000.0)

            except Exception as e:
                print(f"Animation update error: {e}")
                import traceback
                traceback.print_exc()

    def _command_processor_loop(self):
        """Background thread that processes queued commands sequentially."""
        print("[CommandProcessorThread] Started successfully")
        while self.running:
            try:
                # Get next command from queue (timeout to allow graceful shutdown)
                try:
                    command_entry = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Mark as executing
                command_entry.mark_executing()

                try:
                    # Execute the command
                    response = self._process_command_entry(command_entry)

                    # IMPORTANT: Don't wait for animation here!
                    # Animations run independently in the animation thread.
                    # The client is responsible for waiting based on its own animation duration parameter.
                    # If we wait here, we block command processing and slow down the whole system.

                    # Mark as completed with response
                    command_entry.mark_completed(response)

                except Exception as e:
                    # Mark as error
                    command_entry.mark_error(str(e))
                    print(f"[CommandProcessorThread] Error processing command {command_entry.command_id}: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    # Store in history for potential status queries
                    with self.command_lock:
                        self.command_history[command_entry.command_id] = command_entry

            except Exception as e:
                print(f"Command processor error: {e}")
                import traceback
                traceback.print_exc()

    def _wait_for_animation_complete(self, timeout: float = 5.0) -> None:
        """Wait for current animation to complete, with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.state_lock:
                if not self.is_animating:
                    return
            time.sleep(0.01)  # Poll every 10ms
        # Timeout reached, log warning
        print(f"[CommandProcessorThread] WARNING: Animation did not complete within {timeout}s")

    def _wait_for_animation_complete_nonblocking(self, timeout: float = 5.0) -> None:
        """
        Wait for animation to complete with short non-blocking polls.

        This is used in the command processor thread to ensure sequential command execution
        without blocking the GUI thread. Uses 10ms polling intervals for responsiveness.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds (default: 5.0)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.state_lock:
                if not self.is_animating:
                    return
            time.sleep(0.01)  # Poll every 10ms - short enough to stay responsive
        # Timeout reached, log warning but continue anyway
        print(f"[CommandProcessorThread] WARNING: Animation did not complete within {timeout}s, continuing anyway")

    def _process_command_entry(self, command_entry: CommandEntry) -> dict:
        """
        Process a CommandEntry by delegating to _process_command with the command dict.

        Parameters
        ----------
        command_entry : CommandEntry
            The command entry to process.

        Returns
        -------
        dict
            Response dictionary from command processing.
        """
        return self._process_command_json(command_entry.command)

    def _process_command_json(self, command: dict) -> dict:
        """
        Process a command dictionary (internal version that doesn't parse JSON).

        Parameters
        ----------
        command : dict
            Command dictionary.

        Returns
        -------
        dict
            Response dictionary.
        """
        try:
            cmd = command.get("cmd")

            if cmd == "get_state":
                return self._cmd_get_state()

            elif cmd == "set_joints":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_set_joints(q)

            elif cmd == "move_to":
                xy = np.array(command.get("xy", [0, 0]))
                elbow = command.get("elbow", "up")
                return self._cmd_move_to(xy, elbow)

            elif cmd == "get_fk":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_get_fk(q)

            elif cmd == "get_ik":
                xy = np.array(command.get("xy", [0, 0]))
                elbow = command.get("elbow", "up")
                return self._cmd_get_ik(xy, elbow)

            elif cmd == "check_collision":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_check_collision(q)

            elif cmd == "get_limits":
                return self._cmd_get_limits()

            elif cmd == "get_info":
                return self._cmd_get_info()

            elif cmd == "reset":
                return self._cmd_reset()

            elif cmd == "start_mission":
                mission_id = command.get("mission_id")
                return self._cmd_start_mission(mission_id)

            elif cmd == "complete_mission":
                mission_id = command.get("mission_id")
                return self._cmd_complete_mission(mission_id)

            elif cmd == "fail_mission":
                mission_id = command.get("mission_id")
                error_message = command.get("error_message", "")
                return self._cmd_fail_mission(mission_id, error_message)

            elif cmd == "load_scene":
                scene_path = command.get("scene_path")
                return self._cmd_load_scene(scene_path)

            elif cmd == "get_scene_info":
                return self._cmd_get_scene_info()

            elif cmd == "query_command_status":
                command_id = command.get("command_id")
                return self._cmd_query_command_status(command_id)

            elif cmd == "shutdown":
                return {"status": "ok", "message": "Shutting down"}

            else:
                return {"status": "error", "message": f"Unknown command: {cmd}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _process_command(self, command_str: str) -> dict:
        """
        Process a command from the client.

        Parameters
        ----------
        command_str : str
            JSON command string.

        Returns
        -------
        dict
            Response dictionary.
        """
        try:
            command = json.loads(command_str)
            cmd = command.get("cmd")

            if cmd == "get_state":
                return self._cmd_get_state()

            elif cmd == "set_joints":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_set_joints(q)

            elif cmd == "move_to":
                xy = np.array(command.get("xy", [0, 0]))
                elbow = command.get("elbow", "up")
                return self._cmd_move_to(xy, elbow)

            elif cmd == "get_fk":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_get_fk(q)

            elif cmd == "get_ik":
                xy = np.array(command.get("xy", [0, 0]))
                elbow = command.get("elbow", "up")
                return self._cmd_get_ik(xy, elbow)

            elif cmd == "check_collision":
                q = np.array(command.get("q", [0, 0, 0, 0]))
                return self._cmd_check_collision(q)

            elif cmd == "get_limits":
                return self._cmd_get_limits()

            elif cmd == "get_info":
                return self._cmd_get_info()

            elif cmd == "reset":
                return self._cmd_reset()

            elif cmd == "start_mission":
                mission_id = command.get("mission_id")
                return self._cmd_start_mission(mission_id)

            elif cmd == "complete_mission":
                mission_id = command.get("mission_id")
                return self._cmd_complete_mission(mission_id)

            elif cmd == "fail_mission":
                mission_id = command.get("mission_id")
                error_message = command.get("error_message", "")
                return self._cmd_fail_mission(mission_id, error_message)

            elif cmd == "load_scene":
                scene_path = command.get("scene_path")
                return self._cmd_load_scene(scene_path)

            elif cmd == "get_scene_info":
                return self._cmd_get_scene_info()

            elif cmd == "query_command_status":
                command_id = command.get("command_id")
                return self._cmd_query_command_status(command_id)

            elif cmd == "shutdown":
                return {"status": "ok", "message": "Shutting down"}

            else:
                return {"status": "error", "message": f"Unknown command: {cmd}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _cmd_get_state(self) -> dict:
        """Get current robot state."""
        with self.state_lock:
            q = self.current_q.copy()

        xy = self.robot.fk_xy(q)
        positions = self.robot.get_joint_positions(q)

        return {
            "status": "ok",
            "q": q.tolist(),
            "xy": xy.tolist(),
            "joint_positions": [list(p) for p in positions],
            "timestamp": time.time(),
        }

    def _cmd_set_joints(self, q: np.ndarray) -> dict:
        """Set joint positions with smooth animation."""
        # Validate limits
        if not self.robot.within_limits(q):
            failed_mission = self._mark_active_mission_failed("Joint configuration outside limits")
            return {
                "status": "error",
                "message": "Joint configuration outside limits",
                "mission_status": "failed" if failed_mission else None,
                "mission_id": failed_mission,
            }

        # Check collision along the path - but do not cancel animation; we only mark mission failed
        collision = self.collision_checker.check_trajectory(
            [self.current_q.copy(), q.copy()], resolution=30
        )
        collision_message = None
        failed_mid = None
        if collision:
            failed_mid = self._mark_active_mission_failed("Target configuration in collision")
            if failed_mid:
                collision_message = f"Target configuration in collision - mission {failed_mid} failed"
            else:
                collision_message = "Target configuration in collision"

        # Calculate animation duration based on physics
        with self.state_lock:
            current_pos = self.robot.fk_xy(self.current_q)
            target_pos = self.robot.fk_xy(q)
            distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
            animation_duration = get_realistic_delay(distance, max_speed=DEFAULT_PHYSICS.max_ee_speed)

        # Override animation duration for this move
        actual_animation_duration = max(animation_duration, self.animation_duration)

        # Start smooth animation to target (atomically)
        with self.state_lock:
            self.start_q = self.current_q.copy()
            self.target_q = q.copy()
            self.animation_start_time = time.time()
            self.is_animating = True
            # Store the actual animation duration for this movement
            self._current_animation_duration = actual_animation_duration
            self._anim_mission_id = self._get_active_mission_id()
            self._anim_duration = actual_animation_duration
        # Commit the travel time to the active mission immediately (so failures still report duration)
        self._add_elapsed_time(actual_animation_duration, mission_id=self._anim_mission_id)

        xy = self.robot.fk_xy(q)

        # Return success (keep status ok so animation runs; collision info returned separately)
        result = {
            "status": "ok",
            "q": q.tolist(),
            "xy": xy.tolist(),
            "animation_duration": actual_animation_duration,  # Send to client for accurate waiting
        }
        if collision:
            # Flag collision but keep animation running so GUI continues to play it
            result["collision"] = True
            if collision_message:
                result["warning"] = collision_message
                result["message"] = collision_message
            else:
                result["message"] = "Target configuration in collision"
            # Include mission failure info if available
            active_failed = failed_mid or self._mark_active_mission_failed("Target configuration in collision")
            if active_failed:
                result["mission_status"] = "failed"
                result["mission_id"] = active_failed
        # Always include the planned animation duration so clients can wait even if disconnected
        result["animation_duration"] = actual_animation_duration
        return result

    def _cmd_move_to(self, xy: np.ndarray, elbow: str) -> dict:
        """Move to cartesian position via IK with smooth animation."""
        q = self.robot.ik_xy(xy, elbow=elbow)

        if q is None:
            failed_mission = self._mark_active_mission_failed("Target unreachable")
            return {
                "status": "error",
                "message": "Target unreachable",
                "mission_status": "failed" if failed_mission else None,
                "mission_id": failed_mission,
            }

        # Check collision along the path - but do not cancel animation; we only mark mission failed
        collision = self.collision_checker.check_trajectory(
            [self.current_q.copy(), q.copy()], resolution=30
        )
        collision_message = None
        failed_mid = None
        if collision:
            failed_mid = self._mark_active_mission_failed("Target configuration in collision")
            if failed_mid:
                collision_message = f"Target configuration in collision - mission {failed_mid} failed"
            else:
                collision_message = "Target configuration in collision"

        # Calculate animation duration based on physics
        with self.state_lock:
            current_pos = self.robot.fk_xy(self.current_q)
            distance = np.linalg.norm(np.array(xy) - np.array(current_pos))
            animation_duration = get_realistic_delay(distance, max_speed=DEFAULT_PHYSICS.max_ee_speed)

        # Override animation duration for this move
        actual_animation_duration = max(animation_duration, self.animation_duration)

        # Start smooth animation to target (atomically)
        with self.state_lock:
            self.start_q = self.current_q.copy()
            self.target_q = q.copy()
            self.animation_start_time = time.time()
            self.is_animating = True
            # Store the actual animation duration for this movement
            self._current_animation_duration = actual_animation_duration
            self._anim_mission_id = self._get_active_mission_id()
            self._anim_duration = actual_animation_duration
        # Commit the travel time to the active mission immediately (so failures still report duration)
        self._add_elapsed_time(actual_animation_duration, mission_id=self._anim_mission_id)

        # Return success (keep status ok so animation runs; collision info returned separately)
        result = {
            "status": "ok",
            "q": q.tolist(),
            "xy": xy.tolist(),
            "animation_duration": actual_animation_duration,  # Send to client for accurate waiting
        }
        if collision:
            # Flag collision but keep animation running so GUI continues to play it
            result["collision"] = True
            if collision_message:
                result["warning"] = collision_message
                result["message"] = collision_message
            else:
                result["message"] = "Target configuration in collision"
            active_failed = failed_mid or self._mark_active_mission_failed("Target configuration in collision")
            if active_failed:
                result["mission_status"] = "failed"
                result["mission_id"] = active_failed
        # Always include duration so clients can wait even if disconnected
        result["animation_duration"] = actual_animation_duration
        return result

    def _cmd_get_fk(self, q: np.ndarray) -> dict:
        """Compute forward kinematics."""
        xy = self.robot.fk_xy(q)
        positions = self.robot.get_joint_positions(q)

        return {
            "status": "ok",
            "q": q.tolist(),
            "xy": xy.tolist(),
            "joint_positions": [list(p) for p in positions],
        }

    def _cmd_get_ik(self, xy: np.ndarray, elbow: str) -> dict:
        """Compute inverse kinematics."""
        q = self.robot.ik_xy(xy, elbow=elbow)

        if q is None:
            return {"status": "error", "message": "Target unreachable"}

        return {
            "status": "ok",
            "q": q.tolist(),
            "xy": xy.tolist(),
        }

    def _cmd_check_collision(self, q: np.ndarray) -> dict:
        """Check collision for configuration."""
        collision = self.collision_checker.check_configuration(q)

        return {
            "status": "ok",
            "collision": collision,
            "q": q.tolist(),
        }

    def _cmd_get_limits(self) -> dict:
        """Get joint limits."""
        return {
            "status": "ok",
            "joint_limits": self.robot.joint_limits,
        }

    def _cmd_get_info(self) -> dict:
        """Get robot information."""
        return {
            "status": "ok",
            "L1": self.robot.L1,
            "L2": self.robot.L2,
            "link_radius": self.robot.link_radius,
            "max_reach": self.robot.L1 + self.robot.L2,
            "min_reach": abs(self.robot.L1 - self.robot.L2),
        }

    def _cmd_reset(self) -> dict:
        """Reset robot to home position with smooth animation."""
        q = np.array([0.0, 0.0, 0.0, 0.0])

        # Start smooth animation to home position (atomically)
        with self.state_lock:
            self.start_q = self.current_q.copy()
            self.target_q = q.copy()
            self.animation_start_time = time.time()
            self.is_animating = True

        return {
            "status": "ok",
            "q": q.tolist(),
        }

    def _cmd_start_mission(self, mission_id: str) -> dict:
        """Handle mission start command from client."""
        import time as time_module

        # Track mission state
        self.mission_states[mission_id] = {
            "status": "in_progress",
            "start_time": time_module.time(),
            "elapsed_time": 0.0
        }
        self.active_mission_id = mission_id

        if self.mission_callback:
            try:
                self.mission_callback("start", mission_id, None)
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {
            "status": "ok",
            "message": f"Mission {mission_id} started",
            "mission_status": "in_progress"
        }

    def _cmd_complete_mission(self, mission_id: str) -> dict:
        """Handle mission complete command from client."""
        import time as time_module

        # Calculate elapsed time server-side
        if mission_id in self.mission_states:
            current_status = self.mission_states[mission_id].get("status", "pending")
            if current_status == "failed":
                # Preserve failure status and surface as error to client
                mission_status = "failed"
                elapsed_time = self.mission_states[mission_id].get("elapsed_time", 0.0)
                # Emit failure callback here so GUI updates after motion/animation has played
                if self.mission_callback:
                    try:
                        self.mission_callback("fail", mission_id, self.mission_states[mission_id].get("error_message", ""))
                    except Exception:
                        pass
                return {
                    "status": "error",
                    "message": f"Mission {mission_id} failed",
                    "mission_status": mission_status,
                    "elapsed_time": elapsed_time,
                }
            else:
                self.mission_states[mission_id]["status"] = "completed"
                self.mission_states[mission_id]["end_time"] = time_module.time()

                # Calculate elapsed time from start_time to end_time
                start_time = self.mission_states[mission_id].get("start_time")
                end_time = self.mission_states[mission_id]["end_time"]
                elapsed_time = self.mission_states[mission_id].get("elapsed_time", 0.0)
                if elapsed_time == 0.0:
                    elapsed_time = end_time - start_time if start_time else 0.0
                self.mission_states[mission_id]["elapsed_time"] = elapsed_time
                mission_status = "completed"
        else:
            # Mission was never started on server, create entry with 0 elapsed time
            end_time = time_module.time()
            self.mission_states[mission_id] = {
                "status": "completed",
                "elapsed_time": 0.0,
                "end_time": end_time
            }
            elapsed_time = 0.0
            mission_status = "completed"

        if self.mission_callback:
            try:
                if mission_status != "failed":
                    self.mission_callback("complete", mission_id, elapsed_time)
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {
            "status": "ok",
            "message": f"Mission {mission_id} completed",
            "mission_status": mission_status,
            "elapsed_time": elapsed_time
        }

    def _cmd_fail_mission(self, mission_id: str, error_message: str = "") -> dict:
        """Handle mission failure command from client."""
        import time as time_module

        # Mark mission as failed
        if mission_id in self.mission_states:
            self.mission_states[mission_id]["status"] = "failed"
            self.mission_states[mission_id]["end_time"] = time_module.time()
            self.mission_states[mission_id]["error_message"] = error_message

            # Calculate elapsed time from start_time to end_time
            start_time = self.mission_states[mission_id].get("start_time")
            end_time = self.mission_states[mission_id]["end_time"]
            elapsed_time = end_time - start_time if start_time else 0.0
            self.mission_states[mission_id]["elapsed_time"] = elapsed_time
            mission_status = "failed"
        else:
            # Mission was never started on server, create entry with 0 elapsed time
            end_time = time_module.time()
            self.mission_states[mission_id] = {
                "status": "failed",
                "elapsed_time": 0.0,
                "end_time": end_time,
                "error_message": error_message
            }
            elapsed_time = 0.0
            mission_status = "failed"

        if self.mission_callback:
            try:
                self.mission_callback("fail", mission_id, error_message)
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {
            "status": "ok",
            "message": f"Mission {mission_id} failed: {error_message}",
            "mission_status": mission_status,
            "error_message": error_message,
            "elapsed_time": elapsed_time
        }

    def _cmd_load_scene(self, scene_path: str) -> dict:
        """Load a scene from file."""
        try:
            if not scene_path:
                return {"status": "error", "message": "scene_path is required"}

            new_scene = load_scene(scene_path)
            self.scene = new_scene
            self.scene_path = scene_path

            # Reset collision checker with new scene
            self.collision_checker = CollisionChecker(self.robot, self.scene)

            # Notify viewer to update scene display (if viewer exists)
            if self.viewer_callback:
                try:
                    self.viewer_callback((new_scene, scene_path))
                except Exception as e:
                    print(f"Warning: Failed to update viewer: {e}")

            return {
                "status": "ok",
                "message": f"Scene loaded: {scene_path}",
                "scene_path": scene_path,
                "num_targets": len(self.scene.targets),
                "num_obstacles": len(self.scene.obstacles),
            }
        except FileNotFoundError:
            return {"status": "error", "message": f"Scene file not found: {scene_path}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load scene: {str(e)}"}

    def _cmd_get_scene_info(self) -> dict:
        """Get information about the currently loaded scene."""
        targets = []
        for t in self.scene.targets:
            target_info = {
                "label": t.get("label", "Unknown"),
                "x": t.get("x", 0),
                "y": t.get("y", 0),
            }
            targets.append(target_info)

        return {
            "status": "ok",
            "scene_path": self.scene_path,
            "num_targets": len(self.scene.targets),
            "num_obstacles": len(self.scene.obstacles),
            "targets": targets,
        }

    def _cmd_query_command_status(self, command_id: str) -> dict:
        """Query the status of a previously executed command."""
        if not command_id:
            return {"status": "error", "message": "command_id is required"}

        with self.command_lock:
            if command_id not in self.command_history:
                return {
                    "status": "error",
                    "message": f"Command {command_id} not found in history"
                }

            command_entry = self.command_history[command_id]
            return {
                "status": "ok",
                "command_status": command_entry.get_status()
            }

    def get_current_state(self) -> np.ndarray:
        """Get current joint configuration (thread-safe)."""
        with self.state_lock:
            return self.current_q.copy()

    def stop(self):
        """Stop the simulator server with proper cleanup."""
        self.running = False

        # Close client socket first
        client_socket = self.client_socket
        self.client_socket = None
        if client_socket:
            try:
                # Shutdown socket communication
                client_socket.shutdown(socket.SHUT_RDWR)
            except (OSError, socket.error):
                # Socket may already be closed
                pass
            try:
                client_socket.close()
            except (OSError, socket.error):
                pass

        # Close server socket
        if self.server_socket:
            try:
                # Disable further accepts
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except (OSError, socket.error):
                # Socket may already be closed
                pass
            try:
                self.server_socket.close()
            except (OSError, socket.error):
                pass
            self.server_socket = None

        # Wait for threads to finish with timeout
        if self.server_thread and self.server_thread.is_alive():
            try:
                self.server_thread.join(timeout=1.0)
            except:
                pass

        if self.animation_thread and self.animation_thread.is_alive():
            try:
                self.animation_thread.join(timeout=1.0)
            except:
                pass

        print("Simulator server stopped")
