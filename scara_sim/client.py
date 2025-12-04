"""
Client API for connecting to and controlling the simulator.
"""

import socket
import json
import time
from typing import Optional, Tuple, List
import numpy as np


class SimulatorClient:
    """
    Client for connecting to the SCARA simulator server.

    Example usage:
        client = SimulatorClient()
        client.connect()
        client.move_to([0.4, 0.1])
        state = client.get_state()
        client.disconnect()
    """

    def __init__(self, host: str = "localhost", port: int = 8008):
        """
        Initialize simulator client.

        Parameters
        ----------
        host : str
            Server host address.
        port : int
            Server port.
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.buffer = ""

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to simulator server.

        Parameters
        ----------
        timeout : float
            Connection timeout in seconds.

        Returns
        -------
        bool
            True if connected successfully.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to simulator at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from simulator server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
        print("Disconnected from simulator")

    def _send_command(self, command: dict) -> Optional[dict]:
        """
        Send command to server and receive response.

        Parameters
        ----------
        command : dict
            Command dictionary.

        Returns
        -------
        Optional[dict]
            Response dictionary or None on error.
        """
        if not self.connected or not self.socket:
            print("Not connected to simulator")
            return None

        try:
            # Send command
            message = json.dumps(command) + "\n"
            self.socket.sendall(message.encode("utf-8"))

            # Receive response
            while "\n" not in self.buffer:
                data = self.socket.recv(4096).decode("utf-8")
                if not data:
                    print("Connection closed by server")
                    self.connected = False
                    return None
                self.buffer += data

            line, self.buffer = self.buffer.split("\n", 1)
            response = json.loads(line)
            return response

        except Exception as e:
            print(f"Communication error: {e}")
            self.connected = False
            return None

    def get_state(self) -> Optional[dict]:
        """
        Get current robot state.

        Returns
        -------
        Optional[dict]
            {
                "status": "ok",
                "q": [q1, q2, q3, q4],
                "xy": [x, y],
                "joint_positions": [[x, y], ...],
                "timestamp": float
            }
        """
        return self._send_command({"cmd": "get_state"})

    def set_joints(self, q: List[float]) -> Optional[dict]:
        """
        Set joint positions.

        Parameters
        ----------
        q : List[float]
            Joint angles [q1, q2, q3, q4] in radians.

        Returns
        -------
        Optional[dict]
            Response with status and updated state.
        """
        return self._send_command({"cmd": "set_joints", "q": q})

    def move_to(self, xy: List[float], elbow: str = "up") -> Optional[dict]:
        """
        Move end-effector to cartesian position.

        Parameters
        ----------
        xy : List[float]
            Target position [x, y] in meters.
        elbow : str
            Elbow configuration: "up" or "down".

        Returns
        -------
        Optional[dict]
            Response with status and joint configuration.
        """
        return self._send_command({"cmd": "move_to", "xy": xy, "elbow": elbow})

    def get_fk(self, q: List[float]) -> Optional[dict]:
        """
        Compute forward kinematics for given joint configuration.

        Parameters
        ----------
        q : List[float]
            Joint angles [q1, q2, q3, q4] in radians.

        Returns
        -------
        Optional[dict]
            End-effector position and joint positions.
        """
        return self._send_command({"cmd": "get_fk", "q": q})

    def get_ik(self, xy: List[float], elbow: str = "up") -> Optional[dict]:
        """
        Compute inverse kinematics for target position.

        Parameters
        ----------
        xy : List[float]
            Target position [x, y] in meters.
        elbow : str
            Elbow configuration: "up" or "down".

        Returns
        -------
        Optional[dict]
            Joint configuration or error.
        """
        return self._send_command({"cmd": "get_ik", "xy": xy, "elbow": elbow})

    def check_collision(self, q: List[float]) -> Optional[dict]:
        """
        Check if joint configuration collides with obstacles.

        Parameters
        ----------
        q : List[float]
            Joint angles [q1, q2, q3, q4] in radians.

        Returns
        -------
        Optional[dict]
            {"status": "ok", "collision": bool}
        """
        return self._send_command({"cmd": "check_collision", "q": q})

    def get_limits(self) -> Optional[dict]:
        """
        Get joint limits.

        Returns
        -------
        Optional[dict]
            Joint limits dictionary.
        """
        return self._send_command({"cmd": "get_limits"})

    def get_info(self) -> Optional[dict]:
        """
        Get robot information.

        Returns
        -------
        Optional[dict]
            Robot parameters (L1, L2, reach, etc.).
        """
        return self._send_command({"cmd": "get_info"})

    def reset(self) -> Optional[dict]:
        """
        Reset robot to home position.

        Returns
        -------
        Optional[dict]
            Response with status.
        """
        return self._send_command({"cmd": "reset"})

    def query_command_status(self, command_id: str) -> Optional[dict]:
        """
        Query the status of a previously executed command.

        Parameters
        ----------
        command_id : str
            The command ID to query.

        Returns
        -------
        Optional[dict]
            Command status or None on error.
        """
        return self._send_command({"cmd": "query_command_status", "command_id": command_id})

    def wait_for_command(self, command_id: str, timeout: float = 10.0, poll_interval: float = 0.01) -> Optional[dict]:
        """
        Wait for a command to complete execution.

        Parameters
        ----------
        command_id : str
            The command ID to wait for.
        timeout : float
            Maximum time to wait in seconds.
        poll_interval : float
            How often to poll for status in seconds (default: 0.01 = 10ms for responsiveness).

        Returns
        -------
        Optional[dict]
            Command status when completed, or None on timeout/error.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_resp = self.query_command_status(command_id)
            if not status_resp:
                return None

            if status_resp.get("status") == "ok":
                cmd_status = status_resp.get("command_status", {})
                cmd_state = cmd_status.get("status", "unknown")

                # Check if command is complete or errored
                if cmd_state in ("completed", "error"):
                    return cmd_status

            # Sleep before next poll (10ms default for responsive command completion)
            time.sleep(poll_interval)

        # Timeout reached
        return None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience functions for quick scripting
def connect(host: str = "localhost", port: int = 8008) -> SimulatorClient:
    """
    Create and connect to simulator.

    Parameters
    ----------
    host : str
        Server host.
    port : int
        Server port.

    Returns
    -------
    SimulatorClient
        Connected client instance.
    """
    client = SimulatorClient(host, port)
    client.connect()
    return client
