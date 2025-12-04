"""
Utilities for managing simulator server instances and port conflicts.

Provides functions to:
- Check if a port is already in use
- Kill existing processes using a port
- Handle port conflicts gracefully
"""

import socket
import sys
import os


def is_port_in_use(host: str = "localhost", port: int = 5000) -> bool:
    """
    Check if a port is already in use.

    Parameters
    ----------
    host : str
        Host address (default: localhost)
    port : int
        Port number (default: 5000)

    Returns
    -------
    bool
        True if port is in use, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0  # 0 means connection successful (port in use)
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        return False


def get_process_using_port(port: int) -> dict:
    """
    Find process using a specific port.

    Returns
    -------
    dict
        Dictionary with 'pid' and 'name' keys, or empty dict if not found
    """
    try:
        import subprocess

        if sys.platform == "win32":
            # Windows: use netstat command
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) > 0:
                        pid = parts[-1]
                        try:
                            pid_int = int(pid)
                            return {
                                'pid': pid_int,
                                'name': f'Process {pid_int}'
                            }
                        except ValueError:
                            pass
        else:
            # Unix: use lsof command
            result = subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5
            )

            lines = result.stdout.split('\n')
            if len(lines) > 1:  # First line is header
                parts = lines[1].split()
                if len(parts) >= 2:
                    name = parts[0]
                    pid = parts[1]
                    try:
                        pid_int = int(pid)
                        return {
                            'pid': pid_int,
                            'name': name
                        }
                    except ValueError:
                        pass

    except Exception as e:
        print(f"Error finding process using port {port}: {e}")

    return {}


def kill_process_on_port(port: int, force: bool = False) -> bool:
    """
    Kill process using a specific port.

    Parameters
    ----------
    port : int
        Port number
    force : bool
        Use force kill (SIGKILL) instead of normal kill (SIGTERM)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        import subprocess

        process_info = get_process_using_port(port)
        if not process_info:
            return False

        pid = process_info['pid']
        name = process_info['name']

        print(f"Killing process {name} (PID: {pid}) using port {port}...")

        if sys.platform == "win32":
            # Windows: use taskkill command
            cmd = ["taskkill", "/PID", str(pid)]
            if force:
                cmd.append("/F")  # Force kill
        else:
            # Unix: use kill command
            cmd = ["kill"]
            if force:
                cmd.append("-9")  # SIGKILL
            else:
                cmd.append("-15")  # SIGTERM
            cmd.append(str(pid))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            print(f"Successfully killed process {name} (PID: {pid})")
            return True
        else:
            print(f"Failed to kill process: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


def handle_port_conflict(host: str, port: int, auto_kill: bool = False) -> bool:
    """
    Handle port conflict by checking and optionally killing existing process.

    Parameters
    ----------
    host : str
        Host address
    port : int
        Port number
    auto_kill : bool
        Automatically kill existing process (default: False)

    Returns
    -------
    bool
        True if port is now available, False if still in use

    Raises
    ------
    OSError
        If port is in use and user doesn't want to kill the process
    """
    if not is_port_in_use(host, port):
        return True  # Port is free

    # Port is in use
    process_info = get_process_using_port(port)
    name = process_info.get('name', 'Unknown')
    pid = process_info.get('pid', 'Unknown')

    message = f"Port {port} is already in use by {name} (PID: {pid})"

    if auto_kill:
        print(f"{message}")
        print("Attempting to kill existing process...")
        if kill_process_on_port(port):
            import time
            time.sleep(1)  # Wait for port to be released
            if not is_port_in_use(host, port):
                print("Port is now available")
                return True
            else:
                print("Port is still in use, try again")
                return False
        else:
            raise OSError(f"Could not kill process on port {port}")
    else:
        raise OSError(f"{message}. Use --kill-existing to remove the process.")


def prompt_kill_existing(host: str, port: int) -> bool:
    """
    Prompt user to kill existing process using port.

    Parameters
    ----------
    host : str
        Host address
    port : int
        Port number

    Returns
    -------
    bool
        True if port is available after handling, False otherwise
    """
    if not is_port_in_use(host, port):
        return True

    process_info = get_process_using_port(port)
    name = process_info.get('name', 'Unknown')
    pid = process_info.get('pid', 'Unknown')

    print()
    print(f"[WARNING] Port {port} is already in use!")
    print(f"Process: {name} (PID: {pid})")
    print()
    print("Options:")
    print("  1. Kill the existing process and start simulator")
    print("  2. Use a different port")
    print("  3. Exit")
    print()

    try:
        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            if kill_process_on_port(port):
                import time
                time.sleep(1)
                if not is_port_in_use(host, port):
                    print("\nPort is now available. Starting simulator...\n")
                    return True
                else:
                    print("\nPort still in use. Please try again.\n")
                    return False
            else:
                print("\nFailed to kill process. Please kill it manually and try again.\n")
                return False

        elif choice == "2":
            print("\nPlease run the simulator with a different port:")
            print("  python run_simulator_server.py --port 5001\n")
            return False

        else:
            print("\nExiting.\n")
            return False

    except (EOFError, KeyboardInterrupt):
        print("\nExiting.\n")
        return False
