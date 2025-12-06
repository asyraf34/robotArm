#!/usr/bin/env python
"""
Launch the SCARA simulator GUI application.

Usage:
    python run_gui.py [scene_file.json]

Example:
    python run_gui.py
    python run_gui.py scenes/demo.json
    python run_gui.py scenes/problem1.json --solve
"""
import argparse
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from examples.client_tools import RobotController
from scara_sim.gui import SimulatorGUI
from scara_sim.core.robot import ScaraRobot
from scara_sim.io.scene import Scene, load_scene
from scara_sim.server import SimulatorServer

def _build_target_map(scene: Scene) -> Dict[str, Tuple[float, float]]:
    """Return a mapping from target label to (x, y) coordinates."""

    return {t["label"]: (t["x"], t["y"]) for t in scene.targets}


def _problem1_plan(targets: Dict[str, Tuple[float, float]]):
    """Waypoint plan for scenes/problem1.json."""

    return [
        (
            "Mission 1",
            [
                targets["A"],
                (0.24, 0.20),  # skirt left of the central obstacle
                targets["B"],
            ],
        ),
        (
            "Mission 2",
            [
                targets["B"],
                (0.52, -0.24),
                targets["C"],
            ],
        ),
        (
            "Mission 3",
            [
                targets["C"],
                (0.52, -0.26),
                (0.24, -0.26),
                targets["A"],
            ],
        ),
    ]


def _problem2_plan(targets: Dict[str, Tuple[float, float]]):
    """Waypoint plan for scenes/problem2.json."""

    return [
        ("Mission 1", [targets["A"], targets["B"]]),
        ("Mission 2", [targets["B"], targets["C"]]),
        ("Mission 3", [targets["C"], targets["A"]]),
    ]


def _problem3_plan(targets: Dict[str, Tuple[float, float]]):
    """Waypoint plan for scenes/problem3.json."""

    return [
        (
            "Mission 1",
            [
                targets["A"],
                (0.24, 0.26),
                targets["B"],
            ],
        ),
        (
            "Mission 2",
            [
                targets["B"],
                (0.24, 0.26),
                (0.40, 0.26),
                (0.40, -0.22),
                targets["C"],
            ],
        ),
        (
            "Mission 3",
            [
                targets["C"],
                (0.42, -0.26),
                targets["D"],
            ],
        ),
        (
            "Mission 4",
            [
                targets["D"],
                (0.24, -0.18),
                (0.24, 0.26),
                targets["A"],
            ],
        ),
    ]


def _solve_scene(scene_path: str) -> int:
    """Run the autonomous solver for supported problem scenes."""

    scene = load_scene(scene_path)
    if scene is None:
        print(f"Failed to load scene: {scene_path}")
        return 1

    target_map = _build_target_map(scene)
    scene_name = Path(scene_path).name
    if scene_name == "problem1.json":
        plan = _problem1_plan(target_map)
    elif scene_name == "problem2.json":
        plan = _problem2_plan(target_map)
    elif scene_name == "problem3.json":
        plan = _problem3_plan(target_map)
    else:
        print(f"Solver not implemented for scene: {scene_name}")
        return 1

    robot = ScaraRobot(
        L1=scene.robot_config["L1"],
        L2=scene.robot_config["L2"],
        joint_limits=scene.robot_config["joint_limits"],
    )

    server = SimulatorServer(robot, scene, host="localhost", port=8008, scene_path=scene_path)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Give the server a moment to open the socket
    time.sleep(0.5)

    controller = RobotController(verbose=True)
    try:
        if not controller.connect():
            return 1

        if not controller.load_scene(scene_path):
            return 1

        for mission_id, waypoints in plan:
            print(f"Starting {mission_id}")
            if not controller.start_mission(mission_id):
                return 1

            for xy in waypoints:
                if not controller.move_to(list(xy), elbow="up"):
                    return 1

            if not controller.complete_mission(mission_id):
                return 1

        controller.reset()
    finally:
        try:
            controller.disconnect()
        except Exception:
            pass
        server.stop()

    print("Solver finished all missions successfully.")
    return 0


def _parse_args(argv: Iterable[str]):
    parser = argparse.ArgumentParser(description="SCARA simulator GUI and solver")
    parser.add_argument("scene", nargs="?", help="Path to a scene JSON file")
    parser.add_argument("--solve", action="store_true", help="Run the autonomous solver and exit")
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for GUI application."""
    # Get scene path from command line
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    scene_path = None
    if args.scene:
        scene_path = args.scene
        if not Path(scene_path).exists():
            print(f"Warning: Scene file not found: {scene_path}")
            print("Using default scene instead.")
            scene_path = None
    else:
        # Try default scene
        default_scene = Path("scenes/demo.json")
        if default_scene.exists():
            scene_path = str(default_scene)

    # Create root window
    root = tk.Tk()

    # Set icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except Exception:
        pass

    # Create GUI
    try:
        app = SimulatorGUI(root, scene_path)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        # Start main loop
        root.mainloop()

    except Exception as e:
        import traceback
        print("Error starting GUI:")
        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
