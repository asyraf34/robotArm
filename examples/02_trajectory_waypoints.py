#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import connect_robot


def main():
    robot = connect_robot()
    if not robot.load_scene("scenes/simple_scene.json"):
        robot.disconnect()
        return

    waypoints = [
        [0.5, 0.0],      # Start
        [0.4, 0.2],      # Top-left
        [0.3, 0.2],      # Higher-left
        [0.4, 0.0],      # Bottom-middle
        [0.5, 0.0],      # Back to start
    ]

    robot.execute_waypoints(waypoints)
    robot.disconnect()


if __name__ == "__main__":
    main()
