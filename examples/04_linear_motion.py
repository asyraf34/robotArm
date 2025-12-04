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

    robot.line_motion(
        start_xy=[0.5, 0.0],
        end_xy=[0.2, 0.3],
        n_points=8,
    )

    robot.line_motion(
        start_xy=[0.2, 0.3],
        end_xy=[0.5, 0.0],
        n_points=8,
    )

    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
