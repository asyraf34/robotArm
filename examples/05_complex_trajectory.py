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

    corners = [
        [0.5, 0.0],
        [0.5, 0.2],
        [0.3, 0.2],
        [0.3, 0.0],
    ]

    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]

        robot.line_motion(start, end, n_points=5)

    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]

        robot.line_motion(start, end, n_points=3)

    robot.reset()

    robot.disconnect()


if __name__ == "__main__":
    main()
