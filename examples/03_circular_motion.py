#!/usr/bin/env python
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import connect_robot


def generate_safe_circle(center_x, center_y, radius, n_points=16, clockwise=True):
    waypoints = []
    direction = 1 if clockwise else -1

    for i in range(n_points):
        angle = direction * 2 * np.pi * i / n_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        waypoints.append([x, y])

    return waypoints


def main():
    robot = connect_robot()

    if not robot.load_scene("scenes/simple_scene.json"):
        robot.disconnect()
        return

    circle_1 = generate_safe_circle(center_x=0.35, center_y=0.25, radius=0.12, n_points=16, clockwise=True)
    robot.execute_waypoints(circle_1)

    circle_2 = generate_safe_circle(center_x=0.35, center_y=0.25, radius=0.12, n_points=12, clockwise=False)
    robot.execute_waypoints(circle_2)

    robot.reset()
    robot.disconnect()


if __name__ == "__main__":
    main()
