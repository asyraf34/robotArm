#!/usr/bin/env python
"""Minimal client skeleton for quick starts."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController


def main():
    with RobotController() as robot:
        robot.load_scene("scenes/problem1.json")
        robot.start_mission("Mission 1")
        robot.move_to([0.40, 0.10])
        robot.move_to([0.22, -0.30])
        success = robot.complete_mission("Mission 1")
        if success:
            print("Mission 1 completed successfully.")
        else:
            print("Mission 1 failed.")
        robot.reset()


if __name__ == "__main__":
    main()
