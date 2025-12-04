#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController


def main():
    with RobotController() as robot:
        if not robot.load_scene("scenes/simple_scene.json"):
            return

        robot.state()
        robot.move_to([0.4, 0.1])
        robot.move_to([0.35, 0.25])
        robot.position()
        robot.reset()


if __name__ == "__main__":
    main()
