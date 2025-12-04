#!/usr/bin/env python
"""Demo 1: Simple Pick-and-Place"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController

with RobotController() as robot:
    robot.load_scene("scenes/simple_scene.json")
    robot.reset()
    robot.start_mission("Mission 1")
    robot.move_to([0.4, 0.1])
    robot.move_to([0.3, -0.15])
    robot.complete_mission("Mission 1")
    print("[OK] Mission completed!")
