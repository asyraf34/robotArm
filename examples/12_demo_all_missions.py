#!/usr/binenv python
"""Demo 2: Multiple Missions with Waypoints"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController

results = []

with RobotController() as robot:
    robot.load_scene("scenes/easy_scene.json")
    robot.reset()

    robot.start_mission("Mission 1")
    success = robot.execute_waypoints([[0.460, 0.120], [0.220, -0.300]]) == 2
    robot.complete_mission("Mission 1")
    results.append(("Mission 1", success))

    robot.start_mission("Mission 2")
    success = robot.execute_waypoints([[0.220, -0.300], [0.460, 0.120]]) == 2
    robot.complete_mission("Mission 2")
    results.append(("Mission 2", success))

    robot.start_mission("Mission 3 (with stopovers)")
    success = robot.execute_waypoints([[0.460, 0.120], [0.300, 0.260], [0.540, -0.180], [0.220, -0.300]]) == 4
    robot.complete_mission("Mission 3 (with stopovers)")
    results.append(("Mission 3", success))

print("\n" + "="*40)
for name, ok in results:
    status = "[OK]" if ok else "[FAIL]"
    print(f"{status} {name}")
print(f"Total: {sum(1 for _, ok in results if ok)}/{len(results)}")
print("="*40)
