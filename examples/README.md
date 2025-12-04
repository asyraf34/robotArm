# Examples Quick Guide

Below are the core functions you’ll use, their inputs, and what they return.

## Connecting

- `connect_robot(host="localhost", port=8008, scene_path="scenes/demo.json", verbose=True)` → `RobotController`
  - Creates a controller and loads the scene on the server.
  - Usually just change the scene and keep other defaults.
- Context manager: `with RobotController() as robot:` auto-connects and disconnects.

## Movement

- `robot.move_to([x, y], elbow="up")` → `bool`
  - Moves the end-effector to `[x, y]`. Returns `True` on success.
  - The `elbow` parameter specifies the robot elbow direction ("up" or "down"). The default "up" is recommended.

- `robot.execute_waypoints([[x1, y1], ...])` → `int`
  - Runs a list of waypoints sequentially. Returns how many were reached.

- `robot.line_motion(start_xy, end_xy, n_points=8)` → `int`
  - Executes a straight line through `n_points` interpolated waypoints.

- `robot.reset()` → `bool`
  - Moves back to home position `[0, 0, 0, 0]`.
  - Useful for returning to the initial position after completing all missions.

## Missions

- `robot.start_mission("Mission N")` → `bool`
- `robot.complete_mission("Mission N")` → `bool`
  - Use these to mark the start and end of each mission. Mission names must match those defined in the scene JSON.

## Scene and State

- `robot.state()` → `dict`
  - Current joint angles and end-effector position.
- `robot.position()` → `[x, y]`
  - Current end-effector position only.

## Typical Pattern

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController

with RobotController() as robot:
    robot.load_scene("scenes/easy_scene.json")
    robot.start_mission("Mission 1")
    robot.move_to([0.46, 0.12])
    robot.move_to([0.22, -0.30])
    robot.complete_mission("Mission 1")
```

All timing/animation is handled by the server; you don’t need to add sleeps or delays.
