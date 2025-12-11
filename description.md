# GUI and Solver Approach

`run_gui.py` acts as both the entry point for the SCARA simulator GUI and a simple autonomous solver. The script accepts an optional scene path and a `--solve` flag to switch between interactive and automated modes.

## Automated solver flow
- Scenes are loaded and validated before planning begins. A small helper builds a label-to-coordinate map for targets in the scene to simplify planning.
- Problem-specific waypoint plans are embedded for the provided challenge scenes. Each plan is a sequence of missions, and every mission lists Cartesian waypoints the robot should visit in order.
- The solver spins up a `SimulatorServer` with the configured `ScaraRobot` in a background thread, connects a `RobotController`, and executes each mission by issuing `start_mission`, `move_to` commands for every waypoint (using an "elbow up" posture), and `complete_mission` once the path is finished.
- After all missions succeed, the controller resets the robot, and the server shuts down cleanly.
- The result can be seen on the terminal.

## GUI mode flow
- Without `--solve`, the script optionally loads a provided scene (falls back to the default demo) and constructs a `SimulatorGUI` window.
- The window is centered on the screen after initialization, and any startup errors are caught and reported before exiting.
- But currently, for some reason, the GUI is not working

Use `python run_gui.py scenes/[problem.json] --solve` to run the solver on a supported problem scene, or omit `--solve` to open the interactive GUI.

For example, to test on `problem1.json`, run 
```
python run_gui.py scenes/problem1.json --solve
```  
