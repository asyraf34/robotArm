# SCARA Robot Simulator

Lightweight GUI + client/server simulator for a 4-DOF SCARA arm.

- Forward/Inverse kinematics, collision checking, straight-line + RRT planners
- Built-in GUI with mission list and animation
- Simple Python client API with ready-to-run examples

> Tested on Windows; other platforms are not covered.

## Quick Install (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run the GUI

```bash
python run_gui.py
```

Pick a scene from `scenes/` (easy/medium/hard/demo) and use the mission tab to watch progress.

## Run an Example Client

```bash
python examples/01_basic_movement.py
```

More samples are in `examples/` (missions, trajectories, collision demo).

## Scenes

Editable JSON files in `scenes/` define obstacles, targets, and missions. Tweak coordinates to make new layouts.

## Need Help?

See `examples/README.md` for a quick guide to the client functions used in the samples.
