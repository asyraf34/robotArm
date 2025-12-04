#!/usr/bin/env python
"""Demo 3: Collision Detection and Handling"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from client_tools import RobotController

results = []

with RobotController() as robot:
    robot.load_scene("scenes/hard_scene.json")
    robot.reset()

    mission_times = {}

    def complete_with_time(mid: str):
        """Complete mission and return (success, elapsed_time)."""
        resp = robot.client._send_command({"cmd": "complete_mission", "mission_id": mid})
        if not resp:
            return False, None
        if resp.get("status") == "queued":
            cmd_id = resp.get("command_id")
            status = robot.client.wait_for_command(cmd_id, timeout=10.0)
            if not status or status.get("status") != "completed":
                return False, None
            resp = status.get("response", {})
        success = resp.get("status") == "ok"
        elapsed = resp.get("elapsed_time")
        mission_times[mid] = elapsed
        return success, elapsed

    # Mission 1: Unreachable target
    robot.start_mission("Mission 1")
    success = robot.move_to([0.720, 0.350])
    # Always ask server to finalize mission to get elapsed time (even on fail)
    mission_ok, elapsed = complete_with_time("Mission 1")
    success = success and mission_ok
    results.append(("Mission 1", success, elapsed))

    # Mission 2: B -> C
    robot.start_mission("Mission 2")
    s1 = robot.move_to([0.600, 0.300])
    s2 = robot.move_to([0.500, -0.320]) if s1 else False
    mission_ok, elapsed = complete_with_time("Mission 2")
    success = s1 and s2 and mission_ok
    results.append(("Mission 2", success, elapsed))

    # Mission 3: C -> D
    robot.start_mission("Mission 3")
    s1 = robot.move_to([0.500, -0.320])
    s2 = robot.move_to([0.200, -0.360]) if s1 else False
    mission_ok, elapsed = complete_with_time("Mission 3")
    success = s1 and s2 and mission_ok
    results.append(("Mission 3", success, elapsed))

    # Mission 4: D -> A (unreachable)
    robot.start_mission("Mission 4")
    s1 = robot.move_to([0.440, -0.160])  # inside obstacle to trigger collision
    s2 = robot.move_to([0.160, 0.320]) if s1 else False
    mission_ok, elapsed = complete_with_time("Mission 4")
    success = s1 and s2 and mission_ok
    results.append(("Mission 4", success, elapsed))

print("\n" + "="*40)
for name, ok, elapsed in results:
    status = "[OK]" if ok else "[FAIL]"
    t_str = f" ({elapsed:.2f}s)" if elapsed is not None else ""
    print(f"{status} {name}{t_str}")
print(f"Total: {sum(1 for _, ok, _ in results if ok)}/{len(results)}")
# Detailed mission times
if mission_times:
    print("Mission times:")
    for mid, t in mission_times.items():
        if t is not None:
            print(f"  {mid}: {t:.2f}s")
        else:
            print(f"  {mid}: (no time reported)")
print("="*40)
