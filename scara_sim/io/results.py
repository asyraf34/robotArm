"""
Result persistence for simulation runs.
"""

import json
import csv
from pathlib import Path
from typing import Optional
import numpy as np


class ResultWriter:
    """Writes simulation results to disk."""

    def __init__(self, outdir: str):
        """
        Initialize result writer.

        Parameters
        ----------
        outdir : str
            Output directory path.
        """
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        traj: dict,
        meta: dict,
        scene,
        run_id: str,
        video_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
    ) -> None:
        """
        Save complete run results.

        Parameters
        ----------
        traj : dict
            Trajectory data from planner.
        meta : dict
            Metadata (planner name, config, etc.).
        scene : Scene
            Scene used for planning.
        run_id : str
            Unique run identifier.
        video_path : Optional[str]
            Path to video file if generated.
        frames_dir : Optional[str]
            Path to frames directory if generated.
        """
        # Save metadata JSON
        meta_data = {
            "run_id": run_id,
            "planner": meta.get("planner", "unknown"),
            "config": traj.get("meta", {}),
            "success": traj.get("success", False),
            "planning_time": traj.get("planning_time", 0.0),
            "path_length": traj.get("path_length"),
            "clearance": traj.get("clearance"),
            "nodes_explored": traj.get("nodes_explored", 0),
            "video": video_path,
            "frames": frames_dir,
        }

        meta_path = self.outdir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

        # Save scene
        from scara_sim.io.scene import dump_scene

        scene_path = self.outdir / "scene.json"
        dump_scene(scene, str(scene_path))

        # Save trajectory CSV
        if traj.get("success") and traj.get("waypoints"):
            self._save_trajectory_csv(traj)

    def _save_trajectory_csv(self, traj: dict) -> None:
        """
        Save trajectory to CSV file.

        Parameters
        ----------
        traj : dict
            Trajectory with waypoints and times.
        """
        csv_path = self.outdir / "trajectory.csv"
        waypoints = traj["waypoints"]
        times = traj.get("times")

        if times is None or len(times) != len(waypoints):
            times = list(range(len(waypoints)))

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "q1", "q2", "q3", "q4", "x", "y"])

            for t, q in zip(times, waypoints):
                # Compute FK for x, y (assumes 2-DOF)
                q1, q2 = q[0], q[1]
                # Note: would need robot instance for accurate FK
                # For now, write q values
                row = [t] + list(q)
                writer.writerow(row)
