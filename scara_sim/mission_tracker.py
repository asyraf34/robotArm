"""Mission progress tracking system.

This module provides tracking and logging of mission execution, including:
- Robot position and state at each step
- Mission timing and completion status
- Historical mission data

Progress is logged to separate files, not stored in scene descriptions.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class MissionStep:
    """Represents a single step in mission execution."""

    def __init__(self, mission_id: str, step_type: str, xy: List[float],
                 q: List[float], timestamp: float):
        """
        Initialize a mission step.

        Args:
            mission_id: ID of the mission
            step_type: Type of step (e.g., "move", "pickup", "place", "reset")
            xy: End-effector position [x, y]
            q: Joint angles [q1, q2, q3, q4]
            timestamp: Unix timestamp when step was recorded
        """
        self.mission_id = mission_id
        self.step_type = step_type
        self.xy = xy
        self.q = q
        self.timestamp = timestamp
        self.elapsed_time = 0.0  # Will be set relative to mission start

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "mission_id": self.mission_id,
            "step_type": self.step_type,
            "xy": self.xy,
            "q": self.q,
            "timestamp": self.timestamp,
            "elapsed_time": self.elapsed_time
        }


class MissionRecord:
    """Records execution of a complete mission."""

    def __init__(self, mission_id: str, mission_definition: Dict[str, Any]):
        """
        Initialize a mission record.

        Args:
            mission_id: ID of the mission
            mission_definition: Mission definition from scene (pickup_target, delivery_target, etc.)
        """
        self.mission_id = mission_id
        self.mission_definition = mission_definition
        self.start_time = None
        self.end_time = None
        self.steps: List[MissionStep] = []
        self.status = "pending"  # pending, in_progress, completed, failed
        self.error_message = None

    @property
    def duration(self) -> Optional[float]:
        """Get mission duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def start(self):
        """Mark mission as started."""
        self.start_time = time.time()
        self.status = "in_progress"

    def complete(self):
        """Mark mission as completed successfully."""
        self.end_time = time.time()
        self.status = "completed"

    def fail(self, error_message: str = ""):
        """Mark mission as failed."""
        self.end_time = time.time()
        self.status = "failed"
        self.error_message = error_message

    def add_step(self, step_type: str, xy: List[float], q: List[float]):
        """
        Add a step to the mission record.

        Args:
            step_type: Type of step (move, pickup, place, etc.)
            xy: End-effector position [x, y]
            q: Joint angles [q1, q2, q3, q4]
        """
        step = MissionStep(self.mission_id, step_type, xy, q, time.time())

        # Calculate elapsed time from mission start
        if self.start_time:
            step.elapsed_time = step.timestamp - self.start_time

        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert mission record to dictionary."""
        return {
            "mission_id": self.mission_id,
            "mission_definition": self.mission_definition,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error_message": self.error_message,
            "steps": [step.to_dict() for step in self.steps]
        }


class MissionTracker:
    """Tracks missions and logs progress to file."""

    def __init__(self, log_dir: str = "mission_logs"):
        """
        Initialize the mission tracker.

        Args:
            log_dir: Directory where mission logs will be saved
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.missions: Dict[str, MissionRecord] = {}
        self.current_mission_id: Optional[str] = None
        self.log_file: Optional[Path] = None

    def create_log_file(self, scene_name: str = "scene"):
        """Create a new log file for this tracking session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{scene_name}_{timestamp}.json"
        self.log_file = self.log_dir / log_filename

    def start_mission(self, mission_id: str, mission_definition: Dict[str, Any]):
        """
        Start tracking a new mission.

        Args:
            mission_id: ID of the mission to track
            mission_definition: Mission configuration from scene file
        """
        record = MissionRecord(mission_id, mission_definition)
        record.start()
        self.missions[mission_id] = record
        self.current_mission_id = mission_id

    def log_step(self, step_type: str, xy: List[float], q: List[float]):
        """
        Log a step in the current mission.

        Args:
            step_type: Type of step (move, pickup, place, etc.)
            xy: End-effector position [x, y]
            q: Joint angles [q1, q2, q3, q4]
        """
        if not self.current_mission_id:
            return

        mission = self.missions[self.current_mission_id]
        mission.add_step(step_type, xy, q)

    def complete_mission(self, mission_id: str):
        """Mark a mission as completed."""
        if mission_id in self.missions:
            self.missions[mission_id].complete()

    def fail_mission(self, mission_id: str, error_message: str = ""):
        """Mark a mission as failed."""
        if mission_id in self.missions:
            self.missions[mission_id].fail(error_message)

    def get_mission_record(self, mission_id: str) -> Optional[MissionRecord]:
        """Get a mission record."""
        return self.missions.get(mission_id)

    def get_all_missions(self) -> Dict[str, MissionRecord]:
        """Get all mission records."""
        return self.missions.copy()

    def save_progress(self):
        """Save all mission progress to log file."""
        if not self.log_file:
            self.create_log_file()

        progress = {
            "timestamp": datetime.now().isoformat(),
            "total_missions": len(self.missions),
            "missions": {mid: record.to_dict()
                        for mid, record in self.missions.items()}
        }

        with open(self.log_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self, log_file: str) -> Dict[str, Any]:
        """
        Load mission progress from a log file.

        Args:
            log_file: Path to the log file

        Returns:
            Dictionary with mission data
        """
        with open(log_file, 'r') as f:
            data = json.load(f)

        # Load missions from file
        for mission_id, mission_data in data.get("missions", {}).items():
            record = MissionRecord(mission_id, mission_data["mission_definition"])
            record.status = mission_data["status"]
            record.start_time = mission_data.get("start_time")
            record.end_time = mission_data.get("end_time")
            record.error_message = mission_data.get("error_message")

            # Load steps
            for step_data in mission_data.get("steps", []):
                step = MissionStep(
                    step_data["mission_id"],
                    step_data["step_type"],
                    step_data["xy"],
                    step_data["q"],
                    step_data["timestamp"]
                )
                step.elapsed_time = step_data["elapsed_time"]
                record.steps.append(step)

            self.missions[mission_id] = record

        return data

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all missions."""
        completed = [m for m in self.missions.values()
                    if m.status == "completed"]
        failed = [m for m in self.missions.values()
                 if m.status == "failed"]
        in_progress = [m for m in self.missions.values()
                      if m.status == "in_progress"]

        total_time = sum(m.duration for m in completed if m.duration)

        return {
            "total_missions": len(self.missions),
            "completed": len(completed),
            "failed": len(failed),
            "in_progress": len(in_progress),
            "total_execution_time": total_time,
            "log_file": str(self.log_file) if self.log_file else None
        }

    def print_summary(self):
        """Print mission summary to console."""
        summary = self.get_summary()

        print("\n" + "="*50)
        print("MISSION TRACKING SUMMARY")
        print("="*50)
        print(f"Total Missions: {summary['total_missions']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"In Progress: {summary['in_progress']}")
        print(f"Total Time: {summary['total_execution_time']:.2f}s")

        if summary['log_file']:
            print(f"Log File: {summary['log_file']}")

        print("\nMission Details:")
        for mission_id, record in self.missions.items():
            status_icon = "[OK]" if record.status == "completed" else "[FAIL]" if record.status == "failed" else "[IN_PROGRESS]"
            duration_str = f"{record.duration:.2f}s" if record.duration else "N/A"
            print(f"  {status_icon} {mission_id}: {record.status} ({duration_str})")

            if record.steps:
                print(f"      Steps: {len(record.steps)}")
                for i, step in enumerate(record.steps):
                    print(f"        {i+1}. {step.step_type}: xy={step.xy}, elapsed={step.elapsed_time:.2f}s")

            if record.error_message:
                print(f"      Error: {record.error_message}")

        print("="*50 + "\n")
