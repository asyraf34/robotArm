"""
Scene loading and saving (JSON format).
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Scene:
    """
    Scene representation with obstacles, targets, and missions.
    """

    obstacles: list[dict] = field(default_factory=list)
    targets: list[dict] = field(default_factory=list)
    missions: list[dict] = field(default_factory=list)
    robot_config: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "obstacles": self.obstacles,
            "targets": self.targets,
            "robot": self.robot_config,
        }
        if self.missions:
            result["missions"] = self.missions
        return result

    @staticmethod
    def from_dict(data: dict) -> "Scene":
        """Create Scene from dictionary."""
        return Scene(
            obstacles=data.get("obstacles", []),
            targets=data.get("targets", []),
            missions=data.get("missions", []),
            robot_config=data.get("robot"),
        )


def load_scene(path: str) -> Scene:
    """
    Load scene from JSON file.

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    Scene
        Loaded scene.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return Scene.from_dict(data)


def dump_scene(scene: Scene, path: str) -> None:
    """
    Save scene to JSON file.

    Parameters
    ----------
    scene : Scene
        Scene to save.
    path : str
        Output path.
    """
    with open(path, "w") as f:
        json.dump(scene.to_dict(), f, indent=2)
