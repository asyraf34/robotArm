"""
Scene loading and saving (JSON format).
"""

from dataclasses import dataclass, field
from pathlib import Path
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
    # Resolve the scene path. When the simulator is executed from a directory
    # other than the repository root (for example, when launched from a
    # packaged install), relative paths like ``scenes/problem1.json`` may not
    # resolve correctly. To make the behaviour robust, we try multiple
    # locations:
    #   1) The path as provided (absolute or relative to cwd).
    #   2) Relative to the project root (two levels above this file).
    #   3) Inside a ``scenes`` directory under the project root.
    scene_path = Path(path)
    if not scene_path.exists():
        project_root = Path(__file__).resolve().parents[2]

        candidates = [project_root / scene_path]

        # If the provided path is not already prefixed with "scenes", also
        # consider a direct lookup under the scenes directory.
        if scene_path.parts[0] != "scenes":
            candidates.append(project_root / "scenes" / scene_path.name)

        for candidate in candidates:
            if candidate.exists():
                scene_path = candidate
                break
        else:
            raise FileNotFoundError(f"Scene file not found: {path}")

    with open(scene_path, "r") as f:
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
