"""
SCARA Robot Simulator - path planning and visualization toolkit.
"""

__version__ = "0.1.0"
__author__ = "SCARA Sim Contributors"

from scara_sim.core.robot import ScaraRobot
from scara_sim.io.scene import Scene, load_scene, dump_scene
from scara_sim.viz.viewer2d import Viewer
from scara_sim.io.results import ResultWriter

__all__ = [
    "ScaraRobot",
    "Scene",
    "load_scene",
    "dump_scene",
    "Viewer",
    "ResultWriter",
]
