"""
2D visualization using Matplotlib with animation support.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon as MPLPolygon, Circle
import time


class Viewer:
    """2D top-down viewer for SCARA robot simulation."""

    def __init__(self, scene, robot):
        """
        Initialize viewer.

        Parameters
        ----------
        scene : Scene
            Scene with obstacles and targets.
        robot : ScaraRobot
            Robot model.
        """
        self.scene = scene
        self.robot = robot
        self.fig = None
        self.ax = None

    def play(
        self,
        traj: dict,
        realtime: bool = True,
        save_video: Optional[str] = None,
        save_frames_dir: Optional[str] = None,
        headless: bool = False,
    ) -> None:
        """
        Animate trajectory.

        Parameters
        ----------
        traj : dict
            Trajectory from planner with waypoints.
        realtime : bool
            Play at real-time speed.
        save_video : Optional[str]
            Path to save MP4 video.
        save_frames_dir : Optional[str]
            Directory to save PNG frames.
        headless : bool
            Run without displaying window.
        """
        if not traj.get("success"):
            print("Cannot visualize failed trajectory")
            return

        waypoints = traj["waypoints"]
        times = traj.get("times")
        dt = traj.get("dt", 0.01)

        if times is None:
            times = np.arange(len(waypoints)) * dt

        # Setup figure
        if headless or save_video:
            plt.ioff()
        else:
            plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("SCARA Robot Simulator")

        # Set axis limits
        reach = self.robot.L1 + self.robot.L2
        margin = 0.1
        self.ax.set_xlim(-reach - margin, reach + margin)
        self.ax.set_ylim(-reach - margin, reach + margin)

        # Draw static elements
        self._draw_obstacles()
        self._draw_targets()

        # Initialize robot artists
        (self.link_line,) = self.ax.plot([], [], "o-", linewidth=4, markersize=8, color="royalblue")
        (self.ee_marker,) = self.ax.plot([], [], "o", markersize=12, color="red", label="End Effector")
        self.ax.legend()

        # Animation function
        def animate(frame_idx):
            if frame_idx >= len(waypoints):
                return self.link_line, self.ee_marker

            q = waypoints[frame_idx]
            positions = self.robot.get_joint_positions(q)

            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            self.link_line.set_data(xs, ys)
            self.ee_marker.set_data([xs[-1]], [ys[-1]])

            return self.link_line, self.ee_marker

        # Create animation
        if realtime and times is not None:
            interval = dt * 1000  # ms
        else:
            interval = 50  # ms

        anim = FuncAnimation(
            self.fig,
            animate,
            frames=len(waypoints),
            interval=interval,
            blit=True,
            repeat=True,
        )

        # Save video
        if save_video:
            writer = FFMpegWriter(fps=int(1.0 / dt), bitrate=1800)
            anim.save(save_video, writer=writer)
            print(f"Video saved to {save_video}")

        # Save frames
        if save_frames_dir:
            from pathlib import Path

            frames_path = Path(save_frames_dir)
            frames_path.mkdir(parents=True, exist_ok=True)

            for i in range(len(waypoints)):
                animate(i)
                self.fig.canvas.draw()
                frame_path = frames_path / f"frame_{i:04d}.png"
                self.fig.savefig(frame_path, dpi=100)

            print(f"Frames saved to {save_frames_dir}")

        # Display
        if not headless:
            plt.show()
        else:
            plt.close(self.fig)

    def _draw_obstacles(self):
        """Draw obstacles on the plot."""
        for obs in self.scene.obstacles:
            if obs.get("type") == "polygon":
                points = obs.get("points", [])
                if len(points) >= 3:
                    poly = MPLPolygon(
                        points, closed=True, fill=True, facecolor="gray", edgecolor="black", alpha=0.7
                    )
                    self.ax.add_patch(poly)

    def _draw_targets(self):
        """Draw target positions on the plot."""
        for target in self.scene.targets:
            x = target.get("x", 0)
            y = target.get("y", 0)
            circle = Circle((x, y), 0.02, color="green", alpha=0.8, label="Target")
            self.ax.add_patch(circle)
