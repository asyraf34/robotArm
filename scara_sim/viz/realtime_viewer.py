"""
Real-time viewer for simulator server with live updates.

Features:
- Real-time robot visualization
- Start and goal point indicators
- Trajectory history with fading effect
- Smooth interpolation between positions
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon, Circle
from matplotlib.animation import FuncAnimation
from collections import deque


class RealtimeViewer:
    """Real-time 2D viewer for SCARA robot with server integration."""

    def __init__(self, scene, robot, server):
        """
        Initialize real-time viewer.

        Parameters
        ----------
        scene : Scene
            Scene with obstacles and targets.
        robot : ScaraRobot
            Robot model.
        server : SimulatorServer
            Simulator server instance.
        """
        self.scene = scene
        self.robot = robot
        self.server = server
        self.fig = None
        self.ax = None
        self.link_line = None
        self.ee_marker = None
        self.anim = None

        # Trajectory tracking
        self.trajectory_history = deque(maxlen=200)  # Keep last 200 positions
        self.trajectory_line = None
        self.trajectory_array = None  # Cache numpy array to avoid recreating every frame
        self.start_marker = None
        self.goal_marker = None
        self.goal_position = None
        self.start_position = None
        self.last_ee_pos = None
        self.movement_threshold = 0.01  # 1cm to detect goal change

        # Animation FPS (24-60, default 30)
        self.animation_fps = 30
        self.animation_interval = 33  # milliseconds (1000/30 = ~33ms)

    def start(self, interval: int = None, fps: int = 30):
        """
        Start real-time visualization.

        Parameters
        ----------
        interval : int, optional
            Update interval in milliseconds. If provided, fps is ignored.
            Default: None (use fps parameter instead).
        fps : int
            Frames per second (24-60, default 30).
            Only used if interval is None.
        """
        # Set animation FPS (enforce minimum 24 FPS)
        self.animation_fps = max(24, min(60, fps))
        self.animation_interval = max(1, int(1000.0 / self.animation_fps))

        # Use provided interval if specified, otherwise use calculated interval
        if interval is not None:
            animation_interval = interval
        else:
            animation_interval = self.animation_interval
        # Setup figure
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("SCARA Robot Simulator - Real-time View")

        # Set axis limits
        reach = self.robot.L1 + self.robot.L2
        margin = 0.1
        self.ax.set_xlim(-reach - margin, reach + margin)
        self.ax.set_ylim(-reach - margin, reach + margin)

        # Draw static elements
        self._draw_obstacles()
        self._draw_targets()

        # Initialize robot artists
        (self.link_line,) = self.ax.plot(
            [], [], "o-", linewidth=4, markersize=8, color="royalblue", label="Robot"
        )
        (self.ee_marker,) = self.ax.plot(
            [], [], "o", markersize=12, color="red", label="End Effector"
        )

        # Initialize trajectory visualization
        (self.trajectory_line,) = self.ax.plot(
            [], [], "-", linewidth=1.5, color="orange", alpha=0.5, label="Trajectory"
        )

        # Start point marker (green)
        self.start_marker = Circle((0, 0), 0.03, color="lime", alpha=0, zorder=5)
        self.ax.add_patch(self.start_marker)

        # Goal point marker (blue)
        self.goal_marker = Circle((0, 0), 0.035, color="dodgerblue", alpha=0,
                                 edgecolor="darkblue", linewidth=2, zorder=5)
        self.ax.add_patch(self.goal_marker)

        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue",
                      markersize=10, label="Robot"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                      markersize=10, label="End Effector"),
            plt.Line2D([0], [0], color="orange", linewidth=1.5, alpha=0.5, label="Trajectory"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
                      markersize=8, label="Start"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
                      markersize=8, label="Goal"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right")

        # Create animation
        self.anim = FuncAnimation(
            self.fig, self._update, interval=animation_interval, blit=False, cache_frame_data=False
        )

        # Show plot
        plt.show(block=False)

    def _update(self, frame):
        """Update visualization with current robot state."""
        # Get current state from server
        q = self.server.get_current_state()
        positions = self.robot.get_joint_positions(q)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Current end-effector position
        current_ee_pos = np.array([xs[-1], ys[-1]])

        # Initialize start position on first frame
        if self.start_position is None:
            self.start_position = current_ee_pos.copy()
            self.goal_position = current_ee_pos.copy()
            self._update_start_marker()
            self._update_goal_marker()

        self.last_ee_pos = current_ee_pos

        # Check if robot has moved significantly (new goal)
        if self.goal_position is not None:
            distance_to_goal = np.linalg.norm(current_ee_pos - self.goal_position)
            if distance_to_goal > self.movement_threshold:
                # Goal has changed, update start position for new movement
                self.start_position = self.goal_position.copy()
                self.goal_position = current_ee_pos.copy()
                self._update_start_marker()
                self._update_goal_marker()

        # Add position to trajectory history
        self.trajectory_history.append(current_ee_pos.copy())

        # Update trajectory line (cache array to avoid expensive conversion every frame)
        if len(self.trajectory_history) > 1:
            self.trajectory_array = np.array(list(self.trajectory_history))
            self.trajectory_line.set_data(self.trajectory_array[:, 0], self.trajectory_array[:, 1])

        # Update robot visualization
        self.link_line.set_data(xs, ys)
        self.ee_marker.set_data([xs[-1]], [ys[-1]])

        return self.link_line, self.ee_marker, self.trajectory_line, self.start_marker, self.goal_marker

    def _update_start_marker(self):
        """Update start point marker position."""
        if self.start_position is not None and self.start_marker is not None:
            self.start_marker.set_center(self.start_position)
            self.start_marker.set_alpha(0.7)

    def _update_goal_marker(self):
        """Update goal point marker position."""
        if self.goal_position is not None and self.goal_marker is not None:
            self.goal_marker.set_center(self.goal_position)
            self.goal_marker.set_alpha(0.8)

    def _draw_obstacles(self):
        """Draw obstacles on the plot."""
        for obs in self.scene.obstacles:
            if obs.get("type") == "polygon":
                points = obs.get("points", [])
                if len(points) >= 3:
                    poly = MPLPolygon(
                        points,
                        closed=True,
                        fill=True,
                        facecolor="gray",
                        edgecolor="black",
                        alpha=0.7,
                    )
                    self.ax.add_patch(poly)

    def _draw_targets(self):
        """Draw target positions on the plot."""
        for target in self.scene.targets:
            x = target.get("x", 0)
            y = target.get("y", 0)
            circle = Circle((x, y), 0.02, color="green", alpha=0.8)
            self.ax.add_patch(circle)
            # FIX: Skip expensive text labels with styled boxes for performance
            # Text creation with bbox is slow and unnecessary during animation
            # Simple text without bbox is still readable:
            self.ax.text(
                x + 0.03,
                y + 0.03,
                "T",
                fontsize=7,
                color="green",
            )

    def update_scene(self, new_scene):
        """Update the viewer with a new scene (called when client loads a scene).

        Parameters
        ----------
        new_scene : Scene
            New scene to display.
        """
        if not self.ax or not self.fig:
            return  # Viewer not initialized yet

        # Update internal scene reference
        self.scene = new_scene

        # CRITICAL FIX #4: Clear trajectory history when loading new scene
        # This prevents unused data from previous scenes accumulating in memory
        self.trajectory_history.clear()
        self.trajectory_array = None
        self.last_ee_pos = None
        self.start_position = None
        self.goal_position = None

        # Clear the plot but keep axis and other elements
        # Remove all patches (polygons and circles) but keep the plot artists
        # FIX: Use try-except to handle NotImplementedError from matplotlib during rendering
        try:
            for patch in list(self.ax.patches):
                try:
                    patch.remove()
                except (RuntimeError, NotImplementedError):
                    # Patch may be in the middle of rendering, skip it
                    pass
        except Exception as e:
            # If patch removal fails, continue anyway
            print(f"[WARNING] Error removing patches: {e}")

        # Redraw obstacles and targets
        self._draw_obstacles()
        self._draw_targets()

        # Update canvas
        try:
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"[WARNING] Error updating canvas: {e}")

    def stop(self):
        """Stop the real-time viewer."""
        # Stop animation
        if self.anim:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None

        # CRITICALLY: Close the figure (FIX #3 - closes 2-5MB per figure)
        if self.fig:
            try:
                plt.close(self.fig)
            except:
                pass
            self.fig = None
            self.ax = None

        # Clear trajectory history
        self.trajectory_history.clear()
        self.trajectory_array = None
        self.last_ee_pos = None

    def keep_alive(self):
        """Keep viewer window open (blocking)."""
        if self.fig:
            plt.show()
