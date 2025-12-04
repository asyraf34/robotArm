"""
Modern GUI for SCARA robot simulator with integrated visualization and controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MPLPolygon, Circle, Rectangle
from matplotlib.collections import LineCollection
import threading
import queue
import time
from collections import deque

from scara_sim.core.robot import ScaraRobot
from scara_sim.io.scene import load_scene, dump_scene, Scene
from scara_sim.core.collision import CollisionChecker
from scara_sim.core.trajectory import check_trajectory_collision
from scara_sim.planning import rrt, straight
from scara_sim.server import SimulatorServer
from scara_sim.server_utils import is_port_in_use, kill_process_on_port, get_process_using_port
from examples.client_tools import MissionMonitor


class SimulatorGUI:
    """Main GUI window for SCARA robot simulator."""

    def __init__(self, root, scene_path=None):
        """
        Initialize GUI.

        Parameters
        ----------
        root : tk.Tk
            Root tkinter window.
        scene_path : str, optional
            Path to scene file to load on startup.
        """
        self.root = root
        self.root.title("SCARA Robot Simulator")
        self.root.geometry("1000x700")

        # Track current scene path
        self.current_scene_path = scene_path if scene_path else "(Default)"

        # Load scene
        if scene_path:
            self.scene = load_scene(scene_path)
        else:
            # Default scene
            self.scene = Scene(
                obstacles=[
                    {
                        "type": "polygon",
                        "points": [[0.25, 0.05], [0.35, 0.05], [0.35, 0.20], [0.25, 0.20]],
                    }
                ],
                targets=[{"x": 0.40, "y": 0.10}],
                robot_config={
                    "L1": 0.35,
                    "L2": 0.25,
                    "joint_limits": {
                        "q1": [-3.14, 3.14],
                        "q2": [-2.2, 2.2],
                        "q3": [0.0, 0.25],
                        "q4": [-3.14, 3.14],
                    },
                },
            )

        # Create robot
        cfg = self.scene.robot_config
        self.robot = ScaraRobot(
            L1=cfg["L1"], L2=cfg["L2"], joint_limits=cfg["joint_limits"]
        )

        # Current robot state
        self.current_q = np.array([0.0, 0.0, 0.0, 0.0])
        self.collision_checker = CollisionChecker(self.robot, self.scene)

        # Trajectory for animation
        self.trajectory = None
        self.trajectory_index = 0
        self.trajectory_collision_info = None  # Collision data for current trajectory
        self.is_animating = False

        # Trajectory tracking for visualization
        self.trajectory_history = deque(maxlen=200)  # Keep last 200 positions
        self.start_position = None
        self.goal_position = None
        self.movement_threshold = 0.01  # 1cm to detect goal change

        # Animation and smoothing
        self.target_q = None  # Target joint configuration
        self.animation_fps = 30  # Frames per second (24-60 recommended)
        self.animation_duration = 0.5  # Duration of smooth motion in seconds (will be updated dynamically)
        self.current_animation_duration = 0.5  # Current animation duration from server
        self.animation_progress = 0.0  # Progress (0-1) of current animation
        self.last_q = np.array([0.0, 0.0, 0.0, 0.0])  # For interpolation
        self.animation_start_time = None
        self.updating_sliders = False  # Flag to prevent callback loop
        self.active_animation_deadline = 0.0  # Block mission transitions until current animation completes

        # Performance optimization: track when to redraw visualization
        self.last_visualized_q = None  # Track last joint config we visualized
        self.last_visualization_time = time.time()  # Debounce visualization redraws
        self.min_redraw_interval = 0.016  # Minimum 16ms between redraws (~60 FPS)

        # Server update callback throttling
        self.pending_server_update = False  # Flag to avoid queuing multiple updates

        # Frame timing diagnostics
        self.frame_times = deque(maxlen=30)  # Last 30 frame times for averaging
        self.profile_frame = False  # Enable/disable frame profiling
        self.trajectory_update_frequency = 10  # Update trajectory visualization every N frames to reduce IK/collision checks

        # Mission tracking
        self.missions = self.scene.missions.copy() if self.scene.missions else []
        # Reset all missions to pending on startup
        for mission in self.missions:
            mission["status"] = "pending"
        # Also reset scene missions
        for mission in self.scene.missions:
            mission["status"] = "pending"
        self.current_mission_idx = None
        self.last_loaded_scene_path = None  # Track which scene was last loaded to detect changes

        # Delete any persisted mission state file to start fresh
        try:
            import tempfile
            import os
            state_file = os.path.join(tempfile.gettempdir(), "mission_state.json")
            if os.path.exists(state_file):
                os.remove(state_file)
        except Exception:
            pass  # Ignore errors deleting state file
        self.mission_widgets = []  # Store mission display widgets for updates
        self.scene_label = None  # Label widget for scene display in missions tab
        self.mission_start_times = {}  # Track when missions start (by index)
        self.mission_elapsed_times = {}  # Store final elapsed time for completed missions
        self.last_sync_time = 0  # Debounce timer sync checks to avoid excessive updates
        self.pending_mission_events = deque()  # Queue mission events so we can gate them on animation state
        self.mission_event_lock = threading.Lock()  # Protect pending_mission_events across threads

        # Animation frame counter for throttling expensive operations
        self.frame_counter = 0

        # Mission monitoring (for autonomous mission solver events)
        # Will be initialized after scene loads with correct path
        self.mission_monitor = None

        # Server integration
        self.server = None
        self.server_running = False
        self.client_connected = False
        self.server_host = "localhost"
        self.server_port = 8008

        # Setup GUI components
        # Menu bar creation skipped (functions remain intact)
        self._setup_main_layout()
        self._setup_visualization()
        self._setup_control_panel()
        self._setup_status_bar()

        # Initialize server status display
        self._update_server_status_display()

        # Initial draw
        self._update_visualization()

        # Start update loop
        self._schedule_update()

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Auto-start server on GUI startup
        self.root.after(500, self._start_server)

    def _setup_menubar(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Scene...", command=self._load_scene)
        file_menu.add_command(label="Save Scene...", command=self._save_scene)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_command(label="Zoom to Fit", command=self._zoom_to_fit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _setup_main_layout(self):
        """Setup main layout with paned windows."""
        # Main paned window (left: visualization, right: controls)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left frame - Visualization
        self.viz_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.viz_frame, weight=3)

        # Right frame - Controls
        self.control_frame = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(self.control_frame, weight=1)

    def _setup_visualization(self):
        """Setup matplotlib visualization."""
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("SCARA Robot Workspace")

        # Set limits
        reach = self.robot.L1 + self.robot.L2
        margin = 0.1
        self.ax.set_xlim(-reach - margin, reach + margin)
        self.ax.set_ylim(-reach - margin, reach + margin)

        # Draw static elements
        self._draw_obstacles()
        self._draw_targets()

        # Robot artists
        (self.link_line,) = self.ax.plot(
            [], [], "o-", linewidth=4, markersize=8, color="royalblue", label="Robot", zorder=10
        )
        (self.ee_marker,) = self.ax.plot(
            [], [], "o", markersize=12, color="red", label="End Effector", zorder=11
        )

        # Trajectory visualization using LineCollection for collision-aware coloring
        self.trajectory_line_collection = LineCollection(
            [], linewidths=2, zorder=9
        )
        self.ax.add_collection(self.trajectory_line_collection)

        # Keep old trajectory_line for backward compatibility with legends
        (self.trajectory_line,) = self.ax.plot(
            [], [], "-", linewidth=0, color="orange", alpha=0, label="Trajectory", zorder=9
        )

        # Start point marker (green)
        self.start_marker = Circle((0, 0), 0.03, color="lime", alpha=0, zorder=5)
        self.ax.add_patch(self.start_marker)

        # Goal point marker (blue)
        # FIX: Use facecolor instead of color to avoid matplotlib warning
        self.goal_marker = Circle((0, 0), 0.035, facecolor="dodgerblue", alpha=0,
                                 edgecolor="darkblue", linewidth=2, zorder=5)
        self.ax.add_patch(self.goal_marker)

        # Legend disabled - cleans up visualization
        # Users can identify elements from mission task list panel

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(self.viz_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def _setup_control_panel(self):
        """Setup control panel with tabbed interface."""
        # Create notebook (tabs) - no padding to fit nicely
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Mission tab
        self.mission_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mission_tab, text="Mission")
        self.control_panel_frame = self.mission_tab  # For compatibility with _setup_missions_section
        self._setup_missions_section()

        # Server tab
        self.server_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.server_tab, text="Server")
        self._setup_server_tab()

        # Info tab
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="Info")
        self._setup_info_tab()

    def _setup_planning_section(self):
        """Setup planning section."""
        tab = ttk.LabelFrame(self.control_panel_frame, text="Planning", padding=10)
        tab.pack(fill=tk.X, padx=5, pady=5)

        # Planner selection
        planner_frame = ttk.LabelFrame(tab, text="Planner", padding=10)
        planner_frame.pack(fill=tk.X, padx=5, pady=5)

        self.planner_var = tk.StringVar(value="rrt")
        ttk.Radiobutton(
            planner_frame, text="RRT", variable=self.planner_var, value="rrt"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            planner_frame, text="Straight Line", variable=self.planner_var, value="straight"
        ).pack(anchor=tk.W)

        # RRT Parameters
        rrt_frame = ttk.LabelFrame(tab, text="RRT Parameters", padding=10)
        rrt_frame.pack(fill=tk.X, padx=5, pady=5)

        # Seed
        seed_frame = ttk.Frame(rrt_frame)
        seed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT)
        self.seed_entry = ttk.Entry(seed_frame, width=10)
        self.seed_entry.insert(0, "42")
        self.seed_entry.pack(side=tk.LEFT, padx=5)

        # Max nodes
        nodes_frame = ttk.Frame(rrt_frame)
        nodes_frame.pack(fill=tk.X, pady=2)
        ttk.Label(nodes_frame, text="Max Nodes:").pack(side=tk.LEFT)
        self.max_nodes_entry = ttk.Entry(nodes_frame, width=10)
        self.max_nodes_entry.insert(0, "5000")
        self.max_nodes_entry.pack(side=tk.LEFT, padx=5)

        # Goal Configuration
        goal_frame = ttk.LabelFrame(tab, text="Goal", padding=10)
        goal_frame.pack(fill=tk.X, padx=5, pady=5)

        # Goal X, Y
        gx_frame = ttk.Frame(goal_frame)
        gx_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gx_frame, text="Goal X:").pack(side=tk.LEFT)
        self.goal_x_entry = ttk.Entry(gx_frame, width=10)
        self.goal_x_entry.insert(0, "0.4")
        self.goal_x_entry.pack(side=tk.LEFT, padx=5)

        gy_frame = ttk.Frame(goal_frame)
        gy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gy_frame, text="Goal Y:").pack(side=tk.LEFT)
        self.goal_y_entry = ttk.Entry(gy_frame, width=10)
        self.goal_y_entry.insert(0, "0.1")
        self.goal_y_entry.pack(side=tk.LEFT, padx=5)

        # Plan button
        ttk.Button(goal_frame, text="Plan & Execute", command=self._plan_trajectory).pack(
            fill=tk.X, pady=5
        )

        # Test collision visualization button
        ttk.Button(goal_frame, text="Test Collision Viz", command=self._test_collision_visualization).pack(
            fill=tk.X, pady=2
        )

        # Results
        results_frame = ttk.LabelFrame(tab, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(results_frame, height=8, width=30, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def _setup_missions_section(self):
        """Setup missions section to display and track tasks."""
        if not self.missions:
            return  # Don't create section if no missions

        tab = ttk.LabelFrame(self.control_panel_frame, text="Missions", padding=10)
        tab.pack(fill=tk.X, padx=5, pady=5)

        # Display currently loaded scene
        scene_display = ttk.Frame(tab)
        scene_display.pack(fill=tk.X, padx=5, pady=5)

        # Extract scene name from path, handling both forward and backslashes
        if self.current_scene_path == "(Default)":
            scene_name = "Default"
        else:
            # Use Path to handle both Windows and Unix paths
            from pathlib import Path
            scene_name = Path(self.current_scene_path).name
        self.scene_label = ttk.Label(scene_display, text=f"Scene: {scene_name}", font=("TkDefaultFont", 9, "bold"))
        self.scene_label.pack(anchor=tk.W)

        # Mission list
        missions_frame = ttk.LabelFrame(tab, text="Task List", padding=5)
        missions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.mission_widgets = []

        for idx, mission in enumerate(self.missions):
            # Create mission display frame
            mission_frame = ttk.Frame(missions_frame)
            mission_frame.pack(fill=tk.X, pady=3, padx=5)

            # Mission label and status
            pickup = mission.get("pickup_target", "Unknown")
            delivery = mission.get("delivery_target", "Unknown")
            status = mission.get("status", "pending")

            # Color based on status
            status_color = "#ffeb3b" if status == "pending" else "#4caf50" if status == "completed" else "#2196f3"

            # Mission info
            info_text = f"{mission.get('id', f'Mission {idx+1}')}: {pickup} → {delivery}"

            label = ttk.Label(
                mission_frame,
                text=info_text,
                font=("TkDefaultFont", 9, "bold")
            )
            label.pack(side=tk.LEFT, padx=5)

            # Status indicator
            status_label = ttk.Label(
                mission_frame,
                text=status.upper(),
                font=("TkDefaultFont", 8)
            )
            status_label.pack(side=tk.LEFT, padx=5)

            # Complete button (hidden)
            complete_btn = ttk.Button(
                mission_frame,
                text="✓",
                width=3,
                command=lambda i=idx: self._mark_mission_complete(i)
            )
            complete_btn.pack_forget()

            # Store widget references for updates (button kept for logic, not shown)
            self.mission_widgets.append({
                'frame': mission_frame,
                'label': label,
                'status_label': status_label,
                'button': complete_btn,
                'mission': mission
            })

        # Control buttons (hidden)
        button_frame = ttk.Frame(tab)
        button_frame.pack_forget()

        # CRITICAL: After widgets are created, refresh display to show any persisted status
        # This ensures missions that completed while widgets were being recreated
        # still show their completion status immediately
        self._refresh_missions_display()

    def _mark_mission_complete(self, mission_idx: int):
        """Mark a mission as completed or start it (in_progress)."""
        if mission_idx < len(self.missions):
            current_status = self.missions[mission_idx]["status"]

            if current_status == "pending":
                # Start the mission - transition to in_progress and start timer
                self.missions[mission_idx]["status"] = "in_progress"
                self.scene.missions[mission_idx]["status"] = "in_progress"
                self.mission_start_times[mission_idx] = time.time()
                self.current_mission_idx = mission_idx  # Set current mission
            elif current_status == "in_progress":
                # Complete the mission - record elapsed time
                self.missions[mission_idx]["status"] = "completed"
                self.scene.missions[mission_idx]["status"] = "completed"
                if mission_idx in self.mission_start_times:
                    elapsed = time.time() - self.mission_start_times[mission_idx]
                    self.mission_elapsed_times[mission_idx] = elapsed
                    del self.mission_start_times[mission_idx]
                if self.current_mission_idx == mission_idx:
                    self.current_mission_idx = None  # Clear current mission

            self._refresh_missions_display()

    def _clear_mission_progress(self):
        """Clear active mission timers but preserve mission status/completion state.

        This is called when a mission starts/completes to clean up the active timer.
        Mission statuses (pending, in_progress, completed) are preserved.
        Elapsed times are preserved to show completion times.
        """
        # Only clear the active mission timer, not elapsed times
        self.mission_start_times.clear()
        self.current_mission_idx = None  # Clear current mission indicator

        self._refresh_missions_display()

    def _reset_all_missions_to_pending(self):
        """Reset all missions to pending status when a new client script loads.

        This is called when a new scene is loaded via load_scene() to reset
        the mission widget for the new client. Mission completion times are cleared
        and mission statuses are reset to pending.
        """
        # Reset all mission statuses to pending
        for mission in self.missions:
            mission["status"] = "pending"
        if hasattr(self, 'scene') and self.scene and self.scene.missions:
            for mission in self.scene.missions:
                mission["status"] = "pending"

        # Clear all timer and elapsed time data
        self.mission_start_times.clear()
        self.mission_elapsed_times.clear()
        self.current_mission_idx = None

        # Also reset the mission monitor if it exists
        if hasattr(self, 'mission_monitor') and self.mission_monitor is not None:
            self.mission_monitor.reset_all_missions()

        self._refresh_missions_display()

    def _reset_missions(self):
        """Reset all missions to pending status."""
        for mission in self.missions:
            mission["status"] = "pending"
        for mission in self.scene.missions:
            mission["status"] = "pending"
        # Clear all timer data
        self.mission_start_times.clear()
        self.mission_elapsed_times.clear()
        self.current_mission_idx = None  # Clear current mission
        self._refresh_missions_display()

    def _refresh_missions_display(self):
        """Refresh mission display with updated statuses.

        CRITICAL: This method MUST work even if mission_widgets is empty.
        Mission state is stored in self.missions and will be displayed
        when widgets are recreated.
        """
        # Update all missions data first (this persists the state)
        for idx, mission in enumerate(self.missions):
            # Ensure status field exists
            if "status" not in mission:
                mission["status"] = "pending"

        # If widgets don't exist yet, that's OK - they'll display the updated data when created
        if not self.mission_widgets:
            return

        for idx, (widget_info, mission) in enumerate(zip(self.mission_widgets, self.missions)):
            status = mission.get("status", "pending")
            # Determine status text and color
            if status == "pending":
                status_color = "PENDING"
                status_fg_color = "black"
            elif status == "completed":
                status_color = "COMPLETED"
                status_fg_color = "green"
            elif status == "failed":
                status_color = "FAILED"
                status_fg_color = "#d32f2f"  # Red color for failed
            else:  # in_progress
                status_color = "IN PROGRESS"
                status_fg_color = "black"

            try:
                widget_info['status_label'].config(text=status_color, foreground=status_fg_color)
            except tk.TclError:
                # Widget destroyed, skip
                pass

            # Highlight current mission with bold label and colored text
            if self.current_mission_idx == idx:
                # Highlight by making the label bold and blue
                widget_info['label'].config(font=("TkDefaultFont", 9, "bold"), foreground="#1976d2")
                widget_info['status_label'].config(foreground="#1976d2", font=("TkDefaultFont", 8, "bold"))
            else:
                # Reset to normal (but keep the status color)
                widget_info['label'].config(font=("TkDefaultFont", 9, "bold"), foreground="black")
                widget_info['status_label'].config(foreground=status_fg_color, font=("TkDefaultFont", 8))

            # Always keep button hidden/disabled in UI
            widget_info['button'].config(state=tk.DISABLED)

        # Note: Canvas redraw happens in _animate_step() frame updates, not here
        # This keeps mission display updates fast and prevents GUI freezing

    def _update_mission_timers(self):
        """Track elapsed time for in-progress missions (no display).

        This method updates mission_elapsed_times which are used for logging and
        mission completion records. Timer values are not displayed in widgets.
        """
        if not self.missions:
            return

        # Track elapsed time for in-progress missions (no display updates)
        current_time = time.time()
        for idx, mission in enumerate(self.missions):
            status = mission.get("status", "pending")
            if status == "in_progress" and idx in self.mission_start_times:
                # Track elapsed time in data for logging and mission completion records
                elapsed = current_time - self.mission_start_times[idx]
                self.mission_elapsed_times[idx] = elapsed

    def _initialize_mission_monitor(self, scene_path: str = None):
        """Initialize mission monitor and register event callbacks.

        Parameters
        ----------
        scene_path : str, optional
            Path to the scene file to monitor. If None, uses a default scene.
        """
        try:
            # Clean up old monitor if it exists
            self.mission_monitor = None

            # Create mission monitor with the correct scene path
            if scene_path:
                self.mission_monitor = MissionMonitor(scene_path=scene_path, verbose=False)
            else:
                self.mission_monitor = MissionMonitor(verbose=False)

            # Register event callbacks for autonomous mission solver
            self.mission_monitor.on_mission_started.append(self._on_mission_started)
            self.mission_monitor.on_mission_completed.append(self._on_mission_completed)
            # Register generic event handler for mission_failed events
            def handle_mission_event(event_type, mission_id, data):
                if event_type == "mission_failed":
                    self._on_mission_failed(
                        mission_id,
                        data,  # mission_info
                        data.get("elapsed_time", 0.0),
                        data.get("error_message", "")
                    )
            self.mission_monitor.on_mission_event.append(handle_mission_event)
        except Exception as e:
            print(f"Warning: Could not initialize mission monitor: {e}")
            import traceback
            traceback.print_exc()
            self.mission_monitor = None

    def _queue_mission_event(self, event_type: str, mission_id: str, data=None, source: str = "server"):
        """Queue mission events so they are applied when it is safe to update the widget."""
        if not mission_id:
            return

        event = {
            "type": event_type,
            "mission_id": mission_id,
            "data": data,
            "source": source,
        }
        with self.mission_event_lock:
            self.pending_mission_events.append(event)

    def _animation_active(self) -> bool:
        """Check if any animation is currently running."""
        now = time.time()
        return bool(
            self.is_animating
            or self.target_q is not None
            or self.animation_start_time is not None
            or now < self.active_animation_deadline
        )

    def _apply_mission_event(self, event: dict) -> bool:
        """Apply a queued mission event if it is safe, otherwise keep it queued."""
        mission_id = event.get("mission_id")
        event_type = event.get("type")
        data = event.get("data")

        # Find mission index
        mission_idx = None
        for idx, mission in enumerate(self.missions):
            if mission.get("id") == mission_id:
                mission_idx = idx
                break

        # Drop events for unknown missions
        if mission_idx is None:
            return True

        current_status = self.missions[mission_idx].get("status", "pending")

        # Ignore attempts to restart finished missions
        if event_type == "start" and current_status in ("completed", "failed"):
            return True

        # If mission already failed, keep it failed unless explicitly reset
        if current_status == "failed" and event_type != "fail":
            return True

        # Do not switch missions while another one is marked active
        if self.current_mission_idx is not None and self.current_mission_idx != mission_idx:
            return False

        # Wait until animations are finished before changing mission status
        # Allow failure events to post immediately so the widget reflects the failure
        if event_type != "fail" and self._animation_active():
            return False

        if event_type == "start":
            # Ignore if already in progress
            if current_status != "in_progress":
                self.missions[mission_idx]["status"] = "in_progress"
                if hasattr(self, 'scene') and self.scene.missions and mission_idx < len(self.scene.missions):
                    self.scene.missions[mission_idx]["status"] = "in_progress"
                self.mission_start_times[mission_idx] = time.time()
            self.current_mission_idx = mission_idx
            self._refresh_missions_display()
            return True

        if event_type in ("complete", "fail"):
            # Only handle completion/failure for the active mission
            if self.current_mission_idx is not None and self.current_mission_idx != mission_idx:
                return False

            # If mission already failed, do not overwrite failure with completion
            if current_status == "failed" and event_type == "complete":
                return True

            # Determine elapsed time and error message (if any)
            elapsed_time = None
            error_message = ""

            if isinstance(data, dict):
                elapsed_time = data.get("elapsed_time")
                error_message = data.get("error_message", "")
            elif isinstance(data, (int, float)):
                elapsed_time = data
            elif isinstance(data, str) and event_type == "fail":
                error_message = data

            if elapsed_time is None and mission_idx in self.mission_start_times:
                elapsed_time = time.time() - self.mission_start_times[mission_idx]

            # Update status
            new_status = "completed" if event_type == "complete" else "failed"
            if current_status != new_status:
                self.missions[mission_idx]["status"] = new_status
                if hasattr(self, 'scene') and self.scene.missions and mission_idx < len(self.scene.missions):
                    self.scene.missions[mission_idx]["status"] = new_status

            # Store timing and error info
            if elapsed_time is not None:
                self.mission_elapsed_times[mission_idx] = elapsed_time
            if event_type == "fail":
                self.missions[mission_idx]["error_message"] = error_message

            # Clean up timers and active marker
            if mission_idx in self.mission_start_times:
                del self.mission_start_times[mission_idx]
            if self.current_mission_idx == mission_idx:
                self.current_mission_idx = None

            self._refresh_missions_display()
            return True

        # Unknown event type - drop it
        return True

    def _process_pending_mission_events(self):
        """Process queued mission events, allowing failures to apply even if earlier events are gated."""
        while True:
            applied_any = False
            with self.mission_event_lock:
                events = list(self.pending_mission_events)
                self.pending_mission_events.clear()

            for event in events:
                if self._apply_mission_event(event):
                    applied_any = True
                else:
                    # Keep events that still cannot be applied
                    with self.mission_event_lock:
                        self.pending_mission_events.append(event)

            if not applied_any:
                return

    def _on_server_mission_command(self, event_type: str, mission_id: str, data):
        """Handle mission commands from server/client.

        Parameters
        ----------
        event_type : str
            Type of mission event: "start" or "complete"
        mission_id : str
            ID of the mission
        data : any
            Additional data (elapsed time for completion, etc.)
        """
        try:
            # Queue mission events so they are applied only when safe
            self._queue_mission_event(event_type, mission_id, data, source="server")
        except Exception as e:
            print(f"Error handling mission command: {e}")

    def _on_mission_started(self, mission_id, mission_info):
        """Handle mission started event from monitor."""
        try:
            # Defer processing to respect animation timing
            self._queue_mission_event("start", mission_id, mission_info, source="monitor")
        except Exception as e:
            print(f"Error handling mission_started event: {e}")
            import traceback
            traceback.print_exc()

    def _on_mission_completed(self, mission_id, mission_info, elapsed_time):
        """Handle mission completed event from monitor."""
        try:
            # Defer processing to respect animation timing
            self._queue_mission_event("complete", mission_id, {"elapsed_time": elapsed_time}, source="monitor")
        except Exception as e:
            print(f"Error handling mission_completed event: {e}")

    def _on_mission_failed(self, mission_id, mission_info, elapsed_time, error_message=""):
        """Handle mission failed event from monitor."""
        try:
            # Defer processing to respect animation timing
            self._queue_mission_event(
                "fail",
                mission_id,
                {"elapsed_time": elapsed_time, "error_message": error_message},
                source="monitor",
            )
        except Exception as e:
            print(f"Error handling mission_failed event: {e}")

    def _sync_mission_states_from_monitor(self):
        """Sync mission states from monitor to GUI (detect external client changes).

        This allows the GUI to detect when an external mission solver client
        updates mission statuses without triggering the event callbacks.
        Uses debouncing to avoid excessive file I/O and UI updates.
        """
        if not hasattr(self, 'mission_monitor') or self.mission_monitor is None:
            return

        # Debounce: Only sync every 100ms (10 FPS) to avoid excessive updates
        # while running at 40 FPS animation loop
        current_time = time.time()
        if current_time - self.last_sync_time < 0.1:
            return
        self.last_sync_time = current_time

        try:
            # First, load state from shared file (if it exists) to detect solver updates
            import tempfile
            import os
            state_file = os.path.join(tempfile.gettempdir(), "mission_state.json")
            self.mission_monitor.load_state_from_file(state_file)

            # Check each mission in the monitor's tracking
            for mission_id, monitor_state in self.mission_monitor.mission_states.items():
                monitor_status = monitor_state.get("status", "pending")
                monitor_elapsed = monitor_state.get("elapsed_time", 0.0)

                # Find corresponding mission in GUI's mission list
                for idx, gui_mission in enumerate(self.missions):
                    if gui_mission.get("id") == mission_id:
                        gui_status = gui_mission.get("status", "pending")

                        # If status changed, update GUI and sync timing
                        if gui_status != monitor_status:
                            if monitor_status == "in_progress":
                                self._queue_mission_event("start", mission_id, None, source="monitor_sync")
                            elif monitor_status == "completed":
                                self._queue_mission_event(
                                    "complete",
                                    mission_id,
                                    {"elapsed_time": monitor_elapsed},
                                    source="monitor_sync",
                                )
                            elif monitor_status == "failed":
                                self._queue_mission_event(
                                    "fail",
                                    mission_id,
                                    {"elapsed_time": monitor_elapsed},
                                    source="monitor_sync",
                                )
                        break
        except Exception as e:
            # Silently ignore sync errors to prevent breaking animation loop
            pass

    def _setup_server_tab(self):
        """Setup server configuration and control tab."""
        # Create scrollable frame
        canvas = tk.Canvas(self.server_tab, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.server_tab, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        server_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=server_frame, anchor=tk.NW)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make frame expand to fill canvas width and height
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            frame_height = max(event.height, canvas_height)
            canvas.itemconfig(window_id, width=canvas_width, height=frame_height)

        def on_canvas_configure(event):
            canvas_width = event.width
            canvas_height = event.height
            # Get current frame height from scrollregion
            bbox = canvas.bbox("all")
            if bbox:
                frame_height = max(bbox[3] - bbox[1], canvas_height)
            else:
                frame_height = canvas_height
            canvas.itemconfig(window_id, width=canvas_width, height=frame_height)

        server_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        # Server Status
        status_frame = ttk.LabelFrame(server_frame, text="Server Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(status_frame, text="Status:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
        self.server_status_text = tk.Text(status_frame, height=2, width=30, state=tk.DISABLED)
        self.server_status_text.pack(fill=tk.X, pady=5)

        # Server Controls
        control_frame = ttk.LabelFrame(server_frame, text="Server Control", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.start_server_btn = ttk.Button(button_frame, text="Start Server", command=self._start_server)
        self.start_server_btn.pack(side=tk.LEFT, padx=2)

        self.stop_server_btn = ttk.Button(button_frame, text="Stop Server", command=self._stop_server, state=tk.DISABLED)
        self.stop_server_btn.pack(side=tk.LEFT, padx=2)

        # Server Configuration
        config_frame = ttk.LabelFrame(server_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Host
        host_frame = ttk.Frame(config_frame)
        host_frame.pack(fill=tk.X, pady=5)
        ttk.Label(host_frame, text="Host:", width=10).pack(side=tk.LEFT)
        self.server_host_text = ttk.Entry(host_frame, width=20)
        self.server_host_text.insert(0, self.server_host)
        self.server_host_text.pack(side=tk.LEFT, padx=5)

        # Port
        port_frame = ttk.Frame(config_frame)
        port_frame.pack(fill=tk.X, pady=5)
        ttk.Label(port_frame, text="Port:", width=10).pack(side=tk.LEFT)
        self.server_port_text = ttk.Entry(port_frame, width=20)
        self.server_port_text.insert(0, str(self.server_port))
        self.server_port_text.pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(config_frame, text="Apply Configuration", command=self._apply_server_config).pack(fill=tk.X, pady=5)

        # Add vertical spacer to fill remaining space
        spacer = ttk.Frame(server_frame)
        spacer.pack(fill=tk.BOTH, expand=True)

    def _apply_server_config(self):
        """Apply server configuration changes."""
        try:
            new_host = self.server_host_text.get()
            new_port = int(self.server_port_text.get())

            if self.server_running:
                messagebox.showwarning("Warning", "Please stop the server before changing configuration")
                return

            self.server_host = new_host
            self.server_port = new_port
            messagebox.showinfo("Success", "Configuration updated successfully")
        except ValueError:
            messagebox.showerror("Error", "Invalid port number")

    def _setup_info_tab(self):
        """Setup information display tab."""
        # Create scrollable frame
        canvas = tk.Canvas(self.info_tab, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.info_tab, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=info_frame, anchor=tk.NW)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make frame expand to fill canvas width and height
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            frame_height = max(event.height, canvas_height)
            canvas.itemconfig(window_id, width=canvas_width, height=frame_height)

        def on_canvas_configure(event):
            canvas_width = event.width
            canvas_height = event.height
            # Get current frame height from scrollregion
            bbox = canvas.bbox("all")
            if bbox:
                frame_height = max(bbox[3] - bbox[1], canvas_height)
            else:
                frame_height = canvas_height
            canvas.itemconfig(window_id, width=canvas_width, height=frame_height)

        info_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        # Robot Information
        robot_frame = ttk.LabelFrame(info_frame, text="Robot Configuration", padding=10)
        robot_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(robot_frame, text=f"Link 1 Length: {self.robot.L1:.3f} m").pack(anchor=tk.W)
        ttk.Label(robot_frame, text=f"Link 2 Length: {self.robot.L2:.3f} m").pack(anchor=tk.W)
        ttk.Label(robot_frame, text=f"Total Reach: {self.robot.L1 + self.robot.L2:.3f} m").pack(anchor=tk.W)

        # Scene Information
        scene_frame = ttk.LabelFrame(info_frame, text="Scene Information", padding=10)
        scene_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(scene_frame, text=f"Scene File: {self.current_scene_path}").pack(anchor=tk.W)
        ttk.Label(scene_frame, text=f"Targets: {len(self.scene.targets)}").pack(anchor=tk.W)
        ttk.Label(scene_frame, text=f"Obstacles: {len(self.scene.obstacles)}").pack(anchor=tk.W)
        ttk.Label(scene_frame, text=f"Missions: {len(self.missions)}").pack(anchor=tk.W)

        # Mission List
        if self.missions:
            missions_frame = ttk.LabelFrame(info_frame, text="Missions", padding=10)
            missions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            missions_text = tk.Text(missions_frame, height=8, width=35, state=tk.NORMAL)
            missions_text.pack(fill=tk.BOTH, expand=True)

            for mission in self.missions:
                mission_str = "{}: {} -> {} [{}]".format(
                    mission.get('id', 'Unknown'),
                    mission.get('pickup_target', 'Unknown'),
                    mission.get('delivery_target', 'Unknown'),
                    mission.get('status', 'unknown').upper()
                )
                missions_text.insert(tk.END, mission_str + "\n")

            missions_text.config(state=tk.DISABLED)

        # Add vertical spacer to fill remaining space
        spacer = ttk.Frame(info_frame)
        spacer.pack(fill=tk.BOTH, expand=True)

    def _show_server_config_dialog(self):
        """Show server configuration dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Server Configuration")
        dialog.geometry("400x250")
        dialog.resizable(False, False)

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Configuration frame
        config_frame = ttk.LabelFrame(dialog, text="Server Configuration", padding=15)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Host
        host_frame = ttk.Frame(config_frame)
        host_frame.pack(fill=tk.X, pady=8)
        ttk.Label(host_frame, text="Host:", width=15).pack(side=tk.LEFT)
        host_entry = ttk.Entry(host_frame, width=20)
        host_entry.insert(0, getattr(self, 'server_host_entry', ttk.Entry(host_frame)).get() or "localhost")
        host_entry.pack(side=tk.LEFT, padx=5)

        # Port
        port_frame = ttk.Frame(config_frame)
        port_frame.pack(fill=tk.X, pady=8)
        ttk.Label(port_frame, text="Port:", width=15).pack(side=tk.LEFT)
        port_entry = ttk.Entry(port_frame, width=20)
        port_entry.insert(0, getattr(self, 'server_port_entry', ttk.Entry(port_frame)).get() or "8008")
        port_entry.pack(side=tk.LEFT, padx=5)

        # Animation FPS
        fps_frame = ttk.Frame(config_frame)
        fps_frame.pack(fill=tk.X, pady=8)
        ttk.Label(fps_frame, text="Animation FPS:", width=15).pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=getattr(self, 'animation_fps', 30))
        fps_scale = ttk.Scale(
            fps_frame,
            from_=24,
            to=60,
            orient=tk.HORIZONTAL,
            variable=self.fps_var,
            command=lambda v: self._on_fps_change()
        )
        fps_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.fps_label = ttk.Label(fps_frame, text=str(self.fps_var.get()), width=3)
        self.fps_label.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, pady=15)

        def save_config():
            self.server_host_entry = type('obj', (object,), {'get': lambda: host_entry.get()})()
            self.server_port_entry = type('obj', (object,), {'get': lambda: port_entry.get()})()
            self.animation_fps = self.fps_var.get()
            messagebox.showinfo("Configuration", "Server configuration saved!")
            dialog.destroy()

        ttk.Button(button_frame, text="Save", command=save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _draw_obstacles(self):
        """Draw obstacles on plot."""
        for obs in self.scene.obstacles:
            if obs.get("type") == "polygon":
                points = obs.get("points", [])
                if len(points) >= 3:
                    # Draw filled polygon with gray color
                    poly = MPLPolygon(
                        points,
                        closed=True,
                        fill=True,
                        facecolor="gray",          # Gray fill
                        edgecolor="darkgray",      # Dark gray outline
                        alpha=0.6,                 # Medium opacity
                        linewidth=2,               # Border
                        zorder=4,                  # Below robot but visible
                    )
                    self.ax.add_patch(poly)

    def _draw_targets(self):
        """Draw targets on plot - labels only for current mission, gray dots for all other targets."""
        self._draw_targets_impl()

    def _redraw_targets_only(self):
        """Efficiently redraw only target markers when mission changes (without clearing whole plot)."""
        # Remove all existing target-related artists from the axes
        # We need to track target artists to remove only those
        if not hasattr(self, '_target_artists'):
            self._target_artists = []

        # Remove old target artists with error handling
        # FIX: Wrap removal in try-except to handle NotImplementedError from matplotlib rendering
        for artist in list(self._target_artists):
            try:
                artist.remove()
            except (RuntimeError, NotImplementedError):
                # Artist may be in the middle of rendering, skip it
                pass
        self._target_artists.clear()

        # Redraw targets
        self._draw_targets_impl()

        # Trigger canvas redraw
        try:
            self.canvas.draw_idle()
        except Exception as e:
            print(f"[WARNING] Error redrawing targets: {e}")

    def _draw_targets_impl(self):
        """Implementation of target drawing - used by both _draw_targets and _redraw_targets_only."""
        # Get current mission targets if mission is active
        current_mission_targets = set()

        if self.current_mission_idx is not None and self.current_mission_idx < len(self.missions):
            mission = self.missions[self.current_mission_idx]

            # Add pickup, delivery, and stopover targets from current mission
            pickup = mission.get("pickup_target")
            if pickup:
                current_mission_targets.add(pickup)

            delivery = mission.get("delivery_target")
            if delivery:
                current_mission_targets.add(delivery)

            stopovers = mission.get("stopover_points", [])
            for stopover in stopovers:
                current_mission_targets.add(stopover)

        # Draw targets with different styling based on current mission
        for idx, target in enumerate(self.scene.targets):
            x = target.get("x", 0)
            y = target.get("y", 0)
            label = target.get("label", None)

            # Check if this target is in the current mission
            is_current_mission_target = label in current_mission_targets if label else False

            # Only draw labels for current mission targets
            if is_current_mission_target and self.current_mission_idx is not None:
                mission = self.missions[self.current_mission_idx]
                pickup = mission.get("pickup_target")
                delivery = mission.get("delivery_target")
                stopovers = mission.get("stopover_points", [])

                # Determine role and color
                if label == pickup:
                    color = "blue"
                    marker = "P"
                    display_label = f"{label}\n(Pickup)"
                elif label == delivery:
                    color = "red"
                    marker = "X"
                    display_label = f"{label}\n(Delivery)"
                elif label in stopovers:
                    color = "green"
                    marker = "*"
                    display_label = f"{label}\n(Stopover)"
                else:
                    color = "blue"
                    marker = "P"
                    display_label = label

                # Draw colored marker with label
                line, = self.ax.plot(x, y, marker=marker, markersize=14,
                            color=color, alpha=0.9, zorder=6,
                            markeredgecolor='white', markeredgewidth=1.5)
                if hasattr(self, '_target_artists'):
                    self._target_artists.append(line)

                # Draw label with matching color
                text = self.ax.text(
                    x + 0.04,
                    y + 0.04,
                    display_label,
                    fontsize=8,
                    color=color,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             edgecolor=color, alpha=0.85, linewidth=1.2),
                    zorder=7,
                )
                if hasattr(self, '_target_artists'):
                    self._target_artists.append(text)
            else:
                # Draw gray dot without label for all other targets
                line, = self.ax.plot(x, y, marker="o", markersize=10,
                            color="gray", alpha=0.5, zorder=5,
                            markeredgecolor='darkgray', markeredgewidth=1)
                if hasattr(self, '_target_artists'):
                    self._target_artists.append(line)

    def _update_visualization(self):
        """Update robot visualization."""
        frame_start = time.time()

        positions = self.robot.get_joint_positions(self.current_q)
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

        # Update trajectory line with collision-aware coloring only every N frames to avoid expensive IK/collision checks
        # Each trajectory segment does IK + collision check, so this is very expensive
        if len(self.trajectory_history) > 1 and (self.frame_counter % self.trajectory_update_frequency == 0):
            self._update_trajectory_visualization()

        # Update robot visualization
        self.link_line.set_data(xs, ys)
        self.ee_marker.set_data([xs[-1]], [ys[-1]])

        self.canvas.draw_idle()

        # Track frame timing for diagnostics
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)

    def _update_trajectory_visualization(self):
        """Update trajectory line with collision-aware coloring (orange=safe, red=collision).

        Optimized to only recompute collision colors for new trajectory points, avoiding
        expensive IK + collision checks on the entire history every frame.
        """
        if len(self.trajectory_history) < 2:
            return

        traj_array = np.array(list(self.trajectory_history))

        # Create line segments and colors for collision detection
        segments = []
        colors = []

        # Color red if the segment midpoint is inside any obstacle, else orange
        for i in range(len(traj_array) - 1):
            p1 = traj_array[i]
            p2 = traj_array[i + 1]
            mid = (p1 + p2) / 2.0
            in_collision = False
            for obs in self.scene.obstacles:
                if obs.get("type") != "polygon":
                    continue
                pts = obs.get("points", [])
                if len(pts) < 3:
                    continue
                from matplotlib.path import Path
                if Path(pts).contains_point(mid):
                    in_collision = True
                    break
            segments.append([p1, p2])
            colors.append("red" if in_collision else "orange")

        # Update LineCollection with segments and colors
        self.trajectory_line_collection.set_segments(segments)
        self.trajectory_line_collection.set_colors(colors)

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

    def _on_fps_change(self):
        """Handle animation FPS change."""
        fps = self.fps_var.get()
        self.animation_fps = max(24, int(fps))
        self.fps_label.config(text=str(self.animation_fps))

    def _update_server_ui(self):
        """Update server UI buttons based on server_running state."""
        if self.server_running:
            self.start_server_btn.config(state=tk.DISABLED)
            self.stop_server_btn.config(state=tk.NORMAL)
        else:
            self.start_server_btn.config(state=tk.NORMAL)
            self.stop_server_btn.config(state=tk.DISABLED)

    def _update_server_status_display(self):
        """Update server status text display."""
        # Enable the text widget to update it
        self.server_status_text.config(state=tk.NORMAL)
        self.server_status_text.delete("1.0", tk.END)

        if self.server_running:
            status = f"Running on {self.server_host}:{self.server_port}"
        else:
            status = "Stopped"

        self.server_status_text.insert("1.0", status)
        self.server_status_text.config(state=tk.DISABLED)

    def _update_state_display(self):
        """Update state info display."""
        # Only update if state_labels exist (for backward compatibility)
        if not hasattr(self, 'state_labels'):
            return

        xy = self.robot.fk_xy(self.current_q)
        collision = self.collision_checker.check_configuration(self.current_q)

        self.state_labels["q"].config(
            text=f"[{', '.join([f'{x:.3f}' for x in self.current_q[:2]])}]"
        )
        self.state_labels["xy"].config(text=f"[{xy[0]:.3f}, {xy[1]:.3f}]")
        self.state_labels["collision"].config(
            text="YES" if collision else "NO",
            foreground="red" if collision else "green",
        )

    def _on_joint_slider_change(self, joint_idx):
        """Handle joint slider change with smooth animation."""
        # Skip if we're internally updating sliders (prevent callback loop)
        if self.updating_sliders:
            return

        if not self.is_animating:
            q = np.array([slider.get() for slider in self.joint_sliders])
            # Set target for smooth animation
            self.target_q = q
            self.animation_start_time = None  # Reset animation timer
            self.active_animation_deadline = max(
                self.active_animation_deadline,
                time.time() + self.current_animation_duration
            )

    def _move_to_cartesian(self):
        """Move robot to cartesian position with smooth animation."""
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            elbow = self.elbow_var.get()

            q = self.robot.ik_xy(np.array([x, y]), elbow=elbow)

            if q is None:
                messagebox.showerror("Error", "Target position is unreachable!")
                return

            if self.collision_checker.check_configuration(q):
                messagebox.showerror("Error", "Target configuration collides with obstacles!")
                return

            # Set target for smooth animation
            self.target_q = q
            self.animation_start_time = None
            self.status_bar.config(text=f"Moving to ({x:.3f}, {y:.3f})...")
            self.active_animation_deadline = max(
                self.active_animation_deadline,
                time.time() + self.current_animation_duration
            )

        except ValueError:
            messagebox.showerror("Error", "Invalid coordinates. Please enter numbers.")

    def set_animation_duration(self, duration: float):
        """
        Set the animation duration for the next movement.

        Called by the server to synchronize GUI animation with server-side animation timing.

        Parameters
        ----------
        duration : float
            Animation duration in seconds (e.g., 15.0 for a 15-second movement)
        """
        self.current_animation_duration = max(0.01, duration)  # Minimum 10ms to avoid division issues
        self.active_animation_deadline = max(
            self.active_animation_deadline,
            time.time() + self.current_animation_duration
        )

    def _reset_robot(self):
        """Reset robot to home position."""
        self.current_q = np.array([0.0, 0.0, 0.0, 0.0])

        # Clear trajectory history on reset
        self.trajectory_history.clear()
        self.start_position = None
        self.goal_position = None

        for i, slider in enumerate(self.joint_sliders):
            slider.set(0.0)

        self._update_visualization()
        self._update_state_display()
        self.status_bar.config(text="Reset to home position")

    def _plan_trajectory(self):
        """Plan trajectory using selected planner."""
        try:
            # Get goal
            goal_x = float(self.goal_x_entry.get())
            goal_y = float(self.goal_y_entry.get())

            # Compute IK for goal
            goal_q = self.robot.ik_xy(np.array([goal_x, goal_y]), elbow="up")
            if goal_q is None:
                goal_q = self.robot.ik_xy(np.array([goal_x, goal_y]), elbow="down")

            if goal_q is None:
                messagebox.showerror("Error", "Goal position is unreachable!")
                return

            # Plan
            start_q = self.current_q.copy()
            planner = self.planner_var.get()

            self.status_bar.config(text=f"Planning with {planner}...")
            self.root.update()

            if planner == "rrt":
                seed = int(self.seed_entry.get())
                max_nodes = int(self.max_nodes_entry.get())
                cfg = {"seed": seed, "max_nodes": max_nodes}
                result = rrt.plan(self.robot, self.scene, start_q, goal_q, cfg=cfg, seed=seed)
            else:
                result = straight.plan(self.robot, self.scene, start_q, goal_q)

            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            if result["success"]:
                # Check collision along the trajectory
                collision_info = check_trajectory_collision(result["waypoints"], self.collision_checker)

                results_str = f"""Status: SUCCESS
Planning Time: {result['planning_time']:.3f}s
Path Length: {result['path_length']:.3f}
Clearance: {result['clearance']:.4f}
Nodes Explored: {result['nodes_explored']}
Waypoints: {len(result['waypoints'])}
Collision Waypoints: {collision_info['collision_count']} / {len(result['waypoints'])}
"""
                self.results_text.insert(tk.END, results_str)

                # Add collision segments info if any
                if collision_info['collision_segments']:
                    self.results_text.insert(tk.END, "\nCollision Segments:\n")
                    for start_idx, end_idx in collision_info['collision_segments']:
                        self.results_text.insert(tk.END, f"  Waypoints {start_idx}-{end_idx}\n")

                self.trajectory = result["waypoints"]
                self.trajectory_collision_info = collision_info
                self.trajectory_index = 0
                self.is_animating = True

                self.status_bar.config(text="Planning succeeded! Animating...")
            else:
                self.results_text.insert(tk.END, "Status: FAILED\nNo path found.")
                self.status_bar.config(text="Planning failed")

            self.results_text.config(state=tk.DISABLED)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")

    def _test_collision_visualization(self):
        """Test collision visualization with a trajectory containing known collisions."""
        print("\n[TEST] Creating test trajectory with known collisions...")

        # Create a test trajectory with known collision points
        test_waypoints = [
            np.array([0.0, 0.0, 0.0, 0.0]),     # Safe
            np.array([0.2, 0.5, 0.0, 0.0]),     # COLLISION
            np.array([0.4, 0.0, 0.0, 0.0]),     # COLLISION
            np.array([-0.3, 1.8, 0.0, 0.0]),    # COLLISION
            np.array([0.5, 0.0, 0.0, 0.0]),     # Safe
        ]

        # Check collisions
        collision_info = check_trajectory_collision(test_waypoints, self.collision_checker)

        print(f"[TEST] Created trajectory with {len(test_waypoints)} waypoints")
        print(f"[TEST] Detected {collision_info['collision_count']} collision waypoints")

        # Update results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        results_str = f"""TEST TRAJECTORY - Collision Visualization
Waypoints: {len(test_waypoints)}
Collision Waypoints: {collision_info['collision_count']} / {len(test_waypoints)}
"""
        self.results_text.insert(tk.END, results_str)

        if collision_info['collision_segments']:
            self.results_text.insert(tk.END, "\nCollision Segments:\n")
            for start_idx, end_idx in collision_info['collision_segments']:
                self.results_text.insert(tk.END, f"  Waypoints {start_idx}-{end_idx}\n")

        self.results_text.config(state=tk.DISABLED)

        # Store trajectory for animation
        self.trajectory = test_waypoints
        self.trajectory_collision_info = collision_info
        self.trajectory_index = 0
        self.is_animating = True

        print(f"[TEST] Collision visualization test complete")
        self.status_bar.config(text="Test trajectory displayed - red trajectory line shows collision regions")

    def _stop_animation(self):
        """Stop trajectory animation."""
        self.is_animating = False
        self.trajectory = None
        self.status_bar.config(text="Animation stopped")

    def _visualize_collision_points(self, collision_info):
        """Visualize collision waypoints on the trajectory plot - HIGHLY VISIBLE."""
        # Get waypoints and collision status
        waypoints = collision_info["waypoints"]
        collision_status = collision_info["collision_status"]

        # Extract collision waypoints in cartesian space
        collision_xs = []
        collision_ys = []

        for i, (q, is_collision) in enumerate(zip(waypoints, collision_status)):
            if is_collision:
                # Ensure q is a numpy array
                if not isinstance(q, np.ndarray):
                    q = np.array(q)
                xy = self.robot.fk_xy(q)
                collision_xs.append(float(xy[0]))
                collision_ys.append(float(xy[1]))

        # Update collision markers using scatter plot (MUCH more visible)
        if collision_xs:
            self.collision_markers.set_offsets(np.c_[collision_xs, collision_ys])
            print(f"[DISPLAY] Showing {len(collision_xs)} collision markers at:")
            for x, y in zip(collision_xs, collision_ys):
                print(f"  [{x:.4f}, {y:.4f}]")
        else:
            self.collision_markers.set_offsets(np.empty((0, 2)))
            print(f"[DISPLAY] No collision markers to show")

        # Redraw canvas
        self.canvas.draw_idle()

    def _animate_step(self):
        """Animate one step with smooth interpolation."""
        step_start = time.time()
        current_time = step_start

        # Handle trajectory playback animation
        if self.is_animating and self.trajectory:
            if self.trajectory_index < len(self.trajectory):
                # Use smooth interpolation between waypoints
                idx = self.trajectory_index
                current_wp = self.trajectory[idx]

                # Interpolate to next waypoint if available
                if idx + 1 < len(self.trajectory):
                    next_wp = self.trajectory[idx + 1]

                    # Time-based interpolation (0.1 seconds per waypoint)
                    if self.animation_start_time is None:
                        self.animation_start_time = current_time

                    elapsed = current_time - self.animation_start_time
                    progress = min(1.0, elapsed / 0.1)  # 0-1 progress

                    # Linear interpolation
                    self.current_q = current_wp + progress * (next_wp - current_wp)

                    if progress >= 1.0:
                        self.trajectory_index += 1
                        self.animation_start_time = None
                else:
                    self.current_q = current_wp
                    self.trajectory_index += 1

                # Update sliders
                for i, slider in enumerate(self.joint_sliders):
                    slider.set(self.current_q[i])

                self._update_visualization()
                self._update_state_display()

                if self.trajectory_index >= len(self.trajectory):
                    self.is_animating = False
                    self.status_bar.config(text="Animation complete")

        # Handle smooth manual movement animation
        elif self.target_q is not None:
            if self.animation_start_time is None:
                self.animation_start_time = current_time
                self.last_q = self.current_q.copy()

            elapsed = current_time - self.animation_start_time
            # Use current_animation_duration if set, otherwise fall back to animation_duration
            duration = self.current_animation_duration if self.current_animation_duration > 0 else self.animation_duration
            progress = min(1.0, elapsed / duration)

            # Smooth interpolation using cosine easing
            ease = (1.0 - np.cos(np.pi * progress)) / 2.0

            # Interpolate smoothly
            self.current_q = self.last_q + ease * (self.target_q - self.last_q)

            # Update sliders quietly (without triggering callbacks)
            self.updating_sliders = True
            try:
                for i, slider in enumerate(self.joint_sliders):
                    slider.set(self.current_q[i])
            finally:
                self.updating_sliders = False

            self._update_visualization()
            self._update_state_display()

            # Check if reached target
            if progress >= 1.0:
                self.current_q = self.target_q.copy()
                self.target_q = None
                self.animation_start_time = None
                self.status_bar.config(text="Motion complete")
        else:
            # When idle, only redraw if enough time has passed (debounce to reduce CPU)
            # This prevents constant redraws when nothing is moving
            current_time = time.time()
            time_since_last_draw = current_time - self.last_visualization_time

            if time_since_last_draw >= self.min_redraw_interval:
                self._update_visualization()
                self.last_visualization_time = current_time

        # Increment frame counter for throttling expensive operations
        self.frame_counter += 1

        # Sync mission states from monitor (detect external client changes)
        self._sync_mission_states_from_monitor()
        # Apply queued mission events only when animations are finished
        self._process_pending_mission_events()

        # Update mission timers (called every frame)
        self._update_mission_timers()

        # Track total frame timing for diagnostics
        if self.profile_frame and self.frame_counter % 30 == 0:
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(list(self.frame_times))
                max_frame_time = np.max(list(self.frame_times))
                print(f"[GUI Frame] Avg: {avg_frame_time*1000:.1f}ms, Max: {max_frame_time*1000:.1f}ms (sample of {len(self.frame_times)})")

    def _schedule_update(self):
        """Schedule periodic updates with smooth animation."""
        self._animate_step()
        # Calculate interval from FPS (at least 24 FPS for smooth animation)
        interval = max(1, int(1000.0 / max(24, self.animation_fps)))
        self.root.after(interval, self._schedule_update)

    def _load_scene(self):
        """Load scene from file."""
        filename = filedialog.askopenfilename(
            title="Load Scene", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.scene = load_scene(filename)
                # Update current scene path
                self.current_scene_path = filename
                # Reload missions from scene and reset all to pending
                self.missions = self.scene.missions.copy() if self.scene.missions else []
                for mission in self.missions:
                    mission["status"] = "pending"
                for mission in self.scene.missions:
                    mission["status"] = "pending"

                # Clear all mission progress
                self.mission_start_times.clear()
                self.mission_elapsed_times.clear()
                self.current_mission_idx = None

                # Delete mission state file to start fresh
                try:
                    import tempfile
                    import os
                    state_file = os.path.join(tempfile.gettempdir(), "mission_state.json")
                    if os.path.exists(state_file):
                        os.remove(state_file)
                except Exception:
                    pass  # Ignore errors deleting state file

                # Recreate robot
                cfg = self.scene.robot_config
                self.robot = ScaraRobot(
                    L1=cfg["L1"], L2=cfg["L2"], joint_limits=cfg["joint_limits"]
                )
                self.collision_checker = CollisionChecker(self.robot, self.scene)

                # Clear trajectory history and reset trajectory state
                self.trajectory_history.clear()
                self.start_position = None
                self.goal_position = None
                self.trajectory = None
                self.trajectory_index = 0
                self.is_animating = False

                # Redraw
                self.ax.clear()
                self.ax.set_aspect("equal")
                self.ax.grid(True, alpha=0.3)
                self.ax.set_xlabel("X (m)")
                self.ax.set_ylabel("Y (m)")
                self.ax.set_title("SCARA Robot Workspace")
                reach = self.robot.L1 + self.robot.L2
                margin = 0.1
                self.ax.set_xlim(-reach - margin, reach + margin)
                self.ax.set_ylim(-reach - margin, reach + margin)

                self._draw_obstacles()
                self._draw_targets()

                (self.link_line,) = self.ax.plot(
                    [], [], "o-", linewidth=4, markersize=8, color="royalblue", zorder=10
                )
                (self.ee_marker,) = self.ax.plot(
                    [], [], "o", markersize=12, color="red", zorder=11
                )

                # Recreate trajectory line collection (was cleared by ax.clear())
                self.trajectory_line_collection = LineCollection(
                    [], linewidths=2, zorder=9
                )
                self.ax.add_collection(self.trajectory_line_collection)

                # Recreate trajectory line for backward compatibility
                (self.trajectory_line,) = self.ax.plot(
                    [], [], "-", linewidth=0, color="orange", alpha=0, zorder=9
                )

                # Recreate start/goal markers (were removed by ax.clear())
                self.start_marker = Circle((0, 0), 0.03, color="lime", alpha=0, zorder=5)
                self.ax.add_patch(self.start_marker)
                # FIX: Use facecolor instead of color to avoid matplotlib warning
                self.goal_marker = Circle((0, 0), 0.035, facecolor="dodgerblue", alpha=0,
                                         edgecolor="darkblue", linewidth=2, zorder=5)
                self.ax.add_patch(self.goal_marker)

                self._update_visualization()

                # Update server scene if running
                if self.server:
                    self.server.scene = self.scene
                    self.server.scene_path = filename
                    # Also update collision checker with new scene
                    self.server.collision_checker = CollisionChecker(self.server.robot, self.scene)

                # Update scene label if it exists
                if self.scene_label:
                    scene_name = filename.split("/")[-1]
                    self.scene_label.config(text=f"Scene: {scene_name}")

                self.status_bar.config(text=f"Loaded scene: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scene: {e}")

    def _on_client_scene_loaded(self, new_scene, scene_path):
        """Handle scene loaded event from client.

        Updates the GUI when a client script loads a new scene via the load_scene command.

        Parameters
        ----------
        new_scene : Scene
            The newly loaded scene object
        scene_path : str
            Path to the loaded scene file
        """
        # Update internal scene reference and path
        self.scene = new_scene
        self.current_scene_path = scene_path

        # Check if this is a NEW scene (different from the last loaded one)
        scene_changed = (self.last_loaded_scene_path != scene_path)
        old_scene_path = self.last_loaded_scene_path
        self.last_loaded_scene_path = scene_path

        # Update missions from new scene
        self.missions = self.scene.missions.copy() if self.scene.missions else []

        # CRITICAL: Initialize MissionMonitor with the CORRECT scene path
        # This must be done AFTER missions are loaded so callbacks listen to the right scene
        self._initialize_mission_monitor(scene_path)

        # Only reset mission widget when a DIFFERENT scene loads
        # If the same scene reconnects, preserve mission status
        self.mission_widgets.clear()
        if scene_changed:
            print(f"[Mission] Scene changed from {old_scene_path} to {scene_path} - resetting mission status")
            self._reset_all_missions_to_pending()
        else:
            # Same scene reconnected - just reset timers/current mission, preserve status and elapsed times
            print(f"[Mission] Same scene reconnected ({scene_path}) - preserving mission status")
            self.mission_start_times.clear()
            self.current_mission_idx = None
            self._refresh_missions_display()

        # Recreate robot from scene config
        cfg = self.scene.robot_config
        self.robot = ScaraRobot(
            L1=cfg["L1"], L2=cfg["L2"], joint_limits=cfg["joint_limits"]
        )
        self.collision_checker = CollisionChecker(self.robot, self.scene)

        # Get the missions tab (control_panel_frame) and clear it to rebuild missions section
        for widget in self.control_panel_frame.winfo_children():
            widget.destroy()

        # Recreate the missions section with updated missions
        self._setup_missions_section()

        # Update scene label to display actual scene filename (handle both Windows and Unix paths)
        if scene_path:
            from pathlib import Path
            scene_name = Path(scene_path).name
        else:
            scene_name = "Unknown"
        if self.scene_label:
            self.scene_label.config(text=f"Scene: {scene_name}")

        # Redraw visualization
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("SCARA Robot Workspace")
        reach = self.robot.L1 + self.robot.L2
        margin = 0.1
        self.ax.set_xlim(-reach - margin, reach + margin)
        self.ax.set_ylim(-reach - margin, reach + margin)

        self._draw_obstacles()
        self._draw_targets()

        (self.link_line,) = self.ax.plot(
            [], [], "o-", linewidth=4, markersize=8, color="royalblue", zorder=10
        )
        (self.ee_marker,) = self.ax.plot(
            [], [], "o", markersize=12, color="red", zorder=11
        )

        self.trajectory_line_collection = LineCollection(
            [], linewidths=2, zorder=9
        )
        self.ax.add_collection(self.trajectory_line_collection)

        (self.trajectory_line,) = self.ax.plot(
            [], [], "-", linewidth=0, color="orange", alpha=0, zorder=9
        )

        self.start_marker = Circle((0, 0), 0.03, color="lime", alpha=0, zorder=5)
        self.ax.add_patch(self.start_marker)
        # FIX: Use facecolor instead of color to avoid matplotlib warning
        self.goal_marker = Circle((0, 0), 0.035, facecolor="dodgerblue", alpha=0,
                                 edgecolor="darkblue", linewidth=2, zorder=5)
        self.ax.add_patch(self.goal_marker)

        self._update_visualization()

    def _save_scene(self):
        """Save scene to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Scene",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            try:
                dump_scene(self.scene, filename)
                self.status_bar.config(text=f"Saved scene: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scene: {e}")

    def _reset_view(self):
        """Reset visualization view."""
        reach = self.robot.L1 + self.robot.L2
        margin = 0.1
        self.ax.set_xlim(-reach - margin, reach + margin)
        self.ax.set_ylim(-reach - margin, reach + margin)
        self.canvas.draw()

    def _zoom_to_fit(self):
        """Zoom to fit all elements."""
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About SCARA Simulator",
            "SCARA Robot Simulator\nVersion 0.1.0\n\n"
            "A production-ready simulator for path planning\n"
            "and robot control.\n\n"
            "Built with Python, NumPy, and Matplotlib",
        )

    def _auto_start_server(self):
        """Automatically start the server on GUI startup."""
        if not self.server_running:
            try:
                self._start_server()
            except Exception as e:
                print(f"Auto-start server failed: {e}")

    def _start_server(self):
        """Start the simulator server."""
        try:
            host = self.server_host
            port = self.server_port

            # Check if port is already in use
            if is_port_in_use(host, port):
                process_info = get_process_using_port(port)
                name = process_info.get('name', 'Unknown')
                pid = process_info.get('pid', 'Unknown')

                response = messagebox.askyesno(
                    "Port In Use",
                    f"Port {port} is already in use by {name} (PID: {pid}).\n\n"
                    f"Would you like to kill this process and start the simulator?",
                )

                if response:
                    print(f"Attempting to kill process using port {port}...")
                    if kill_process_on_port(port):
                        time.sleep(1)  # Wait for port to be released
                        if is_port_in_use(host, port):
                            messagebox.showerror(
                                "Port Still In Use",
                                f"Port {port} is still in use. Please close the process manually.",
                            )
                            return
                        else:
                            print("Port is now available, starting server...")
                    else:
                        messagebox.showerror(
                            "Failed to Kill Process",
                            f"Could not kill process using port {port}.\n"
                            f"Please close it manually and try again.",
                        )
                        return
                else:
                    return  # User cancelled

            # Create server callback to update GUI when state changes
            def update_callback(q):
                self.current_q = q
                # Only queue one update per frame to avoid overloading event queue
                if not self.pending_server_update:
                    self.pending_server_update = True
                    self.root.after(0, self._update_from_server)

            # Create viewer callback to update GUI when scene is loaded
            def viewer_callback(scene_data):
                # scene_data is a tuple of (new_scene, scene_path)
                # Update missions tab when a new scene is loaded from client
                self.root.after(0, lambda: self._on_client_scene_loaded(scene_data[0], scene_data[1]))

            # Create and start server
            self.server = SimulatorServer(
                robot=self.robot,
                scene=self.scene,
                host=host,
                port=port,
                update_callback=update_callback,
                mission_callback=self._on_server_mission_command,
                scene_path=self.current_scene_path if self.current_scene_path != "(Default)" else None,
                viewer_callback=viewer_callback,
                animation_duration_callback=self.set_animation_duration,
            )

            self.server.start()
            self.server_running = True

            # Update UI to reflect server running state
            self._update_server_ui()
            self._update_server_status_display()

            # Update status bar
            self.status_bar.config(text=f"Server started on {host}:{port}")
            print(f"[OK] Server started on {host}:{port}")

            # Start checking for client connections
            self._check_client_connection()

        except Exception as e:
            messagebox.showerror("Server Error", f"Failed to start server: {e}")

    def _stop_server(self):
        """Stop the simulator server."""
        if self.server:
            try:
                self.server.stop()
                self.server = None
                self.server_running = False
                self.client_connected = False

                # Update UI to reflect server stopped state
                self._update_server_ui()
                self._update_server_status_display()

                # Update status bar
                self.status_bar.config(text="Server stopped")
                print("[OK] Server stopped")

            except Exception as e:
                messagebox.showerror("Server Error", f"Failed to stop server: {e}")

    def _check_client_connection(self):
        """Check if a client is connected and update indicator."""
        if self.server_running and self.server:
            # Check if client socket exists
            is_connected = self.server.client_socket is not None

            if is_connected != self.client_connected:
                self.client_connected = is_connected
                if is_connected:
                    self.status_bar.config(text="Client connected to server")
                    print("[OK] Client connected to server")
                else:
                    self.status_bar.config(text="Server waiting for client connection...")

            # Schedule next check
            self.root.after(500, self._check_client_connection)

    def _update_from_server(self):
        """Update GUI from server state changes."""
        try:
            # Update visualization with new robot state from server
            self._update_visualization()
            self._update_state_display()
        finally:
            # Clear the pending flag so next update can be queued
            self.pending_server_update = False

    def _on_closing(self):
        """Handle window closing."""
        try:
            # Always attempt to stop the server so background threads exit cleanly
            self._stop_server()
        except Exception:
            pass
        try:
            self.root.quit()
        finally:
            self.root.destroy()


def main():
    """Launch GUI application."""
    import sys

    root = tk.Tk()

    # Load scene from command line if provided
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "scenes/demo.json"

    app = SimulatorGUI(root, scene_path)
    root.mainloop()


if __name__ == "__main__":
    main()
