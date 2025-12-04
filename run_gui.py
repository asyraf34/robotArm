#!/usr/bin/env python
"""
Launch the SCARA simulator GUI application.

Usage:
    python run_gui.py [scene_file.json]

Example:
    python run_gui.py
    python run_gui.py scenes/demo.json
"""

import sys
import tkinter as tk
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scara_sim.gui import SimulatorGUI


def main():
    """Main entry point for GUI application."""
    # Get scene path from command line
    scene_path = None
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
        if not Path(scene_path).exists():
            print(f"Warning: Scene file not found: {scene_path}")
            print("Using default scene instead.")
            scene_path = None
    else:
        # Try default scene
        default_scene = Path("scenes/demo.json")
        if default_scene.exists():
            scene_path = str(default_scene)

    # Create root window
    root = tk.Tk()

    # Set icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass

    # Create GUI
    try:
        app = SimulatorGUI(root, scene_path)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        # Start main loop
        root.mainloop()

    except Exception as e:
        import traceback
        print("Error starting GUI:")
        print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
