"""
Frame capture and video export utilities.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def export_video(
    fig, animate_func, n_frames: int, output_path: str, fps: int = 30
) -> None:
    """
    Export animation to MP4 video.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    animate_func : callable
        Animation function taking frame index.
    n_frames : int
        Number of frames.
    output_path : str
        Output video path.
    fps : int
        Frames per second.
    """
    writer = FFMpegWriter(fps=fps, bitrate=1800)

    with writer.saving(fig, output_path, dpi=100):
        for i in range(n_frames):
            animate_func(i)
            writer.grab_frame()

    print(f"Video exported to {output_path}")


def export_frames(
    fig, animate_func, n_frames: int, output_dir: str
) -> None:
    """
    Export animation frames as PNG images.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    animate_func : callable
        Animation function taking frame index.
    n_frames : int
        Number of frames.
    output_dir : str
        Output directory.
    """
    frames_path = Path(output_dir)
    frames_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        animate_func(i)
        fig.canvas.draw()
        frame_path = frames_path / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=100)

    print(f"Frames exported to {output_dir}")
