"""
Command-line interface for SCARA simulator.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import glob

from scara_sim.core.robot import ScaraRobot
from scara_sim.io.scene import load_scene
from scara_sim.io.results import ResultWriter
from scara_sim.viz.viewer2d import Viewer
from scara_sim import planning


def create_robot_from_scene(scene):
    """Create robot from scene configuration."""
    if scene.robot_config is None:
        raise ValueError("Scene missing robot configuration")

    cfg = scene.robot_config
    return ScaraRobot(
        L1=cfg["L1"],
        L2=cfg["L2"],
        joint_limits=cfg["joint_limits"],
    )


def run_single(args):
    """Run a single planning task."""
    # Load scene
    scene = load_scene(args.scene)
    robot = create_robot_from_scene(scene)

    # Get target
    if not scene.targets:
        print("Error: Scene has no targets")
        return 1

    target = scene.targets[0]
    target_xy = np.array([target["x"], target["y"]])

    # Compute IK
    start_q = np.array([0.0, 0.0, 0.0, 0.0])
    goal_q = robot.ik_xy(target_xy, elbow="up")

    if goal_q is None:
        goal_q = robot.ik_xy(target_xy, elbow="down")

    if goal_q is None:
        print(f"Error: Target {target_xy} is unreachable")
        return 1

    # Plan
    planner_name = args.planner
    if planner_name == "straight":
        from scara_sim.planning import straight

        traj = straight.plan(robot, scene, start_q, goal_q)
    elif planner_name == "rrt":
        from scara_sim.planning import rrt

        traj = rrt.plan(robot, scene, start_q, goal_q, seed=args.seed)
    else:
        print(f"Error: Unknown planner '{planner_name}'")
        return 1

    # Print results
    if traj["success"]:
        print(f"Planning succeeded in {traj['planning_time']:.3f}s")
        print(f"  Path length: {traj['path_length']:.3f}")
        print(f"  Clearance: {traj['clearance']:.4f}")
        print(f"  Nodes explored: {traj['nodes_explored']}")
    else:
        print("Planning failed")
        return 1

    # Save results
    if args.export:
        writer = ResultWriter(args.export)
        video_path = None
        frames_dir = None

        if not args.no_viz:
            video_path = str(Path(args.export) / "video.mp4")
            if args.save_frames:
                frames_dir = str(Path(args.export) / "frames")

        meta = {"planner": planner_name, "seed": args.seed}
        writer.save_run(traj, meta, scene, "run", video_path, frames_dir)
        print(f"Results saved to {args.export}")

    # Visualize
    if not args.no_viz:
        viewer = Viewer(scene, robot)
        video_path = None
        frames_dir = None

        if args.export:
            video_path = str(Path(args.export) / "video.mp4")
            if args.save_frames:
                frames_dir = str(Path(args.export) / "frames")

        viewer.play(
            traj,
            realtime=not args.fast,
            save_video=video_path,
            save_frames_dir=frames_dir,
            headless=args.headless,
        )

    return 0


def run_bench(args):
    """Run benchmark across multiple scenes and planners."""
    # Expand scene patterns
    scene_files = []
    for pattern in args.scenes:
        scene_files.extend(glob.glob(pattern))

    if not scene_files:
        print("Error: No scene files found")
        return 1

    print(f"Benchmarking {len(scene_files)} scenes with {len(args.planners)} planners")
    print(f"Repetitions: {args.reps}")
    print()

    results = []

    for scene_path in scene_files:
        scene = load_scene(scene_path)
        robot = create_robot_from_scene(scene)

        if not scene.targets:
            continue

        target = scene.targets[0]
        target_xy = np.array([target["x"], target["y"]])

        start_q = np.array([0.0, 0.0, 0.0, 0.0])
        goal_q = robot.ik_xy(target_xy, elbow="up")
        if goal_q is None:
            goal_q = robot.ik_xy(target_xy, elbow="down")
        if goal_q is None:
            continue

        for planner_name in args.planners:
            for rep in range(args.reps):
                seed = args.seed + rep if args.seed is not None else None

                if planner_name == "straight":
                    from scara_sim.planning import straight

                    traj = straight.plan(robot, scene, start_q, goal_q)
                elif planner_name == "rrt":
                    from scara_sim.planning import rrt

                    traj = rrt.plan(robot, scene, start_q, goal_q, seed=seed)
                else:
                    continue

                result = {
                    "scene": Path(scene_path).stem,
                    "planner": planner_name,
                    "rep": rep,
                    "success": traj["success"],
                    "time": traj["planning_time"],
                    "path_length": traj.get("path_length"),
                    "nodes": traj.get("nodes_explored", 0),
                }
                results.append(result)

                status = "OK" if traj["success"] else "FAIL"
                print(
                    f"  {Path(scene_path).stem:20} {planner_name:10} rep={rep} {status:4} "
                    f"time={traj['planning_time']:.3f}s"
                )

    # Summary
    print("\nSummary:")
    for planner_name in args.planners:
        planner_results = [r for r in results if r["planner"] == planner_name]
        successes = sum(1 for r in planner_results if r["success"])
        avg_time = np.mean([r["time"] for r in planner_results if r["success"]])
        print(f"  {planner_name}: {successes}/{len(planner_results)} success, avg time {avg_time:.3f}s")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SCARA Robot Simulator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run single planning task")
    run_parser.add_argument("--scene", required=True, help="Scene JSON file")
    run_parser.add_argument(
        "--planner", default="rrt", choices=["straight", "rrt"], help="Planner to use"
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--export", help="Export results to directory")
    run_parser.add_argument("--headless", action="store_true", help="Run without display")
    run_parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    run_parser.add_argument("--fast", action="store_true", help="Fast playback")
    run_parser.add_argument("--save-frames", action="store_true", help="Save frames as PNGs")

    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("--scenes", nargs="+", required=True, help="Scene file patterns")
    bench_parser.add_argument(
        "--planners", nargs="+", default=["straight", "rrt"], help="Planners to benchmark"
    )
    bench_parser.add_argument("--reps", type=int, default=3, help="Repetitions per config")
    bench_parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    if args.command == "run":
        return run_single(args)
    elif args.command == "bench":
        return run_bench(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
