"""
Geometric utilities: polygons, capsules, transforms, and SAT collision.
"""

from typing import Optional, Tuple
import numpy as np


def point_to_segment_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """
    Minimum distance from point p to line segment ab.

    Parameters
    ----------
    p : np.ndarray
        Point [x, y].
    a, b : np.ndarray
        Segment endpoints [x, y].

    Returns
    -------
    float
        Distance.
    """
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def capsule_to_capsule_distance(
    c1: dict, c2: dict
) -> float:
    """
    Minimum distance between two capsules.

    Parameters
    ----------
    c1, c2 : dict
        {"p1": (x, y), "p2": (x, y), "radius": r}.

    Returns
    -------
    float
        Distance between closest points on capsule surfaces.
    """
    p1a = np.array(c1["p1"])
    p1b = np.array(c1["p2"])
    p2a = np.array(c2["p1"])
    p2b = np.array(c2["p2"])

    # Closest points on segments
    ab = p1b - p1a
    cd = p2b - p2a
    ab_sq = np.dot(ab, ab)
    cd_sq = np.dot(cd, cd)

    if ab_sq < 1e-12 and cd_sq < 1e-12:
        dist = np.linalg.norm(p1a - p2a)
    elif ab_sq < 1e-12:
        dist = point_to_segment_distance(p1a, p2a, p2b)
    elif cd_sq < 1e-12:
        dist = point_to_segment_distance(p2a, p1a, p1b)
    else:
        # Approximate: closest points on segments
        ac = p2a - p1a
        t = np.clip(np.dot(ac, ab) / ab_sq, 0.0, 1.0)
        s = np.clip(np.dot(ac + t * ab, cd) / cd_sq, 0.0, 1.0)
        closest1 = p1a + t * ab
        closest2 = p2a + s * cd
        dist = np.linalg.norm(closest1 - closest2)

    return dist - (c1["radius"] + c2["radius"])


def polygon_centroid(points: list[Tuple[float, float]]) -> np.ndarray:
    """
    Compute centroid of a polygon.

    Parameters
    ----------
    points : list[Tuple[float, float]]
        Polygon vertices in order.

    Returns
    -------
    np.ndarray
        Centroid [x, y].
    """
    pts = np.array(points)
    return np.mean(pts, axis=0)


def polygon_to_capsule_collision(
    poly_points: list[Tuple[float, float]], capsule: dict
) -> bool:
    """
    Check collision between convex polygon and capsule using SAT.

    Parameters
    ----------
    poly_points : list[Tuple[float, float]]
        Polygon vertices.
    capsule : dict
        {"p1": (x, y), "p2": (x, y), "radius": r}.

    Returns
    -------
    bool
        True if collision detected.
    """
    # For MVP, use simplified approach: check if capsule's segment
    # intersects polygon or if endpoints are inside.

    p1 = np.array(capsule["p1"])
    p2 = np.array(capsule["p2"])
    r = capsule["radius"]

    pts = np.array(poly_points)
    n = len(pts)

    # Check distance from segment to each edge
    for i in range(n):
        edge_a = pts[i]
        edge_b = pts[(i + 1) % n]
        dist = point_to_segment_distance(p1, edge_a, edge_b)
        if dist < r:
            return True
        dist = point_to_segment_distance(p2, edge_a, edge_b)
        if dist < r:
            return True

    # Check if endpoints are inside polygon (ray casting)
    if point_in_polygon(p1, poly_points) or point_in_polygon(
        p2, poly_points
    ):
        return True

    return False


def point_in_polygon(
    point: np.ndarray, polygon: list[Tuple[float, float]]
) -> bool:
    """
    Check if point is inside convex polygon using ray casting.

    Parameters
    ----------
    point : np.ndarray
        Point [x, y].
    polygon : list[Tuple[float, float]]
        Polygon vertices in order.

    Returns
    -------
    bool
        True if inside.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def aabb_collision(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> bool:
    """
    Check if two axis-aligned bounding boxes collide.

    Parameters
    ----------
    box1, box2 : Tuple[float, float, float, float]
        (min_x, min_y, max_x, max_y).

    Returns
    -------
    bool
        True if collision detected.
    """
    return (
        box1[0] <= box2[2]
        and box1[2] >= box2[0]
        and box1[1] <= box2[3]
        and box1[3] >= box2[1]
    )
