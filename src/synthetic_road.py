"""
Synthetic road scene generator for demo purposes.
Produces photorealistic-ish road images with:
  - Asphalt texture
  - Lane markings (white dashed center, solid yellow/white edges)
  - Configurable curve radius
  - Night / rain effects
"""

import cv2
import numpy as np
from typing import Optional


def generate_road_scene(
    width: int = 1280,
    height: int = 720,
    curve_radius: float = 0,
    offset: float = 0.0,
    night: bool = False,
    rain: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic road frame.

    Args:
        curve_radius: Radius of curvature in pixels (0 = straight).
        offset:       Lateral offset of the car within the lane (-1..1).
        night:        Apply night-time lighting.
        rain:         Add rain streaks and wet reflections.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky
    sky_color = (30, 20, 15) if night else (180, 210, 240)
    img[:] = sky_color

    # Road trapezoid
    road_pts = np.array([
        [int(0.10 * width), height],
        [int(0.40 * width), int(0.58 * height)],
        [int(0.60 * width), int(0.58 * height)],
        [int(0.90 * width), height],
    ], dtype=np.int32)
    road_color = (30, 30, 30) if not rain else (20, 22, 22)
    cv2.fillPoly(img, [road_pts], road_color)

    # Asphalt noise texture
    noise = rng.integers(0, 20, (height, width, 3), dtype=np.uint8)
    road_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(road_mask, [road_pts], 255)
    road_zone = road_mask > 0
    img[road_zone] = np.clip(img[road_zone].astype(int) + noise[road_zone] - 10, 0, 255).astype(np.uint8)

    # Horizon line
    h_y = int(0.58 * height)
    cv2.line(img, (0, h_y), (width, h_y), (50, 50, 50), 1)

    # Lane markings
    _draw_lane_markings(img, width, height, curve_radius, offset, rng)

    # Wet reflections
    if rain:
        _apply_rain(img, width, height, rng)

    # Night: darken + add headlight cone
    if night:
        img = (img * 0.35).astype(np.uint8)
        _apply_headlights(img, width, height)

    # Mild blur to simulate motion
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _lane_x(base_x: float, y: int, height: int, curve_radius: float, direction: int = 1) -> int:
    """Compute x position for a lane marking given vertical position and curve."""
    if curve_radius == 0:
        return int(base_x)
    t = (height - y) / height
    curve_offset = direction * (t ** 2) * (height ** 2 / (2 * curve_radius))
    return int(base_x + curve_offset)


def _draw_lane_markings(img, width, height, curve_radius, offset, rng):
    h_y = int(0.58 * height)
    lane_width_bottom = int(0.28 * width)

    center_bottom = int(width / 2 + offset * 0.5 * lane_width_bottom)
    center_top = int(width / 2 + offset * 0.2 * lane_width_bottom)

    left_bottom  = center_bottom - lane_width_bottom // 2
    right_bottom = center_bottom + lane_width_bottom // 2
    left_top  = center_top - int(0.08 * width)
    right_top = center_top + int(0.08 * width)

    # Solid edge lines
    left_pts  = np.array([[left_bottom, height], [left_top, h_y]], dtype=np.int32)
    right_pts = np.array([[right_bottom, height], [right_top, h_y]], dtype=np.int32)
    cv2.polylines(img, [left_pts.reshape(-1, 1, 2)],  False, (255, 255, 255), 6)
    cv2.polylines(img, [right_pts.reshape(-1, 1, 2)], False, (255, 255, 255), 6)

    # Dashed center line
    dash_len = int(0.04 * height)
    gap_len  = int(0.035 * height)
    y = height
    drawing = True
    while y > h_y:
        y_end = max(y - dash_len, h_y)
        t_start = (height - y) / height
        t_end   = (height - y_end) / height
        cx_start = int(left_bottom + (left_top - left_bottom) * t_start +
                       (right_bottom - left_bottom) / 2 +
                       ((right_top - left_top) - (right_bottom - left_bottom)) / 2 * t_start)
        cx_end = int(left_bottom + (left_top - left_bottom) * t_end +
                     (right_bottom - left_bottom) / 2 +
                     ((right_top - left_top) - (right_bottom - left_bottom)) / 2 * t_end)
        if drawing:
            cv2.line(img, (cx_start, y), (cx_end, y_end), (255, 255, 0), 3)
        y = y_end - (0 if drawing else gap_len)
        drawing = not drawing


def _apply_headlights(img, width, height):
    cone = np.zeros((height, width, 3), dtype=np.uint8)
    pts = np.array([
        [width // 2 - 30, height - 5],
        [width // 2 + 30, height - 5],
        [int(0.65 * width), int(0.58 * height)],
        [int(0.35 * width), int(0.58 * height)],
    ], dtype=np.int32)
    cv2.fillPoly(cone, [pts], (80, 80, 60))
    img[:] = np.clip(img.astype(int) + cone.astype(int), 0, 255).astype(np.uint8)


def _apply_rain(img, width, height, rng):
    rain = np.zeros_like(img)
    for _ in range(200):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        length = rng.integers(10, 30)
        cv2.line(rain, (x, y), (x + 2, y + length), (150, 150, 160), 1)
    img[:] = np.clip(img.astype(int) + rain.astype(int) * 0.5, 0, 255).astype(np.uint8)
    # Slight blue tint for wet road
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 10, 0, 255).astype(np.uint8)
