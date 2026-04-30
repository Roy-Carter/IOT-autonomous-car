"""
Lane Detection Pipeline for Autonomous Vehicles
Core detection module implementing:
  - Color thresholding (HLS color space)
  - Canny edge detection
  - Region of Interest masking
  - Hough Line Transform
  - Lane line averaging & extrapolation
  - Perspective transform (bird's-eye view)
  - Lane curvature estimation
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LaneInfo:
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    left_curvature_m: float
    right_curvature_m: float
    center_offset_m: float
    lane_detected: bool


class LaneDetector:
    def __init__(self, image_width: int = 1280, image_height: int = 720):
        self.width = image_width
        self.height = image_height

        # Perspective transform source/destination points (normalized ratios)
        self._src_pts = np.float32([
            [0.44 * self.width, 0.65 * self.height],
            [0.56 * self.width, 0.65 * self.height],
            [0.15 * self.width, self.height],
            [0.85 * self.width, self.height],
        ])
        self._dst_pts = np.float32([
            [0.25 * self.width, 0],
            [0.75 * self.width, 0],
            [0.25 * self.width, self.height],
            [0.75 * self.width, self.height],
        ])

        self.M = cv2.getPerspectiveTransform(self._src_pts, self._dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self._dst_pts, self._src_pts)

        # Real-world scale (approximate for highway footage at 720p)
        self.ym_per_pix = 30.0 / self.height   # meters per pixel vertical
        self.xm_per_pix = 3.7 / (0.5 * self.width)  # meters per pixel horizontal

        # Smoothing buffers
        self._left_fit_history = []
        self._right_fit_history = []
        self._history_len = 8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, LaneInfo]:
        """Full pipeline: raw frame → annotated frame + lane info."""
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))

        binary = self._preprocess(frame)
        warped = cv2.warpPerspective(binary, self.M, (self.width, self.height))
        left_fit, right_fit, debug_img = self._fit_polynomial(warped)

        left_fit = self._smooth(left_fit, self._left_fit_history)
        right_fit = self._smooth(right_fit, self._right_fit_history)

        lane_info = self._compute_metrics(left_fit, right_fit)
        result = self._draw_overlay(frame, warped, left_fit, right_fit)
        result = self._draw_hud(result, lane_info)
        return result, lane_info

    # ------------------------------------------------------------------
    # Step 1 – Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur + color threshold → binary mask."""
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        color_mask = self._color_threshold(blurred)
        edges = self._canny_edges(blurred)
        combined = cv2.bitwise_or(color_mask, edges)
        return self._region_of_interest(combined)

    def _color_threshold(self, img: np.ndarray) -> np.ndarray:
        """Isolate white and yellow lane markings in HLS color space."""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # White: high lightness regardless of hue
        white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
        # Yellow: hue 15–35, moderate saturation
        yellow_mask = cv2.inRange(hls, np.array([15, 80, 80]), np.array([35, 255, 255]))
        return cv2.bitwise_or(white_mask, yellow_mask)

    def _canny_edges(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, threshold1=50, threshold2=150)

    def _region_of_interest(self, binary: np.ndarray) -> np.ndarray:
        """Trapezoidal mask covering road area."""
        mask = np.zeros_like(binary)
        poly = np.array([[
            (int(0.05 * self.width), self.height),
            (int(0.45 * self.width), int(0.60 * self.height)),
            (int(0.55 * self.width), int(0.60 * self.height)),
            (int(0.95 * self.width), self.height),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        return cv2.bitwise_and(binary, mask)

    # ------------------------------------------------------------------
    # Step 2 – Hough Lines (for simple/overlay mode)
    # ------------------------------------------------------------------

    def hough_lane_lines(self, frame: np.ndarray) -> np.ndarray:
        """Return frame annotated with Hough-detected lane lines."""
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        binary = self._preprocess(frame)
        lines = cv2.HoughLinesP(binary, rho=2, theta=np.pi / 180,
                                threshold=50, minLineLength=40, maxLineGap=150)
        left, right = self._average_lane_lines(frame, lines)
        line_img = np.zeros_like(frame)
        for lane in [left, right]:
            if lane is not None:
                x1, y1, x2, y2 = lane
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
        return cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)

    def _average_lane_lines(self, frame, lines):
        left_lines, right_lines = [], []
        if lines is None:
            return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < -0.3:
                left_lines.append((slope, intercept, length))
            elif slope > 0.3:
                right_lines.append((slope, intercept, length))

        def weighted_avg(lines_data):
            if not lines_data:
                return None
            lengths = np.array([l[2] for l in lines_data])
            slopes = np.average([l[0] for l in lines_data], weights=lengths)
            intercepts = np.average([l[1] for l in lines_data], weights=lengths)
            y1 = self.height
            y2 = int(0.60 * self.height)
            x1 = int((y1 - intercepts) / slopes)
            x2 = int((y2 - intercepts) / slopes)
            return x1, y1, x2, y2

        return weighted_avg(left_lines), weighted_avg(right_lines)

    # ------------------------------------------------------------------
    # Step 3 – Polynomial Fit (sliding window)
    # ------------------------------------------------------------------

    def _fit_polynomial(self, warped_binary: np.ndarray):
        histogram = np.sum(warped_binary[self.height // 2:, :], axis=0)
        mid = self.width // 2
        leftx_base = np.argmax(histogram[:mid])
        rightx_base = np.argmax(histogram[mid:]) + mid

        nwindows = 9
        margin = 100
        minpix = 50
        window_height = self.height // nwindows

        nonzero = warped_binary.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        debug_img = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR) if len(warped_binary.shape) == 2 else warped_binary.copy()

        for window in range(nwindows):
            win_y_low = self.height - (window + 1) * window_height
            win_y_high = self.height - window * window_height

            for cx, inds_list, color in [
                (leftx_current, left_lane_inds, (0, 255, 0)),
                (rightx_current, right_lane_inds, (0, 0, 255)),
            ]:
                win_x_low = cx - margin
                win_x_high = cx + margin
                cv2.rectangle(debug_img, (win_x_low, win_y_low), (win_x_high, win_y_high), color, 2)
                good = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                inds_list.append(good)
                if len(good) > minpix:
                    if color == (0, 255, 0):
                        leftx_current = int(np.mean(nonzerox[good]))
                    else:
                        rightx_current = int(np.mean(nonzerox[good]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 50 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 50 else None

        return left_fit, right_fit, debug_img

    # ------------------------------------------------------------------
    # Step 4 – Smoothing
    # ------------------------------------------------------------------

    def _smooth(self, fit, history):
        if fit is None:
            return history[-1] if history else None
        history.append(fit)
        if len(history) > self._history_len:
            history.pop(0)
        return np.mean(history, axis=0)

    # ------------------------------------------------------------------
    # Step 5 – Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, left_fit, right_fit) -> LaneInfo:
        if left_fit is None and right_fit is None:
            return LaneInfo(None, None, 0, 0, 0, False)

        ploty = np.linspace(0, self.height - 1, self.height)

        def curvature(fit):
            if fit is None:
                return 0.0
            y_eval = self.height * self.ym_per_pix
            A = fit[0] * (self.xm_per_pix / self.ym_per_pix ** 2)
            B = fit[1] * (self.xm_per_pix / self.ym_per_pix)
            return ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / abs(2 * A + 1e-6)

        left_curv = curvature(left_fit)
        right_curv = curvature(right_fit)

        # Lane center offset
        left_x = left_fit[0] * self.height**2 + left_fit[1] * self.height + left_fit[2] if left_fit is not None else 0
        right_x = right_fit[0] * self.height**2 + right_fit[1] * self.height + right_fit[2] if right_fit is not None else self.width
        lane_center = (left_x + right_x) / 2
        offset = (self.width / 2 - lane_center) * self.xm_per_pix

        return LaneInfo(left_fit, right_fit, left_curv, right_curv, offset, True)

    # ------------------------------------------------------------------
    # Step 6 – Drawing
    # ------------------------------------------------------------------

    def _draw_overlay(self, original, warped_binary, left_fit, right_fit) -> np.ndarray:
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if left_fit is None or right_fit is None:
            return original

        ploty = np.linspace(0, self.height - 1, self.height)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
        lane_pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(overlay, lane_pts, (0, 200, 0))
        cv2.polylines(overlay, left_pts, False, (255, 50, 50), 12)
        cv2.polylines(overlay, right_pts, False, (50, 50, 255), 12)

        unwarped = cv2.warpPerspective(overlay, self.M_inv, (self.width, self.height))
        return cv2.addWeighted(original, 1.0, unwarped, 0.4, 0)

    def _draw_hud(self, img: np.ndarray, info: LaneInfo) -> np.ndarray:
        panel = img.copy()
        cv2.rectangle(panel, (10, 10), (450, 130), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.6, panel, 0.4, 0)

        avg_curv = (info.left_curvature_m + info.right_curvature_m) / 2
        direction = "LEFT" if info.center_offset_m < 0 else "RIGHT"
        status = "DETECTED" if info.lane_detected else "NOT DETECTED"
        color = (0, 255, 0) if info.lane_detected else (0, 0, 255)

        texts = [
            (f"Lane: {status}", color),
            (f"Curvature: {avg_curv:.0f} m", (255, 255, 255)),
            (f"Offset: {abs(info.center_offset_m):.2f} m {direction}", (255, 255, 255)),
        ]
        for i, (text, c) in enumerate(texts):
            cv2.putText(img, text, (20, 45 + i * 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, c, 2)
        return img
