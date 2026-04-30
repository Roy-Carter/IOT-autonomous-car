"""
Lane Detection Pipeline - Main Entry Point

Usage:
  python main.py                        # Run synthetic demo (no video needed)
  python main.py --video path/to/video  # Process a video file
  python main.py --image path/to/image  # Process a single image
  python main.py --demo                 # Generate demo output images
"""

import argparse
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lane_detector import LaneDetector
from synthetic_road import generate_road_scene


def process_video(video_path: str, output_path: str = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = LaneDetector(image_width=width, image_height=height)
    writer = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print(f"Processing {video_path} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, info = detector.process_frame(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} | Curvature: {(info.left_curvature_m + info.right_curvature_m)/2:.0f}m "
                  f"| Offset: {info.center_offset_m:.2f}m")

        if writer:
            writer.write(result)

        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames.")


def process_image(image_path: str, output_path: str = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return

    h, w = img.shape[:2]
    detector = LaneDetector(image_width=w, image_height=h)
    result, info = detector.process_frame(img)

    print(f"Lane detected: {info.lane_detected}")
    print(f"Left curvature:  {info.left_curvature_m:.1f} m")
    print(f"Right curvature: {info.right_curvature_m:.1f} m")
    print(f"Center offset:   {info.center_offset_m:.3f} m")

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved to {output_path}")
    cv2.imshow("Lane Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_synthetic_demo(output_dir: str = "../output/samples"):
    os.makedirs(output_dir, exist_ok=True)
    detector = LaneDetector(image_width=1280, image_height=720)

    scenarios = [
        ("straight_road",    dict(curve_radius=0,     offset=0.0)),
        ("gentle_left_curve",dict(curve_radius=800,   offset=-0.1)),
        ("sharp_right_curve",dict(curve_radius=300,   offset=0.2)),
        ("night_road",       dict(curve_radius=500,   offset=0.0,  night=True)),
        ("rainy_road",       dict(curve_radius=0,     offset=0.05, rain=True)),
    ]

    print("Generating synthetic road demo frames...")
    for name, params in scenarios:
        frame = generate_road_scene(width=1280, height=720, **params)
        result, info = detector.process_frame(frame)

        # Save side-by-side comparison
        side = np.hstack([frame, result])
        path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(path, side)
        print(f"  [{name}] curvature={((info.left_curvature_m+info.right_curvature_m)/2):.0f}m "
              f"offset={info.center_offset_m:.3f}m → {path}")

    # Also save pipeline step visualization
    frame = generate_road_scene(width=1280, height=720)
    _save_pipeline_steps(frame, detector, output_dir)
    print(f"\nAll outputs saved to {output_dir}/")


def _save_pipeline_steps(frame, detector, output_dir):
    import cv2
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    yellow = cv2.inRange(hls, np.array([15, 80, 80]), np.array([35, 255, 255]))
    color_mask = cv2.bitwise_or(white, yellow)
    color_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Hough lines result
    hough_result = detector.hough_lane_lines(frame.copy())

    # Warped (bird's eye)
    binary = detector._preprocess(frame)
    warped = cv2.warpPerspective(binary, detector.M, (1280, 720))
    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Full pipeline result
    result, _ = detector.process_frame(frame.copy())

    steps = [
        ("1_original.jpg",     frame),
        ("2_grayscale.jpg",    gray_bgr),
        ("3_color_mask.jpg",   color_bgr),
        ("4_canny_edges.jpg",  edges_bgr),
        ("5_hough_lines.jpg",  hough_result),
        ("6_birdseye.jpg",     warped_bgr),
        ("7_final_result.jpg", result),
    ]
    for filename, img in steps:
        cv2.imwrite(os.path.join(output_dir, filename), img)
    print(f"  Pipeline step images saved.")


def main():
    parser = argparse.ArgumentParser(description="Lane Detection Pipeline")
    parser.add_argument("--video", help="Input video file path")
    parser.add_argument("--image", help="Input image file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    args = parser.parse_args()

    if args.video:
        process_video(args.video, args.output)
    elif args.image:
        process_image(args.image, args.output)
    else:
        run_synthetic_demo(args.output or "../output/samples")


if __name__ == "__main__":
    main()
