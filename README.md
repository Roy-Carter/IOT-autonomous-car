# Lane Detection Pipeline for Autonomous Vehicles

**Course:** Software Development for Autonomous Vehicles | HIT 2026  
**Student:** Roy Carter  
**GitHub:** https://github.com/Roy-Carter/IOT-autonomous-car

---

## What Is This?

Real-time lane detection system for autonomous vehicles. Upload a video or image, click **Run Detection**, and see lane boundaries, road curvature, and lateral offset — all in a browser-based web app. No neural network. No GPU required.

**Pipeline:** Blur → Color Detection → Edge Detection → Region Mask → Bird's-Eye View → Sliding-Window Fit → Curvature & Offset

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/Roy-Carter/IOT-autonomous-car.git
cd IOT-autonomous-car
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the web app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

The app auto-downloads three Udacity demo clips into the library on first launch — no manual setup needed.

---

## Web App Features

| Tab | What it does |
|---|---|
| **Detection** | Run detection on an image or video. Shows lane status, curvature, offset. Live frame preview for videos. |
| **Pipeline Steps** | Shows all 7 processing stages side by side. |
| **Results History** | Browse and compare every saved detection run. |

**Sidebar controls:**
- Upload to Library / Browse Library
- Advanced Settings — tune all 8 detection thresholds with live sliders
- Visualization mode selector

---

## CLI Usage (optional)

Generate synthetic demo images (no video needed):
```bash
python src/main.py
```

Process a video file:
```bash
python src/main.py --video path/to/road.mp4 --output output/result.mp4
```

Process a single image:
```bash
python src/main.py --image path/to/frame.jpg --output output/result.jpg
```

---

## Results

| Scenario | Lane Detected | Notes |
|---|---|---|
| Straight road | ✅ | Offset = 0.00 m |
| Gentle left curve | ✅ | ~800 m radius |
| Sharp right curve | ✅ | ~300 m radius |
| Night driving | ✅ | Color mask finds markings in headlight zone |
| Rainy conditions | ✅ | Edge + color combined handle reflections |

**Performance:** 35–50 FPS @ 1280×720 — laptop CPU, no GPU.

---

## Deliverables

| Item | Location |
|---|---|
| Presentation | `presentation/` |
| Project documentation | `docs/` |
| Source code | This repository |
| Recording | `data/Screencast from 2026-04-30 21-58-57.mp4` |

---

## Technologies

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| OpenCV | All image processing |
| NumPy | Array math, curve fitting |
| Streamlit | Web interface |
| SQLite | Media library & results storage |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| Video window won't open | Ensure display available (X11/Wayland on Linux) |
| `Cannot open video` | Verify path and format (mp4, avi, mov) |
