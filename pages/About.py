"""About page — Lane Detection Pipeline project."""

import streamlit as st

st.set_page_config(page_title="About — Lane Detection", page_icon="ℹ️", layout="wide")

st.title("Lane Detection Pipeline")
st.caption("HIT Software Development for Autonomous Vehicles · Roy Carter · 2026")

st.divider()

st.markdown("""
## What Is This Project?

A real-time lane detection system for autonomous vehicles. A front-facing camera records the road ahead.
This software analyzes each video frame and identifies where the lane boundaries are, how curved the road
is, and whether the car is drifting from the lane center — all in real time, without any neural network
or training data.
""")

st.markdown("## The 7-Step Pipeline")

steps = [
    ("1. Noise Reduction",   "🔵", "Gently blurs the image to remove camera sensor noise so only real lane edges remain."),
    ("2. Color Detection",   "🟡", "Isolates white and yellow lane markings using the HLS color model, which is stable under changing lighting."),
    ("3. Edge Detection",    "🔴", "Finds sharp brightness changes (lane marking borders) using the Canny algorithm."),
    ("4. Region Focus",      "🟢", "Masks out everything outside the road area — sky, trees, other lanes."),
    ("5. Bird's-Eye View",   "🟣", "Digitally warps the image to a top-down perspective, removing converging-lane distortion."),
    ("6. Lane Line Fitting", "🟠", "Searches for lane pixels using a sliding window, then fits a smooth curve through them."),
    ("7. Measurement",       "⚪", "Calculates road curvature and lateral offset in real-world meters; overlays results on the frame."),
]

for name, icon, desc in steps:
    col_icon, col_text = st.columns([1, 10])
    with col_icon:
        st.markdown(f"## {icon}")
    with col_text:
        st.markdown(f"**{name}**")
        st.markdown(desc)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("## Tech Stack")
    st.markdown("""
| Component | Technology |
|---|---|
| Computer Vision | OpenCV 4.13 |
| Numerical | NumPy 2.x |
| Interface | Streamlit |
| Database | SQLite (via Python `sqlite3`) |
| Language | Python 3.12 |
""")

with col2:
    st.markdown("## Key Results")
    st.markdown("""
| Scenario | Result |
|---|---|
| Straight road | Lane detected, offset = 0.000 m |
| Gentle left curve | Smooth curve fitted correctly |
| Sharp right curve | Detection succeeds on tight curves |
| Night driving | Color mask finds high-contrast markings |
| Rainy conditions | Edge + color combination handles reflections |

**Performance:** 35–50 FPS at 1280×720 on a laptop CPU — no GPU required.
""")

st.divider()

st.divider()
st.markdown("## GitHub Repository")
st.markdown("[https://github.com/Roy-Carter/IOT-autonomous-car](https://github.com/Roy-Carter/IOT-autonomous-car)")
st.caption("Recording link: add your demo video URL here before presenting.")
