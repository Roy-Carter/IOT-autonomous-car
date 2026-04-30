"""
Lane Detection Pipeline — Streamlit Web UI
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
import tempfile
import json
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lane_detector import LaneDetector, LaneInfo
import db as _db

_db.init_db()

st.set_page_config(
    page_title="Lane Detection Pipeline",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_VIDEOS = [
    {
        "name": "Udacity — solidWhiteRight",
        "url": "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4",
        "ext": ".mp4",
    },
    {
        "name": "Udacity — solidYellowLeft",
        "url": "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidYellowLeft.mp4",
        "ext": ".mp4",
    },
    {
        "name": "Udacity — challenge",
        "url": "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/challenge.mp4",
        "ext": ".mp4",
    },
]

VIZ_LABELS = ["Final Result", "Original", "Color Mask",
               "Canny Edges", "ROI Binary", "Bird's Eye", "Sliding Window"]
VIZ_KEYS   = {
    "Final Result":   "final_result",
    "Original":       "original",
    "Color Mask":     "color_mask",
    "Canny Edges":    "canny_edges",
    "ROI Binary":     "roi_binary",
    "Bird's Eye":     "birds_eye",
    "Sliding Window": "sliding_window",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
:root{--bg-card:#1c2333;--muted:#8892b0;--border:#2a3050;}
.metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;
             padding:20px 16px 16px;text-align:center;margin-bottom:12px;}
.metric-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px;}
.metric-value{font-size:2rem;font-weight:700;letter-spacing:-.02em;line-height:1.1;}
.metric-sub{font-size:.8rem;color:var(--muted);margin-top:4px;}
.green{color:#06d6a0;} .yellow{color:#ffd166;} .red{color:#ef476f;} .blue{color:#4dabf7;}
.badge{display:inline-block;padding:5px 16px;border-radius:20px;font-size:.82rem;font-weight:700;
       letter-spacing:.08em;margin-top:4px;}
.badge-ok{background:rgba(6,214,160,.15);color:#06d6a0;border:1px solid #06d6a0;}
.badge-err{background:rgba(239,71,111,.15);color:#ef476f;border:1px solid #ef476f;}
.step-label{text-align:center;font-size:.72rem;color:var(--muted);text-transform:uppercase;
            letter-spacing:.07em;margin-top:5px;margin-bottom:14px;}
[data-testid="stImage"] img{border-radius:10px;}
[data-testid="stTabs"] button{font-size:.92rem;font-weight:600;}
</style>""", unsafe_allow_html=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def frame_to_jpeg(img) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

def get_detector(width, height):
    if st.session_state.get("detector_size") != (width, height):
        st.session_state["detector"]      = LaneDetector(image_width=width, image_height=height)
        st.session_state["detector_size"] = (width, height)
    det = st.session_state["detector"]
    det._history_len = st.session_state.get("smooth_win", 8)
    return det

def collect_params():
    return {
        "white_l":      st.session_state.get("white_l",   200),
        "yellow_h_min": st.session_state.get("yh_min",     15),
        "yellow_h_max": st.session_state.get("yh_max",     35),
        "canny_lo":     st.session_state.get("canny_lo",   50),
        "canny_hi":     st.session_state.get("canny_hi",  150),
        "nwindows":     st.session_state.get("nwindows",    9),
        "margin":       st.session_state.get("margin",    100),
        "minpix":       st.session_state.get("minpix",     50),
    }

def _ensure_demo_videos_in_db():
    """Download any missing demo videos into the DB at startup."""
    existing = {row["name"] for row in _db.get_all_media()}
    missing  = [d for d in DEMO_VIDEOS if d["name"] not in existing]
    if not missing:
        return
    with st.spinner(f"Setting up demo library ({len(missing)} video(s))…"):
        for demo in missing:
            tmp = tempfile.NamedTemporaryFile(suffix=demo["ext"], delete=False)
            urllib.request.urlretrieve(demo["url"], tmp.name)
            tmp.close()
            with open(tmp.name, "rb") as f:
                video_bytes = f.read()
            os.unlink(tmp.name)
            _db.save_media(demo["name"], "video", video_bytes, demo["ext"])


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _fit_poly_param(warped_binary, detector, nwindows, margin, minpix):
    h, w     = detector.height, detector.width
    histogram = np.sum(warped_binary[h // 2:, :], axis=0)
    mid       = w // 2
    lx_base   = np.argmax(histogram[:mid])
    rx_base   = np.argmax(histogram[mid:]) + mid
    wh        = h // nwindows
    nzy       = np.array(warped_binary.nonzero()[0])
    nzx       = np.array(warped_binary.nonzero()[1])
    lx, rx    = lx_base, rx_base
    l_inds, r_inds = [], []
    debug = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR) if warped_binary.ndim == 2 else warped_binary.copy()
    for win in range(nwindows):
        y_lo, y_hi = h - (win + 1) * wh, h - win * wh
        for cx, lst, col in [(lx, l_inds, (0,220,100)), (rx, r_inds, (0,100,220))]:
            xl, xr = cx - margin, cx + margin
            cv2.rectangle(debug, (xl, y_lo), (xr, y_hi), col, 2)
            good = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl) & (nzx < xr)).nonzero()[0]
            lst.append(good)
            if len(good) > minpix:
                mx = int(np.mean(nzx[good]))
                if col == (0,220,100): lx = mx
                else:                  rx = mx
    li = np.concatenate(l_inds); ri = np.concatenate(r_inds)
    lfit = np.polyfit(nzy[li], nzx[li], 2) if len(li) > 50 else None
    rfit = np.polyfit(nzy[ri], nzx[ri], 2) if len(ri) > 50 else None
    return lfit, rfit, debug


def run_pipeline(frame, detector, params):
    if (frame.shape[1], frame.shape[0]) != (detector.width, detector.height):
        frame = cv2.resize(frame, (detector.width, detector.height))
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hls     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    white   = cv2.inRange(hls, np.array([0,   params["white_l"],      0]),
                               np.array([180, 255,               255]))
    yellow  = cv2.inRange(hls, np.array([params["yellow_h_min"], 80,  80]),
                               np.array([params["yellow_h_max"], 255, 255]))
    color   = cv2.bitwise_or(white, yellow)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, params["canny_lo"], params["canny_hi"])
    roi     = detector._region_of_interest(cv2.bitwise_or(color, edges))
    warped  = cv2.warpPerspective(roi, detector.M, (detector.width, detector.height))
    lfit, rfit, sw = _fit_poly_param(warped, detector, params["nwindows"],
                                     params["margin"], params["minpix"])
    lfit   = detector._smooth(lfit, detector._left_fit_history)
    rfit   = detector._smooth(rfit, detector._right_fit_history)
    info   = detector._compute_metrics(lfit, rfit)
    result = detector._draw_overlay(frame, warped, lfit, rfit)
    result = detector._draw_hud(result, info)
    def b(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    return {"original": frame.copy(), "color_mask": b(color), "canny_edges": b(edges),
            "roi_binary": b(roi), "birds_eye": b(warped), "sliding_window": sw,
            "final_result": result, "lane_info": info}


# ── Render helpers ────────────────────────────────────────────────────────────

def render_metrics(info: LaneInfo):
    avg_curv = (info.left_curvature_m + info.right_curvature_m) / 2
    off_abs  = abs(info.center_offset_m)
    dirn     = "RIGHT ▶" if info.center_offset_m > 0 else "◀ LEFT"
    cc = "green" if avg_curv > 500 else ("yellow" if avg_curv > 200 else "red")
    oc = "green" if off_abs < 0.15 else ("yellow" if off_abs < 0.4 else "red")
    ct = f"{avg_curv:.0f} m" if avg_curv < 9000 else "Straight"
    c1, c2, c3 = st.columns(3)
    with c1:
        bc = "badge-ok" if info.lane_detected else "badge-err"
        bt = "DETECTED"     if info.lane_detected else "NOT DETECTED"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Lane Status</div>'
                    f'<div><span class="badge {bc}">{bt}</span></div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Road Curvature</div>'
                    f'<div class="metric-value {cc}">{ct}</div>'
                    f'<div class="metric-sub">radius of curve</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Lane Offset</div>'
                    f'<div class="metric-value {oc}">{off_abs:.2f} m</div>'
                    f'<div class="metric-sub">{dirn}</div></div>',
                    unsafe_allow_html=True)


def render_steps(pipeline):
    steps = [("1 · Original","original"),("2 · Color Mask","color_mask"),
             ("3 · Canny Edges","canny_edges"),("4 · ROI Binary","roi_binary"),
             ("5 · Bird's Eye","birds_eye"),("6 · Sliding Window","sliding_window"),
             ("7 · Final Result","final_result")]
    for i in range(0, len(steps), 3):
        cols = st.columns(3)
        for col, (lbl, key) in zip(cols, steps[i:i+3]):
            with col:
                st.image(bgr_to_rgb(pipeline[key]), use_container_width=True)
                st.markdown(f'<div class="step-label">{lbl}</div>', unsafe_allow_html=True)


def render_history(media_id):
    """Render saved results for a given media_id with run picker."""
    results = _db.get_results_for_media(media_id)
    if not results:
        st.info("No saved results yet. Run detection to generate a result.")
        return

    options = {f"Run #{r['id']} — {r['processed_at'][:16]}": dict(r) for r in results}
    sel_label = st.selectbox(
        f"Select run ({len(results)} saved)",
        list(options.keys()),
        key=f"history_sel_{media_id}",
    )
    r = options[sel_label]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Curvature", f"{r['avg_curvature_m']:.0f} m")
    c2.metric("Avg Offset",    f"{r['avg_offset_m']:.2f} m")
    c3.metric("Detected",      "Yes" if r["lane_detected"] else "No")
    c4.metric("Frames",        r["frames_processed"])

    if r["result_thumb"]:
        arr = np.frombuffer(r["result_thumb"], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            st.image(bgr_to_rgb(img), use_container_width=True)

    if r["params_json"]:
        p = json.loads(r["params_json"])
        st.caption(
            f"Params — Canny: {p.get('canny_lo',50)}/{p.get('canny_hi',150)} · "
            f"White L>{p.get('white_l',200)} · "
            f"Windows: {p.get('nwindows',9)}"
        )


def _save_result_button(media_id, info, frame_count=1):
    """Explicit save-to-history button so we don't spam DB on every slider touch."""
    if st.button("💾 Save to History", use_container_width=True):
        avg_c = (info.left_curvature_m + info.right_curvature_m) / 2
        rid   = _db.save_result(
            media_id=media_id,
            params=collect_params(),
            avg_curvature=avg_c,
            avg_offset=abs(info.center_offset_m),
            lane_detected=info.lane_detected,
            frames_processed=frame_count,
            result_thumb=b"",
        )
        st.success(f"Saved — result #{rid}")
        st.rerun()


# ── Video processing (stores result in session_state) ─────────────────────────

def process_video(input_path: str, params: dict, result_key: str,
                  media_id: int = None, media_name: str = ""):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = get_detector(width, height)

    tmp_out  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    tmp_out.close()
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    bar       = st.progress(0, text="Processing…")
    preview   = st.empty()
    key_frames, curvatures, offsets = [], [], []
    detected_ct = 0
    last_pipe   = None
    idx         = 0
    sample_every = max(1, total // 12)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pipe = run_pipeline(frame, detector, params)
        writer.write(pipe["final_result"])
        last_pipe = pipe
        info      = pipe["lane_info"]
        avg_c     = (info.left_curvature_m + info.right_curvature_m) / 2
        curvatures.append(avg_c)
        offsets.append(abs(info.center_offset_m))
        if info.lane_detected:
            detected_ct += 1
        if idx % sample_every == 0:
            key_frames.append(pipe["final_result"].copy())
        idx += 1
        bar.progress(min(int(idx / total * 100), 100), text=f"Frame {idx} / {total}")
        if idx % 12 == 0:
            preview.image(bgr_to_rgb(pipe["final_result"]),
                          caption=f"Live — frame {idx}", use_container_width=True)

    cap.release()
    writer.release()
    bar.progress(100, text="Done!")
    preview.empty()

    avg_curv   = float(np.mean(curvatures)) if curvatures else 0.0
    avg_offset = float(np.mean(offsets))    if offsets    else 0.0
    det_pct    = detected_ct / max(idx, 1) * 100

    with open(out_path, "rb") as f:
        video_bytes = f.read()

    db_result_id = None
    if media_id is not None and last_pipe is not None:
        thumb = frame_to_jpeg(last_pipe["final_result"])
        db_result_id = _db.save_result(
            media_id=media_id, params=collect_params(),
            avg_curvature=avg_curv, avg_offset=avg_offset,
            lane_detected=det_pct > 50, frames_processed=idx,
            result_thumb=thumb,
        )

    st.session_state["vid_result"] = {
        "key_frames":   key_frames,
        "avg_curv":     avg_curv,
        "avg_offset":   avg_offset,
        "det_pct":      det_pct,
        "frames":       idx,
        "last_pipe":    last_pipe,
        "video_bytes":  video_bytes,
        "media_id":     media_id,
        "media_name":   media_name,
        "db_result_id": db_result_id,
        "result_key":   result_key,
    }


def render_video_result():
    """Render the stored video result from session_state."""
    r = st.session_state.get("vid_result")
    if not r:
        return False

    avg_c   = r["avg_curv"]
    off_abs = r["avg_offset"]
    det_pct = r["det_pct"]
    frames  = r["frames"]
    dirn    = "RIGHT ▶" if r["avg_offset"] > 0 else "◀ LEFT"
    cc      = "green" if avg_c > 500 else ("yellow" if avg_c > 200 else "red")
    oc      = "green" if off_abs < 0.15 else ("yellow" if off_abs < 0.4 else "red")

    c1, c2, c3 = st.columns(3)
    with c1:
        bc = "badge-ok" if det_pct > 50 else "badge-err"
        bt = f"DETECTED ({det_pct:.0f}%)"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Lane Status</div>'
                    f'<div><span class="badge {bc}">{bt}</span></div></div>',
                    unsafe_allow_html=True)
    with c2:
        ct = f"{avg_c:.0f} m" if avg_c < 9000 else "Straight"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Curvature</div>'
                    f'<div class="metric-value {cc}">{ct}</div>'
                    f'<div class="metric-sub">over {frames} frames</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Lane Offset</div>'
                    f'<div class="metric-value {oc}">{off_abs:.2f} m</div>'
                    f'<div class="metric-sub">{dirn}</div></div>',
                    unsafe_allow_html=True)

    st.markdown(f"**{frames} frames processed** · {det_pct:.0f}% with lane detected")

    if r.get("db_result_id"):
        st.success(f"✅ Result saved to library (result #{r['db_result_id']})")

    if r["key_frames"]:
        st.markdown("#### Key Frames Preview")
        for row_start in range(0, len(r["key_frames"]), 4):
            row = r["key_frames"][row_start:row_start+4]
            cols = st.columns(4)
            for col, kf in zip(cols, row):
                col.image(bgr_to_rgb(kf), use_container_width=True)

    dl_name = f"lane_detection_{r['media_name'] or 'output'}.mp4"
    st.download_button("⬇️  Download Processed Video",
                       data=r["video_bytes"], file_name=dl_name,
                       mime="video/mp4", use_container_width=True)
    st.caption("Open the downloaded file with VLC, QuickTime, or Windows Media Player.")

    if r["last_pipe"]:
        st.divider()
        st.markdown("#### Pipeline Steps — Last Frame")
        render_steps(r["last_pipe"])

    return True


# Seed demo videos once per session (downloads any missing clips silently)
if "demos_seeded" not in st.session_state:
    _ensure_demo_videos_in_db()
    st.session_state["demos_seeded"] = True

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛣️ Lane Detection")
    st.caption("HIT AV Course · 2026 · Roy Carter")
    st.divider()

    st.subheader("Input Source")
    source = st.radio("source",
                      ["Upload to Library", "Browse Library"],
                      label_visibility="collapsed")
    st.divider()

    uploaded_file = None
    upload_name   = ""

    if source == "Upload to Library":
        st.subheader("Upload File")
        uploaded_file = st.file_uploader(
            "Image or Video",
            type=["jpg","jpeg","png","bmp","mp4","avi","mov","mkv"],
        )
        if uploaded_file:
            upload_name = st.text_input("Name / Label",
                                        value=uploaded_file.name.rsplit(".", 1)[0])

    elif source == "Browse Library":
        st.subheader("Media Library")
        items = _db.get_all_media()
        if items:
            options = {
                f"{item['name']} ({'video' if item['media_type']=='video' else 'image'}) — {item['uploaded_at'][:16]}":
                item["id"] for item in items
            }
            sel_label = st.selectbox("Select item", list(options.keys()))
            new_sel   = options[sel_label]
            if st.session_state.get("selected_media_id") != new_sel:
                st.session_state.pop("vid_result", None)
            st.session_state["selected_media_id"] = new_sel
            if st.button("🗑️ Delete selected", use_container_width=True):
                _db.delete_media(new_sel)
                st.session_state.pop("selected_media_id", None)
                st.session_state.pop("vid_result", None)
                st.rerun()
        else:
            st.info("Library is empty.")


    st.divider()

    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.caption("Tune detection thresholds live")
        st.slider("White brightness (L)",  100, 255, 200, key="white_l")
        st.markdown("**Yellow hue range**")
        yc1, yc2 = st.columns(2)
        yc1.slider("Min", 0, 50, 15, key="yh_min")
        yc2.slider("Max", 0, 50, 35, key="yh_max")
        st.slider("Canny low",          0, 300,  50, key="canny_lo")
        st.slider("Canny high",         0, 300, 150, key="canny_hi")
        st.slider("Smoothing (frames)", 1,  20,   8, key="smooth_win")
        st.slider("Sliding windows",    3,  15,   9, key="nwindows")
        st.slider("Window margin",     50, 200, 100, key="margin")
        st.slider("Min lane pixels",   10, 200,  50, key="minpix")

    st.divider()
    st.subheader("Visualization Mode")
    st.radio("viz_mode", VIZ_LABELS, index=0, label_visibility="collapsed", key="viz_mode")

    st.divider()
    stats = _db.db_stats()
    st.caption(f"📦 Library: {stats['media']} files · {stats['results']} results")


# ── Main tabs ─────────────────────────────────────────────────────────────────

params = collect_params()
tab_detect, tab_steps, tab_history = st.tabs(["🎯 Detection", "🔬 Pipeline Steps", "📊 Results History"])

pipeline   = None
history_id = None


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD TO LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
if source == "Upload to Library":
    if uploaded_file is None:
        with tab_detect:
            st.info("Choose a file in the sidebar to upload and process.")
        with tab_steps:
            st.info("Upload a file first.")
        with tab_history:
            st.info("No item selected yet.")
    else:
        file_bytes = uploaded_file.getvalue()
        ext        = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
        is_video   = ext in (".mp4", ".avi", ".mov", ".mkv")
        media_type = "video" if is_video else "image"

        cache_key = f"upload_{uploaded_file.name}_{len(file_bytes)}"
        if st.session_state.get("last_upload_key") != cache_key:
            media_id = _db.save_media(upload_name or uploaded_file.name,
                                      media_type, file_bytes, ext)
            st.session_state.update({
                "last_upload_key":   cache_key,
                "last_upload_id":    media_id,
                "last_upload_bytes": file_bytes,
                "last_upload_ext":   ext,
                "last_upload_type":  media_type,
            })
            st.session_state.pop("vid_result", None)

        media_id   = st.session_state["last_upload_id"]
        file_bytes = st.session_state["last_upload_bytes"]
        ext        = st.session_state["last_upload_ext"]
        media_type = st.session_state["last_upload_type"]
        history_id = media_id
        result_key = f"upload_{media_id}"

        with tab_detect:
            st.success(f"✅ Saved to library — **{upload_name or uploaded_file.name}** (ID: {media_id})")
            st.divider()

            if media_type == "video":
                stored = st.session_state.get("vid_result")
                if stored and stored.get("result_key") == result_key:
                    render_video_result()
                    if st.button("🔄 Re-run with current settings", use_container_width=True):
                        st.session_state.pop("vid_result", None)
                        st.rerun()
                else:
                    st.markdown("### Ready to process")
                    if media_row_thumb := _db.get_media(media_id):
                        if media_row_thumb["thumbnail"]:
                            arr = np.frombuffer(media_row_thumb["thumbnail"], dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                st.image(bgr_to_rgb(img), caption="Preview (first frame)", width=480)
                    if st.button("▶  Run Detection", use_container_width=True, type="primary"):
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        process_video(tmp_path, params, result_key,
                                      media_id=media_id, media_name=upload_name)
                        st.rerun()
            else:
                arr      = np.frombuffer(file_bytes, dtype=np.uint8)
                frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                h, w     = frame.shape[:2]
                detector = get_detector(w, h)
                pipeline = run_pipeline(frame, detector, params)
                viz_key  = VIZ_KEYS.get(st.session_state.get("viz_mode","Final Result"), "final_result")

                render_metrics(pipeline["lane_info"])
                st.divider()
                col_in, col_out = st.columns(2)
                with col_in:
                    st.caption("📷 Input")
                    st.image(bgr_to_rgb(pipeline["original"]), use_container_width=True)
                with col_out:
                    st.caption(f"🔍 {st.session_state.get('viz_mode','Final Result')}")
                    st.image(bgr_to_rgb(pipeline[viz_key]), use_container_width=True)
                _save_result_button(media_id, pipeline["lane_info"])

        with tab_steps:
            if pipeline:
                render_steps(pipeline)
            else:
                stored = st.session_state.get("vid_result")
                if stored and stored.get("result_key") == result_key and stored.get("last_pipe"):
                    render_steps(stored["last_pipe"])
                else:
                    st.info("Run detection first.")

        with tab_history:
            render_history(history_id)


# ══════════════════════════════════════════════════════════════════════════════
# BROWSE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
elif source == "Browse Library":
    selected_id = st.session_state.get("selected_media_id")
    media_row   = _db.get_media(selected_id) if selected_id else None
    history_id  = selected_id

    if media_row is None:
        with tab_detect:
            st.info("Select an item from the sidebar.")
        with tab_steps:
            st.info("Select an item first.")
        with tab_history:
            st.info("Select an item first.")
    else:
        file_bytes = bytes(media_row["file_data"])
        ext        = media_row["file_ext"]
        media_type = media_row["media_type"]
        name       = media_row["name"]
        result_key = f"browse_{selected_id}"

        with tab_detect:
            tag = "🎬" if media_type == "video" else "🖼️"
            st.markdown(f"### {tag} {name}")
            st.caption(f"Added: {media_row['uploaded_at'][:16]}  ·  Type: {media_type}")
            st.divider()

            if media_type == "video":
                stored = st.session_state.get("vid_result")
                if stored and stored.get("result_key") == result_key:
                    render_video_result()
                    if st.button("🔄 Re-run with current settings", use_container_width=True):
                        st.session_state.pop("vid_result", None)
                        st.rerun()
                else:
                    if media_row["thumbnail"]:
                        arr = np.frombuffer(media_row["thumbnail"], dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            st.image(bgr_to_rgb(img), caption="Preview (first frame)", width=640)
                    st.info("Click **Run Detection** to process this video.")
                    if st.button("▶  Run Detection", use_container_width=True, type="primary"):
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        process_video(tmp_path, params, result_key,
                                      media_id=selected_id, media_name=name)
                        st.rerun()
            else:
                arr      = np.frombuffer(file_bytes, dtype=np.uint8)
                frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                h, w     = frame.shape[:2]
                detector = get_detector(w, h)
                pipeline = run_pipeline(frame, detector, params)
                viz_key  = VIZ_KEYS.get(st.session_state.get("viz_mode","Final Result"), "final_result")

                render_metrics(pipeline["lane_info"])
                st.divider()
                col_in, col_out = st.columns(2)
                with col_in:
                    st.caption("📷 Input")
                    st.image(bgr_to_rgb(pipeline["original"]), use_container_width=True)
                with col_out:
                    st.caption(f"🔍 {st.session_state.get('viz_mode','Final Result')}")
                    st.image(bgr_to_rgb(pipeline[viz_key]), use_container_width=True)
                _save_result_button(selected_id, pipeline["lane_info"])

        with tab_steps:
            if pipeline:
                render_steps(pipeline)
            else:
                stored = st.session_state.get("vid_result")
                if stored and stored.get("result_key") == result_key and stored.get("last_pipe"):
                    render_steps(stored["last_pipe"])
                else:
                    st.info("Run detection first.")

        with tab_history:
            render_history(selected_id)
