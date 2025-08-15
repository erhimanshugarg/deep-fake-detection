import streamlit as st
import tempfile
import time
import cv2
from datetime import datetime
from liveness_utils import load_all_models, run_inference, log_session, MODEL_EMOTIONS

import streamlit as st
st.set_page_config(page_title="Real-Time Liveness Verification", layout="wide")

# ---- Constants ----
CAPTURE_DURATION_SECS = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# st.set_page_config(page_title="Real-Time Liveness Detection", layout="wide")
st.title("🎥 Real-Time Liveness Detection")

# ---- Cache model loading ----
@st.cache_resource
def load_models():
    return load_all_models()

deepfake_model, micro_model, rf_clf, xgb_clf, scaler = load_models()

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames <= 0:
        return []
    step = max(1, total_frames // max_frames)
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        if len(frames) == max_frames:
            break
    cap.release()
    while len(frames) < max_frames:
        frames.append(frames[-1])
    return frames

# ---- Session state init ----
for key, default in [
    ("video_path", None),
    ("video_captured", False),
    ("capturing", False),
    ("analysis_done", False),
    ("results", None),
    ("last_mode", None),
    ("analysis_started", False),
    ("show_time", False),
    ("uploaded_sig", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Sidebar ----
mode = st.sidebar.radio("Select Input Method", ["Upload Videos", "Live / Webcam"])

# ---- Reset state when switching modes ----
if st.session_state.last_mode is None:
    st.session_state.last_mode = mode

if mode != st.session_state.last_mode:
    st.session_state.video_path = None
    st.session_state.video_captured = False
    st.session_state.capturing = False
    st.session_state.analysis_done = False
    st.session_state.analysis_started = False
    st.session_state.results = None
    st.session_state.last_mode = mode

# ---------------- Upload Videos ----------------
if mode == "Upload Videos":
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video", type=None)

    if uploaded_file:
        # Only initialize when a new file is uploaded (avoid resetting flags on rerun)
        size_attr = getattr(uploaded_file, "size", None)
        file_sig = f"{uploaded_file.name}:{size_attr}"
        if st.session_state.uploaded_sig != file_sig:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            st.session_state.video_path = tfile.name
            st.session_state.video_captured = True
            st.session_state.analysis_done = False
            st.session_state.analysis_started = False
            st.session_state.results = None
            st.session_state.uploaded_sig = file_sig

        # Show uploaded video preview in player
        if st.session_state.video_path:
            st.video(st.session_state.video_path)

        if not st.session_state.analysis_done:
            if not st.session_state.analysis_started:
                if st.button("📊 Start Analysis", key="start_analysis_upload"):
                    st.session_state.analysis_started = True
                    st.rerun()
            else:
                with st.spinner("🔍 Running analysis for KYC verification..."):
                    frames = extract_frames(st.session_state.video_path)
                    st.session_state.results = run_inference(
                        deepfake_model, micro_model, rf_clf, xgb_clf, scaler, frames
                    )
                st.session_state.analysis_done = True
                st.rerun()

# ---------------- Live / Webcam ----------------
elif mode == "Live / Webcam":
    st.header("WebCam / Live")

    if not st.session_state.capturing and not st.session_state.video_captured:
        if st.button("🎥 Start Camera"):
            st.session_state.capturing = True
            st.session_state.analysis_done = False
            st.session_state.analysis_started = False
            st.session_state.results = None
            st.session_state.video_path = None
            st.session_state.show_time = True
            st.rerun()

    if st.session_state.capturing:
        preview_window = st.empty()
        progress_placeholder = st.empty()
        frames = []

        # Initial spinner to indicate camera starting
        with st.spinner("🎬 Starting the camera..."):
            time.sleep(0.8)

        cap = cv2.VideoCapture(0)
        start_time = time.time()

        # Progress bar to simulate spinner while allowing live preview
        progress_bar = progress_placeholder.progress(0, text="⏳ Capturing video...")

        while time.time() - start_time < CAPTURE_DURATION_SECS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            preview_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Preview")
            elapsed = time.time() - start_time
            remaining = CAPTURE_DURATION_SECS - int(elapsed)
            percent = min(100, int((elapsed / CAPTURE_DURATION_SECS) * 100))
            progress_bar.progress(percent, text=f"⏳ Capturing video... {remaining} seconds left")
            frames.append(frame)

        cap.release()
        preview_window.empty()
        progress_placeholder.empty()

        # Save video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_path = temp_video.name
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (VIDEO_WIDTH, VIDEO_HEIGHT))
        for f in frames:
            out.write(f)
        out.release()
        temp_video.close()

        st.session_state.video_path = video_path
        st.session_state.video_captured = True
        st.session_state.capturing = False
        st.session_state.show_time = False
        st.rerun()

    if st.session_state.video_captured and not st.session_state.analysis_done:
        with st.spinner("⏳ Please wait..."):
            time.sleep(2.0)
        st.success("✅ Video is captured successfully !!")

        if not st.session_state.analysis_done:
            if not st.session_state.analysis_started:
                if st.button("📊 Start Analysis", key="start_analysis_webcam"):
                    st.session_state.analysis_started = True
                    st.rerun()
            else:
                with st.spinner("🔍 Running analysis for KYC verification..."):
                    frames = extract_frames(st.session_state.video_path)
                    st.session_state.results = run_inference(
                        deepfake_model, micro_model, rf_clf, xgb_clf, scaler, frames
                    )
                st.session_state.analysis_done = True
                st.rerun()

# ---------------- Results Section ----------------
if st.session_state.analysis_done and st.session_state.results:
    res = st.session_state.results
    st.markdown("### 🧪 KYC Liveness Analysis Results")

    # Deepfake-related stats (if available)
    st.markdown("#### 🎭 Deepfake Stats")
    if 'deep_score' in res and res['deep_score'] is not None:
        st.markdown(f"- Deepfake score: `{res['deep_score']:.4f}`")
    # Print any other deepfake-related keys for transparency
    for k, v in res.items():
        if k.startswith('deep_') and k not in ['deep_score']:
            try:
                st.markdown(f"- {k}: `{float(v):.4f}`")
            except Exception:
                st.markdown(f"- {k}: `{v}`")

    # Microexpression probabilities
    st.markdown("#### 😐 Micro-expression Probabilities")
    if 'micro_probs' in res and res['micro_probs'] is not None:
        for i, emo in enumerate(MODEL_EMOTIONS):
            try:
                st.markdown(f"- **{emo}**: {res['micro_probs'][i]:.4f}")
            except Exception:
                pass

    # Fusion summary
    st.markdown("#### 🔗 Fusion Decision")
    st.markdown(
        f"""
        - Decision: `{res['fusion_label']}`  
        - Confidence: `{res['confidence_score']:.4f}`  
        - Tier: `{res['confidence_tier']}`
        """
    )

    if res['fusion_label'].upper() == "LIVE":
        st.success("🟢 Final Decision: LIVE")
    else:
        st.error("🔴 Final Decision: SPOOF")

    log_session(
        timestamp=datetime.now().isoformat(),
        fusion_label=res['fusion_label'],
        conf_tier=res['confidence_tier'],
        conf_score=res['confidence_score'],
        deep_score=res.get('deep_score', None),
        micro_probs=res.get('micro_probs', None)
    )
