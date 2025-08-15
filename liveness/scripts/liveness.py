import streamlit as st
import tempfile
import time
import cv2
import os
from datetime import datetime
from streamlit_lottie import st_lottie
from liveness_utils import load_all_models, run_inference, log_session, MODEL_EMOTIONS

# ---- Page config & title ----
st.set_page_config(
    page_title="Real-Time Liveness Verification",
    page_icon="🛡️",
    layout="centered"
)
st.title("🎥 Real-Time Liveness Detection")

# ---- Custom CSS for blocking loader ----
CUSTOM_CSS = """
<style>
.loader-backdrop {
    position: fixed;
    top: 0; left: 0;
    height: 100vh; width: 100vw;
    z-index: 9999;
    background: rgba(20, 40, 56, 0.65);
    display: flex; align-items: center; justify-content: center;
}
.loader-backdrop .loader {
    border: 10px solid #38d9e7;
    border-top: 10px solid #152238;
    border-radius: 50%;
    width: 72px;
    height: 72px;
    animation: spin 1s linear infinite;
    margin: auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg);}
}
.loader-backdrop .text {
    color: #f3f7fa;
    font-size: 1.4rem;
    margin-top: 24px;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def show_fullscreen_loader(text="Processing..."):
    return f"""
        <div class="loader-backdrop">
            <div>
                <div class="loader"></div>
                <div class="text">{text}</div>
            </div>
        </div>
    """

@st.cache_resource
def load_models():
    return load_all_models()
deepfake_model, micro_model, rf_clf, xgb_clf, scaler = load_models()

CAPTURE_DURATION_SECS = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Component decision thresholds
DEEP_LIVE_THRESHOLD = 0.50         # if deep_score >= threshold => LIVE else SPOOF
MICRO_TOP_PROB_THRESHOLD = 0.50    # if max(micro_probs) >= threshold => PASS else FAIL


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
    while frames and len(frames) < max_frames:
        frames.append(frames[-1])
    return frames

def load_lottiefile(filepath: str):
    import json
    with open(filepath, "r") as f:
        return json.load(f)

def show_results(res):
    import pandas as pd
    st.markdown("### 🧪 KYC Liveness Analysis Results")
    with st.expander("🎭 Deepfake Stats", expanded=True):
        st.info(
            "Deepfake Stats: These metrics indicate how real or synthetic the video appears. Higher scores generally suggest authenticity.",
            icon="ℹ️"
        )
        deep_rows = []
        deep_score = res.get("deep_score", None)
        if deep_score is not None:
            try:
                deep_rows.append({"Metric": "Deepfake score", "Value": f"{float(deep_score):.4f}"})
            except Exception:
                deep_rows.append({"Metric": "Deepfake score", "Value": f"{deep_score}"})
        deep_probs = res.get("deep_pred_probs") or res.get("deep_probs")
        if deep_probs is not None:
            for i, p in enumerate(deep_probs):
                try:
                    deep_rows.append({"Metric": f"deep_pred_probs[{i}]", "Value": f"{float(p):.4f}"})
                except Exception:
                    deep_rows.append({"Metric": f"deep_pred_probs[{i}]", "Value": f"{p}"})
        for k, v in res.items():
            if k.startswith("deep_") and k not in ["deep_score", "deep_pred_probs", "deep_probs"]:
                try:
                    deep_rows.append({"Metric": k, "Value": f"{float(v):.4f}"})
                except Exception:
                    deep_rows.append({"Metric": k, "Value": f"{v}"})
        if deep_rows:
            st.table(pd.DataFrame(deep_rows))
        # Deepfake decision in block
        if deep_score is not None:
            try:
                ds = float(deep_score)
                deep_verdict = "LIVE" if ds >= DEEP_LIVE_THRESHOLD else "SPOOF"
                if deep_verdict == "LIVE":
                    st.success(f"Deepfake Decision: {deep_verdict} (score={ds:.4f}, thr={DEEP_LIVE_THRESHOLD:.2f})")
                else:
                    st.error(f"Deepfake Decision: {deep_verdict} (score={ds:.4f}, thr={DEEP_LIVE_THRESHOLD:.2f})")
            except Exception:
                st.warning(f"Deepfake Decision: unavailable (unparsable score: {deep_score})")

    with st.expander("😐 Micro‑expression Probabilities", expanded=True):
        st.info(
            "Micro-expressions: Probabilities of facial emotions detected from the video.",
            icon="ℹ️"
        )
        micro = res.get("micro_probs")
        if micro is not None:
            micro_rows = []
            top_prob = None
            top_emo = None
            for i, emo in enumerate(MODEL_EMOTIONS):
                val = None
                if i < len(micro):
                    try:
                        prob_val = float(micro[i])
                        val = f"{float(micro[i]):.4f}"
                        if top_prob is None or prob_val > top_prob:
                            top_prob = prob_val
                            top_emo = emo
                    except Exception:
                        val = f"{micro[i]}"
                micro_rows.append({"Emotion": emo, "Probability": val if val is not None else "N/A"})
                lottie_path = f"assets/lottie/{emo}.json"
                try:
                    lottie_json = load_lottiefile(lottie_path)
                    st_lottie(lottie_json, speed=1, loop=True, height=45)
                except Exception:
                    pass
            st.table(pd.DataFrame(micro_rows))

            # Micro-expression decision in block
            if top_prob is not None:
                micro_verdict = "PASS" if top_prob >= MICRO_TOP_PROB_THRESHOLD else "FAIL"
                if micro_verdict == "PASS":
                    st.success(
                        f"Micro‑expression Decision: {micro_verdict} (top={top_emo}, p={top_prob:.4f}, thr={MICRO_TOP_PROB_THRESHOLD:.2f})")
                else:
                    st.error(
                        f"Micro‑expression Decision: {micro_verdict} (top={top_emo}, p={top_prob:.4f}, thr={MICRO_TOP_PROB_THRESHOLD:.2f})")
            else:
                st.warning("Micro‑expression Decision: unavailable (no valid probabilities)")

    with st.expander("🔗 Fusion Decision", expanded=True):
        st.info(
            "Fusion Decision combines all model outputs to give a final liveness verdict.",
            icon="ℹ️"
        )
        decision = res.get("fusion_label", "N/A")
        conf_val = res.get("confidence_score", None)
        try:
            conf_str = f"{float(conf_val):.4f}" if conf_val is not None else "N/A"
        except Exception:
            conf_str = f"{conf_val}" if conf_val is not None else "N/A"
        tier = res.get("confidence_tier", "N/A")
        fusion_rows = [
            {"Field": "Decision", "Value": f"{decision}"},
            {"Field": "Confidence", "Value": conf_str},
            {"Field": "Tier", "Value": f"{tier}"},
        ]
        st.table(pd.DataFrame(fusion_rows))
        if str(res.get("fusion_label", "")).upper() == "LIVE":
            st.success("🟢 Final Decision: LIVE")
        else:
            st.error("🔴 Final Decision: SPOOF")

defaults = {
    "upload_video_path": None,
    "upload_uploaded_sig": None,
    "upload_analysis_started": False,
    "upload_analysis_done": False,
    "upload_results": None,
    "live_video_path": None,
    "live_video_captured": False,
    "live_capturing": False,
    "live_analysis_started": False,
    "live_analysis_done": False,
    "live_results": None,
    "live_show_time": False,
    "start_video_capture": False,
    "live_stats_ready": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.container(border=True, width="stretch"):
    tab_upload, tab_live = st.tabs(["📤 Upload Video", "🎥 Live/Webcam"])

    # --------- Upload Video Tab ----------
    with tab_upload:
        st.header("Upload Video")
        uploaded_file = st.file_uploader("Choose a video", type=None)
        if uploaded_file:
            size_attr = getattr(uploaded_file, "size", None)
            file_sig = f"{uploaded_file.name}:{size_attr}"
            if st.session_state.upload_uploaded_sig != file_sig:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                st.session_state.upload_video_path = tfile.name
                st.session_state.upload_analysis_started = False
                st.session_state.upload_analysis_done = False
                st.session_state.upload_results = None
                st.session_state.upload_uploaded_sig = file_sig
                st.query_params["dummy"] = str(time.time())
                st.rerun()

        # Always show the video preview if a video path is set
        if st.session_state.upload_video_path:
            st.video(st.session_state.upload_video_path)

        # If analysis not done, run analysis automatically and show loader
        if st.session_state.upload_video_path and not st.session_state.upload_analysis_done:
            loader_holder = st.empty()
            loader_holder.markdown(show_fullscreen_loader("🔍 Running analysis for KYC verification..."),
                                   unsafe_allow_html=True)
            frames = extract_frames(st.session_state.upload_video_path)
            st.session_state.upload_results = run_inference(
                deepfake_model, micro_model, rf_clf, xgb_clf, scaler, frames
            )
            loader_holder.empty()
            st.session_state.upload_analysis_done = True
            st.query_params["dummy"] = str(time.time())
            st.rerun()

        # Once analysis is done, show results and reset option
        if st.session_state.upload_analysis_done and st.session_state.upload_results:
            res = st.session_state.upload_results
            show_results(res)
            log_session(
                timestamp=datetime.now().isoformat(),
                fusion_label=res.get('fusion_label'),
                conf_tier=res.get('confidence_tier'),
                conf_score=res.get('confidence_score'),
                deep_score=res.get('deep_score'),
                micro_probs=res.get('micro_probs')
            )

    # --------- Live/Webcam Tab ----------
    with tab_live:
        left_h, right_h = st.columns([0.7, 0.3])
        with left_h:
            st.header("WebCam / Live")
        with right_h:
            # Show Restart button only once stats/results are available
            if st.session_state.get("live_stats_ready"):
                if st.button("🔄 Restart"):
                    # Reset all webcam and result state
                    for k in [
                        "live_video_path", "live_video_captured", "live_capturing",
                        "live_analysis_started", "live_analysis_done", "live_results",
                        "start_video_capture", "live_stats_ready", "captured_frames"
                    ]:
                        if k in st.session_state:
                            st.session_state[k] = defaults.get(k, None)
                    st.query_params.clear()
                    st.rerun()

        # Step 1: Show loader briefly when "Start Camera" clicked, then trigger actual capture
        if st.session_state.live_capturing:
            loader_holder = st.empty()
            loader_holder.markdown(show_fullscreen_loader("🎬 Initializing camera ..."), unsafe_allow_html=True)
            time.sleep(0.6)
            loader_holder.empty()
            st.session_state.live_capturing = False
            st.session_state.start_video_capture = True
            st.query_params["dummy"] = str(time.time())
            st.rerun()

        # Step 2: Actual camera capture
        if st.session_state.get("start_video_capture", False):
            cap = cv2.VideoCapture(0)
            start_time = time.time()
            frames = []
            preview_window = st.empty()
            progress_placeholder = st.empty()
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
            st.session_state.captured_frames = frames
            st.session_state.live_video_captured = True
            st.session_state.start_video_capture = False
            st.session_state.live_stats_ready = False
            st.query_params["dummy"] = str(time.time())
            st.rerun()

        # Step 3: After capture, loader while analyzing, show stats, provide restart
        if st.session_state.live_video_captured and not st.session_state.live_stats_ready:
            loader_holder = st.empty()
            loader_holder.markdown(show_fullscreen_loader("Processing video, please wait..."), unsafe_allow_html=True)
            frames = st.session_state.get("captured_frames") or []
            # Save captured frames to video file for results consistency
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = temp_video.name
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (VIDEO_WIDTH, VIDEO_HEIGHT))
            for f in frames:
                out.write(f)
            out.release()
            temp_video.close()
            st.session_state.live_video_path = video_path
            st.session_state.live_results = run_inference(
                deepfake_model, micro_model, rf_clf, xgb_clf, scaler, frames
            )
            loader_holder.empty()
            st.session_state.live_stats_ready = True
            st.query_params["dummy"] = str(time.time())
            st.rerun()

        # After webcam video capture, before showing stats/results:
        if st.session_state.live_video_captured and st.session_state.live_stats_ready and st.session_state.live_results:
            st.success("✅ Video is captured successfully !!")
            res = st.session_state.live_results
            show_results(res)
            log_session(
                timestamp=datetime.now().isoformat(),
                fusion_label=res.get('fusion_label'),
                conf_tier=res.get('confidence_tier'),
                conf_score=res.get('confidence_score'),
                deep_score=res.get('deep_score'),
                micro_probs=res.get('micro_probs')
            )

        # Start Camera button appears only if ready for new capture
        if not st.session_state.live_capturing and not st.session_state.live_video_captured and not st.session_state.start_video_capture:
            if st.button("🎥 Start Camera"):
                st.session_state.live_capturing = True
                st.session_state.live_analysis_started = False
                st.session_state.live_analysis_done = False
                st.session_state.live_results = None
                st.session_state.live_video_path = None
                st.session_state.live_show_time = True
                st.query_params["dummy"] = str(time.time())
                st.rerun()
