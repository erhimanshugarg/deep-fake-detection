# liveness/app.py

import streamlit as st
import tempfile
import os
import shutil
import time
import cv2
from inference import predict_liveness_from_frames, preprocess_video

st.set_page_config(page_title="Liveness Detection", layout="centered")

st.title("🔍 Liveness Detection System")
st.markdown("Analyze videos using **Deepfake Detection** and **Microexpression Analysis**")

input_type = st.radio("Choose input method:", ["Upload Video", "Webcam"])

video_path = None
temp_dir = "temp_input"

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)

if input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(tfile.name)

        if st.button("📊 Analyze Video"):
            with st.spinner("📦 Extracting frames..."):
                preprocess_video(video_path, temp_dir)
                st.success("✅ Frames extracted.")

            with st.spinner("🔍 Running prediction..."):
                try:
                    results = predict_liveness_from_frames(temp_dir)
                    st.success("✅ Prediction complete")

                    st.markdown("### 🧪 Prediction Results")
                    st.markdown(f"""
                    🎭 **Deepfake Detection**: `{results['deep_class']}`  
                    - Confidence: `{results['deep_confidence']:.4f}`  

                    😐 **Micro-expression**: `{results['micro_class']}`  
                    - Confidence: `{results['micro_confidence']:.4f}`  

                    🔗 **Fusion Decision**: `{results['fusion_result']}`  
                    - Confidence: `{results['confidence']:.4f}`
                    """)

                    # Final decision highlight
                    if results["fusion_result"] == "LIVE":
                        st.success("🟢 Final Decision: ✅ LIVE")
                    else:
                        st.error("🔴 Final Decision: ❌ SPOOF")

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

elif input_type == "Webcam":
    st.warning("📷 Starting webcam. Press 'q' in the preview window to stop.")

    if st.button("🎥 Start Webcam Capture"):
        cap = cv2.VideoCapture(0)
        saved = 0
        max_frames = 60
        interval = 5
        frame_id = 0

        st.info("📸 Capturing frames from webcam...")
        while cap.isOpened() and saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % interval == 0:
                frame_path = os.path.join(temp_dir, f"frame_{saved:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            frame_id += 1

            cv2.imshow("Webcam - Press 'q' to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success(f"✅ Saved {saved} frames from webcam")

        with st.spinner("🔍 Running prediction..."):
            try:
                results = predict_liveness_from_frames(temp_dir)
                st.success("✅ Prediction complete")

                st.markdown("### 🧪 Prediction Results")
                st.markdown(f"""
                🎭 **Deepfake Detection**: `{results['deep_class']}`  
                - Confidence: `{results['deep_confidence']:.4f}`  

                😐 **Micro-expression**: `{results['micro_class']}`  
                - Confidence: `{results['micro_confidence']:.4f}`  

                🔗 **Fusion Decision**: `{results['fusion_result']}`  
                - Confidence: `{results['confidence']:.4f}`
                """)

                if results["fusion_result"] == "LIVE":
                    st.success("🟢 Final Decision: ✅ LIVE")
                else:
                    st.error("🔴 Final Decision: ❌ SPOOF")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
