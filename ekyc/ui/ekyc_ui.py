import streamlit as st
import requests
import tempfile
import os

API_URL = os.environ.get("EKYC_API_URL", "http://127.0.0.1:8000/liveness_check")
HEALTH_URL = API_URL.replace("/liveness_check", "/health")

st.set_page_config(page_title="eKYC Liveness Verification", layout="centered")
st.title("🔍 eKYC Liveness Verification")
st.caption(f"API endpoint: {API_URL}")

# Optional health check
try:
    r = requests.get(HEALTH_URL, timeout=3)
    if r.ok:
        st.success("API online")
    else:
        st.warning(f"API health check failed: {r.status_code}")
except Exception as e:
    st.warning(f"API unreachable: {e}")

user_id = st.text_input("User ID", value="test_user_001")
uploaded_file = st.file_uploader("Upload a face video for liveness check", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

if st.button("Run Liveness Check"):
    if not uploaded_file:
        st.error("Please upload a video first.")
    elif not user_id.strip():
        st.error("Please enter a user ID.")
    else:
        temp_path = None
        with st.spinner("Uploading and running prediction..."):
            try:
                suffix = os.path.splitext(uploaded_file.name or "")[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
                    temp_path = tmpf.name
                    tmpf.write(uploaded_file.getbuffer())

                # Use with so file is closed before API call to avoid Windows lock
                with open(temp_path, "rb") as f:
                    files = {"file": (os.path.basename(temp_path), f, "video/mp4")}
                    data = {"user_id": user_id}
                    response = requests.post(API_URL, files=files, data=data)
                    try:
                        result = response.json()
                    except Exception:
                        st.error(f"Invalid API response (status {response.status_code}): {response.text[:500]}")
                        st.stop()

                    if result.get("status") != "ok":
                        st.error(f"API error: {result.get('message', 'Unknown')}")
                        st.stop()

                content_type = response.headers.get("Content-Type", "").lower()
                if "application/json" not in content_type:
                    st.error(f"Invalid API response (not JSON): {response.text[:500]}")
                    st.stop()

                result = response.json()
                if response.status_code != 200 or result.get("status") != "ok":
                    st.error(f"API error: {result.get('message', 'Unknown error')}")
                    st.stop()

                st.subheader(f"Fusion Decision: {result['fusion_label']} ({result['confidence_tier']})")
                try:
                    st.write(f"Confidence Score: {float(result['confidence_score']):.2%}")
                except Exception:
                    st.write(f"Confidence Score: {result.get('confidence_score')}")

                st.write(f"Deep Score: {result.get('deep_score')}")
                st.write(f"Deepfake Model Probs: {result.get('deep_pred_probs')}")
                st.write(f"Micro-expression Probs: {result.get('micro_probs')}")

                if result["fusion_label"] == "LIVE":
                    st.success("✅ LIVE - User passed liveness check")
                else:
                    st.error("❌ SPOOF detected")

            except requests.RequestException as e:
                st.error(f"Error calling API: {e}")
            except ValueError as e:
                st.error(f"Invalid JSON from API: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
