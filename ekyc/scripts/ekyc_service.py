import os
import shutil
import tempfile
import traceback
import cv2
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from liveness.scripts.liveness_utils import load_all_models, run_inference, log_session

app = FastAPI(title="eKYC Liveness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production to your UI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.on_event("startup")
def load_models():
    print("Loading models at startup...")
    # print(f"Loading deepfake model from {DEEPFAKE_MODEL_PATH}")
    deepfake_model, micro_model, rf_clf, xgb_clf, scaler = load_all_models()
    app.state.deepfake_model = deepfake_model
    app.state.micro_model = micro_model
    app.state.rf_clf = rf_clf
    app.state.xgb_clf = xgb_clf
    app.state.scaler = scaler
    print("Models loaded successfully.")

def to_float_or_orig(x):
    try:
        return float(x)
    except Exception:
        return x

def to_list_or_orig(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [to_float_or_orig(v) for v in x]
    return x

@app.post("/liveness_check")
async def liveness_check(file: UploadFile, user_id: str = Form(...)):
    temp_video_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
            temp_video_path = tmpf.name
            shutil.copyfileobj(file.file, tmpf)

        cap = cv2.VideoCapture(temp_video_path, cv2.CAP_FFMPEG)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frames.append(frame)
        cap.release()

        print(f"Video {file.filename}: extracted {len(frames)} frames")

        if len(frames) == 0:
            raise RuntimeError("No frames extracted; invalid or corrupt video?")

        result = run_inference(
            app.state.deepfake_model,
            app.state.micro_model,
            app.state.rf_clf,
            app.state.xgb_clf,
            app.state.scaler,
            frames,
        )

        micro_labels = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
        micro_probs_map = {emo: to_float_or_orig(prob) for emo, prob in zip(micro_labels, result.get("micro_probs", []))}

        payload = {
            "status": "ok",
            "user_id": user_id,
            "fusion_label": result.get("fusion_label"),
            "confidence_tier": result.get("confidence_tier"),
            "confidence_score": to_float_or_orig(result.get("confidence_score")),
            "deep_score": to_float_or_orig(result.get("deep_score")),
            "deep_pred_probs": to_list_or_orig(result.get("deep_pred_probs")),
            "micro_probs": micro_probs_map,
            "weighted_proba": to_list_or_orig(result.get("weighted_proba")),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        log_session(
            payload["timestamp"],
            payload["fusion_label"],
            payload["confidence_tier"],
            payload["confidence_score"],
            payload["deep_score"],
            result.get("micro_probs", []),
        )

        return JSONResponse(content=jsonable_encoder(payload), status_code=200)

    except Exception as e:
        print(f"[ERROR] Exception in /liveness_check: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({"status": "error", "message": str(e)}),
        )

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ekyc.scripts.ekyc_service:app", host="0.0.0.0", port=8000, reload=True)
