from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from overlay_utils import overlay_results
import cv2
import os
import uuid

# --- Initialize app ---
app = FastAPI(title="Road & Vehicle Detection API")

# --- Load models ---
det_model = YOLO("models/yolo11n.pt")      # Vehicle Detection
seg_model = YOLO("models/best_road.pt")     # Road Segmentation

# --- Directories ---
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/videos", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)

# =====================================================
# HOME ENDPOINT
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Road & Vehicle Detection API is running!",
        "available_endpoints": {
            "image": "/predict/image",
            "video": "/predict/video",
            "camera": "/predict/camera",
            "docs": "/docs"
        },
        "usage": "Go to /docs to test the API interactively."
    }

# =====================================================
# IMAGE ENDPOINT
# =====================================================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    file_path = f"uploads/images/{uuid.uuid4()}_{file.filename}"
    output_path = f"outputs/images/result_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = det_model(img_rgb)
    segs = seg_model(img_rgb)

    overlay = overlay_results(img_rgb, dets, segs, det_model)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return FileResponse(output_path, media_type="image/jpeg")

# =====================================================
# VIDEO ENDPOINT
# =====================================================
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    file_path = f"uploads/videos/{uuid.uuid4()}_{file.filename}"
    output_path = f"outputs/videos/result_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = det_model(frame_rgb)
        segs = seg_model(frame_rgb)
        overlay = overlay_results(frame_rgb, dets, segs, det_model)
        out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    return FileResponse(output_path, media_type="video/mp4")

# =====================================================
# CAMERA (LIVE STREAM)
# =====================================================
@app.get("/predict/camera")
def predict_camera():
    cap = cv2.VideoCapture(0)  # 0 = internal cam, use 1 or 2 for external

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = det_model(frame_rgb)
        segs = seg_model(frame_rgb)
        overlay = overlay_results(frame_rgb, dets, segs, det_model)

        cv2.imshow("Live Detection", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Camera stream ended."}
