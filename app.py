from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
import io
import base64
from src.detector import DrowsinessDetector
from src.face_analyzer import FaceAnalyzer
from src.utils import draw_status, draw_alert_box, encode_frame

# ── Initialize App ─────────────────────────────────────────
app = FastAPI(title="Drowsiness Detection System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load Models Once at Startup ────────────────────────────
print("Loading models...")
detector = DrowsinessDetector()
analyzer = FaceAnalyzer()
print("All models ready! ✅")

# ── Routes ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image
    Returns drowsiness analysis result
    """
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Step 1 — MediaPipe face analysis
    face_result = analyzer.analyze(frame)

    if not face_result['face_detected']:
        return JSONResponse({
            "error": "No face detected",
            "label": "unknown",
            "confidence": 0,
            "ear": 0,
            "mar": 0,
            "needs_llama": False
        })

    # Step 2 — YOLOv8 prediction
    yolo_result = detector.predict(frame)

    # Step 3 — Check if LLaMA needed
    needs_llama = not yolo_result['is_confident']

    # Final label decision
    if needs_llama:
        # Will be handled by LLaMA in Section 6
        final_label = yolo_result['label']
        source = "yolo_uncertain"
    else:
        final_label = yolo_result['label']
        source = "yolo_confident"

    # Override with MediaPipe if clearly drowsy
    if face_result['is_drowsy']:
        final_label = "drowsy"
        source = "mediapipe"

    # Draw on frame
    annotated = face_result['annotated_frame']
    annotated = draw_status(
        annotated,
        final_label,
        yolo_result['confidence'],
        face_result['ear'],
        face_result['mar']
    )

    if final_label == "drowsy":
        annotated = draw_alert_box(annotated)

    # Convert frame to base64 for sending to browser
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "label": final_label,
        "confidence": round(yolo_result['confidence'], 3),
        "ear": face_result['ear'],
        "mar": face_result['mar'],
        "source": source,
        "needs_llama": needs_llama,
        "annotated_image": img_base64
    })


@app.get("/health")
async def health():
    """Check if system is running"""
    return {"status": "running", "models": "loaded"}