import cv2
import numpy as np
from datetime import datetime

# Alert colors
COLOR_ALERT  = (0, 255, 0)    # Green
COLOR_DROWSY = (0, 0, 255)    # Red
COLOR_TEXT   = (255, 255, 255) # White

def draw_status(frame, label, confidence, ear, mar):
    """Draw status overlay on frame"""
    h, w = frame.shape[:2]

    # Background bar
    color = COLOR_DROWSY if label == 'drowsy' else COLOR_ALERT
    cv2.rectangle(frame, (0, 0), (w, 80), color, -1)

    # Status text
    status = f"STATUS: {label.upper()}"
    cv2.putText(frame, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)

    # Confidence
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(frame, conf_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

    # EAR and MAR
    cv2.putText(frame, f"EAR: {ear:.3f}", (w-200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    cv2.putText(frame, f"MAR: {mar:.3f}", (w-200, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return frame

def draw_alert_box(frame):
    """Draw big red alert when drowsy detected"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
    cv2.putText(frame, "⚠ DROWSINESS ALERT!", (w//2-200, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame

def encode_frame(frame):
    """Convert frame to JPEG bytes for FastAPI streaming"""
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()