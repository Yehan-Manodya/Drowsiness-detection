import requests
import base64
import cv2
import numpy as np

class OllamaReader:
    def __init__(self):
        self.model = "llama3.2-vision:11b"
        self.url = "http://localhost:11434/api/generate"
        print("OllamaReader initialized ✅")

    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, frame):
        """
        Send frame to LLaMA 3.2 Vision for edge case analysis
        Returns: 'alert' or 'drowsy'
        """
        print("Sending to LLaMA 3.2 Vision...")
        
        img_base64 = self.frame_to_base64(frame)
        
        prompt = """Look at this image of a driver. 
        Analyze their face carefully and determine if they are:
        - DROWSY: eyes closing/closed, head drooping, yawning, unfocused
        - ALERT: eyes open, looking forward, attentive
        
        Reply with ONLY one word: either 'drowsy' or 'alert'"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=30
            )
            result = response.json()['response'].strip().lower()
            
            # Clean the response
            if 'drowsy' in result:
                label = 'drowsy'
            elif 'alert' in result:
                label = 'alert'
            else:
                label = 'alert'  # default to alert if unclear
                
            print(f"LLaMA 3.2 Vision result: {label} ✅")
            return label

        except Exception as e:
            print(f"LLaMA error: {e}")
            return None