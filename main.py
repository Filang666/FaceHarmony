import cv2
import torch
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from engine import MobileFaceLandmarker

class LooksmaxApp(App):
    def build(self):
        self.img_widget = Image()
        self.capture = cv2.VideoCapture(0)
        
        # Device config
        self.device = torch.device("cpu")
        
        # Load Model
        self.model = MobileFaceLandmarker().to(self.device).eval()
        # self.model.load_state_dict(torch.load('landmarker.pth'))

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img_widget

    def calculate_metrics(self, pts):
        """
        Implements professional Looksmaxing math.
        pts: (68, 2) array of landmarks
        """
        results = []
        
        # 1. Golden Ratio (Face length / width)
        face_width = np.linalg.norm(pts[16] - pts[0])
        face_height = np.linalg.norm(pts[8] - pts[24]) # Chin to brow avg
        ratio = face_height / face_width if face_width != 0 else 0
        results.append(f"Ratio: {ratio:.2f} (Ideal 1.61)")

        # 2. Eye Spacing (Horizontal Fifths)
        eye_width = np.linalg.norm(pts[39] - pts[36])
        inter_eye_dist = np.linalg.norm(pts[42] - pts[39])
        results.append(f"Eye Gap: {inter_eye_dist/eye_width:.1f}x (Ideal 1.0)")

        # 3. Lower Third Breakdown (Nose to Lip / Lip to Chin)
        # Nose: 33, Lip Center: 62, Chin: 8
        d1 = np.linalg.norm(pts[33] - pts[62])
        d2 = np.linalg.norm(pts[62] - pts[8])
        results.append(f"Lower 1/3: {d1/d2:.2f}")

        return results

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        h, w, _ = frame.shape
        # Prep for model
        input_res = 224
        img_input = cv2.resize(frame, (input_res, input_res))
        img_tensor = torch.from_numpy(img_input).permute(2,0,1).float() / 255.0
        
        with torch.no_grad():
            landmarks = self.model(img_tensor.unsqueeze(0)).numpy().reshape(68, 2)

        # Draw Landmarks & Metrics
        metrics = self.calculate_metrics(landmarks)
        
        for i, pt in enumerate(landmarks):
            x, y = int(pt[0] * w / input_res), int(pt[1] * h / input_res)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, 50 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert to Kivy Texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(w, h), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    LooksmaxApp().run()
