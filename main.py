import cv2
import torch
import numpy as np
from engine import PNet, RNet, ONet


class FaceDetectorApp:
    """Main Application to handle Video Stream and MTCNN Inference."""
    
    def __init__(self, camera_idx=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Cascade
        self.pnet = PNet().to(self.device).eval()
        self.rnet = RNet().to(self.device).eval()
        self.onet = ONet().to(self.device).eval()

        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video stream.")

    @staticmethod
    def _preprocess(image, size):
        """Prepares image for the neural network."""
        img = cv2.resize(image, (size, size))
        img = torch.FloatTensor(img).permute(2, 0, 1)
        img = (img - 127.5) * 0.0078125  # Normalization
        return img.unsqueeze(0)

    def process_frame(self, frame):
        """Runs the image through the P-Net stage (Simplified demo)."""
        h, w, _ = frame.shape
        input_t = self._preprocess(frame, 12).to(self.device)

        with torch.no_grad():
            cls_map, box_map = self.pnet(input_t)
        
        # Get max confidence score from the heatmap
        score = cls_map[0, 1, :, :].max().item()

        if score > 0.90:
            # Drawing a demo rectangle if a face is suspected
            cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {score:.2f}", (60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main loop for camera capture and display."""
        print("Application started. Press 'Q' to exit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                output_frame = self.process_frame(frame)
                cv2.imshow("Lightweight MTCNN (Face Detection)", output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceDetectorApp()
    app.run()
