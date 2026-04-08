
## Lightweight MTCNN & Facial Looksmaxing AI

[![Python 3.14+](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-red.svg)](https://pytorch.org/)
[![Buildozer](https://img.shields.io/badge/Buildozer-1.5.0-8A2BE2?logo=kivy&logoColor=white)](https://buildozer.readthedocs.io)
[![Docker](https://img.shields.io/badge/docker-✓-2496ED.svg)](https://www.docker.com/)


A mobile-ready facial detection and aesthetic analysis system built with PyTorch and Kivy. The project implements a multi-stage CNN cascade for face localization and a MobileNetV2-based landmark regressor (68 points) for calculating facial proportions and "Looksmaxing" metrics.

## 🚀 Key Features

- **MTCNN Detection**: Efficient face localization using P-Net.
- **Aesthetic Scoring**: Automated calculation of:
  - **Golden Ratio**: Face height/width balance.
  - **Vertical Thirds**: Evaluation of facial harmony from hairline to chin.
  - **Horizontal Fifths**: Eye spacing and symmetry analysis.
  - **Lower Third Breakdown**: Professional-grade ratio between nose, lips, and chin.
- **Mobile-Optimized**: Designed for Android/iOS deployment via Buildozer.
- **Dockerized Environment**: One-command setup for training and development.

## 📂 Project Structure

```text
├── engine.py           # Neural Network architectures (P-Net, Landmarks, Scorer)
├── training.py         # Training pipelines and loss functions
├── main.py             # Kivy application and real-time inference logic
├── Dockerfile          # Environment isolation
├── docker-compose.yml  # GPU-accelerated container orchestration
├── requirements.txt    # Python dependencies
└── buildozer.spec      # Android/iOS build configuration
```

## 🛠 Installation & Setup

### Using Docker (Recommended for Training)

Ensure you have NVIDIA Container Toolkit installed for GPU support.

```bash
# Build and start training
docker-compose up --build
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the desktop version
python main.py
```

## 📱 Mobile Deployment (Android)

The application uses Kivy for the UI and camera interface to ensure cross-platform compatibility.

1. **Prepare Buildozer**  
   Use a Linux environment (Ubuntu recommended).  
   ```bash
   buildozer init
   ```

2. **Configure `buildozer.spec`**  
   Ensure requirements include: `python3, kivy, pytorch, torchvision, opencv-python, numpy`.

3. **Build APK**  
   ```bash
   buildozer -v android debug deploy run
   ```

## 🧪 Scientific Metrics Implementation

The "Looksmaxing" logic is based on the following anthropometric markers:

- **Golden Ratio** ($1.618$): Measured as $FaceHeight / FaceWidth$.
- **Eye Spacing**: Ideal ratio is $1.0$ (one eye width between the eyes).
- **Lower Face Ratio**: Proportions between the subnasale-to-labrale and labrale-to-menton distances.

## 👨‍💻 Contributing

1. Add new model architectures to `engine.py`.
2. Test inference performance on mobile CPUs (use `torch.quantization` if needed).
3. Ensure all code follows PEP8 standards.

## ⚖️ License

MIT License. Created for educational and research purposes in facial aesthetics.
