# Self-Driving Vision Core 🚗🤖

A real-time computer vision system for autonomous driving that combines **road segmentation** and **vehicle detection** using YOLOv8 and YOLO11 models. Built with FastAPI for easy deployment and scalability.

## 🎬 Demo

### Sample Output

The system overlays:
- **Blue transparent mask** for drivable road areas
- **Green bounding boxes** for detected vehicles/objects
- **Confidence scores** for each detection

![Demo Output](gif/demo.gif)

*Example: Real-time detection on highway footage*

## 🎯 Overview

This project implements a fundamental perception system for self-driving vehicles by combining two critical computer vision tasks:

1. **Road Segmentation**: Identifies drivable road surfaces using a custom-trained YOLOv8 segmentation model
2. **Vehicle Detection**: Detects and classifies vehicles, pedestrians, and traffic signs using YOLO11

The system provides real-time inference through a FastAPI backend, supporting images, videos, and live camera feeds.

## ✨ Features

- **Dual-Model Architecture**: Road segmentation + object detection in one pipeline
- **RESTful API**: FastAPI-powered endpoints for easy integration
- **Multiple Input Modes**:
  - Static image processing
  - Video file analysis
  - Real-time camera streaming
- **Optimized Performance**: GPU-accelerated inference with CUDA support
- **Production-Ready**: Clean code structure with modular design
- **Interactive Docs**: Auto-generated Swagger UI at `/docs`

## 🏗️ Architecture

```
Input Image/Video
       ↓
┌──────────────────┐
│  Pre-processing  │
└──────────────────┘
       ↓
┌──────────────────┐     ┌──────────────────┐
│  YOLOv8n-seg     │     │    YOLO11n       │
│  (Road Seg)      │     │  (Detection)     │
└──────────────────┘     └──────────────────┘
       ↓                         ↓
┌─────────────────────────────────┐
│     Overlay & Visualization     │
└─────────────────────────────────┘
       ↓
   Output Result
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ RAM

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/adeel-iqbal/self-driving-vision-core.git
cd self-driving-vision-core
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download models**

Place the following models in the `models/` directory:
- `best_road.pt` (Road segmentation model)
- `yolo11n.pt` (Vehicle detection model)

> **Note**: The custom road segmentation model (`best_road.pt`) was trained on the BDD10K dataset. Pre-trained weights available on request.

## 🚀 Usage

### Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. **Home** - `GET /`

Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

#### 2. **Image Prediction** - `POST /predict/image`

Process a single image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  --output result.jpg
```

**Response:** Processed image with overlays

#### 3. **Video Prediction** - `POST /predict/video`

Process a video file.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/video" \
  -H "accept: video/mp4" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4" \
  --output result.mp4
```

**Response:** Processed video with overlays

#### 4. **Camera Stream** - `GET /predict/camera`

Real-time processing from webcam (opens local window).

```bash
curl http://localhost:8000/predict/camera
```

**Note:** Press `q` to quit the camera stream.

### Testing with Swagger UI

Navigate to `http://localhost:8000/docs` for interactive API documentation where you can:
- Upload files directly
- Test endpoints
- View request/response schemas

### Python Client Example

```python
import requests

# Image prediction
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f}
    )
    
with open("output.jpg", "wb") as out:
    out.write(response.content)

print("✅ Processed image saved!")
```

## 📁 Project Structure

```
self-driving-vision-core/
├── models/
│   ├── best_road.pt          # Road segmentation model
│   └── yolo11n.pt            # Vehicle detection model
├── notebooks/
│   ├── road-segmentation.ipynb         # Training notebook
│   └── self-driving-vision-core.ipynb  # Inference notebook
├── uploads/
│   ├── images/               # Temporary uploaded images
│   └── videos/               # Temporary uploaded videos
├── outputs/
│   ├── images/               # Processed images
│   └── videos/               # Processed videos
├── gif/
│   └── demo.gif              # Demo visualization
├── app.py                    # FastAPI application
├── overlay_utils.py          # Overlay processing utilities
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🤖 Models

### 1. Road Segmentation Model (YOLOv8n-seg)

- **Dataset**: BDD10K Road Segmentation (YOLO format)
- **Architecture**: YOLOv8n-seg (nano segmentation model)
- **Training**:
  - 50 epochs
  - Image size: 640×640
  - Batch size: 8
  - AdamW optimizer
- **Performance**:
  - mAP50 (Box): 0.478
  - mAP50 (Mask): 0.475
  - Inference: ~7.5ms per image

### 2. Vehicle Detection Model (YOLO11n)

- **Pre-trained**: COCO dataset (80 classes)
- **Classes detected**: cars, trucks, buses, motorcycles, bicycles, traffic lights, stop signs, pedestrians
- **Inference**: ~8-10ms per image

## 📊 Training Details

The road segmentation model was trained using:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="road_segmentation"
)
```

**Training Results:**
- Final loss: Box=0.6427, Seg=1.376, Cls=0.573
- Training time: 1.574 hours (Tesla T4 GPU)
- Dataset: 6,955 training images, 999 validation images

Full training logs available in `notebooks/road-segmentation.pdf`

## 🔮 Future Improvements

This project serves as a foundation for a complete autonomous driving perception system. Planned enhancements include:

### Core Perception Features
- [ ] **Lane Detection**: Identify lane markings and boundaries
- [ ] **Traffic Sign Recognition**: Classify and interpret traffic signs
- [ ] **Depth Estimation**: Calculate distance to objects using monocular/stereo vision
- [ ] **Speed Estimation**: Track vehicle velocities using optical flow

### Advanced Features
- [ ] **3D Object Detection**: Convert 2D bounding boxes to 3D space
- [ ] **Sensor Fusion**: Integrate LiDAR, radar, and camera data
- [ ] **Path Planning Visualization**: Display predicted vehicle trajectory
- [ ] **Weather/Night Adaptation**: Robust detection in adverse conditions
- [ ] **Multi-Camera Support**: 360° perception with multiple cameras

### System Improvements
- [ ] **Tracking System**: Add object tracking (DeepSORT/ByteTrack)
- [ ] **WebSocket Streaming**: Real-time browser-based visualization
- [ ] **Metrics Dashboard**: Live performance monitoring (FPS, latency, accuracy)

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## 📧 Contact

**Adeel Iqbal**

- **Email**: [adeelmemon096@yahoo.com](mailto:adeelmemon096@yahoo.com)
- **LinkedIn**: [linkedin.com/in/adeeliqbalmemon](https://linkedin.com/in/adeeliqbalmemon)
- **GitHub**: [@adeel-iqbal](https://github.com/adeel-iqbal)

---

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8/YOLO11 framework
- **BDD10K Dataset**: Berkeley DeepDrive for road segmentation data
- **FastAPI**: Modern web framework for building APIs
- **OpenCV**: Computer vision library

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{self_driving_vision_core,
  author = {Adeel Iqbal},
  title = {Self-Driving Vision Core: Real-time Road Segmentation and Vehicle Detection},
  year = {2025},
  url = {https://github.com/adeel-iqbal/self-driving-vision-core}
}
```
