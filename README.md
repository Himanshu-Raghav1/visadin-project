# Visadin AI 🛡️

> **Edge AI Security Camera System**  
> Dynamic frame throttling + real-time weapon & threat detection on your own hardware.

---

## Features

| Feature | Detail |
|---|---|
| **Dynamic FPS Throttling** | IDLE at 5 FPS, ACTIVE at 30 FPS — cuts storage ~80% with zero quality loss on events |
| **Pre-Event Buffer** | Saves the 5 s leading up to any trigger (the approach is never missed) |
| **YOLOv8 Threat Detection** | Knife / gun / weapon detection; sub-second on CUDA or CPU |
| **Auto Backend Selection** | TensorRT → CUDA → OpenVINO → CPU — zero code change |
| **GStreamer Ingestion** | Sub-100 ms RTSP latency; falls back to OpenCV/FFMPEG automatically |
| **Auto-Reconnect** | Streams restart automatically after any drop |
| **Async Webhook Alerts** | JSON payload + base64 thumbnail; configurable cooldown |
| **Live MJPEG Dashboard** | Browser-accessible at `http://device-ip:8000` |
| **Runtime Controls** | Adjust FPS, quality, and state without restarting |

---

## Project Structure

```
visadin/
├── config.py          # All config via env vars / .env
├── stream_ingest.py   # RTSP ingestion, circular buffer, auto-reconnect
├── throttler.py       # IDLE/ACTIVE state machine, video chunk writer
├── detector.py        # YOLOv8 inference pipeline + annotation
├── alerter.py         # Async webhook & thumbnail alerter
├── pipeline.py        # Per-camera orchestration thread
├── main.py            # FastAPI server + entry point
├── dashboard.html     # Browser dashboard (served at /)
├── requirements.txt
├── .env.example       # Copy to .env and edit
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Himanshu-Raghav1/visadin-project.git
cd visadin-project
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env – at minimum set RTSP_URLS
```

### 3. Run

```bash
python main.py
```

Open your browser at **http://localhost:8000**

---

## RTSP URL Examples

```
# Hikvision
rtsp://admin:password@192.168.1.64/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.108/cam/realmonitor?channel=1&subtype=0

# Demo (public test stream)
rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov
```

---

## GPU Acceleration

```bash
# CUDA (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then set in .env:
YOLO_DEVICE=cuda

# Intel OpenVINO
pip install openvino
yolo export model=yolov8n.pt format=openvino
YOLO_MODEL_PATH=yolov8n_openvino_model/
YOLO_DEVICE=cpu  # OpenVINO runs on CPU automatically
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/api/cameras` | List all cameras + stats |
| GET | `/api/cameras/{id}/stats` | Per-camera stats JSON |
| POST | `/api/cameras/{id}/control` | Override FPS / quality / state |
| GET | `/stream/{id}` | MJPEG live stream |
| GET | `/snapshot/{id}` | Single JPEG snapshot |
| GET | `/api/alerts/recent` | Recent threat thumbnails |
| GET | `/api/alerts/{file}` | Serve thumbnail image |
| GET | `/api/recordings` | List saved video chunks |
| POST | `/api/system/shutdown` | Graceful shutdown |

### Control payload example

```json
POST /api/cameras/cam0/control
{
  "fps": 15,
  "quality": 90,
  "state": "active"
}
```

Pass `null` for any field to revert to automatic control.

---

## Webhook Payload

```json
{
  "event": "threat_detected",
  "timestamp": "2024-08-08T03:00:00Z",
  "camera_id": "cam0",
  "frame_index": 12345,
  "threat_class": "knife",
  "confidence": 0.87,
  "bbox": [120, 80, 240, 320],
  "inference_ms": 18.4,
  "thumbnail_b64": "<base64-jpeg>"
}
```

---

## Adding Custom Threat Classes

Edit `THREAT_CLASSES` in `.env`:

```
THREAT_CLASSES=knife,gun,pistol,rifle,weapon,mask,backpack
```

The strings must match class names in your YOLO model.  
For a custom dataset, train with [Roboflow](https://roboflow.com/) and point `YOLO_MODEL_PATH` at your `.pt` file.

---

## License

MIT — see [LICENSE](LICENSE).
