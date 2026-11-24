# Quick start

## 1. Install

```bash
git clone https://github.com/vincenzo-afk/crowd-detection.git
cd crowd-detection
chmod +x setup.sh run.sh start.sh install-quick.sh
./setup.sh
```

The full setup creates a Python virtual environment, installs `requirements.txt`, creates runtime directories, and attempts to prepare the default `yolov8n.pt` model.

## 2. Run

```bash
# Live webcam, camera index 0
./run.sh

# Synthetic crowd simulation without a camera
./run.sh --sim

# Video file, camera index, or RTSP URL
./run.sh --source /path/to/video.mp4
./run.sh --source 1
./run.sh --source rtsp://camera-host/stream

# Dashboard without a local OpenCV window
./run.sh --sim --no-window
```

Open the dashboard at <http://localhost:8000>. Use `--port 8080` to select another port.

## 3. Simulation controls

In simulation mode, the local OpenCV window accepts `1` for normal, `2` for building, `3` for dense, and `4` for panic scenarios. Press `q` to quit, `h` to hide the window, or `s` to toggle simulation mode.

The web dashboard also exposes simulation actions through `POST /api/control/sim_normal`, `POST /api/control/sim_building`, `POST /api/control/sim_dense`, and `POST /api/control/sim_panic`.

## Troubleshooting

If dependencies are missing, run `./setup.sh`. If a camera cannot be opened, try another numeric source, use a video file, or run `./run.sh --sim`. If port `8000` is already in use, run `./run.sh --sim --port 8080`.

For configuration, API routes, runtime outputs, limitations, and contribution guidance, see [`README.md`](README.md).
