# Project status

`crowd-detection` is an experimental Python application with a FastAPI dashboard and a synthetic simulation mode. The current main pipeline connects video or simulation input to person detection, tracking, density and movement analysis, alert evaluation, risk prediction, logging, recording, and dashboard state.

## Available now

- Webcam, video-file, RTSP, and simulation inputs through `main.py`.
- YOLO-based person detection with configurable model and thresholds.
- Tracking, density grid, occupancy, heatmaps, movement analysis, and alert levels.
- Risk prediction, explanation data, social-distance checks, cluster analysis, restricted-zone checks, and optional notification integrations.
- FastAPI dashboard with MJPEG stream, WebSocket updates, REST endpoints, and a mobile view.
- JSONL and SQLite persistence, incident timelines, CSV metric export, and MP4 incident recording.
- Full setup through `setup.sh`, quick setup through `install-quick.sh`, and launchers through `run.sh` and `start.sh`.

## Validation status

The repository currently has no automated test suite or pinned lockfile. The available static checks are:

```bash
python3 -m compileall -q config src web main.py launch.py
bash -n setup.sh install-quick.sh run.sh start.sh
```

The CI workflow runs those checks on pushes to `main` and on pull requests.

## Known limitations

The FastAPI server has no authentication, authorization, TLS, or rate limiting. Weather values are mocked rather than retrieved from a live weather service. Detection quality and risk scores have not been presented as validated against a labeled benchmark in this repository. Several analysis modules are experimental and require venue-specific calibration before any real-world use.

See [`README.md`](README.md) for installation, configuration, API routes, deployment guidance, security considerations, and contribution instructions.
