# TFLite Object Detection with Latency Overlay + GStreamer Streaming

Real-time object detection on a Raspberry Pi using TensorFlow Lite (with optional
Coral Edge TPU acceleration), with the annotated video stream sent out over the
network via GStreamer (H.264/RTP over UDP) instead of displayed locally. The
overlay shows live FPS and AI inference latency.

This project is based on [`TFLite_detection_webcam.py`](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
from EdjeElectronics' *TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi*
tutorial repo (Copyright Evan Juras, Apache License 2.0), modified to stream
output over the network via GStreamer instead of showing a local window, and
to add the AI-latency readout. See [LICENSE](LICENSE) for full license text
and attribution.

## Contents

| File | Purpose |
|---|---|
| `TFLite_detection_webcam_gst_ObjName_latency.py` | Main detection script |
| `setup_new_pi.sh` | One-shot environment setup script |
| `tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl` | Prebuilt tflite-runtime wheel for Python 3.9 / aarch64 |
| `TFLite_detection_webcam_gst_ObjName_latency.png` | Example output screenshot |
| `LICENSE` | Apache License 2.0, plus attribution to the original project |
| `.gitignore` | Excludes the generated `tflite1-env/` virtualenv, etc. |

## Requirements

- Raspberry Pi (aarch64) running **Debian 13 "trixie"**
- A USB webcam
- A receiver machine on the same network to view the stream (e.g. via
  `gst-launch-1.0` or VLC/ffplay listening on UDP port 5000)
- A TFLite model directory containing your model file (default name
  `edgetpu.tflite`) and a `labelmap.txt`

The reference environment this project was built and tested against:

- Debian 13 (trixie), aarch64
- Python **3.9.18** (installed via `pyenv`, since apt's default Python on
  trixie is 3.13, which is incompatible with the provided `tflite-runtime` wheel)
- `opencv-python` **4.12.0.88**
- `numpy` **1.19.5**
- `tflite-runtime` **2.5.0.post1** (from the `.whl` included in this folder)
- GStreamer **1.26.2** (`gst-launch-1.0` + `plugins-good`, `plugins-bad`, `libav`)

---

## Option 1: Install with the setup script (recommended)

1. Copy this entire folder onto the Raspberry Pi — it must contain
   `setup_new_pi.sh`, the `.whl` file, and the `.py` script together.
2. Make the script executable and run it:
   ```bash
   chmod +x setup_new_pi.sh
   ./setup_new_pi.sh
   ```
   This will:
   - Install required system packages (build tools, GStreamer, etc.) via `apt`
   - Install `pyenv` and use it to build Python 3.9.18
   - Create a virtual environment named `tflite1-env` in this folder
   - Install `opencv-python==4.12.0.88`, `numpy==1.19.5`, and `tflite-runtime`
     from the bundled wheel into that virtual environment
3. Activate the virtual environment:
   ```bash
   source tflite1-env/bin/activate
   ```
4. Run the script (see [Usage](#usage) below).

> The `apt-get` steps require `sudo`. Building Python 3.9.18 with `pyenv` can
> take several minutes on a Raspberry Pi.

---

## Option 2: Manual install

Use this if you want full control over each step, or are setting up on
hardware/OS that doesn't fit the scripted flow. To reproduce the exact same
environment as the script, follow these steps in order.

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y \
    git curl build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libatlas-base-dev libjpeg-dev libopenjp2-7 \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

### 2. Install pyenv and Python 3.9.18

The bundled `tflite-runtime` wheel is built for **CPython 3.9** — apt's system
Python on Debian 13 is 3.13, so a separate Python 3.9 must be built via `pyenv`.

```bash
curl https://pyenv.run | bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.9.18
```

> Add the three `export`/`eval` lines to your `~/.bashrc` if you want `pyenv`
> available in future shells.

### 3. Create and activate a virtual environment

Run this from inside the project folder (so the venv sits alongside the script):

```bash
"$PYENV_ROOT/versions/3.9.18/bin/python3.9" -m venv tflite1-env
source tflite1-env/bin/activate
pip install --upgrade pip
```

### 4. Install Python dependencies

```bash
pip install opencv-python==4.12.0.88 numpy==1.19.5
pip install tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```

The wheel filename encodes the exact ABI it was built for: CPython 3.9
(`cp39-cp39`), Linux aarch64. It will only install successfully into a
Python 3.9 environment on aarch64 — this is why step 2/3 are required rather
than using the system Python.

---

## Usage

With the virtual environment activated:

```bash
python TFLite_detection_webcam_gst_ObjName_latency.py \
    --modeldir <model_folder> \
    --ip <receiver_ip>
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--modeldir` | Yes | — | Path to the folder containing the model file and label map |
| `--ip` | Yes | — | IP address of the machine that will receive the video stream |
| `--graph` | No | `edgetpu.tflite` | Model filename inside `--modeldir` |
| `--labels` | No | `labelmap.txt` | Label map filename inside `--modeldir` |
| `--threshold` | No | `0.35` | Minimum confidence score to draw a detection |
| `--resolution` | No | `1280x720` | Capture/stream resolution, `WxH` |
| `--edgetpu` | No | off | Pass this flag to use a Coral Edge TPU delegate |

### Viewing the stream

On the receiver machine (the `--ip` you passed in), listen for the incoming
H.264/RTP stream on UDP port 5000, for example:

```bash
gst-launch-1.0 -v udpsrc port=5000 \
    ! application/x-rtp,encoding-name=H264,payload=96 \
    ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

### Stopping

Press `Ctrl+C` in the terminal running the script. This releases the camera,
closes the GStreamer pipeline, and terminates cleanly.

---

## Notes

- If you have a real TensorFlow (not `tflite-runtime`) installed instead, the
  script falls back to `tensorflow.lite.python.interpreter` automatically.
- The on-screen overlay shows both the end-to-end **FPS** and the isolated
  **AI inference latency** (time spent purely in `interpreter.invoke()`),
  useful for comparing CPU vs. Edge TPU performance.

## Credits & License

`TFLite_detection_webcam_gst_ObjName_latency.py` is a modified version of
[`TFLite_detection_webcam.py`](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
from Evan Juras' (EdjeElectronics) *TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi*
tutorial. Modifications in this version: GStreamer UDP streaming output in
place of a local display window, an AI-inference-only latency readout, and
an FPS overlay.

This repository is licensed under the **Apache License, Version 2.0** — the
same license as the original project. See [LICENSE](LICENSE) for the full
text and attribution notice.
