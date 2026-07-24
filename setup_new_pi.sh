#!/bin/bash
# One-shot setup for a brand-new Raspberry Pi to run TFLite_detection_webcam_gst_ObjName_latency
#
# Reproduces the exact environment used on the reference Pi:
#   - Debian 13 (trixie), aarch64
#   - Python 3.9.18 (via pyenv, since apt's default python3 is 3.13)
#   - opencv-python 4.12.0.88, numpy 1.19.5
#   - tflite-runtime 2.5.0.post1 (installed from the .whl in this folder)
#   - GStreamer 1.26.2 (gst-launch-1.0 + plugins, for streaming video out)
#
# Usage:
#   1. Copy this whole "example code" folder onto the new Pi
#      (it must contain this script + the .whl file + TFLite_detection_webcam_gst_ObjName_latency.py)
#   2. chmod +x setup_new_pi.sh && ./setup_new_pi.sh
#   3. source tflite1-env/bin/activate
#   4. python TFLite_detection_webcam_gst_ObjName_latency.py --modeldir <model_folder> --ip <receiver_ip>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== 1/5: Installing system packages =="
sudo apt-get update
sudo apt-get install -y \
    git curl build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libatlas-base-dev libjpeg-dev libopenjp2-7 \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav

echo "== 2/5: Installing pyenv (for Python 3.9, required by the tflite-runtime wheel) =="
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "== 3/5: Installing Python 3.9.18 =="
pyenv install -s 3.9.18

echo "== 4/5: Creating virtualenv 'tflite1-env' =="
"$PYENV_ROOT/versions/3.9.18/bin/python3.9" -m venv "$SCRIPT_DIR/tflite1-env"
source "$SCRIPT_DIR/tflite1-env/bin/activate"
pip install --upgrade pip

echo "== 5/5: Installing Python dependencies =="
pip install opencv-python==4.12.0.88 numpy==1.19.5
pip install "$SCRIPT_DIR/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl"

echo ""
echo "Setup complete. To run the script:"
echo "  source $SCRIPT_DIR/tflite1-env/bin/activate"
echo "  python TFLite_detection_webcam_gst_ObjName_latency.py --modeldir <model_folder> --ip <receiver_ip>"
