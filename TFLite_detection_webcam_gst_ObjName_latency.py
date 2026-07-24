# Modified from EdjeElectronics' TFLite_detection_webcam.py
# (https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
# Original work Copyright Evan Juras (EdjeElectronics), licensed under
# the Apache License, Version 2.0. See LICENSE for details.
# Changes: added GStreamer UDP streaming output, AI-inference-only
# latency readout, and FPS overlay in place of the original local display.

import os
import argparse
import cv2
import numpy as np
import time
import importlib.util
from threading import Thread
import subprocess

class VideoStream:
    def __init__(self, resolution=(640, 320), framerate=30):
        self.stream = cv2.VideoCapture(0)  # Use video source
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='edgetpu.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.35)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')
parser.add_argument('--ip',required=True)
args = parser.parse_args()

IP_Receiver = args.ip
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = map(int, args.resolution.split('x'))
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'model.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)



if use_TPU:
    print("✅ Edge TPU is enabled. Using edgetpu.tflite")
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
else:
    print("⚠️  TPU not used. Running on CPU only.")
    interpreter = Interpreter(model_path=PATH_TO_CKPT)



with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()



input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

resW, resH = map(int, args.resolution.split('x'))  # <-- Define first
freq = cv2.getTickFrequency()
frame_rate_calc = 1
videostream = VideoStream(resolution=(resW, resH), framerate=30).start()
time.sleep(1)


# GStreamer command using rawvideoparse (fixes "not-negotiated" error)

gst_cmd = [
    'gst-launch-1.0', '-v',
    'fdsrc', 'fd=0',
    '!', 'rawvideoparse',
    f'format=bgr', f'width={resW}', f'height={resH}', 'framerate=30/1',
    '!', 'videoconvert',
    '!', 'queue', 'leaky=downstream' , 'max-size-buffers=1', 'max-size-bytes=0',
    'max-size-time=0',
    '!', 'x264enc', 'tune=zerolatency', 'speed-preset=ultrafast',
    'bitrate=4096', 'key-int-max=15', 'bframes=0',
    '!', 'rtph264pay', 'config-interval=1', 'pt=96',
    '!', 'udpsink', f'host={IP_Receiver}', 'port=5000', 'sync=false', 'async=false'
]


gst_process = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE)

frame_count = 0
while True:
    t1 = cv2.getTickCount()
    frame_count += 1
    frame1 = videostream.read()
    if frame1 is None:
        continue
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # --- AI (inference-only) latency: set_tensor + invoke, excludes capture/streaming ---
    t_inf_start = cv2.getTickCount()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    t_inf_end = cv2.getTickCount()
    inference_ms = (t_inf_end - t_inf_start) / freq * 1000.0

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * resH)))
            xmin = int(max(1, (boxes[i][1] * resW)))
            ymax = int(min(resH, (boxes[i][2] * resH)))
            xmax = int(min(resW, (boxes[i][3] * resW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- Draw AI latency readout, right-aligned in the top-right corner ---
    lat_label = 'AI latency: %.1f ms' % inference_ms
    lat_size, lat_base = cv2.getTextSize(lat_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    lat_x = resW - lat_size[0] - 15
    lat_y = 30
    cv2.rectangle(frame, (lat_x - 8, lat_y - lat_size[1] - 8),
                  (lat_x + lat_size[0] + 8, lat_y + lat_base + 4), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, lat_label, (lat_x, lat_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Draw FPS readout, top-left corner, green ---
    fps_label = 'FPS: %.2f' % frame_rate_calc
    cv2.putText(frame, fps_label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    try:
        gst_process.stdin.write(frame.tobytes())
    except Exception as e:
        print("GStreamer write error:", e)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

videostream.stop()
gst_process.stdin.close()
gst_process.terminate()
cv2.destroyAllWindows()
