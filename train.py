import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolov8s-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8s-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='widerface.yaml', epochs=300, imgsz=640, batch=16, device=[0,1])
