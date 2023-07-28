"""
verify_txts.py

For verifying correctness of the generated YOLO txt annotations.
"""

import random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import os

#Dataset path
dataset_dir = "/mnt/FaceDetectionDatasets/WIDER_test_gt_label_emami_points_yolo_format"

for i in range(len(os.listdir(dataset_dir))):
    p = random.uniform(0, 1)
    # if p > 0.5:
    if os.listdir(dataset_dir)[i].split(".")[1] == "txt":
        text_file_name = os.listdir(dataset_dir)[i]
    else:
        text_file_name = os.listdir(dataset_dir)[i].replace("jpg", "txt")

    img_name = text_file_name.replace("txt", "jpg")
    img = cv2.imread(os.path.join(dataset_dir, img_name))
    img_h, img_w, _ = img.shape
    
    with open(os.path.join(dataset_dir, text_file_name), 'r') as f:
        obj_lines = [l.strip() for l in f.readlines()]
    for line in obj_lines:
        if line.startswith("#"):
            continue
        else:
            points = []
            points = [item for item in line.split(' ')]

            x_min = int((float(points[2]) - (float(points[4])  / 2.0)) * img_w)
            y_min = int((float(points[3]) - (float(points[5]) / 2.0)) * img_h)
            x_max = int((float(points[2]) + (float(points[4]) / 2.0)) * img_w)
            y_max = int((float(points[3])  + (float(points[5]) / 2.0)) * img_h)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            if float(points[6]) > 0:
                cv2.circle(img, center=(int(float(points[6])*img_w), int(float(points[7])*img_h)), thickness=-1, color=(255, 255, 0), radius=1)
                cv2.circle(img, center=(int(float(points[8])*img_w), int(float(points[9])*img_h)), thickness=-1, color=(255, 255, 0), radius=1)
                cv2.circle(img, center=(int(float(points[10])*img_w), int(float(points[11])*img_h)), thickness=-1, color=(255, 255, 0), radius=1)
                cv2.circle(img, center=(int(float(points[12])*img_w), int(float(points[13])*img_h)), thickness=-1, color=(255, 255, 0), radius=1)
                cv2.circle(img, center=(int(float(points[14])*img_w), int(float(points[15])*img_h)), thickness=-1, color=(255, 255, 0), radius=1)
                cv2.imwrite(os.path.join("./verify_texts", f"{img_name}.jpg"), img)
