from roboflow import Roboflow
import os

# Initialize Roboflow (using public guest access)
rf = Roboflow(api_key="7EdWUUZ4KVSWAQSdLDzn")

print("Downloading pre-annotated Rubik's Cube dataset from Roboflow...")

# Download dataset configured specifically for YOLOv8
project = rf.workspace("michael-song").project("rubik-s-cube-detection")
dataset = project.version(1).download("yolov8")

print("\nDataset successfully downloaded!")
print("Dataset location:", dataset.location)