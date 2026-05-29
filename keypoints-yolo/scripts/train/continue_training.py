# continue training from a checkpoint
from ultralytics import YOLO

checkpoint_path = ".../runs/pose/uncropped/quadruped/weights/last.pt"

model = YOLO(checkpoint_path)
model.train(resume=True)