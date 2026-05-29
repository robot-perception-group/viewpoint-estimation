# This script is used to train a Yolo keypoint detection model on uncropped images
from ultralytics import YOLO
DATA = ".../config/basic.yaml"


IMGSZ = 640       
BATCH = -1       
DEVICE = 0        
WORKERS = 16

# Training Hyperparameters

EPOCHS = 300     
LR0 = 0.001        
LRF = 0.01        # final LR will be 0.00001
OPTIMIZER = 'AdamW'
PATIENCE=30      

POSE = 12.0       # focus on keypoint accuracy
KOBJ = 2.0        # focus on predicting IF a limb is visible 

# Augmentations
DEGREES = 15.0    
SCALE = 0.7       
MOSAIC = 1.0      
MIXUP = 0.1      
LABEL_SMOOTHING = 0.05
FLIPLR = 0.5

def main():
    model = YOLO("yolo26n-pose.pt") 

    model.train(
        data=DATA,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        pose=POSE,
        kobj=KOBJ,
        degrees=DEGREES,
        scale=SCALE,          
        mosaic=MOSAIC,
        mixup=MIXUP,
        fliplr=FLIPLR,
        label_smoothing=LABEL_SMOOTHING,
        project="uncropped",
        name="quadruped",
        save=True,
        deterministic=True,   
        pretrained=True       
    )

if __name__ == "__main__":
    main()