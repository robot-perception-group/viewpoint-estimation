# This script finetunes models for cropped images with uav data and no sideview images, resulting in overfitting
from ultralytics import YOLO

MODEL_PATH = ".../runs/pose/cropped_models/quadruped-standalone-minimal-cropped/weights/best.pt" 

DATA = ".../config/basic.yaml"
FLIPLR = 0.5
def main():
    model = YOLO(MODEL_PATH)
    
    model.train(
        data=DATA,
        epochs=300,
        imgsz=224,
        batch=-1,         
        device=0, 
    
        freeze=20,        


        lr0=0.00001,      
        lrf=0.01,
        warmup_epochs=20, # Longer warmup to stabilize the gradients
        
        mosaic=0.6,       
        overlap_mask=True, 
        fliplr=FLIPLR,

                
        project="overfit_finetune",
        name="overfit_minimal_cropped",
        save=True,
        patience=50
    )

if __name__ == "__main__":
    main()