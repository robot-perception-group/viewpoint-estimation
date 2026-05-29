# Run YOLO keypoint inference on a single image and  visualize

from pathlib import Path
import cv2

# Add script and parent directories to sys.path for local imports
from infer_frame import infer_frame
from visualization.visualize_keypoints import visualize_keypoints

IMAGE_PATH = Path(".../demo.jpg")
WEIGHTS_PATH = Path(".../keypoints-yolo/models/pose/cropped_models/quadruped-standalone-minimal-cropped/weights/best.pt")
DEVICE = "cpu"
CONF = 0.25


def main():
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        print(f"Failed to read image: {IMAGE_PATH}")
        return
    results = infer_frame(img, WEIGHTS_PATH, device=DEVICE, conf=CONF)
    if not results:
        print("No keypoints detected.")
        return
    print(results)
    for det in results:
        bbox = det.get("bbox", None)
        visualize_keypoints(img, det["keypoints"], title=f"Keypoints: {IMAGE_PATH.name}", bbox=bbox)

if __name__ == "__main__":
    main()
