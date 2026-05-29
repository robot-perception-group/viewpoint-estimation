"""
Visualize keypoints on an image (frame) using the result from infer_frame.
"""
import cv2
import matplotlib.pyplot as plt

def visualize_keypoints(img, keypoints, title=None, bbox=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    
    # Draw bounding box if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)
    
    # Draw keypoints
    for kp in keypoints:
        if kp["conf"] > 0:
            plt.scatter(kp["x"], kp["y"], c='r', s=20)
            label = kp.get("name", "")
            plt.text(kp["x"]+3, kp["y"]-3, label, color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()
