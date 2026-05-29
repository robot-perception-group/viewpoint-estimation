# Run YOLO keypoint inference on a video and optionally visualize in real-time or save output video

from pathlib import Path
import cv2



from infer_frame import infer_frame

from yolo_keypoints_utils import SKELETON_EDGES


VIDEO_PATH = Path(".../DemoVideo.mp4") 
WEIGHTS_PATH = Path(".../keypoints-yolo/models/pose/cropped_models/quadruped-standalone-minimal-cropped/weights/best.pt")
DEVICE = "cuda:0"
CONF = 0.5
VISUALIZE = True

REALTIME = True  # If True, show each frame in real time; if False, save output video
SAVE_VIDEO = False  # If True, save the output video (works with REALTIME on or off)

OUTPUT_VIDEO_PATH = Path("output.mp4")  

SHOW_FPS = True  # Set to True to display FPS on video
FRAME_STRIDE = 1  # Process every nth frame 


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return
    
    video_name = VIDEO_PATH.stem  
    
    weights_parts = WEIGHTS_PATH.parts
    if "weights" in weights_parts:
        weights_idx = weights_parts.index("weights")
        model_name = weights_parts[weights_idx - 1]
    else:
        model_name = "model"
    
    output_filename = f"{video_name}_{model_name}.mp4"
    output_path = Path(output_filename)
    
    writer = None
    if VISUALIZE and SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        print(f"Saving output video to: {output_path}")
    import time
    frame_idx = 0
    prev_time = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Only process every nth frame
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue
        vis_frame = frame.copy()
        # Draw frame number in top left
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # FPS calculation
        if SHOW_FPS:
            curr_time = time.time()
            if frame_idx > 0:
                fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(vis_frame, f"FPS: {fps:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Inference
        results = infer_frame(vis_frame, WEIGHTS_PATH, device=DEVICE, conf=CONF)
        if results and VISUALIZE:
            # Visualize all detected animals with skeleton
            for det in results:
                kps = det["keypoints"]
                # Draw skeleton edges first
                for start_idx, end_idx in SKELETON_EDGES:
                    kp1 = kps[start_idx]
                    kp2 = kps[end_idx]
                    if kp1["conf"] > 0 and kp2["conf"] > 0:
                        cv2.line(vis_frame, (int(kp1["x"]), int(kp1["y"])), (int(kp2["x"]), int(kp2["y"])), (255,255,255), 2)
                # Draw keypoints
                for i, kp in enumerate(kps):
                    if kp["conf"] > 0:
                        cv2.circle(vis_frame, (int(kp["x"]), int(kp["y"])), 4, (0,0,255), -1)
                        label = kp.get("name", str(i))
                        cv2.putText(vis_frame, label, (int(kp["x"])+5, int(kp["y"])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if VISUALIZE:
            if writer:
                writer.write(vis_frame)
            if REALTIME:
                cv2.imshow("YOLO Keypoint Inference", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    if VISUALIZE and REALTIME:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

