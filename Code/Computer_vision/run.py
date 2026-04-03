# Code/Computer_vision/run.py
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Make relative imports work when imported as a module
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import src.DoDecahedronUtils as dodecapen
import src.Tracker as tracker

# Intrinsics (unchanged from your file)
CAM_DIR = BASE_DIR / "camera_matrix"
COLOR_CAM_MATRIX_PATH = CAM_DIR / "color_cam_matrix.npy"
COLOR_CAM_DIST_PATH   = CAM_DIR / "color_cam_dist.npy"
color_cam_matrix = np.load(COLOR_CAM_MATRIX_PATH)
color_cam_dist   = np.load(COLOR_CAM_DIST_PATH)

# >>> shared state the bridge reads <<<
# shape (1,12): [tx,ty,tz, r00 r01 r02 r10 r11 r12 r20 r21 r22]
object_pose: np.ndarray | None = None

# >>> shared state for pen tip positions from IMU app <<<
raw_pen_tip_position: np.ndarray | None = None
smoothed_pen_tip_position: np.ndarray | None = None

# >>> shutdown flag for graceful termination <<<
cv_shutdown_requested: bool = False

def _publish_pose(obj_1x12: np.ndarray) -> None:
    """Make the latest pose visible to dodeca_bridge in-process."""
    global object_pose
    object_pose = obj_1x12

def _publish_pen_tip_positions(raw_pos: np.ndarray = None, smoothed_pos: np.ndarray = None) -> None:
    """Make the pen tip positions visible for visualization in CV window."""
    global raw_pen_tip_position, smoothed_pen_tip_position
    if raw_pos is not None:
        raw_pen_tip_position = raw_pos
    if smoothed_pos is not None:
        smoothed_pen_tip_position = smoothed_pos

def _request_shutdown() -> None:
    """Signal that CV window is shutting down."""
    global cv_shutdown_requested
    cv_shutdown_requested = True

def _is_shutdown_requested() -> bool:
    """Check if CV window shutdown has been requested."""
    global cv_shutdown_requested
    return cv_shutdown_requested

def _draw_pen_tip_positions(rgb_frame, ddc_params):
    """Draw raw and smoothed pen tip positions on the CV frame."""
    global raw_pen_tip_position, smoothed_pen_tip_position
    
    frame_height, frame_width = rgb_frame.shape[:2]
    
    raw_2d = None
    smoothed_2d = None
    
    # Convert 3D positions to 2D image coordinates
    if raw_pen_tip_position is not None:
        try:
            # Ensure the position is a numpy array and has the right shape
            pos_3d = np.asarray(raw_pen_tip_position, dtype=np.float32)
            if pos_3d.size >= 3:
                pos_3d = pos_3d.flatten()[:3]  # Take first 3 elements
                # Convert from meters to mm for projection
                pos_3d = pos_3d * 1000.0
                pos_2d, _ = cv2.projectPoints(
                    pos_3d.reshape(1, 1, 3), 
                    np.zeros((3, 1)), np.zeros((3, 1)),  # No additional rotation/translation
                    ddc_params.mtx, ddc_params.dist
                )
                x, y = int(pos_2d[0, 0, 0]), int(pos_2d[0, 0, 1])
                
                # Only draw if within frame bounds
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    raw_2d = (x, y)
                    # Draw raw position as red circle
                    cv2.circle(rgb_frame, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
                    cv2.putText(rgb_frame, "Raw", (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            pass  # Ignore projection errors
    
    if smoothed_pen_tip_position is not None:
        try:
            # Ensure the position is a numpy array and has the right shape
            pos_3d = np.asarray(smoothed_pen_tip_position, dtype=np.float32)
            if pos_3d.size >= 3:
                pos_3d = pos_3d.flatten()[:3]  # Take first 3 elements
                # Convert from meters to mm for projection
                pos_3d = pos_3d * 1000.0
                pos_2d, _ = cv2.projectPoints(
                    pos_3d.reshape(1, 1, 3), 
                    np.zeros((3, 1)), np.zeros((3, 1)),  # No additional rotation/translation
                    ddc_params.mtx, ddc_params.dist
                )
                x, y = int(pos_2d[0, 0, 0]), int(pos_2d[0, 0, 1])
                
                # Only draw if within frame bounds
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    smoothed_2d = (x, y)
                    # Draw smoothed position as green circle
                    cv2.circle(rgb_frame, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
                    cv2.putText(rgb_frame, "Smoothed", (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            pass  # Ignore projection errors
    
    # Draw line connecting raw and smoothed, and show 3D distance
    if raw_2d is not None and smoothed_2d is not None and raw_pen_tip_position is not None and smoothed_pen_tip_position is not None:
        # Draw connecting line
        cv2.line(rgb_frame, raw_2d, smoothed_2d, (255, 255, 0), 2)  # Yellow line
        
        # Calculate and display 3D distance
        raw_3d = np.asarray(raw_pen_tip_position).flatten()[:3]
        smoothed_3d = np.asarray(smoothed_pen_tip_position).flatten()[:3]
        distance_m = np.linalg.norm(raw_3d - smoothed_3d)
        distance_mm = distance_m * 1000.0
        
        # Display distance at midpoint
        mid_x = (raw_2d[0] + smoothed_2d[0]) // 2
        mid_y = (raw_2d[1] + smoothed_2d[1]) // 2
        cv2.putText(rgb_frame, f"{distance_mm:.1f}mm", (mid_x, mid_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def start(headless: bool = True, cam_index: int = 0, video_file: str = None) -> None:
    """
    Main vision loop.
    Continuously captures frames, runs marker tracking, converts pose to (1,12),
    and publishes it to the shared global 'object_pose' for EKF fusion.
    
    Args:
        headless: If True, runs without displaying the video window
        cam_index: Camera index for live capture (ignored if video_file is specified)
        video_file: Path to offline video file. If specified, uses this instead of live camera.
    """
    global object_pose

    # >>> MODIFICATION: Support offline video input <<<
    if video_file is not None:
        print(f"[CV] Using offline video file: {video_file}")
        cap = cv2.VideoCapture(video_file)
    else:
        # >>> ORIGINAL: Live camera capture <<<
        cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        if video_file is not None:
            print(f"[CV] Could not open video file: {video_file}")
        else:
            print(f"[CV] Could not open camera index {cam_index}")
        return

    if not headless:
        w, h = int(cap.get(3)), int(cap.get(4))
        cv2.namedWindow("RGB-D Live Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB-D Live Stream", w, h)
        cv2.moveWindow("RGB-D Live Stream", 20, 20)

    # Load DodecaPen calibration and params
    ddc_text_data = dodecapen.txt_data()
    ddc_params    = dodecapen.parameters()
    tip_loc_cent  = np.array([0.15100563, 137.52252061, -82.07403558, 1]).reshape(4, 1)
    post = 1

    frames, dets = 0, 0
    t0 = time.time()

    try:
        while True:
            ok, rgb = cap.read()
            if not ok or rgb is None:
                time.sleep(0.01)
                continue

            # Run object tracking
            obj = tracker.object_tracking(rgb, ddc_params, ddc_text_data, post)

            if obj is not None:
                arr = np.asarray(obj, dtype=float).reshape(-1)
                if arr.size == 6:
                    # [rvec(3), tvec(3)] → [t, R]
                    rvec = arr[:3].astype(np.float64).reshape(3, 1)
                    t    = arr[3:].astype(np.float64).reshape(3,)
                    R, _ = cv2.Rodrigues(rvec)
                elif arr.size == 12:
                    t = arr[:3].astype(np.float64).reshape(3,)
                    R = arr[3:].astype(np.float64).reshape(3, 3)
                elif arr.size == 16:
                    T = arr.reshape(4, 4)
                    R = T[:3, :3].astype(np.float64)
                    t = T[:3, 3].astype(np.float64)
                else:
                    R = None; t = None

                if R is not None and t is not None:
                    row_publish = np.empty((1, 12), dtype=np.float64)
                    row_publish[0, :3] = t
                    row_publish[0, 3:] = R.reshape(9)
                    _publish_pose(row_publish)
                    dets += 1

                    if not headless and (frames % 5) != 0:
                        # Draw coordinate axes
                        rvec, _ = cv2.Rodrigues(R)
                        cv2.drawFrameAxes(
                            rgb, ddc_params.mtx, ddc_params.dist,
                            rvec.reshape(3, 1), t.reshape(3, 1), 20
                        )

            # Display / performance logging
            frames += 1
            if time.time() - t0 > 1.0:
                print(f"[CV] fps={frames}, detections={dets}")
                frames, dets, t0 = 0, 0, time.time()

            # Draw pen tip positions if available
            if not headless:
                _draw_pen_tip_positions(rgb, ddc_params)
                
            if not headless:
                cv2.imshow("RGB-D Live Stream", rgb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    _request_shutdown()
                    break
            else:
                time.sleep(0.001)

    finally:
        _request_shutdown()  # Ensure shutdown is signaled
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main(headless: bool = False, cam_index: int = 0, video_file: str = None) -> None:
    """
    Entry point for the CV tracking system.
    
    Args:
        headless: If True, runs without displaying the video window
        cam_index: Camera index for live capture (default: 0)
        video_file: Path to offline video file. If None, uses live camera.
                   Example: "offline_test.mp4" or "/path/to/video.mp4"
    """
    start(headless=headless, cam_index=cam_index, video_file=video_file)

if __name__ == "__main__":
    main(headless=False)
