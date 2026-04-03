"""
Offline CV Processing for Recorded Video
Processes a recorded video file and generates CV data in my_data.json format.
This uses the same CV pipeline that works well for offline videos.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# Add project directories to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "IMU"))
sys.path.append(str(repo_root / "Computer_vision" / "src"))

try:
    from app.dodeca_bridge import CENTER_TO_TIP_BODY, IMU_OFFSET_BODY
    from filter import OneEuroFilter
    import src.DoDecahedronUtils as dodecapen
    import src.Tracker as tracker
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure you are running this from the Code/IMU directory or paths are correct.")
    sys.exit(1)


class OfflineCVProcessor:
    def __init__(self, video_path, output_file="cv_data.json", apply_filter=True):
        self.video_path = Path(video_path)
        self.output_file = output_file
        self.apply_filter = apply_filter
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.data = {
            "metadata": {
                "start_time": time.time(),
                "tip_offset_body": CENTER_TO_TIP_BODY.tolist(),
                "imu_to_tip_body": IMU_OFFSET_BODY.tolist(),
                "filtered": apply_filter,
                "filter_type": "OneEuro" if apply_filter else "None",
                "video_source": str(video_path)
            },
            "cv_readings": []
        }
        
        # One-Euro filters (initialized on first reading)
        self.filters_initialized = False
        self.filter_x = None
        self.filter_y = None
        self.filter_z = None
        self.filter_qw = None
        self.filter_qx = None
        self.filter_qy = None
        self.filter_qz = None
    
    def process_video(self, video_start_timestamp=None, t_cv_start_system=None, sync_offset=None):
        """
        Process the video file and extract CV data.
        Args:
            video_start_timestamp: The absolute system timestamp when the video recording started.
            t_cv_start_system: The monotonic system timestamp when the video recording started.
            sync_offset: The offset (t_sensor - t_system) established during recording.
        """
        print(f"[CV Processor] Opening video: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"[CV Processor] Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        # Load DodecaPen calibration
        ddc_text_data = dodecapen.txt_data()
        ddc_params = dodecapen.parameters()
        post = 1
        
        frame_count = 0
        detection_count = 0
        start_time_proc = time.time()
        
        # If no start timestamp is provided, we use 0 (the merge script will handle alignment)
        if video_start_timestamp is None:
            video_start_timestamp = 0
            print("[CV Processor] No video start timestamp provided. Using 0.")
        else:
            print(f"[CV Processor] Using provided video start timestamp: {video_start_timestamp:.3f}")
        
        # Use sync_offset if provided to establish master clock timestamps
        if t_cv_start_system is not None and sync_offset is not None:
            print(f"[CV Processor] Using Master Clock sync:")
            print(f"  t_cv_start_system: {t_cv_start_system:.6f}")
            print(f"  sync_offset: {sync_offset:.6f}")
            self.data["metadata"]["master_clock"] = "IMU_SENSOR"
            self.data["metadata"]["sync_offset"] = sync_offset
            self.data["metadata"]["t_cv_start_system"] = t_cv_start_system
            
        # Update metadata
        self.data["metadata"]["start_time"] = video_start_timestamp
        
        print("[CV Processor] Processing frames...")
        
        try:
            while True:
                ret, rgb = cap.read()
                if not ret or rgb is None:
                    break
                
                frame_count += 1
                
                # Calculate timestamp based on frame number and FPS
                if t_cv_start_system is not None and sync_offset is not None:
                    # Master Clock domain: t_sensor = (t_cv_start_system + frame_index / fps) + offset
                    frame_timestamp = (t_cv_start_system + (frame_count / fps)) + sync_offset
                else:
                    # Fallback to absolute system time
                    frame_timestamp = video_start_timestamp + (frame_count / fps)
                
                # Run object tracking
                obj = tracker.object_tracking(rgb, ddc_params, ddc_text_data, post)
                
                if obj is not None:
                    arr = np.asarray(obj, dtype=float).reshape(-1)
                    
                    # Parse tracking result
                    if arr.size == 6:
                        # [rvec(3), tvec(3)] → [t, R]
                        rvec = arr[:3].astype(np.float64).reshape(3, 1)
                        tvec = arr[3:].astype(np.float64).reshape(3,)
                        R_cam, _ = cv2.Rodrigues(rvec)
                    elif arr.size == 12:
                        tvec = arr[:3].astype(np.float64).reshape(3,)
                        R_cam = arr[3:].astype(np.float64).reshape(3, 3)
                    elif arr.size == 16:
                        T = arr.reshape(4, 4)
                        R_cam = T[:3, :3].astype(np.float64)
                        tvec = T[:3, 3].astype(np.float64)
                    else:
                        continue
                    
                    # Convert tvec from mm to meters (dodecapen outputs in mm)
                    center_pos = tvec / 1000.0
                    
                    # Convert rotation matrix to quaternion [w, x, y, z]
                    quat_xyzw = R.from_matrix(R_cam).as_quat()  # [x, y, z, w]
                    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
                    
                    # Apply One-Euro filter if enabled
                    if self.apply_filter:
                        if not self.filters_initialized:
                            # Initialize filters on first detection
                            self.filter_x = OneEuroFilter(frame_timestamp, center_pos[0])
                            self.filter_y = OneEuroFilter(frame_timestamp, center_pos[1])
                            self.filter_z = OneEuroFilter(frame_timestamp, center_pos[2])
                            self.filter_qw = OneEuroFilter(frame_timestamp, quat[0])
                            self.filter_qx = OneEuroFilter(frame_timestamp, quat[1])
                            self.filter_qy = OneEuroFilter(frame_timestamp, quat[2])
                            self.filter_qz = OneEuroFilter(frame_timestamp, quat[3])
                            self.filters_initialized = True
                            
                            # Store first reading as-is
                            filtered_center = center_pos
                            filtered_R = R_cam
                        else:
                            # Apply One-Euro filter
                            filtered_x = self.filter_x.filter_signal(frame_timestamp, center_pos[0])
                            filtered_y = self.filter_y.filter_signal(frame_timestamp, center_pos[1])
                            filtered_z = self.filter_z.filter_signal(frame_timestamp, center_pos[2])
                            
                            filtered_qw = self.filter_qw.filter_signal(frame_timestamp, quat[0])
                            filtered_qx = self.filter_qx.filter_signal(frame_timestamp, quat[1])
                            filtered_qy = self.filter_qy.filter_signal(frame_timestamp, quat[2])
                            filtered_qz = self.filter_qz.filter_signal(frame_timestamp, quat[3])
                            
                            # Normalize quaternion
                            filtered_quat = np.array([filtered_qw, filtered_qx, filtered_qy, filtered_qz])
                            quat_norm = np.linalg.norm(filtered_quat)
                            if quat_norm > 1e-6:
                                filtered_quat = filtered_quat / quat_norm
                            else:
                                filtered_quat = quat  # Fallback to unfiltered
                            
                            # Reconstruct rotation matrix
                            filtered_R = R.from_quat([filtered_quat[1], filtered_quat[2], 
                                                       filtered_quat[3], filtered_quat[0]]).as_matrix()
                            
                            filtered_center = np.array([filtered_x, filtered_y, filtered_z])
                    else:
                        # No filtering
                        filtered_center = center_pos
                        filtered_R = R_cam
                    
                    # Calculate tip position in camera frame
                    # Tip = Center + R_cam @ CENTER_TO_TIP_BODY
                    filtered_tip = filtered_center + filtered_R @ CENTER_TO_TIP_BODY
                    
                    # Create CV reading entry (matching my_data.json structure)
                    # We include both center and tip positions for full compatibility
                    cv_entry = {
                        "timestamp": frame_timestamp,
                        "local_timestamp": frame_timestamp,
                        "center_pos_cam": filtered_center.tolist(),
                        "imu_pos_cam": filtered_center.tolist(), # Backward compatibility
                        "tip_pos_cam": filtered_tip.tolist(),
                        "R_cam": filtered_R.tolist(),
                    }
                    
                    self.data["cv_readings"].append(cv_entry)
                    detection_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"[CV Processor] Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - Detections: {detection_count}")
        
        finally:
            cap.release()
        
        processing_time = time.time() - start_time_proc
        
        print(f"\n[CV Processor] Processing complete:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Successful detections: {detection_count}")
        print(f"  Detection rate: {(detection_count/frame_count)*100:.1f}%")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Processing speed: {frame_count/processing_time:.1f} FPS")
    
    def save(self):
        """Save CV data to JSON file"""
        self.data["metadata"]["end_time"] = time.time()
        self.data["metadata"]["cv_count"] = len(self.data["cv_readings"])
        
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=2)
        
        print(f"\n[CV Processor] CV data saved to: {output_path}")
        print(f"[CV Processor] Total CV readings: {len(self.data['cv_readings'])}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process video file to generate CV data")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="outputs/cv_data.json", help="Output CV data JSON file")
    parser.add_argument("--no-filter", action="store_true", help="Disable One-Euro filtering")
    args = parser.parse_args()
    
    processor = OfflineCVProcessor(
        args.video,
        args.output,
        apply_filter=not args.no_filter
    )
    
    # If processing the default outputs/video.mp4, try to find its start time from imu_data.json
    video_start_time = None
    t_cv_start_system = None
    sync_offset = None
    
    if Path(args.video).name == "video.mp4":
        imu_json = Path(args.video).parent / "imu_data.json"
        if imu_json.exists():
            try:
                with open(imu_json, 'r') as f:
                    imu_data = json.load(f)
                    video_start_time = imu_data.get("metadata", {}).get("start_time")
                    
                    # Try to get Master Clock sync info
                    video_meta = imu_data.get("video_metadata", {})
                    t_cv_start_system = video_meta.get("t_cv_start_system")
                    sync_offset = video_meta.get("sync_offset")
                    
                    if video_start_time:
                        print(f"[CV Processor] Found video start time: {video_start_time}")
                    if t_cv_start_system and sync_offset:
                        print(f"[CV Processor] Found Master Clock sync info in imu_data.json")
            except Exception as e:
                print(f"[CV Processor] Error reading imu_data.json: {e}")
                pass

    processor.process_video(
        video_start_timestamp=video_start_time,
        t_cv_start_system=t_cv_start_system,
        sync_offset=sync_offset
    )
    processor.save()


if __name__ == "__main__":
    main()
