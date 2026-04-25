"""
Enhanced Raw Data Recorder with Video Recording
Records raw IMU data and captures raw video for offline processing.
This avoids real-time CV processing overhead and ensures high-quality trajectories.
"""

import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

# Add project directories to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "IMU"))

# Try to import project modules
try:
    from app.monitor_ble import monitor_ble, StopCommand, get_sync_offset
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure you are running this from the Code/IMU directory or paths are correct.")
    sys.exit(1)

class RawDataRecorder:
    def __init__(self, imu_output="outputs/imu_data.json", video_output="outputs/video.mp4"):
        self.imu_output = imu_output
        self.video_output = video_output
        self.data = {
            "metadata": {
                "start_time": time.time(),
                "filtered": False,
                "note": "Raw IMU and Video recording for offline processing",
                "sync_info": {}
            },
            "imu_readings": [],
            "video_metadata": {
                "fps": 30,
                "t_cv_start_system": 0,
                "sync_offset": 0
            }
        }
        self.should_stop = False
        
    def record_imu(self, ble_queue):
        print("[Recorder] IMU recording started.")
        imu_count = 0
        while not self.should_stop:
            try:
                reading = ble_queue.get(timeout=0.1)
                if hasattr(reading, "to_json"):
                    reading_dict = reading.to_json()
                else:
                    reading_dict = {
                        "accel": reading.accel.tolist(),
                        "gyro": reading.gyro.tolist(),
                        "mag": None if reading.mag is None else reading.mag.tolist(),
                        "t": reading.t,
                        "pressure": reading.pressure
                    }
                # Store with absolute system timestamp for fallback/reference
                reading_dict["local_timestamp"] = time.time()
                self.data["imu_readings"].append(reading_dict)
                
                # Update metadata if offset was just established
                if not self.data["metadata"]["sync_info"]:
                    offset = get_sync_offset()
                    if offset is not None:
                        self.data["metadata"]["sync_info"] = {
                            "offset": offset,
                            "master_clock": "IMU_SENSOR"
                        }
                imu_count += 1
                if imu_count % 100 == 0:
                    print(f"[Recorder] IMU: received {imu_count} samples")
            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] IMU Error: {e}")

    def save_imu(self):
        self.data["metadata"]["end_time"] = time.time()
        self.data["metadata"]["imu_count"] = len(self.data["imu_readings"])
        
        os.makedirs(os.path.dirname(self.imu_output), exist_ok=True)
        with open(self.imu_output, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"\n[Recorder] IMU data saved to {self.imu_output}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record raw IMU data and capture video")
    parser.add_argument("--imu", default="outputs/imu_data.json", help="Output IMU JSON file")
    parser.add_argument("--video", default="outputs/video.mp4", help="Output video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.imu).parent.mkdir(parents=True, exist_ok=True)
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    # Initialize Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Standard recording FPS

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(args.video, fourcc, fps, (width, height))

    # CRITICAL: Record the exact start time for both IMU and Video
    t_cv_start_system = time.monotonic()
    
    recorder = RawDataRecorder(args.imu, args.video)
    recorder.data["video_metadata"]["fps"] = fps
    recorder.data["video_metadata"]["t_cv_start_system"] = t_cv_start_system
    
    # Start BLE monitoring
    ble_queue = mp.Queue()
    ble_command_queue = mp.Queue()
    ble_thread = threading.Thread(
        target=monitor_ble, 
        args=(ble_queue, ble_command_queue),
        daemon=True
    )
    ble_thread.start()

    # Start IMU recording thread
    imu_rec_thread = threading.Thread(target=recorder.record_imu, args=(ble_queue,))
    imu_rec_thread.start()

    print("\n=== Recording Started ===")
    print(f"Saving video to: {args.video}")
    print(f"Saving IMU data to: {args.imu}")
    print("Press 'q' in the camera window or Ctrl+C to stop recording\n")

    try:
        while not recorder.should_stop:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame to video
            video_out.write(frame)

            # Display preview
            cv2.putText(frame, "RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Dodeca-pen Recording', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        recorder.should_stop = True
        
        # Cleanup
        ble_command_queue.put(StopCommand())
        imu_rec_thread.join(timeout=1.0)
        
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()
        
        # Finalize sync info in video metadata before saving
        offset = get_sync_offset()
        if offset is not None:
            recorder.data["video_metadata"]["sync_offset"] = offset
            
        recorder.save_imu()
        print(f"[Recorder] Video saved to {args.video}")

if __name__ == "__main__":
    main()
