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


def terminal_stop_requested():
    """Return True if q was pressed in the terminal on Windows."""
    if os.name != "nt":
        return False
    try:
        import msvcrt
    except ImportError:
        return False
    if not msvcrt.kbhit():
        return False
    key = msvcrt.getwch()
    return key.lower() == "q"


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
        self._logged_first_format = False
        self.skipped_unsynced = 0
        
    def record_imu(self, ble_queue):
        print("[Recorder] IMU recording started.")
        imu_count = 0
        skipped_unsynced = 0
        while not self.should_stop:
            try:
                reading = ble_queue.get(timeout=0.1)
                aligned_host_time = getattr(reading, "aligned_host_time", None)
                if aligned_host_time is None:
                    skipped_unsynced += 1
                    self.skipped_unsynced = skipped_unsynced
                    if skipped_unsynced == 1:
                        print("[Recorder] Waiting for IMU/host clock sync before saving samples...")
                    continue

                if hasattr(reading, "to_json"):
                    reading_dict = reading.to_json()
                else:
                    reading_dict = {
                        "accel": None if reading.accel is None else reading.accel.tolist(),
                        "gyro": None if reading.gyro is None else reading.gyro.tolist(),
                        "mag": None if reading.mag is None else reading.mag.tolist(),
                        "quat": None if reading.quat is None else reading.quat.tolist(),
                        "t": reading.t,
                        "pressure": reading.pressure,
                        "aligned_host_time": aligned_host_time,
                    }
                reading_dict["aligned_host_time"] = aligned_host_time
                reading_dict["local_timestamp"] = aligned_host_time
                self.data["imu_readings"].append(reading_dict)

                if not self._logged_first_format:
                    if skipped_unsynced:
                        print(f"[Recorder] Skipped {skipped_unsynced} unsynced IMU samples.")
                    keys = sorted(reading_dict.keys())
                    print(f"[Recorder] First IMU payload keys: {keys}")
                    if reading_dict.get("quat") is not None:
                        print("[Recorder] Detected quaternion IMU packet format.")
                    else:
                        print("[Recorder] Detected legacy raw-IMU BLE packet format.")
                    self._logged_first_format = True
                
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
        self.data["metadata"]["skipped_unsynced_imu_count"] = self.skipped_unsynced

        timing = self._imu_timing_summary()
        if timing:
            self.data["metadata"]["imu_timing"] = timing
        
        os.makedirs(os.path.dirname(self.imu_output), exist_ok=True)
        with open(self.imu_output, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"\n[Recorder] IMU data saved to {self.imu_output}")
        print(f"[Recorder] Skipped unsynced IMU samples: {self.skipped_unsynced}")
        if not self.data["imu_readings"]:
            print("[Recorder] Warning: no IMU samples were recorded.")
        if timing:
            print(
                "[Recorder] IMU timing: "
                f"mean={timing['mean_rate_hz']:.1f} Hz, "
                f"median={timing['median_rate_hz']:.1f} Hz, "
                f"p95 gap={timing['p95_gap_ms']:.0f} ms, "
                f"max gap={timing['max_gap_ms']:.0f} ms, "
                f">100ms={timing['gaps_over_100ms']}, "
                f">200ms={timing['gaps_over_200ms']}"
            )

    def _imu_timing_summary(self):
        readings = self.data["imu_readings"]
        timestamps = [
            float(reading["t"]) / 1000.0
            for reading in readings
            if reading.get("t") is not None
        ]
        if len(timestamps) < 2:
            return None

        deltas = np.diff(np.asarray(timestamps, dtype=float))
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return None

        median_delta = float(np.median(deltas))
        summary = {
            "duration_s": float(duration),
            "mean_rate_hz": float((len(timestamps) - 1) / duration),
            "median_rate_hz": float(1.0 / median_delta) if median_delta > 0 else 0.0,
            "p95_gap_ms": float(np.percentile(deltas, 95) * 1000.0),
            "max_gap_ms": float(np.max(deltas) * 1000.0),
            "gaps_over_100ms": int(np.sum(deltas > 0.100)),
            "gaps_over_200ms": int(np.sum(deltas > 0.200)),
        }

        return summary

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record raw IMU data and capture video")
    parser.add_argument("--imu", default="outputs/imu_data.json", help="Output IMU JSON file")
    parser.add_argument("--video", default="outputs/video.mp4", help="Output video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--transport",
        choices=["ble", "usb"],
        default="ble",
        help="IMU transport to use.",
    )
    parser.add_argument("--port", help="Serial port for --transport usb, for example COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate for USB IMU transport")
    args = parser.parse_args()

    if args.transport == "usb":
        if not args.port:
            print("Error: --port is required when using --transport usb")
            return
        try:
            from app.monitor_usb import list_serial_ports, open_usb_serial
        except ImportError:
            print(
                "Error: pyserial is required for USB IMU transport. "
                "Install it with: python -m pip install pyserial"
            )
            return
        try:
            print(f"Opening USB serial IMU on {args.port} at {args.baud} baud...")
            usb_serial_port = open_usb_serial(args.port, args.baud)
            # Opening a USB CDC serial port can reset the Nicla. Give the sketch
            # a moment to restart before dropping banners/partial lines.
            time.sleep(2.0)
            usb_serial_port.reset_input_buffer()
            print("USB serial IMU connected!")
        except Exception as exc:
            ports = list_serial_ports()
            port_text = ", ".join(ports) if ports else "none detected"
            print(f"Error: could not open {args.port}: {exc}")
            print(f"Available serial ports: {port_text}")
            print("Close Arduino Serial Monitor or any other app using the port, then retry.")
            return
    else:
        usb_serial_port = None

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
    recorder.data["metadata"]["imu_transport"] = args.transport
    if args.transport == "usb":
        recorder.data["metadata"]["imu_serial_port"] = args.port
        recorder.data["metadata"]["imu_serial_baud"] = args.baud
    
    # Start IMU monitoring
    imu_queue = mp.Queue()
    imu_command_queue = mp.Queue()
    if args.transport == "usb":
        from app.monitor_usb import monitor_usb_serial

        imu_monitor_target = monitor_usb_serial
        imu_monitor_args = (imu_queue, imu_command_queue, usb_serial_port)
    else:
        imu_monitor_target = monitor_ble
        imu_monitor_args = (imu_queue, imu_command_queue)

    imu_monitor_thread = threading.Thread(
        target=imu_monitor_target,
        args=imu_monitor_args,
        daemon=True,
        name=f"{args.transport}-imu-monitor",
    )
    imu_monitor_thread.start()

    # Start IMU recording thread
    imu_rec_thread = threading.Thread(
        target=recorder.record_imu,
        args=(imu_queue,),
        daemon=True,
        name="imu-recorder",
    )
    imu_rec_thread.start()

    print("\n=== Recording Started ===")
    print(f"Saving video to: {args.video}")
    print(f"Saving IMU data to: {args.imu}")
    print("Press 'q' in the camera window or terminal, or Ctrl+C, to stop recording\n")

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

            if (cv2.waitKey(1) & 0xFF == ord('q')) or terminal_stop_requested():
                print("\n[Main] Stopping...")
                break
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        recorder.should_stop = True
        
        # Cleanup
        imu_command_queue.put(StopCommand())
        imu_rec_thread.join(timeout=2.0)
        imu_monitor_thread.join(timeout=2.0)
        
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()
        
        # Finalize sync info in video metadata before saving
        offset = get_sync_offset()
        if offset is not None:
            recorder.data["video_metadata"]["sync_offset"] = offset
            
        recorder.save_imu()
        print(f"[Recorder] Video saved to {args.video}")

        # Multiprocessing queues create feeder threads on Windows; closing and
        # joining them avoids the process hanging after all visible work is done.
        imu_queue.close()
        imu_queue.join_thread()
        imu_command_queue.close()
        imu_command_queue.join_thread()

if __name__ == "__main__":
    main()
