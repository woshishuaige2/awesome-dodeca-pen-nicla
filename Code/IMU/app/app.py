import datetime
import json
from pathlib import Path
import time
import os
import sys
import argparse
import threading
import queue
import multiprocessing as mp
from typing import NamedTuple
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt

import numpy as np
import pandas as pd

# Vispy imports
import vispy
from vispy import scene
from vispy.io import read_mesh
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy.util import quaternion
from vispy.visuals import transforms

# Local imports
from app.color_button import ColorButton
from app.filter import DpointFilter, blend_new_data
from app.marker_tracker import CameraReading, run_tracker
from app.monitor_ble import StopCommand, StylusReading, monitor_ble
from app.dodeca_bridge import make_ekf_measurements, CENTER_TO_TIP_BODY, IMU_OFFSET_BODY, publish_pen_tip_positions, is_cv_shutdown_requested
from app import dodeca_bridge

_CODE_DIR = Path(__file__).resolve().parents[2]
_CV_DIR = _CODE_DIR / "Computer_vision"
if str(_CV_DIR) not in sys.path:
    sys.path.insert(0, str(_CV_DIR))

import run as cv_run
dodeca_bridge.dcv_run = cv_run

CANVAS_SIZE = (1080, 1080)
TRAIL_POINTS = 12000
USE_3D_LINE = False

class CameraUpdateData(NamedTuple):
    position_replace: list[np.ndarray]

class StylusUpdateData(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray
    pressure: float

ViewUpdateData = CameraUpdateData | StylusUpdateData

class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE, vsync=False)
        self.grid = self.canvas.central_widget.add_grid()
        self.view_top = self.grid.add_view(0, 0, bgcolor="white")
        self.view_top.camera = scene.TurntableCamera(up="z", fov=0, center=(0.10, 0.13, 0), elevation=90, azimuth=0, scale_factor=0.3)
        APP_DIR = Path(__file__).resolve().parent
        MESH_PATH = (APP_DIR / "mesh" / "pen.obj").resolve()
        vertices, faces, normals, texcoords = read_mesh(MESH_PATH.as_posix())
        self.pen_mesh = visuals.Mesh(vertices, faces, color=(0.8, 0.8, 0.8, 1), parent=self.view_top.scene)
        self.pen_mesh.transform = transforms.MatrixTransform()
        self.line_color = (0, 0, 0, 1)
        self.line_data_pos = np.zeros((TRAIL_POINTS, 3), dtype=np.float32)
        self.line_data_col = np.zeros((TRAIL_POINTS, 4), dtype=np.float32)
        self.trail_line = visuals.Line(width=3, parent=self.view_top.scene, method="agg", antialias=False)

    def update_data(self, new_data: ViewUpdateData):
        match new_data:
            case StylusUpdateData(position=pos, orientation=orientation, pressure=pressure):
                orientation_quat = quaternion.Quaternion(*orientation).inverse()
                self.pen_mesh.transform.matrix = orientation_quat.get_matrix() @ vispy.util.transforms.translate(pos)
                append_line_point(self.line_data_pos, pos)
            case CameraUpdateData(position_replace):
                if len(position_replace) == 0: return
                view = self.line_data_pos[-len(position_replace) :, :]
                view[:, :] = blend_new_data(view, position_replace, 1.5) # Increased alpha from 0.5 to 1.5 for smoother blending

def append_line_point(line: np.ndarray, new_point: np.array):
    line[:-1, :] = line[1:, :]
    line[-1, :] = new_point

class QueueConsumer(QtCore.QObject):
    new_data = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()
    def __init__(self, tracker_queue, imu_queue, parent=None):
        super().__init__(parent)
        self._should_end = False
        self._tracker_queue = tracker_queue
        self._imu_queue = imu_queue
        self._filter = DpointFilter(dt=1/30, smoothing_length=5, camera_delay=0) # Reverted smoothing_length to 5 for controlled testing
        self._trajectory = []

    def run_queue_consumer(self):
        print("Queue consumer is starting")
        while not self._should_end:
            if is_cv_shutdown_requested(): break
            try:
                # Efficiently drain the queue to avoid lag and processing stale data
                vis = None
                while True:
                    # In offline mode, we want to process every frame, but in real-time we'd skip.
                    # However, make_ekf_measurements currently just reads the latest global state.
                    # We add a small check to avoid tight-looping if no new data is present.
                    new_vis = make_ekf_measurements(CENTER_TO_TIP_BODY, IMU_OFFSET_BODY)
                    if new_vis is None: break
                    
                    # If this is the same timestamp as before, don't re-process
                    if vis and new_vis['timestamp'] == vis['timestamp']:
                        break
                    
                    vis = new_vis
                    # If we are lagging behind, we only care about the latest pose
                    break 

                if vis is not None:
                    # CV provides dodecahedron center position
                    # Filter tracks this position and fuses it with IMU
                    smoothed_tip_pos = self._filter.update_camera(
                        np.asarray(vis["center_pos_cam"]).flatten(),
                        np.asarray(vis["R_cam"])
                    )
                    if smoothed_tip_pos:
                        tip = smoothed_tip_pos[-1]
                        self._trajectory.append({
                            "t": time.time(),
                            "x": tip[0], "y": tip[1], "z": tip[2]
                        })
                        self.new_data.emit(CameraUpdateData(position_replace=smoothed_tip_pos))
            except Exception as e:
                print(f"[QueueConsumer] Error: {e}")
            time.sleep(0.01)
        
        print("Queue consumer finishing")
        if self._trajectory:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            df = pd.DataFrame(self._trajectory)
            df.to_csv(output_dir / "trajectory.csv", index=False)
            print(f"Trajectory saved to {output_dir / 'trajectory.csv'}")
        self.finished.emit()

    def stop_data(self):
        self._should_end = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="normal")
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    has_display = "DISPLAY" in os.environ
    # A QApplication instance is needed for QThread even in headless mode.
    # We manage its lifecycle to ensure clean shutdown.
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)
    
    if not has_display:
        print("No display detected, running in headless mode")

    tracker_queue = mp.Queue()
    ble_queue = mp.Queue()
    ble_command_queue = mp.Queue()
    
    if has_display:
        canvas_wrapper = CanvasWrapper()
        data_thread = QtCore.QThread()
    else:
        canvas_wrapper = None
        data_thread = QtCore.QThread()

    queue_consumer = QueueConsumer(tracker_queue, ble_queue)
    queue_consumer.moveToThread(data_thread)

    # Use live camera (video_file=None) unless --video is specified
    video_file = args.video
    # Set daemon to False to ensure it's handled gracefully
    cv_thread = threading.Thread(target=cv_run.main, kwargs={"headless": True, "cam_index": 0, "video_file": video_file}, daemon=False)
    cv_thread.start()

    # Start BLE monitoring unless in cv_only mode
    ble_thread = None
    if args.mode != "cv_only":
        ble_thread = threading.Thread(target=monitor_ble, args=(ble_queue, ble_command_queue), daemon=False)
        ble_thread.start()
        print("BLE monitoring started")

    data_thread.started.connect(queue_consumer.run_queue_consumer)
    queue_consumer.finished.connect(data_thread.quit)
    
    try:
        data_thread.start()
        # Run for a fixed duration or until video ends
        start_time = time.time()
        while data_thread.isRunning():
            # Check if the CV thread is alive. If not, we stop the consumer.
            if not cv_thread.is_alive():
                queue_consumer.stop_data()
                # Give the consumer a moment to process the remaining queue and emit finished
                time.sleep(0.5)
                break
            
            # Timeout safety
            if time.time() - start_time > 300: # 5 minute hard timeout for long videos
                print("[Main] Hard timeout reached, stopping...")
                queue_consumer.stop_data()
                break
                
            time.sleep(0.1)
    finally:
        # Stop BLE thread if it was started
        if ble_thread is not None and ble_thread.is_alive():
            ble_command_queue.put(StopCommand())
            ble_thread.join(timeout=2.0)
            print("BLE monitoring stopped")
        
        # Forceful cleanup to ensure the process exits
        if data_thread.isRunning():
            data_thread.quit()
            data_thread.wait(1000)
        
        # In headless mode, we should exit explicitly
        if not has_display:
            print("[Main] Analysis complete. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    main()