# Dodeca Pen: Detailed Execution Trace

This document provides a step-by-step execution trace of the Dodeca Pen system, starting from `Code/IMU/main.py`. It includes data flow, coordinate frames, transformations, and real-time dependencies.

## Table of Contents

1. [Entry Point and Initialization](#1-entry-point-and-initialization)
2. [Process and Thread Architecture](#2-process-and-thread-architecture)
3. [Data Ingestion](#3-data-ingestion)
4. [Data Fusion Pipeline](#4-data-fusion-pipeline)
5. [Coordinate Frames and Transformations](#5-coordinate-frames-and-transformations)
6. [Real-Time Dependencies](#6-real-time-dependencies)
7. [Key Assumptions](#7-key-assumptions)

---

## 1. Entry Point and Initialization

### 1.1 `Code/IMU/main.py` (Lines 1-6)

The application starts here with a simple wrapper:

```python
from app.app import main

if __name__ == "__main__":
    main()
```

**Action:** Immediately calls `main()` from `app.app`.

### 1.2 `Code/IMU/app/app.py::main()` (Lines 431-506)

This is the true entry point. The function performs the following initialization:

**Lines 432-436: Environment Setup**
```python
np.set_printoptions(precision=3, suppress=True, formatter={"float": "{: >5.2f}".format})
app = use_app("pyqt6")
app.create()
```
- Sets NumPy print options for debugging
- Initializes the PyQt6 application framework
- Creates the VisPy application context

**Lines 452-456: Queue Creation**
```python
tracker_queue = mp.Queue()
ble_queue = mp.Queue()
ble_command_queue = mp.Queue()
canvas_wrapper = CanvasWrapper()
win = MainWindow(canvas_wrapper)
```
- Creates multiprocessing queues for inter-process communication
- Initializes the 3D visualization canvas
- Creates the main window (kept hidden to avoid lag)

**Lines 458-461: Data Consumer Thread**
```python
data_thread = QtCore.QThread(parent=win)
queue_consumer = QueueConsumer(tracker_queue, ble_queue)
queue_consumer.moveToThread(data_thread)
```
- Creates a Qt thread for data processing
- Instantiates the `QueueConsumer` object (the data fusion engine)
- Moves the consumer to the separate thread

---

## 2. Process and Thread Architecture

The system uses a hybrid multiprocessing and multithreading architecture:

### 2.1 BLE Process (Lines 472-475)

```python
ble_process = mp.Process(
    target=monitor_ble, args=(ble_queue, ble_command_queue), daemon=False
)
ble_process.start()
```

**Purpose:** Connects to the Dodeca Pen via Bluetooth Low Energy and streams IMU data.

**Implementation:** `Code/IMU/app/monitor_ble.py`
- Runs in a separate process to avoid blocking the main thread
- Uses the `bleak` library for BLE communication
- Searches for a device named "DPOINT"
- Subscribes to a specific BLE characteristic UUID
- Unpacks binary IMU data packets and puts them into `ble_queue`

### 2.2 CV Thread (Lines 467-478)

```python
OFFLINE_VIDEO_FILE = None  # <<< CHANGE THIS to enable offline mode

cv_thread = threading.Thread(
    target=cv_run.main, 
    kwargs={"headless": False, "cam_index": 0, "video_file": OFFLINE_VIDEO_FILE}, 
    daemon=True
)
cv_thread.start()
```

**Purpose:** Captures camera frames and tracks ArUco markers on the pen.

**Implementation:** `Code/Computer_vision/run.py`
- Runs in a separate thread (not a process) to share memory with the main thread
- Opens the camera using OpenCV (`cv2.VideoCapture`)
- Continuously reads frames and detects ArUco markers
- Estimates the 6-DoF pose of the pen's center
- Stores the pose in a global variable `object_pose` that can be read by the main thread

**Key Insight:** This thread runs in the same process as the main application, allowing the `dodeca_bridge` module to directly access the `object_pose` variable without inter-process communication.

### 2.3 Data Consumer Thread (Lines 479-495)

```python
queue_consumer.new_data.connect(canvas_wrapper.update_data)
data_thread.started.connect(queue_consumer.run_queue_consumer)
queue_consumer.finished.connect(data_thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
win.closing.connect(queue_consumer.stop_data, QtCore.Qt.ConnectionType.DirectConnection)
```

**Purpose:** Fuses IMU and camera data using an Extended Kalman Filter.

**Implementation:** `Code/IMU/app/app.py::QueueConsumer`
- Runs in a Qt thread
- Continuously polls the `ble_queue` for IMU data
- Periodically calls `dodeca_bridge.make_ekf_measurements()` to get camera data
- Updates the EKF with both data sources
- Emits signals to update the 3D visualization

---

## 3. Data Ingestion

### 3.1 IMU Data Flow

**Source:** Dodeca Pen hardware (Arduino-based IMU)

**Path:**
1. **Hardware → BLE:** The pen's microcontroller reads the IMU (accelerometer + gyroscope) and pressure sensor, packs the data into a binary struct, and transmits it via BLE.

2. **BLE → `monitor_ble.py`:** The `monitor_ble_async()` function (lines 64-100) receives BLE notifications and calls `unpack_imu_data_packet()` (lines 53-58).

3. **Unpacking:** The binary packet is unpacked into a `StylusReading` object:
   ```python
   class StylusReading(NamedTuple):
       accel: np.ndarray  # (3,) in m/s²
       gyro: np.ndarray   # (3,) in rad/s
       t: int             # timestamp (currently unused, set to 0)
       pressure: float    # normalized [0, 1]
   ```

4. **Queue → Data Consumer:** The `StylusReading` is placed in `ble_queue` and retrieved by `QueueConsumer.run_queue_consumer()` (lines 357-391).

**Data Format:**
- **Accelerometer:** 3-axis, in m/s² (after calibration)
- **Gyroscope:** 3-axis, in rad/s (after calibration)
- **Pressure:** Single value, normalized to [0, 1]

**Frequency:** Approximately 100-200 Hz (depends on BLE and microcontroller settings)

### 3.2 Camera Data Flow

**Source:** USB webcam (or offline video file)

**Path:**
1. **Camera → OpenCV:** `cv2.VideoCapture` reads frames from the camera or video file.

2. **Frame → Marker Detection:** Each frame is passed to `tracker.object_tracking()` in `Code/Computer_vision/src/Tracker.py` (lines 6-101).

3. **Marker Detection:**
   - Converts frame to grayscale
   - Uses `cv2.aruco.detectMarkers()` to find ArUco markers
   - For each detected marker, estimates its 6-DoF pose using `cv2.aruco.estimatePoseSingleMarkers()`
   - Transforms marker poses to the pen's center coordinate frame
   - Averages multiple marker estimates to get a robust center pose

4. **Pose → Global Variable:** The estimated pose is stored in `run.py::object_pose` (line 25):
   ```python
   object_pose: np.ndarray | None = None  # shape (1, 12)
   ```
   Format: `[tx, ty, tz, r00, r01, r02, r10, r11, r12, r20, r21, r22]`
   - `tx, ty, tz`: Translation in mm (camera frame)
   - `r00...r22`: 3x3 rotation matrix (camera frame)

5. **Bridge → Data Consumer:** `dodeca_bridge.make_ekf_measurements()` reads `object_pose` and transforms it into EKF-friendly measurements (lines 309-354 in `app.py`).

**Data Format:**
- **Position:** (3,) in mm, then converted to meters
- **Orientation:** (3, 3) rotation matrix

**Frequency:** Approximately 10-30 Hz (depends on camera frame rate and processing time)

---

## 4. Data Fusion Pipeline

### 4.1 Extended Kalman Filter (EKF)

The EKF is implemented in `Code/IMU/app/filter.py` and `Code/IMU/app/filter_core.py`.

**State Vector (21 dimensions):**
```python
state = [
    pos_x, pos_y, pos_z,           # Position (m)
    vel_x, vel_y, vel_z,           # Velocity (m/s)
    acc_x, acc_y, acc_z,           # Acceleration (m/s²)
    av_x, av_y, av_z,              # Angular velocity (rad/s)
    quat_w, quat_x, quat_y, quat_z, # Orientation (quaternion)
    accbias_x, accbias_y, accbias_z, # Accelerometer bias (m/s²)
    gyrobias_x, gyrobias_y, gyrobias_z # Gyroscope bias (rad/s)
]
```

**Note:** Due to `IMU_TO_TIP_BODY = [0, 0, 0]` in `dodeca_bridge.py`, the filter tracks the pen tip position directly.

### 4.2 Prediction Step (IMU Update)

**Trigger:** New IMU data arrives in `ble_queue`

**Process:** `QueueConsumer.run_queue_consumer()` (lines 357-391)
1. Retrieve `StylusReading` from `ble_queue`
2. Call `self._filter.update_imu(reading.accel, reading.gyro)`
3. Inside `filter.py::DpointFilter.update_imu()` (lines 111-126):
   - Predict the next state using the motion model: `ekf_predict()`
   - Fuse the IMU measurement: `fuse_imu()`
   - Append the state to the history buffer

**Motion Model:**
- Integrates acceleration to update velocity and position
- Integrates angular velocity to update orientation (quaternion)
- Accounts for accelerometer and gyroscope biases

**Noise Model:**
```python
accel_noise = 2e-3
gyro_noise = 5e-4
```

### 4.3 Correction Step (Camera Update)

**Trigger:** New camera pose is available (checked every loop iteration)

**Process:** `QueueConsumer.run_queue_consumer()` (lines 309-354)
1. Call `dodeca_bridge.make_ekf_measurements()`
2. Inside `dodeca_bridge.py::make_ekf_measurements()` (lines 55-87):
   - Read `object_pose` from `run.py`
   - Extract translation `t_cam` (mm) and rotation `R_cam` (3x3 matrix)
   - Convert translation to meters: `t_cam_m = t_cam * 0.001`
   - Transform pen center pose to IMU pose:
     ```python
     tip_pos_cam = t_cam_m + R_cam @ TIP_OFFSET_BODY
     imu_pos_cam = t_cam_m + R_cam @ (TIP_OFFSET_BODY - IMU_TO_TIP_BODY)
     ```
   - Convert rotation matrix to quaternion: `q_cam = _rotmat_to_quat(R_cam)`
   - Return a dictionary with `imu_pos_cam`, `R_cam`, `q_cam`, etc.

3. Call `self._filter.update_camera(imu_pos, orientation_mat)`
4. Inside `filter.py::DpointFilter.update_camera()` (lines 128-192):
   - **Rollback:** Remove recent IMU updates from history (to account for camera delay)
   - **Fuse:** Incorporate the camera measurement at the correct time: `fuse_camera()`
   - **Smooth:** Apply Rauch-Tung-Striebel (RTS) smoothing to past states: `ekf_smooth()`
   - **Replay:** Re-apply the rolled-back IMU updates
   - Return smoothed tip positions for visualization

**Noise Model:**
```python
camera_noise_pos = 1e-6  # Very low (camera is trusted)
camera_noise_or = 1e-4
```

**Outlier Rejection:**
If the camera measurement is too far from the current state estimate, the filter resets:
```python
if pos_error > 0.05 or or_error > 0.4:
    print(f"Resetting state, errors: pos={pos_error:.4f}m, or={or_error:.4f}rad")
    self.fs = initial_state(imu_pos, or_quat_smoothed)
```

### 4.4 Smoothing

The system uses **Rauch-Tung-Striebel (RTS) smoothing** to refine past state estimates after a camera measurement arrives. This is a backward pass through the filter history that propagates information from the camera measurement back in time.

**Purpose:** Correct for IMU drift that accumulated between camera measurements.

**Implementation:** `filter_core.py::ekf_smooth()` (called from `filter.py` line 168)

---

## 5. Coordinate Frames and Transformations

### 5.1 Coordinate Frames

The system uses multiple coordinate frames:

1. **Camera Frame:**
   - Origin: Camera optical center
   - Axes: Standard OpenCV convention (X right, Y down, Z forward)
   - Units: mm (from ArUco pose estimation), converted to meters for EKF

2. **Pen Body Frame:**
   - Origin: Center of the dodecahedron (pen body)
   - Axes: Aligned with the pen's physical structure
   - Units: mm (for geometry), meters (for EKF)

3. **IMU Frame:**
   - Origin: Location of the IMU sensor inside the pen
   - Axes: Aligned with the IMU chip's axes
   - Units: m/s² (accel), rad/s (gyro)

4. **Pen Tip Frame:**
   - Origin: Physical tip of the pen
   - Axes: Same as pen body frame
   - Units: meters (for EKF)

### 5.2 Key Transformations

**Pen Center → Pen Tip (Body Frame):**
```python
TIP_OFFSET_BODY = np.array([0.0, 137.52252061, -82.07403558]) * 1e-3  # meters
```
This vector represents the offset from the dodecahedron center to the pen tip in the pen's body frame.

**IMU → Pen Tip (Body Frame):**
```python
IMU_TO_TIP_BODY = np.array([0.0, 0.0, 0.0])  # meters
```
**Important:** This is currently set to zero, meaning the system assumes the IMU is at the pen tip. In reality, the IMU is inside the pen body, but this simplification is acceptable if the IMU is close to the tip or if orientation is more important than absolute position.

**Camera Frame → Pen Tip (World Frame):**
```python
tip_pos_cam = t_cam_m + R_cam @ TIP_OFFSET_BODY
```
- `t_cam_m`: Pen center position in camera frame (meters)
- `R_cam`: Pen body orientation (rotation matrix)
- `TIP_OFFSET_BODY`: Offset from center to tip in body frame

This transformation rotates the tip offset into the camera frame and adds it to the center position.

### 5.3 Rotation Representations

The system uses multiple rotation representations:

- **Rotation Vector (Rodrigues):** Used by OpenCV's ArUco pose estimation. A 3D vector where the direction is the rotation axis and the magnitude is the rotation angle.
- **Rotation Matrix:** 3x3 orthogonal matrix. Used internally for transformations.
- **Quaternion:** 4D unit vector [w, x, y, z]. Used by the EKF for smooth interpolation and to avoid gimbal lock.

**Conversions:**
- Rodrigues ↔ Matrix: `cv2.Rodrigues()`
- Matrix → Quaternion: `transforms3d.quaternions.mat2quat()` (in `dodeca_bridge.py`)
- Quaternion → Matrix: `Quaternion.get_matrix()` (in `filter.py`)

---

## 6. Real-Time Dependencies

### 6.1 Coupling Between CV and IMU

**Tight Coupling:**
- The CV thread runs in the same process as the main application, allowing direct memory access to `object_pose`.
- The `dodeca_bridge` module reads `object_pose` synchronously in the data consumer thread.
- No inter-process communication (IPC) is needed for camera data.

**Timing:**
- IMU data arrives at ~100-200 Hz
- Camera data arrives at ~10-30 Hz
- The EKF accounts for the timing difference by maintaining a history buffer and applying smoothing.

**Latency Compensation:**
- The `camera_delay` parameter (default: a few samples) accounts for the time it takes to process a camera frame.
- The filter "rolls back" recent IMU updates, inserts the camera measurement at the correct time, and then "replays" the IMU updates.

### 6.2 Shutdown Coordination

The system uses a shared flag to coordinate shutdown:

```python
# In run.py
cv_shutdown_requested: bool = False

def _request_shutdown() -> None:
    global cv_shutdown_requested
    cv_shutdown_requested = True
```

**Trigger:** User presses 'q' in the CV window or closes the main window.

**Propagation:**
1. CV window sets `cv_shutdown_requested = True`
2. `QueueConsumer.run_queue_consumer()` checks `is_cv_shutdown_requested()` every loop iteration
3. If True, the data consumer stops and signals the main application to quit
4. The main application terminates the BLE process and exits

---

## 7. Key Assumptions

### 7.1 Geometry Assumptions

1. **IMU at Tip:** The system assumes `IMU_TO_TIP_BODY = [0, 0, 0]`, meaning the IMU is located at the pen tip. This is a simplification; in reality, the IMU is inside the pen body.

2. **Rigid Body:** The pen is assumed to be a rigid body with no flex or deformation.

3. **Known Tip Offset:** The offset from the dodecahedron center to the pen tip (`TIP_OFFSET_BODY`) is assumed to be known and constant. This was likely measured during calibration.

### 7.2 Sensor Assumptions

1. **IMU Calibration:** The accelerometer and gyroscope are assumed to be calibrated. The `calc_accel()` and `calc_gyro()` functions in `monitor_ble.py` apply scaling factors based on the sensor range settings.

2. **Camera Calibration:** The camera intrinsic parameters (focal length, principal point, distortion coefficients) are assumed to be known and stored in `color_cam_matrix.npy` and `color_cam_dist.npy`.

3. **Marker Geometry:** The positions of the ArUco markers on the dodecahedron are assumed to be known and stored in `dodecapen_assets/end_face_transforms_*.txt`.

### 7.3 Timing Assumptions

1. **Constant IMU Rate:** The IMU is assumed to send data at a roughly constant rate. The filter uses a fixed `dt` (time step) for prediction.

2. **Camera Delay:** The camera measurement is assumed to have a small, constant delay relative to the IMU. This is compensated by the `camera_delay` parameter.

3. **No Dropped Packets:** The system assumes that BLE packets are not dropped or that dropped packets are rare enough to not significantly affect performance.

### 7.4 Environment Assumptions

1. **Static Camera:** The camera is assumed to be stationary. All motion is relative to the camera frame.

2. **Good Lighting:** The camera is assumed to have sufficient lighting to reliably detect ArUco markers.

3. **Marker Visibility:** At least 2 ArUco markers must be visible for the system to estimate the pen's pose.

### 7.5 Noise Model Assumptions

1. **Gaussian Noise:** All sensor noise is assumed to be Gaussian (normally distributed).

2. **Constant Noise:** The noise characteristics are assumed to be constant over time. The noise covariance matrices (`Q`, `imu_noise`, `camera_noise`) are fixed.

3. **Independent Noise:** Noise in different sensors and different axes is assumed to be independent.

---

## Summary

The Dodeca Pen system is a sophisticated real-time tracking application that fuses high-frequency IMU data with lower-frequency camera data using an Extended Kalman Filter. The execution flow is:

1. **Initialization:** Set up processes, threads, and queues
2. **Data Ingestion:** Receive IMU data via BLE and camera data via OpenCV
3. **Data Fusion:** Use EKF to combine both sources, with RTS smoothing for past states
4. **Visualization:** Display the pen's 3D position and drawn trail in real-time

The system makes several assumptions about geometry, sensor calibration, and timing, but these are reasonable for a research prototype. The modifications to support offline video input are minimal and non-destructive, preserving all original functionality while adding flexibility for reproducible analysis.
