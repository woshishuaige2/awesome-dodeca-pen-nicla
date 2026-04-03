# Dodeca Pen Project: System Analysis

This document provides a detailed analysis of the Dodeca Pen project, focusing on its system architecture, data flow, and key components. The analysis is based on the code in the `woshishuaige2/awesome-dodeca-pen` repository.

## 1. High-Level System Overview

The Dodeca Pen system is designed to track the 3D position and orientation of a custom-built pen-like device. It achieves this by fusing data from two primary sources: an Inertial Measurement Unit (IMU) embedded in the pen and an external camera that tracks ArUco markers on the pen's body. The system is implemented in Python and uses a multi-process and multi-threaded architecture to handle real-time data acquisition and processing.

### Key Components:

*   **IMU Data Acquisition:** A dedicated process (`monitor_ble.py`) connects to the pen via Bluetooth Low Energy (BLE) to receive real-time IMU data (accelerometer, gyroscope, and pressure).
*   **Computer Vision (CV) Tracking:** A separate thread (`run.py`) uses a standard webcam to detect ArUco markers on the pen. It estimates the pen's 6-DoF pose (position and orientation) relative to the camera.
*   **Data Fusion:** An Extended Kalman Filter (EKF) implemented in `filter.py` fuses the high-frequency IMU data with the lower-frequency but more stable CV pose estimates. This corrects for IMU drift and provides a smooth and accurate track of the pen's tip.
*   **3D Visualization:** The system uses the VisPy library to provide a real-time 3D visualization of the pen's movement and the path it draws.

## 2. Data Flow and Execution Trace

The execution of the Dodeca Pen application starts with `Code/IMU/main.py`, which immediately calls the `main()` function in `Code/IMU/app/app.py`. The following is a step-by-step trace of the execution flow:

1.  **Initialization:** The `main()` function in `app.py` initializes the PyQt6 application, creates a VisPy canvas for visualization (`CanvasWrapper`), and sets up several multiprocessing queues for inter-process communication (`tracker_queue`, `ble_queue`, `ble_command_queue`).

2.  **Process/Thread Creation:** The `main()` function then spawns the following processes and threads:
    *   **BLE Process:** A `multiprocessing.Process` is started to run the `monitor_ble()` function. This process connects to the "DPOINT" BLE device, receives IMU data packets, unpacks them into `StylusReading` objects, and puts them into the `ble_queue`.
    *   **CV Thread:** A `threading.Thread` is started to run the `main()` function from `Code/Computer_vision/run.py`. This thread continuously captures frames from the camera, detects ArUco markers, and estimates the pen's pose. The pose is stored in a global variable `object_pose` within the `run.py` module.
    *   **Data Consumer Thread:** A `QThread` is created to run the `QueueConsumer` object. This is the heart of the data fusion logic.

3.  **Data Fusion in `QueueConsumer`:** The `run_queue_consumer()` method of the `QueueConsumer` runs in a loop and performs the following actions:
    *   **IMU Data Processing:** It retrieves `StylusReading` objects from the `ble_queue` and uses them to update the EKF via the `_filter.update_imu()` method.
    *   **CV Data Processing:** It calls `dodeca_bridge.make_ekf_measurements()`, which reads the `object_pose` from the `run.py` module. The pose data is then used to update the EKF via the `_filter.update_camera()` method. This step also triggers a smoothing algorithm to correct past state estimates.
    *   **Data Emission:** The `QueueConsumer` emits a `new_data` signal containing either `StylusUpdateData` (from the IMU) or `CameraUpdateData` (from the CV system). This signal is connected to the `CanvasWrapper` to update the 3D visualization.

4.  **Visualization:** The `CanvasWrapper` receives the `new_data` signal and updates the 3D scene, including the position and orientation of the pen model and the drawn trail.

## 3. Modules Dependent on Live Camera Input

The following modules and code sections are directly involved in handling the live camera feed:

*   **`Code/Computer_vision/run.py`:** This is the primary module responsible for camera interaction. The `start()` function initializes the camera capture using `cv2.VideoCapture(cam_index)`. The main loop of this function reads frames from the camera, performs marker detection, and displays the live video feed in a window if not running in headless mode.
*   **`Code/IMU/app/app.py`:** The `main()` function in this file is responsible for starting the CV thread: `cv_thread = threading.Thread(target=cv_run.main, kwargs={"headless": False, "cam_index": 0}, daemon=True)`. This line directly initiates the live camera capture.

## 4. Modification for Offline Analysis

To facilitate offline analysis and collaboration, the live camera feed can be replaced with a pre-recorded video file. This requires the following modifications:

1.  **Identify and Comment Out Live Camera Logic:** The primary code to modify is in `Code/Computer_vision/run.py`. The line `cap = cv2.VideoCapture(cam_index)` should be commented out and replaced with `cap = cv2.VideoCapture("offline_test.mp4")`.

2.  **Specify Placeholder Video:** The placeholder video file should be named `offline_test.mp4` and placed in the `Code/Computer_vision/` directory. The video should be a standard MP4 file containing a recording of the Dodeca Pen with its ArUco markers clearly visible.

By making these changes, the system will process the video file frame by frame, allowing for reproducible analysis of the tracking and fusion algorithms without the need for a live camera setup.
