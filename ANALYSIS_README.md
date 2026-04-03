# Dodeca Pen Project Analysis: Documentation Package

**Date:** January 10, 2026  
**Prepared by:** Manus AI  
**Project:** awesome-dodeca-pen (woshishuaige2/awesome-dodeca-pen)

---

## Overview

This documentation package provides a comprehensive analysis of the Dodeca Pen project, including system architecture, execution flow, and modifications to enable offline video analysis. The analysis was conducted with thesis-level rigor, focusing on traceability and reproducibility.

## Documentation Files

### 1. **EXECUTION_TRACE.md**
A detailed step-by-step execution trace starting from `Code/IMU/main.py`. This document covers:
- Entry point and initialization sequence
- Process and thread architecture
- Data ingestion from IMU and camera
- Extended Kalman Filter data fusion pipeline
- Coordinate frames and transformations
- Real-time dependencies and coupling
- Key assumptions about geometry, sensors, and timing

**Use this document to:** Understand how the system works internally and trace the flow of data from sensors to visualization.

### 2. **OFFLINE_MODE_GUIDE.md**
A practical guide for enabling offline video analysis. This document covers:
- Summary of modifications made to the codebase
- Detailed explanation of changes in each file
- Step-by-step instructions for enabling offline mode
- Video format requirements and specifications
- Troubleshooting common issues

**Use this document to:** Set up and run the system with pre-recorded video files instead of a live camera.

### 3. **system_analysis.md**
A high-level overview of the system architecture. This document covers:
- Key components (IMU, CV, EKF, visualization)
- Data flow between components
- Modules dependent on live camera input
- Modification strategy for offline analysis

**Use this document to:** Get a quick understanding of the system architecture before diving into details.

### 4. **modifications.diff**
A Git diff file showing all code changes made to enable offline video support.

**Use this file to:** Review the exact changes made to the codebase or apply them to a different branch.

---

## Modified Files

The following files were modified to enable offline video support:

### 1. `Code/Computer_vision/run.py`
**Changes:**
- Added `video_file` parameter to `start()` and `main()` functions
- Added conditional logic to use `cv2.VideoCapture(video_file)` when `video_file` is specified
- Preserved all original live camera functionality

**Lines modified:** 132-158, 242-252

### 2. `Code/IMU/app/app.py`
**Changes:**
- Added `OFFLINE_VIDEO_FILE` configuration variable in `main()` function
- Passed this variable to the CV thread when starting it
- Added comments explaining how to enable offline mode

**Lines modified:** 466-478

---

## Phase 1: Code Understanding (Completed)

### What Data is Being Ingested

1. **IMU Data (via BLE):**
   - **Accelerometer:** 3-axis, ~100-200 Hz, in m/s²
   - **Gyroscope:** 3-axis, ~100-200 Hz, in rad/s
   - **Pressure:** Single value, ~100-200 Hz, normalized [0, 1]
   - **Source:** Arduino-based IMU in the pen, transmitted via Bluetooth Low Energy

2. **Camera Data (via OpenCV):**
   - **RGB Frames:** 640x480 or higher, ~10-30 fps
   - **ArUco Markers:** Detected in each frame
   - **Pose Estimate:** 6-DoF (position + orientation), ~10-30 Hz
   - **Source:** USB webcam or video file

3. **Timestamps:**
   - IMU timestamps are currently unused (set to 0)
   - Camera timestamps are generated using `time.time()` when pose is published
   - EKF uses a fixed time step `dt` for prediction

### How Data Flows Through the Pipeline

```
┌─────────────────┐         ┌─────────────────┐
│   Dodeca Pen    │         │   USB Camera    │
│   (Hardware)    │         │   (or Video)    │
└────────┬────────┘         └────────┬────────┘
         │ BLE                       │ USB/File
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  monitor_ble    │         │     run.py      │
│   (Process)     │         │    (Thread)     │
└────────┬────────┘         └────────┬────────┘
         │ ble_queue                 │ object_pose
         │                           │ (global var)
         ▼                           ▼
┌────────────────────────────────────────────┐
│          QueueConsumer (Thread)            │
│  ┌──────────────────────────────────────┐  │
│  │   Extended Kalman Filter (EKF)       │  │
│  │  - Prediction: IMU data (~100 Hz)    │  │
│  │  - Correction: Camera data (~20 Hz)  │  │
│  │  - Smoothing: RTS backward pass      │  │
│  └──────────────────────────────────────┘  │
└────────────────┬───────────────────────────┘
                 │ new_data signal
                 ▼
         ┌───────────────┐
         │ CanvasWrapper │
         │  (VisPy 3D)   │
         └───────────────┘
```

### Key Assumptions

1. **Geometry:**
   - IMU is assumed to be at the pen tip (simplification)
   - Pen is a rigid body with no deformation
   - Tip offset from dodecahedron center is known and constant

2. **Sensors:**
   - IMU and camera are calibrated
   - ArUco marker positions on the pen are known
   - At least 2 markers must be visible for pose estimation

3. **Timing:**
   - IMU sends data at a roughly constant rate
   - Camera has a small, constant delay (compensated by filter)
   - BLE packet loss is negligible

4. **Environment:**
   - Camera is stationary
   - Lighting is sufficient for marker detection
   - All motion is relative to the camera frame

### Coordinate Frames

1. **Camera Frame:** Origin at camera optical center, X right, Y down, Z forward (OpenCV convention)
2. **Pen Body Frame:** Origin at dodecahedron center, aligned with pen structure
3. **IMU Frame:** Origin at IMU sensor location, aligned with IMU chip axes
4. **Pen Tip Frame:** Origin at physical pen tip, same axes as body frame

### Real-Time Dependencies

- **Tight Coupling:** CV thread runs in the same process as main app, allowing direct memory access
- **No IPC Overhead:** Camera data is read directly from a global variable, not through a queue
- **Latency Compensation:** EKF maintains a history buffer and applies smoothing to account for camera delay
- **Shutdown Coordination:** Shared flag (`cv_shutdown_requested`) coordinates graceful shutdown

---

## Phase 2: Offline Analysis Modifications (Completed)

### Sections Enabling Live CV Window and Camera Capture

1. **`Code/Computer_vision/run.py::start()` (Line 140):**
   ```python
   cap = cv2.VideoCapture(cam_index)  # Opens live camera
   ```

2. **`Code/Computer_vision/run.py::start()` (Lines 160-164):**
   ```python
   if not headless:
       w, h = int(cap.get(3)), int(cap.get(4))
       cv2.namedWindow("RGB-D Live Stream", cv2.WINDOW_NORMAL)
       cv2.resizeWindow("RGB-D Live Stream", w, h)
       cv2.moveWindow("RGB-D Live Stream", 20, 20)
   ```

3. **`Code/Computer_vision/run.py::start()` (Lines 212-216):**
   ```python
   if not headless:
       cv2.imshow("RGB-D Live Stream", rgb)
       if cv2.waitKey(1) & 0xFF == ord("q"):
           _request_shutdown()
           break
   ```

4. **`Code/IMU/app/app.py::main()` (Lines 467-478):**
   ```python
   cv_thread = threading.Thread(
       target=cv_run.main, 
       kwargs={"headless": False, "cam_index": 0, "video_file": OFFLINE_VIDEO_FILE}, 
       daemon=True
   )
   cv_thread.start()
   ```

### Modifications Made

**No code was deleted.** All modifications are additive and preserve the original functionality:

1. **Added `video_file` parameter** to `run.py::start()` and `run.py::main()`
2. **Added conditional logic** to use video file when specified, otherwise use live camera
3. **Added `OFFLINE_VIDEO_FILE` variable** in `app.py::main()` for easy configuration

### How to Enable Offline Mode

1. **Prepare your video file:**
   - Format: MP4 (H.264 codec recommended)
   - Content: Dodeca Pen with ArUco markers clearly visible
   - Recommended filename: `offline_test.mp4`

2. **Place the video file:**
   - Recommended location: `Code/Computer_vision/offline_test.mp4`
   - Or use an absolute path

3. **Modify `Code/IMU/app/app.py` line 471:**
   ```python
   # Change from:
   OFFLINE_VIDEO_FILE = None
   
   # To:
   OFFLINE_VIDEO_FILE = "offline_test.mp4"
   ```

4. **Run the application:**
   ```bash
   cd Code/IMU
   python main.py
   ```

### Expected Placeholder Video

**Filename:** `offline_test.mp4`  
**Location:** `Code/Computer_vision/offline_test.mp4`

**Format Requirements:**
- **Container:** MP4, AVI, MOV, or MKV
- **Codec:** H.264 (for MP4) or other OpenCV-compatible codec
- **Resolution:** 640x480 or higher
- **Frame Rate:** 30 fps or higher
- **Content:** Dodeca Pen with ArUco markers clearly visible throughout

**Recording Tips:**
- Ensure good lighting (avoid shadows on markers)
- Keep markers in focus
- Move the pen slowly to avoid motion blur
- Ensure at least 2 markers are visible in most frames
- Use the same camera that was used for calibration (or recalibrate)

---

## Next Steps

1. **Record a test video** showing the Dodeca Pen with ArUco markers
2. **Name the video** `offline_test.mp4`
3. **Place it** in `Code/Computer_vision/`
4. **Update** line 471 in `Code/IMU/app/app.py` to enable offline mode
5. **Run** the application and verify the system processes the video correctly

---

## Additional Notes

### Optimization Opportunities (Not Implemented)

The following optimizations were deliberately **not** implemented to keep changes minimal:

1. **Frame rate control:** The system processes video frames as fast as possible. For more accurate timing, you could add frame rate control based on the video's metadata.

2. **Video looping control:** OpenCV automatically loops videos. You could add logic to stop after one pass.

3. **Progress indicator:** You could display the current frame number and total frames.

4. **Multiple video support:** You could add a command-line interface to specify the video file without editing code.

These can be added later if needed, but the current implementation focuses on minimal, non-destructive changes.

### Testing Recommendations

1. **Test with live camera first** to ensure the system works correctly
2. **Record a short test video** (10-30 seconds) with good marker visibility
3. **Verify offline mode** processes the video and tracks markers correctly
4. **Compare results** between live and offline modes to ensure consistency

### Troubleshooting

See `OFFLINE_MODE_GUIDE.md` for detailed troubleshooting steps.

---

## Contact and Collaboration

This analysis was prepared to facilitate thesis-level research collaboration. All documentation emphasizes:

- **Rigor:** Detailed technical analysis with precise terminology
- **Traceability:** Clear references to specific files and line numbers
- **Reproducibility:** Step-by-step instructions for replicating the setup

For questions or issues, refer to the specific documentation files or examine the code with the provided line number references.

---

**End of Analysis Package**
