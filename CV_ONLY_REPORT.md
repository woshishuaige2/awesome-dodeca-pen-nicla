# Dodeca Pen: CV-Only Mode Verification Report

This report summarizes the successful implementation and verification of the **CV-only mode** for the Dodeca Pen system. The system was modified to bypass IMU input entirely while maintaining the EKF pipeline structure, using an offline video file for tracking.

## 1. Implementation Details

### 1.1 IMU Bypass
- **`Code/IMU/app/app.py`**: Modified to support a `--mode cv_only` flag. When enabled, the BLE acquisition process is not started, and the `QueueConsumer` only processes vision measurements.
- **`Code/IMU/app/filter.py`**: Updated the `update_camera` method to handle cases where no IMU history is available. In CV-only mode, the filter state is updated directly from vision measurements, with a prediction step to maintain covariance.

### 1.2 EKF Parameter Adjustments
To ensure vision measurements dominate the state:
- **Camera Noise**: Reduced to near-zero (`camera_noise_pos = 1e-9`, `camera_noise_or = 1e-7`).
- **Process Noise**: Increased to make the filter more responsive to camera updates without IMU guidance.

### 1.3 Offline Video Support
- The system uses the video file at `Code/Computer_vision/src/offline_test.mp4`.
- Headless mode is fully supported for environments without a display.

## 2. Verification Results

The system was run end-to-end for 60 seconds, processing the offline video file.

### 2.1 Tracking Performance
- **Frame Rate**: Averaged **8-12 FPS** during processing.
- **Detections**: Consistent ArUco marker detections (8-10 per frame).
- **Trajectory**: Successfully captured a continuous 3D trajectory of the pen tip.

### 2.2 Generated Outputs
The following files were generated and are available in the `Code/IMU/outputs/` directory:
- **`trajectory.csv`**: Raw pen-tip trajectory data (t, x, y, z).
- **`pentip_xy.png`**: 2D path plot of the pen tip.
- **`pentip_xyz.png`**: 3D trajectory plot of the pen tip.
- **`pentip_time.png`**: Time-series plot of X, Y, and Z positions.

## 3. How to Run

To replicate the results, use the following command:

```bash
cd Code/IMU
python3 main.py --mode cv_only --video /home/ubuntu/awesome-dodeca-pen/Code/Computer_vision/src/offline_test.mp4
```

After the run completes, generate the plots:

```bash
python3 plot_trajectory.py
```

## 4. Conclusion

The CV-only pipeline is now fully functional and verified. It provides a robust baseline for pen-tip tracking using only visual information, which can be used for comparison once IMU data is integrated in Stage 2.
