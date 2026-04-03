# Dodeca Pen: EKF Strategy Comparison and Accuracy Analysis

## Overview
This report summarizes the implementation and testing of the **Decoupled Extended Kalman Filter (EKF)** strategy for the Dodeca Pen project. The goal was to mitigate vision-based jitter that amplifies pen-tip position errors by decoupling orientation and position updates.

## Workflow Comparison Results

| Workflow | Strategy | Characteristics |
| :--- | :--- | :--- |
| **CV Only (Raw)** | Baseline | Uses raw computer vision data. High jitter, especially in the Z-axis. |
| **Standard EKF** | 7D Coupled | Fuses IMU and CV (Position + Orientation). High orientation jitter from CV propagates into position errors. |
| **Decoupled EKF** | 3D Position-Only | IMU handles orientation; CV updates only position. Significantly reduces orientation-induced position jitter. |

### Visualization Findings
- **3D Trajectory**: The Decoupled EKF provides a smoother path compared to the raw CV data, though it still follows the general trend of the recorded hand-drawn rectangle.
- **Z-Axis Stability**: The decoupled approach shows a marked reduction in high-frequency noise in the Z-axis, which is critical for accurate pen-lift detection.
- **XY Plane Accuracy**: While smoother, the trajectory still shows a "rough" rectangle. This indicates that while jitter is reduced, systematic errors remain.

## Remaining Accuracy Challenges

Despite the improvements from the decoupled EKF, several challenges persist:

1.  **IMU-to-Tip Offset**: The current `IMU_TO_TIP_BODY` offset is set to `[0, 0, 0]`. In reality, the IMU is located some distance from the pen tip. This misalignment causes "lever arm" effects where rotation is misinterpreted as translation.
2.  **Sensor Bias Drift**: IMU biases (accelerometer and gyroscope) drift over time. While the EKF estimates these, a short recording may not provide enough excitation for accurate convergence.
3.  **CV Latency**: There is a known delay in the computer vision pipeline. While the EKF uses a rollback/replay mechanism, inaccuracies in the delay estimation can cause temporal misalignment between IMU and CV data.
4.  **Static Calibration**: The system would benefit from a more rigorous initial calibration of the camera-to-world transform and the IMU's resting orientation.

## Proposed Solutions

- **Calibrate Lever Arm**: Measure the physical distance from the IMU center to the pen tip and update the transformation matrices in the filter.
- **Adaptive Noise Tuning**: Implement an adaptive EKF that adjusts measurement noise (`R`) based on the confidence of the CV marker detection (e.g., marker size or reprojection error).
- **Dynamic Bias Compensation**: Use a pre-recording "still period" to better estimate initial IMU biases.
