# Dodeca Pen Recording and Analysis Workflow Guide

## Overview

This guide explains the complete workflow for recording pen sketching data and analyzing it through different EKF strategies. The workflow includes **integrated One-Euro filtering** during recording to produce clean, smooth CV trajectories.

---

## Important: Coordinate System

**What Computer Vision Detects:**
- CV tracks ArUco markers on the dodecahedron and outputs the **DODECAHEDRON CENTER** position and orientation
- The recorded data contains `center_pos_cam` which is this detected center position in the camera frame

**Key Offsets (defined in body frame):**
- `CENTER_TO_TIP_BODY` = [0, 137.5mm, -82.1mm] - offset from dodeca center to pen tip
- `IMU_OFFSET_BODY` = [0, 0, 0] - offset from dodeca center to IMU (currently simplified as zero)

**Current Simplification:**
- We assume the IMU is located at the dodecahedron center (IMU_OFFSET_BODY = [0,0,0])
- This means the filter tracks the dodeca center position, not the actual IMU or tip position
- For accurate tip tracking, the workflow applies the CENTER_TO_TIP_BODY offset

---

## File Structure and Purpose

| File | Purpose |
|------|---------|
| **record_raw_data_filtered.py** | **[RECOMMENDED]** Records both CV and IMU data with One-Euro filter applied to CV readings in real-time. Produces clean trajectories. |
| **record_raw_data.py** | Records raw, unfiltered CV and IMU data. Use only if you want to test without filtering. |
| **playback_raw_data.py** | Replays recorded data through the EKF and outputs a CSV trajectory file. |
| **plot_trajectory.py** | (Optional) Generates XY plot, 3D plot, and time series plots using "decoupled EKF". |
| **compare_workflows.py** | Compares three EKF strategies (CV only, Standard EKF, Decoupled EKF) and generates comparative visualizations. |

---

## Step-by-Step Workflow

### Step 1: Record Data with Filtering

Use the **new filtered recording script** to capture clean CV data:

```bash
cd Code/IMU
python3 record_raw_data_filtered.py --output outputs/my_data.json
```

**What this does:**
- Connects to your pen via BLE and captures IMU data (accel, gyro, pressure)
- Captures computer vision pose estimates from the camera
- **Applies One-Euro filtering** to the CV position and orientation in real-time
- Saves both streams to `outputs/my_data.json`

**During recording:**
- You'll see status messages showing marker detection and sample counts
- Draw your rectangle (or any shape) on the sketching plane
- Lift the pen to demonstrate the Z-axis motion
- Press `Ctrl+C` to stop and save

**Expected output:**
```
[Recorder] IMU recording started.
[Recorder] CV recording started with One-Euro filtering.
=== Recording Started ===
Press Ctrl+C to stop recording

[12:34:56] Marker: DETECTED | Pos: (0.123, 0.456, 0.789)
[Recorder] IMU: received 100 samples
[Recorder] CV: received 50 filtered samples
...
^C
[Main] Stopping...
[Recorder] Data saved to outputs/my_data.json
[Recorder] Recorded 509 IMU and 420 filtered CV samples.
```

---

### Step 2: Compare EKF Strategies

Run the comparison script to test all three workflows:

```bash
cd Code/IMU
python3 compare_workflows.py
```

**What this does:**
- Loads `outputs/my_data.json`
- Runs three different EKF strategies:
  1. **CV Only (Raw):** Uses only the filtered CV data, no IMU fusion
  2. **Standard EKF (Coupled):** Fuses CV position + orientation with IMU (7D updates)
  3. **Decoupled EKF (Proposed):** Fuses CV position only with IMU (3D updates, orientation from IMU only)
- Generates `outputs/workflow_comparison.png` with 4 subplots

**Expected output:**
```
Running CV Only workflow...
Running Standard EKF workflow...
Running Decoupled EKF workflow...
Workflow CV Only (Raw): 420 points
  CV Only (Raw) - X mean: 0.1234, Y mean: 0.5678, Z mean: 0.5079
Workflow Standard EKF (Coupled): 420 points
  Standard EKF (Coupled) - X mean: 0.1235, Y mean: 0.5679, Z mean: 0.5008
Workflow Decoupled EKF (Proposed): 420 points
  Decoupled EKF (Proposed) - X mean: 0.1233, Y mean: 0.5677, Z mean: 0.4955
Comparison plot saved to ./outputs/workflow_comparison.png
```

---

### Step 3: Interpret the Results

Open `outputs/workflow_comparison.png` to see four plots:

#### 1. **3D Pen-Tip Trajectory** (Top-Left)
Shows the full 3D motion of the pen tip. You should see:
- A rectangular path in the XY plane
- An upward motion in Z when you lifted the pen

#### 2. **XY Plane (Top View)** (Top-Right)
Shows the top-down view of your sketching. The rectangle should be clearly visible here.
- **CV Only** may show some jitter
- **Standard EKF** should be smoother but may have "wandering"
- **Decoupled EKF** should be the smoothest and most stable

#### 3. **Z-Axis (Absolute Height)** (Bottom-Left)
Shows the pen's height over time. You should see:
- A relatively flat region while drawing the rectangle
- A clear upward trend when you lifted the pen
- **Decoupled EKF** should show the cleanest lift-off

#### 4. **Z-Axis Jitter (High-pass filtered)** (Bottom-Right)
Shows only the high-frequency noise in the Z-axis.
- **CV Only** will be the most noisy
- **Standard EKF** will be smoother but still show some vibration
- **Decoupled EKF** should be the flattest (least jitter)

---

## Key Differences Between Workflows

| Workflow | CV Position | CV Orientation | IMU Accel/Gyro | Result |
|----------|-------------|----------------|----------------|--------|
| **CV Only** | ✅ Used | ✅ Used | ❌ Ignored | High jitter, no sensor fusion |
| **Standard EKF** | ✅ Fused (7D) | ✅ Fused (7D) | ✅ Fused | Smooth but orientation jitter propagates to position |
| **Decoupled EKF** | ✅ Fused (3D) | ❌ Ignored | ✅ Fused (orientation only) | Best stability, orientation from IMU only |

---

## Troubleshooting

### Problem: "Marker: NOT DETECTED" during recording
**Solution:**
- Ensure good lighting on the dodecahedron markers
- Keep the pen within the camera's field of view
- Check that the camera is connected and working

### Problem: Very few CV readings recorded
**Solution:**
- Make sure the dodecahedron is clearly visible throughout the recording
- Move the pen more slowly to ensure consistent detection
- Check camera focus and exposure settings

### Problem: Trajectory doesn't look like a rectangle
**Solution:**
- The One-Euro filter should now produce clean trajectories
- If still noisy, check that you're using `record_raw_data_filtered.py` (not the old `record_raw_data.py`)
- Ensure you drew a clear rectangle during recording

### Problem: "Array must not contain infs or NaNs" error
**Solution:**
- This was caused by the old post-processing filter
- The new integrated filter in `record_raw_data_filtered.py` should not have this issue
- If you still see this, check that the CV readings in your JSON file have valid numbers

---

## Next Steps

Once you have clean trajectories from the **Decoupled EKF**:

1. **Tune Parameters:** Adjust `camera_noise_pos` in `app/filter.py` to optimize the balance between responsiveness and smoothness
2. **Test Different Motions:** Record different sketching patterns (circles, lines, 3D spirals) to validate the EKF across various scenarios
3. **Deploy to Live System:** Once satisfied with offline results, integrate the decoupled EKF into your live application

---

## Summary of Changes

### What Was Fixed:
- **One-Euro Filter Integration:** The filter is now applied during recording, not as a post-processing step
- **Timestamp Handling:** Fixed the NaN issue caused by improper filter initialization
- **Simplified Workflow:** No need for separate post-processing scripts

### What You Should Use:
- **For Recording:** `python3 record_raw_data_filtered.py`
- **For Analysis:** `python3 compare_workflows.py`

This workflow ensures repeatable, clean results for your EKF development and testing.
