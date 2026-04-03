# Complete Workflow Guide: Offline Recording and Processing

This guide explains the new workflow for recording and processing pen tracking data with improved CV quality.

## Problem Summary

The previous `record_raw_data_filtered.py` performed real-time CV processing during recording, which resulted in noisy trajectories. Analysis showed that while frame drops were minimal (0.1%), the real-time CV detection quality was inferior to offline processing.

## Solution

Separate recording from processing:
1. **Record** IMU data + raw video (no CV processing)
2. **Process** video offline using proven CV pipeline
3. **Merge** IMU and CV data with proper synchronization
4. **Analyze** using existing `compare_workflows.py`

## New Workflow

### Step 1: Record IMU Data and Video

Use the new `record_imu_and_video.py` script:

```bash
cd Code/IMU
python record_imu_and_video.py --output outputs/recording --camera 0
```

**What it does:**
- Records IMU data from BLE device
- Records raw video from camera (no CV processing)
- Saves to separate files:
  - `outputs/recording/imu_data.json` - IMU readings
  - `outputs/recording/video.mp4` - Raw video
  - `outputs/recording/metadata.json` - Recording metadata

**When to stop:**
- Press `Ctrl+C` when finished recording
- Recommended: Record 10-15 seconds of pen movement

### Step 2: Process Video to Extract CV Data

Use the new `process_video_to_cv_data.py` script:

```bash
cd Code/IMU
python process_video_to_cv_data.py outputs/recording/video.mp4 --output outputs/recording/cv_data.json
```

**What it does:**
- Processes video frame-by-frame using the same CV pipeline as offline mode
- Applies One-Euro filtering for smooth trajectories
- Extracts dodecahedron center position and orientation
- Saves CV readings in my_data.json format

**Options:**
- `--no-filter` - Disable One-Euro filtering (not recommended)

### Step 3: Merge IMU and CV Data

Use the new `merge_imu_cv_data.py` script:

```bash
cd Code/IMU
python merge_imu_cv_data.py outputs/recording/imu_data.json outputs/recording/cv_data.json --output outputs/my_data.json
```

**What it does:**
- Synchronizes IMU and CV timestamps
- Aligns both streams to start at first CV detection
- Creates unified `my_data.json` in the format expected by `compare_workflows.py`

**Synchronization methods:**
- `--sync first_detection` (default) - Align at first CV marker detection
- `--sync manual --offset X.X` - Manual offset in seconds (CV - IMU)

### Step 4: Calibrate IMU Axis Alignment

Use one or more merged stationary recordings captured in clearly different pen orientations:

```bash
cd Code/IMU
python calibrate_imu_alignment.py outputs/calib_pose_1.json outputs/calib_pose_2.json outputs/calib_pose_3.json
```

**What it does:**
- Estimates a fixed IMU-to-body rotation from stationary gravity observations
- Saves the calibration to `calibration/imu_to_body.json`
- Prints a warning if you only provide one pose, because that only constrains gravity alignment and not the full 3D yaw relationship

**Recommended calibration data:**
- Record `6 to 10` stationary batches
- Hold each pose for `5 to 10 s`
- Make the orientations clearly different
- Keep the camera fixed during the calibration session and the validation run, because the calibration also estimates gravity in the camera frame
- Treat large residuals as a bad calibration; low residuals mean the stationary poses agree under one rigid IMU-to-body transform

### Step 5: Analyze Results

Use the existing `compare_workflows.py`:

```bash
cd Code/IMU
python compare_workflows.py outputs/my_data.json
```

**What it does:**
- Generates comparison plots for:
  - CV Only (Raw)
  - Standard EKF (Coupled)
  - Decoupled EKF (Proposed)
- Saves visualization to `outputs/comparison.png`
- Automatically loads `calibration/imu_to_body.json` if it exists

## Complete Example

```bash
# Navigate to IMU directory
cd /path/to/awesome-dodeca-pen/Code/IMU

# Step 1: Record (press Ctrl+C to stop)
python record_imu_and_video.py --output outputs/recording

# Step 2: Process video
python process_video_to_cv_data.py outputs/recording/video.mp4 --output outputs/recording/cv_data.json

# Step 3: Merge data
python merge_imu_cv_data.py outputs/recording/imu_data.json outputs/recording/cv_data.json --output outputs/my_data.json

# Step 4: Calibrate IMU alignment
python calibrate_imu_alignment.py outputs/calib_pose_1.json outputs/calib_pose_2.json outputs/calib_pose_3.json

# Step 5: Analyze
python compare_workflows.py outputs/my_data.json
```

## Advantages of New Workflow

1. **Better CV Quality**: Uses proven offline CV pipeline that produces clean trajectories
2. **No Real-time Overhead**: Recording doesn't compete with CV processing for resources
3. **Reproducible**: Can re-process same video with different parameters
4. **Debuggable**: Can inspect raw video and intermediate results
5. **Flexible**: Can apply different filters or CV algorithms offline

## Troubleshooting

### Low CV Detection Rate

If `process_video_to_cv_data.py` reports low detection rate (<50%):
- Check lighting conditions during recording
- Ensure markers are clearly visible
- Avoid rapid movements that cause motion blur
- Check camera focus and exposure settings

### Synchronization Issues

If trajectories don't align properly:
- Try manual synchronization: `--sync manual --offset X.X`
- Check metadata.json for timing information
- Verify both IMU and CV have data in overlapping time range

### Video Quality Issues

If video quality is poor:
- Increase camera resolution in `record_imu_and_video.py`
- Improve lighting conditions
- Use a higher quality camera
- Reduce camera auto-exposure/auto-focus

## File Structure

```
outputs/
├── recording/
│   ├── imu_data.json      # Raw IMU readings
│   ├── video.mp4          # Raw video recording
│   ├── cv_data.json       # Processed CV readings
│   └── metadata.json      # Recording metadata
├── my_data.json           # Merged IMU + CV data
└── comparison.png         # Analysis visualization
```

## Migration from Old Workflow

**Old workflow:**
```bash
python record_raw_data_filtered.py --output outputs/my_data.json
python compare_workflows.py outputs/my_data.json
```

**New workflow:**
```bash
python record_imu_and_video.py --output outputs/recording
python process_video_to_cv_data.py outputs/recording/video.mp4 --output outputs/recording/cv_data.json
python merge_imu_cv_data.py outputs/recording/imu_data.json outputs/recording/cv_data.json --output outputs/my_data.json
python compare_workflows.py outputs/my_data.json
```

The new workflow has more steps but produces significantly better results.

## Technical Details

### Timestamp Alignment

The merge script uses `first_detection` method by default:
1. Finds first IMU reading timestamp: `t_imu_0`
2. Finds first CV detection timestamp: `t_cv_0`
3. Uses `max(t_imu_0, t_cv_0)` as sync point
4. Aligns all timestamps relative to sync point

This ensures both streams have valid data at t=0.

### One-Euro Filter Parameters

The CV processing uses One-Euro filter with default parameters:
- `min_cutoff`: 1.0 Hz
- `beta`: 0.007
- `d_cutoff`: 1.0 Hz

These can be adjusted in `process_video_to_cv_data.py` if needed.

### Data Format Compatibility

The merged `my_data.json` maintains full compatibility with existing tools:
- `compare_workflows.py` - No changes needed
- Field names match original format
- Metadata includes all required parameters
