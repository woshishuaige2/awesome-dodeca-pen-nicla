#!/bin/bash

# run_offline_pipeline.sh - Complete offline processing workflow
# This script processes a raw video to generate CV data, then merges it with IMU data.

# 1. Configuration
VIDEO_FILE=${1:-"outputs/video.mp4"}
IMU_FILE=${2:-"outputs/imu_data.json"}
CV_FILE="outputs/cv_data.json"
MERGED_FILE="outputs/my_data.json"

echo "============================================================"
echo "DODECA-PEN OFFLINE PIPELINE"
echo "============================================================"

# 2. Process Video
echo "[Step 1/3] Processing video to extract CV data..."
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file $VIDEO_FILE not found."
    exit 1
fi
python3.11 process_video_to_cv_data.py "$VIDEO_FILE" --output "$CV_FILE"

# 3. Merge IMU and CV Data
echo "[Step 2/3] Merging IMU and CV data..."
if [ ! -f "$IMU_FILE" ]; then
    echo "Error: IMU data file $IMU_FILE not found."
    exit 1
fi
python3.11 merge_imu_cv_data.py "$IMU_FILE" "$CV_FILE" --output "$MERGED_FILE" --sync first_detection

# 4. Run Workflows Comparison
echo "[Step 3/3] Generating workflow comparison plots..."
python3.11 compare_workflows.py "$MERGED_FILE"

echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Results saved to: outputs/workflow_comparison.png"
echo "Merged data: $MERGED_FILE"
echo "============================================================"
