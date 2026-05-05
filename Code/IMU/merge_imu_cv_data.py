"""
Merge IMU and CV Data with Synchronization
Combines separately recorded IMU and CV data into a single my_data.json format.
Handles timestamp alignment and synchronization.
"""

import json
import sys
from pathlib import Path
import numpy as np
import argparse


def timestamp_stats(readings, timestamp_key="timestamp"):
    """Return basic timing stats for a stream."""
    timestamps = [
        float(reading[timestamp_key])
        for reading in readings
        if timestamp_key in reading and reading[timestamp_key] is not None
    ]
    if len(timestamps) < 2:
        return None

    deltas = np.diff(np.asarray(timestamps, dtype=float))
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None

    return {
        "duration": float(duration),
        "mean_rate": float((len(timestamps) - 1) / duration),
        "median_rate": float(1.0 / np.median(deltas)) if np.median(deltas) > 0 else 0.0,
        "p95_gap": float(np.percentile(deltas, 95)),
        "max_gap": float(np.max(deltas)),
        "gaps_over_100ms": int(np.sum(deltas > 0.100)),
        "gaps_over_200ms": int(np.sum(deltas > 0.200)),
    }


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON file"""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def find_sync_point(imu_data, cv_data, method="first_detection"):
    """
    Find synchronization point between IMU and CV data.
    
    Methods:
    - "master_clock": Use IMU sensor clock synchronization (preferred)
    - "first_detection": Align at first CV detection (default)
    - "manual": Use manually specified offset
    - "cross_correlation": Use motion correlation (future enhancement)
    
    Returns:
        (imu_offset, cv_offset) - timestamps to use as t=0 for each stream
    """
    imu_readings = imu_data.get("imu_readings", [])
    cv_readings = cv_data.get("cv_readings", [])
    
    if len(imu_readings) == 0:
        print("[Sync] Warning: No IMU readings found")
        imu_start = 0
    else:
        imu_start = imu_readings[0]["local_timestamp"]
    
    if len(cv_readings) == 0:
        print("[Sync] Warning: No CV readings found")
        cv_start = 0
    else:
        cv_start = cv_readings[0]["local_timestamp"]
    
    if method == "master_clock":
        print(f"[Sync] Using master_clock (IMU Sensor) method")
        # In master_clock mode, timestamps are already in the same domain (t_sensor)
        # We align t=0 to the first CV detection for consistency with plotting
        return cv_start, cv_start

    elif method == "first_detection":
        # Align both streams to start at the first CV detection
        # CRITICAL: Use the SAME sync point for both IMU and CV to ensure exact alignment
        print(f"[Sync] Using first_detection method")
        print(f"  IMU start: {imu_start:.3f}")
        print(f"  CV start: {cv_start:.3f}")
        
        # If the timestamps are very far apart (e.g. > 1000s), they likely use different clocks
        if abs(cv_start - imu_start) > 100:
            print(f"  [Warning] Large time difference ({abs(cv_start - imu_start):.1f}s).")
            print(f"  [Sync] Treating first readings as simultaneous.")
            return imu_start, cv_start
        
        # Otherwise, assume they share the same clock
        sync_point = cv_start
        print(f"  [Sync] Using shared clock. Sync point: {sync_point:.3f}")
        return sync_point, sync_point
    
    else:
        raise ValueError(f"Unknown sync method: {method}")


def should_use_master_clock(imu_data, cv_data, tolerance_seconds=10.0):
    """
    Validate that CV and IMU timestamps are genuinely in the same sensor clock domain.
    """
    if cv_data.get("metadata", {}).get("master_clock") != "IMU_SENSOR":
        return False

    imu_readings = imu_data.get("imu_readings", [])
    cv_readings = cv_data.get("cv_readings", [])
    if not imu_readings or not cv_readings:
        return False

    if "t" not in imu_readings[0]:
        return False

    imu_sensor_start = imu_readings[0]["t"] / 1000.0
    cv_start = cv_readings[0]["local_timestamp"]
    delta = abs(imu_sensor_start - cv_start)
    if delta > tolerance_seconds:
        print(
            f"[Sync] Master clock metadata present, but timestamps disagree by "
            f"{delta:.3f}s. Falling back to standard synchronization."
        )
        return False

    return True


def align_timestamps(readings, sync_offset, allow_negative=False, use_sensor_t=False):
    """
    Align timestamps to start from sync point.
    CRITICAL: Both 'timestamp' and 'local_timestamp' must be aligned for EKF to work correctly.
    
    Args:
        readings: List of readings with timestamp fields
        sync_offset: Timestamp to use as t=0
        allow_negative: If True, include readings before sync point (with negative timestamps)
        use_sensor_t: If True, use the sensor 't' field (converted to seconds) instead of local_timestamp
    
    Returns:
        List of readings with aligned timestamps
    """
    aligned = []
    for reading in readings:
        # Determine the source timestamp
        if use_sensor_t and "t" in reading:
            t_src = reading["t"] / 1000.0  # Convert ms to seconds
        else:
            t_src = reading["local_timestamp"]
            
        # Include readings at or after sync point, or before if allow_negative=True
        if allow_negative or t_src >= sync_offset:
            aligned_reading = reading.copy()
            # CRITICAL: Update both timestamp fields to maintain consistency
            aligned_reading["timestamp"] = t_src - sync_offset
            aligned_reading["local_timestamp"] = t_src - sync_offset
            aligned.append(aligned_reading)
    
    return aligned


def merge_data(imu_file, cv_file, output_file, sync_method="first_detection", manual_offset=0.0):
    """
    Merge IMU and CV data into unified format.
    
    Args:
        imu_file: Path to IMU data JSON
        cv_file: Path to CV data JSON
        output_file: Path to output merged JSON
        sync_method: Synchronization method
        manual_offset: Manual time offset (CV - IMU) in seconds
    """
    print("=" * 60)
    print("MERGING IMU AND CV DATA")
    print("=" * 60)
    
    # Load data
    print(f"[Merge] Loading IMU data from: {imu_file}")
    imu_data = load_json(imu_file)
    
    print(f"[Merge] Loading CV data from: {cv_file}")
    cv_data = load_json(cv_file)
    
    # Detect if master clock was used
    is_master_clock = should_use_master_clock(imu_data, cv_data)
    if is_master_clock:
        print("[Sync] Master Clock (IMU_SENSOR) detected in CV data")
        sync_method = "master_clock"

    # Find synchronization point
    if sync_method == "manual":
        print(f"[Sync] Using manual offset: {manual_offset:.3f} seconds")
        imu_readings = imu_data.get("imu_readings", [])
        cv_readings = cv_data.get("cv_readings", [])
        
        if len(imu_readings) > 0 and len(cv_readings) > 0:
            imu_offset = imu_readings[0]["local_timestamp"]
            cv_offset = cv_readings[0]["local_timestamp"] + manual_offset
        else:
            imu_offset = 0
            cv_offset = manual_offset
    else:
        imu_offset, cv_offset = find_sync_point(imu_data, cv_data, sync_method)
    
    # Align timestamps
    # CRITICAL: Allow IMU readings before CV start (negative timestamps) for proper EKF initialization
    print(f"[Merge] Aligning IMU timestamps (offset: {imu_offset:.3f}, allow_negative=True, use_sensor_t={is_master_clock})")
    aligned_imu = align_timestamps(
        imu_data.get("imu_readings", []), 
        imu_offset, 
        allow_negative=True, 
        use_sensor_t=is_master_clock
    )
    
    print(f"[Merge] Aligning CV timestamps (offset: {cv_offset:.3f})")
    aligned_cv = align_timestamps(
        cv_data.get("cv_readings", []), 
        cv_offset, 
        allow_negative=False,
        use_sensor_t=False # CV is already in sensor domain if master_clock is used
    )
    
    # Create merged data structure (matching my_data.json format)
    merged_data = {
        "metadata": {
            "start_time": 0.0,
            "end_time": max(
                aligned_imu[-1]["local_timestamp"] if aligned_imu else 0.0,
                aligned_cv[-1]["local_timestamp"] if aligned_cv else 0.0
            ),
            "tip_offset_body": cv_data["metadata"].get("tip_offset_body", [0, 0, 0]),
            "imu_to_tip_body": cv_data["metadata"].get("imu_to_tip_body", [0, 0, 0]),
            "filtered": cv_data["metadata"].get("filtered", False),
            "filter_type": cv_data["metadata"].get("filter_type", "None"),
            "imu_count": len(aligned_imu),
            "cv_count": len(aligned_cv),
            "sync_method": sync_method,
            "imu_sync_offset": imu_offset,
            "cv_sync_offset": cv_offset,
        },
        "imu_readings": aligned_imu,
        "cv_readings": aligned_cv
    }
    
    # Save merged data
    print(f"[Merge] Saving merged data to: {output_file}")
    save_json(merged_data, output_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"IMU readings: {len(aligned_imu)}")
    print(f"CV readings: {len(aligned_cv)}")
    
    if aligned_imu and aligned_cv:
        imu_stats = timestamp_stats(aligned_imu)
        cv_stats = timestamp_stats(aligned_cv)
        if imu_stats:
            print(f"IMU duration: {imu_stats['duration']:.2f} seconds")
            print(f"IMU mean sample rate: {imu_stats['mean_rate']:.1f} Hz")
            print(f"IMU median sample rate: {imu_stats['median_rate']:.1f} Hz")
            print(
                f"IMU gaps: p95={imu_stats['p95_gap']*1000:.0f} ms, "
                f"max={imu_stats['max_gap']*1000:.0f} ms, "
                f">100ms={imu_stats['gaps_over_100ms']}, "
                f">200ms={imu_stats['gaps_over_200ms']}"
            )
        if cv_stats:
            print(f"CV duration: {cv_stats['duration']:.2f} seconds")
            print(f"CV mean sample rate: {cv_stats['mean_rate']:.1f} Hz")
            print(f"CV median sample rate: {cv_stats['median_rate']:.1f} Hz")
            print(
                f"CV gaps: p95={cv_stats['p95_gap']*1000:.0f} ms, "
                f"max={cv_stats['max_gap']*1000:.0f} ms, "
                f">100ms={cv_stats['gaps_over_100ms']}, "
                f">200ms={cv_stats['gaps_over_200ms']}"
            )
    
    print(f"\nOutput saved to: {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge IMU and CV data with synchronization")
    parser.add_argument("imu_file", help="Path to IMU data JSON file")
    parser.add_argument("cv_file", help="Path to CV data JSON file")
    parser.add_argument("--output", default="outputs/merged_data.json", help="Output merged JSON file (use my_data.json for compare_workflows.py)")
    parser.add_argument("--sync", default="first_detection", 
                       choices=["first_detection", "manual", "master_clock"],
                       help="Synchronization method")
    parser.add_argument("--offset", type=float, default=0.0,
                       help="Manual time offset in seconds (CV - IMU), only used with --sync manual")
    
    args = parser.parse_args()
    
    merge_data(
        args.imu_file,
        args.cv_file,
        args.output,
        sync_method=args.sync,
        manual_offset=args.offset
    )


if __name__ == "__main__":
    main()
