
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyquaternion import Quaternion

# Add project directories to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "IMU"))

from app.filter import DpointFilter, initial_state
from app.imu_alignment import (
    DEFAULT_ALIGNMENT_PATH,
    estimate_alignment_from_streams,
    load_alignment_calibration,
)
from app.monitor_ble import StylusReading
from app.dodeca_bridge import CENTER_TO_TIP_BODY
import app.filter_core as fc

DEFAULT_DT = 1.0 / 60.0


def _rotation_matrix_to_quaternion(rotation_matrix):
    return Quaternion(matrix=rotation_matrix).elements


def _compute_camera_world_rotation(imu_quat, imu_to_body, camera_rotation):
    """
    Bridge the Nicla fused orientation frame into the camera frame.

    Assumptions:
    - imu_quat encodes IMU-frame -> fused-world rotation.
    - imu_to_body maps IMU frame -> stylus body frame.
    - camera_rotation maps body frame -> camera frame.

    This bridge is only as accurate as the current imu_to_body calibration and
    should be revisited after the CAD/mechanical update for the Nicla mount.
    """
    world_from_imu = Quaternion(imu_quat).rotation_matrix
    world_from_body = world_from_imu @ imu_to_body.T
    return camera_rotation @ world_from_body.T


def _align_imu_quaternion_to_camera_frame(imu_quat, imu_to_body, camera_world_rotation):
    world_from_imu = Quaternion(imu_quat).rotation_matrix
    world_from_body = world_from_imu @ imu_to_body.T
    camera_from_body = camera_world_rotation @ world_from_body
    return _rotation_matrix_to_quaternion(camera_from_body)


def _estimate_nominal_dt(imu_readings):
    """
    Estimate a stable IMU timestep from the positive timestamp deltas.

    Offline recordings can contain duplicate timestamps when multiple BLE packets
    are received inside the same clock tick. Using the first delta directly can
    therefore produce dt=0 and effectively break the EKF dynamics.
    """
    if len(imu_readings) < 2:
        return DEFAULT_DT

    timestamps = np.array([r["local_timestamp"] for r in imu_readings], dtype=float)
    deltas = np.diff(timestamps)
    positive_deltas = deltas[deltas > 1e-6]
    if positive_deltas.size == 0:
        return DEFAULT_DT

    return float(np.median(positive_deltas))


def _update_filter_dt(filter_obj, delta_t, nominal_dt):
    """
    Keep the filter timestep sane for irregular offline data.
    """
    if delta_t is None:
        filter_obj.dt = nominal_dt
        return

    if delta_t <= 1e-6:
        return

    # Clamp large gaps so the EKF does not take a single oversized integration
    # step after periods without IMU updates.
    filter_obj.dt = min(float(delta_t), 5.0 * nominal_dt)


def _resolve_imu_alignment(imu_readings, cv_readings, calibration_path=None):
    explicit_path = Path(calibration_path) if calibration_path else None
    candidate_paths = []
    if explicit_path is not None:
        candidate_paths.append(explicit_path)
    candidate_paths.append(DEFAULT_ALIGNMENT_PATH)

    seen_paths = set()
    for path in candidate_paths:
        if str(path) in seen_paths:
            continue
        seen_paths.add(str(path))
        calibration = load_alignment_calibration(path)
        if calibration is not None:
            print(
                f"[IMU Align] Loaded calibration from {calibration['path']} "
                f"(method={calibration['method']}, observable_axes={calibration['observable_axes']})"
            )
            return calibration["rotation_matrix"], calibration["gravity_camera"]

    calibration = estimate_alignment_from_streams(imu_readings, cv_readings)
    print(
        f"[IMU Align] No saved calibration found. "
        f"Using fallback {calibration['method']} estimate "
        f"(mean_residual={calibration['mean_residual']:.4f}, max_residual={calibration['max_residual']:.4f})"
    )
    return calibration["rotation_matrix"], calibration["gravity_camera"]


def run_workflow(input_file, mode="decoupled", calibration_path=None):
    """
    Modes: 
    - 'cv_only': Only use CV updates, no IMU.
    - 'standard': Use 7D CV updates (Pos + Quat).
    - 'decoupled': Use 3D CV updates (Pos only).
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    
    imu_readings = data.get("imu_readings", [])
    cv_readings = data.get("cv_readings", [])
    
    # Backward compatibility: rename old field names to new ones
    for cv_reading in cv_readings:
        if "imu_pos_cam" in cv_reading and "center_pos_cam" not in cv_reading:
            cv_reading["center_pos_cam"] = cv_reading["imu_pos_cam"]
    
    all_events = []
    
    if mode != "cv_only":
        for r in imu_readings:
            all_events.append(("IMU", r["local_timestamp"], r))
    
    for r in cv_readings:
        all_events.append(("CV", r["local_timestamp"], r))
    
    all_events.sort(key=lambda x: x[1])

    # Configuration for different modes
    original_camera_measurement = fc.camera_measurement

    # Standard mode logic helper
    def standard_fuse_camera(fs, imu_pos, orientation_quat):
        h = np.concatenate((fs.state[fc.i_pos], fs.state[fc.i_quat]))
        H = np.zeros((7, fc.STATE_SIZE))
        H[0:3, fc.i_pos] = np.eye(3)
        H[3:7, fc.i_quat] = np.eye(4)
        z = np.concatenate((imu_pos.flatten(), orientation_quat))
        
        # FIX: Standard EKF was over-damped. 
        # Increasing R allows the filter to be more flexible and follow CV motion.
        R = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3])
        
        state, statecov = fc.ekf_correct(fs.state, fs.statecov, h, H, z, R)
        state[fc.i_quat] = fc.repair_quaternion(state[fc.i_quat])
        return fc.FilterState(state, statecov)

    # Initialize Filter
    dt = _estimate_nominal_dt(imu_readings)
    
    # FIX: Tuning noise parameters for both EKF modes.
    # We need high process noise for position and velocity to follow CV accurately,
    # but low noise for acceleration to prevent random drift when static.
    import app.filter as filter_mod
    q_diag = np.zeros(fc.STATE_SIZE)
    q_diag[fc.i_quat] = 1e-7   # Keep orientation stable
    q_diag[fc.i_av] = 1e-4     # Low angular velocity noise
    q_diag[fc.i_pos] = 1.0      # High trust in CV position changes
    q_diag[fc.i_vel] = 1.0      # High trust in CV velocity changes
    q_diag[fc.i_acc] = 0.01     # LOW acceleration noise to fight gravity leakage
    q_diag[fc.i_accbias] = 1e-6
    q_diag[fc.i_gyrobias] = 1e-7
    q_diag[fc.i_magbias] = 1e-4
    filter_mod.Q = np.diag(q_diag)
    
    imu_alignment, gravity_camera = _resolve_imu_alignment(imu_readings, cv_readings, calibration_path)
    filter = DpointFilter(dt=dt, smoothing_length=15, camera_delay=5, gravity_vector=gravity_camera)
    trajectory = []
    previous_imu_ts = None
    latest_imu_quat = None
    last_cv_rotation = None
    camera_world_rotation = None

    first_cv = True
    for i, (type, ts, reading) in enumerate(all_events):
        if type == "IMU":
            imu_dt = None if previous_imu_ts is None else (ts - previous_imu_ts)
            _update_filter_dt(filter, imu_dt, dt)
            sr = StylusReading.from_json(reading)
            if sr.quat is not None:
                latest_imu_quat = sr.quat
                if camera_world_rotation is None and last_cv_rotation is not None:
                    camera_world_rotation = _compute_camera_world_rotation(
                        sr.quat, imu_alignment, last_cv_rotation
                    )
                if camera_world_rotation is not None:
                    aligned_quat = _align_imu_quaternion_to_camera_frame(
                        sr.quat, imu_alignment, camera_world_rotation
                    )
                    filter.update_imu_quaternion(aligned_quat)
            else:
                mag = None if sr.mag is None else imu_alignment @ sr.mag
                filter.update_imu(imu_alignment @ sr.accel, imu_alignment @ sr.gyro, mag)
            previous_imu_ts = ts
            
            # If we haven't seen CV yet, we can't initialize position
            if first_cv:
                continue
                
            # Add tip position even for IMU updates to have high-frequency trajectory
            if mode != "cv_only":
                center = filter.fs.state[fc.i_pos]
                q = Quaternion(filter.fs.state[fc.i_quat])
                tip = center + q.rotation_matrix @ CENTER_TO_TIP_BODY
                trajectory.append({"t": ts, "x": tip[0], "y": tip[1], "z": tip[2]})
        elif type == "CV":
            # CV reading contains dodecahedron center position
            center_pos = np.array(reading["center_pos_cam"])
            r_cam = np.array(reading["R_cam"])
            last_cv_rotation = r_cam
            q_cam = Quaternion(matrix=r_cam).elements
            
            if first_cv:
                # Initialize filter with first CV reading
                filter.fs = fc.FilterState(initial_state(center_pos, q_cam).state, filter.fs.statecov)
                filter.last_camera_position = center_pos.copy()
                filter.last_camera_timestamp = ts
                if latest_imu_quat is not None and camera_world_rotation is None:
                    camera_world_rotation = _compute_camera_world_rotation(
                        latest_imu_quat, imu_alignment, r_cam
                    )
                first_cv = False
                # Append first point
                tip = center_pos + r_cam @ CENTER_TO_TIP_BODY
                trajectory.append({"t": ts, "x": tip[0], "y": tip[1], "z": tip[2]})
                continue
            
            if mode == "cv_only":
                # CV-only: just use the center position directly
                filter.fs = fc.FilterState(initial_state(center_pos, q_cam).state, filter.fs.statecov)
                filter.observe_camera_pose(center_pos, ts)
                # Calculate tip from center and rotation
                tip = center_pos + r_cam @ CENTER_TO_TIP_BODY
                trajectory.append({"t": ts, "x": tip[0], "y": tip[1], "z": tip[2]})
            elif mode == "standard":
                # Standard mode uses the custom 7D fuse
                filter.fs = standard_fuse_camera(filter.fs, center_pos, q_cam)
                filter.observe_camera_pose(center_pos, ts)
                # Add trajectory point after fusion
                center = filter.fs.state[fc.i_pos]
                q = Quaternion(filter.fs.state[fc.i_quat])
                tip = center + q.rotation_matrix @ CENTER_TO_TIP_BODY
                trajectory.append({"t": ts, "x": tip[0], "y": tip[1], "z": tip[2]})
            else:
                # Decoupled mode uses update_camera
                filter.update_camera(center_pos, r_cam, timestamp=ts)
                # Add trajectory point after update
                center = filter.fs.state[fc.i_pos]
                q = Quaternion(filter.fs.state[fc.i_quat])
                tip = center + q.rotation_matrix @ CENTER_TO_TIP_BODY
                trajectory.append({"t": ts, "x": tip[0], "y": tip[1], "z": tip[2]})

    # Restore original functions if modified
    fc.camera_measurement = original_camera_measurement
    
    return pd.DataFrame(trajectory)

def visualize(results_dict, output_path):
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Plot
    ax = fig.add_subplot(221, projection='3d')
    # XY Plane (Top view)
    ax2 = fig.add_subplot(222)
    # Z-axis over time (Absolute to see lifting)
    ax3 = fig.add_subplot(223)
    # Z-axis jitter (Normalized)
    ax4 = fig.add_subplot(224)

    # Define plotting styles for each workflow
    plot_styles = {
        "CV Only (Raw)": {"alpha": 0.8, "linewidth": 2.0, "color": 'blue', "linestyle": ':', "zorder": 1},
        "Standard EKF (Coupled)": {"alpha": 0.8, "linewidth": 2.0, "color": 'orange', "linestyle": '-', "zorder": 2},
        "Decoupled EKF (Proposed)": {"alpha": 0.9, "linewidth": 2.5, "color": 'green', "linestyle": '-', "zorder": 3}
    }

    # Plot order: CV Only (Raw) first, then Standard EKF, then Decoupled on top
    # This ensures the EKF results are clearly visible over the noisy raw data
    plot_order = ["CV Only (Raw)", "Standard EKF (Coupled)", "Decoupled EKF (Proposed)"]
    sorted_results = {k: results_dict[k] for k in plot_order if k in results_dict}

    for name, df in sorted_results.items():
        if not df.empty:
            style = plot_styles.get(name, {})
            
            # 3D Plot
            ax.plot(df["x"], df["y"], df["z"], label=name, **style)
            
            # XY Plane (Top view)
            ax2.plot(df['x'], df['y'], label=name, **style)
            
            # Z-axis over time (Absolute to see lifting)
            ax3.plot(df['t'] - df['t'].iloc[0], df['z'], label=name, alpha=style.get('alpha', 0.8))
            
            # Z-axis jitter (Normalized)
            z_norm = df['z'] - df['z'].rolling(window=10, center=True).mean()
            ax4.plot(df['t'] - df['t'].iloc[0], z_norm, label=name, alpha=style.get('alpha', 0.8))

    ax.set_title("3D Pen-Tip Trajectory")
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    ax2.set_title("XY Plane (Top View)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.axis('equal') # Maintain aspect ratio for the rectangle
    ax2.grid(True)
    ax2.legend()

    ax3.set_title("Z-Axis (Absolute Height)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Z (m)")
    ax3.grid(True)
    ax3.legend()

    ax4.set_title("Z-Axis Jitter (High-pass filtered)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Z Noise (m)")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare offline CV/EKF workflows")
    parser.add_argument("data_file", nargs="?", default="outputs/my_data.json")
    parser.add_argument(
        "--imu-calibration",
        default=None,
        help=f"Path to IMU alignment calibration JSON. Defaults to {DEFAULT_ALIGNMENT_PATH}",
    )
    args = parser.parse_args()

    data_file = args.data_file
    print(f"Using data file: {data_file}")
    
    print("Running CV Only workflow...")
    df_cv = run_workflow(data_file, mode="cv_only", calibration_path=args.imu_calibration)
    
    print("Running Standard EKF workflow...")
    df_std = run_workflow(data_file, mode="standard", calibration_path=args.imu_calibration)
    
    print("Running Decoupled EKF workflow...")
    df_dec = run_workflow(data_file, mode="decoupled", calibration_path=args.imu_calibration)
    
    results = {
        "CV Only (Raw)": df_cv,
        "Standard EKF (Coupled)": df_std,
        "Decoupled EKF (Proposed)": df_dec
    }
    
    for name, df in results.items():
        print(f"Workflow {name}: {len(df)} points")
        if not df.empty:
            print(f"  {name} - X mean: {df['x'].mean():.4f}, Y mean: {df['y'].mean():.4f}, Z mean: {df['z'].mean():.4f}")
    
    # Plot order: CV Only (Raw) first, then Standard EKF, then Decoupled on top
    # This ensures the EKF results are clearly visible over the noisy raw data
    plot_order = ["CV Only (Raw)", "Standard EKF (Coupled)", "Decoupled EKF (Proposed)"]
    sorted_results = {}
    for k in plot_order:
        if k in results:
            sorted_results[k] = results[k]
    
    visualize(sorted_results, str(Path(__file__).resolve().parent / "outputs" / "workflow_comparison.png"))
