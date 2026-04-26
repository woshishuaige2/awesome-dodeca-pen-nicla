import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import app.filter_core as fc


DEFAULT_ALIGNMENT_PATH = Path(__file__).resolve().parents[1] / "calibration" / "imu_to_body.json"


def _shortest_arc_rotation(source_vec, target_vec):
    """
    Return the minimum-angle rotation that maps source_vec onto target_vec.
    """
    src_norm = np.linalg.norm(source_vec)
    dst_norm = np.linalg.norm(target_vec)
    if src_norm < 1e-9 or dst_norm < 1e-9:
        return np.eye(3)

    src = source_vec / src_norm
    dst = target_vec / dst_norm
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))

    if cross_norm < 1e-9:
        if dot > 0:
            return np.eye(3)

        reference = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, reference)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi).as_matrix()

    axis = cross / cross_norm
    angle = np.arccos(dot)
    return R.from_rotvec(axis * angle).as_matrix()


def _rotation_matrix_to_wxyz(rotation_matrix):
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _signed_permutation_matrices():
    matrices = []
    for perm in __import__("itertools").permutations(range(3)):
        base = np.zeros((3, 3))
        for i, j in enumerate(perm):
            base[i, j] = 1.0
        for signs in __import__("itertools").product([-1.0, 1.0], repeat=3):
            candidate = np.diag(signs) @ base
            # Keep only proper right-handed rotations. Reflections have
            # determinant -1 and are rejected by scipy Rotation.from_matrix().
            if np.linalg.det(candidate) > 0.0:
                matrices.append(candidate)
    return matrices


SIGNED_PERMUTATION_MATRICES = _signed_permutation_matrices()


def _solve_alignment_and_gravity(summaries, initial_rotation=None):
    gravity_target = float(
        np.mean([np.linalg.norm(summary["measured_accel"]) for summary in summaries])
    )

    def residual(params):
        rot = R.from_rotvec(params[:3]).as_matrix()
        gravity_camera = params[3:6]
        values = []
        for summary in summaries:
            predicted_gravity = summary["mean_rotation"] @ (rot @ summary["measured_accel"])
            values.extend(np.sqrt(summary["weight"]) * (predicted_gravity - gravity_camera))
        values.append(np.linalg.norm(gravity_camera) - gravity_target)
        return np.asarray(values, dtype=float)

    if initial_rotation is None:
        initial_rotation = np.eye(3)
    initial_rotvec = R.from_matrix(initial_rotation).as_rotvec()
    initial_gravity = np.mean(
        [summary["mean_rotation"] @ (initial_rotation @ summary["measured_accel"]) for summary in summaries],
        axis=0,
    )

    result = least_squares(
        residual,
        np.concatenate((initial_rotvec, initial_gravity)),
        method="lm",
        max_nfev=5000,
    )
    rotation_matrix = R.from_rotvec(result.x[:3]).as_matrix()
    gravity_camera = result.x[3:6]

    per_recording_residuals = []
    for summary in summaries:
        predicted_gravity = summary["mean_rotation"] @ (rotation_matrix @ summary["measured_accel"])
        per_recording_residuals.append(
            float(np.linalg.norm(predicted_gravity - gravity_camera))
        )

    return {
        "rotation_matrix": rotation_matrix,
        "gravity_camera": gravity_camera,
        "mean_residual": float(np.mean(per_recording_residuals)),
        "max_residual": float(np.max(per_recording_residuals)),
        "per_recording_residuals": per_recording_residuals,
        "optimization_cost": float(result.cost),
        "success": bool(result.success),
    }


def summarize_stationary_recording(data):
    imu_readings = data.get("imu_readings", [])
    cv_readings = data.get("cv_readings", [])
    if not imu_readings or not cv_readings:
        raise ValueError("Recording must contain both IMU and CV readings")

    accel = np.array([r["accel"] for r in imu_readings], dtype=float)
    gyro = np.array([r["gyro"] for r in imu_readings], dtype=float)
    rotations = np.array([r["R_cam"] for r in cv_readings], dtype=float)

    mean_rotation = R.from_matrix(rotations).mean().as_matrix()
    measured_accel = np.mean(accel, axis=0)

    accel_std_norm = float(np.linalg.norm(np.std(accel, axis=0)))
    gyro_std_norm = float(np.linalg.norm(np.std(gyro, axis=0)))
    center_std_norm = float(
        np.linalg.norm(np.std(np.array([r["center_pos_cam"] for r in cv_readings], dtype=float), axis=0))
    )
    weight = 1.0 / max(accel_std_norm, 1e-6)

    return {
        "imu_count": len(imu_readings),
        "cv_count": len(cv_readings),
        "measured_accel": measured_accel,
        "mean_rotation": mean_rotation,
        "accel_std_norm": accel_std_norm,
        "gyro_std_norm": gyro_std_norm,
        "center_std_norm": center_std_norm,
        "weight": weight,
    }


def estimate_alignment_from_streams(imu_readings, cv_readings, calibration_samples=60):
    """
    Fallback estimator for a single stationary recording.
    """
    if not imu_readings or not cv_readings:
        return {
            "rotation_matrix": np.eye(3),
            "gravity_camera": fc.DEFAULT_GRAVITY_VECTOR.copy(),
            "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "method": "identity",
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "observable_axes": 0,
        }

    sample_count = min(len(imu_readings), calibration_samples)
    accel = np.array([r["accel"] for r in imu_readings[:sample_count]], dtype=float)
    rotations = np.array([r["R_cam"] for r in cv_readings], dtype=float)
    summary = {
        "measured_accel": np.mean(accel, axis=0),
        "mean_rotation": R.from_matrix(rotations).mean().as_matrix(),
        "weight": 1.0,
    }
    fit = _solve_alignment_and_gravity([summary])

    return {
        "rotation_matrix": fit["rotation_matrix"],
        "gravity_camera": fit["gravity_camera"],
        "quaternion_wxyz": _rotation_matrix_to_wxyz(fit["rotation_matrix"]),
        "method": "single_recording_gravity_fit",
        "mean_residual": fit["mean_residual"],
        "max_residual": fit["max_residual"],
        "observable_axes": 2,
    }


def estimate_alignment_from_recordings(recording_paths):
    summaries = []
    for path in recording_paths:
        file_path = Path(path)
        data = json.loads(file_path.read_text())
        summary = summarize_stationary_recording(data)
        summary["path"] = str(file_path)
        summaries.append(summary)

    if not summaries:
        raise ValueError("At least one recording is required for calibration")

    measured = np.array([s["measured_accel"] for s in summaries], dtype=float)
    best_fit = None
    for initial_rotation in SIGNED_PERMUTATION_MATRICES:
        fit = _solve_alignment_and_gravity(summaries, initial_rotation)
        if best_fit is None or fit["mean_residual"] < best_fit["mean_residual"]:
            best_fit = fit

    rotation_matrix = best_fit["rotation_matrix"]
    gravity_camera = best_fit["gravity_camera"]
    residual_norms = np.asarray(best_fit["per_recording_residuals"], dtype=float)
    method = "joint_gravity_fit"
    observable_axes = 2 if len(summaries) == 1 else 3

    return {
        "rotation_matrix": rotation_matrix,
        "gravity_camera": gravity_camera,
        "quaternion_wxyz": _rotation_matrix_to_wxyz(rotation_matrix),
        "method": method,
        "observable_axes": observable_axes,
        "rmsd": float(best_fit["optimization_cost"]),
        "mean_residual": float(np.mean(residual_norms)),
        "max_residual": float(np.max(residual_norms)),
        "recordings": [
            {
                "path": s["path"],
                "imu_count": s["imu_count"],
                "cv_count": s["cv_count"],
                "measured_accel": s["measured_accel"].tolist(),
                "accel_std_norm": s["accel_std_norm"],
                "gyro_std_norm": s["gyro_std_norm"],
                "center_std_norm": s["center_std_norm"],
                "residual_norm": float(best_fit["per_recording_residuals"][idx]),
            }
            for idx, s in enumerate(summaries)
        ],
    }


def save_alignment_calibration(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "rotation_matrix": np.asarray(result["rotation_matrix"], dtype=float).tolist(),
        "gravity_camera": np.asarray(result["gravity_camera"], dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(result["quaternion_wxyz"], dtype=float).tolist(),
        "method": result["method"],
        "observable_axes": int(result["observable_axes"]),
        "rmsd": float(result.get("rmsd", 0.0)),
        "mean_residual": float(result.get("mean_residual", 0.0)),
        "max_residual": float(result.get("max_residual", 0.0)),
        "recordings": result.get("recordings", []),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def load_alignment_calibration(path=None):
    calibration_path = Path(path) if path is not None else DEFAULT_ALIGNMENT_PATH
    if not calibration_path.exists():
        return None

    payload = json.loads(calibration_path.read_text())
    payload["rotation_matrix"] = np.array(payload["rotation_matrix"], dtype=float)
    payload["gravity_camera"] = np.array(
        payload.get("gravity_camera", fc.DEFAULT_GRAVITY_VECTOR),
        dtype=float,
    )
    payload["quaternion_wxyz"] = np.array(payload["quaternion_wxyz"], dtype=float)
    payload["path"] = str(calibration_path)
    return payload
