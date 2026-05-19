import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from app.imu_alignment import DEFAULT_ALIGNMENT_PATH, load_alignment_calibration


DEFAULT_CAMERA_WORLD_PATH = (
    Path(__file__).resolve().parents[1] / "calibration" / "camera_from_world.json"
)
DEFAULT_DODECA_BODY_PATH = (
    Path(__file__).resolve().parents[1] / "calibration" / "dodeca_to_body.json"
)
DEFAULT_IMU_CV_BIAS_PATH = (
    Path(__file__).resolve().parents[1] / "calibration" / "imu_cv_bias.json"
)


def _wxyz_to_rotation_matrix(quat):
    quat = np.asarray(quat, dtype=float)
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _rotation_matrix_to_wxyz(rotation_matrix):
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _rotation_matrix_to_euler_xyz_deg(rotation_matrix):
    return R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)


def _rotation_angle_deg(rotation_matrix):
    return float(np.degrees(np.linalg.norm(R.from_matrix(rotation_matrix).as_rotvec())))


def _rotation_spread_stats(rotation_matrices):
    rotation_matrices = np.asarray(rotation_matrices, dtype=float)
    mean_rotation = R.from_matrix(rotation_matrices).mean().as_matrix()
    errors = np.array(
        [_rotation_angle_deg(mean_rotation.T @ matrix) for matrix in rotation_matrices],
        dtype=float,
    )
    return {
        "mean_rotation": mean_rotation,
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "p95_error_deg": float(np.percentile(errors, 95)),
        "max_error_deg": float(np.max(errors)),
    }


def _as_rotation_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def load_dodeca_body_calibration(path=DEFAULT_DODECA_BODY_PATH):
    calibration_path = Path(path)
    if not calibration_path.exists():
        return {
            "body_from_dodeca": np.eye(3),
            "dodeca_from_body": np.eye(3),
            "path": None,
            "method": "identity",
        }

    payload = json.loads(calibration_path.read_text())
    body_from_dodeca = _as_rotation_matrix(payload["body_from_dodeca"])
    dodeca_from_body = _as_rotation_matrix(
        payload.get("dodeca_from_body", body_from_dodeca.T),
    )
    payload["body_from_dodeca"] = body_from_dodeca
    payload["dodeca_from_body"] = dodeca_from_body
    payload["path"] = str(calibration_path)
    return payload


def load_imu_cv_bias(path=DEFAULT_IMU_CV_BIAS_PATH):
    bias_path = Path(path)
    if not bias_path.exists():
        return {
            "rotation_matrix": np.eye(3),
            "quaternion_wxyz": _rotation_matrix_to_wxyz(np.eye(3)),
            "euler_xyz_deg": np.zeros(3),
            "angle_deg": 0.0,
            "path": None,
            "method": "identity",
        }

    payload = json.loads(bias_path.read_text())
    rotation_matrix = _as_rotation_matrix(payload["rotation_matrix"])
    payload["rotation_matrix"] = rotation_matrix
    payload["quaternion_wxyz"] = np.asarray(
        payload.get("quaternion_wxyz", _rotation_matrix_to_wxyz(rotation_matrix)),
        dtype=float,
    )
    payload["euler_xyz_deg"] = np.asarray(
        payload.get("euler_xyz_deg", _rotation_matrix_to_euler_xyz_deg(rotation_matrix)),
        dtype=float,
    )
    payload["angle_deg"] = float(
        payload.get("angle_deg", _rotation_angle_deg(rotation_matrix))
    )
    payload["path"] = str(bias_path)
    return payload


def save_imu_cv_bias(result, output_path=DEFAULT_IMU_CV_BIAS_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_matrix = np.asarray(result["rotation_matrix"], dtype=float)
    payload = {
        "rotation_matrix": rotation_matrix.tolist(),
        "quaternion_wxyz": np.asarray(
            result.get("quaternion_wxyz", _rotation_matrix_to_wxyz(rotation_matrix)),
            dtype=float,
        ).tolist(),
        "euler_xyz_deg": np.asarray(
            result.get("euler_xyz_deg", _rotation_matrix_to_euler_xyz_deg(rotation_matrix)),
            dtype=float,
        ).tolist(),
        "angle_deg": float(result.get("angle_deg", _rotation_angle_deg(rotation_matrix))),
        "method": result.get("method", "relative_pose_hand_eye_bias"),
        "recordings": result.get("recordings", []),
        "pose_pairs": result.get("pose_pairs", []),
        "residuals": result.get("residuals", {}),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _world_from_imu_rotation(imu_quat, convention):
    rotation = _wxyz_to_rotation_matrix(imu_quat)
    if convention == "world_from_imu":
        return rotation
    if convention == "imu_from_world":
        return rotation.T
    raise ValueError(f"Unknown IMU quaternion convention: {convention}")


def _paired_camera_world_estimates(
    data,
    imu_to_body,
    dodeca_from_body,
    max_pair_dt_s,
    imu_quat_convention="world_from_imu",
    imu_cv_bias=None,
):
    imu_readings = [
        reading for reading in data.get("imu_readings", []) if reading.get("quat") is not None
    ]
    cv_readings = data.get("cv_readings", [])
    if not imu_readings or not cv_readings:
        raise ValueError("Recording must contain both quaternion IMU readings and CV readings")

    imu_times = np.array([reading["local_timestamp"] for reading in imu_readings], dtype=float)
    imu_quats = np.array([reading["quat"] for reading in imu_readings], dtype=float)

    estimates = []
    pair_dts = []
    cv_rotations = []
    imu_rotations = []
    world_body_rotations = []
    bias = np.eye(3) if imu_cv_bias is None else np.asarray(imu_cv_bias, dtype=float)
    for cv_reading in cv_readings:
        cv_time = float(cv_reading["local_timestamp"])
        imu_idx = int(np.argmin(np.abs(imu_times - cv_time)))
        pair_dt = abs(float(imu_times[imu_idx] - cv_time))
        if pair_dt > max_pair_dt_s:
            continue

        camera_from_dodeca = np.array(cv_reading["R_cam"], dtype=float)
        camera_from_body = camera_from_dodeca @ dodeca_from_body
        world_from_imu = _world_from_imu_rotation(
            imu_quats[imu_idx],
            imu_quat_convention,
        )
        world_from_body = world_from_imu @ imu_to_body.T
        camera_from_world = camera_from_body @ bias.T @ world_from_body.T

        estimates.append(camera_from_world)
        pair_dts.append(pair_dt)
        cv_rotations.append(camera_from_body)
        imu_rotations.append(world_from_imu)
        world_body_rotations.append(world_from_body)

    if not estimates:
        raise ValueError(
            f"No CV/IMU pairs were within {max_pair_dt_s * 1000.0:.1f} ms"
        )

    return {
        "camera_from_world_estimates": np.asarray(estimates, dtype=float),
        "pair_dts": np.asarray(pair_dts, dtype=float),
        "cv_rotations": np.asarray(cv_rotations, dtype=float),
        "imu_rotations": np.asarray(imu_rotations, dtype=float),
        "world_body_rotations": np.asarray(world_body_rotations, dtype=float),
        "imu_count": len(imu_readings),
        "cv_count": len(cv_readings),
    }


def estimate_camera_world_from_recordings(
    recording_paths,
    imu_to_body_path=DEFAULT_ALIGNMENT_PATH,
    dodeca_body_path=DEFAULT_DODECA_BODY_PATH,
    max_pair_dt_s=0.04,
    imu_quat_convention="world_from_imu",
    imu_cv_bias_path=DEFAULT_IMU_CV_BIAS_PATH,
):
    imu_alignment = load_alignment_calibration(imu_to_body_path)
    if imu_alignment is None:
        raise FileNotFoundError(f"IMU-to-body calibration not found: {imu_to_body_path}")

    imu_to_body = np.asarray(imu_alignment["rotation_matrix"], dtype=float)
    dodeca_body = load_dodeca_body_calibration(dodeca_body_path)
    dodeca_from_body = np.asarray(dodeca_body["dodeca_from_body"], dtype=float)
    imu_cv_bias = load_imu_cv_bias(imu_cv_bias_path)
    bias_rotation = np.asarray(imu_cv_bias["rotation_matrix"], dtype=float)
    all_estimates = []
    recording_summaries = []

    for path in recording_paths:
        recording_path = Path(path)
        data = json.loads(recording_path.read_text())
        paired = _paired_camera_world_estimates(
            data,
            imu_to_body,
            dodeca_from_body,
            max_pair_dt_s,
            imu_quat_convention=imu_quat_convention,
            imu_cv_bias=bias_rotation,
        )
        estimates = paired["camera_from_world_estimates"]
        all_estimates.extend(estimates)

        estimate_stats = _rotation_spread_stats(estimates)
        cv_stats = _rotation_spread_stats(paired["cv_rotations"])
        imu_stats = _rotation_spread_stats(paired["imu_rotations"])
        pair_dts = paired["pair_dts"]
        recording_summaries.append(
            {
                "path": str(recording_path),
                "imu_count": int(paired["imu_count"]),
                "cv_count": int(paired["cv_count"]),
                "paired_count": int(len(estimates)),
                "mean_pair_dt_ms": float(np.mean(pair_dts) * 1000.0),
                "max_pair_dt_ms": float(np.max(pair_dts) * 1000.0),
                    "camera_from_world_euler_xyz_deg": _rotation_matrix_to_euler_xyz_deg(
                        estimate_stats["mean_rotation"]
                    ),
                    "camera_from_world_angle_deg": _rotation_angle_deg(
                        estimate_stats["mean_rotation"]
                    ),
                "camera_from_world_mean_error_deg": estimate_stats["mean_error_deg"],
                "camera_from_world_p95_error_deg": estimate_stats["p95_error_deg"],
                "camera_from_world_max_error_deg": estimate_stats["max_error_deg"],
                    "camera_from_world_quaternion_wxyz": _rotation_matrix_to_wxyz(
                        estimate_stats["mean_rotation"]
                    ),
                "cv_rotation_p95_error_deg": cv_stats["p95_error_deg"],
                "imu_rotation_p95_error_deg": imu_stats["p95_error_deg"],
            }
        )

    all_estimates = np.asarray(all_estimates, dtype=float)
    overall_stats = _rotation_spread_stats(all_estimates)
    camera_from_world = overall_stats["mean_rotation"]

    return {
        "camera_from_world": camera_from_world,
        "quaternion_wxyz": _rotation_matrix_to_wxyz(camera_from_world),
        "method": "stationary_cv_imu_rotation_average",
        "imu_quat_convention": imu_quat_convention,
        "imu_to_body_path": str(Path(imu_to_body_path)),
        "dodeca_body_path": dodeca_body["path"],
        "dodeca_body_method": dodeca_body["method"],
        "imu_cv_bias_path": imu_cv_bias["path"],
        "imu_cv_bias_method": imu_cv_bias["method"],
        "max_pair_dt_ms": float(max_pair_dt_s * 1000.0),
        "sample_count": int(len(all_estimates)),
        "mean_error_deg": overall_stats["mean_error_deg"],
        "median_error_deg": overall_stats["median_error_deg"],
        "p95_error_deg": overall_stats["p95_error_deg"],
        "max_error_deg": overall_stats["max_error_deg"],
        "recordings": recording_summaries,
    }


def save_camera_world_calibration(result, output_path=DEFAULT_CAMERA_WORLD_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: _to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(item) for item in value]
        return value

    payload = {
        "camera_from_world": np.asarray(result["camera_from_world"], dtype=float),
        "quaternion_wxyz": np.asarray(result["quaternion_wxyz"], dtype=float),
        "method": result["method"],
        "imu_quat_convention": result["imu_quat_convention"],
        "imu_to_body_path": result["imu_to_body_path"],
        "dodeca_body_path": result["dodeca_body_path"],
        "dodeca_body_method": result["dodeca_body_method"],
        "imu_cv_bias_path": result.get("imu_cv_bias_path"),
        "imu_cv_bias_method": result.get("imu_cv_bias_method", "identity"),
        "max_pair_dt_ms": float(result["max_pair_dt_ms"]),
        "sample_count": int(result["sample_count"]),
        "mean_error_deg": float(result["mean_error_deg"]),
        "median_error_deg": float(result["median_error_deg"]),
        "p95_error_deg": float(result["p95_error_deg"]),
        "max_error_deg": float(result["max_error_deg"]),
        "recordings": result["recordings"],
    }
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2))


def load_camera_world_calibration(path=DEFAULT_CAMERA_WORLD_PATH):
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None

    payload = json.loads(calibration_path.read_text())
    payload["camera_from_world"] = np.array(payload["camera_from_world"], dtype=float)
    payload["quaternion_wxyz"] = np.array(payload["quaternion_wxyz"], dtype=float)
    payload["path"] = str(calibration_path)
    return payload
