import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from app.camera_world_calibration import (
    DEFAULT_DODECA_BODY_PATH,
    DEFAULT_IMU_CV_BIAS_PATH,
    _paired_camera_world_estimates,
    _rotation_matrix_to_euler_xyz_deg,
    _rotation_matrix_to_wxyz,
    _rotation_spread_stats,
    load_dodeca_body_calibration,
    save_imu_cv_bias,
)
from app.imu_alignment import DEFAULT_ALIGNMENT_PATH, load_alignment_calibration


def _rotation_angle_deg(rotation_matrix):
    return float(np.degrees(np.linalg.norm(R.from_matrix(rotation_matrix).as_rotvec())))


def _mean_rotation(rotation_matrices):
    return R.from_matrix(np.asarray(rotation_matrices, dtype=float)).mean().as_matrix()


def _pose_summaries(
    recording_paths,
    imu_to_body,
    dodeca_from_body,
    max_pair_dt_s,
    imu_quat_convention,
):
    summaries = []
    for path in recording_paths:
        recording_path = Path(path)
        data = json.loads(recording_path.read_text())
        paired = _paired_camera_world_estimates(
            data,
            imu_to_body,
            dodeca_from_body,
            max_pair_dt_s,
            imu_quat_convention=imu_quat_convention,
        )
        summaries.append(
            {
                "path": str(recording_path),
                "paired_count": int(len(paired["pair_dts"])),
                "mean_pair_dt_ms": float(np.mean(paired["pair_dts"]) * 1000.0),
                "cam_from_body": _mean_rotation(paired["cv_rotations"]),
                "world_from_body": _mean_rotation(paired["world_body_rotations"]),
                "sample_cam_from_body": paired["cv_rotations"],
                "sample_world_from_body": paired["world_body_rotations"],
            }
        )
    return summaries


def _relative_pairs(pose_summaries):
    pairs = []
    for i, j in combinations(range(len(pose_summaries)), 2):
        cam_i = pose_summaries[i]["cam_from_body"]
        cam_j = pose_summaries[j]["cam_from_body"]
        world_i = pose_summaries[i]["world_from_body"]
        world_j = pose_summaries[j]["world_from_body"]
        pairs.append(
            {
                "i": i,
                "j": j,
                "cv_relative": cam_j.T @ cam_i,
                "imu_relative": world_j.T @ world_i,
            }
        )
    return pairs


def _fit_bias_from_pairs(pairs):
    def residual(rotvec):
        bias = R.from_rotvec(rotvec).as_matrix()
        errors = []
        for pair in pairs:
            predicted_cv_relative = bias.T @ pair["imu_relative"] @ bias
            error_matrix = predicted_cv_relative.T @ pair["cv_relative"]
            errors.extend(R.from_matrix(error_matrix).as_rotvec())
        return np.asarray(errors, dtype=float)

    seeds = [
        np.zeros(3),
        np.array([0.05, 0.0, 0.0]),
        np.array([0.0, 0.05, 0.0]),
        np.array([0.0, 0.0, 0.05]),
        np.array([0.1, 0.1, 0.0]),
        np.array([-0.1, 0.1, 0.05]),
    ]
    best = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=10000,
        )
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result

    bias = R.from_rotvec(best.x).as_matrix()
    pair_residuals = []
    for pair in pairs:
        predicted_cv_relative = bias.T @ pair["imu_relative"] @ bias
        error_matrix = predicted_cv_relative.T @ pair["cv_relative"]
        pair_residuals.append(
            {
                "i": int(pair["i"]),
                "j": int(pair["j"]),
                "error_deg": _rotation_angle_deg(error_matrix),
            }
        )
    return bias, pair_residuals


def _camera_world_residuals(pose_summaries, bias):
    cam_samples = []
    world_samples = []
    for summary in pose_summaries:
        cam_samples.extend(summary["sample_cam_from_body"])
        world_samples.extend(summary["sample_world_from_body"])

    cam_samples = np.asarray(cam_samples, dtype=float)
    world_samples = np.asarray(world_samples, dtype=float)
    baseline_estimates = np.array(
        [cam @ world.T for cam, world in zip(cam_samples, world_samples)]
    )
    corrected_estimates = np.array(
        [cam @ bias.T @ world.T for cam, world in zip(cam_samples, world_samples)]
    )

    baseline = _rotation_spread_stats(baseline_estimates)
    corrected = _rotation_spread_stats(corrected_estimates)
    return {
        "baseline_mean_deg": baseline["mean_error_deg"],
        "baseline_median_deg": baseline["median_error_deg"],
        "baseline_p95_deg": baseline["p95_error_deg"],
        "baseline_max_deg": baseline["max_error_deg"],
        "corrected_mean_deg": corrected["mean_error_deg"],
        "corrected_median_deg": corrected["median_error_deg"],
        "corrected_p95_deg": corrected["p95_error_deg"],
        "corrected_max_deg": corrected["max_error_deg"],
    }


def estimate_imu_cv_bias(
    recording_paths,
    imu_to_body_path=DEFAULT_ALIGNMENT_PATH,
    dodeca_body_path=DEFAULT_DODECA_BODY_PATH,
    max_pair_dt_s=0.04,
    imu_quat_convention="world_from_imu",
):
    if len(recording_paths) < 2:
        raise ValueError("At least two stationary pose recordings are required")

    imu_alignment = load_alignment_calibration(imu_to_body_path)
    if imu_alignment is None:
        raise FileNotFoundError(f"IMU-to-body calibration not found: {imu_to_body_path}")
    imu_to_body = np.asarray(imu_alignment["rotation_matrix"], dtype=float)
    dodeca_body = load_dodeca_body_calibration(dodeca_body_path)
    dodeca_from_body = np.asarray(dodeca_body["dodeca_from_body"], dtype=float)

    pose_summaries = _pose_summaries(
        recording_paths,
        imu_to_body,
        dodeca_from_body,
        max_pair_dt_s,
        imu_quat_convention,
    )
    pairs = _relative_pairs(pose_summaries)
    bias, pair_residuals = _fit_bias_from_pairs(pairs)
    rotvec = R.from_matrix(bias).as_rotvec()
    angle_deg = float(np.degrees(np.linalg.norm(rotvec)))
    axis = (
        (rotvec / np.linalg.norm(rotvec)).tolist()
        if np.linalg.norm(rotvec) > 1e-12
        else [0.0, 0.0, 0.0]
    )

    return {
        "rotation_matrix": bias,
        "quaternion_wxyz": _rotation_matrix_to_wxyz(bias),
        "euler_xyz_deg": _rotation_matrix_to_euler_xyz_deg(bias),
        "angle_deg": angle_deg,
        "axis": axis,
        "method": "relative_pose_hand_eye_bias",
        "recordings": [
            {
                "path": summary["path"],
                "paired_count": summary["paired_count"],
                "mean_pair_dt_ms": summary["mean_pair_dt_ms"],
            }
            for summary in pose_summaries
        ],
        "pose_pairs": pair_residuals,
        "residuals": _camera_world_residuals(pose_summaries, bias),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a fixed body-frame bias B between IMU-derived and CV-derived "
            "pen orientations from stationary calibration poses."
        )
    )
    parser.add_argument("recordings", nargs="+", help="Stationary merged pose recordings")
    parser.add_argument(
        "--imu-to-body",
        default=str(DEFAULT_ALIGNMENT_PATH),
        help="Path to the fixed IMU-to-body calibration JSON.",
    )
    parser.add_argument(
        "--dodeca-body",
        default=str(DEFAULT_DODECA_BODY_PATH),
        help="Path to the dodeca-to-pen-body calibration JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_IMU_CV_BIAS_PATH),
        help="Output IMU/CV body-frame bias JSON path.",
    )
    parser.add_argument(
        "--max-pair-dt-ms",
        type=float,
        default=40.0,
        help="Maximum allowed timestamp gap between a CV frame and nearest IMU sample.",
    )
    parser.add_argument(
        "--imu-quat-convention",
        choices=["world_from_imu", "imu_from_world"],
        default="world_from_imu",
        help="Interpretation of the streamed IMU quaternion matrix.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the fitted bias without writing the output JSON.",
    )
    args = parser.parse_args()

    result = estimate_imu_cv_bias(
        args.recordings,
        imu_to_body_path=args.imu_to_body,
        dodeca_body_path=args.dodeca_body,
        max_pair_dt_s=args.max_pair_dt_ms / 1000.0,
        imu_quat_convention=args.imu_quat_convention,
    )

    print("=" * 60)
    print("IMU/CV BODY-FRAME BIAS")
    print("=" * 60)
    print(f"Method: {result['method']}")
    print(f"Angle: {result['angle_deg']:.3f} deg")
    print(f"Axis: {np.array2string(np.asarray(result['axis']), precision=4)}")
    print(
        "Euler xyz: "
        f"{np.array2string(np.asarray(result['euler_xyz_deg']), precision=3)} deg"
    )
    print("Residual spread:")
    residuals = result["residuals"]
    print(
        "  baseline:  "
        f"mean={residuals['baseline_mean_deg']:.3f}, "
        f"median={residuals['baseline_median_deg']:.3f}, "
        f"p95={residuals['baseline_p95_deg']:.3f}, "
        f"max={residuals['baseline_max_deg']:.3f} deg"
    )
    print(
        "  corrected: "
        f"mean={residuals['corrected_mean_deg']:.3f}, "
        f"median={residuals['corrected_median_deg']:.3f}, "
        f"p95={residuals['corrected_p95_deg']:.3f}, "
        f"max={residuals['corrected_max_deg']:.3f} deg"
    )
    print("Pose-pair residuals:")
    for pair in result["pose_pairs"]:
        print(f"  {pair['i']} <-> {pair['j']}: {pair['error_deg']:.3f} deg")

    if not args.no_save:
        save_imu_cv_bias(result, args.output)
        print(f"Saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
