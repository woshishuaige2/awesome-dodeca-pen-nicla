"""Overlay CV- and IMU-inferred stylus body axes on raw calibration videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from app.camera_world_calibration import (
    DEFAULT_CAMERA_WORLD_PATH,
    DEFAULT_DODECA_BODY_PATH,
    DEFAULT_IMU_CV_BIAS_PATH,
    load_camera_world_calibration,
    load_dodeca_body_calibration,
    load_imu_cv_bias,
)
from app.imu_alignment import DEFAULT_ALIGNMENT_PATH, load_alignment_calibration


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_MATRIX_PATH = REPO_ROOT / "Computer_vision" / "cam_mtx.npy"
DEFAULT_DIST_COEFFS_PATH = REPO_ROOT / "Computer_vision" / "cam_dist.npy"

AXIS_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 180, 0),
    "z": (255, 0, 0),
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wxyz_to_rotation_matrix(quat_wxyz) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return R.from_quat(quat_xyzw).as_matrix()


def world_from_imu_rotation(imu_quat, convention: str) -> np.ndarray:
    rotation = wxyz_to_rotation_matrix(imu_quat)
    if convention == "world_from_imu":
        return rotation
    if convention == "imu_from_world":
        return rotation.T
    raise ValueError(f"Unknown IMU quaternion convention: {convention}")


def nearest_reading(readings: list[dict], times: np.ndarray, timestamp: float):
    if len(readings) == 0:
        return None, None

    idx = int(np.searchsorted(times, timestamp))
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(times):
        candidates.append(idx)
    if not candidates:
        return None, None

    best_idx = min(candidates, key=lambda i: abs(times[i] - timestamp))
    return readings[best_idx], abs(float(times[best_idx]) - timestamp)


def project_points(points_cam: np.ndarray, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(
        points_cam.reshape(-1, 3),
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def draw_dashed_line(frame, start, end, color, thickness=2, dash_px=10):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return

    direction = delta / length
    distance = 0.0
    while distance < length:
        seg_start = start + direction * distance
        seg_end = start + direction * min(distance + dash_px, length)
        cv2.line(
            frame,
            tuple(np.round(seg_start).astype(int)),
            tuple(np.round(seg_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash_px * 2


def draw_axis_set(
    frame,
    origin_cam: np.ndarray,
    camera_from_body: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
    label_prefix: str,
    dashed: bool = False,
):
    points_cam = [origin_cam]
    for axis_idx in range(3):
        points_cam.append(origin_cam + camera_from_body[:, axis_idx] * axis_length_m)
    points_cam = np.asarray(points_cam, dtype=float)

    if np.any(points_cam[:, 2] <= 0):
        return False

    points_px = project_points(points_cam, camera_matrix, dist_coeffs)
    origin_px = points_px[0]
    labels = ["x", "y", "z"]

    for axis_idx, axis_name in enumerate(labels, start=1):
        end_px = points_px[axis_idx]
        color = AXIS_COLORS[axis_name]
        if dashed:
            draw_dashed_line(frame, origin_px, end_px, color, thickness=2)
        else:
            cv2.line(
                frame,
                tuple(np.round(origin_px).astype(int)),
                tuple(np.round(end_px).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
        cv2.circle(frame, tuple(np.round(end_px).astype(int)), 4, color, -1)
        cv2.putText(
            frame,
            f"{label_prefix}{axis_name.upper()}",
            tuple(np.round(end_px + np.array([5.0, -5.0])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.circle(frame, tuple(np.round(origin_px).astype(int)), 4, (255, 255, 255), -1)
    return True


def rotation_error_deg(camera_from_body_cv: np.ndarray, camera_from_body_imu: np.ndarray) -> float:
    delta = camera_from_body_cv.T @ camera_from_body_imu
    return float(np.degrees(R.from_matrix(delta).magnitude()))


def draw_status(frame, lines: list[str]):
    x, y = 12, 24
    line_height = 22
    width = max(360, max((len(line) for line in lines), default=0) * 9)
    height = 12 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def render_overlay(
    merged_json_path: Path,
    video_path: Path,
    output_path: Path,
    video_zero_time_s: float,
    camera_from_world: np.ndarray,
    dodeca_from_body: np.ndarray,
    imu_to_body: np.ndarray,
    imu_cv_bias: np.ndarray,
    imu_quat_convention: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
    max_pair_dt_s: float,
):
    data = load_json(merged_json_path)
    imu_readings = [reading for reading in data.get("imu_readings", []) if reading.get("quat") is not None]
    cv_readings = data.get("cv_readings", [])
    imu_times = np.asarray([reading["local_timestamp"] for reading in imu_readings], dtype=float)
    cv_times = np.asarray([reading["local_timestamp"] for reading in cv_readings], dtype=float)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError(f"Video FPS is invalid for {video_path}: {fps}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    frame_index = 0
    rendered = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # process_video_to_cv_data.py timestamps frame_count / fps after
            # incrementing frame_count, so frame 0 corresponds to 1 / fps.
            video_elapsed_s = (frame_index + 1) / fps
            merged_time_s = video_elapsed_s - video_zero_time_s

            cv_reading, cv_dt = nearest_reading(cv_readings, cv_times, merged_time_s)
            imu_reading, imu_dt = nearest_reading(imu_readings, imu_times, merged_time_s)

            if (
                cv_reading is not None
                and imu_reading is not None
                and cv_dt is not None
                and imu_dt is not None
                and cv_dt <= max_pair_dt_s
                and imu_dt <= max_pair_dt_s
            ):
                origin_cam = np.asarray(cv_reading["center_pos_cam"], dtype=float)
                camera_from_dodeca = np.asarray(cv_reading["R_cam"], dtype=float)
                camera_from_body_cv = camera_from_dodeca @ dodeca_from_body

                world_from_imu = world_from_imu_rotation(
                    imu_reading["quat"],
                    imu_quat_convention,
                )
                world_from_body = world_from_imu @ imu_to_body.T
                camera_from_body_imu = camera_from_world @ world_from_body @ imu_cv_bias

                draw_axis_set(
                    frame,
                    origin_cam,
                    camera_from_body_cv,
                    camera_matrix,
                    dist_coeffs,
                    axis_length_m,
                    "CV ",
                    dashed=False,
                )
                draw_axis_set(
                    frame,
                    origin_cam,
                    camera_from_body_imu,
                    camera_matrix,
                    dist_coeffs,
                    axis_length_m,
                    "IMU ",
                    dashed=True,
                )
                error_deg = rotation_error_deg(camera_from_body_cv, camera_from_body_imu)
                draw_status(
                    frame,
                    [
                        f"{merged_json_path.name}",
                        "CV axes: solid   IMU axes: dashed",
                        f"t={merged_time_s:.3f}s  cv_dt={cv_dt*1000:.1f}ms  imu_dt={imu_dt*1000:.1f}ms",
                        f"body-axis disagreement={error_deg:.2f} deg",
                    ],
                )
                rendered += 1
            else:
                draw_status(
                    frame,
                    [
                        f"{merged_json_path.name}",
                        "No paired CV/IMU sample for this frame",
                        f"t={merged_time_s:.3f}s",
                    ],
                )

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return {
        "frames": frame_index,
        "rendered_frames": rendered,
        "output": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render raw videos with overlaid stylus body axes inferred from CV and IMU."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("MERGED_JSON", "VIDEO"),
        required=True,
        help="Merged calibration pose JSON and the corresponding raw video. Repeat for multiple poses.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/body_axis_overlays",
        help="Directory for generated overlay videos.",
    )
    parser.add_argument(
        "--video-zero-times",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional per-pair raw-video elapsed times, in seconds, that correspond "
            "to merged JSON t=0. Defaults to 0 for every pair."
        ),
    )
    parser.add_argument(
        "--camera-world-calibration",
        default=str(DEFAULT_CAMERA_WORLD_PATH),
        help=f"Path to camera_from_world calibration JSON. Defaults to {DEFAULT_CAMERA_WORLD_PATH}.",
    )
    parser.add_argument(
        "--imu-calibration",
        default=str(DEFAULT_ALIGNMENT_PATH),
        help=f"Path to imu_to_body calibration JSON. Defaults to {DEFAULT_ALIGNMENT_PATH}.",
    )
    parser.add_argument(
        "--dodeca-body",
        default=str(DEFAULT_DODECA_BODY_PATH),
        help=f"Path to dodeca/body calibration JSON. Defaults to {DEFAULT_DODECA_BODY_PATH}.",
    )
    parser.add_argument(
        "--imu-cv-bias",
        default=str(DEFAULT_IMU_CV_BIAS_PATH),
        help=f"Path to IMU/CV body bias JSON. Missing path means identity. Defaults to {DEFAULT_IMU_CV_BIAS_PATH}.",
    )
    parser.add_argument(
        "--imu-quat-convention",
        choices=["world_from_imu", "imu_from_world"],
        default=None,
        help="Override IMU quaternion convention. Defaults to the camera-world calibration value.",
    )
    parser.add_argument(
        "--camera-matrix",
        default=str(DEFAULT_CAMERA_MATRIX_PATH),
        help=f"Camera matrix .npy path. Defaults to {DEFAULT_CAMERA_MATRIX_PATH}.",
    )
    parser.add_argument(
        "--dist-coeffs",
        default=str(DEFAULT_DIST_COEFFS_PATH),
        help=f"Camera distortion .npy path. Defaults to {DEFAULT_DIST_COEFFS_PATH}.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.05,
        help="Axis length to draw in meters.",
    )
    parser.add_argument(
        "--max-pair-dt-ms",
        type=float,
        default=50.0,
        help="Maximum nearest-sample time gap for drawing an overlay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = [(Path(json_path), Path(video_path)) for json_path, video_path in args.pair]
    if args.video_zero_times is None:
        video_zero_times = [0.0] * len(pairs)
    else:
        if len(args.video_zero_times) != len(pairs):
            raise ValueError("--video-zero-times must have one value per --pair")
        video_zero_times = args.video_zero_times

    camera_world_calibration = load_camera_world_calibration(args.camera_world_calibration)
    if camera_world_calibration is None:
        raise FileNotFoundError(f"camera_from_world calibration not found: {args.camera_world_calibration}")
    camera_from_world = np.asarray(camera_world_calibration["camera_from_world"], dtype=float)
    imu_quat_convention = args.imu_quat_convention or camera_world_calibration.get(
        "imu_quat_convention",
        "world_from_imu",
    )

    dodeca_body = load_dodeca_body_calibration(args.dodeca_body)
    dodeca_from_body = np.asarray(dodeca_body["dodeca_from_body"], dtype=float)

    imu_alignment = load_alignment_calibration(args.imu_calibration)
    if imu_alignment is None:
        raise FileNotFoundError(f"IMU calibration not found: {args.imu_calibration}")
    imu_to_body = np.asarray(imu_alignment["rotation_matrix"], dtype=float)

    imu_cv_bias = np.asarray(load_imu_cv_bias(args.imu_cv_bias)["rotation_matrix"], dtype=float)

    camera_matrix = np.load(args.camera_matrix)
    dist_coeffs = np.load(args.dist_coeffs)

    output_dir = Path(args.output_dir)
    for (merged_json_path, video_path), video_zero_time_s in zip(pairs, video_zero_times):
        output_path = output_dir / f"{merged_json_path.stem}_body_axes_overlay.mp4"
        summary = render_overlay(
            merged_json_path=merged_json_path,
            video_path=video_path,
            output_path=output_path,
            video_zero_time_s=video_zero_time_s,
            camera_from_world=camera_from_world,
            dodeca_from_body=dodeca_from_body,
            imu_to_body=imu_to_body,
            imu_cv_bias=imu_cv_bias,
            imu_quat_convention=imu_quat_convention,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            axis_length_m=args.axis_length,
            max_pair_dt_s=args.max_pair_dt_ms / 1000.0,
        )
        print(
            f"[Overlay] {merged_json_path.name}: wrote {summary['rendered_frames']}/"
            f"{summary['frames']} annotated frames to {summary['output']}"
        )


if __name__ == "__main__":
    main()
