import argparse
from pathlib import Path

import numpy as np

from app.camera_world_calibration import (
    DEFAULT_CAMERA_WORLD_PATH,
    DEFAULT_DODECA_BODY_PATH,
    estimate_camera_world_from_recordings,
    save_camera_world_calibration,
)
from app.imu_alignment import DEFAULT_ALIGNMENT_PATH


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a fixed camera-from-world rotation from stationary merged "
            "CV/IMU recordings."
        )
    )
    parser.add_argument(
        "recordings",
        nargs="+",
        help="Merged recording files (my_data.json style), preferably stationary poses.",
    )
    parser.add_argument(
        "--imu-to-body",
        default=str(DEFAULT_ALIGNMENT_PATH),
        help="Path to the fixed IMU-to-body calibration JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CAMERA_WORLD_PATH),
        help="Output camera-from-world calibration JSON path.",
    )
    parser.add_argument(
        "--dodeca-body",
        default=str(DEFAULT_DODECA_BODY_PATH),
        help="Path to the dodeca-to-pen-body calibration JSON.",
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
        help=(
            "Interpretation of the streamed IMU quaternion matrix. "
            "'imu_from_world' uses the transpose before alignment."
        ),
    )
    parser.add_argument(
        "--diagnose-conventions",
        action="store_true",
        help="Evaluate both IMU quaternion conventions without saving a calibration.",
    )
    args = parser.parse_args()

    if args.diagnose_conventions:
        print("=" * 60)
        print("CAMERA-FROM-WORLD CONVENTION DIAGNOSIS")
        print("=" * 60)
        for convention in ["world_from_imu", "imu_from_world"]:
            result = estimate_camera_world_from_recordings(
                args.recordings,
                imu_to_body_path=args.imu_to_body,
                dodeca_body_path=args.dodeca_body,
                max_pair_dt_s=args.max_pair_dt_ms / 1000.0,
                imu_quat_convention=convention,
            )
            print(
                f"{convention}: "
                f"mean={result['mean_error_deg']:.3f} deg, "
                f"median={result['median_error_deg']:.3f} deg, "
                f"p95={result['p95_error_deg']:.3f} deg, "
                f"max={result['max_error_deg']:.3f} deg"
            )
            for recording in result["recordings"]:
                print(
                    f"  - {recording['path']}: "
                    f"p95={recording['camera_from_world_p95_error_deg']:.3f} deg, "
                    f"pairs={recording['paired_count']}"
                )
        return

    result = estimate_camera_world_from_recordings(
        args.recordings,
        imu_to_body_path=args.imu_to_body,
        dodeca_body_path=args.dodeca_body,
        max_pair_dt_s=args.max_pair_dt_ms / 1000.0,
        imu_quat_convention=args.imu_quat_convention,
    )
    save_camera_world_calibration(result, args.output)

    print("=" * 60)
    print("CAMERA-FROM-WORLD CALIBRATION")
    print("=" * 60)
    print(f"Method: {result['method']}")
    print(f"IMU quaternion convention: {result['imu_quat_convention']}")
    print(f"Dodeca/body calibration: {result['dodeca_body_path']}")
    print(f"Samples used: {result['sample_count']}")
    print(
        "Residual spread: "
        f"mean={result['mean_error_deg']:.3f} deg, "
        f"median={result['median_error_deg']:.3f} deg, "
        f"p95={result['p95_error_deg']:.3f} deg, "
        f"max={result['max_error_deg']:.3f} deg"
    )
    print("camera_from_world:")
    print(np.array2string(result["camera_from_world"], precision=6, suppress_small=True))
    print(f"Saved to: {Path(args.output).resolve()}")

    print("\nRecordings:")
    for recording in result["recordings"]:
        print(
            f"- {recording['path']}: pairs={recording['paired_count']}, "
            f"mean dt={recording['mean_pair_dt_ms']:.1f} ms, "
            f"p95 spread={recording['camera_from_world_p95_error_deg']:.3f} deg"
        )


if __name__ == "__main__":
    main()
