import argparse
from pathlib import Path

import numpy as np

from app.imu_alignment import (
    DEFAULT_ALIGNMENT_PATH,
    estimate_alignment_from_recordings,
    save_alignment_calibration,
)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a fixed IMU-to-body rotation from stationary merged recordings."
    )
    parser.add_argument(
        "recordings",
        nargs="+",
        help="Merged recording files (my_data.json style), each captured while stationary in a distinct orientation.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ALIGNMENT_PATH),
        help="Output calibration JSON path.",
    )
    args = parser.parse_args()

    result = estimate_alignment_from_recordings(args.recordings)
    save_alignment_calibration(result, args.output)

    print("=" * 60)
    print("IMU ALIGNMENT CALIBRATION")
    print("=" * 60)
    print(f"Method: {result['method']}")
    print(f"Observable axes: {result['observable_axes']}/3")
    print(f"Mean residual: {result['mean_residual']:.6f}")
    print(f"Max residual: {result['max_residual']:.6f}")
    if result["method"] == "joint_quaternion_cv_fit":
        print("(Quaternion residuals are rotation-vector magnitudes in radians.)")
    print("Rotation matrix:")
    print(np.array2string(result["rotation_matrix"], precision=6, suppress_small=True))
    print("Gravity in camera frame:")
    print(np.array2string(result["gravity_camera"], precision=6, suppress_small=True))
    print(f"Saved to: {Path(args.output).resolve()}")

    if result["observable_axes"] < 3:
        print(
            "[Warning] Only one stationary orientation was provided. "
            "This constrains gravity alignment but not the full 3D IMU yaw."
        )


if __name__ == "__main__":
    main()
