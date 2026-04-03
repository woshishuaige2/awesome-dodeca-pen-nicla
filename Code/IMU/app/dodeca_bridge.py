# dodeca_bridge.py  — unified, guarded bridge for Dodecaball → IMU EKF
#
# COORDINATE SYSTEM DOCUMENTATION:
# ================================
# 
# Body Frame: Fixed to the dodecahedron, origin at its geometric center
#   - X, Y, Z axes aligned with dodecahedron geometry
#   - Rotates with the pen as it moves
#
# Camera Frame: Fixed to the camera sensor
#   - Z-axis points into the scene (depth)
#   - X-axis points right, Y-axis points down (standard computer vision convention)
#
# Key Points in Body Frame:
#   - DODECA CENTER: [0, 0, 0] (origin)
#   - PEN TIP: CENTER_TO_TIP_BODY = [0, 137.5mm, -82.1mm]
#   - IMU LOCATION: IMU_OFFSET_BODY = [0, 0, 0] (currently simplified as center)
#
# What Computer Vision Detects:
#   - CV tracks ArUco markers on the dodecahedron surface
#   - Output: position and orientation of DODECA CENTER in camera frame
#   - The variable 'center_pos_cam' represents this detected center position
#
# Transformations:
#   - center_pos_cam = what CV outputs directly (dodeca center in camera frame)
#   - tip_pos_cam = center_pos_cam + R_cam @ CENTER_TO_TIP_BODY
#   - imu_pos_cam = center_pos_cam + R_cam @ IMU_OFFSET_BODY
#                 = center_pos_cam (since IMU_OFFSET_BODY = [0,0,0])
#

from pathlib import Path
import sys
import time
import numpy as np

# --- Geometry configuration (kept local for simplicity) ---
# Vector from Dodecaball CENTER → PEN TIP in the body frame (mm→m)
CENTER_TO_TIP_BODY = np.array([0.0, 137.52252061, -82.07403558]) * 1e-3
# Vector from Dodecaball CENTER → IMU in the body frame (meters). 
# Currently [0,0,0] assumes IMU is at dodeca center (simplification)
IMU_OFFSET_BODY = np.array([0.0, 0.0, 0.0])

# --- Make Computer_vision importable (once) ---
repo_root = Path(__file__).resolve().parents[2]      # .../Code
cv_dir = repo_root / "Computer_vision"
if str(cv_dir) not in sys.path:
    sys.path.append(str(cv_dir))

# Guarded import: this should NOT auto-run the CV window if run.py has an __main__ guard
try:
    import run as dcv_run   # Code/Computer_vision/run.py
except Exception:
    dcv_run = None

# --- Quaternion helper with sign continuity ---
_prev_q = None
def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(R)  # [w, x, y, z]
    global _prev_q
    if _prev_q is not None and float(np.dot(q, _prev_q)) < 0.0:
        q = -q
    _prev_q = q
    return q

# --- Vision access ---
def get_vision_reading():
    """
    Returns (t_cam, R_cam, ts) from Dodecaball vision, or None if not available.
    t_cam: (3,) in meters; R_cam: (3,3)
    Requires that Computer_vision/run.py defines a module-level `object_pose`
    updated by its loop, but does NOT auto-run on import.
    """
    if dcv_run is None or not hasattr(dcv_run, "object_pose"):
        return None
    obj = dcv_run.object_pose
    if obj is None:
        return None
    t = obj[0, :3].astype(float)
    R = obj[0, 3:].reshape(3, 3).astype(float)
    return t, R, time.time()

# --- EKF measurement packaging ---
def make_ekf_measurements(center_to_tip_body: np.ndarray = CENTER_TO_TIP_BODY,
                          imu_offset_body: np.ndarray = IMU_OFFSET_BODY):
    """
    Map camera pose (dodeca center) → EKF-friendly measurements.
    CV tracks the dodecahedron CENTER via ArUco markers.
    Returns dict with:
      - center_pos_cam: (3,) - dodecahedron center position in camera frame
      - tip_pos_cam: (3,) - pen tip position in camera frame
      - R_cam: (3,3) - rotation matrix
      - q_cam: (4,) [w,x,y,z] - quaternion
      - timestamp: float
      - quality: float
    or None if no vision reading available.
    """
    out = get_vision_reading()
    if out is None:
        return None 
    t_cam, R_cam, ts = out
    # CV detects dodecahedron center position (in mm, convert to m)
    center_pos_cam = t_cam * 0.001

    # Transform offsets from body frame to camera frame
    # Tip = Center + R × (Center→Tip offset)
    tip_pos_cam = center_pos_cam + R_cam @ center_to_tip_body
    # IMU = Center + R × (IMU offset from center)
    # Since IMU_OFFSET_BODY = [0,0,0], this currently equals center_pos_cam

    q_cam = _rotmat_to_quat(R_cam)
    return {
        "center_pos_cam": center_pos_cam,  # What CV actually detects
        "tip_pos_cam": tip_pos_cam,
        "R_cam": R_cam,
        "q_cam": q_cam,
        "timestamp": ts,
        "quality": 1.0,
    }

# --- Pen tip position publishing ---
def publish_pen_tip_positions(raw_pos: np.ndarray = None, smoothed_pos: np.ndarray = None):
    """
    Send pen tip positions back to the CV window for visualization.
    """
    if dcv_run is not None and hasattr(dcv_run, "_publish_pen_tip_positions"):
        dcv_run._publish_pen_tip_positions(raw_pos, smoothed_pos)

def is_cv_shutdown_requested() -> bool:
    """
    Check if the CV window has requested shutdown.
    """
    if dcv_run is not None and hasattr(dcv_run, "_is_shutdown_requested"):
        return dcv_run._is_shutdown_requested()
    return False

# Expose geometry constants if callers import them
__all__ = [
    "CENTER_TO_TIP_BODY",
    "IMU_OFFSET_BODY",
    "get_vision_reading",
    "make_ekf_measurements",
    "publish_pen_tip_positions",
    "is_cv_shutdown_requested",
]
