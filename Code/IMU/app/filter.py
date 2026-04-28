from collections import deque
from typing import Deque, Tuple
import numpy as np
from numpy import typing as npt
from pyquaternion import Quaternion
from numba.typed.typedlist import List

from app.dimensions import IMU_OFFSET, STYLUS_LENGTH
from app.filter_core import (
    STATE_SIZE,
    FilterState,
    HistoryItem,
    SmoothingHistoryItem,
    ekf_predict,
    ekf_smooth,
    fuse_camera,
    fuse_imu,
    fuse_quaternion,
    i_acc,
    i_accbias,
    i_av,
    i_gyrobias,
    i_magbias,
    i_pos,
    i_quat,
    i_vel,
    DEFAULT_GRAVITY_VECTOR,
)

Mat = npt.NDArray[np.float64]

additive_noise = np.zeros(STATE_SIZE)
additive_noise[i_pos] = 1e-6
additive_noise[i_vel] = 4e-4
additive_noise[i_acc] = 1000
additive_noise[i_av] = 50
additive_noise[i_quat] = 1e-5
additive_noise[i_accbias] = 0.5e-4
additive_noise[i_gyrobias] = 1e-5
additive_noise[i_magbias] = 1e-3
Q = np.diag(additive_noise)

accel_noise = 2e-3
gyro_noise = 5e-4
imu_noise = np.diag([accel_noise] * 3 + [gyro_noise] * 3)
mag_noise = np.diag([25.0] * 3)
quat_noise = np.diag([2e-4] * 4)
# >>> MODIFICATION: Optimized noise for EKF performance <<<
# We make camera noise small so the filter trusts the high-quality offline CV
camera_noise_pos = 1e-6 # High trust in CV position
camera_noise_or = 1e-6  # High trust in orientation
camera_noise = np.diag([camera_noise_pos] * 3 + [camera_noise_or] * 4)

# We make process noise larger to allow the filter to follow motion accurately
additive_noise = np.zeros(STATE_SIZE)
# Position/Velocity noise allows following the camera tightly
additive_noise[i_pos] = 1e-3 
additive_noise[i_vel] = 1e-3 
# Accel/AV noise allows the IMU to fill gaps between CV frames smoothly
additive_noise[i_acc] = 10.0    
additive_noise[i_av] = 1.0     
additive_noise[i_quat] = 1e-6
additive_noise[i_accbias] = 1e-5
additive_noise[i_gyrobias] = 1e-6
additive_noise[i_magbias] = 1e-3
Q = np.diag(additive_noise)


def initial_state(position=None, orientation=None):
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[i_quat] = [1, 0, 0, 0]
    if position is not None:
        state[i_pos] = position.flatten()
    if orientation is not None:
        state[i_quat] = orientation.flatten()
    covdiag = np.ones(STATE_SIZE, dtype=np.float64) * 0.0001
    covdiag[i_accbias] = 1e-2
    covdiag[i_gyrobias] = 1e-4
    covdiag[i_magbias] = 1e-1
    statecov = np.diag(covdiag)
    return FilterState(state, statecov)


def get_tip_pose(state: Mat) -> Tuple[Mat, Mat]:
    """
    Return the pen tip position and orientation from the filter state.
    
    Note: The filter tracks the position that the vision system provides as
    'center_pos_cam' (dodecahedron center detected by CV). Due to IMU_OFFSET_BODY = [0,0,0]
    in dodeca_bridge.py, we currently assume the IMU is at the dodeca center, so the filter
    state position tracks the dodeca center. To get actual tip position, you would need to
    add CENTER_TO_TIP_BODY offset rotated by current orientation.
    """
    pos = state[i_pos]
    orientation = state[i_quat]
    return (pos, orientation)


def get_orientation_quat(orientation_mat_opencv: Mat):
    return Quaternion(matrix=orientation_mat_opencv).normalised


def nearest_quaternion(reference: Mat, new: Mat):
    """
    Find the sign for new that makes it as close to reference as possible.
    Changing the sign of a quaternion does not change its rotation, but affects
    the difference from the reference quaternion.
    """
    error1 = np.linalg.norm(reference - new)
    error2 = np.linalg.norm(reference + new)
    return (new, error1) if error1 < error2 else (-new, error2)


def blend_new_data(old: np.ndarray, new: np.ndarray, alpha: float):
    """Blends between old and new based on a power curve.
    Abruptly stopping smoothing can sometimes cause jumps, so we fade out the correction.
    This isn't mathematically optimal, but it looks a bit nicer.
    """
    N = old.shape[0]
    # This is just an arbitrary function that starts close to zero and ends at one.
    mix_factor = np.linspace(1 / 2 / N, 1, N)[:, np.newaxis] ** alpha
    return old * (1 - mix_factor) + new * mix_factor


def normalize_vector(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def as_float_vector(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    result = np.asarray(vector, dtype=np.float64)
    if np.linalg.norm(result) < 1e-9:
        return None
    return result


class DpointFilter:
    history: Deque[HistoryItem]

    def __init__(self, dt, smoothing_length: int, camera_delay: int, gravity_vector=None):
        self.history = deque()
        self.fs = initial_state()
        self.dt = dt
        self.smoothing_length = smoothing_length
        self.camera_delay = camera_delay
        if gravity_vector is None:
            gravity_vector = DEFAULT_GRAVITY_VECTOR
        self.gravity_vector = np.asarray(gravity_vector, dtype=np.float64)
        self.last_mag = None
        self.magnetic_field_vector = None
        self.last_camera_position = None
        self.last_camera_timestamp = None

    def update_imu(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray | None = None):
        predicted = ekf_predict(self.fs, self.dt, Q)
        raw_mag = as_float_vector(mag)
        if raw_mag is not None:
            self.last_mag = raw_mag

        magnetic_field_vector = (
            np.empty(0, dtype=np.float64)
            if self.magnetic_field_vector is None
            else self.magnetic_field_vector
        )
        mag_argument = (
            np.empty(0, dtype=np.float64)
            if raw_mag is None
            else raw_mag
        )
        mag_noise_argument = (
            np.empty((0, 0), dtype=np.float64)
            if raw_mag is None or self.magnetic_field_vector is None
            else mag_noise
        )

        self.fs = fuse_imu(
            predicted,
            accel,
            gyro,
            imu_noise,
            self.gravity_vector,
            mag_argument,
            mag_noise_argument,
            magnetic_field_vector,
        )
        self.history.append(
            HistoryItem(
                self.fs.state,
                self.fs.statecov,
                predicted.state,
                predicted.statecov,
                accel=accel,
                gyro=gyro,
                mag=raw_mag,
            )
        )
        max_history_len = self.smoothing_length + self.camera_delay + 1
        if len(self.history) > max_history_len:
            self.history.popleft()

    def update_imu_quaternion(self, quat: np.ndarray):
        predicted = ekf_predict(self.fs, self.dt, Q)
        orientation_quat = normalize_vector(quat)
        if orientation_quat is None:
            return

        aligned_quat, _ = nearest_quaternion(
            predicted.state[i_quat], orientation_quat
        )
        self.fs = fuse_quaternion(predicted, aligned_quat, quat_noise)
        self.history.append(
            HistoryItem(
                self.fs.state,
                self.fs.statecov,
                predicted.state,
                predicted.statecov,
                quat=aligned_quat,
            )
        )
        max_history_len = self.smoothing_length + self.camera_delay + 1
        if len(self.history) > max_history_len:
            self.history.popleft()

    def _update_velocity_from_camera(
        self, imu_pos: np.ndarray, timestamp: float | None
    ) -> None:
        if timestamp is None:
            return
        if self.last_camera_position is None or self.last_camera_timestamp is None:
            self.last_camera_position = imu_pos.copy()
            self.last_camera_timestamp = timestamp
            return

        dt = timestamp - self.last_camera_timestamp
        if dt <= 1e-6:
            return

        cv_velocity = (imu_pos - self.last_camera_position) / dt
        blend = 0.35
        self.fs.state[i_vel] = (1 - blend) * self.fs.state[i_vel] + blend * cv_velocity
        self.last_camera_position = imu_pos.copy()
        self.last_camera_timestamp = timestamp

    def observe_camera_pose(
        self, imu_pos: np.ndarray, timestamp: float | None
    ) -> None:
        self._update_velocity_from_camera(imu_pos, timestamp)

    def update_camera(
        self, imu_pos: np.ndarray, orientation_mat: np.ndarray, timestamp: float | None = None
    ) -> list[np.ndarray]:
        # >>> MODIFICATION: Support CV-only mode <<<
        # If no history (no IMU updates), we just update the state directly
        or_quat = get_orientation_quat(orientation_mat)
        
        if len(self.history) == 0:
            # Direct state update if no IMU data is present
            self.fs = initial_state(imu_pos, or_quat.elements)
            self.last_camera_position = imu_pos.copy()
            self.last_camera_timestamp = timestamp
            # We still need to predict to update the covariance
            self.fs = ekf_predict(self.fs, self.dt, Q)
            return [get_tip_pose(self.fs.state)[0]]

        # Rollback and store recent IMU measurements
        replay: Deque[HistoryItem] = deque()
        for _ in range(min(len(self.history) - 1, self.camera_delay)):
            replay.appendleft(self.history.pop())

        if self.last_mag is not None:
            magnetic_world = as_float_vector(orientation_mat @ self.last_mag)
            if magnetic_world is not None:
                if self.magnetic_field_vector is None:
                    self.magnetic_field_vector = magnetic_world
                else:
                    blend = 0.05
                    self.magnetic_field_vector = as_float_vector(
                        (1 - blend) * self.magnetic_field_vector + blend * magnetic_world
                    )

        # Fuse camera in its rightful place
        h = self.history[-1]
        fs = FilterState(h.updated_state, h.updated_statecov)
        or_quat_smoothed, or_error = nearest_quaternion(
            fs.state[i_quat], or_quat.elements
        )
        pos_error = np.linalg.norm(imu_pos - fs.state[i_pos])
        
        # In CV-only mode, we might want to be more lenient with resets
        if pos_error > 0.5 or or_error > 1.0: 
            print(f"Resetting state, errors: pos={pos_error:.4f}m, or={or_error:.4f}rad")
            self.fs = initial_state(imu_pos, or_quat_smoothed)
            self.last_camera_position = imu_pos.copy()
            self.last_camera_timestamp = timestamp
            self.history = deque()
            return [get_tip_pose(self.fs.state)[0]]
            
        self.fs = fuse_camera(fs, imu_pos, or_quat_smoothed, camera_noise)
        self._update_velocity_from_camera(imu_pos, timestamp)
        previous = self.history.pop()  # Replace last item
        self.history.append(
            HistoryItem(
                self.fs.state,
                self.fs.statecov,
                previous.predicted_state,
                previous.predicted_statecov,
            )
        )

        # Apply smoothing
        smoothed_estimates = ekf_smooth(
            List(
                [
                    SmoothingHistoryItem(
                        h.updated_state,
                        h.updated_statecov,
                        h.predicted_state,
                        h.predicted_statecov,
                    )
                    for h in self.history
                ]
            ),
            self.dt,
        )

        # Replay the IMU measurements
        predicted_estimates = []
        for item in replay:
            if item.quat is not None:
                self.update_imu_quaternion(item.quat)
            elif item.accel is not None and item.gyro is not None:
                self.update_imu(item.accel, item.gyro, item.mag)
            predicted_estimates.append(self.fs.state)
            
        return [
            get_tip_pose(state)[0] for state in smoothed_estimates + predicted_estimates
        ]

    def get_tip_pose(self) -> Tuple[Mat, Mat]:
        return get_tip_pose(self.fs.state)
