from typing import NamedTuple, Optional

import numpy as np
from numba import njit
from numpy import typing as npt

# This file is separate from filter.py, so that it doesn't need to be re-compiled so often.

Mat = npt.NDArray[np.float64]

i_quat = np.array([0, 1, 2, 3])
i_pos = np.array([4, 5, 6])
i_vel = np.array([7, 8, 9])

STATE_SIZE = 10
DEFAULT_GRAVITY_VECTOR = np.array([0, 0, 9.81], dtype=np.float64)
GRAVITY_VECTOR = DEFAULT_GRAVITY_VECTOR.copy()


class FilterState(NamedTuple):
    state: Mat
    statecov: Mat


class SmoothingHistoryItem(NamedTuple):
    updated_state: Mat
    updated_statecov: Mat
    predicted_state: Mat
    predicted_statecov: Mat


class HistoryItem(NamedTuple):
    updated_state: Mat
    updated_statecov: Mat
    predicted_state: Mat
    predicted_statecov: Mat
    accel: Optional[Mat] = None
    gyro: Optional[Mat] = None
    mag: Optional[Mat] = None
    quat: Optional[Mat] = None


@njit(cache=True)
def state_transition(state: Mat = np.array([])):
    statedot = np.zeros_like(state)
    statedot[i_pos] = state[i_vel]
    return statedot


@njit(cache=True)
def state_transition_jacobian(state: Mat):
    dfdx = np.zeros((STATE_SIZE, STATE_SIZE), dtype=state.dtype)
    dfdx[i_pos[0], i_vel[0]] = 1.0
    dfdx[i_pos[1], i_vel[1]] = 1.0
    dfdx[i_pos[2], i_vel[2]] = 1.0
    return dfdx


@njit(cache=True)
def camera_measurement(state: Mat):
    m_camera = state[i_pos]
    mj_camera = np.zeros((3, len(state)))
    mj_camera[0:3, i_pos] = np.eye(3)
    return (m_camera, mj_camera)


@njit(cache=True)
def quaternion_measurement(state: Mat):
    m_quat = state[i_quat]
    mj_quat = np.zeros((4, len(state)))
    mj_quat[:, i_quat] = np.eye(4)
    return (m_quat, mj_quat)


@njit(cache=True)
def repair_quaternion(q: Mat):
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    return q / norm


@njit(cache=True)
def predict_cov_derivative(P: Mat, dfdx: Mat, Q: Mat):
    pDot = dfdx @ P + P @ (dfdx.T) + Q
    pDot = 0.5 * (pDot + pDot.T)
    return pDot


@njit(cache=True)
def euler_integrate(x: Mat, xdot: Mat, dt: float):
    return x + xdot * dt


@njit(cache=True)
def ekf_predict(fs: FilterState, dt: float, Q: np.ndarray):
    xdot = state_transition(fs.state)
    dfdx = state_transition_jacobian(fs.state)
    P = fs.statecov
    Pdot = predict_cov_derivative(P, dfdx, Q)
    state = euler_integrate(fs.state, xdot, dt)
    state[i_quat] = repair_quaternion(state[i_quat])
    statecov = euler_integrate(P, Pdot, dt)
    statecov = 0.5 * (statecov + statecov.T)
    return FilterState(state, statecov)


@njit(cache=True)
def ekf_correct(x: Mat, P: Mat, h: Mat, H: Mat, z: Mat, R: Mat):
    S = H @ P @ H.T + R
    W = P @ H.T @ np.linalg.inv(S)
    x2 = x + W @ (z - h)
    P2 = P - W @ H @ P
    P2 = 0.5 * (P2 + P2.T)
    return x2, P2


def fuse_camera(
    fs: FilterState,
    imu_pos: np.ndarray,
    orientation_quat: np.ndarray,
    meas_noise: np.ndarray,
):
    """
    Fuses camera data using the decoupled approach.
    Only position is used in the correction step; orientation_quat is accepted
    for call-site compatibility but intentionally ignored.
    """
    h, H = camera_measurement(fs.state)
    z = imu_pos.flatten()
    R = meas_noise[0:3, 0:3] if meas_noise.shape[0] == 7 else meas_noise
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, R)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


def fuse_quaternion(
    fs: FilterState,
    orientation_quat: np.ndarray,
    quat_noise: np.ndarray,
):
    h, H = quaternion_measurement(fs.state)
    z = orientation_quat.flatten()
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, quat_noise)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


@njit(cache=True)
def ekf_smooth(history: list[HistoryItem], dt: float):
    smoothed_state = [h.updated_state for h in history]

    for i in range(len(history) - 2, -1, -1):
        h = history[i]
        F = np.eye(STATE_SIZE) + state_transition_jacobian(h.updated_state) * dt
        A = h.updated_statecov @ F.T @ np.linalg.inv(history[i + 1].predicted_statecov)
        correction = A @ (smoothed_state[i + 1] - history[i + 1].predicted_state)
        smoothed_state[i] = h.updated_state + correction
        smoothed_state[i][i_quat] = repair_quaternion(smoothed_state[i][i_quat])
    return smoothed_state
