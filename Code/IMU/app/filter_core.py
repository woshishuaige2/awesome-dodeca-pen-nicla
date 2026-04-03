from typing import NamedTuple, Optional

import numpy as np
from numba import njit
from numpy import typing as npt
from pyquaternion import Quaternion

# This file is separate from filter.py, so that it doesn't need to be re-compiled so often.

Mat = npt.NDArray[np.float64]

i_quat = np.array([0, 1, 2, 3])
i_av = np.array([4, 5, 6])
i_pos = np.array([7, 8, 9])
i_vel = np.array([10, 11, 12])
i_acc = np.array([13, 14, 15])
i_accbias = np.array([16, 17, 18])
i_gyrobias = np.array([19, 20, 21])

STATE_SIZE = 22
DEFAULT_GRAVITY_VECTOR = np.array([0, 0, 9.81], dtype=np.float64)
# GRAVITY_VECTOR points UP in the world frame because the accelerometer
# measures the normal force pushing the stylus UP against gravity.
# This matches the CV world frame where Z+ is UP.
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


@njit(cache=True)
def state_transition(state: Mat = np.array([])):
    av = state[i_av]
    quat = state[i_quat]
    q0, q1, q2, q3 = quat
    acc = state[i_acc]
    vel = state[i_vel]

    qdot = np.array(
        [
            np.dot(av, np.array([-q1, -q2, -q3])) / 2,
            np.dot(av, np.array([q0, -q3, q2])) / 2,
            np.dot(av, np.array([q3, q0, -q1])) / 2,
            np.dot(av, np.array([-q2, q1, q0])) / 2,
        ]
    )
    pdot = vel
    vdot = acc
    statedot = np.zeros_like(state)
    statedot[i_quat] = qdot
    statedot[i_av] = 0
    statedot[i_pos] = pdot
    statedot[i_vel] = vdot
    statedot[i_acc] = 0
    statedot[i_accbias] = 0
    statedot[i_gyrobias] = 0
    return statedot


@njit(cache=True)
def state_transition_jacobian(state: Mat):
    av = state[i_av]
    avx, avy, avz = av
    quat = state[i_quat]
    q0, q1, q2, q3 = quat

    N = len(state)

    # Orientation
    dorientfuncdx = np.zeros((4, N), dtype=state.dtype)
    dorientfuncdx[0, i_quat] = [0, -avx / 2, -avy / 2, -avz / 2]
    dorientfuncdx[0, i_av] = [-q1 / 2, -q2 / 2, -q3 / 2]
    dorientfuncdx[1, i_quat] = [avx / 2, 0, avz / 2, -avy / 2]
    dorientfuncdx[1, i_av] = [q0 / 2, -q3 / 2, q2 / 2]
    dorientfuncdx[2, i_quat] = [avy / 2, -avz / 2, 0, avx / 2]
    dorientfuncdx[2, i_av] = [q3 / 2, q0 / 2, -q1 / 2]
    dorientfuncdx[3, i_quat] = [avz / 2, avy / 2, -avx / 2, 0]
    dorientfuncdx[3, i_av] = [-q2 / 2, q1 / 2, q0 / 2]

    # Position
    dposfuncdx = np.zeros((3, N), dtype=state.dtype)
    dposfuncdx[0, i_vel] = [1, 0, 0]
    dposfuncdx[1, i_vel] = [0, 1, 0]
    dposfuncdx[2, i_vel] = [0, 0, 1]

    # Velocity
    dvelfuncdx = np.zeros((3, N), dtype=state.dtype)
    dvelfuncdx[0, i_acc] = [1, 0, 0]
    dvelfuncdx[1, i_acc] = [0, 1, 0]
    dvelfuncdx[2, i_acc] = [0, 0, 1]

    dfdx = np.zeros((STATE_SIZE, STATE_SIZE), dtype=state.dtype)
    dfdx[i_quat, :] = dorientfuncdx
    dfdx[i_pos, :] = dposfuncdx
    dfdx[i_vel, :] = dvelfuncdx
    return dfdx


@njit(cache=True)
def imu_measurement(state: Mat, gravity_vector: Mat = DEFAULT_GRAVITY_VECTOR):
    av = state[i_av]
    acc = state[i_acc]
    quat = state[i_quat]
    q0, q1, q2, q3 = quat
    accbias = state[i_accbias]

    m_gyro = av + state[i_gyrobias]
    mj_gyro = np.zeros((3, len(state)))
    mj_gyro[:, i_av] = np.eye(3)
    mj_gyro[:, i_gyrobias] = np.eye(3)

    grav = gravity_vector

    m_accel = np.array(
        [
            accbias[0]
            - (acc[0] - grav[0]) * (2 * q0**2 + 2 * q1**2 - 1)
            - (acc[1] - grav[1]) * (2 * q0 * q3 + 2 * q1 * q2)
            + (acc[2] - grav[2]) * (2 * q0 * q2 - 2 * q1 * q3),
            accbias[1]
            - (acc[1] - grav[1]) * (2 * q0**2 + 2 * q2**2 - 1)
            + (acc[0] - grav[0]) * (2 * q0 * q3 - 2 * q1 * q2)
            - (acc[2] - grav[2]) * (2 * q0 * q1 + 2 * q2 * q3),
            accbias[2]
            - (acc[2] - grav[2]) * (2 * q0**2 + 2 * q3**2 - 1)
            - (acc[0] - grav[0]) * (2 * q0 * q2 + 2 * q1 * q3)
            + (acc[1] - grav[1]) * (2 * q0 * q1 - 2 * q2 * q3),
        ]
    )

    mj_accel = np.zeros((3, len(state)))
    mj_accel[0, i_quat] = [
        2 * q2 * (acc[2] - grav[2])
        - 2 * q3 * (acc[1] - grav[1])
        - 4 * q0 * (acc[0] - grav[0]),
        -4 * q1 * (acc[0] - grav[0])
        - 2 * q2 * (acc[1] - grav[1])
        - 2 * q3 * (acc[2] - grav[2]),
        2 * q0 * (acc[2] - grav[2]) - 2 * q1 * (acc[1] - grav[1]),
        -2 * q0 * (acc[1] - grav[1]) - 2 * q1 * (acc[2] - grav[2]),
    ]
    mj_accel[1, i_quat] = [
        2 * q3 * (acc[0] - grav[0])
        - 4 * q0 * (acc[1] - grav[1])
        - 2 * q1 * (acc[2] - grav[2]),
        -2 * q2 * (acc[0] - grav[0]) - 2 * q0 * (acc[2] - grav[2]),
        -2 * q1 * (acc[0] - grav[0])
        - 4 * q2 * (acc[1] - grav[1])
        - 2 * q3 * (acc[2] - grav[2]),
        2 * q0 * (acc[0] - grav[0]) - 2 * q2 * (acc[2] - grav[2]),
    ]
    mj_accel[2, i_quat] = [
        2 * q1 * (acc[1] - grav[1])
        - 2 * q2 * (acc[0] - grav[0])
        - 4 * q0 * (acc[2] - grav[2]),
        2 * q0 * (acc[1] - grav[1]) - 2 * q3 * (acc[0] - grav[0]),
        -2 * q0 * (acc[0] - grav[0]) - 2 * q3 * (acc[1] - grav[1]),
        -2 * q1 * (acc[0] - grav[0])
        - 2 * q2 * (acc[1] - grav[1])
        - 4 * q3 * (acc[2] - grav[2]),
    ]
    mj_accel[0, i_acc] = [
        1 - 2 * q1**2 - 2 * q0**2,
        -2 * q0 * q3 - 2 * q1 * q2,
        2 * q0 * q2 - 2 * q1 * q3,
    ]
    mj_accel[1, i_acc] = [
        2 * q0 * q3 - 2 * q1 * q2,
        1 - 2 * q2**2 - 2 * q0**2,
        -2 * q0 * q1 - 2 * q2 * q3,
    ]
    mj_accel[2, i_acc] = [
        -2 * q0 * q2 - 2 * q1 * q3,
        2 * q0 * q1 - 2 * q2 * q3,
        1 - 2 * q3**2 - 2 * q0**2,
    ]
    mj_accel[:, i_accbias] = np.eye(3)

    m_combined = np.concatenate((m_accel, m_gyro))
    mj_combined = np.vstack((mj_accel, mj_gyro))
    return (m_combined, mj_combined)


@njit(cache=True)
def camera_measurement(state: Mat):
    """
    Decoupled camera measurement model.
    Only uses position for the update to avoid orientation jitter propagation.
    """
    pos = state[i_pos]
    m_camera = pos
    
    mj_camera = np.zeros((3, len(state)))
    mj_camera[0:3, i_pos] = np.eye(3)
    return (m_camera, mj_camera)


@njit(cache=True)
def repair_quaternion(q: Mat):
    # Note: we can't change the sign here, because it will affect the smoothing
    return q / np.linalg.norm(q)


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
    # statecov = P + dt * (dfdx @ P + P @ dfdx.T + Q) + dt**2 * (dfdx @ P @ dfdx.T)
    # statecov = 0.5 * (statecov + statecov.T)

    return FilterState(state, statecov)


@njit(cache=True)
def ekf_correct(x: Mat, P: Mat, h: Mat, H: Mat, z: Mat, R: Mat):
    S = H @ P @ H.T + R  # innovation covariance
    W = P @ H.T @ np.linalg.inv(S)
    x2 = x + W @ (z - h)
    P2 = P - W @ H @ P
    return x2, P2


@njit(cache=True)
def fuse_imu(
    fs: FilterState,
    accel: np.ndarray,
    gyro: np.ndarray,
    meas_noise: np.ndarray,
    gravity_vector: Mat = DEFAULT_GRAVITY_VECTOR,
):
    h, H = imu_measurement(fs.state, gravity_vector)
    # Correct mapping for Dodeca-pen to Camera Frame:
    # Camera: X right, Y down, Z forward
    # Pen: X forward, Y right, Z up
    # To match Camera Frame:
    # Correct mapping for Dodeca-pen to Camera Frame:
    # Camera Frame: X+ right, Y+ down, Z+ forward
    # Pen Frame: X+ forward, Y+ right, Z+ up
    # Correct mapping:
    # Pen X+ -> Camera X+
    # Pen Y+ -> Camera Y+
    # Pen Z+ -> Camera Z+
    # Assuming the Pen and Camera frames are already aligned or the rotation
    # is handled by the initial orientation.
    accel2 = accel
    gyro2 = gyro
    z = np.concatenate((accel2, gyro2))  # actual measurement
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, meas_noise)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


def fuse_camera(
    fs: FilterState,
    imu_pos: np.ndarray,
    orientation_quat: np.ndarray, # Kept in signature for compatibility, but unused for correction
    meas_noise: np.ndarray,
):
    """
    Fuses camera data using a decoupled approach.
    Only the position (imu_pos) is used to update the state.
    The orientation_quat is ignored in the correction step to prevent jitter propagation.
    """
    h, H = camera_measurement(fs.state)
    z = imu_pos.flatten() # Only use position
       # Slicing the noise matrix to match the 3D position measurement
    # If meas_noise is 7x7, we take the top-left 3x3
    if meas_noise.shape[0] == 7:
        R = meas_noise[0:3, 0:3]
    else:
        R = meas_noise
        
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, R)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


@njit(cache=True)
def ekf_smooth(history: list[HistoryItem], dt: float):
    # We only need the last item to be set, but we do all of them to make numba happy
    smoothed_state = [h.updated_state for h in history]

    for i in range(len(history) - 2, -1, -1):
        h = history[i]
        F = np.eye(STATE_SIZE) + state_transition_jacobian(h.updated_state) * dt
        A = h.updated_statecov @ F.T @ np.linalg.inv(history[i + 1].predicted_statecov)
        correction = A @ (smoothed_state[i + 1] - history[i + 1].predicted_state)
        smoothed_state[i] = h.updated_state + correction
    return smoothed_state
