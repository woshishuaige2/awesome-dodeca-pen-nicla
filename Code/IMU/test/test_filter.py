from approvaltests.approvals import verify
import numpy as np

from app.filter import initial_state
from app.filter_core import (
    FilterState,
    STATE_SIZE,
    camera_measurement,
    imu_measurement,
    i_acc,
    i_av,
    i_pos,
    i_quat,
    i_vel,
    state_transition,
    state_transition_jacobian,
)


def initial_state_for_tests():
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[i_quat] = [1, 0, 0, 0]
    statecov = np.eye(STATE_SIZE) * 0.01
    return FilterState(state, statecov)


def test_initial_state():
    fs = initial_state()
    assert len(fs.state) == STATE_SIZE
    assert fs.statecov.shape == (STATE_SIZE, STATE_SIZE)


def test_state_transition_jacobian():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    jacobian = state_transition_jacobian(fs.state)
    verify(jacobian)


def test_state_transition():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    st = state_transition(fs.state)
    verify(st)


def test_imu_measurement():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    measurement = imu_measurement(fs.state)
    verify(measurement)


def test_camera_measurement():
    fs = initial_state_for_tests()
    fs.state[i_pos] = [1, 2, 3]
    fs.state[i_quat] = [4, 5, 6, 7]
    measurement = camera_measurement(fs.state)
    verify(measurement)
