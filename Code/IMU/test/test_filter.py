from approvaltests.approvals import verify
import numpy as np

from app.filter import initial_state
from app.filter_core import (
    FilterState,
    STATE_SIZE,
    camera_measurement,
    i_pos,
    i_quat,
    i_vel,
    quaternion_measurement,
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
    jacobian = state_transition_jacobian(fs.state)
    verify(jacobian)
    assert jacobian[i_pos[0], i_vel[0]] == 1
    assert jacobian[i_pos[1], i_vel[1]] == 1
    assert jacobian[i_pos[2], i_vel[2]] == 1


def test_state_transition():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    st = state_transition(fs.state)
    verify(st)
    np.testing.assert_allclose(st[i_pos], [1, 2, 3])
    np.testing.assert_allclose(st[i_vel], [0, 0, 0])


def test_quaternion_measurement():
    fs = initial_state_for_tests()
    measurement = quaternion_measurement(fs.state)
    verify(measurement)


def test_camera_measurement():
    fs = initial_state_for_tests()
    fs.state[i_pos] = [1, 2, 3]
    fs.state[i_quat] = [4, 5, 6, 7]
    measurement = camera_measurement(fs.state)
    verify(measurement)
