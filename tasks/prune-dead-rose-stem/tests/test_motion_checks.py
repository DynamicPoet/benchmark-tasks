import numpy as np
import pytest

from motion_checks import points_at_rest,finger_pad_points


def stationary(samples=21):
    return [np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.4]) for _ in range(samples)]


@pytest.mark.parametrize("samples", [0, 1, 20])
def test_requires_a_full_second(samples):
    assert not points_at_rest(stationary(samples), 0.005, 20)


def test_stationary_cutting_passes():
    assert points_at_rest(stationary(), 0.005, 20)


@pytest.mark.parametrize("endpoint", [0, 1])
def test_motion_and_return_within_window_fails(endpoint):
    history = stationary()
    history[10][endpoint * 3] += 0.02
    assert not points_at_rest(history, 0.005, 20)


def test_diagonal_motion_uses_spatial_distance():
    history = [np.full(6, 0.004 * i / 20) for i in range(21)]
    assert not points_at_rest(history, 0.005, 20)


def test_motion_on_opposite_sides_of_start_fails():
    history = stationary()
    history[5][0] += 0.004
    history[15][0] -= 0.004
    assert not points_at_rest(history, 0.005, 20)


@pytest.mark.parametrize("distance, expected", [(0.0049, True), (0.005, False), (0.0051, False)])
def test_threshold_is_strict(distance, expected):
    history = [np.zeros(6) for _ in range(21)]
    history[10][0] = distance
    assert points_at_rest(history, 0.005, 20) is expected


def test_small_motion_passes():
    history = stationary()
    history[5][:3] += [0.001, 0.001, 0.001]
    history[15][3:] += [0.001, 0.001, 0.001]
    assert points_at_rest(history, 0.005, 20)


def test_motion_before_window_does_not_count():
    history = [np.ones(6)] + stationary()
    assert points_at_rest(history, 0.005, 20)


def test_non_finite_position_fails():
    history = stationary()
    history[10][0] = np.nan
    assert not points_at_rest(history, 0.005, 20)


@pytest.mark.parametrize('finger_coordinates',[(-.0031,-.0322),(0.,-.0045)])
def test_independent_fingers_agree_with_actual_urdf_frames(finger_coordinates):
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS,READY,TCP_OFFSET,PAD_TIP_PAST_TCP
    from yam_arena.urdf import forward_kinematics
    q=dict(zip(ARM_JOINTS,READY));q.update(joint7=0.,joint8=0.)
    zero=forward_kinematics(YAM_URDF,q)
    refs=[np.array([0.,0.,TCP_OFFSET[2]+PAD_TIP_PAST_TCP-d])for d in(.012,0.)]
    local={name:[np.linalg.solve(zero[name][:3,:3],zero['gripper'][:3,:3]@ref+zero['gripper'][:3,3]-zero[name][:3,3])for ref in refs]
           for name in('tip_left','tip_right')}
    q.update(joint7=finger_coordinates[0],joint8=finger_coordinates[1]);actual=forward_kinematics(YAM_URDF,q);G=actual['gripper']
    points,gap=finger_pad_points(G[:3,3],G[:3,:3],*finger_coordinates,*refs)
    expected=np.array([actual[name][:3,:3]@local[name][i]+actual[name][:3,3]for i in range(2)for name in('tip_left','tip_right')])
    assert np.allclose(points,expected,atol=1e-12)
    assert gap==pytest.approx(np.linalg.norm(expected[0]-expected[1]))
    assert not np.allclose(points[:2].mean(0),G[:3,:3]@refs[0]+G[:3,3])


def test_actual_rim_projection_uses_the126mm_opening_at_its_real_tilted_pose():
    from motion_checks import middle_over_rim
    centre=np.array([.2,.37,.7475]);R=np.eye(3)
    assert middle_over_rim(centre+[.125,0,.15],centre,R,.12,.126)
    assert not middle_over_rim(centre+[.127,0,.15],centre,R,.12,.126)
    angle=np.radians(25);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    for radius,expected in ((.125,True),(.127,False)):
        actual=centre+R@np.array([radius,0.,.12])
        assert middle_over_rim(actual+[0,0,.06],centre,R,.12,.126)is expected
    assert not middle_over_rim(centre+[.125,0,.15],centre+[.3,0,0],np.eye(3),.12,.126)
