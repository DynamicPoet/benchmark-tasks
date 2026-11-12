import numpy as np
import pytest

from body_distance import certify_surface, ellipsoid_closest, union_distance_bounds


def test_sphere_distances_and_inside_sign():
    points = np.array([[0, 0, 0], [0, 0, 1.2], [0.8, 0, 0], [0, -1, 0]])
    low, high = union_distance_bounds(points, [((0, 0, 0), (1, 1, 1))])
    expected = np.array([-1.0, 0.2, -0.2, 0.0])
    assert np.all(low <= expected + 1e-8)
    assert np.all(high >= expected - 1e-8)
    np.testing.assert_allclose(low, expected, atol=2e-9)
    np.testing.assert_allclose(high, expected, atol=2e-9)


def test_ellipsoid_axis_and_centre_cases():
    p = np.array([[0.0, 0, 0], [0, 0, 1.1], [3.2, 0, 0]])
    _, distance, _ = ellipsoid_closest(p, (0, 0, 0), (3, 2, 1))
    np.testing.assert_allclose(distance, [1.0, 0.1, 0.2], atol=1e-10)


def test_internal_overlap_faces_are_not_exterior_surface():
    bodies = [((-0.5, 0, 0), (1, 1, 1)), ((0.5, 0, 0), (1, 1, 1))]
    low, high = union_distance_bounds(np.array([[0, 0, 0]]), bodies)
    true_depth = np.sqrt(0.75)
    assert low[0] <= -true_depth <= high[0]
    assert low[0] < -0.8  # the buried individual sphere faces are only 0.5 away


def test_face_interior_cannot_hide_deep_penetration():
    bodies = [((0, 0, 0), (0.1, 0.1, 0.1))]
    # All three vertices lie on the sphere, but the displayed flat triangle
    # crosses deeply through its interior.
    triangle = np.array([[[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0]]])
    passed, _, _ = certify_surface(triangle, bodies)
    assert not passed


def test_unresolved_face_fails_closed():
    bodies = [((0, 0, 0), (0.1, 0.1, 0.1))]
    triangle = np.array([[[0, 0, 0.101], [0.01, 0, 0.101], [0, 0.01, 0.101]]])
    assert not certify_surface(triangle, bodies, max_depth=0)[0]
