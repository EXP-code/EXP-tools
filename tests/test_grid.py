import numpy as np
import pytest
import numpy.testing as npt

from EXPtools.visuals import Grid3D

def test_spherical_angle_ranges():
    grid = Grid3D(
        "spherical",
        ranges=[(0.5, 2.0), None, None],
        num_points=[4, 16, 9],
    )

    r, theta, phi = grid.get().T

    assert np.all(phi >= 0.0)
    assert np.all(phi < 2 * np.pi)

    assert np.all(theta >= 0.0)
    assert np.all(theta <= np.pi)

def test_cylindrical_theta_range():
    grid = Grid3D(
        "cylindrical",
        ranges=[(0.2, 1.0), None, (-1.0, 1.0)],
        num_points=[5, 16, 7],
    )

    rho, phi, z = grid.get().T

    assert np.all(phi >= 0.0)
    assert np.all(phi < 2 * np.pi)

def test_spherical_radius_monotonicity():
    grid = Grid3D(
        "spherical",
        ranges=[(1.0, 5.0), None, None],
        num_points=[10, 4, 4],
    )

    r = grid.get()[:, 0]

    # Extract unique radial shells
    unique_r = np.unique(r)

    assert np.all(np.diff(unique_r) > 0.0)

def test_spherical_to_cartesian_radius_consistency():
    grid = Grid3D(
        "spherical",
        ranges=[(0.5, 3.0), None, None],
        num_points=[6, 8, 6],
    )

    spherical = grid.get()
    cartesian = grid.to("cartesian")

    r_sph = spherical[:, 0]
    r_cart = np.linalg.norm(cartesian, axis=1)

    npt.assert_allclose(r_sph, r_cart, rtol=1e-6, atol=1e-8)

def test_cylindrical_to_cartesian_rho_consistency():
    grid = Grid3D(
        "cylindrical",
        ranges=[(0.3, 2.0), None, (-1.0, 1.0)],
        num_points=[6, 12, 5],
    )

    cylindrical = grid.get()
    cartesian = grid.to("cartesian")

    rho = cylindrical[:, 0]
    x, y = cartesian[:, 0], cartesian[:, 1]

    rho_cart = np.sqrt(x**2 + y**2)

    npt.assert_allclose(rho, rho_cart, rtol=1e-6, atol=1e-8)

def test_cartesian_identity_transform():
    grid = Grid3D(
        "cartesian",
        ranges=[(-1, 1), (-2, 2), (-3, 3)],
        num_points=[4, 5, 6],
    )

    cart = grid.get()
    cart_to_cart = grid.to("cartesian")

    npt.assert_array_equal(cart, cart_to_cart)

@pytest.mark.parametrize("source,target", [
    ("cartesian", "spherical"),
    ("cartesian", "cylindrical"),
    ("spherical", "cartesian"),
    ("cylindrical", "cartesian"),
])
def test_shape_preservation(source, target):
    ranges = {
        "cartesian": [(-1, 1), (-1, 1), (-1, 1)],
        "spherical": [(1, 2), None, None],
        "cylindrical": [(0.2, 1), None, (-1, 1)],
    }

    grid = Grid3D(source, ranges[source], [4, 6, 5])
    transformed = grid.to(target)

    assert transformed.shape == grid.get().shape

