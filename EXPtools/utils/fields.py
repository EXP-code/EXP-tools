import numpy as np
import pyEXP
from EXPtools.visuals import Grid3D

def bfe_density_profiles(
    basis,
    coefs,
    r_bins,
    theta_bins=20,
    phi_bins=40,
    time=0.0,
    field="dens",
    statistic="median",
):
    """
    Compute a spherically averaged density profile from a BFE by
    sampling the density field on a spherical grid at specified
    radial bin centers.

    Parameters
    ----------
    basis : pyEXP basis object
    coefs : pyEXP coefficients
    r_bins : array-like
        Radial bin centers at which the density is evaluated.
    theta_bins : int, optional
        Number of azimuthal samples.
    phi_bins : int, optional
        Number of polar samples.
    time : float, optional
        Snapshot time.
    field : str, optional
        Field name to extract (default: 'dens').
    statistic : {'mean', 'median'}, optional
        Angular statistic to compute at each radius.

    Returns
    -------
    ndarray
        Spherically averaged density evaluated at each radius.
    """

    r_bins = np.asarray(r_bins)

    # Build spherical grid at bin centers
    grid = Grid3D(
        system="spherical",
        ranges=[(r_bins.min(), r_bins.max()), None, None],
        num_points=[len(r_bins), theta_bins, phi_bins],
        r_bins=r_bins,
    )

    xyz = grid.to("cartesian")

    # Evaluate field
    fields = pyEXP.field.FieldGenerator([time], xyz)
    field_data = fields.points(basis, coefs)[time]

    if field not in field_data:
        raise ValueError(
            f"Field '{field}' not available. "
            f"Available fields: {list(field_data.keys())}"
        )

    dens = field_data[field]

    # Reshape: (nr, ntheta * nphi)
    nr = len(r_bins)
    dens = dens.reshape(nr, -1)

    if statistic == "mean":
        profile = np.mean(dens, axis=1)
    elif statistic == "median":
        profile = np.median(dens, axis=1)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    return profile
