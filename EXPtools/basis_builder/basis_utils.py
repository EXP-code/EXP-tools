import os
import sys
import yaml
import numpy as np
import pyEXP
from scipy.optimize import minimize

def write_table(tablename, radius, density, mass, potential, fmt="%.6e"):
    """
    Write a table of radius, density, mass, and potential values to a text file.

    Parameters
    ----------
    tablename : str
        Output filename.
    radius, density, mass, potential : array-like
        Arrays of physical quantities, all with the same length.
    fmt : str, optional
        Format string for numerical values. Defaults to scientific notation with 6 decimals.

    Notes
    -----
    Writes the table in the following format:
        ! <tablename>
        ! R    D    M    P
        <Nrows>
        <radius> <density> <mass> <potential>
    """
    # Convert inputs to NumPy arrays (for safety and performance)
    radius = np.asarray(radius)
    density = np.asarray(density)
    mass = np.asarray(mass)
    potential = np.asarray(potential)

    # Stack data into a single 2D array for fast writing
    data = np.column_stack((radius, density, mass, potential))

    header = f"! {tablename}\n! R    D    M    P\n{len(radius)}"
    np.savetxt(tablename, data, fmt=fmt, header=header, comments="")

def make_config(basis_id, lmax, nmax, rmapping=1,
    modelname="", cachename=".slgrid_sph_cache",
    float_fmt_rmin="{:.7f}", float_fmt_rmax="{:.3f}", float_fmt_rmapping="{:.3f}"):
    """
    Create a YAML configuration file string for building a basis model.

    Parameters
    ----------
    basis_id : str
        Identity of the basis model.
    numr : int
        Number of radial grid points.
    rmin : float
        Minimum radius value.
    rmax : float
        Maximum radius value.
    lmax : int
        Maximum `l` value of the basis.
    nmax : int
        Maximum `n` value of the basis.
    rmapping : float
        Radial mapping parameter.
    modelname : str, optional
        Name of the model. Default is an empty string.
    cachename : str, optional
        Name of the cache file. Default is '.slgrid_sph_cache'.
    float_fmt_rmin, float_fmt_rmax, float_fmt_rmapping : str, optional
        Format strings for controlling float precision.

    Returns
    -------
    str
        YAML configuration file contents.
    """
    R = np.loadtxt(modelname, skiprows=3, usecols=0)
    rmin = R[0]
    rmax = R[-1]
    numr = len(R)

    config_dict = {
        "id": basis_id,
        "parameters": {
            "numr": numr,
            "rmin": float_fmt_rmin.format(rmin),
            "rmax": float_fmt_rmax.format(rmax),
            "Lmax": lmax,
            "nmax": nmax,
            "rmapping": float_fmt_rmapping.format(rmapping),
            "modelname": modelname,
            "cachename": cachename,
        }
    }
    return yaml.dump(config_dict, sort_keys=False)

def make_Dfit(r_data, rho_data, fit_func, 
              params_guess=None, params_bounds=None):
    """
    Fit a density profile to data using least-squares in log space.

    Parameters
    ----------
    r_data : ndarray
        Radial grid where the density data is defined.
    rho_data : ndarray
        Observed density values at `r_data`.
    fit_func : callable
        Model function with signature `fit_func(params, *fun_params)`.
    fun_params : tuple
        Extra arguments to pass to `fit_func` (e.g., analytic profile).
    params_guess : list of float, optional
        Initial parameter guess (default: [1.0, 0.5]).
    params_bounds : list of tuple, optional
        Bounds for parameters (default: [(1e-4, 1e-2), (0, 2)]).

    Returns
    -------
    rho_fit : ndarray
        Best-fit model evaluated at `r_data`.
    best_fit_params : ndarray
        Optimized parameter values.
    """
    
    log_rho_data = np.log10(np.maximum(rho_data, 1e-12))

    def objective(params):
        rho_model = fit_func(params, r_data)
        log_rho_model = np.log10(np.maximum(rho_model, 1e-12))
        return np.sum((log_rho_model - log_rho_data) ** 2)

    res = minimize(
        objective,
        x0=params_guess,
        method="L-BFGS-B",
        bounds=params_bounds
    )

    best_fit_params = res.x
    rho_fit = fit_func(best_fit_params, r_data)
    return rho_fit, best_fit_params


def make_model(radius, density, Mtotal, output_filename='', physical_units=False, verbose=True):
    """
    Generate an EXP-compatible spherical basis function table.

    Parameters
    ----------
    radius : array-like
        Radii at which the density values are evaluated.
    density : array-like
        Density values corresponding to radius.
    Mtotal : float
        Total mass of the model, used for normalization.
    output_filename : str, optional
        Name of the output file to save the table. If empty, no file is written.
    physical_units : bool, optional
        If True, disables scaling and returns physical values (default: False).
    verbose : bool, optional
        If True, prints scaling information.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
        - 'radius' : ndarray
            Scaled radius values.
        - 'density' : ndarray
            Scaled density values.
        - 'mass' : ndarray
            Scaled enclosed mass values.
        - 'potential' : ndarray
            Scaled potential values.
    """
    EPS_MASS = 1e-15
    EPS_R = 1e-10
    
    Rmax = np.nanmax(radius)
    
    mass = np.zeros_like(density)
    pwvals = np.zeros_like(density)

    mass[0] = 1.e-15
    pwvals[0] = 0.

    dr = np.diff(radius)  

    # Midpoint integration for enclosed mass and potential
    mass_contrib = 2.0 * np.pi * (
        radius[:-1]**2 * density[:-1] + radius[1:]**2 * density[1:]
    ) * dr

    pwvals_contrib = 2.0 * np.pi * (
        radius[:-1] * density[:-1] + radius[1:] * density[1:]
    ) * dr

    # Now cumulative sum to get the arrays
    mass = np.concatenate(([EPS_MASS], EPS_MASS + np.cumsum(mass_contrib)))
    pwvals = np.concatenate(([0.0], np.cumsum(pwvals_contrib)))
    
    potential = -mass / (radius + EPS_R) - (pwvals[-1] - pwvals)

    M0 = mass[-1]
    R0 = radius[-1]

    Beta = (Mtotal / M0) * (R0 / Rmax)
    Gamma = np.sqrt((M0 * R0) / (Mtotal * Rmax)) * (R0 / Rmax)

    if verbose:
        print(f"! Scaling: R = {Rmax}  M = {Mtotal}")

    rfac = Beta**-0.25 * Gamma**-0.5
    dfac = Beta**1.5 * Gamma
    mfac = Beta**0.75 * Gamma**-0.5
    pfac = Beta

    if physical_units:
        rfac = dfac = mfac = pfac = 1.0

    if verbose:
        print(f"Scaling factors: rfac = {rfac}, dfac = {dfac}, mfac = {mfac}, pfac = {pfac}")

    if output_filename:
        write_table(
            output_filename,
            radius * rfac,
            density * dfac,
            mass * mfac,
            potential * pfac
        )

    return {
        "radius": radius * rfac,
        "density": density * dfac,
        "mass": mass * mfac,
        "potential": potential * pfac,
    }
    
    
def make_basis(R, D, Mtotal, basis_params, modelname="test_model.txt"):
    """
    Construct a basis from a given radial density profile.

    Parameters
    ----------
    R : array_like
        Radial grid points (e.g., radii at which density `D` is defined).
    D : array_like
        Density values corresponding to each radius in `R`.
    Mtotal : float, optional
        Total mass normalization (default is 1.0).
    modelname : str, optional
        Name of the model, used for intermediate output files (default is "model").
    lmax : int, optional
        Maximum spherical harmonic degree `l` for the expansion (default is 10).
    nmax : int, optional
        Maximum radial order `n` for the expansion (default is 10).

    Returns
    -------
    basis : pyEXP.basis.Basis
        A SCF basis object initialized with the given density model.

    Notes
    -----
    - This function wraps `makemodel.makemodel` to generate a model from 
      the supplied density profile and total mass.
    - It then builds a spherical basis (`sphereSL`) using `EXPtools.make_config`
      and returns the corresponding `pyEXP` basis object.
    """
    R, D, M, P = make_model(
        D, R, Mtotal=Mtotal, 
        output_filename=modelname, 
        pfile=modelname
    )

    config = make_config(
        basis_id=basis_params['basis_id'], 
        lmax=basis_params['lmax'], 
        nmax=basis_params['nmax'], 
        rmapping=R[-1], 
        modelname=modelname
    )

    basis = pyEXP.basis.Basis.factory(config)
    return basis

