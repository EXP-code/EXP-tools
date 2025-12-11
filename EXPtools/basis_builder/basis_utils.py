import os
import sys
import yaml
import numpy as np
import pyEXP
from scipy.optimize import minimize
from EXPtools.basis_builder.makemodel import make_model

def check_basis_params(basis_id, **kwargs):
    """
    Check that the required keyword arguments for a given basis are provided.

    Parameters
    ----------
    basis_id : str
        The identifier of the basis set. 
        Accepted values are:
        - ``'sphereSL'`` : Spherical basis with Stäckel-like mapping.
        - ``'cylinder'`` : Cylindrical basis.
    **kwargs : dict
        Arbitrary keyword arguments corresponding to the basis parameters.
        The required keys depend on ``basis_id``:

        - For ``'sphereSL'``:
          ['lmax', 'mmax', 'modelname', 'rmapping', 'cachename']

        - For ``'cylinder'``:
          ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax', 'ncylodd',
           'ncylnx', 'ncylny', 'rnum', 'pmun', 'tnum', 'vflag', 'logr', 'cachename']

    Returns
    -------
    bool
        Returns ``True`` if all mandatory parameters are present.

    Raises
    ------
    KeyError
        If one or more mandatory keyword arguments are missing for the selected basis.
    AttributeError
        If ``basis_id`` is not recognized (must be either 'sphereSL' or 'cylinder').

    Examples
    --------
    check_basis_params('sphereSL', lmax=4, nmax=4, modelname='hernquist',
    ...                    rmapping='linear', cachename='cache_sph')
    True

    check_basis_params('cylinder', acyl=1.0, hcyl=2.0)  
    Traceback (most recent call last):
    ...
    KeyError: "Missing mandatory keyword arguments missing: [...]"
    """
    
    if basis_id == 'sphereSL':
        mandatory_keys = ['lmax', 'nmax', 'modelname', 'rmapping', 'cachename']
        missing = [key for key in mandatory_keys if key not in kwargs]
        if missing:
            raise KeyError(f"Missing mandatory keyword arguments missing: {missing}")
        return True
    elif basis_id == 'cylinder':
        mandatory_keys = ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 
                           'mmax', 'nmax', 'ncylodd', 'ncylnx', 
                           'ncylny', 'rnum', 'pmun', 'tnum', 'vflag', 'logr', 'cachename']  
        missing = [key for key in mandatory_keys if key not in kwargs]
        if missing:
            raise KeyError(f"Missing mandatory keyword arguments missing: {missing}")
        return True
    else: 
        raise AttributeError(f"basis id {basis_id} not found. Please chose between sphereSL or cylinder")
	
def make_config(basis_id, float_fmt_rmin="{:.7f}", float_fmt_rmax="{:.3f}",
                float_fmt_rmapping="{:.3f}", **params):
    """
    Create a YAML configuration file string for building a basis model.

    Parameters
    ----------
    basis_id : str
        Identifier of the basis model. Must be either 'sphereSL' or 'cylinder'.
    float_fmt_rmin : str, optional
        Format string for rmin (default ``"{:.7f}"``).
    float_fmt_rmax : str, optional
        Format string for rmax (default ``"{:.3f}"``).
    float_fmt_rmapping : str, optional
        Format string for rmapping (default ``"{:.3f}"``).
    **kwargs : dict
        Additional keyword arguments required depending on the basis type:

        - For ``sphereSL``:
          ['lmax', 'nmax', 'rmapping', 'modelname', 'cachename']

        - For ``cylinder``:
          ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax',
           'ncylodd', 'ncylnx', 'ncylny', 'rnum', 'pnum', 'tnum',
           'vflag', 'logr', 'cachename']

    Returns
    -------
    str
        YAML configuration file contents.

    Raises
    ------
    KeyError
        If mandatory parameters for the given basis are missing.
    FileNotFoundError
        If ``modelname`` is required but cannot be opened.
    ValueError
        If the model file does not contain valid radius data.
    """

    check_basis_params(basis_id, **kwargs)

    if basis_id == "sphereSL":
        modelname = kwargs["modelname"]
        try:
            R = np.loadtxt(modelname, skiprows=3, usecols=0)
        except OSError as e:
            raise FileNotFoundError(f"Could not open model file '{modelname}'") from e
        if R.size == 0:
            raise ValueError(f"Model file '{modelname}' contains no radius data")

        rmin, rmax, numr = R[0], R[-1], len(R)

        # TODO this should be done in yaml directly! 
        # Write some log here! 
    config_dict = {
        "id": basis_id,
        "parameters": params
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

  
def make_basis(R, D, Mtotal, basis_params, modelname="test_model.txt", cachename='test_cache.txt'):
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
    basis_params : dict 
        basis parameters e.g., basis_id, nmax, lmax

    Returns
    -------
    basis : pyEXP.basis.Basis
        A basis object initialized with the given density model.

    Notes
    -----
    - This function wraps `makemodel.makemodel` to generate a model from 
      the supplied density profile and total mass.
    - It then builds a basis either spherical (`sphereSL`) or cylindrical using `EXPtools.make_config`
      and returns the corresponding `pyEXP` basis object.
    """

    R, D, M, P = make_model(
        R, D, Mtotal=Mtotal, 
        output_filename=modelname
    )

    config = make_config(
        basis_id=basis_params['basis_id'], 
        lmax=basis_params['lmax'], 
        nmax=basis_params['nmax'], 
        rmapping=R[-1], 
        modelname=modelname,
        cachename=cachename
    )

    basis = pyEXP.basis.Basis.factory(config)
    return basis

