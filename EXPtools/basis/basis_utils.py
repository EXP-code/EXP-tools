import os
import re
import yaml
import numpy as np
import pyEXP
from EXPtools.basis.makemodel import make_model

def load_basis(config_name, cache_dir=None):
    """
    Load a basis configuration from a YAML file and initialize a Basis object.

    Parameters
    ----------
    config_name : str
        Path to the YAML configuration file. If the provided filename does not 
        end with `.yaml`, the extension is automatically appended.
    cache_dir : str (optional)
        Path to modelname and cachename (assumes both are in the same directory)
        if None assumes it is the current working directory.

    Returns
    -------
    basis : pyEXP.basis.Basis
        An initialized Basis object created from the configuration.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    """

    # Check file existence
    if not os.path.exists(config_name):
        raise FileNotFoundError(f"Configuration file not found: {config_name}")

    # Load YAML safely
    with open(config_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if cache_dir:
        config = re.sub(r"(modelname:\s*)(\S+)", rf"\1{cache_dir}\2", config)
        config = re.sub(r"(cachename:\s*)(\S+)", rf"\1{cache_dir}\2", config)
    # Build basis from configuration
    basis = pyEXP.basis.Basis.factory(config)
    return basis
    
def check_basis_params(basis_params):
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
          ['Lmax', 'mmax', 'modelname', 'rmapping', 'cachename']

        - For ``'cylinder'``:
          ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax', 'ncylodd',
           'ncylnx', 'ncylny', 'rnum', 'pmun', 'tnum', 'vflag', 'logr', 'cachename']

    Returns
    -------Pipeline repo: https://github.com/jngaravitoc/XMC-Atlas/tree/main/scripts  


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
    check_basis_params('sphereSL', Lmax=4, nmax=4, modelname='hernquist.txt',
    ...                    rmapping=1, cachename='cache_hernquist.txt')
    True

    check_basis_params('cylinder', acyl=1.0, hcyl=2.0)  
    Traceback (most recent call last):
    ...
    KeyError: "Missing mandatory keyword arguments missing: [...]"
    """
    
    if basis_params['basis_id'] == 'sphereSL':
        mandatory_keys = ['Lmax', 'nmax', 'modelname', 'rmapping', 'cachename']
        missing = [key for key in mandatory_keys if key not in basis_params]
        if missing:
            raise KeyError(f"Missing mandatory keyword arguments missing: {missing}")
        return True
    elif basis_params['basis_id'] == 'cylinder':
        mandatory_keys = ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 
                           'mmax', 'nmax', 'ncylodd', 'ncylnx', 
                           'ncylny', 'rnum', 'pnum', 'tnum', 'vflag', 'logr', 'cachename']  
        missing = [key for key in mandatory_keys if key not in basis_params]
        if missing:
            raise KeyError(f"Missing mandatory keyword arguments missing: {missing}")
        return True
    else: 
        raise AttributeError(f"basis id {basis_params['basis_id']} not found. Please chose between sphereSL or cylinder")


def write_config(basis_params):
    """
    Create a YAML configuration file string for building a basis model.

    Parameters
    ----------
    basis_id : str
        Identifier of the basis model. Must be either 'sphereSL' or 'cylinder'.

    **params : dict
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
    print(basis_params)
    check_basis_params(basis_params)

    
    if basis_params['basis_id'] == "sphereSL":
        modelname = basis_params["modelname"]
        try:
            R = np.loadtxt(modelname, skiprows=3, usecols=0)
        except OSError as e:
            raise FileNotFoundError(f"Could not open model file '{modelname}'") from e
        if R.size == 0:
            raise ValueError(f"Model file '{modelname}' contains no radius data")

        rmin, rmax, numr = R[0], R[-1], len(R)
        basis_params["rmin"] = float("{:.7f}".format(rmin))
        basis_params["rmax"] = float("{:.3f}".format(rmax))
        basis_params["numr"] = int(numr)

    basis_id = basis_params['basis_id']
    basis_params.pop('basis_id')
    config_dict = {
        "id": basis_id,
        "parameters": basis_params
        }
    
    print(config_dict)
    
    
    return yaml.dump(config_dict, sort_keys=False)

def make_basis(R, D, Mtotal, **basis_params):
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

    if "modelname" not in basis_params.keys():
        basis_params['modelname']="test_model.txt"

    if "cachename" not in basis_params.keys():
        basis_params['cachename']="test_cache.txt"
    
    R, D, _, _ = make_model(
        R, D, Mtotal=Mtotal, 
        output_filename=basis_params['modelname']
    )


    config = write_config(basis_params)

    basis = pyEXP.basis.Basis.factory(config)
    return basis