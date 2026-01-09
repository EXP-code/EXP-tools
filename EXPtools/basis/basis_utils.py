import os
import re
import yaml
import numpy as np
import pyEXP
from EXPtools.basis.makemodel import make_model

def load_basis(config_file, cache_dir=None):
    """
    Load a basis configuration from a YAML file and initialize a Basis object.

    Parameters
    ----------
    config_file : str
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
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load YAML safely
    #with open(config_file) as f:
    #    config_yaml = f.read()
    with open(config_file, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    
    modelfile = config_yaml["parameters"]["modelname"]
    if not os.path.exists(modelfile):
        raise FileNotFoundError(f"Modelname file not found: {modelfile}")

    #if cache_dir:
    #    config_yaml = re.sub(r"(modelname:\s*)(\S+)", rf"\1{cache_dir}\2", config_yaml)
    #    config_yaml = re.sub(r"(cachename:\s*)(\S+)", rf"\1{cache_dir}\2", config_yaml)
    # Build basis from configuration
    config_str = yaml.safe_dump(config_yaml)
    
    
    basis = pyEXP.basis.Basis.factory(config_str)
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
        mandatory_keys = [
            'acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax',
            'ncylodd', 'ncylnx', 'ncylny', 'rnum', 'pnum', 'tnum',
            'vflag', 'logr', 'cachename',
        ]
        missing = [key for key in mandatory_keys if key not in basis_params]
        if missing:
            raise KeyError(f"Missing mandatory keyword arguments missing: {missing}")
        return True
    else: 
        raise AttributeError(f"basis id {basis_params['basis_id']} not found. Please chose between sphereSL or cylinder")


def write_config(basis, basis_filename):
    with open(basis_filename, 'w') as file:
        yaml.safe_dump(basis, file, default_flow_style=False)


def config_dict_to_yaml(basis_params):

    """
    Create a YAML configuration file string for building a basis model.

    Parameters
    ----------
    basis_params : dict
        Dictionary containing basis configuration parameters. Must include
        the key ``'basis_id'`` and all required parameters for the chosen basis.
    
        - For ``sphereSL``:
          ['lmax', 'nmax', 'rmapping', 'modelname', 'cachename']

        - For ``cylinder``:
          ['acyl', 'hcyl', 'nmaxfid', 'lmaxfid', 'mmax', 'nmax',
           'ncylodd', 'ncylnx', 'ncylny', 'rnum', 'pnum', 'tnum',
           'vflag', 'logr', 'cachename']
    
    write_config : bool, optional
        If True, write the YAML configuration to disk. Default is True.

    filename : str, optional
        Output filename for the YAML configuration. Only used if
        ``write_config=True``. Default is ``'basis_config.yaml'``.

    
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
    basis_params = basis_params.copy()  
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

    #remove id
    basis_id = basis_params.pop("basis_id")
    config_dict = {
        "id": basis_id,
        "parameters": basis_params
        }
    config_yaml = yaml.dump(config_dict, sort_keys=False)
    return config_yaml
    

def make_basis(radii, density, Mtotal, basis_params, physical_units=True, write_basis=False, basis_filename='test_config.yaml'):
    """
    Construct a basis from a density profile.

    Parameters
    ----------
    radii : array_like
        Radial grid points (e.g., radii at which density `D` is defined).
    density : array_like
        Density values corresponding to each radius in `R`.
    Mtotal : float, optional
        Total mass normalization (default is 1.0).
    basis_params : dict 
        basis parameters e.g., basis_id, nmax, lmax
        For the descriptions of the basis_params please see the EXP docs:
        https://github.com/EXP-code/EXP-docs/blob/93207da758d34cf10092650a84
        0cdb23180b859a/topics/yamlconfig.rst#L229


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
    
    _ = make_model(
        radii, density, Mtotal=Mtotal, 
        output_filename=basis_params['modelname'], 
        physical_units=physical_units
    )
    print('Done making model')
    config = config_dict_to_yaml(basis_params)
    #if write_basis == True:
    #    yaml_config = yaml.safe_load(config)
    #    write_config(yaml_config, basis_filename=basis_filename)
    #basis = load_basis(basis_filename)
    
    if write_basis:
        yaml_config = yaml.safe_load(config)
        write_config(yaml_config, basis_filename)
        return load_basis(basis_filename)

    return pyEXP.basis.Basis.factory(config)
