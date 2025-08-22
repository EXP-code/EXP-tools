"""
TODO: Add plotting functions
"""

import os
import yaml
import pyEXP

def write_basis(basis, basis_name):
    """
    Write a basis configuration dictionary to a YAML file.

    Parameters
    ----------
    basis : dict
        Dictionary containing the basis configuration.
    conf_name : str
        Name of the YAML file to write. If the provided name does not
        end with `.yaml`, the extension is automatically appended.

    Returns
    -------
    str
        The final filename used to save the YAML configuration.
    """
    # Ensure file has .yaml extension
    if not basis_name.endswith(".yaml"):
        basis_name += ".yaml"

    # Write to file
    with open(basis_name, "w") as file:
        yaml.dump(basis, file, default_flow_style=False, sort_keys=False)

    return 0

def load_basis(conf_name):
    """
    Load a basis configuration from a YAML file and initialize a Basis object.

    Parameters
    ----------
    conf_name : str
        Path to the YAML configuration file. If the provided filename does not 
        end with `.yaml`, the extension is automatically appended.

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
    if not os.path.exists(conf_name):
        raise FileNotFoundError(f"Configuration file not found: {conf_name}")

    # Load YAML safely
    with open(conf_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build basis from configuration
    basis = pyEXP.basis.Basis.factory(config)
    return basis
    

