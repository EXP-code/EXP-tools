"""
Modules to read coefficients and basis from EXP, Gala, and AGAMA

TODO: 
read gala coefficients into EXP
read AGAMA coefficients into EXP
"""

import pyEXP
from EXPtools.basis.basis_utils import write_config

def exp_coefficients(coeffile, config, **kwargs):
    """
    Reads EXP coefficients
    Parameters:
    -----------
    coeffile
    config 
    **kwargs

    TODO:
    Have a cache reader

    """
    basis = pyEXP.basis.Basis.factory(config)
    coefs = pyEXP.coefs.Coefs.factory(coeffile)
    return basis, coefs



