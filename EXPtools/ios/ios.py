"""
Modules to read coefficients and basis from EXP, Gala, and AGAMA

TODO: 
read gala coefficients into EXP
read AGAMA coefficients into EXP
"""


import numpy as np
import pyEXP
from EXPtools.basis.basis_utils import make_config

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
    if config is None:
        config = make_config(kwargs['basis_id'],
                             kwargs['numr'],
                             kwargs['rmin'],
                             kwargs['rmax'],
                             kwargs['lmax'],
                             kwargs['nmax'],
                             kwargs['scale'],
                             kwargs['modelname'],
                             kwargs['cachename'])
    basis = pyEXP.basis.Basis.factory(config)
    coefs = pyEXP.coefs.Coefs.factory(coeffile)
    return basis, coefs



