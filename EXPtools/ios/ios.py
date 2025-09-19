"""
Modules to read coefficients and basis from EXP, Gala, and AGAMA

TODO: 
read gala coefficients into EXP
read AGAMA coefficients into EXP
"""

import os
import numpy as np
from pytest import approx
import pyEXP
from EXPtools.basis_builder.basis_utils import make_config

DATAPATH = "../../tests/data/"


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

## tests

def _exp_coefficients():
    """
    Test 
    """
    coeftest = os.path.join(DATAPATH, "spherical_hernquist_halo_coef.h5")
    config = make_config('sphereSL', numr=51, rmin=0.02,
                        rmax=1.9799999999999995, lmax=5, nmax=10, scale=1,
                        modelname=DATAPATH+'SLGrid.empirical_spherical_hernquist_halo',
                        cachename=DATAPATH+".slgrid_spherical_hernquist_halo")
    basis, coefs = exp_coefficients(coeftest, config)

    basis_density = np.log10(basis.getBasis()[0][0]['density'].flatten()[0::20])
    basis_true = np.array([2.52452473, 2.52007596,  2.51333528, 2.50305991,
                           2.4872475, 2.46254365, 2.42296872, 2.35670352,
                           2.23560574, 2.02788039, 1.82455463, 1.61151223,
                           1.38374334,  1.13708294, 0.86224642, 0.55234052,
                           0.19739773, -0.20560017, -0.66084427, -1.1953064 ])


    assert basis_density == approx(basis_true)

    assert coefs.getAllCoefs().shape == approx((21, 10, 1))

    first_coefs = np.array([1.58866402e+00, 1.61446863e-02, -1.21213618e-03,
                            1.94620389e-03, 8.00640342e-04, -9.05078238e-04,
                            -1.89598446e-03, -8.74281279e-04, 1.00354378e-03,
                            4.03461871e-04])

    assert coefs.getAllCoefs()[0].real.flatten() == approx(first_coefs)
    return basis, coefs

if __name__ == "__main__":
    _exp_coefficients()
