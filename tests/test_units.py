"""
Tests for unit conversions
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from EXPtools.utils import units

def test_define_exp_units():

	tol = 1e-10
	expL, expM, expV, expT = units.define_exp_units(M_phys=1.*u.Msun, R_phys=1.*u.kpc, G=const.G.to(u.kpc/(1e10*u.Msun)*u.km**2/u.s**2))
	assert (np.abs((1.*expV).to(u.km/u.s).value - 0.002073865296984421) < tol), "Unit definitions in define_exp_units failed!"

	return None
