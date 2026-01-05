"""
Routines for defining and handling unit systems.
"""

import astropy.units as u
import astropy.constants as const
import numpy as np

def define_exp_units(M_phys, R_phys, G, verbose=False):
    """
    Defines EXP (G=1) units in astropy for a given halo,
    when given mass and length in physical units. 
    Base units are called expM, expL, expV, expT. 

    Parameters
    ----------
    M_phys : astropy quantity
        Desired EXP mass unit in some physical unit of mass, 
        i.e. a virial mass in solar masses or the mass unit of an external simulation
        
    R_phys : astropy quantity
        Desired EXP length unit in some physical unit of length, 
        i.e. a virial radius in kpc or the length unit of an external simulation
        
    G : astropy quantity
        G in the physical unit system of the external simulation
        
    verbose : bool, optional
        If True, print out the EXP unit system's conversion to physical units

    Returns
    -------
    expL : astropy unit
        EXP length unit
        
    expM : astropy unit
        EXP mass unit
        
    expV : astropy unit
        EXP velocity unit
        
    expT : astropy unit
        EXP time unit
    
    Example
    -------
    To define the conversion from e.g. Gadget default units, use
    expL, expM, expV, expT = define_exp_units(M_phys=1.*u.Msun, R_phys=1.*u.kpc, G=const.G.to(u.kpc/(1e10*u.Msun)*u.km**2/u.s**2))
    """
    
    if verbose:
        print("Using physical G = ", G)
    
    # define EXP velocity and time 
    expL = u.def_unit('expL', R_phys)
    expM = u.def_unit('expM', M_phys)
    expV = u.def_unit('expV', np.sqrt(G*expM/expL))
    expT = u.def_unit('expT', expL/expV)
    
    # print and return the unit system
    if verbose:
        print('Conversions between EXP units and physical units:')
        print('1 expL = ', 1.*expL.to(u.kpc), 'kpc')
        print('1 expM = ', 1.*expM.to(u.Msun), 'Msun')
        print('1 expV = ', 1.*expV.to(u.km/u.s), 'km/s')
        print('1 expT = ', 1.*expT.to(u.Gyr), 'Gyr')

    return expL, expM, expV, expT
