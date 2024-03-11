import os, sys, pickle, pyEXP
import numpy as np
from EXPtools.basis_builder import makemodel

def make_config(basis_id, numr, rmin, rmax, lmax, nmax, scale, 
                modelname='', cachename='.slgrid_sph_cache'):
    """
    Creates a configuration file required to build a basis model.

    Args:
    basis_id (str): The identity of the basis model.
    numr (int): The number of radial grid points.
    rmin (float): The minimum radius value.
    rmax (float): The maximum radius value.
    lmax (int): The maximum l value of the basis.
    nmax (int): The maximum n value of the basis.
    scale (float): Scaling factor for the basis.
    modelname (str, optional): Name of the model. Default is an empty string.
    cachename (str, optional): Name of the cache file. Default is '.slgrid_sph_cache'.

    Returns:
    str: A string representation of the configuration file.

    Raises:
    None
    """
    
    config = '\n---\nid: {:s}\n'.format(basis_id)
    config += 'parameters:\n'
    config += '  numr: {:d}\n'.format(numr)
    config += '  rmin: {:.7f}\n'.format(rmin)
    config += '  rmax: {:.3f}\n'.format(rmax)
    config += '  Lmax: {:d}\n'.format(lmax)
    config += '  nmax: {:d}\n'.format(nmax)
    config += '  scale: {:.3f}\n'.format(scale)
    config += '  modelname: {}\n'.format(modelname)
    config += '  cachename: {}\n'.format(cachename)
    config += '...\n'
    return config

def empirical_density_profile(pos, mass, nbins=500, rmin=0, rmax=600, log_space=False):
    """
    Computes the number density radial profile assuming all particles have the same mass.

    Args:
        pos (ndarray): array of particle positions in cartesian coordinates with shape (n,3).
        mass (ndarray): array of particle masses with shape (n,).
        nbins (int, optional): number of bins in the radial profile. Default is 500.
        rmin (float, optional): minimum radius of the radial profile. Default is 0.
        rmax (float, optional): maximum radius of the radial profile. Default is 600.
        log_space (bool, optional): whether to use logarithmic binning. Default is False.

    Returns:
        tuple: a tuple containing the arrays of radius and density with shapes (nbins,) and (nbins,), respectively.

    Raises:
        ValueError: if pos and mass arrays have different lengths or if nbins is not a positive integer.
    """
    if len(pos) != len(mass):
        raise ValueError("pos and mass arrays must have the same length")
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")

    # Compute radial distances
    r_p = np.sqrt(np.sum(pos**2, axis=1))

    # Compute bin edges and shell volumes
    if log_space:
        bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    else:
        bins = np.linspace(rmin, rmax, nbins+1)
    V_shells = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)

    # Compute density profile
    density, _ = np.histogram(r_p, bins=bins, weights=mass)
    density /= V_shells

    # Compute bin centers and return profile
    radius = 0.5 * (bins[1:] + bins[:-1])
    return radius, density

def make_exp_basis_table(radius, density, outfile='', plabel='', 
                          verbose=True, physical_units=True, return_values=False):
    
    """
    Create a table of basis functions for an exponential density profile, compatible with EXP.

    Parameters:
    -----------
    radius : array-like
        Array of sampled radius points.
    density : array-like
        Array of density values at radius points.
    outfile : str, optional
        The name of the output file. If not provided, the file will not be written.
    plabel : str, optional
        A comment string to add to the output file.
    verbose : bool, optional
        Whether to print scaling factors and other details during execution.
    physical_units : bool, optional
        Whether to use physical units (True) or normalized units (False) in the output file.
    return_values : bool, optional
        Whether to return the radius, density, mass, and potential arrays (True) or not (False).

    Returns:
    --------
    If return_values is True:
        radius : array-like
            The radius values.
        density : array-like
            The density values.
        mass : array-like
            The mass enclosed at each radius.
        potential : array-like
            The potential energy at each radius.
    If return_values is False (default):
        None

    Notes:
    ------
    This function assumes an exponential density profile:

        rho(r) = rho_0 * exp(-r/a)

    where rho_0 and a are constants.

    The table of basis functions is used by EXP to create a density profile.

    Reference:
    ----------
    https://gist.github.com/michael-petersen/ec4f20641eedac8f63ec409c9cc65ed7
    """    
    

    # make the mass and potential arrays
    rvals, dvals = radius, density 
    mvals = np.zeros(density.size)
    pvals = np.zeros(density.size)
    pwvals = np.zeros(density.size)

    M = 1.
    R = np.nanmax(rvals)


    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1, dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R)
    Gamma = np.sqrt( (M0*R0)/(M*R) ) * (R0/R)
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if outfile != '':
        f = open(outfile,'w')
        print('! ', plabel, file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)
        
        if physical_units:
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rvals[indx],\
                                              dvals[indx],\
                                              mvals[indx],\
                                              pvals[indx]),file=f)
        else:
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rfac*rvals[indx],\
                  dfac*dvals[indx],\
                  mfac*mvals[indx],\
                  pfac*pvals[indx]),file=f)

        f.close()
    
    if return_values:
        if physical_units:
            return rvals, dvals, mvals, pvals

        return rvals*rfac, dfac*dvals, mfac*mvals, pfac*pvals

    
def makebasis(pos, mass, basis_model, config=None, basis_id='sphereSL', time=0, 
               numr=500, rmin=0.61, rmax=599, lmax=4, nmax=20, scale=22.5, 
               norm_mass_coef = True, modelname='dens_table.txt', cachename='.slgrid_sph_cache', add_coef = False, coef_file=''):
    """
    Create a BFE expansion for a given set of particle positions and masses.
    
    Parameters:
    pos (numpy.ndarray): The positions of particles. Each row represents one particle, 
                         and each column represents the coordinate of that particle.
    mass (numpy.ndarray): The masses of particles. The length of this array should be the same 
                          as the number of particles.
    basismodel ():
    config (pyEXP.config.Config, optional): A configuration object that specifies the basis set. 
                                             If not provided, an empirical density profile will be computed 
                                             and a configuration object will be created automatically.
    basis_id (str, optional): The type of basis set to be used. Default is 'sphereSL'.
    time (float, optional): The time at which the expansion is being computed. Default is 0.
    numr (int, optional): The number of radial grid points in the basis set. Default is 200.
    rmin (float, optional): The minimum radius of the basis set. Default is 0.61.
    rmax (float, optional): The maximum radius of the basis set. Default is 599.
    lmax (int, optional): The maximum harmonic order in the basis set. Default is 4.
    nmax (int, optional): The maximum number of polynomials in the basis set. Default is 20.
    scale (float, optional): The scale of the basis set in physical units.
    modelname (str, optional): The name of the file containing the density profile model. 
                               Default is 'dens_table.txt'.
    cachename (str, optional): The name of the file that will be used to cache the basis set. 
                               Default is '.slgrid_sph_cache'.
    save_dir (str, optional): The name of the file if provided that will be used to save the coef files as .h5.
                              Default is ''. 
    Returns:
    tuple: A tuple containing the basis and the coefficients of the expansion.
           The basis is an instance of pyEXP.basis.Basis, and the coefficients are 
           an instance of pyEXP.coefs.Coefs.
    """
    
    if os.path.isfile(modelname) == False:
        print("-> File model not found so we are computing one \n")
        if basis_model == "empirical":
            print('-> Computing empirical model')
            rad, rho = empirical_density_profile(pos, mass, nbins=numr)
            R, D, M, P = makemodel('empirical', func=None, dvals=rho, rvals=np.logspace(np.log10(rmin),  np.log10(rmax), numr), M=np.sum(mass), outfile=modelname, return_values=True)
        elif empirical == "Hernquist":
            print('-> Computing analytical Hernquist model')
            #makemodel.hernquist_halo()
            #R, D, M, P = makemodel.makemodel(hernquist_halo, 1, [scale], rvals=, pfile=modelname)
        
        print('-> Model computed: rmin={}, rmax={}, numr={}'.format(R[0], R[-1], len(R)))
    else:
        R, D, M, P  = np.loadtxt(modelname, skiprows=3, unpack=True) 
    # check if config file is passed
    if config is None:
        print('No config file provided.')
        print(f'Computing empirical density')
        #rad, rho = empirical_density_profile(pos, mass, nbins=500)
        #makemodel_empirical(r_exact, rho, outfile=modelname)
        #R = [0.01, 600]
        config = make_config(basis_id, numr, R[0], R[-1], lmax, nmax, scale, 
                             modelname, cachename)

    # Construct the basis instances
    basis = pyEXP.basis.Basis.factory(config)

    # Prints info from Cache
    basis.cacheInfo(cachename)
    
    #compute coefficients
    if norm_mass_coef == True :
        coef = basis.createFromArray(mass/np.sum(mass), pos, time=time)
    elif norm_mass_coef == False : 
        coef = basis.createFromArray(mass, pos, time=time)

    coefs = pyEXP.coefs.Coefs.makecoefs(coef, 'dark halo')
    coefs.add(coef)
    if add_coef == False:
      coefs.WriteH5Coefs(coef_file)
    elif add_coef == True:
      coefs.ExtendH5Coefs(coef_file)
    
    return basis, coefs
