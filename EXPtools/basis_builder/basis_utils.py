import os, sys, pickle, pyEXP
import numpy as np
from scipy.linalg import norm
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
    return density


    
def makebasis(pos, mass, basis_model, config=None, basis_id='sphereSL', time=0, 
               numr=500, rmin=0.61, rmax=599, lmax=4, nmax=20, scale=1, 
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
            R, D, M, P = makemodel(func=empirical_density_profile, M=np.sum(mass), funcargs=[0], rvals=np.logspace(np.log10(rmin),  np.log10(rmax), numr), pfile=modelname)
        elif basis_model == "Hernquist":
            print('-> Computing analytical Hernquist model')
            #makemodel.hernquist_halo()
            #R, D, M, P = makemodel.makemodel(hernquist_halo, 1, [scale], rc=0, alpha=1.0, beta=2.0)
        
        elif basis_model == "powerlaw":
            print('-> Computing analytical Hernquist model')
            #makemodel.hernquist_halo()
            rbins = np.logspace(np.log10(rmin), np.log10(rmax), numr+1)
            
            R, D, M, P = makemodel.makemodel(makemodel.powerhalo, M=np.sum(mass),
                                             funcargs=[1, 0, 1.0, 2.0], rvals = rbins,
                                             pfile=modelname)
            
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
