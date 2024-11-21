import os, sys, pickle, pyEXP
import numpy as np
from scipy.linalg import norm
from EXPtools.basis_builder import makemodel

def old_make_config(basis_id, numr, rmin, rmax, lmax, nmax, scale, 
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
    
    config = 'id: {:s}\n'.format(basis_id)
    config += 'parameters:\n'
    config += '  numr: {:d}\n'.format(numr)
    config += '  rmin: {:.7f}\n'.format(rmin)
    config += '  rmax: {:.3f}\n'.format(rmax)
    config += '  Lmax: {:d}\n'.format(lmax)
    config += '  nmax: {:d}\n'.format(nmax)
    config += '  scale: {:.3f}\n'.format(scale)
    config += '  modelname: {}\n'.format(modelname)
    config += '  cachename: {}\n'.format(cachename)
    return config


def make_halo_config(basis_id, numr, rmin, rmax, lmax, nmax, rmapping, 
                modelname, cachename):
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
    
    config = 'id: {:s}\n'.format(basis_id)
    config += 'parameters:\n'
    config += '  numr: {:d}\n'.format(numr)
    config += '  rmin: {:.7f}\n'.format(rmin)
    config += '  rmax: {:.3f}\n'.format(rmax)
    config += '  Lmax: {:d}\n'.format(lmax)
    config += '  nmax: {:d}\n'.format(nmax)
    config += '  rmapping: {:.3f}\n'.format(rmapping)
    config += '  modelname: {}\n'.format(modelname)
    config += '  cachename: {}\n'.format(cachename)
    return config

    
def makebasis(pos, mass, basis_model, config=None, basis_id='sphereSL', time=0,
              r_s=1.0, r_c=0.0,
              nbins=500, rmin=0.61, rmax=599, log_space=True, lmax=4, nmax=20, scale=1,
              norm_mass_coef = True, modelname='dens_table.txt', cachename='.slgrid_sph_cache', add_coef = False,
              coef_file=''
              ):
    """
    Create a BFE expansion for a given set of particle positions and masses.
    
    Parameters:
    pos (numpy.ndarray): The positions of particles. Each row represents one particle, 
                         and each column represents the coordinate of that particle.
    mass (numpy.ndarray): The masses of particles. The length of this array should be the same 
                          as the number of particles.
    basismodel (string): The model to compute, NFW,Hernquist, singlepowerlaw and empirical are available
                        A modelname file can be used to specify a particular model if needed.
    config (pyEXP.config.Config, optional): A configuration object that specifies the basis set. 
                                             If not provided, an empirical density profile will be computed 
                                             and a configuration object will be created automatically.
    basis_id (str, optional): The type of basis set to be used. Default is 'sphereSL'.
    time (float, optional): The time at which the expansion is being computed. Default is 0.
    r_s (float,optional): scale radius used in the computation of the model.
    r_c (float,optional): core radius used in the computation of the model.
    nbins (int, optional): The number of radial grid points in the basis set. Default is 500.
    rmin (float, optional): The minimum radius of the basis set. Default is 0.61.
    rmax (float, optional): The maximum radius of the basis set. Default is 599.
    lmax (int, optional): The maximum harmonic order in the basis set. Default is 4.
    nmax (int, optional): The maximum number of polynomials in the basis set. Default is 20.
    scale (float, optional): The scale of the basis set in physical units.
    modelname (str, optional): The name of the file containing the density profile model. 
                               Default is 'dens_table.txt'.
    cachename (str, optional): The name of the file that will be used to cache the basis set. 
                               Default is '.slgrid_sph_cache'.
    coef_file (str, optional): The name of the file if provided that will be used to save the coef files as .h5.
                              Default is ''. 
    Returns:
    tuple: A tuple containing the basis and the coefficients of the expansion.
           The basis is an instance of pyEXP.basis.Basis, and the coefficients are 
           an instance of pyEXP.coefs.Coefs.
    """
    

    if log_space == True:
        rbins =  np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    elif log_space == False:
        rbins = np.linspace(rmin, rmax, nbins+1)

    if os.path.isfile(modelname) == False:
        print("-> File model not found so we are computing one \n")

        if basis_model == "empirical":
            print('-> Computing empirical model')
            #rho = empirical_density_profile(pos, mass, rbins)
            
            R, D, M, P = makemodel.makemodel(makemodel.empirical_density_profile, M=np.sum(mass),
                                   funcargs=[pos, mass], rvals=rbins)
        
        elif basis_model == "Hernquist":
            print('-> Computing analytical Hernquist model')
            R, D, M, P = makemodel.makemodel(makemodel.powerhalo, M=np.sum(mass),
                                             funcargs=[r_s, r_c, 1.0, 3.0], rvals = rbins,
                                             pfile=modelname)
        
        elif basis_model == "NFW":
            print('-> Computing analytical NFW model')
            R, D, M, P = makemodel.makemodel(makemodel.powerhalo, M=np.sum(mass),
                                             funcargs=[r_s, r_c, 1.0, 2.0], rvals = rbins,
                                             pfile=modelname)
        elif basis_model == "singlepowerlaw":
            print('-> Computing analytical Hernquist model') 
            R, D, M, P = makemodel.makemodel(makemodel.powerhalo, M=np.sum(mass),
                                             funcargs=[r_s, r_c, 2.5, 0.0], rvals = rbins,
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
        config = make_config(basis_id, nbins+1, R[0], R[-1], lmax, nmax, scale, 
                             modelname, cachename)

    # Construct the basis instances
    basis = pyEXP.basis.Basis.factory(config)

    # Prints info from Cache
    basis.cacheInfo(cachename)
    
    #compute coefficients
    if norm_mass_coef == True :
        coef = basis.createFromArray(mass/np.sum(mass), pos.T, time=time)
    elif norm_mass_coef == False : 
        coef = basis.createFromArray(mass, pos.T, time=time)

    coefs = pyEXP.coefs.Coefs.makecoefs(coef, 'dark halo')
    coefs.add(coef)
    if add_coef == False:
      coefs.WriteH5Coefs(coef_file)
    elif add_coef == True:
      coefs.ExtendH5Coefs(coef_file)
    
    return basis, coefs
