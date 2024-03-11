import numpy as np

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

def makemodel(basis_type, func, dvals, rvals, M, 
        unit_physical=False, return_values=False,
        outfile='', plabel = '', verbose=True):

    """
    Make an EXP-compatible spherical basis function table
    
    Parameters
    ----------
    basis_type: str
        empirical, analytical
    func: function (optional)
        the callable functional form of the density
    M: float 
        the total mass of the model, sets normalizations
    funargs: list (optional)
        a list of arguments for the density function.
    rvals: array of floats
        radius values to evaluate the density function
                     = 10.**np.linspace(-2.,4.,2000)
    pfile: string
        the name of the output file. If '', will not print file
    plabel: (string)
        comment string
    verbose: boolean

    Returns
    -------
    R: array of floats 
        Radius values
    D: array of floats
        Density profile
    M: array of floats
        Enclosed mass profile
    P: array of floats
        Potential profile
    
    """
    
    R = np.nanmax(rvals)
    
    # query out the density values


    #vals = empirical_density_profile(rvals)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)



    # initialize the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1, dvals.size):
        mvals[indx] = mvals[indx-1] + 2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] + rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1])

        pwvals[indx] = pwvals[indx-1] + 2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
    
    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=", R,"  M=", M)

    rfac = np.power(Beta, -0.25) * np.power(Gamma, -0.5)
    dfac = np.power(Beta, 1.5) * Gamma
    mfac = np.power(Beta, 0.75) * np.power(Gamma, -0.5)
    pfac = Beta

    if verbose:
        print(rfac, dfac, mfac, pfac)

    # save file if desired
    if outfile != '':
        f = open(outfile,'w')
        print('! ', plabel, file=f)
        print('! R    D    M    P', file=f)

        print(rvals.size, file=f)

        if unit_physical == True:
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rvals[indx],
                                               dvals[indx],
                                               mvals[indx],
                                               pvals[indx]),file=f)
        else: 
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rvals[indx],
                                               dvals[indx]*dfac,
                                               mvals[indx]*mfac,
                                               pvals[indx]*pfac),file=f)
    
        f.close()
    
    if return_values==True:
        if unit_physical==True:
            return rvals, dvals, mvals, pvals
        else:
            return rvals*rfac, dfac*dvals, mfac*mvals, pfac*pvals
      
