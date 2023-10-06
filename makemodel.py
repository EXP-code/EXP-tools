import numpy as np

def makemodel(func,M,funcargs,rvals = 10.**np.linspace(-2.,4.,2000),pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table
    
    inputs
    -------------
    func        : (function) the callable functional form of the density
    M           : (float) the total mass of the model, sets normalisations
    funcargs    : (list) a list of arguments for the density function.
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string
    verbose     : (boolean)

    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential
    
    """
    
    R = np.nanmax(rvals)
    
    # query out the density values
    dvals = func(rvals,*funcargs)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
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
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)
    
        f.close()
    
    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals



def powerhalo(r,rs=1.,rc=0.,alpha=1.,beta=1.e-7):
    """return generic twopower law distribution
    
    inputs
    ----------
    r      : (float) radius values
    rs     : (float, default=1.) scale radius 
    rc     : (float, default=0. i.e. no core) core radius
    alpha  : (float, default=1.) inner halo slope
    beta   : (float, default=1.e-7) outer halo slope
    
    returns
    ----------
    densities evaluated at r
    
    notes
    ----------
    different combinations are known distributions.
    alpha=1,beta=2 is NFW
    alpha=1,beta=3 is Hernquist
    alpha=2.5,beta=0 is a typical single power law halo
    
    
    """
    ra = r/rs
    return 1./(((ra+rc/rs)**alpha)*((1+ra)**beta))
    
 
def powerhalorolloff(r,rs=1.,rc=0.,alpha=1.,beta=1.e-7):
    """return generic twopower law distribution with an erf rolloff
    
    inputs
    ----------
    r      : (float) radius values
    rs     : (float, default=1.) scale radius 
    rc     : (float, default=0. i.e. no core) core radius
    alpha  : (float, default=1.) inner halo slope
    beta   : (float, default=1.e-7) outer halo slope
    
    returns
    ----------
    densities evaluated at r
    
    notes
    ----------
    different combinations are known distributions.
    alpha=1,beta=2 is NFW
    alpha=1,beta=3 is Hernquist
    alpha=2.5,beta=0 is a typical single power law halo
    
    
    """
    ra = r/rs
    dens = 1./(((ra+rc/rs)**alpha)*((1+ra)**beta))
    rtrunc = 25*rs
    wtrunc = rtrunc*0.2
    rolloff = 0.5 - 0.5*special.erf((r-rtrunc)/wtrunc)
    return dens*rolloff


def plummer_density(radius,scale_radius=1.0,mass=1.0,astronomicalG=1.0):
    """basic plummer density profile"""
    return ((3.0*mass)/(4*np.pi))*(scale_radius**2.)*((scale_radius**2 + radius**2)**(-2.5))

def twopower_density_withrolloff(r,a,alpha,beta,rcen,wcen):
    """a twopower density profile"""
    ra = r/a
    prefac = 0.5*(1.-scipy.special.erf((ra-rcen)/wcen))
    return prefac*(ra**-alpha)*(1+ra)**(-beta+alpha)

def hernquist_halo(r, a):
    return 1 / ( 2*np.pi * (r/a) * (1 + r/a)**3)


"""
plummer_b = 1.0
R,D,M,P = makemodel(plummer_density,1.,[plummer_b],rvals = 10.**np.linspace(-3.,1.,2000))

alpha=1
beta=2
concentration = 15.
rs = 1/concentration
rc = 0.0001
R,D,M,P = makemodel(powerhalo,1.,[rs,rc,alpha,beta],rvals = 10.**np.linspace(-5.,0.3,2000)pfile='NFWModelc{}.txt'.format(concentration),plabel = 'NFWModelc{}.txt'.format(concentration))
"""