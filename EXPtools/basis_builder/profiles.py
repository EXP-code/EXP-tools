import numpy as np
from scipy import special

Class Profiles:
    def __init__(self, radius, scale_radius, alpha=1., beta=2.0):
        """
        Class with density profiles of Dark Matter halos

        inputs
        ----------
        r      : (float) radius values
        rs     : (float, default=1.) scale radius 
        alpha  : (float, default=1.) inner halo slope
        beta   : (float, default=1.e-7) outer halo slope
        
        notes
        ----------
        different combinations are known distributions.
        alpha=1,beta=2 is NFW  (default)
        alpha=1,beta=3 is Hernquist
        alpha=2.5, beta=0 is a typical single power law halo

        """
        
        self.radius = radius
        self.scale_radius = scale_radius
        self.alpha = alpha
        self.beta = beta
        self.ra = self.radius/self.scale_radius


    def powerhalo(self, rc=0.0):
        """
        Generic two power law distribution
        
        inputs
        ----------
        rc     : (float, default=0. i.e. no core) core radius
        
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
        return 1./(((self.ra+rc/self.scale_radius)**self.alpha)*((1+self.ra)**self.beta))
        
    
    def powerhalorolloff(self, rc=0.0):
        """
        Generic twopower law distribution with an erf rolloff
        
        
        inputs
        ----------
        rc     : (float, default=0. i.e. no core) core radius
        
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
        dens = 1./(((self.ra+rc/self.scale_radius)**self.alpha)*((1+self.ra)**self.beta))
        rtrunc = 25*self.scale_radius 
        wtrunc = rtrunc*0.2
        rolloff = 0.5 - 0.5*special.erf((self.radius-rtrunc)/wtrunc)
        return dens*rolloff


    def plummer_density(self):
        """
        Plummer density profile

        inputs
        ---------

        
        returns
        ----------
        densities evaluated at r
        """
        return ((3.0)/(4*np.pi))*(self.scale_radius**2.)*((self.scale_radius**2 + self.radius**2)**(-2.5))

    def twopower_density_withrolloff(self, rcen, wcen):
        """
        inputs
        ---------

        
        returns
        ----------
        densities evaluated at r
        """
        
        
        prefac = 0.5*(1.-scipy.special.erf((ra-rcen)/wcen))
        return prefac*(ra**-self.alpha)*(1+ra)**(-self.beta+self.alpha)

    def hernquist_halo(self):
        """
        TODO: DO we need these profile
        Hernquist halo
        
        inputs
        ---------
        
        returns
        ----------
        densities evaluated at r
        """
        return 1 / ( 2*np.pi * (self.radius/self.scale_radius) * (1 + self.radius/self.scale_radius)**3)