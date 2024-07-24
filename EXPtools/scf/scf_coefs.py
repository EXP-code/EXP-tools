"""
routines to translate coefficients between agama, gala, EXP, galpy

"""

import numpy as np
from scipy import special
import pyEXP


def agama_to_gala(agama_coefs, nmax, lmax):
    """
    Translate AGAMA scf coefficients to Gala 
    Adapted from: https://github.com/GalacticDynamics-Oxford/Agama/blob/master/py/example_basis_set.py

    """
    
    Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Tnlm = np.zeros((nmax+1, lmax+1, lmax+1))
    for n in range(nmax+1):
        for l in range(lmax+1):
            j=1 # j=0 is the value of n order
            for m in range(l):
                Tnlm[n, l, l-m] = agama_coefs[n, j]*2**0.5
                j+=1
            for m in range(l+1):
                #if m<0:
                Snlm[n, l, m] = agama_coefs[n, j]*(2**0.5 if m>0 else 1)
                j+=1
    return Snlm, Tnlm

    
    return coefs


def exp(config):
    return 0

class Coeff_properties:
    def __init__(self, S, T, nmax, lmax):
        self.S = S
        self.T = T
        self.nmax = nmax
        self.lmax = lmax
        
    def Anl(self, n, l):
        knl = (0.5*n*(n+4*l+3)) + ((l+1)*(2*l+1))
        A_nl = - (2**(8*l+6)/(4*np.pi*knl)) *  ((special.factorial(n)*(n+2*l+3/2.)*(special.gamma(2*l+3/2.))**2)/(special.gamma(n+4*l+3)))
        return A_nl

    def Anl_array(self):
        A_nl_array = np.zeros((self.nmax, self.lmax))
        for n in range(self.nmax):
            for l in range(self.lmax):
                A_nl_array[n][l] = self.Anl(n, l)
        return A_nl_array

    def coeff_energy(self,  m):
        A_nl = self.Anl_array()
        if m==0:
            A = (self.S[:,:,m]**2 + self.T[:,:,m]**2)/A_nl
        else:
            A = (self.S[:,:,m]**2 + self.T[:,:,m]**2)/(2*A_nl)
        return A
        
    def U_all(self): 
        """
        
        """
        U_all = np.zeros((self.nmax, self.lmax, self.lmax))
        for i in range(self.lmax):
            U_all[:,:,i] = self.coeff_energy(i)
        return U_all
       
    def energy_nlm(self):
        """

        """
        U = self.U_all()
        U_l = np.zeros(self.lmax)
        U_n = np.zeros(self.nmax)
        U_m = np.zeros(self.lmax)
        for i in range(0, self.lmax):
            U_l[i] = np.sum(U[:,i,:])
            U_m[i] = np.sum(U[:,:,i])
            
        for i in range(0, self.nmax):
            U_n[i] = np.sum(U[i,:,:])
        
        return U_n, U_l, U_m
