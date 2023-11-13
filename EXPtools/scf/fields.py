#!/usr/bin/env python3.8
"""
Script that computes parallel potential using the BFE formalism as outlined
in Ben Lowing et al+11 equation 14 (MNRAS 416, 2697-2711).


"""

import numpy as np
import sys
from scipy import special
import math
import time

class SCFfields:
    def __init__(self, pos, S, T, rs, nmax, lmax, G, M):
        """
        Computes parallel BFE potential and density
        Attributes:
            nlm_list     Creates arrays of indices n, l, m from 3d to 1d.
            bfe_pot      Core function that computed the BFE potential.
            potential    Computes the potential bfe_pot by receiving a task
                         with the arguments that are going to run in parallel.
            main         Runs potential in parallel using a pool to be defined
                         by the user.

        Parameters:
        -----------
        pos : numpy.ndarray with shape
        S : numpy.ndarray
        T : numpy.ndarray
        rs : float
            Hernquist halo scale length
        nmax : int
            nmax in the expansion
        lmax : int
            lmax in the expansion
        G : float
            Value of the gravitational constant
        M : float
            Total mass of the halo (M=1) if the masses
            of each particle where already used for computing the
            coefficients.

        """
        self.pos = pos
        self.rs = rs
        self.nmax = nmax
        self.lmax = lmax
        self.r = (self.pos[:,0]**2 + self.pos[:,1]**2 + self.pos[:,2]**2)**0.5
        self.theta = np.arccos(self.pos[:,2]/self.r) # cos(theta)

        self.phi = np.arctan2(self.pos[:,1], self.pos[:,0])
        self.s = self.r/self.rs
        self.G = G
        self.M = M
        self.S = S
        self.T = T
        self.nparticles = len(self.s)

    def nlm_list(self, ncoeff, nmax, lmax):
        n_list = np.zeros(ncoeff)
        l_list = np.zeros(ncoeff)
        m_list = np.zeros(ncoeff)
        i=0
        for n in range(nmax+1):
            for l in range(lmax+1):
                for m in range(l+1):
                    n_list[i] = n
                    l_list[i] = l
                    m_list[i] = m
                    i+=1
        return n_list, l_list, m_list

    def bfe_pot(self, n, l, m, s, theta):
        """
        Check this! 
        """
        #r, theta, phi = spherical_coordinates(pos)

        phi_l = -s**l * (1+s)**(-2*l-1)
        #phi_nlm = special.eval_gegenbauer(n, 2*l+1.5, (s-1)/(s+1)) * special.sph_harm(m, l, 0, theta)
        phi_nlm = special.eval_gegenbauer(n, 2*l+1.5, ((s-1)/(s+1))) * special.lpmv(m, l, np.cos(theta))
        factor = ((2*l+1) * math.factorial(l-m)/math.factorial(l+m))**0.5
        #factor=1
        #factor = 1#(4*np.pi)**0.5
        return  factor*phi_l*phi_nlm #np.sqrt(4*np.pi)


    def potential_nlm(self, n, l, m):
        pot_nlm = self.bfe_pot(n, l, m, self.s, self.theta)\
                * (self.S[n,l,m]*np.cos(m*self.phi)+self.T[n,l,m]*np.sin(m*self.phi))
        return pot_nlm*self.G*self.M/self.rs

    def bfe_rho(self, n, l, m, s, theta):
        # Eq 10 in Lowing+11

        rho_l = s**(l-1) * (1+s)**(-2*l-3)
        rho_nlm = special.eval_gegenbauer(n, 2*l+1.5, ((s-1)/(s+1))) * special.lpmv(m, l, np.cos(theta))
        factor = ((2*l+1) * math.factorial(l-m)/math.factorial(l+m))**0.5
        #factor=1
        Knl = 0.5*n*(n+4*l+3) + (l+1)*(2*l+1)
        return  factor*Knl*rho_l*rho_nlm/(2*np.pi)

    def density_nlm(self, n, l, m):
        rho = self.bfe_rho(n, l, m, self.s, self.theta)\
                * (self.S[n,l,m]*np.cos(m*self.phi)+self.T[n,l,m]*np.sin(m*self.phi))
        return rho*self.M/self.rs

    def d_phi(self, n, l, m):
        """
        TODO: Add all the .self
        """
        factor = np.sqrt(4*np.pi)
        A_factor = (-l-3*l*self.s-self.s)
        A = self.s**(l-1)/(1+self.s)**(2*l+2) * special.eval_gegenbauer(n, 2*l+1.5, ((self.s-1)/(self.s+1)))
        B_factor = -4*(2*l+3/2.)
        B = self.s**l/(1+self.s)**(2*l+3) * special.eval_gegenbauer(n-1, 2*l+2.5, ((self.s-1)/(self.s+1)))
        d_phi_dr = factor * ( A_factor*A + B_factor*B) * special.lpmv(m, l, np.cos(self.theta))
        d_phi_dtheta = m*np.cot(self.theta)*np.sqrt()

    def acceleration(self, n, l, m, s, theta):
        # Computes bfe accelerations Eqn: 17-19 in Lowing+11

        special.lpmv(m, l, np.cos(theta))
        return 0



