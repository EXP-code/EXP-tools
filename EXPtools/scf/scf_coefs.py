"""
routines to translate coefficients between agama, gala, EXP, galpy

"""

import numpy as np
import pyEXP



def agama_to_gala(agama_coefs):
    """

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



