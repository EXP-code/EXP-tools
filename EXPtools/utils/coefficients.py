"""
Functionality to handle coefficients
"""

import numpy as np


def remove_terms(original_coefficients, n, l, m):
    """
    Remove coefficients 
    """ 
    assert len(l) == len(m) == len(n)
    copy_coeffcients = original_coefficients.deepcopy()
    coefs_matrix = original_coefficients.getAllCoefs()
    t_snaps = original_coefficients.Times()
    
    for t in range(len(t_snaps)):
        for i in range(len(l)):
            lm_idx = int(l[i]*(l[i]+1) / 2) + m[i]
            print(n[i],l[i],m[i], lm_idx)
            try: coefs_matrix[n[i], lm_idx, t] = np.complex128(0)
            except IndexError: continue
        copy_coeffcients.setMatrix(mat=coefs_matrix[:,:, t], time=t_snaps[t])
    
    return copy_coeffcients
    
def reorder_nlm(coefficients, nmax, lmax):
    """
    return coefficients in order (n, l, l+1)
    """
    new_order = np.zeros((2, nmax+1, lmax+1, lmax+1))
    coefs_matrix = coefficients
    for n in range(nmax+1):
        print(n)
        for l in range(lmax+1):
            for m in range(l+1):
                lm_idx = int(l*(l+1) / 2) + m
                new_order[0][n][l][m] = coefs_matrix[n, lm_idx].real
                new_order[1][n][l][m] = coefs_matrix[n, lm_idx].imag
    return new_order


# Plot phase of coefficients! 

# 
