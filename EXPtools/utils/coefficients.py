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
            print(n[i],l[i],m[i],lm_idx)
            try: coefs_matrix[n[i], lm_idx, t] = np.complex128(0)
            except IndexError: continue
        copy_coeffcients.setMatrix(mat=coefs_matrix[:,:, t], time=t_snaps[t])
    
    return copy_coeffcients


# Plot phase of coefficients! 

# 
