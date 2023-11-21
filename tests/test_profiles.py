import numpy as np
from EXPtools.basis_builder import profiles

# Test output of functions
# Test document test functions 

def test_profile_class():
    """
    Density profiles tests
    """
    r = np.logspace(0, 3, 100)
    scale_length = 1

    alpha1 = 1
    beta2 = 2
    alpha2 = 2.5
    beta3 = 3
    beta0 = 0

    # NFW 
    NFW = profiles.Profiles(r, scale_length, alpha1, beta2)
    rho_NFW = NFW.powerhalo()

    # Hernquist 
    Hernquist = profiles.Profiles(r, scale_length, alpha1, beta3)
    rho_Hern = Hernquist.powerhalo()

    # Single power law
    SPL = profiles.Profiles(r, scale_length, alpha2, beta0)
    rho_SPL = SPL.powerhalo()

    return None
