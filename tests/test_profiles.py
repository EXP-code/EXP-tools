import numpy as np
from EXPtools.utils import profiles

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
    NFW.power_halo()

    # Hernquist 
    Hernquist = profiles.Profiles(r, scale_length, alpha1, beta3)
    Hernquist.power_halo()

    # Single power law
    SPL = profiles.Profiles(r, scale_length, alpha2, beta0)
    SPL.power_halo()

    return None
