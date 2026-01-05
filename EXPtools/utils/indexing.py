"""
This module provides functions to work with index-based calculations for spherical harmonics coef series.
"""
import numpy as np, math

def total_terms(l_max):
    """
    Calculate the total number of terms up to a given maximum angular number.

    Parameters:
        l_max (int): The maximum angular quantum number.

    Returns:
        float: The total number of terms.
    """
    return l_max * (l_max + 1) / 2 + l_max + 1  # Sigma(lmax+1) + 1 (the extra one for the 0th term)

def I(l, m):
    """
    Calculate the index of a spherical harmonic element given the angular numbers l and m .

    Parameters:
        l (int): The angular number.
        m (int): The magnetic quantum number, ranging from 0 to l.

    Returns:
        int: The index corresponding to the specified angular numbers.
    """
    import math
    assert isinstance(l, int) and isinstance(m, int), "l and m must be integers"
    assert l >= 0, "l must be greater than 0"
    assert abs(m) <= l, "m must be less than or equal to l"
    return int(l * (l + 1) / 2) + abs(m)

def I_nlm(n, l, m, lmax):
    """
    Calculate the index of a spherical harmonic element given the angular numbers n, l, and m .

    Parameters
    ----------
        n : int 
            The radial number
        l : int
            The angular number
        m : int 
            The magnetic quantum number, ranging from 0 to l.
        lmax : int 
            The maximum value of l in the basis

    Returns:
    --------
        idx : int 
            The index corresponding to the specified angular numbers.
    """
    assert isinstance(n, int) and isinstance(l, int) and isinstance(m, int), "all args must be integers"
    assert n >= 0, "n must be greater than 0"
    assert l >= 0, "l must be greater than 0"
    assert abs(m) <= l, "m must be less than or equal to l"

    # determine the index of the n=n, l=0, m=0 term
    n_idx = int(total_terms(lmax)*n)

    # add the number of indices to reach the desired l,m
    return I(l, m) + n_idx

def inverse_I(I):
    """
    Calculate the angular numbers l and m given the index of a spherical harmonic element.

    Parameters:
        I (int): The index of the spherical harmonic element.

    Returns:
        tuple: A tuple containing the angular numbers (l, m).
    """
    import math
    assert isinstance(I, int) and I >=0, "I must be an interger greater than or equal to 0"
    l = math.floor((-1 + math.sqrt(1 + 8 * I)) / 2)  # Calculate l using the inverse of the formula
    m = I - int(l * (l + 1) / 2)  # Calculate m using the given formula
    return l, m

def inverse_Inlm(I, lmax):
    """
    Calculate the angular numbers n, l, and m, given the index of a spherical harmonic element.
    
    Parameters
    ----------
        I : int 
            The index of the spherical harmonic element
        lmax : int 
            The maximum value of l in the basis

    Returns:
    --------
        tuple 
            A 3-tuple containing the angular numbers (n, l, m)
    """
    assert isinstance(I, int) and I >=0, "I must be an interger greater than or equal to 0"
    terms_per_n = int(total_terms(l_max=lmax))
    n = I//terms_per_n
    Irem = I%terms_per_n
    l,m = inverse_I(Irem)
    return n, l, m

def set_specific_lm_non_zero(data, lm_pairs_to_set):
    """
    Sets specific (l, m) pairs in the input data to non-zero values.

    Parameters:
        data (np.ndarray): Input data, an array of complex numbers.
        lm_pairs_to_set (list): List of tuples representing (l, m) pairs to set to non-zero.

    Returns:
        np.ndarray: An array with selected (l, m) pairs set to non-zero values.

    Raises:
        ValueError: If any of the provided (l, m) pairs are out of bounds for the input data.
    """
    assert isinstance(lm_pairs_to_set, list), "lm_pairs_to_set must be a list"
    for pair in lm_pairs_to_set:
        assert isinstance(pair, tuple), "Each element in lm_pairs_to_set must be a tuple"
        assert len(pair) == 2, "Each tuple in lm_pairs_to_set must contain two elements"
        assert all(isinstance(x, int) for x in pair), "Each element in each tuple must be an integer"
    
    # Get a zeros array of the same shape as the input data
    arr_filt = np.zeros(data.shape, dtype=complex)

    # Check if the provided (l, m) pairs are within the valid range
    for l, m in lm_pairs_to_set:
        if l < 0 or l >= data.shape[0] or m < 0 or m > l:
            raise ValueError(f"Invalid (l, m) pair: ({l}, {m}). Out of bounds for the input data shape.")
        arr_filt[I(l, m), :, :] = data[I(l, m), :, :] 

    return arr_filt

def listStates(n, lmax, allStates=True):
    """
    List all (n, l, m) combinations either up to or for
    a given value of n and lmax

    Parameters
    ----------
    n : int
        Radial number
    lmax : int
        The maximum value of l in the basis
    allStates : bool, optional
        If True, list (n,l,m) combinations for the
        the specified value of n only, by default True

    Returns
    -------
    list
        List of 3-tuples of the combinations, represented
        as (n,l,m)
    """
    if allStates==False:
        nmin, nmax = n, n+1
    else:
        nmin, nmax = 0, n+1
    states = [(n,l,m) for n in range(nmin, nmax) 
                       for l in range(0, lmax +  1) 
                         for m in range(0, lmax + 1) if m<=l]
    return states
