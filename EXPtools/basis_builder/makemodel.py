import numpy as np

def write_table(tablename, radius, density, mass, potential, fmt="%.6e"):
    """
    Write a table of radius, density, mass, and potential values to a text file.

    Parameters
    ----------
    tablename : str
        Output filename.
    radius, density, mass, potential : array-like
        Arrays of physical quantities, all with the same length.
    fmt : str, optional
        Format string for numerical values. Defaults to scientific notation with 6 decimals.

    Notes
    -----
    Writes the table in the following format:
        ! <tablename>
        ! R    D    M    P
        <Nrows>
        <radius> <density> <mass> <potential>
    """
    # Convert inputs to NumPy arrays (for safety and performance)
    radius = np.asarray(radius)
    density = np.asarray(density)
    mass = np.asarray(mass)
    potential = np.asarray(potential)

    # Stack data into a single 2D array for fast writing
    data = np.column_stack((radius, density, mass, potential))

    header = f"! {tablename}\n! R    D    M    P\n{len(radius)}"
    np.savetxt(tablename, data, fmt=fmt, header=header, comments="")

def makemodel(radius, density, Mtotal, output_filename='', physical_units=False, verbose=True):
    """
    Generate an EXP-compatible spherical basis function table.

    Parameters
    ----------
    radius : array-like
        Radii at which the density values are evaluated.
    density : array-like
        Density values corresponding to radius.
    Mtotal : float
        Total mass of the model, used for normalization.
    output_filename : str, optional
        Name of the output file to save the table. If empty, no file is written.
    verbose : bool, optional
        If True, prints scaling information.

    Returns
    -------
    radius_scaled : ndarray
        Scaled radius values.
    density_scaled : ndarray
        Scaled density values.
    mass_scaled : ndarray
        Scaled enclosed mass values.
    potential_scaled : ndarray
        Scaled potential values.
    """
    Rmax = np.nanmax(radius)

    mass = np.zeros_like(density)
    pwvals = np.zeros_like(density)

    mass[0] = 1.e-15
    pwvals[0] = 0.

    #dr = radius[indx] - radius[indx - 1]
    dr = np.diff(radius)  # differences between consecutive radii

    # Midpoint mass contribution terms
    mass_contrib = 2.0 * np.pi * (
        radius[:-1]**2 * density[:-1] + radius[1:]**2 * density[1:]
    ) * dr

    pwvals_contrib = 2.0 * np.pi * (
        radius[:-1] * density[:-1] + radius[1:] * density[1:]
    ) * dr

    # Now cumulative sum to get the arrays
    mass = np.concatenate(([1e-15], 1e-15 + np.cumsum(mass_contrib)))
    pwvals = np.concatenate(([0.0], np.cumsum(pwvals_contrib)))
    
    potential = -mass / (radius + 1.e-10) - (pwvals[-1] - pwvals)

    M0 = mass[-1]
    R0 = radius[-1]

    Beta = (Mtotal / M0) * (R0 / Rmax)
    Gamma = np.sqrt((M0 * R0) / (Mtotal * Rmax)) * (R0 / Rmax)

    if verbose:
        print(f"! Scaling: R = {Rmax}  M = {Mtotal}")

    rfac = Beta**-0.25 * Gamma**-0.5
    dfac = Beta**1.5 * Gamma
    mfac = Beta**0.75 * Gamma**-0.5
    pfac = Beta

    if physical_units == True:
        rfac=1
        dfac=1
        mfac=1
        pfac=1

    if verbose:
        print(f"Scaling factors: rfac = {rfac}, dfac = {dfac}, mfac = {mfac}, pfac = {pfac}")

    if output_filename:
        write_table(
            output_filename,
            radius * rfac,
            density * dfac,
            mass * mfac,
            potential * pfac
        )

    return radius * rfac, density * dfac, mass * mfac, potential * pfac

