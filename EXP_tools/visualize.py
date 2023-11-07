import os, sys, pickle, pyEXP
import numpy as np
import makemodel
from makemodel import hernquist_halo


###Field computations for plotting###
def make_basis_plot(basis, savefile=None, nsnap='mean', y=0.92, dpi=200):
    """
    Plots the potential of the basis functions for different values of l and n.

    Args:
    basis (obj): object containing the basis functions for the simulation
    savefile (str, optional): name of the file to save the plot as
    nsnap (str, optional): description of the snapshot being plotted
    y (float, optional): vertical position of the main title
    dpi (int, optional): resolution of the plot in dots per inch

    Returns:
    None

    """
    # Set up grid for plotting potential
    lrmin, lrmax, rnum = 0.5, 2.7, 100
    halo_grid = basis.getBasis(lrmin, lrmax, rnum)
    r = np.linspace(lrmin, lrmax, rnum)
    r = np.power(10.0, r)

    # Create subplots and plot potential for each l and n
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 5, figsize=(6,6), dpi=dpi, 
                            sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax = ax.flatten()

    for l in range(len(ax)):
        ax[l].set_title(f"$\ell = {l}$", y=0.8, fontsize=6)    
        for n in range(20):
            ax[l].semilogx(r, halo_grid[l][n]['potential'], '-', label="n={}".format(n), lw=0.5)

    # Add labels and main title
    fig.supylabel('Potential', weight='bold', x=-0.02)
    fig.supxlabel('Radius', weight='bold', y=0.02)
    fig.suptitle(f'nsnap = {nsnap}', 
                 fontsize=12, 
                 weight='bold', 
                 y=y,
                )
    
    # Save plot if a filename was provided
    if savefile:
        plt.savefig(f'{savefile}', bbox_inches='tight')

def find_field(basis, coefficients, time=0, xyz=(0, 0, 0), property='dens', include_monopole=True):
    """
    Finds the value of the specified property of the field at the given position.

    Args:
    basis (obj): Object containing the basis functions for the simulation.
    coefficients (obj): Object containing the coefficients for the simulation.
    time (float, optional): The time at which to evaluate the field. Default is 0.
    xyz (tuple or list, optional): The (x, y, z) position at which to evaluate the field. Default is (0, 0, 0).
    property (str, optional): The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.
    include_monopole (bool, optional): Whether to return the monopole contribution to the property only. Default is True.

    Returns:
    float or list: The value of the specified property of the field at the given position. If property is 'force', a list of
    three values is returned representing the force vector in (x, y, z) directions.

    Raises:
    ValueError: If the property argument is not 'dens', 'pot', or 'force'.
    """

    coefficients.set_coefs(coefficients.getCoefStruct(time))
    dens0, pot0, dens, pot, fx, fy, fz = basis.getFields(xyz[0], xyz[1], xyz[2])
    
    if property == 'dens':
        if include_monopole:
            return dens0
        return dens + dens0

    elif property == 'pot':
        if include_monopole:
            return pot0
        return pot + pot0

    elif property == 'force':
        return [fx, fy, fz]

    else:
        raise ValueError("Invalid property specified. Possible values are 'dens', 'pot', and 'force'.")
    
def spherical_avg_prop(basis, coefficients, time=0, radius=np.linspace(0.1, 600, 100), property='dens'):
    """
    Computes the spherically averaged value of the specified property of the field over the given radii.

    Args:
    basis (obj): Object containing the basis functions for the simulation.
    coefficients (obj): Object containing the coefficients for the simulation.
    time (float, optional): The time at which to evaluate the field. Default is 0.
    radius (ndarray, optional): An array of radii over which to compute the spherically averaged property. Default is an
        array of 100 values logarithmically spaced between 0.1 and 600.
    property (str, optional): The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.

    Returns:
    ndarray: An array of spherically averaged values of the specified property over the given radii.

    Raises:
    ValueError: If the property argument is not 'dens', 'pot', or 'force'.
    """

    coefficients.set_coefs(coefficients.getCoefStruct(time))
    field = [find_field(basis, np.hstack([[rad], [0], [0]]), property=property, include_monopole=True) for rad in radius]

    if property == 'force':
        return np.vstack(field), radius

    return np.array(field), radius


def slice_fields(basis, coefficients, time=0, 
                 projection='XY', proj_plane=0, npoints=300, 
                 grid_limits=(-300, 300), prop='dens', monopole_only=False):
    """
    Plots a slice projection of the fields of a simulation.

    Args:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    time (float): the time at which to plot the fields
    projection (str): the slice projection to plot. Can be 'XY', 'XZ', or 'YZ'.
    proj_plane (float, optional): the value of the coordinate that is held constant in the slice projection
    npoints (int, optional): the number of grid points in each dimension
    grid_limits (tuple, optional): the limits of the grid in the x and y dimensions, in the form (x_min, x_max)
    prop (str, optional): the property to return. Can be 'dens' (density), 'pot' (potential), or 'force' (force).
    monopole_only (bool, optional): whether to return the monopole component in the returned property value.

    Returns:
    array or list: the property specified by `prop`. If `prop` is 'force', a list of the x, y, and z components of the force is returned.
                    Also returns the grid used to compute slice fields. 
    """
    x = np.linspace(grid_limits[0], grid_limits[1], npoints)
    xgrid = np.meshgrid(x, x)
    xg = xgrid[0].flatten()
    yg = xgrid[1].flatten()

    
    if projection not in ['XY', 'XZ', 'YZ']:
        raise ValueError("Invalid projection specified. Possible values are 'XY', 'XZ', and 'YZ'.")

    N = len(xg)
    rho0 = np.zeros_like(xg)
    pot0 = np.zeros_like(xg)
    rho = np.zeros_like(xg)
    pot = np.zeros_like(xg)
    basis.set_coefs(coefficients.getCoefStruct(time))

    for k in range(0, N):
        if projection == 'XY':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(xg[k], yg[k], proj_plane)
        elif projection == 'XZ':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(xg[k], proj_plane, yg[k])
        elif projection == 'YZ':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(proj_plane, xg[k], yg[k])
    
    dens = rho.reshape(npoints, npoints)
    pot = pot.reshape(npoints, npoints)
    dens0 = rho0.reshape(npoints, npoints)
    pot0 = pot0.reshape(npoints, npoints)

    if prop == 'dens':
        if monopole_only:
            return dens0
        return dens0, dens, xgrid

    if prop == 'pot':
        if monopole_only:
            return pot0
        return pot0, pot, xgrid

    if prop == 'force':
        return [fx.reshape(npoints, npoints), fy.reshape(npoints, npoints), fz.reshape(npoints, npoints)], xgrid
    

def slice_3d_fields(basis, coefficients, time=0, 
                 projection='XY', proj_plane=0, npoints=300, 
                 grid_limits=(-300, 300), prop='dens', monopole_only=False):
    """
    Plots a slice projection of the fields of a simulation.

    Args:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    time (float): the time at which to plot the fields
    projection (str): the slice projection to plot. Can be 'XY', 'XZ', or 'YZ'.
    proj_plane (float, optional): the value of the coordinate that is held constant in the slice projection
    npoints (int, optional): the number of grid points in each dimension
    grid_limits (tuple, optional): the limits of the grid in the x and y dimensions, in the form (x_min, x_max)
    prop (str, optional): the property to return. Can be 'dens' (density), 'pot' (potential), or 'force' (force).
    monopole_only (bool, optional): whether to return the monopole component in the returned property value.

    Returns:
    array or list: the property specified by `prop`. If `prop` is 'force', a list of the x, y, and z components of the force is returned.
                    Also returns the grid used to compute slice fields. 
    """
    x = np.linspace(grid_limits[0], grid_limits[1], npoints)
    xgrid = np.meshgrid(x, x, x)
    xg = xgrid[0].flatten()
    yg = xgrid[1].flatten() 
    zg = xgrid[2].flatten()
  
    

    N = len(xg)
    rho0 = np.zeros_like(xg)
    pot0 = np.zeros_like(xg)
    rho = np.zeros_like(xg)
    pot = np.zeros_like(xg)
    basis.set_coefs(coefficients.getCoefStruct(time))

    for k in range(0, N):
        rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(xg[k], yg[k], zg[k])
    
    dens = rho.reshape(npoints, npoints, npoints)
    pot = pot.reshape(npoints, npoints, npoints)
    dens0 = rho0.reshape(npoints, npoints, npoints)
    pot0 = pot0.reshape(npoints, npoints, npoints)

    if prop == 'dens':
        if monopole_only:
            return dens0
        return dens0, dens, xgrid

    if prop == 'pot':
        if monopole_only:
            return pot0
        return pot0, pot, xgrid

    if prop == 'force':
        return [fx.reshape(npoints, npoints, npoints), fy.reshape(npoints, npoints, npoints), fz.reshape(npoints, npoints, npoints)], xgrid
    