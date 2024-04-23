import os,  sys, pickle, pyEXP
import numpy as np
import matplotlib.pyplot as plt

def make_basis_plot(basis, savefile=None, nsnap='mean', y=0.92, dpi=200):
    """
    Plots the potential of the basis functions for different values of l and n.

    Parameters
    ----------
    basis: object
        object containing the basis functions for the simulation
        savefile (str, optional): name of the file to save the plot as
        nsnap (str, optional): description of the snapshot being plotted
    y: float (optional
        vertical position of the main title 
    dpi: int (optional)
        resolution of the plot in dots per inch

    Returns
    -------
    None
        None

    """
    # Set up grid for plotting potential
    lrmin, lrmax, rnum = 0.5, 2.7, 100
    halo_grid = basis.getBasis(lrmin, lrmax, rnum)
    r = np.linspace(lrmin, lrmax, rnum)
    r = np.power(10.0, r)

    # Create subplots and plot potential for each l and n
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

    Parameters
    ----------
        basis: object 
            Object containing the basis functions for the simulation.
        coefficients: oject 
            Object containing the coefficients for the simulation.
        time: float (optional) 
            The time at which to evaluate the field. Default is 0.
        xyz: tuple or list (optional) 
            The (x, y, z) position at which to evaluate the field. Default is (0, 0, 0).
        property: str (optional) 
            The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.
        include_monopole: bool (optional) 
            Whether to return the monopole contribution to the property only. Default is True.

    Returns
    -------
        field: float or list
            The value of the specified property of the field at the given position. If property is 'force', a list of three values is returned representing the force vector in (x, y, z) directions.

    Raises
    ------
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

    Parameters
    ----------
    basis: object
        bject containing the basis functions for the simulation.
    coefficients: object 
        bject containing the coefficients for the simulation.
    time: float (optional) 
        The time at which to evaluate the field. Default is 0.
    radius: ndarray (optional) 
        An array of radii over which to compute the spherically averaged property. Default is an array of 100 values logarithmically spaced between 0.1 and 600.
    property:  str (optional)
        The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.

    Returns
    -------
    field_r: Array  
        An array of spherically averaged values of the specified field over the given radii.

    radius: array
        Radius at where the field is evaluated
    Raises:
    ValueError: 
        If the property argument is not 'dens', 'pot', or 'force'.
    """

    coefficients.set_coefs(coefficients.getCoefStruct(time))
    field = [find_field(basis, np.hstack([[rad], [0], [0]]), property=property, include_monopole=True) for rad in radius]

    if property == 'force':
        return np.vstack(field), radius

    return np.array(field), radius


def make_grid(gridtype, gridspecs, rgrid, representation='cartesian'):
    """
    Make a variety of grids in different coordinate representations

    Parameters
    ----------
    gridtype:
    npoints:
    rgrid:
    representation:


    Returns:
    --------
    coordinates 
    
    """
    
    if gridtype == 'spherical':
        arcostheta = np.linspace(-1, 1, gridspecs['theta_bins'])
        phi = np.linspace(-np.pi, np.pi, gridspecs['phi_bins'])
        theta_mesh, phi_mesh = np.meshgrid(np.arccos(arcostheta)-np.pi/2., phi)

        if representation == 'cartesian':
            x = rgrid  * np.sin(theta_mesh) * np.cos(phi_mesh)
            y = rgrid  * np.sin(theta_mesh) * np.sin(phi_mesh)
            z = rgrid  * np.cos(theta_mesh) 
            ## TODO: fix this return to allow the user to chose what to return
            return np.array([x, y, z]), theta_mesh, phi_mesh
        
        elif representation == 'spherical':
            return np.array([theta_mesh, phi_mesh])
  
    else:
        print('gridtype {} not implemented'.format(gridtype))  
   





def return_fields_in_grid(basis, coefficients, times=[0], 
                       projection='3D', proj_plane=0, 
                       grid_lim=300, npoints=150,):

    """
    Returns the 2D or 3D field grids as a dict at each time step as a key.

    Args:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    times (list): list of the time to compute the field grid.
    projection (str): the slice projection to plot. Can be '3D', XY', 'XZ', or 'YZ'.
    proj_plane (float, optional): the value of the coordinate that is held constant in the 2D slice projection.
    npoints (int, optional): the number of grid points in each dimension. It's +1 in 3D projection.
    grid_limits (float, optional): the limits of the grid in the dimensions.

    Returns:
    heirarchical dict structure with for each time:
        dict_keys(['d', 'd0', 'd1', 'dd', 'fp', 'fr', 'ft', 'p', 'p0', 'p1'])
        where 
        - 'd': Total density at each point.
        - 'd0' : Density in the monopole term.
        - 'd1' : Density in l>0 terms.
        - 'dd': Density difference.
        - 'fp': Force in the phi direction.
        - 'fr': Force in the radial direction.
        - 'ft': Force in the theta direction.
        - 'p': Total potential at each point.
        - 'p0': Potential in the monopole term.
        - 'p1': Potential in l>0 terms.
        
    xgrid used to make the fields. 
    """

    assert projection in ['3D', 'XY', 'XZ', 'YZ'], "Invalid value for 'projection'. Must be one of '3D', 'XY', 'XZ', or 'YZ'."
    
    if projection == '3D':
        pmin  = [-grid_lim, -grid_lim, -grid_lim]
        pmax  = [grid_lim, grid_lim, grid_lim]
        grid  = [npoints, npoints, npoints]
        
        ##create a projection grid along with it
        x = np.linspace(-grid_lim, grid_lim, npoints+1) ##pyEXP creates a grid of npoints+1 for 3D spacing.
        xgrid = np.meshgrid(x, x, x)
        
        field_gen = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)
        return field_gen.volumes(basis, coefficients), xgrid

    else:
        x = np.linspace(-grid_lim, grid_lim, npoints)
        xgrid = np.meshgrid(x, x)
                
        if projection == 'XY':        
            pmin  = [-grid_lim, -grid_lim, proj_plane]
            pmax  = [grid_lim, grid_lim, proj_plane]
            grid  = [npoints, npoints, 0]
            
        elif projection == 'XZ':
            pmin  = [-grid_lim,  proj_plane, -grid_lim]
            pmax  = [grid_lim, proj_plane, grid_lim]
            grid  = [npoints, 0, npoints]

        elif projection == 'YZ':
            pmin  = [proj_plane,  -grid_lim, -grid_lim]
            pmax  = [proj_plane, grid_lim, grid_lim]

    
    field_gen = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)

    return field_gen.volumes(basis, coefficients), xgrid


def spherical_slice(basis, coefficients, gridspecs, rgrid):
    """
    Compute fields in a a spherical grid

    Parameters:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    gridspces (dict): Dictionary specifying the number of bins in theta_bins and phi_bins as integers
    rgrid (float): radius where to build the spherical grid

    return:
    fields evaluated in the spherical grid. Dens, Dens0, phi, pho0
    spherical grid, theta, phi
    
    """

    grid, theta_grid, phi_grid = make_grid(gridtype='spherical', gridspecs=gridspecs, 
                                    rgrid=rgrid, representation='cartesian')
    
    xg = grid[0].flatten()
    yg = grid[1].flatten()
    zg = grid[2].flatten()

    theta_bins = gridspecs['theta_bins']
    phi_bins = gridspecs['phi_bins']

    ngrid = int(theta_bins*phi_bins)
    assert len(xg) == ngrid


    rho0 = np.zeros_like(xg)
    pot0 = np.zeros_like(xg)
    rho = np.zeros_like(xg)
    pot = np.zeros_like(xg)
    fx = np.zeros_like(xg)
    fy = np.zeros_like(xg)
    fz = np.zeros_like(xg)
    
    for k in range(ngrid):
        rho0[k], pot0[k], rho[k], pot[k], fx[k], fy[k], fz[k] = basis.getFields(xg[k], yg[k], zg[k])

    dens = rho.reshape(phi_bins, theta_bins)
    dens0 = rho0.reshape(phi_bins, theta_bins)


    phi= pot.reshape(phi_bins, theta_bins)
    phi0 = pot0.reshape(phi_bins, theta_bins)


    return  dens, dens0, phi, phi0, [theta_grid, phi_grid]

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
    fx = np.zeros_like(xg)
    fy = np.zeros_like(xg)
    fz = np.zeros_like(xg)
    try:
        basis.set_coefs(coefficients.getCoefStruct(time))
    except AttributeError:
        basis.set_coefs(coefficients)

    for k in range(0, N):
        if projection == 'XY':
            rho0[k], pot0[k], rho[k], pot[k], fx[k], fy[k], fz[k] = basis.getFields(xg[k], yg[k], proj_plane)
        elif projection == 'XZ':
            rho0[k], pot0[k], rho[k], pot[k], fx[k], fy[k], fz[k] = basis.getFields(xg[k], proj_plane, yg[k])
        elif projection == 'YZ':
            rho0[k], pot0[k], rho[k], pot[k], fx[k], fy[k], fz[k] = basis.getFields(proj_plane, xg[k], yg[k])
    
    dens = rho.reshape(npoints, npoints)
    pot = pot.reshape(npoints, npoints)
    dens0 = rho0.reshape(npoints, npoints)
    pot0 = pot0.reshape(npoints, npoints)
    fx = fx.reshape(npoints,npoints)
    fy = fy.reshape(npoints,npoints)
    fz = fz.reshape(npoints,npoints)

    if prop == 'dens':
        if monopole_only:
            return dens0
        return dens0, dens, xgrid

    if prop == 'pot':
        if monopole_only:
            return pot0
        return pot0, pot, xgrid

    if prop == 'force':
        return [fx, fy, fz], xgrid
    

def slice_3d_fields(basis, coefficients, time=0,  npoints=50, 
                 grid_limits=(-300, 300), prop='dens', monopole_only=False):
    """
    Plots a slice projection of the fields of a simulation.

    Args:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    time (float): the time at which to plot the fields
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
    fx = np.zeros_like(xg)
    fy = np.zeros_like(xg)
    fz = np.zeros_like(xg)
    basis.set_coefs(coefficients.getCoefStruct(time))

    for k in range(0, N):
        rho0[k], pot0[k], rho[k], pot[k], fx[k], fy[k], fz[k] = basis.getFields(xg[k], yg[k], zg[k])
    
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
    
