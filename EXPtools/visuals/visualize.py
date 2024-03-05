import os,  sys, pickle, pyEXP
import numpy as np

###Field computations for plotting###
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
            grid  = [0, npoints, npoints]
        
        field_gen = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)
        return field_gen.slices(basis, coefficients), xgrid
