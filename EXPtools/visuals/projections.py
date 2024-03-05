"""
Dependencies:
  - Matplitlib
  - Healpy
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

    
    
def mollweide(l, b, bmin, bmax, title="", nside=24, smooth=1, q=[0], **kwargs):

    """
    Makes mollweide plot using healpix
    
    Parameters
    ---------- 
    l : numpy.array 
        Longitude in degrees 
    b : numpy.array 
        Latitude in degrees [-90, 90]

    Returns
    -------
    None 
        None

    """
 

    mwlmc_indices = hp.ang2pix(nside,  (90-b)*np.pi/180., l*np.pi/180.)
    npix = hp.nside2npix(nside)
 
    idx, counts = np.unique(mwlmc_indices, return_counts=True)
    degsq = hp.nside2pixarea(nside, degrees=True)
    # filling the full-sky map
    hpx_map = np.zeros(npix, dtype=float)
    if q[0] != 0 :    
        counts = np.zeros_like(idx, dtype=float)
        k=0
        for i in idx:
            pix_ids = np.where(mwlmc_indices==i)[0]
            counts[k] = np.mean(q[pix_ids])
            k+=1
        hpx_map[idx] = counts
    else :
       hpx_map[idx] = counts/degsq
    
    map_smooth = hp.smoothing(hpx_map, fwhm=smooth*np.pi/180)
    
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap='viridis'
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.close()
    projview(
      map_smooth,
      coord=["G"], # Galactic
      graticule=True,
      graticule_labels=True,
      rot=(0, 0, 0),
      unit=" ",
      #xlabel="Galactic Longitude (l) ",
      ylabel="Galactic Latitude (b)",
      cb_orientation="horizontal",
      min=bmin,
      max=bmax,
      latitude_grid_spacing=45,
      projection_type="mollweide",
      title=title,
      cmap=cmap,
      fontsize={
              "xlabel": 25,
              "ylabel": 25,
              "xtick_label": 20,
              "ytick_label": 20,
              "title": 25,
              "cbar_label": 20,
              "cbar_tick_label": 20,
              },
      )
	
    if 'l2' in kwargs.keys():
        l2 = kwargs['l2']
        b2 = kwargs['b2']
    newprojplot(theta=np.radians(90-(b2)), phi=np.radians(l2), marker="o", color="yellow", markersize=5, lw=0, mfc='none')
    if 'l3' in kwargs.keys():
        l3 = kwargs['l3']
        b3 = kwargs['b3']
        newprojplot(theta=np.radians(90-(b3)), phi=np.radians(l3), marker="o", color="yellow", markersize=5, lw=0)
    elif 'l4' in kwargs.keys():
        l4 = kwargs['l4']
        b4 = kwargs['b4']
        newprojplot(theta=np.radians(90-(b4)), phi=np.radians(l4), marker="*", color="r", markersize=8, lw=0)

    
    if 'figname' in kwargs.keys():
        print("* Saving figure in ", kwargs['figname'])
        plt.savefig(kwargs['figname'], bbox_inches='tight')
        plt.close()


