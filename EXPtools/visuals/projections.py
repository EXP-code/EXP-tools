"""
Dependencies:
  - Matplitlib
  - Healpy
"""

import numpy as np
import matplotlib.pyplot as plt
#from astropy import units as u
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

    
class Projections:
    def __init__(self, coords, field):
        """
        Parameters
        ----------
        coords : mapping-like
            Coordinates with keys 'l' (longitude) and 'b' (latitude) in degrees.
        field : array-like
            Field values corresponding to the coordinates.
        """
        self.coords = coords
        self.field = field

    def cartesian(self, proj):
        plt.contourf(self.coords[0], self.coords[1], self.field)

    def mollweide(self, bmin, bmax, title="", nside=24, smooth=1, **kwargs):
        """
        Makes a Mollweide projection plot using Healpy.

        Parameters
        ----------
        bmin, bmax : float
            Minimum and maximum values for the color scale.
        title : str, optional
            Plot title.
        nside : int, optional
            Healpy nside parameter.
        smooth : float, optional
            Smoothing FWHM in degrees.

        Notes
        -----
        Expects `self.coords` to provide arrays with keys 'l' (longitude)
        and 'b' (latitude), both in degrees.
        """
        lat = self.coords['b']
        lon = self.coords['l']

        mwlmc_indices = hp.ang2pix(
            nside,
            (90.0 - lat) * np.pi / 180.0,
            lon * np.pi / 180.0,
        )
        npix = hp.nside2npix(nside)

        idx, _ = np.unique(mwlmc_indices, return_counts=True)
        # filling the full-sky map with mean values per pixel
        hpx_map = np.zeros(npix, dtype=float)
        mean_vals = np.zeros_like(idx, dtype=float)

        for k_index, pix in enumerate(idx):
            pix_ids = np.where(mwlmc_indices == pix)[0]
            mean_vals[k_index] = np.mean(self.field[pix_ids])
        hpx_map[idx] = mean_vals

        map_smooth = hp.smoothing(hpx_map, fwhm=smooth * np.pi / 180.0)

        cmap = kwargs.get('cmap', 'viridis')
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plt.close()

        projview(
            map_smooth,
            coord=["G"],  # Galactic
            graticule=True,
            graticule_labels=True,
            rot=(0, 0, 0),
            unit=" ",
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

        if 'l2' in kwargs and 'b2' in kwargs:
            l2 = kwargs['l2']
            b2 = kwargs['b2']
            newprojplot(
                theta=np.radians(90.0 - b2),
                phi=np.radians(l2),
                marker="o",
                color="yellow",
                markersize=5,
                lw=0,
                mfc='none',
            )

        if 'l3' in kwargs and 'b3' in kwargs:
            l3 = kwargs['l3']
            b3 = kwargs['b3']
            newprojplot(
                theta=np.radians(90.0 - b3),
                phi=np.radians(l3),
                marker="o",
                color="yellow",
                markersize=5,
                lw=0,
            )
        elif 'l4' in kwargs and 'b4' in kwargs:
            l4 = kwargs['l4']
            b4 = kwargs['b4']
            newprojplot(
                theta=np.radians(90.0 - b4),
                phi=np.radians(l4),
                marker="*",
                color="r",
                markersize=8,
                lw=0,
            )

        if 'figname' in kwargs:
            print(f"* Saving figure in {kwargs['figname']}")
            plt.savefig(kwargs['figname'], bbox_inches='tight')
            plt.close()
