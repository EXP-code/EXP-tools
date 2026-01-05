"""
Common halo profiles
"""
import numpy as np
from scipy import special


class Profiles:
    """
    Density profiles of dark matter halos.

    Parameters
    ----------
    radius : array-like
        Radial distances at which to evaluate profiles. Units: length.
    scale_radius : float
        Characteristic scale radius of the halo. Units: length.
    alpha : float, optional
        Inner slope of the halo. Default is 1.0 (NFW).
    beta : float, optional
        Outer slope of the halo. Default is 2.0 (NFW-like).
    amplitude : float, optional
        normalization. Default is 1

    Notes
    -----
    Common parameter choices:
        - NFW: alpha=1, beta=2
        - Hernquist: alpha=1, beta=3
        - Single power law: alpha=2.5, beta=0
    
    Modified from Mike Petersen's original code:
    https://gist.github.com/michael-petersen/ec4f20641eedac8f63ec409c9cc65ed7#file-simple_density_functions-py
    """

    def __init__(self, radius, scale_radius, amplitud=1.0, alpha=1.0, beta=2.0):
        self.radius = np.asarray(radius, dtype=float)
        self.scale_radius = float(scale_radius)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ra = self.radius / self.scale_radius
        self.amp = float(amplitud)

    def power_halo(self, rc=0.0):
        """
        Generic two-power-law distribution.

        Parameters
        ----------
        rc : float, optional
            Core radius. Default is 0 (no core).

        Returns
        -------
        ndarray
            Density values at each radius.
        """
        rc_scaled = rc / self.scale_radius
        return self.amp / ((self.ra + rc_scaled) ** self.alpha *
                      (1 + self.ra) ** self.beta)

    def power_halo_rolloff(self, rc=0.0, rtrunc_factor=25.0, wtrunc_factor=0.2):
        """
        Two-power-law distribution with an error-function rolloff.

        Parameters
        ----------
        rc : float, optional
            Core radius. Default is 0.
        rtrunc_factor : float, optional
            Truncation radius in units of scale radius. Default is 25.
        wtrunc_factor : float, optional
            Width of rolloff in units of rtrunc. Default is 0.2.

        Returns
        -------
        ndarray
            Density values with truncation applied.
        """
        dens = self.power_halo(rc=rc)
        rtrunc = rtrunc_factor * self.scale_radius
        wtrunc = wtrunc_factor * rtrunc
        rolloff = 0.5 - 0.5 * special.erf((self.radius - rtrunc) / wtrunc)
        return dens * rolloff

    def plummer_density(self):
        """
        Plummer density profile.

        Returns
        -------
        ndarray
            Density values at each radius.
        """
        rs2 = self.scale_radius ** 2
        return (3.0 * self.amp / (4 * np.pi)) * rs2 * (rs2 + self.radius**2)**(-2.5)

    def two_power_density_with_rolloff(self, rcen, wcen):
        """
        Two-power-law distribution with a central rolloff.

        Parameters
        ----------
        rcen : float
            Center radius for rolloff in units of scale radius.
        wcen : float
            Width of rolloff in units of scale radius.

        Returns
        -------
        ndarray
            Density values with central rolloff applied.
        """
        prefac = 0.5 * (1.0 - special.erf((self.ra - rcen) / wcen))
        return prefac * self.amp * self.ra ** -self.alpha * (1 + self.ra) ** (-self.beta + self.alpha)

def evaluate_density_profile(radii, amplitude, scale_radius, profile_func="power_halo", **kwargs):
    """
    Evaluate a dark matter density profile using any method of the Profiles class.

    Parameters
    ----------
    radii : array-like
        Radial distances at which to evaluate the density profile.
    amplitude : float
        Normalization of the profile.
    scale_radius : float
        Characteristic scale radius of the halo.
    profile_func : str or callable, optional
        Profile method to use. Can be:
        - A string matching any method name in `Profiles` (e.g., "power_halo",
          "power_halo_rolloff", "plummer_density", "two_power_density_with_rolloff").
        - A callable that accepts a `Profiles` instance and returns an ndarray.
        Default is "power_halo".
    **kwargs :
        Additional keyword arguments to pass to the profile method
        (e.g., core radius, rolloff parameters, etc.)

    Returns
    -------
    ndarray
        Density values at each radius.
    """

    # Initialize the profile
    profile = Profiles(radii, scale_radius, amplitud=amplitude, alpha=1.0, beta=3.0)

    # Determine the function to call
    if isinstance(profile_func, str):
        if not hasattr(profile, profile_func):
            raise ValueError(f"Profiles has no method '{profile_func}'")
        func = getattr(profile, profile_func)
    elif callable(profile_func):
        func = lambda: profile_func(profile)
    else:
        raise TypeError("profile_func must be a string method name or a callable")

    # Call the selected profile function with provided kwargs
    return func(**kwargs)

