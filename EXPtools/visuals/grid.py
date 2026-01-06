import numpy as np


class Grid3D:
    """
    Generate and transform 3D coordinate grids in Cartesian, Spherical, or Cylindrical systems.

    This class constructs structured 3D grids in one of three coordinate systems and
    provides utilities to transform between them. Grids are stored internally as a
    flattened array of shape ``(N, 3)``, where ``N`` is the total number of grid points.

    Coordinate Conventions (Physics convention)
    --------------------------------------------
    - Cartesian: (x, y, z)
    - Spherical: (r, theta, phi)
        * r >= 0
        * theta ∈ [0, π) (polar angle in the xy-plane)
        * phi ∈ [0, 2π]  (azimuthal angle from +z axis)
    - Cylindrical: (rho, phi, z)
        * rho >= 0
        * phi ∈ [0, 2π)
        * z ∈ ℝ

    Notes
    -----
    - Spherical grids use volume-uniform radial sampling by default
      (uniform in :math:`r^3`).
    - All grids are generated using ``numpy.meshgrid`` with ``indexing='ij'``.
    - Angular coordinates always follow standard *physics* conventions.

    Parameters
    ----------
    system : str
        Coordinate system for the grid. Must be one of
        ``{'cartesian', 'spherical', 'cylindrical'}``.
    ranges : list of tuple or None
        Axis limits. Interpretation depends on `system`:

        * cartesian:
          ``[(x_min, x_max), (y_min, y_max), (z_min, z_max)]``

        * cylindrical:
          ``[(rho_min, rho_max), None, (z_min, z_max)]``
          (phi is always [0, 2π))

        * spherical:
          ``[(r_min, r_max), None, None]``
          (theta ∈ [0, π), phi ∈ [0, 2π])
    num_points : int or list of int, optional
        Number of grid points per dimension. If an integer is provided,
        the same value is used for all dimensions. Default is 10.

    Attributes
    ----------
    system : str
        Coordinate system of the grid.
    ranges : list
        Axis limits used to construct the grid.
    num_points : list of int
        Number of grid points per axis.
    grid : ndarray, shape (N, 3)
        Flattened grid of coordinates in the native system.
    """

    def __init__(self, system, ranges, num_points=10, r_bins=None):
        """
        Initialize a 3D coordinate grid.

        Parameters
        ----------
        system : str
            Coordinate system of the grid. Must be one of
            ``{'cartesian', 'spherical', 'cylindrical'}``.

        ranges : list of tuple or None
            Axis limits defining the extent of the grid. Interpretation
            depends on the coordinate system:

            * cartesian:
            ``[(x_min, x_max), (y_min, y_max), (z_min, z_max)]``

            * cylindrical:
            ``[(rho_min, rho_max), None, (z_min, z_max)]``
            (phi is always [0, 2π))

            * spherical:
            ``[(r_min, r_max), None, None]``
            (theta ∈ [0, π), phi ∈ [0, 2π])

            When ``r_bins`` is provided for spherical grids, ``ranges[0]`` is
            ignored.

        num_points : int or list of int, optional
            Number of grid points per axis. If an integer is provided, the
            same value is used for all dimensions. Default is 10.

            For spherical grids, ``num_points[0]`` is ignored when ``r_bins``
            is supplied.

        r_bins : array_like, optional
            Explicit radial bin centers or edges for spherical grids.
            If provided, the radial coordinate ``r`` is taken directly from
            this array instead of being generated internally.

            Notes:
                * Only applicable when ``system='spherical'``.
                * Must be one-dimensional and strictly non-negative.
                * Enables non-uniform, user-controlled radial sampling
                (e.g. logarithmic bins, adaptive bins, simulation outputs).
                * Useful when matching observational or simulation-derived
                radial grids.

        Raises
        ------
        ValueError
            If ``r_bins`` is provided for a non-spherical coordinate system.
        """


        self.system = system.lower()
        self.supported_systems = ['cartesian', 'spherical', 'cylindrical']

        if self.system not in self.supported_systems:
            raise ValueError(
                f"Unsupported system: {self.system}. "
                f"Supported systems are {self.supported_systems}"
            )


        # Normalize num_points
        if isinstance(num_points, int):
            num_points = [num_points] * 3
        if len(num_points) != 3 or not all(isinstance(n, int) and n > 0 for n in num_points):
            raise ValueError("num_points must be an int or a list of 3 positive integers")

        # Validate ranges
        if self.system == 'cartesian':
            if len(ranges) != 3:
                raise ValueError("Cartesian requires 3 ranges")
        elif self.system == 'cylindrical':
            if len(ranges) != 3:
                raise ValueError("Cylindrical requires 3 entries")
        elif self.system == 'spherical':
            if len(ranges) != 3 or not isinstance(ranges[0], tuple):
                raise ValueError("Spherical requires [(r_min, r_max), None, None]")

        if r_bins is not None:
            if self.system != 'spherical':
                raise ValueError("r_bins can only be used with spherical grids")

            r_bins = np.asarray(r_bins, dtype=float)
            if r_bins.ndim != 1 or np.any(r_bins <= 0):
                raise ValueError("r_bins must be a 1D array of positive values")
            if not np.all(np.diff(r_bins) > 0):
                raise ValueError("r_bins must be strictly increasing")

        self.r_bins = r_bins
        self.ranges = ranges
        self.num_points = num_points
        self.grid = self._generate_grid()

    def _generate_grid(self):
        """
        Generate the grid in the native coordinate system.

        Returns
        -------
        ndarray, shape (N, 3)
            Flattened coordinate grid.
        """
        if self.system == 'spherical':

            if self.r_bins is not None:
                r = self.r_bins
            else:
                # Evenly distributed by volume
                r_min, r_max = self.ranges[0]
                r_points = np.linspace(r_min**3, r_max**3, self.num_points[0]) 
                r = r_points ** (1/3)

            u = np.linspace(-1, 1, self.num_points[1])
            theta = np.arccos(u)
            phi = np.linspace(0, 2 * np.pi, self.num_points[2], endpoint=False)
            
            R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
            return np.stack([R, Theta, Phi], axis=-1).reshape(-1, 3)

        elif self.system == 'cylindrical':
            rho = np.linspace(*self.ranges[0], self.num_points[0])
            phi = np.linspace(0, 2 * np.pi, self.num_points[1], endpoint=False)
            z = np.linspace(*self.ranges[2], self.num_points[2])
            Rho, Phi, Z = np.meshgrid(rho, phi, z, indexing='ij')
            return np.stack([Rho, Phi, Z], axis=-1).reshape(-1, 3)

        elif self.system == 'cartesian':
            axes = [
                np.linspace(start, stop, n)
                for (start, stop), n in zip(self.ranges, self.num_points)
            ]
            mesh = np.meshgrid(*axes, indexing='ij')
            return np.stack(mesh, axis=-1).reshape(-1, 3)


    def to(self, target_system):
        """
        Transform the grid to a different coordinate system.

        Parameters
        ----------
        target_system : str
            Target coordinate system. Must be one of
            ``{'cartesian', 'spherical', 'cylindrical'}``.

        Returns
        -------
        ndarray, shape (N, 3)
            Grid transformed to the target coordinate system.
        """
        target_system = target_system.lower()
        if target_system == self.system:
            return self.grid

        if self.system != 'cartesian':
            cartesian = self._to_cartesian(self.grid, self.system)
        else:
            cartesian = self.grid

        if target_system == 'cartesian':
            return cartesian
        else:
            return self._from_cartesian(cartesian, target_system)

    def _to_cartesian(self, grid, system):
        """
        Convert a grid from spherical or cylindrical to Cartesian coordinates.

        Parameters
        ----------
        grid : ndarray, shape (N, 3)
            Input grid.
        system : {'spherical', 'cylindrical'}
            Original coordinate system.

        Returns
        -------
        ndarray, shape (N, 3)
            Cartesian coordinates.
        """
        if system == 'spherical':
            r, theta, phi = grid[:, 0], grid[:, 1], grid[:, 2]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return np.column_stack((x, y, z))
        elif system == 'cylindrical':
            rho, phi, z = grid[:, 0], grid[:, 1], grid[:, 2]
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return np.column_stack((x, y, z))
        else:
            raise ValueError("Invalid system for Cartesian conversion")

    def _from_cartesian(self, cartesian, target_system):
        """
        Convert Cartesian coordinates to another coordinate system.

        Parameters
        ----------
        cartesian : ndarray, shape (N, 3)
            Cartesian coordinates.
        target_system : {'spherical', 'cylindrical'}
            Desired coordinate system.

        Returns
        -------
        ndarray, shape (N, 3)
            Transformed coordinates.
        """
        x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]

        if target_system == 'spherical':
            r = np.sqrt(x**2 + y**2 + z**2)
            # Avoid division by zero
            with np.errstate(invalid='ignore', divide='ignore'):
                phi = np.arctan2(y, x)
                theta = np.arccos(np.clip(z / np.where(r == 0, 1, r), -1.0, 1.0))
            return np.column_stack((r, theta, phi))

        elif target_system == 'cylindrical':
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return np.column_stack((rho, phi, z))

        else:
            raise ValueError("Invalid target system")

    def get(self):
        """
        Return the raw grid in its native coordinate system.

        Returns
        -------
        ndarray, shape (N, 3)
            Grid coordinates.
        """
        return self.grid

