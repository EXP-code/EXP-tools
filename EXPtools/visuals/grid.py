import numpy as np

class Grid3D:
    """
    Generate and transform 3D coordinate grids in Cartesian, Spherical, or Cylindrical systems.

    Conventions
    -----------
    - Spherical coordinates: (r, theta, phi)
        r >= 0
        theta ∈ [0, 2pi)  (azimuthal angle in the xy-plane)
        phi ∈ [0, pi]     (polar angle from +z axis)

    - Cylindrical coordinates: (rho, theta, z)
        rho >= 0
        theta ∈ [0, 2pi)
        z is unbounded

    - Cartesian coordinates: (x, y, z)

    Parameters
    ----------
    system : str
        One of {'cartesian', 'spherical', 'cylindrical'}.
    ranges : list of tuple
        Axis limits. Interpretation depends on `system`:
            - cartesian: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
              e.g. [(-1, 1), (-1, 1), (-1, 1)]
            - cylindrical: [(rho_min, rho_max), None, (z_min, z_max)]
              e.g. [(0, 5), None, (-2, 2)]
              (theta is always [0, 2π))
            - spherical: [(r_min, r_max), None, None]
              e.g. [(0, 10), None, None]
              (theta ∈ [0, 2π), phi ∈ [0, π] by default)
    num_points : int or list of int, optional
        Number of grid points per dimension. If int, applied to all.
        For example:
            - cartesian: [nx, ny, nz]
            - cylindrical: [nrho, ntheta, nz]
            - spherical: [nr, ntheta, nphi]
    """

    def __init__(self, system, ranges, num_points=10):
        self.system = system.lower()
        self.supported_systems = ['cartesian', 'spherical', 'cylindrical']

        if self.system not in self.supported_systems:
            raise ValueError(f"Unsupported system: {self.system}. "
                             f"Supported systems are {self.supported_systems}")

        # Normalize num_points
        if isinstance(num_points, int):
            num_points = [num_points] * 3
        if len(num_points) != 3 or not all(isinstance(n, int) and n > 0 for n in num_points):
            raise ValueError("num_points must be an int or a list of 3 positive integers")

        # Validate ranges
        if self.system == 'cartesian':
            if len(ranges) != 3:
                raise ValueError("Cartesian requires 3 ranges [(x_min, x_max), (y_min, y_max), (z_min, z_max)]")
        elif self.system == 'cylindrical':
            if len(ranges) != 3:
                raise ValueError("Cylindrical requires 3 entries [(rho_min, rho_max), None, (z_min, z_max)]")
        elif self.system == 'spherical':
            if len(ranges) != 3 or not isinstance(ranges[0], tuple):
                raise ValueError("Spherical requires [(r_min, r_max), None, None]")

        self.ranges = ranges
        self.num_points = num_points
        self.grid = self._generate_grid()

    def _generate_grid(self):
        if self.system == 'spherical':
            r_min, r_max = self.ranges[0]
            r_points = np.linspace(r_min**3, r_max**3, self.num_points[0])  # uniform in volume
            r = r_points**(1/3)

            theta = np.linspace(0, 2 * np.pi, self.num_points[1], endpoint=False)
            u = np.linspace(-1, 1, self.num_points[2])
            phi = np.arccos(u)

            R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
            return np.stack([R, Theta, Phi], axis=-1).reshape(-1, 3)

        elif self.system == 'cylindrical':
            rho = np.linspace(*self.ranges[0], self.num_points[0])
            theta = np.linspace(0, 2 * np.pi, self.num_points[1], endpoint=False)
            z = np.linspace(*self.ranges[2], self.num_points[2])
            Rho, Theta, Z = np.meshgrid(rho, theta, z, indexing='ij')
            return np.stack([Rho, Theta, Z], axis=-1).reshape(-1, 3)

        elif self.system == 'cartesian':
            axes = [np.linspace(start, stop, n) for (start, stop), n in zip(self.ranges, self.num_points)]
            mesh = np.meshgrid(*axes, indexing='ij')
            return np.stack(mesh, axis=-1).reshape(-1, 3)

    def to(self, target_system):
        """
        Transform the current grid to a new coordinate system.

        Parameters
        ----------
        target_system : str
            One of {'cartesian', 'spherical', 'cylindrical'}

        Returns
        -------
        np.ndarray
            Transformed (N, 3) grid.
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
        if system == 'spherical':
            r, theta, phi = grid[:, 0], grid[:, 1], grid[:, 2]
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            return np.column_stack((x, y, z))
        elif system == 'cylindrical':
            rho, theta, z = grid[:, 0], grid[:, 1], grid[:, 2]
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            return np.column_stack((x, y, z))
        else:
            raise ValueError("Invalid system for Cartesian conversion")

    def _from_cartesian(self, cartesian, target_system):
        x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]

        if target_system == 'spherical':
            r = np.sqrt(x**2 + y**2 + z**2)
            # Avoid division by zero
            with np.errstate(invalid='ignore', divide='ignore'):
                theta = np.arctan2(y, x)
                phi = np.arccos(np.clip(z / np.where(r == 0, 1, r), -1.0, 1.0))
            return np.column_stack((r, theta, phi))

        elif target_system == 'cylindrical':
            rho = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            return np.column_stack((rho, theta, z))

        else:
            raise ValueError("Invalid target system")

    def get(self):
        """Return the raw grid in its original coordinate system."""
        return self.grid

