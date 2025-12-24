"""
Routines to generate a Hernquist halo
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

class ICHernquist:
    """
    Class to sample the positions of a Hernquist halo
    initialized with the number of particles in the halo
    """

    def __init__(self, size):
        self.size = int(size)
    
    def sample_profile(self):
        """
        Sample 1d Hernquist density profile
        
        Parameters:
        ----------
        size : int
            Number of points used to sample the density profile.

        Returns:
        --------
	nu : array-like
            Number density array of the Hernquist halo of a given size. 
	
        """

    
        mu = np.random.random(size=self.size)
        nu = mu**0.5 / (1-mu**0.5)
        return nu

    def triaxial(self, axis_ratios, rot_angle, 
		 rot_axis = 'z', center = [0, 0, 0]):
        """
	Parameters:
        -----------
        axis_ratios : list
            List of principal axis ratios in the following order: [a, b, c] 
            axis will be computes as x/a, y/b, z/c
	    rot_angle : float
            Angle at which the halo is going to be rotated in degrees
        rot_axis : str
            Rotation axis it could be: 'x', 'y', 'z'
        center: list
            List with the 3d center of the halo [x, y, z] (default: [0, 0, 0])

	Returns: 
        --------
        xyz: array-like
	    Array with 3d-positions of the particles in the halo     	
        """
        
        a, b, c = axis_ratios
        phi = np.random.uniform(0, 2*np.pi, size=self.size)
        theta = np.arccos(2*np.random.random(size=self.size) - 1)
        r = self.sample_profile()
        
        xyz = np.zeros((self.size, 3))
        xyz[:,0] = r * np.cos(phi) * np.sin(theta) / a + center[0]
        xyz[:,1] = r * np.sin(phi) * np.sin(theta) / b + center[1]
        xyz[:,2] = r * np.cos(theta) / c   + center[2]
        
        rot = Rot.from_euler(rot_axis, rot_angle, degrees=True)
        rot_xyz = rot.apply(xyz)
        
        return rot_xyz


