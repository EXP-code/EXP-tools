import numpy as np
import pytest
import os

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


@pytest.fixture(scope="session")
def particle_data():
    """
    Load particle data from ASCII file.

    Expected format:
    x  y  z  mass
    """
    data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "data", "particle_data.txt")
    )

    pos = data[:, 0:3]
    mass = data[:, 3]

    return {
        "pos": pos,
        "mass": mass,
        "snapshot_time": 0.0,
    }


@pytest.fixture(scope="session")
def reference_coefs_file():
    return os.path.join(
        os.path.dirname(__file__), "data", "coef_test.hdf5"
    )


@pytest.fixture(scope="session")
def mpi_comm():
    if not MPI_AVAILABLE:
        pytest.skip("mpi4py not available")

    return MPI.COMM_WORLD

