import numpy as np
import tempfile
import os
from EXPtools.basis_builder import *

def test_write_table():
    # Prepare dummy data
    rvals = np.array([1.0, 2.0, 3.0])
    dvals = np.array([10.0, 20.0, 30.0])
    mvals = np.array([0.1, 0.2, 0.3])
    pvals = np.array([-1.0, -2.0, -3.0])

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp:
        filename = tmp.name
        _write_table(filename, rvals, dvals, mvals, pvals)

    # Read back the file and check contents
    with open(filename, 'r') as f:
        lines = f.readlines()

    assert lines[0].startswith('!')  # Header line
    assert lines[1].strip() == '! R    D    M    P'  # Column description
    assert lines[2].strip() == '3'  # Number of entries
    assert len(lines[3:]) == 3  # 3 data lines

    # Clean up
    os.remove(filename)

def test_makemodel():
    # Create simple synthetic density profile
    radius = np.logspace(-1, 1, 5)  # 5 points from 0.1 to 10
    density = 1.0 / radius**2       # rho ~ r^-2 (common in astrophysics)

    Mtotal = 10.0  # desired total mass

    # Run model generation
    r_scaled, d_scaled, m_scaled, p_scaled = makemodel(radius, density, Mtotal, verbose=False)

    # Assertions
    assert r_scaled.shape == radius.shape
    assert d_scaled.shape == density.shape
    assert m_scaled.shape == radius.shape
    assert p_scaled.shape == radius.shape

    assert np.all(np.isfinite(r_scaled))
    assert np.all(np.isfinite(d_scaled))
    assert np.all(np.isfinite(m_scaled))
    assert np.all(np.isfinite(p_scaled))

    assert np.all(r_scaled > 0)
    assert np.all(d_scaled > 0)
    assert np.all(m_scaled > 0)

    # Check mass conservation approximately
    np.testing.assert_allclose(m_scaled[-1], Mtotal, rtol=1e-2)

