import numpy as np
import tempfile
import os
from pathlib import Path
from EXPtools.basis import make_model, write_table
from EXPtools.utils import Profiles

def test_write_table():
    # Prepare dummy data
    rvals = np.array([1.0, 2.0, 3.0])
    dvals = np.array([10.0, 20.0, 30.0])
    mvals = np.array([0.1, 0.2, 0.3])
    pvals = np.array([-1.0, -2.0, -3.0])

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp:
        filename = tmp.name
        write_table(filename, rvals, dvals, mvals, pvals)

    # Read back the file and check contents
    with open(filename, 'r') as f:
        lines = f.readlines()

    assert lines[0].startswith('!')  # Header line
    assert lines[1].strip() == '! R    D    M    P'  # Column description
    assert lines[2].strip() == '3'  # Number of entries
    assert len(lines[3:]) == 3  # 3 data lines

    # Clean up
    os.remove(filename)


def test_make_model():
    DATA_DIR = Path(__file__).parent / "data"
    TEST_PLUMMER = "plummer_halo_hf.txt"
    test_data = os.path.join(DATA_DIR, TEST_PLUMMER)
    R_plummer, D_plummer, M_plummer, P_plummer = np.loadtxt(test_data, skiprows=3, unpack=True)
    
    # Build plummer model
    radius = np.logspace(-2, 2.5, 500)  #
    halo_model = Profiles(radius, scale_radius=10)
    density = halo_model.plummer_density()
    Mtotal = 0.998  # desired total mass

    # Run make_model 
    model = make_model(radius, density, Mtotal, physical_units=True, verbose=False)
    r_scaled = model['radius']
    d_scaled = model['density']
    m_scaled = model['mass']
    p_scaled = model['potential']
    
    np.testing.assert_allclose(np.log10(R_plummer), np.log10(r_scaled), atol=1e-6)
    np.testing.assert_allclose(np.log10(D_plummer), np.log10(d_scaled), atol=1e-6)
    np.testing.assert_allclose(np.log10(m_scaled), np.log10(M_plummer),atol=1e-3)
    np.testing.assert_allclose(np.log10(np.abs(p_scaled)), np.log10(np.abs(P_plummer)), atol=1e-3)

    assert np.all(np.isfinite(r_scaled))
    assert np.all(np.isfinite(d_scaled))
    assert np.all(np.isfinite(m_scaled))
    assert np.all(np.isfinite(p_scaled))
    #print(r_scaled)
    # Check mass conservation approximately
    np.testing.assert_allclose(m_scaled[-1], Mtotal, rtol=1e-3)


    # Build plummer model
    radius = np.logspace(-2, 2.5, 500)  #
    halo_model = Profiles(radius, scale_radius=10)
    density = halo_model.plummer_density()
    Mtotal = 1.0  # desired total mass

    # Run make_model 
    model = make_model(radius, density, Mtotal, physical_units=True, verbose=False)
    r_scaled = model['radius']
    d_scaled = model['density']
    m_scaled = model['mass']
    p_scaled = model['potential']
    test_data = os.path.join(DATA_DIR, "plummer_halo_hf.txt")
    write_table(test_data, r_scaled, d_scaled, m_scaled, p_scaled)    

