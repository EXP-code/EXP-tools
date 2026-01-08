import numpy as np
import tempfile
import os
import yaml
from EXPtools.basis.basis_utils import write_config
from EXPtools.basis.makemodel import make_model, write_table
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


def test_write_config():
    """Test the make_config function for both 'sphereSL' and 'cylinder'."""

    # ---- Test sphereSL ----
    R = np.linspace(1e-3, 300, 400)
    np.savetxt("dummy_model.txt", np.c_[R, R, R])  # 3 columns, like your models

    yaml_str = write_config(
        basis_id="sphereSL",
        lmax=8, nmax=10, rmapping=R[-1],
        modelname="dummy_model.txt",
        cachename=".cache_test"
    )
    cfg = yaml.safe_load(yaml_str)

    # Check top-level keys
    assert cfg["id"] == "sphereSL"
    params = cfg["parameters"]
    assert params["numr"] == len(R)
    assert isinstance(params["Lmax"], int)
    assert isinstance(params["nmax"], int)
    assert "rmin_str" in params and "rmax_str" in params

    # ---- Test cylinder ----
    yaml_str_cyl = write_config(
        basis_id="cylinder",
        acyl=1.5, hcyl=2.5,
        nmaxfid=20, lmaxfid=10, mmax=5, nmax=8,
        ncylodd=1, ncylnx=32, ncylny=32,
        rnum=64, pnum=32, tnum=16, vflag=0,
        logr=True,
        cachename="cyl_cache"
    )
    cfg_cyl = yaml.safe_load(yaml_str_cyl)

    assert cfg_cyl["id"] == "cylinder"
    params_cyl = cfg_cyl["parameters"]
    assert isinstance(params_cyl["acyl"], float)
    assert isinstance(params_cyl["hcyl"], float)
    assert isinstance(params_cyl["logr"], bool)
    assert isinstance(params_cyl["ncylnx"], int)
    assert isinstance(params_cyl["ncylny"], int)
    assert isinstance(params_cyl["cachename"], str)

    print("✅ All tests passed for make_config")


def test_make_model():
    from pathlib import Path

    DATA_DIR = Path(__file__).parent / "data"
    # Create simple synthetic density profile
    TEST_PLUMMER = "test_plummer_halo.txt"
    test_data = os.path.join(DATA_DIR, TEST_PLUMMER)
    R_plummer, D_plummer, M_plummer, P_plummer = np.loadtxt(test_data, skiprows=3, unpack=True)
    
    radius = np.logspace(-2, 2.5, 300)  #
    Profiles(radius, scale_radius=10)
    density = Profiles.plummer_density()

    Mtotal = 10.0  # desired total mass

    # Run model generation
    r_scaled, d_scaled, m_scaled, p_scaled = make_model(radius, density, Mtotal,
            physical_units=True, verbose=False)
    # Assertions
    assert r_scaled == R_plummer
    assert d_scaled == D_plummer
    assert m_scaled == M_plummer
    assert p_scaled == P_plummer

    assert np.all(np.isfinite(r_scaled))
    assert np.all(np.isfinite(d_scaled))
    assert np.all(np.isfinite(m_scaled))
    assert np.all(np.isfinite(p_scaled))


    # Check mass conservation approximately
    np.testing.assert_allclose(m_scaled[-1], Mtotal, rtol=1e-2)

