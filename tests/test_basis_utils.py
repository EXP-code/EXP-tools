import numpy as np
import tempfile
import os
import yaml
from EXPtools.basis_builder import make_model, make_config, _write_table

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


def test_make_config():
    """Test the make_config function for both 'sphereSL' and 'cylinder'."""

    # ---- Test sphereSL ----
    R = np.linspace(1e-3, 300, 400)
    np.savetxt("dummy_model.txt", np.c_[R, R, R])  # 3 columns, like your models

    yaml_str = make_config(
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
    yaml_str_cyl = make_config(
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


def test_makemodel():
    # Create simple synthetic density profile
    radius = np.logspace(-1, 1, 5)  # 5 points from 0.1 to 10
    density = 1.0 / radius**2       # rho ~ r^-2 (common in astrophysics)

    Mtotal = 10.0  # desired total mass

    # Run model generation
    r_scaled, d_scaled, m_scaled, p_scaled = make_model(radius, density, Mtotal, verbose=False)

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

