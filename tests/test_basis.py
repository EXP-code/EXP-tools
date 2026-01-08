import numpy as np
import yaml
import tempfile
import os
from pathlib import Path
from EXPtools.basis import load_basis, write_config
import pyEXP


def test_write_config(tmp_path):

    R = np.linspace(1e-3, 300, 400)
    model = tmp_path / "dummy_model.txt"
    np.savetxt(model, np.c_[R, R, R, R])

    basis_params = {
        "basis_id": "sphereSL",
        "Lmax": 2,
        "nmax": 10,
        "rmapping": 1.0,
        "modelname": str(model),
        "cachename": "dummy_model_cache.txt",
    }

    yaml_str = write_config(basis_params, write_yaml=False)

    # ✅ Parse YAML back into a dict
    cfg = yaml.safe_load(yaml_str)

    # ---- structural checks ----
    assert cfg["id"] == "sphereSL"
    assert "parameters" in cfg

    # ---- parameter checks ----
    params = cfg["parameters"]
    assert params["Lmax"] == 2
    assert params["nmax"] == 10
    assert params["modelname"] == str(model)
    assert params["cachename"] == "dummy_model_cache.txt"

    # ---- derived values ----
    assert params["numr"] == len(R)-3
    assert np.isclose(params["rmin"], R[3])
    assert np.isclose(params["rmax"], R[-1])


def test_check_basis_params():
    

"""
def test_load_basis():

    DATA_DIR = Path(__file__).parent / "data"
    spherical_basis = os.path.join(DATA_DIR, "test_spherical_basis.yaml")
    cylindrical_basis = os.path.join(DATA_DIR, "test_cylindrical_basis.yaml")
    
    sph_basis = load_basis(spherical_basis)
    cyl_basis = load_basis(cylindrical_basis)
    
    assert type(sph_basis) == pyEXP.basis.SphericalSL
    assert type(cyl_basis) == pyEXP.basis.Cylindrical

    print("✅ All tests passed for load_basis")

"""


#def test_write_config():
    #"""Test the write_config function for both 'sphereSL' and 'cylinder'."""

    # ---- Test sphereSL ----
    #R = np.linspace(1e-3, 300, 400)
    #np.savetxt("dummy_model.txt", np.c_[R, R, R])  # 3 columns, like your models

    #basis_params = {
    #    'basis_id': 'sphereSL',
    #    'nmax': int(10),
    #    'Lmax': int(2),
    #    'modelname': "dummy_model.txt", 
    #    'cachename': "dummy_model.cache",
    #    'rmapping': R[-1]
    #    }


    #cfg = write_config(basis_params)
    
    #params = cfg["parameters"]
    #print("---")
    #print(cfg)
    #print(cfg["id"])
    #print(basis_params["numr"])
    # Check top-level keys
    #assert cfg["id"] == "sphereSL"
    #assert basis_params["numr"] == len(R)
    #assert isinstance(basis_params["Lmax"], int)
    #assert isinstance(basis_params["nmax"], int)
    #assert "rmin_str" in basis_params and "rmax_str" in basis_params

    # ---- Test cylinder ----
    #"""
    #yaml_str_cyl = write_config(
    #    basis_id="cylinder",
    #    acyl=1.5, hcyl=2.5,
    #    nmaxfid=20, lmaxfid=10, mmax=5, nmax=8,
    #    ncylodd=1, ncylnx=32, ncylny=32,
    #    rnum=64, pnum=32, tnum=16, vflag=0,
    #    logr=True,
    #    cachename="cyl_cache"
    #)
    #cfg_cyl = yaml.safe_load(yaml_str_cyl)

    #assert cfg_cyl["id"] == "cylinder"
    #params_cyl = cfg_cyl["parameters"]
    #assert isinstance(params_cyl["acyl"], float)
    #assert isinstance(params_cyl["hcyl"], float)
    #assert isinstance(params_cyl["logr"], bool)
    #assert isinstance(params_cyl["ncylnx"], int)
    #assert isinstance(params_cyl["ncylny"], int)
    #assert isinstance(params_cyl["cachename"], str)
    #"""
    #print("✅ All tests passed for write_config")
