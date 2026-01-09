import numpy as np
import yaml
import tempfile
import os
from pathlib import Path
from EXPtools.basis import load_basis, write_config, make_basis, config_dict_to_yaml
import pyEXP


def test_config_dict_to_yaml():

    DATA_DIR = Path(__file__).parent / "data"
    model = os.path.join(DATA_DIR, "test_plummer_halo_hf.txt")
    cache = os.path.join(DATA_DIR, "test_cache_plummer_halo.txt")
    
    basis_params = {
        "basis_id": "sphereSL",
        "Lmax": 1,
        "nmax": 5,
        "rmapping": 1,
        "modelname": str(model),
        "cachename": str(cache),
    }

    yaml_str = config_dict_to_yaml(basis_params)

    # ✅ Parse YAML back into a dict
    cfg = yaml.safe_load(yaml_str)

    # ---- structural checks ----
    assert cfg["id"] == "sphereSL"
    assert "parameters" in cfg

    # ---- parameter checks ----
    params = cfg["parameters"]
    assert params["Lmax"] == 1
    assert params["nmax"] == 5
    assert params["modelname"] == str(model)
    assert params["cachename"] == str(cache)

    # ---- derived values ----
    assert params["numr"] == 500
    config_file = os.path.join(DATA_DIR, "test_plummer_halo_config.yaml") 
    write_config(cfg, basis_filename=config_file)
    print("✅ All tests passed for config_dict_to_yaml")


def test_load_basis():
    DATA_DIR = Path(__file__).parent / "data"
    config_file = os.path.join(DATA_DIR, "test_plummer_halo_config.yaml")    
    plummer_basis = load_basis(config_file)
    #basis = pyEXP.basis.Basis.factory(plummer_basis)
    assert type(plummer_basis) == pyEXP.basis.SphericalSL
    print("✅ All tests passed for load_basis")

 
def test_make_basis():
    DATA_DIR = Path(__file__).parent / "data"
    TEST_PLUMMER = "test_plummer_halo_hf.txt"
    test_data = os.path.join(DATA_DIR, TEST_PLUMMER)
    test_cache = os.path.join(DATA_DIR, "test_cache_plummer_halo.txt")
    
    R_plummer, D_plummer, M_plummer, _ = np.loadtxt(test_data, skiprows=3, unpack=True)
    basis_params = {
        'basis_id': 'sphereSL',
        'Lmax': 1, 
        'nmax': 5, 
        'rmapping': 1.0,
        'modelname': str(test_data),
        'cachename': str(test_cache)}
    
    plummer_basis = make_basis(R_plummer, D_plummer, Mtotal=M_plummer[-1],
                                basis_params=basis_params, write_basis=True,
                                basis_filename='test_basis_n.yaml')
    assert type(plummer_basis) == pyEXP.basis.SphericalSL

    print("✅ All tests passed for make_basis")
