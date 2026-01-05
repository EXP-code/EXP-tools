import matplotlib
matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
from importlib import resources

import EXPtools.visuals as visuals


def test_exptools_style_file_is_packaged():
    """
    Test that the exptools Matplotlib style file is included
    in the installed package.
    """
    style_path = resources.files("EXPtools.visuals").joinpath(
        "exptools.mplstyle"
    )

    assert style_path.is_file(), (
        "exptools.mplstyle is missing from the installed package. "
        "Ensure it is included as package_data."
    )


def test_use_exptools_style_applies_without_error():
    """
    Test that use_exptools_style() applies the style and sets rcParams.
    """
    # Reset rcParams to default
    plt.rcdefaults()

    # Default call
    visuals.use_exptools_style()

    # Minimal sanity check
    assert "axes.linewidth" in plt.rcParams
    # Default mathtext.fontset should be 'cm'
    assert plt.rcParams["mathtext.fontset"] == "cm"
    # Default usetex should be False
    assert plt.rcParams["text.usetex"] is False


def test_use_exptools_style_mathfont_and_usetex():
    """
    Test that use_exptools_style() correctly overrides math font and usetex.
    """
    plt.rcdefaults()

    # Test with STIX font and usetex enabled
    visuals.use_exptools_style(mathtext_fontset="stix", usetex=True)

    assert plt.rcParams["mathtext.fontset"] == "stix"
    assert plt.rcParams["text.usetex"] is True

    # Test with DejaVu Sans and usetex disabled
    plt.rcdefaults()
    visuals.use_exptools_style(mathtext_fontset="dejavusans", usetex=False)

    assert plt.rcParams["mathtext.fontset"] == "dejavusans"
    assert plt.rcParams["text.usetex"] is False

