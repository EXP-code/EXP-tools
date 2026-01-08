from importlib import resources
import matplotlib.pyplot as plt

def use_exptools_style(mathtext_fontset="cm", usetex=False):
    """
    Apply the exptools Matplotlib style, optionally overriding math font and LaTeX usage.

    Parameters
    ----------
    mathtext_fontset : str, optional
        One of the supported Matplotlib mathtext fontsets: 'cm', 'stix', 'stixsans',
        'dejavusans', 'dejavuserif'. Default is 'cm'.
    usetex : bool, optional
        If True, enable LaTeX rendering for all text via `text.usetex`. Default is False.

    Notes
    -----
    This function modifies Matplotlib rcParams globally for the
    current Python session. It is entirely optional and must be
    called explicitly by the user.
    """
    # Load the base style
    style_pkg = resources.files("EXPtools.visuals")
    style_path = style_pkg.joinpath("exptools.mplstyle")
    plt.style.use(str(style_path))

    # Override mathtext font
    if mathtext_fontset is not None:
        plt.rcParams["mathtext.fontset"] = mathtext_fontset

    # Optionally enable LaTeX rendering
    plt.rcParams["text.usetex"] = usetex

