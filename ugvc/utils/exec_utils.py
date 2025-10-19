import pkgutil
from os.path import dirname
from os.path import join as pjoin


def find_scripts_path() -> str:
    """Locates the absolute path of the scripts installation

    Parameters
    ----------
    None

    Returns
    -------
    str
        The path
    """
    package = pkgutil.get_loader("ugvc")
    return pjoin(dirname(package.get_filename()), "bash")
