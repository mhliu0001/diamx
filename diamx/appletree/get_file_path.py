import os
import importlib_resources
import appletree as apt


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder."""
    return importlib_resources.files("diamx.appletree") / sub_directory


def get_file_path_diamx(fname):
    try:
        return apt.utils.get_file_path(fname)
    except RuntimeError:
        for sub_dir in ("maps", "parameters", "instructs", "model"):
            p = os.path.join(_package_path(sub_dir), fname)
            if os.path.exists(p):
                return p
        raise RuntimeError(f"Cannot find {fname}")
