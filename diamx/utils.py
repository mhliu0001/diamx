import json
import os
import hashlib
import numpy as np
import inference_interface
from multihist import Histdd

save_folder = "."
template_folder = "templates"

from typing import Type
import functools
import warnings
import sys
import math


def generate_bin_array(bin_spec):
    """Converts the bin specification into a standardized NumPy array."""
    if isinstance(bin_spec, str):
        # Assumes valid NumPy expressions
        bins = np.array(eval(bin_spec))
    elif isinstance(bin_spec, (list, np.ndarray)):
        bins = np.array(bin_spec)
    else:
        raise ValueError(f"Unsupported bin specification type: {type(bin_spec)}")
    return bins


def serialize_config(roi, **kwargs):
    """Serializes the experiment configuration for hashing."""
    # Process and standardize the binning
    binning = {}
    for key in roi:
        binning[key] = generate_bin_array(roi[key])

    bin_strings = {
        key: np.array2string(
            binning[key], separator=",", precision=8, suppress_small=True
        )
        for key in binning
    }

    # Serialize parameters from keyword arguments
    args_json = json.dumps(kwargs, sort_keys=True)

    # Combine binning and args
    combined_string = json.dumps(
        {"bins": bin_strings, "args": args_json}, sort_keys=True
    )

    return combined_string


def create_hash(roi, **kwargs):
    """Creates a hash for the given configuration."""
    serialized_config = serialize_config(roi, **kwargs)
    return hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()[:12]


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# https://stackoverflow.com/questions/32314071/how-to-block-warnings-inside-a-method
def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.

    Args:
        warning (Warning): warning type to ignore.

    Returns:
        the inner function

    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=warning)
                return func(*args, **kwargs)

        return wrapper

    return inner


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# disable stderr (for tqdm)
class HiddenTqdm:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


def get_wimp_masses(model):
    if model == "wimp":
        wimp_masses = list(range(6, 130))
        wimp_masses += [150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370]
        wimp_masses += [400, 500, 600, 800, 1000]
    elif model == "axion_med":
        wimp_masses = list(range(10, 1000, 10))
        wimp_masses += list(range(1000, 10001, 100))
    else:
        raise ValueError(f"Unknown model {model}!")
    return wimp_masses


def csv_to_apt_map(csv_file, coordinate_name="energy"):
    # Assume that the csv_file ends with ".csv"
    base_name, ext = os.path.splitext(csv_file)
    assert ext == ".csv", "File name should end with .csv!"
    json_file_path = base_name + "." + "json"
    if os.path.exists(json_file_path):
        return json_file_path

    data = np.loadtxt(csv_file, delimiter=",")
    apt_map_json = dict()
    apt_map_json["coordinate_system"] = list(data[:, 0])
    apt_map_json["coordinate_type"] = "point"
    apt_map_json["coordinate_name"] = coordinate_name
    apt_map_json["map"] = list(data[:, 1])

    with open(json_file_path, "w") as f:
        json.dump(apt_map_json, f)

    return json_file_path


def make_template(cs1, cs2, eff, rate_multiplier, output_path, fname, bin_edges=None):
    if bin_edges == None:
        bin_edges = [
            np.linspace(0, 100, 101),
            np.linspace(2.1, 4.1, 81),
            # np.append(np.array([0]), np.linspace(45**2, 63**2, 20) ** 0.5),
            # np.linspace(-straxen.tpc_z, 0, 21),
        ]
    mh = Histdd(
        cs1,
        np.log10(cs2),
        # self.df_apt_sim["r"],
        # self.df_apt_sim["z"],
        bins=bin_edges,
        weights=eff,
        axis_names=["cs1", "log10_cs2"],
    )
    mh.histogram = mh.histogram / np.sum(mh.histogram) * eff.sum() * rate_multiplier
    inference_interface.multihist_to_template(
        [mh],
        os.path.join(output_path, "template", fname),
        histogram_names=["cs1-log10_cs2"],
    )


def nest_csv_to_template(csv_file, output_path, fname, experiment="lz_ws2022"):
    cs1, cs2 = np.loadtxt(csv_file, delimiter=",").T
    eff = cs1 < 80
    eff &= cs1 > 3
    eff &= cs2 > 600
    if experiment == "lz_ws2024":
        eff &= cs2 > 645
        bin_edges = [
            np.linspace(3, 80, 101),
            np.linspace(2.5, 4.5, 101),
            # np.append(np.array([0]), np.linspace(45**2, 63**2, 20) ** 0.5),
            # np.linspace(-straxen.tpc_z, 0, 21),
        ]
    elif experiment == "lz_ws2022":
        bin_edges = [
            np.linspace(3, 80, 101),
            np.linspace(2.75, 4.5, 101),
            # np.append(np.array([0]), np.linspace(45**2, 63**2, 20) ** 0.5),
            # np.linspace(-straxen.tpc_z, 0, 21),
        ]
    else:
        raise ValueError(f"Unrecognized experiment {experiment}!")
    rate_multiplier = 1 / eff.sum()
    make_template(
        cs1, cs2, eff, rate_multiplier, output_path, fname, bin_edges=bin_edges
    )


def format_value_uncertainty(val, unc=None):
    """
    Format a value and its uncertainty according to significant-figure rules:
    - If unc is None:
        * Round the value to the nearest integer and return it as a string.
    - Else:
        1. If the first significant digit of the uncertainty is not 1, keep 1 sig fig.
        2. If the first significant digit is 1, keep 2 sig figs.
       The value is rounded to the same decimal place as the uncertainty.
    Returns a string like "1.23 ± 0.04" or "42" if unc is None.
    """
    if unc is None:
        # No uncertainty: round to integer
        return f"{round(val):.0f}"

    if unc <= 0:
        raise ValueError("Uncertainty must be positive")

    # 1) Order of magnitude of the uncertainty
    exp = math.floor(math.log10(unc))
    factor = 10**exp
    scaled_unc = unc / factor

    # 2) Determine sig figs for uncertainty
    sig_digits = 2 if int(scaled_unc) == 1 else 1

    # 3) Round the uncertainty
    new_scaled_unc = round(scaled_unc, sig_digits - 1)
    new_unc = new_scaled_unc * factor

    # 4) Decimal places for formatting
    decimal_places = -exp + (sig_digits - 1)

    # 5) Round the value to match
    new_val = round(val, decimal_places)

    if decimal_places > 0:
        val_str = f"{new_val:.{decimal_places}f}"
        unc_str = f"{new_unc:.{decimal_places}f}"
    else:
        # integer formatting for tens, hundreds, etc.
        val_str = f"{int(new_val)}"
        unc_str = f"{int(new_unc)}"

    # 6) Format output
    return f"{val_str} ± {unc_str}"
