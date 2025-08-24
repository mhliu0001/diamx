import diamx
from diamx.experiment import Experiment
from diamx.nest import lz_model
from diamx.utils import generate_bin_array, csv_to_apt_map
from multihist import Histdd
import inference_interface
import warnings
from scipy.interpolate import interp1d
import importlib.resources
import os
import numpy as np
from appletree.config import Map
from appletree import randgen
import appletree as apt
import jax
from typing import Union, Optional


class LZ(Experiment):
    """LZ experiment base class."""

    experiment_name = None
    default_eff_file = [
        "lz_ws2022_eff_median.csv",
        "lz_ws2022_eff_lower.csv",
        "lz_ws2022_eff_upper.csv",
    ]
    default_data_file = "lz_ws2022_wimp_data.csv"

    def __init__(
        self, config: Union[str, dict], output_path: Optional[str] = "./diamx_output"
    ):
        super().__init__(config, output_path)
        # Prepare a flat ER spectrum
        spectrum_file_path = os.path.join(self.output_path, "flat_er_spectrum.csv")
        if not os.path.exists(spectrum_file_path):
            spectrum = np.ones((101, 2))
            spectrum[:, 0] = np.linspace(0.01, 60, 101)
            np.savetxt(spectrum_file_path, spectrum, delimiter=",")

    def _generate_template(self, name, rate, template_file_path, hist_name, **kwargs):
        batch_size = kwargs.get("batch_size", int(1e7))
        supported_bkgs = ["er_mono", "er_flat", "er_bkg"]
        if name == "er_mono":
            assert (
                "energy" in kwargs
            ), "Energy must be provided for ER monoenergetic bkg"
            energy = kwargs["energy"]
            assert (
                "int_type" in kwargs
            ), "Interaction type must be provided for ER monoenergetic bkg"
            int_type = kwargs["int_type"]
            assert int_type in [
                "ER",
                "EC",
                "DEC",
            ], "Interaction type must be one of ER, EC, DEC for ER monoenergetic bkg"
            if isinstance(energy, dict):
                assert "branching_ratio" in kwargs, (
                    "Branching ratio must be provided for ER monoenergetic bkg "
                    "with multiple energies"
                )
                branching_ratio = kwargs["branching_ratio"]
                quenching_factor = kwargs.get("quenching_factor", {})
                assert isinstance(quenching_factor, dict), (
                    "quenching_factor must be a dict with keys as energy names"
                    "and values as quenching factors"
                )
            else:
                energy = {"default": energy}
                branching_ratio = {"default": 1.0}
                quenching_factor = {
                    "default": kwargs.get(
                        "quenching_factor", 0.87 if int_type == "ER" else 1.0
                    )
                }

            s1c_phd, s2c_phd, eff = [], [], []
            for branch_name, branch_energy in energy.items():
                sim_result = lz_model(
                    self.experiment_name,
                    int_type,
                    0,
                    str(branch_energy),
                    int(batch_size * branching_ratio[branch_name]),
                    quenching_factor.setdefault(
                        branch_name, 1.0 if int_type == "ER" else 0.87
                    ),
                )
                s1c_phd.append(np.array(sim_result.s1c_phd))
                s2c_phd.append(np.array(sim_result.s2c_phd))
                eff.append(
                    np.bitwise_and(
                        np.array(sim_result.s1c_phd) > 0,
                        np.array(sim_result.s2c_phd) > 0,
                    ).astype(float)
                )
            s1c_phd = np.concatenate(s1c_phd)
            s2c_phd = np.concatenate(s2c_phd)
            eff = np.concatenate(eff)

        elif name == "er_flat":
            spectrum_file_path = os.path.join(self.output_path, "flat_er_spectrum.csv")
            assert os.path.exists(spectrum_file_path), (
                f"{spectrum_file_path} not found. Maybe you have deleted it by accident? "
                "Please rerun the program to regenerate."
            )
            sim_result = lz_model(
                self.experiment_name, "ER", 0, spectrum_file_path, batch_size, 1.0
            )
            s1c_phd = np.array(sim_result.s1c_phd)
            s2c_phd = np.array(sim_result.s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)

        elif name == "er_bkg":
            spectrum_file_path = kwargs["spectrum_file_path"]
            if isinstance(spectrum_file_path, dict):
                assert (
                    "component_ratio" in kwargs
                ), "component_ratio must be provided for ER background with multiple components."
                component_ratio = kwargs["component_ratio"]
            else:
                spectrum_file_path = {"default": spectrum_file_path}
                component_ratio = {"default": 1.0}

            s1c_phd, s2c_phd, eff = [], [], []
            for component_name, component_spectrum_file in spectrum_file_path.items():
                assert os.path.exists(
                    component_spectrum_file
                ), f"{component_spectrum_file} not found. Please check the path."
                sim_result = lz_model(
                    self.experiment_name,
                    "ER",
                    0,
                    str(component_spectrum_file),
                    int(batch_size * component_ratio[component_name]),
                    1.0,
                )
                s1c_phd.append(np.array(sim_result.s1c_phd))
                s2c_phd.append(np.array(sim_result.s2c_phd))
                eff.append(
                    np.bitwise_and(
                        np.array(sim_result.s1c_phd) > 0,
                        np.array(sim_result.s2c_phd) > 0,
                    ).astype(float)
                )
            s1c_phd = np.concatenate(s1c_phd)
            s2c_phd = np.concatenate(s2c_phd)
            eff = np.concatenate(eff)

        else:  # NR signal
            spectrum_file_path = kwargs["signal_spectrum_path"].format(**kwargs)
            spectrum = np.loadtxt(spectrum_file_path, delimiter=",")
            if np.any(spectrum[:, 1] < 0) or np.all(spectrum[:, 1] == 0):
                # Generate an empty template with zero histogram
                warnings.warn(
                    f"Invalid input spectrum found at {spectrum_file_path}. "
                    "The template will be generated with zero histogram."
                )
                s1c_phd = np.zeros(batch_size, dtype=np.float64)
                s2c_phd = np.ones(batch_size, dtype=np.float64)
                eff = np.zeros(batch_size, dtype=np.float64)
            else:
                sim_result = lz_model(
                    self.experiment_name, "NR", 0, spectrum_file_path, batch_size, 1.0
                )
                s1c_phd = np.array(sim_result.s1c_phd)
                s2c_phd = np.array(sim_result.s2c_phd)
                eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)
                energy_rec = np.array(sim_result.energy_rec)
                eff_medians = np.loadtxt(
                    importlib.resources.files(diamx)
                    / "data"
                    / self.default_eff_file[0],
                    delimiter=",",
                )
                eff_interpolator = interp1d(
                    eff_medians[:, 0],
                    eff_medians[:, 1],
                    kind="linear",
                    fill_value=0,
                    bounds_error=False,
                )
                eff = eff * eff_interpolator(energy_rec)

        roi = self.config["roi"]
        if "s1c" not in roi or ("s2c" not in roi and "logs2c" not in roi):
            raise ValueError("ROI must contain s1c, s2c/logs2c")
        if "s2c" in roi:
            s2c_for_binning = s2c_phd
            s2c_bin = generate_bin_array(roi["s2c"])
            y_axis_name = "s2c"
        else:
            s2c_for_binning = np.log10(s2c_phd)
            s2c_bin = generate_bin_array(roi["s2c"])
            y_axis_name = "logs2c"
        bin_edges = [generate_bin_array(roi["s1c"]), s2c_bin]
        mh = Histdd(
            s1c_phd,
            s2c_for_binning,
            bins=bin_edges,
            weights=eff,
            axis_names=["s1c", y_axis_name],
        )
        if name not in supported_bkgs:
            mh.histogram = (
                mh.histogram
                / batch_size
                * np.trapz(spectrum[:, 1], spectrum[:, 0])
                * kwargs["fiducial_mass"]
            )
        else:
            if rate is None:
                raise ValueError("rate must be provided for normalizing bkg histogram")
            mh.histogram = mh.histogram / np.sum(mh.histogram) * rate
        inference_interface.multihist_to_template(
            [mh],
            template_file_path,
            histogram_names=[hist_name],
        )

    def get_data(self):
        if os.path.exists(self.config["data"]):
            data = np.loadtxt(self.config["data"], delimiter=",")
        else:
            if self.config["data"] != self.default_data_file:
                warnings.warn(
                    f"Specified data path {self.config['data']} not found. "
                    f"Using {self.default_data_file} instead."
                )
            data = np.loadtxt(
                importlib.resources.files("diamx") / "data" / self.default_data_file,
                delimiter=",",
            )
        s1c = data[:, 0]
        if "s2c" in self.config["roi"]:
            s2c = 10 ** data[:, 1]
            data_alea = np.zeros(
                data.shape[0],
                dtype=[("s1c", "<f8"), ("s2c", "<f8"), ("source", "<i8")],
            )
            data_alea["s1c"] = s1c
            data_alea["s2c"] = s2c
        else:
            logs2c = data[:, 1]
            data_alea = np.zeros(
                data.shape[0],
                dtype=[("s1c", "<f8"), ("logs2c", "<f8"), ("source", "<i8")],
            )
            data_alea["s1c"] = s1c
            data_alea["logs2c"] = logs2c

        return data_alea

    def get_eff_uncertainty(self, signal_config):
        eff_untertainty = {}
        for signal_parameter_value in generate_bin_array(
            signal_config["parameter_range"]
        ).tolist():
            signal_spectrum = signal_config["args"]["signal_spectrum_path"].format(
                **{signal_config["parameter_name"]: signal_parameter_value}
            )
            signal_spectrum_apt = csv_to_apt_map(signal_spectrum, coordinate_name="pdf")
            signal_spectrum_map = Map(
                name="signal_spectrum", method="LERP", default=signal_spectrum_apt
            )
            signal_spectrum_map.build()
            key = jax.random.PRNGKey(0)
            key, probability = randgen.uniform(key, 0.0, 1.0, shape=(1000000,))
            sampled_energies = np.array(signal_spectrum_map.apply(probability))

            eff_lower = np.loadtxt(
                importlib.resources.files(diamx) / "data" / self.default_eff_file[1],
                delimiter=",",
            )
            eff_median = np.loadtxt(
                importlib.resources.files(diamx) / "data" / self.default_eff_file[0],
                delimiter=",",
            )
            eff_upper = np.loadtxt(
                importlib.resources.files(diamx) / "data" / self.default_eff_file[2],
                delimiter=",",
            )
            eff_lower_interpolator = interp1d(
                eff_lower[:, 0],
                eff_lower[:, 1],
                kind="linear",
                fill_value=0,
                bounds_error=False,
            )
            eff_median_interpolator = interp1d(
                eff_median[:, 0],
                eff_median[:, 1],
                kind="linear",
                fill_value=0,
                bounds_error=False,
            )
            eff_upper_interpolator = interp1d(
                eff_upper[:, 0],
                eff_upper[:, 1],
                kind="linear",
                fill_value=0,
                bounds_error=False,
            )

            eff_lower_mean = np.mean(eff_lower_interpolator(sampled_energies))
            eff_median_mean = np.mean(eff_median_interpolator(sampled_energies))
            eff_upper_mean = np.mean(eff_upper_interpolator(sampled_energies))
            eff_untertainty[signal_parameter_value] = float(
                np.clip(
                    (eff_upper_mean - eff_lower_mean) / eff_median_mean / 2, 0, None
                )
            )

            apt.share.clear_cache()

        return eff_untertainty


class LZWS2022(LZ):
    experiment_name = "lz_ws2022"
    default_eff_file = [
        "lz_ws2022_eff_median.csv",
        "lz_ws2022_eff_lower.csv",
        "lz_ws2022_eff_upper.csv",
    ]
    default_data_file = "lz_ws2022_wimp_data.csv"

    def generate_template(self, name, rate, template_file_path, **kwargs):
        supported_bkgs = ["beta", "neutrino", "xe124", "xe127", "ar37", "xe136"]
        flat_er_name = ["beta", "neutrino"]
        if name in flat_er_name:
            self._generate_template("er_flat", rate, template_file_path, name, **kwargs)

        elif name == "xe124":
            branching_ratio = {"lm": 7.1 / 19.4, "ll": 12.3 / 19.4}
            energy = {"lm": 5.98, "ll": 10.0}
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy=energy,
                int_type="DEC",
                branching_ratio=branching_ratio,
                quenching_factor={"lm": 0.87, "ll": 0.87},
                **kwargs,
            )

        elif name == "xe127":
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy=5.2,
                int_type="EC",
                quenching_factor=0.87,
                **kwargs,
            )

        elif name == "ar37":
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy={"k-shell": 2.82, "m-shell": 0.27},
                branching_ratio={"k-shell": 0.902, "m-shell": 0.087},
                int_type="ER",
                **kwargs,
            )

        elif name == "xe136":
            # https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors
            spectrum_file_path = (
                importlib.resources.files(diamx) / "data" / "xe136_spectrum.csv"
            )
            self._generate_template(
                "er_bkg",
                rate,
                template_file_path,
                name,
                spectrum_file_path=spectrum_file_path,
                **kwargs,
            )

        elif "template_path" in kwargs:
            if "hist_name" not in kwargs:
                raise ValueError(f"hist_name must be provided for external templates.")
            self.get_template_from_file(
                name,
                rate,
                kwargs["template_path"],
                kwargs["hist_name"],
                template_file_path,
            )

        else:  # NR signal
            self._generate_template(
                name,
                rate,
                template_file_path,
                name,
                **kwargs,
            )


class LZWS2024(LZ):
    experiment_name = "lz_ws2024"
    default_eff_file = [
        "lz_ws2024_eff_median.csv",
        "lz_ws2024_eff_lower.csv",
        "lz_ws2024_eff_upper.csv",
    ]
    default_data_file = "lz_ws2024_wimp_data.csv"

    def generate_template(self, name, rate, template_file_path, **kwargs):
        batch_size = kwargs.get("batch_size", int(1e7))
        supported_bkgs = [
            "pb214",
            "kr85_ar39_detgamma",
            "solar_neutrino_er",
            "pb212_po218",
            "tritium_c14",
            "xe124",
            "xe127_xe125",
            "ar37",
            "xe136",
        ]
        flat_er_name = [
            "pb214",
            "kr85_ar39_detgamma",
            "solar_neutrino_er",
            "pb212_po218",
        ]
        if name in flat_er_name:
            # Check flat er spectrum
            self._generate_template("er_flat", rate, template_file_path, name, **kwargs)

        elif name == "tritium_c14":
            # Tritium: https://link.springer.com/article/10.1140/epjc/s10052-019-6686-7/figures/1
            # C-14: BetaShape https://github.com/IAEA-NSDDNetwork/BetaShape/
            component_ratio = {"tritium": 8 / 9, "c14": 1 / 9}
            spectrums = {
                "tritium": str(
                    importlib.resources.files(diamx) / "data" / "tritium_spectrum.csv"
                ),
                "c14": str(
                    importlib.resources.files(diamx) / "data" / "c14_spectrum.csv"
                ),
            }
            self._generate_template(
                "er_bkg",
                rate,
                template_file_path,
                name,
                spectrum_file_path=spectrums,
                component_ratio=component_ratio,
                **kwargs,
            )

        elif name == "xe124":
            # DEC model
            branching_ratio = {"lm": 7.1 / 19.4, "ll": 12.3 / 19.4}
            energy = {"lm": 5.98, "ll": 10.0}
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy=energy,
                int_type="DEC",
                branching_ratio=branching_ratio,
                quenching_factor={
                    "lm": 0.87,
                    "ll": kwargs["dec_quenching_factor"] / 100,
                },
                **kwargs,
            )

        elif name == "xe127_xe125":
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy=5.2,
                int_type="EC",
                quenching_factor=0.87,
                **kwargs,
            )

        elif name == "ar37":
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                energy={"k-shell": 2.82, "m-shell": 0.27},
                branching_ratio={"k-shell": 0.902, "m-shell": 0.087},
                int_type="ER",
                **kwargs,
            )

        elif name == "xe136":
            # https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors
            spectrum_file_path = (
                importlib.resources.files(diamx) / "data" / "xe136_spectrum.csv"
            )
            self._generate_template(
                "er_bkg",
                rate,
                template_file_path,
                name,
                spectrum_file_path=spectrum_file_path,
                **kwargs,
            )

        elif "template_path" in kwargs:
            if "hist_name" not in kwargs:
                raise ValueError(f"hist_name must be provided for external templates.")
            self.get_template_from_file(
                name,
                rate,
                kwargs["template_path"],
                kwargs["hist_name"],
                template_file_path,
            )

        else:
            self._generate_template(
                name,
                rate,
                template_file_path,
                name,
                **kwargs,
            )
