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


class LZWS2024(Experiment):
    experiment_name = "lz_ws2024"

    def __init__(self, config, output_path):
        super().__init__(config, output_path)
        # Prepare a flat ER spectrum
        spectrum_file_path = os.path.join(self.output_path, "flat_er_spectrum.csv")
        if not os.path.exists(spectrum_file_path):
            spectrum = np.ones((101, 2))
            spectrum[:, 0] = np.linspace(0.01, 60, 101)
            np.savetxt(spectrum_file_path, spectrum, delimiter=",")

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
            spectrum_file_path = os.path.join(self.output_path, "flat_er_spectrum.csv")
            if not os.path.exists(spectrum_file_path):
                raise RuntimeError(
                    f"{spectrum_file_path} not found. Maybe you have deleted it by accident? "
                    "Please rerun the program to regenerate."
                )

            sim_result = lz_model(
                self.experiment_name, "ER", 0, spectrum_file_path, batch_size, 1.0
            )
            s1c_phd = sim_result.s1c_phd
            s2c_phd = sim_result.s2c_phd
            eff = np.bitwise_and(np.array(s1c_phd) > 0, np.array(s2c_phd) > 0)

        elif name == "tritium_c14":
            # Tritium: https://link.springer.com/article/10.1140/epjc/s10052-019-6686-7/figures/1
            # C-14: BetaShape https://github.com/IAEA-NSDDNetwork/BetaShape/
            element_ratio = {"tritium": 8 / 9, "c14": 1 / 9}
            spectrums = {
                "tritium": str(
                    importlib.resources.files(diamx) / "data" / "tritium_spectrum.csv"
                ),
                "c14": str(
                    importlib.resources.files(diamx) / "data" / "c14_spectrum.csv"
                ),
            }
            s1c_phd = []
            s2c_phd = []
            for element in ["tritium", "c14"]:
                sim_result = lz_model(
                    self.experiment_name,
                    "ER",
                    0,
                    spectrums[element],
                    int(batch_size * element_ratio[element]),
                    1.0,
                )
                s1c_phd.append(sim_result.s1c_phd)
                s2c_phd.append(sim_result.s2c_phd)
            s1c_phd = np.concatenate(s1c_phd)
            s2c_phd = np.concatenate(s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)

        elif name == "xe124":
            # DEC model
            branching_ratio = {"lm": 7.1 / 19.4, "ll": 12.3 / 19.4}
            energies = {"lm": 5.98, "ll": 10.0}
            s1c_phd = []
            s2c_phd = []
            for decay_mode in ["lm", "ll"]:
                if decay_mode == "lm":
                    dec_quenching_factor = 0.87
                else:
                    dec_quenching_factor = kwargs["dec_quenching_factor"] / 100
                sim_result = lz_model(
                    self.experiment_name,
                    "DEC",
                    0,
                    str(energies[decay_mode]),
                    int(batch_size * branching_ratio[decay_mode]),
                    dec_quenching_factor,
                )
                s1c_phd.append(sim_result.s1c_phd)
                s2c_phd.append(sim_result.s2c_phd)
            s1c_phd = np.concatenate(s1c_phd)
            s2c_phd = np.concatenate(s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)

        elif name in ["xe127_xe125", "ar37"]:
            # monoenergetic line
            energies = {"xe127_xe125": 5.2, "ar37": 2.82}
            int_type = {"xe127_xe125": "EC", "ar37": "beta"}
            quenching_factor = {"xe127_xe125": 0.87, "ar37": 1.0}
            sim_result = lz_model(
                self.experiment_name,
                int_type[name],
                0,
                str(energies[name]),
                batch_size,
                quenching_factor[name],
            )
            s1c_phd = np.array(sim_result.s1c_phd)
            s2c_phd = np.array(sim_result.s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)

        elif name == "xe136":
            # https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors
            spectrum_file_path = (
                importlib.resources.files(diamx) / "data" / "xe136_spectrum.csv"
            )
            sim_result = lz_model(
                self.experiment_name, "ER", 0, str(spectrum_file_path), batch_size, 1.0
            )
            s1c_phd = np.array(sim_result.s1c_phd)
            s2c_phd = np.array(sim_result.s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)

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
            return

        else:  # NR signal
            spectrum_file_path = kwargs["signal_spectrum_path"].format(**kwargs)
            spectrum = np.loadtxt(spectrum_file_path, delimiter=",")
            if spectrum_file_path is None:
                raise ValueError(
                    f"spectrum_file_path is required for NR signal {name}."
                )
            sim_result = lz_model(
                self.experiment_name, "NR", 0, spectrum_file_path, batch_size, 1.0
            )
            s1c_phd = np.array(sim_result.s1c_phd)
            s2c_phd = np.array(sim_result.s2c_phd)
            eff = np.bitwise_and(s1c_phd > 0, s2c_phd > 0).astype(float)
            energy_rec = np.array(sim_result.energy_rec)
            eff_medians = np.loadtxt(
                importlib.resources.files(diamx) / "data" / "lz_ws2024_eff_median.csv",
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
            histogram_names=[name],
        )

    def get_data(self):
        if self.config["data"] == "lz_ws2024_wimp_data.csv":
            data = np.loadtxt(
                importlib.resources.files("diamx")
                / "data"
                / f"lz_ws2024_wimp_data.csv",
                delimiter=",",
            )
        else:
            try:
                data = np.loadtxt(self.config["data"], delimiter=",")
            except FileNotFoundError:
                warnings.warn(
                    f"Specified data path {self.config['data']} not found."
                    "Using lz_ws2024_wimp_data.csv instead."
                )
                data = np.loadtxt(
                    importlib.resources.files("diamx")
                    / "data"
                    / f"lz_ws2024_wimp_data.csv",
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
            signal_spectrum_apt = csv_to_apt_map(signal_spectrum)
            signal_spectrum_map = Map(
                name="signal_spectrum", method="LERP", default=signal_spectrum_apt
            )
            signal_spectrum_map.build()
            key = jax.random.PRNGKey(0)
            key, probability = randgen.uniform(key, 0.0, 1.0, shape=(1000000,))
            sampled_energies = np.array(signal_spectrum_map.apply(probability))

            eff_lower = np.loadtxt(
                importlib.resources.files(diamx) / "data" / "lz_ws2024_eff_lower.csv",
                delimiter=",",
            )
            eff_median = np.loadtxt(
                importlib.resources.files(diamx) / "data" / "lz_ws2024_eff_median.csv",
                delimiter=",",
            )
            eff_upper = np.loadtxt(
                importlib.resources.files(diamx) / "data" / "lz_ws2024_eff_upper.csv",
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
