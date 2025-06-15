import importlib.resources
import os
import json
import copy
import importlib
import warnings
import diamx
import appletree as apt
import numpy as np
import jax
from jax import numpy as jnp
from multihist import Histdd
import inference_interface
from diamx.experiment import Experiment
from diamx.utils import make_template, generate_bin_array, csv_to_apt_map, HiddenPrints
from diamx.appletree import get_file_path_diamx
from appletree import ComponentSim
from appletree.config import Map
from scipy.interpolate import interp1d
from appletree import randgen


class XENONnT(Experiment):
    """XENONnT experiment base class. Default to XENONnTSR0, but can be used for other SRs."""

    experiment_name = None
    default_batch_size = int(1e7)
    default_eff_batch_size = int(1e4)
    default_instruct_file = {
        "er_mono": "xenonnt_sr0_er_mono.json",
        "er_flat": "xenonnt_sr0_er_flat.json",
        "er_bkg": "xenonnt_sr0_er_bkg.json",
        "neutron": "xenonnt_sr0_neutron.json",
        "nr": "xenonnt_sr0_wimp.json",
        "s1_recon_eff": "xenonnt_sr0_wimp.json",
    }
    default_yield_file = {
        "er_mono": "xenonnt_sr0_er_model.json",
        "er_flat": "xenonnt_sr0_er_model.json",
        "er_bkg": "xenonnt_sr0_er_model.json",
        "neutron": "xenonnt_sr0_nr_model.json",
        "nr": "xenonnt_sr0_nr_model.json",
        "s1_recon_eff": "xenonnt_sr0_nr_model.json",
    }
    default_param_file = {
        "er_mono": "xenonnt_sr0_er_par.json",
        "er_flat": "xenonnt_sr0_er_par.json",
        "er_bkg": "xenonnt_sr0_er_par.json",
        "neutron": "xenonnt_sr0_nr_par.json",
        "nr": "xenonnt_sr0_nr_par.json",
        "s1_recon_eff": "xenonnt_sr0_nr_par.json",
    }
    default_cut_acc_file = [
        "xenonnt_sr0_cut_acc_median.json",
        "xenonnt_sr0_cut_acc_lower.json",
        "xenonnt_sr0_cut_acc_upper.json",
    ]
    default_eff_file = [
        "xenonnt_sr0_eff_median.csv",
        "xenonnt_sr0_eff_lower.csv",
        "xenonnt_sr0_eff_upper.csv",
    ]
    default_data_file = "xenonnt_sr0_wimp_data.csv"

    def file_path_substitution(self, config):
        """Hack to solve file path problem."""
        new_config = copy.deepcopy(config)
        for key, value in config.items():
            if isinstance(value, list):
                new_value = []
                for value_element in value:
                    try:
                        new_value.append(get_file_path_diamx(value_element))
                    except RuntimeError:
                        new_value.append(value_element)
                new_config[key] = new_value
            elif isinstance(value, str):
                try:
                    new_config[key] = get_file_path_diamx(value)
                except RuntimeError:
                    new_config[key] = value
            else:
                new_config[key] = value
        return new_config

    def run_appletree(
        self,
        batch_size,
        instruct_file_path,
        yield_file_path,
        param_file_path,
        runmode,
        **kwargs,
    ):
        with HiddenPrints():  # suppress annoying appletree print
            supported_runmodes = ["er_mono", "er_flat", "er_bkg", "neutron", "nr"]
            if runmode not in supported_runmodes:
                raise ValueError(
                    f"Unsupported runmode {runmode}. Possible options are {supported_runmodes}."
                )
            with open(instruct_file_path, "r") as instruct_file:
                apt_config = json.load(instruct_file)

            # Keyword argument substitution
            for key, value in kwargs.items():
                apt_config[key] = value

            # Overlap with SR0 best fit yield model
            with open(yield_file_path, "r") as yield_file:
                yield_model = json.load(yield_file)

            param_manager = apt.Parameter(get_file_path_diamx(param_file_path))
            param_manager.sample_prior()
            parameters = param_manager.get_all_parameter()
            for field in yield_model.keys():
                parameters[field] = yield_model[field]

            # Set shifters to 0
            # for ER, set s1_recon_eff shifter to 0
            # for NR, set s1_recon_eff & cut_acc shifter to 0
            s1_recon_eff_shifter_name = "s1_recon_eff_sigma"
            if (
                isinstance(apt_config["s1_recon_eff"], list)
                and len(apt_config["s1_recon_eff"]) == 4
            ):
                s1_recon_eff_shifter_name = apt_config["s1_recon_eff"][3]
            parameters[s1_recon_eff_shifter_name] = 0
            if runmode == "nr":
                cut_acc_shifter_name = "cut_acc_sigma"
                if (
                    isinstance(apt_config["cut_acc"], list)
                    and len(apt_config["cut_acc"]) == 4
                ):
                    cut_acc_shifter_name = apt_config["cut_acc"][3]
                parameters[cut_acc_shifter_name] = 0

            if runmode == "er_mono":
                component = diamx.appletree.ERMonoXENONnT(f"er_mono")
            if runmode == "er_flat":
                component = diamx.appletree.ERFlatXENONnT(f"er_flat")
            elif runmode == "er_bkg":
                component = diamx.appletree.ERBkgXENONnT(f"er_bkg")
            elif runmode == "neutron":
                component = diamx.appletree.NeutronXENONnT(f"neutron")
            elif runmode == "nr":
                component = diamx.appletree.WIMPXENONnT(f"nr")

            component.deduce()

            default_configs = {}
            for plugin_name, _, _ in component.worksheet:
                plugin_cls = eval(f"apt.plugins.{plugin_name}")
                for config_name, config in plugin_cls.takes_config.items():
                    try:
                        default_value = config.get_default()
                        default_configs[config_name] = default_value
                    except ValueError:
                        # No default value set
                        continue
            default_configs.update(apt_config)
            config_with_default = self.file_path_substitution(default_configs)

            apt.share.clear_cache()
            component.set_config(config_with_default)
            component.deduce()
            component.compile()
            key = apt.get_key()
            key, df_apt_sim = component.simulate(key, batch_size, parameters)

            # clear all cache
            apt.share.clear_cache()

        return df_apt_sim

    def _generate_template(self, name, rate, template_file_path, hist_name, **kwargs):
        batch_size = kwargs.get("batch_size", self.default_batch_size)
        supported_bkgs = ["er_mono", "er_flat", "er_bkg", "neutron"]
        if name == "er_mono":
            instruct_file_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file[name])
            )
            yield_file_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file[name])
            )
            param_file_path = get_file_path_diamx(
                kwargs.get("param_file", self.default_param_file[name])
            )
            apt_kwargs = {}
            if "mono_energy" in kwargs:
                if np.isscalar(kwargs["mono_energy"]):
                    apt_kwargs.update({"mono_energy": kwargs["mono_energy"]})
                    cs1, cs2, eff = self.run_appletree(
                        batch_size,
                        instruct_file_path,
                        yield_file_path,
                        param_file_path,
                        "er_mono",
                        **apt_kwargs,
                    )
                elif isinstance(kwargs["mono_energy"], (list, tuple)):
                    if "branching_ratio" not in kwargs:
                        raise ValueError(
                            "Branching ratio must be provided for multiple mono energies."
                        )
                    if len(kwargs["mono_energy"]) != len(kwargs["branching_ratio"]):
                        raise ValueError(
                            "Number of mono energies must match number of branching ratios."
                        )
                    cs1 = []
                    cs2 = []
                    eff = []
                    for mono_energy, branching_ratio in zip(
                        kwargs["mono_energy"], kwargs["branching_ratio"]
                    ):
                        apt_kwargs.update({"mono_energy": mono_energy})
                        cs1_temp, cs2_temp, eff_temp = self.run_appletree(
                            int(batch_size * branching_ratio),
                            instruct_file_path,
                            yield_file_path,
                            param_file_path,
                            "er_mono",
                            **apt_kwargs,
                        )
                        cs1.append(cs1_temp)
                        cs2.append(cs2_temp)
                        eff.append(eff_temp)
                    cs1 = jnp.concatenate(cs1)
                    cs2 = jnp.concatenate(cs2)
                    eff = jnp.concatenate(eff)
                else:
                    raise ValueError("mono_energy must be a scalar or a list.")

        elif name == "er_flat":
            instruct_file_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file[name])
            )
            yield_file_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file[name])
            )
            param_file_path = get_file_path_diamx(
                kwargs.get("param_file", self.default_param_file[name])
            )
            apt_kwargs = {}
            if "upper_energy" in kwargs:
                apt_kwargs.update({"upper_energy": kwargs["upper_energy"]})
            if "lower_energy" in kwargs:
                apt_kwargs.update({"lower_energy": kwargs["lower_energy"]})
            cs1, cs2, eff = self.run_appletree(
                batch_size,
                instruct_file_path,
                yield_file_path,
                param_file_path,
                "er_flat",
                **apt_kwargs,
            )

        elif name == "er_bkg":
            instruct_file_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file[name])
            )
            yield_file_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file[name])
            )
            param_file_path = get_file_path_diamx(
                kwargs.get("param_file", self.default_param_file[name])
            )
            apt_kwargs = {}
            if "energy_spectrum" in kwargs:
                apt_kwargs.update(
                    {
                        "energy_spectrum": csv_to_apt_map(
                            kwargs["energy_spectrum"], "pdf"
                        )
                    }
                )
            cs1, cs2, eff = self.run_appletree(
                batch_size,
                instruct_file_path,
                yield_file_path,
                param_file_path,
                "er_bkg",
                **apt_kwargs,
            )

        elif name == "neutron":
            instruct_file_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file[name])
            )
            yield_file_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file[name])
            )
            param_file_path = get_file_path_diamx(
                kwargs.get("param_file", self.default_param_file[name])
            )

            try:
                for file_name in self.default_cut_acc_file:
                    get_file_path_diamx(file_name)
            except RuntimeError:
                self.calculate_cut_acceptance(
                    batch_size=kwargs.get(
                        "eff_batch_size", self.default_eff_batch_size
                    ),
                    instruct_file_path=self.default_instruct_file["neutron"],
                    yield_file_path=self.default_yield_file["neutron"],
                    param_file_path=self.default_param_file["neutron"],
                )
                for file_name in self.default_cut_acc_file:
                    get_file_path_diamx(file_name)

            cs1, cs2, eff = self.run_appletree(
                batch_size,
                instruct_file_path,
                yield_file_path,
                param_file_path,
                "neutron",
            )

        else:  # NR signal
            instruct_file_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file["nr"])
            )
            yield_file_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file["nr"])
            )
            param_file_path = get_file_path_diamx(
                kwargs.get("param_file", self.default_param_file["nr"])
            )

            try:
                for file_name in self.default_cut_acc_file:
                    get_file_path_diamx(file_name)
            except RuntimeError:
                self.calculate_cut_acceptance(
                    batch_size=kwargs.get(
                        "eff_batch_size", self.default_eff_batch_size
                    ),
                    instruct_file_path=self.default_instruct_file["neutron"],
                    yield_file_path=self.default_yield_file["neutron"],
                    param_file_path=self.default_param_file["neutron"],
                )
                for file_name in self.default_cut_acc_file:
                    get_file_path_diamx(file_name)

            cs1, cs2, eff = self.run_appletree(
                batch_size,
                instruct_file_path,
                yield_file_path,
                param_file_path,
                "nr",
                energy_spectrum=csv_to_apt_map(
                    kwargs["signal_spectrum_path"].format(**kwargs), "pdf"
                ),
            )

        roi = self.config["roi"]
        if "cs1" not in roi or ("cs2" not in roi and "logcs2" not in roi):
            raise ValueError("ROI must contain cs1, cs2/logcs2")
        cs1_bin = generate_bin_array(roi["cs1"])
        if "cs2" in roi:
            cs2_for_binning = cs2
            cs2_bin = generate_bin_array(roi["cs2"])
            cs2_axis_name = "cs2"
        else:
            cs2_for_binning = np.log10(cs2)
            cs2_bin = generate_bin_array(roi["logcs2"])
            cs2_axis_name = "logcs2"
        bin_edges = [cs1_bin, cs2_bin]
        mh = Histdd(
            cs1,
            cs2_for_binning,
            bins=bin_edges,
            weights=eff,
            axis_names=["cs1", cs2_axis_name],
        )
        if name not in supported_bkgs:
            spectrum = np.loadtxt(
                kwargs["signal_spectrum_path"].format(**kwargs), delimiter=","
            )
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
        if self.config["data"] == self.default_data_file:
            data = np.loadtxt(
                importlib.resources.files("diamx") / "data" / self.default_data_file,
                delimiter=",",
            )
        else:
            try:
                data = np.loadtxt(self.config["data"], delimiter=",")
            except FileNotFoundError:
                warnings.warn(
                    f"Specified data path {self.config['data']} not found."
                    f"Using {self.default_data_file} instead."
                )
                data = np.loadtxt(
                    importlib.resources.files("diamx")
                    / "data"
                    / self.default_data_file,
                    delimiter=",",
                )
        cs1 = data[:, 0]
        if "cs2" in self.config["roi"]:
            cs2 = data[:, 1]
            data_alea = np.zeros(
                data.shape[0],
                dtype=[("cs1", "<f8"), ("cs2", "<f8"), ("source", "<i8")],
            )
            data_alea["cs1"] = cs1
            data_alea["cs2"] = cs2
            cs2_bin = generate_bin_array(self.config["roi"]["cs2"])
            roi_mask = data_alea["cs2"] > cs2_bin[0]
            roi_mask &= data_alea["cs2"] < cs2_bin[-1]
        else:
            logcs2 = np.log10(data[:, 1])
            data_alea = np.zeros(
                data.shape[0],
                dtype=[("cs1", "<f8"), ("logcs2", "<f8"), ("source", "<i8")],
            )
            data_alea["cs1"] = cs1
            data_alea["logcs2"] = logcs2
            logcs2_bin = generate_bin_array(self.config["roi"]["logcs2"])
            roi_mask = data_alea["logcs2"] > logcs2_bin[0]
            roi_mask &= data_alea["logcs2"] < logcs2_bin[-1]

        cs1_bin = generate_bin_array(self.config["roi"]["cs1"])
        roi_mask &= data_alea["cs1"] > cs1_bin[0]
        roi_mask &= data_alea["cs1"] < cs1_bin[-1]
        data_alea = data_alea[roi_mask]

        return data_alea

    def calculate_cut_acceptance(self, **kwargs):
        batch_size = kwargs.get("batch_size", self.default_eff_batch_size)
        instruct_file_path = get_file_path_diamx(
            kwargs.get("instruct_file", self.default_instruct_file["nr"])
        )
        yield_file_path = get_file_path_diamx(
            kwargs.get("yield_file", self.default_yield_file["nr"])
        )
        param_file_path = get_file_path_diamx(
            kwargs.get("param_file", self.default_param_file["nr"])
        )

        # Need to load instruct file
        with open(instruct_file_path, "r") as instruct_file:
            instruct = json.load(instruct_file)

        test_energy_range = generate_bin_array(
            kwargs.get("energy_range", "np.linspace(0.01, 100, 51)")
        )
        s1_recon_eff = []

        with open(instruct_file_path, "r") as instruct_file:
            apt_config = json.load(instruct_file)

            # Keyword argument substitution
            for key, value in kwargs.items():
                apt_config[key] = value

            # Overlap with SR0 best fit yield model
            with open(yield_file_path, "r") as yield_file:
                yield_model = json.load(yield_file)

            param_manager = apt.Parameter(get_file_path_diamx(param_file_path))
            param_manager.sample_prior()
            parameters = param_manager.get_all_parameter()
            for field in yield_model.keys():
                parameters[field] = yield_model[field]

            # Set shifters to 0
            s1_recon_eff_shifter_name = "s1_recon_eff_sigma"
            if (
                isinstance(apt_config["s1_recon_eff"], list)
                and len(apt_config["s1_recon_eff"]) == 4
            ):
                s1_recon_eff_shifter_name = apt_config["s1_recon_eff"][3]
            parameters[s1_recon_eff_shifter_name] = 0
            cut_acc_shifter_name = "cut_acc_sigma"
            if (
                isinstance(apt_config["cut_acc"], list)
                and len(apt_config["cut_acc"]) == 4
            ):
                cut_acc_shifter_name = apt_config["cut_acc"][3]
            parameters[cut_acc_shifter_name] = 0

            component = diamx.appletree.S1ReconEffXENONnT(f"eff")
            component.deduce(
                ["acc_s1_recon_eff"], nodep_data_name="energy", force_no_eff=True
            )

            default_configs = {}
            for plugin_name, _, _ in component.worksheet + [
                ["CutAcc", None, None],
                ["EnergyEff", None, None],
            ]:
                plugin_cls = eval(f"apt.plugins.{plugin_name}")
                for config_name, config in plugin_cls.takes_config.items():
                    try:
                        default_value = config.get_default()
                        default_configs[config_name] = default_value
                    except ValueError:
                        # No default value set
                        continue
            default_configs.update(apt_config)
            config_with_default = self.file_path_substitution(default_configs)

            apt.share.clear_cache()
            component.set_config(config_with_default)
            component.deduce(
                ["acc_s1_recon_eff"], nodep_data_name="energy", force_no_eff=True
            )
            component.compile()

        for test_energy in test_energy_range:
            key = apt.get_key()
            key, df_apt_sim = component.simulate(
                key, jnp.ones(batch_size, dtype=float) * test_energy, parameters
            )

            s1_recon_eff.append(np.mean(df_apt_sim))

        s1_recon_eff = np.array(s1_recon_eff)
        s1_recon_eff_zero = s1_recon_eff <= 1e-6

        from appletree.plugins import EnergyEff

        EnergyEff_plugin = EnergyEff("eff")
        key = apt.get_key()
        energy_eff_config = instruct.get("energy_eff", None)
        if (
            energy_eff_config is not None
            and isinstance(energy_eff_config, list)
            and len(energy_eff_config) == 4
        ):
            energy_eff_shifter_name = energy_eff_config[3]
        else:
            energy_eff_shifter_name = "xenonnt_eff_sigma"

        energy_eff_shifters = [0, 1, -1]
        cut_acceptances = []
        for energy_eff_shifter in energy_eff_shifters:
            key, energy_eff = EnergyEff_plugin.simulate(
                key, {energy_eff_shifter_name: 0}, test_energy_range
            )
            cut_acceptance = np.array(energy_eff / s1_recon_eff)
            cut_acceptance[s1_recon_eff_zero] = 0
            cut_acceptances.append(cut_acceptance)

        apt.share.clear_cache()

        # Output result to diamx.appletree
        def save_apt_map(energy, map_data, file_name):
            apt_map_json = dict()
            apt_map_json["coordinate_system"] = energy.tolist()
            apt_map_json["coordinate_type"] = "point"
            apt_map_json["coordinate_name"] = "energy"
            apt_map_json["map"] = map_data.tolist()

            json_file_path = (
                importlib.resources.files("diamx.appletree") / "maps" / file_name
            )
            with open(json_file_path, "w") as f:
                json.dump(apt_map_json, f)

        print("Saving cut acceptance map.")
        for idx, cut_acceptance in enumerate(cut_acceptances):
            save_apt_map(
                test_energy_range, cut_acceptance, self.default_cut_acc_file[idx]
            )

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
            key, probability = randgen.uniform(
                key, 0.0, 1.0, shape=(self.default_eff_batch_size,)
            )
            sampled_energies = np.array(signal_spectrum_map.apply(probability))

            eff_mean = []
            for eff_file in self.default_eff_file:
                eff_data = np.loadtxt(
                    importlib.resources.files(diamx) / "data" / eff_file, delimiter=","
                )
                eff_interpolator = interp1d(
                    eff_data[:, 0],
                    eff_data[:, 1],
                    kind="linear",
                    fill_value=0,
                    bounds_error=False,
                )
                eff_mean.append(np.mean(eff_interpolator(sampled_energies)))
            eff_untertainty[signal_parameter_value] = float(
                np.clip((eff_mean[2] - eff_mean[1]) / eff_mean[0] / 2, 1e-6, None)
            )

            apt.share.clear_cache()

        return eff_untertainty


class XENONnTSR0(XENONnT):
    experiment_name = "xenonnt_sr0"

    def generate_template(self, name, rate, template_file_path, **kwargs):
        if name == "er" or name == "er_flat":
            self._generate_template("er_flat", rate, template_file_path, name, **kwargs)

        elif name == "neutron" or name == "xenonnt_neutron":
            self._generate_template("neutron", rate, template_file_path, name, **kwargs)

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
            self._generate_template(name, rate, template_file_path, name, **kwargs)


class XENONnTSR1(XENONnT):
    experiment_name = None
    default_yield_file = {
        "er_mono": "xenonnt_sr1_er_model.json",
        "er_flat": "xenonnt_sr1_er_model.json",
        "er_bkg": "xenonnt_sr1_er_model.json",
        "neutron": "xenonnt_sr1_nr_model.json",
        "nr": "xenonnt_sr1_nr_model.json",
        "s1_recon_eff": "xenonnt_sr1_nr_model.json",
    }

    def generate_template(self, name, rate, template_file_path, **kwargs):
        if name == "tritium":
            spectrum_path = (
                importlib.resources.files("diamx") / "data" / "tritium_spectrum.csv"
            )
            self._generate_template(
                "er_bkg",
                rate,
                template_file_path,
                name,
                energy_spectrum=spectrum_path,
                **kwargs,
            )

        elif name == "ar37":
            # Considering K-shell and L-shell, not M-shell because energy is too low
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                mono_energy=[2.82, 0.27],
                branching_ratio=[0.902, 0.087],
                **kwargs,
            )

        elif name == "neutron" or name == "xenonnt_neutron":
            self._generate_template("neutron", rate, template_file_path, name, **kwargs)

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

        else:  # er_flat and NR signal
            self._generate_template(name, rate, template_file_path, name, **kwargs)


class XENONnTSR1a(XENONnTSR1):
    experiment_name = "xenonnt_sr1a"
    default_data_file = "xenonnt_sr1a_wimp_data.csv"
    default_instruct_file = {
        "er_mono": "xenonnt_sr0_er_mono.json",
        "er_flat": "xenonnt_sr0_er_flat.json",
        "er_bkg": "xenonnt_sr0_er_bkg.json",
        "neutron": "xenonnt_sr0_neutron.json",
        "nr": "xenonnt_sr1a_wimp.json",
        "s1_recon_eff": "xenonnt_sr1a_wimp.json",
    }
    default_cut_acc_file = [
        "xenonnt_sr1a_cut_acc_median.json",
        "xenonnt_sr1a_cut_acc_lower.json",
        "xenonnt_sr1a_cut_acc_upper.json",
    ]
    default_eff_file = [
        "xenonnt_sr1a_eff_median.csv",
        "xenonnt_sr1a_eff_lower.csv",
        "xenonnt_sr1a_eff_upper.csv",
    ]


class XENONnTSR1b(XENONnTSR1):
    experiment_name = "xenonnt_sr1b"
    default_data_file = "xenonnt_sr1b_wimp_data.csv"
    default_instruct_file = {
        "er_mono": "xenonnt_sr0_er_mono.json",
        "er_flat": "xenonnt_sr0_er_flat.json",
        "er_bkg": "xenonnt_sr0_er_bkg.json",
        "neutron": "xenonnt_sr0_neutron.json",
        "nr": "xenonnt_sr1b_wimp.json",
        "s1_recon_eff": "xenonnt_sr1b_wimp.json",
    }
    default_cut_acc_file = [
        "xenonnt_sr1b_cut_acc_median.json",
        "xenonnt_sr1b_cut_acc_lower.json",
        "xenonnt_sr1b_cut_acc_upper.json",
    ]
    default_eff_file = [
        "xenonnt_sr1b_eff_median.csv",
        "xenonnt_sr1b_eff_lower.csv",
        "xenonnt_sr1b_eff_upper.csv",
    ]
