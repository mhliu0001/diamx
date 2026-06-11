"""PandaX-4T experiments (Run0, Run1) for diamx.

Mirrors ``diamx.experiments.xenonnt`` but with three PandaX-specific differences:

1. **Analysis space** ``(cs1, log10(cs2_b / cs1))`` -- a *ratio* second axis
   (named ``logcs2_s1``), with ``cs2`` the **bottom** S2 (the detector ``g2``
   parameter is PandaX's ``g2_b``). XENONnT uses ``(cs1, log10(cs2))``.
2. **In-chain selection efficiency.** The per-event ``eff`` weight comes directly
   from the appletree chain -- ``eff = cut_acc_s1 * cut_acc_s2 * r(xi) * recon(xi)``
   via ``S1CutAccept`` / ``S2CutAccept`` (Fig. 12) + ``QualityResidualPandaX4T`` /
   ``ReconEffPandaX4T`` (Fig. 15) -> ``EffPandaX4T`` (PRD Eq. 16) -- so there is no
   energy-based cut-acceptance back-calculation like XENONnT's
   ``calculate_cut_acceptance``.
3. **P4-NEST components** (``diamx.appletree.components.{nr,er}_pandax4t``) with
   the hit-clustering photon detection (PRD Eq. 9).

Run1 differs from Run0 by the drift field and drift velocity, the ``g1`` / ``g2_b``
scale factors, the Run1 recombination parameters (the NR Legendre coefficients
``p0..p3_nr`` and the shifts ``d_er`` / ``d_nr``), and its data / maps -- all
carried in the Run1 instruct / model / param JSONs.
"""

import copy
import importlib.resources
import json
import os
import warnings

import diamx
import appletree as apt
import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map
from multihist import Histdd
from scipy.interpolate import interp1d
import inference_interface

from appletree import randgen
from appletree.config import Map
from diamx.experiment import Experiment
from diamx.appletree import get_file_path_diamx
from diamx.utils import make_template, generate_bin_array, csv_to_apt_map, HiddenPrints


def _p4_er_qy(energy, model):
    """ER charge yield Qy [e-/keV] at ``energy`` from the P4-NEST mean ER yields
    (PRD Appendix A1; mirrors ``diamx.appletree.ERYieldParamsP4NEST``). This is
    the bare mean yield (before the Eq. 17 recombination correction); it is used
    only for the quanta-conserving light-yield compensation when a line's charge
    yield is quenched (a sub-percent effect on the line position), so the
    omission of the small Eq. 17 correction is negligible."""
    F = model["field"]
    rho = model["liquid_xe_density"]
    W_eV = model["w"] * 1000.0
    eta = 1.0 + 0.4607 / (1.0 + (F / 621.74) ** (-2.2717)) ** 53.502
    Y0 = 1000.0 / W_eV + 6.5 * (1.0 - 1.0 / (1.0 + (F / 47.408) ** 1.9851))
    Y1 = 32.988 * eta * (
        1.0 - 1.0 / (1.0 + (F / (0.026715 * np.exp(rho / 0.33926))) ** 0.6705)
    )
    tau = (
        1652.264 + (1.415935e10 - 1652.264) / (1.0 + (F / 0.02673144) ** 1.564691)
    ) * energy ** (-2.0)
    return Y1 + (Y0 - Y1) / (1.0 + 1.304 * energy ** 2.1393) ** 0.35535 + 28.0 / (1.0 + tau)


def _p4_quench_ly_scalar(energy, qy_scalar, model):
    """Light-yield scalar that conserves total quanta when a mono-energetic line's
    charge yield is quenched by ``qy_scalar`` (= Q/Q_beta). The PandaX-4T DEC/EC
    quenching is modelled exactly as in XENONnT: ``g2 -> g2 * qy_scalar`` reduces
    the charge, and ``g1 -> g1 * ly_scalar`` puts the freed quanta into light, so
    that ``N_ph + N_e`` (hence the deposited energy) is preserved. For ER,
    ``LY + QY = 1/w``, so ``ly_new = 1/w - QY * qy_scalar``."""
    inv_w = 1.0 / model["w"]
    qy = _p4_er_qy(energy, model)
    return (inv_w - qy * qy_scalar) / (inv_w - qy)


class PandaX4T(Experiment):
    """PandaX-4T base experiment (defaults to Run0 configuration)."""

    experiment_name = None
    default_batch_size = int(1e7)
    default_eff_batch_size = int(1e7)

    # analysis-space second axis is the ratio log10(cs2_b / cs1)
    cs2_axis_name = "logcs2_s1"

    supported_runmodes = ("er_mono", "er_flat", "er_bkg", "neutron", "nr")

    # component class per runmode (resolved on diamx.appletree at call time)
    runmode_component = {
        "er_mono": "ERMonoPandaX4T",
        "er_flat": "ERFlatPandaX4T",
        "er_bkg": "ERBkgPandaX4T",
        "neutron": "NeutronPandaX4T",
        "nr": "WIMPPandaX4T",
    }

    default_instruct_file = {
        "er_mono": "pandax4t_run0_er.json",
        "er_flat": "pandax4t_run0_er.json",
        "er_bkg": "pandax4t_run0_er.json",
        "neutron": "pandax4t_run0_nr.json",
        "nr": "pandax4t_run0_nr.json",
    }
    default_yield_file = {
        "er_mono": "pandax4t_run0_er_model.json",
        "er_flat": "pandax4t_run0_er_model.json",
        "er_bkg": "pandax4t_run0_er_model.json",
        "neutron": "pandax4t_run0_nr_model.json",
        "nr": "pandax4t_run0_nr_model.json",
    }
    default_param_file = {
        "er_mono": "pandax4t_run0_er_par.json",
        "er_flat": "pandax4t_run0_er_par.json",
        "er_bkg": "pandax4t_run0_er_par.json",
        "neutron": "pandax4t_run0_nr_par.json",
        "nr": "pandax4t_run0_nr_par.json",
    }
    # signal (NR) total-efficiency band vs energy (digitized PRD Fig. 15), with the
    # 1-sigma band regularized at the low/high ends (relative diff held constant past
    # the digitized band range -- see notebooks/pandax4t_efficiency.ipynb).
    default_eff_file = [
        "pandax4t_run0_eff_nr_total_median.csv",
        "pandax4t_run0_eff_nr_total_lower_reg.csv",
        "pandax4t_run0_eff_nr_total_upper_reg.csv",
    ]
    default_data_file = "pandax4t_run0_wimp_data.csv"

    # selection-efficiency nuisances set to nominal (0) for template generation
    eff_shifters = ("s1_cut_acc_sigma", "s2_cut_acc_sigma", "elife_sigma")

    # ------------------------------------------------------------------ helpers
    def file_path_substitution(self, config):
        """Resolve map filenames to absolute paths (appletree resources kept as-is)."""
        new_config = copy.deepcopy(config)
        for key, value in config.items():
            if isinstance(value, list):
                new_value = []
                for element in value:
                    try:
                        new_value.append(get_file_path_diamx(element))
                    except RuntimeError:
                        new_value.append(element)
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
        apt_config,
        yield_model,
        param_file_path,
        runmode,
        max_batch_size=int(1e7),
    ):
        """Run the PandaX-4T appletree chain, returning (cs1, cs2_b, eff) arrays."""
        with HiddenPrints():
            if runmode not in self.supported_runmodes:
                raise ValueError(
                    f"Unsupported runmode {runmode}. Options: {self.supported_runmodes}."
                )

            # parameters: priors from the par file, overlaid with the fixed model values
            param_manager = apt.Parameter(get_file_path_diamx(param_file_path))
            param_manager.sample_prior()
            parameters = param_manager.get_all_parameter()
            for field in yield_model.keys():
                parameters[field] = yield_model[field]
            # nominal selection-efficiency / detector nuisances
            for shifter in self.eff_shifters:
                parameters.setdefault(shifter, 0)
                parameters[shifter] = 0

            component_cls = getattr(diamx.appletree, self.runmode_component[runmode])
            component = component_cls(runmode)
            component.deduce()  # default observables (cs1, cs2, eff)

            # assemble config: plugin defaults <- instruct, then resolve paths
            default_configs = {}
            for plugin_name, _, _ in component.worksheet:
                plugin_cls = eval(f"apt.plugins.{plugin_name}")
                for config_name, config in plugin_cls.takes_config.items():
                    try:
                        default_configs[config_name] = config.get_default()
                    except ValueError:
                        continue
            default_configs.update(apt_config)
            config_with_default = self.file_path_substitution(default_configs)

            apt.share.clear_cache()
            component.set_config(config_with_default)
            component.deduce()
            component.compile()
            key = apt.get_key()

            if batch_size <= max_batch_size:
                key, df_apt_sim = component.simulate(key, batch_size, parameters)
            else:
                n_batches = batch_size // max_batch_size
                remainder = batch_size % max_batch_size
                df_apt_sim = None
                for _ in range(n_batches):
                    key, batch = component.simulate(key, max_batch_size, parameters)
                    batch = tree_map(np.asarray, jax.device_get(batch))
                    df_apt_sim = (
                        batch
                        if df_apt_sim is None
                        else tree_map(
                            lambda x, y: np.concatenate([x, y], axis=0), df_apt_sim, batch
                        )
                    )
                if remainder > 0:
                    key, batch = component.simulate(key, remainder, parameters)
                    batch = tree_map(np.asarray, jax.device_get(batch))
                    df_apt_sim = (
                        batch
                        if df_apt_sim is None
                        else tree_map(
                            lambda x, y: np.concatenate([x, y], axis=0), df_apt_sim, batch
                        )
                    )

            apt.share.clear_cache()

        return df_apt_sim  # (cs1, cs2_b, eff)

    # ------------------------------------------------------- template histogram
    def _histogram_and_save(
        self, cs1, cs2, eff, name, rate, template_file_path, hist_name, **kwargs
    ):
        """Bin (cs1, log10(cs2/cs1)) weighted by eff, normalize, and write."""
        roi = self.config["roi"]
        if "cs1" not in roi or self.cs2_axis_name not in roi:
            raise ValueError(f"ROI must contain 'cs1' and '{self.cs2_axis_name}'.")
        cs1_bin = generate_bin_array(roi["cs1"])
        y_bin = generate_bin_array(roi[self.cs2_axis_name])
        with np.errstate(divide="ignore", invalid="ignore"):
            logcs2_s1 = np.log10(np.asarray(cs2) / np.asarray(cs1))
        mh = Histdd(
            np.asarray(cs1),
            logcs2_s1,
            bins=[cs1_bin, y_bin],
            weights=np.asarray(eff),
            axis_names=["cs1", self.cs2_axis_name],
        )
        if name not in ("er_mono", "er_flat", "er_bkg", "neutron"):
            # signal: normalize by the recoil-spectrum integral x fiducial mass
            spectrum = np.loadtxt(
                kwargs["signal_spectrum_path"].format(**kwargs), delimiter=","
            )
            mh.histogram = (
                mh.histogram
                / kwargs.get("batch_size", self.default_batch_size)
                * np.trapz(spectrum[:, 1], spectrum[:, 0])
                * kwargs["fiducial_mass"]
            )
        else:
            if rate is None:
                raise ValueError("rate must be provided for normalizing bkg histogram")
            mh.histogram = mh.histogram / np.sum(mh.histogram) * rate
        inference_interface.multihist_to_template(
            [mh], template_file_path, histogram_names=[hist_name]
        )

    def _quench_yield_model(self, yield_model, energy, qy_scalar):
        """Return ``yield_model`` with g1/g2 rescaled to model a charge yield
        quenched by ``qy_scalar`` = Q/Q_beta at ``energy`` (quanta conserved; see
        ``_p4_quench_ly_scalar``). ``qy_scalar`` None or 1.0 -> returned unchanged."""
        if qy_scalar is None or qy_scalar == 1.0:
            return yield_model
        quenched = copy.deepcopy(yield_model)
        quenched["g1"] = yield_model["g1"] * _p4_quench_ly_scalar(
            energy, qy_scalar, yield_model
        )
        quenched["g2"] = yield_model["g2"] * qy_scalar
        return quenched

    def _generate_template(self, name, rate, template_file_path, hist_name, **kwargs):
        batch_size = kwargs.get("batch_size", self.default_batch_size)

        def load(runmode):
            instruct_path = get_file_path_diamx(
                kwargs.get("instruct_file", self.default_instruct_file[runmode])
            )
            with open(instruct_path) as f:
                apt_config = json.load(f)
            yield_path = get_file_path_diamx(
                kwargs.get("yield_file", self.default_yield_file[runmode])
            )
            with open(yield_path) as f:
                yield_model = json.load(f)
            param_path = kwargs.get("param_file", self.default_param_file[runmode])
            return apt_config, yield_model, param_path

        if name == "er_mono":
            apt_config, yield_model, param_path = load("er_mono")
            mono = kwargs.get("mono_energy")
            branching = kwargs.get("branching_ratio")
            # qy_scalar = Q/Q_beta quenches the line's charge yield: a scalar for a
            # single line, or a list (one per line) for a multi-line source such as
            # the Xe124 DEC (LM + LL shells). None -> no quenching.
            qy_scalar = kwargs.get("qy_scalar")
            if np.isscalar(mono):
                apt_config["mono_energy"] = mono
                cs1, cs2, eff = self.run_appletree(
                    batch_size,
                    apt_config,
                    self._quench_yield_model(yield_model, mono, qy_scalar),
                    param_path,
                    "er_mono",
                )
            else:
                if branching is None or len(branching) != len(mono):
                    raise ValueError("branching_ratio must match mono_energy.")
                if qy_scalar is not None and (
                    np.isscalar(qy_scalar) or len(qy_scalar) != len(mono)
                ):
                    raise ValueError("qy_scalar must be a list matching mono_energy.")
                cs1, cs2, eff = [], [], []
                for idx, (e, br) in enumerate(zip(mono, branching)):
                    apt_config["mono_energy"] = e
                    qs = None if qy_scalar is None else qy_scalar[idx]
                    c1, c2, ef = self.run_appletree(
                        int(batch_size * br),
                        apt_config,
                        self._quench_yield_model(yield_model, e, qs),
                        param_path,
                        "er_mono",
                    )
                    cs1.append(c1); cs2.append(c2); eff.append(ef)
                cs1 = np.concatenate(cs1); cs2 = np.concatenate(cs2); eff = np.concatenate(eff)

        elif name in ("er_flat", "er_bkg", "neutron"):
            apt_config, yield_model, param_path = load(name)
            for opt in ("upper_energy", "lower_energy"):
                if opt in kwargs:
                    apt_config[opt] = kwargs[opt]
            if "energy_spectrum" in kwargs:
                spectrum = str(kwargs["energy_spectrum"])
                # accept a CSV (converted to an apt map) or an existing apt map JSON
                # (e.g. the reused XENONnT neutron recoil spectrum)
                apt_config["energy_spectrum"] = (
                    spectrum if spectrum.endswith(".json")
                    else csv_to_apt_map(spectrum, "pdf")
                )
            cs1, cs2, eff = self.run_appletree(
                batch_size, apt_config, yield_model, param_path, name
            )

        else:  # NR signal
            apt_config, yield_model, param_path = load("nr")
            signal_spectrum_path = kwargs["signal_spectrum_path"].format(**kwargs)
            input_spectrum = np.loadtxt(signal_spectrum_path, delimiter=",")
            if np.any(input_spectrum[:, 1] < 0) or np.all(input_spectrum[:, 1] == 0):
                warnings.warn(
                    f"Invalid input spectrum at {signal_spectrum_path}; zero template."
                )
                cs1 = np.zeros(batch_size); cs2 = np.ones(batch_size); eff = np.zeros(batch_size)
            else:
                apt_config["energy_spectrum"] = csv_to_apt_map(signal_spectrum_path, "pdf")
                cs1, cs2, eff = self.run_appletree(
                    batch_size, apt_config, yield_model, param_path, "nr"
                )

        self._histogram_and_save(
            cs1, cs2, eff, name, rate, template_file_path, hist_name, **kwargs
        )

    # ----------------------------------------------------------------- data / eff
    def get_data(self):
        data_path = self.config["data"]
        if not os.path.exists(data_path):
            if self.config["data"] != self.default_data_file:
                warnings.warn(
                    f"Data path {self.config['data']} not found; using {self.default_data_file}."
                )
            data_path = importlib.resources.files("diamx") / "data" / self.default_data_file
        data = np.loadtxt(data_path, delimiter=",")
        cs1 = data[:, 0]
        cs2_b = data[:, 1]
        logcs2_s1 = np.log10(cs2_b / cs1)
        data_alea = np.zeros(
            data.shape[0],
            dtype=[("cs1", "<f8"), (self.cs2_axis_name, "<f8"), ("source", "<i8")],
        )
        data_alea["cs1"] = cs1
        data_alea[self.cs2_axis_name] = logcs2_s1

        cs1_bin = generate_bin_array(self.config["roi"]["cs1"])
        y_bin = generate_bin_array(self.config["roi"][self.cs2_axis_name])
        mask = (
            (cs1 > cs1_bin[0])
            & (cs1 < cs1_bin[-1])
            & (logcs2_s1 > y_bin[0])
            & (logcs2_s1 < y_bin[-1])
        )
        return data_alea[mask]

    def get_eff_uncertainty(self, signal_config):
        """Relative signal-efficiency uncertainty per signal value, from the net
        efficiency band vs energy (digitized PRD Fig. 15) integrated over each
        recoil spectrum -- same construction as the XENONnT experiment."""
        eff_uncertainty = {}
        for signal_parameter_value in generate_bin_array(
            signal_config["parameter_range"]
        ).tolist():
            signal_spectrum = signal_config["args"]["signal_spectrum_path"].format(
                **{signal_config["parameter_name"]: signal_parameter_value}
            )
            signal_spectrum_map = Map(
                name="signal_spectrum",
                method="LERP",
                default=csv_to_apt_map(signal_spectrum, coordinate_name="pdf"),
            )
            signal_spectrum_map.build()
            key = jax.random.PRNGKey(0)
            key, probability = randgen.uniform(
                key, 0.0, 1.0, shape=(self.default_eff_batch_size,)
            )
            sampled_energies = np.array(signal_spectrum_map.apply(probability))

            eff_mean = []
            for eff_file in self.default_eff_file:
                try:
                    eff_path = get_file_path_diamx(eff_file)
                except RuntimeError:
                    eff_path = importlib.resources.files("diamx") / "data" / eff_file
                eff_data = np.loadtxt(eff_path, delimiter=",")
                interp = interp1d(
                    eff_data[:, 0], eff_data[:, 1], kind="linear",
                    fill_value=0, bounds_error=False,
                )
                eff_mean.append(np.mean(interp(sampled_energies)))
            if eff_mean[0] == 0:
                eff_uncertainty[signal_parameter_value] = 1e-6
                continue
            eff_uncertainty[signal_parameter_value] = float(
                np.clip((eff_mean[2] - eff_mean[1]) / eff_mean[0] / 2, 1e-6, None)
            )
            apt.share.clear_cache()
        return eff_uncertainty


class PandaX4TRun0(PandaX4T):
    experiment_name = "pandax4t_run0"

    def generate_template(self, name, rate, template_file_path, **kwargs):
        if name in ("er", "er_flat", "other_er"):
            self._generate_template("er_flat", rate, template_file_path, name, **kwargs)
        elif name == "tritium":
            spectrum_path = importlib.resources.files("diamx") / "data" / "tritium_spectrum.csv"
            self._generate_template(
                "er_bkg", rate, template_file_path, name,
                energy_spectrum=spectrum_path, **kwargs,
            )
        elif name == "xe124":
            if "ll_quenching_factor" in kwargs or "lm_quenching_factor" in kwargs:
                # DEC model (as in XENONnT and LZ): the LM shell (5.98 keV) and LL
                # shell (10.0 keV), same energies and branching ratios as the other
                # experiments. The DEC charge yield is quenched relative to beta
                # decay: the LM shell shares the Xe127 L-shell ratio
                # Q_L/Q_beta = Q_LM/Q_beta = 0.88 (LZ WS2024, similar drift field),
                # while the LL shell has the lower, free Q_LL/Q_beta (shared with LZ).
                quench = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("lm_quenching_factor", "ll_quenching_factor")
                }
                self._generate_template(
                    "er_mono",
                    rate,
                    template_file_path,
                    name,
                    mono_energy=[5.98, 10.0],
                    branching_ratio=[7.1 / 19.4, 12.3 / 19.4],
                    qy_scalar=[
                        kwargs.get("lm_quenching_factor", 0.88),
                        kwargs.get("ll_quenching_factor", 0.70),
                    ],
                    **quench,
                )
            else:
                # simple single-line treatment (LL shell only, no quenching), as in
                # the PandaX-4T paper -- used by the non-DEC configs.
                self._generate_template(
                    "er_mono", rate, template_file_path, name, mono_energy=10.0, **kwargs
                )
        elif name == "xe127":
            # Xe127 L-shell electron capture (5.2 keV). The DEC configs quench the
            # L-shell charge yield (l_quenching_factor = Q_L/Q_beta = 0.88, as for
            # the Xe124 LM shell); without it the line is unquenched.
            self._generate_template(
                "er_mono",
                rate,
                template_file_path,
                name,
                mono_energy=5.2,
                qy_scalar=kwargs.get("l_quenching_factor", 1.0),
                **{k: v for k, v in kwargs.items() if k != "l_quenching_factor"},
            )
        elif name in ("neutron", "pandax_neutron"):
            # radiogenic-neutron NR background -- reuse the XENONnT neutron recoil
            # spectrum (an appletree map JSON), per the PandaX-4T integration plan
            spectrum_path = get_file_path_diamx("xenonnt_sr0_neutron_spectrum.json")
            self._generate_template(
                "neutron", rate, template_file_path, name,
                energy_spectrum=spectrum_path, **kwargs,
            )
        elif name in ("b8", "8b"):
            # solar 8B CEvNS NR background -- recoil spectrum digitized from the
            # PandaX-4T 8B CEvNS measurement (Phys. Rev. Lett. 133, 191001)
            spectrum_path = importlib.resources.files("diamx") / "data" / "pandax4t_b8_spectrum.csv"
            self._generate_template(
                "neutron", rate, template_file_path, name,
                energy_spectrum=str(spectrum_path), **kwargs,
            )
        elif "template_path" in kwargs:
            if "hist_name" not in kwargs:
                raise ValueError("hist_name must be provided for external templates.")
            self.get_template_from_file(
                name, rate, kwargs["template_path"], kwargs["hist_name"], template_file_path
            )
        else:  # NR signal (WIMP, 8B, ...)
            self._generate_template(name, rate, template_file_path, name, **kwargs)


class PandaX4TRun1(PandaX4TRun0):
    experiment_name = "pandax4t_run1"

    default_instruct_file = {
        "er_mono": "pandax4t_run1_er.json",
        "er_flat": "pandax4t_run1_er.json",
        "er_bkg": "pandax4t_run1_er.json",
        "neutron": "pandax4t_run1_nr.json",
        "nr": "pandax4t_run1_nr.json",
    }
    default_yield_file = {
        "er_mono": "pandax4t_run1_er_model.json",
        "er_flat": "pandax4t_run1_er_model.json",
        "er_bkg": "pandax4t_run1_er_model.json",
        "neutron": "pandax4t_run1_nr_model.json",
        "nr": "pandax4t_run1_nr_model.json",
    }
    default_param_file = {
        "er_mono": "pandax4t_run1_er_par.json",
        "er_flat": "pandax4t_run1_er_par.json",
        "er_bkg": "pandax4t_run1_er_par.json",
        "neutron": "pandax4t_run1_nr_par.json",
        "nr": "pandax4t_run1_nr_par.json",
    }
    default_eff_file = [
        "pandax4t_run1_eff_nr_total_median.csv",
        "pandax4t_run1_eff_nr_total_lower_reg.csv",
        "pandax4t_run1_eff_nr_total_upper_reg.csv",
    ]
    default_data_file = "pandax4t_run1_wimp_data.csv"
