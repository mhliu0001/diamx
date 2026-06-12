import gc
import json
from json import JSONDecodeError
import os
import diamx.experiment
import numpy as np
import pandas as pd
from diamx.utils import (
    HiddenPrints,
    ignore_warning,
    csv_to_apt_map,
    make_template,
    get_shape_parameter_config,
    get_local_pdf_from_template,
    replace_alias,
)
from glob import glob
import yaml
import copy
import warnings
from tqdm import tqdm
from appletree.utils import integrate_midpoint
from appletree.share import _cached_functions
from scipy.stats import norm, chi2
from inference_interface import template_to_multihist
import multihist as mh
from alea.utils import signal_multiplier_estimator
from diamx.model import DiamxModel
from blueice.utils import arrays_to_grid
from scipy.interpolate import RegularGridInterpolator

import diamx
from diamx.utils import (
    create_hash,
    generate_bin_array,
    template_folder,
    HiddenTqdm,
    format_value_uncertainty,
)
from matplotlib import pyplot as plt


class Context(object):
    """context manages all experiments and performs inference"""

    registered_experiments = {}

    def __init__(self, config, output_path="./diamx_output"):
        if isinstance(config, str):
            try:
                with open(config, "r") as config_file:
                    self.config = json.load(config_file)
            except OSError:
                raise ValueError(f"Context config file {config} cannot be opened.")
            except JSONDecodeError:
                raise ValueError(
                    f"Context config file {config} is not a valid JSON file."
                )
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Context config must be str or dict.")
        self.check_config_sanity()
        self.output_path = output_path

        os.makedirs(os.path.join(output_path, template_folder), exist_ok=True)

    def prepare(self):
        self.experiment_instances = []
        for experiment_config in self.config["experiments"]:
            try:
                self.experiment_instances.append(
                    self.registered_experiments[experiment_config["experiment_name"]](
                        experiment_config, self.output_path
                    )
                )
            except KeyError:
                raise ValueError(
                    f"Missing experiment name or experiment not registered"
                    f"for config: {experiment_config}"
                )

    def generate_templates(self):
        self.prepare()
        for experiment_instance in self.experiment_instances:
            experiment_instance.get_bkg_templates()
            experiment_instance.get_shaped_bkg_templates()
            experiment_instance.get_signal_templates(self.config["signal"])

    @staticmethod
    def _get_rate_name(bkg_config, experiment_name=None):
        """Internal method to get the right parameter name for rate_multiplier."""
        if "bkg_name" in bkg_config:
            bkg_name = bkg_config["bkg_name"]
        elif "shaped_bkg_name" in bkg_config:
            bkg_name = bkg_config["shaped_bkg_name"]
        if bkg_config.get("shared_rate", False):
            return f"{bkg_name}_rate_multiplier"
        else:
            if experiment_name is None:
                raise ValueError(
                    "experiment_name must be provided unless shared_rate=True"
                )
            return f"{experiment_name}_{bkg_name}_rate_multiplier"

    @staticmethod
    def _get_shape_parameter_name(shape_parameter_config, experiment_name=None):
        """Internal method to get the right parameter name for shape_parameter."""
        if shape_parameter_config.get("shape_parameter_shared", False):
            return shape_parameter_config["shape_parameter_name"]
        else:
            if experiment_name is None:
                raise ValueError(
                    "experiment_name must be provided unless shape_parameter_shared=True"
                )
            return f"{experiment_name}_{shape_parameter_config['shape_parameter_name']}"

    def generate_alea_config(self):
        """Generate configuration file for combined fit with placeholder signal."""
        alea_config = {}
        alea_config["parameter_definition"] = {}
        alea_config["likelihood_config"] = {}
        alea_config["likelihood_config"]["template_folder"] = None
        likelihood_terms = []

        # Signal parameters
        signal_name = self.config["signal"]["signal_name"]
        signal_parameter_name = self.config["signal"]["parameter_name"]
        alea_config["parameter_definition"][f"{signal_name}_rate_multiplier"] = {
            "nominal_value": 1.0,  # placeholder
            "ptype": "rate",
            "fittable": True,
            "fit_limits": self.config["signal"]["rate_fit_limits"],
            "parameter_interval_bounds": self.config["signal"][
                "rate_parameter_interval_bounds"
            ],
            "description": f"Rate multiplier for signal {signal_name}",
        }
        alea_config["parameter_definition"][signal_parameter_name] = {
            "nominal_value": 1.0,  # placeholder
            "ptype": "needs_reinit",
            "fittable": False,
            "description": f"Parameter {signal_parameter_name} for signal {signal_name}",
        }
        for experiment_instance in self.experiment_instances:
            experiment_name = experiment_instance.experiment_name

            # Livetime
            alea_config["parameter_definition"][f"{experiment_name}_livetime"] = {
                "nominal_value": experiment_instance.config["livetime"],
                "ptype": "livetime",
                "fittable": False,
                "description": f"Livetime of {experiment_name} in years",
            }

            # Rate
            for bkg_config in (
                experiment_instance.config["bkgs"]
                + experiment_instance.config["shaped_bkgs"]
            ):
                if "bkg_name" in bkg_config:
                    bkg_name = bkg_config["bkg_name"]
                elif "shaped_bkg_name" in bkg_config:
                    bkg_name = bkg_config["shaped_bkg_name"]
                if "rate_fit_limits" not in bkg_config:
                    rate_fit_limits = [0, None]
                else:
                    rate_fit_limits = []
                    for rate_fit_limit in bkg_config["rate_fit_limits"]:
                        if rate_fit_limit is None:
                            rate_fit_limits.append(None)
                        else:
                            rate_fit_limits.append(
                                rate_fit_limit / bkg_config["rate_nominal"]
                            )
                rate_config = {
                    "nominal_value": 1.0,
                    "ptype": "rate",
                    "fittable": bkg_config.get("rate_fittable", True),
                    "fit_limits": rate_fit_limits,
                    "fit_guess": 1.0,
                    "description": f"Rate of {bkg_name} background in {experiment_name} (events per year)",
                }
                if "rate_uncertainty" in bkg_config:
                    if bkg_config.get("rate_relative_uncertainty", False):
                        rate_config["uncertainty"] = bkg_config["rate_uncertainty"]
                    else:
                        rate_config["uncertainty"] = (
                            bkg_config["rate_uncertainty"] / bkg_config["rate_nominal"]
                        )

                rate_name = self._get_rate_name(bkg_config, experiment_name)
                if bkg_config.get("shared_rate", False):
                    if rate_name in alea_config["parameter_definition"]:
                        # Check whether the definition is consistent
                        old_def = alea_config["parameter_definition"][rate_name]

                        # Compare all keys except "description"
                        old_keys = set(old_def) - {"description"}
                        new_keys = set(rate_config) - {"description"}
                        if old_keys != new_keys:
                            raise ValueError(
                                f"Inconsistent shared-rate keys for '{rate_name}': "
                                f"old={old_keys}, new={new_keys}"
                            )
                        for key in old_keys:
                            if old_def[key] != rate_config[key]:
                                raise ValueError(
                                    f"Inconsistent shared-rate definition for '{rate_name}' "
                                    f"(key='{key}'): old={old_def[key]}, "
                                    f"new={rate_config[key]}"
                                )

                        # Update description
                        rate_config["description"] = (
                            f"Shared rate for {bkg_name} background"
                        )
                alea_config["parameter_definition"][rate_name] = rate_config

            # Shape parameter
            for shaped_bkg_config in experiment_instance.config["shaped_bkgs"]:
                for shape_parameter_config in shaped_bkg_config["shape_parameters"]:
                    shape_parameter_alea_config = get_shape_parameter_config(
                        shaped_bkg_config["shaped_bkg_name"],
                        shape_parameter_config["shape_parameter_nominal"],
                        shape_parameter_config.get("shape_parameter_fittable", True),
                        shape_parameter_config.get("shape_parameter_range", None),
                        shape_parameter_config.get("shape_parameter_fit_limits", None),
                        shape_parameter_config.get("shape_parameter_uncertainty", None),
                        shape_parameter_config.get(
                            "shape_parameter_relative_uncertainty", False
                        ),
                        experiment_name,
                    )
                    if shape_parameter_config.get("shape_parameter_shared", False):
                        shape_parameter_name = shape_parameter_config[
                            "shape_parameter_name"
                        ]
                        if shape_parameter_name in alea_config["parameter_definition"]:
                            # Check whether the definition is consistent
                            old_def = alea_config["parameter_definition"][
                                shape_parameter_name
                            ]
                            # Compare all keys except "description"
                            old_keys = set(old_def) - {"description"}
                            new_keys = set(shape_parameter_alea_config) - {
                                "description"
                            }
                            if old_keys != new_keys:
                                raise ValueError(
                                    f"Inconsistent shared-shape-parameter keys for '{shape_parameter_name}': "
                                    f"old={old_keys}, new={new_keys}"
                                )
                            for key in old_keys:
                                if old_def[key] != shape_parameter_alea_config[key]:
                                    raise ValueError(
                                        f"Inconsistent shared-shape-parameter definition for '{shape_parameter_name}' "
                                        f"(key='{key}'): old={old_def[key]}, "
                                        f"new={shape_parameter_alea_config[key]}"
                                    )
                            # Update description
                            shape_parameter_alea_config["description"] = (
                                f"Shared shape parameter for {shaped_bkg_config['shaped_bkg_name']} background"
                            )

                    else:
                        shape_parameter_name = f"{experiment_name}_{shape_parameter_config['shape_parameter_name']}"
                    alea_config["parameter_definition"][
                        shape_parameter_name
                    ] = shape_parameter_alea_config

            # Efficiency
            alea_config["parameter_definition"][
                f"{experiment_name}_signal_efficiency"
            ] = {
                "conditioning_parameter_name": self.config["signal"]["parameter_name"],
                "nominal_value": 1.0,
                "ptype": "efficiency",
                "uncertainty": experiment_instance.get_eff_uncertainty(
                    self.config["signal"]
                ),
                "relative_uncertainty": True,
                "fittable": experiment_instance.config["eff"].get("fittable", True),
                "fit_limits": experiment_instance.config["eff"]["fit_limits"],
                "parameter_interval_bounds": experiment_instance.config["eff"][
                    "parameter_interval_bounds"
                ],
                "fit_guess": 1.0,
                "description": f"Efficiency uncertainty for signal given a cross-section for {experiment_name}",
            }

            # Likelihood
            experiment_likelihood = {
                "name": experiment_name,
                "default_source_class": "alea.template_source.TemplateSource",
                "likelihood_type": "blueice.likelihood.UnbinnedLogLikelihood",
                "analysis_space": [
                    {key: value}
                    for key, value in experiment_instance.config["roi"].items()
                ],
                "in_events_per_bin": True,
                "livetime_parameter": f"{experiment_name}_livetime",
                "slice_args": {},
                "sources": [],
            }
            experiment_sources = []

            # Bkg components
            for bkg_config in experiment_instance.config["bkgs"]:
                args = bkg_config.get("args", {})
                file_hash = create_hash(experiment_instance.config["roi"], **args)
                template_file_name = (
                    f"{experiment_name}_bkg_{bkg_config['bkg_name']}"
                    f"_{file_hash}.ii.h5"
                )
                template_file_path = os.path.join(
                    self.output_path, template_folder, template_file_name
                )
                experiment_sources.append(
                    {
                        "name": (
                            bkg_config["bkg_name"]
                            if bkg_config.get("shared_rate", False)
                            else f"{experiment_name}_{bkg_config['bkg_name']}"
                        ),
                        "histname": replace_alias(bkg_config)["bkg_name"],
                        "parameters": [
                            self._get_rate_name(bkg_config, experiment_name)
                        ],
                        "template_filename": os.path.abspath(template_file_path),
                    }
                )

            # Shaped bkg components
            for shaped_bkg_config in experiment_instance.config["shaped_bkgs"]:
                args = shaped_bkg_config.get("args", {})
                file_hash = create_hash(experiment_instance.config["roi"], **args)
                template_suffix_parts = []
                for shape_parameter_config in shaped_bkg_config["shape_parameters"]:
                    # shape_parameter_name = (
                    #     f"{experiment_name}_{shape_parameter_config['shape_parameter_name']}"
                    #     if not shape_parameter_config.get(
                    #         "shape_parameter_shared", False
                    #     )
                    #     else shape_parameter_config["shape_parameter_name"]
                    # )
                    shape_parameter_name = self._get_shape_parameter_name(
                        shape_parameter_config, experiment_name
                    )
                    template_suffix_parts.append(
                        f"{shape_parameter_name}_{{{shape_parameter_name}:{shape_parameter_config['formatter']}}}"
                    )
                template_suffix = "_".join(template_suffix_parts)

                template_file_name = (
                    f"{experiment_name}_shaped_bkg_{shaped_bkg_config['shaped_bkg_name']}"
                    f"_{file_hash}_{template_suffix}.ii.h5"
                )
                template_file_path = os.path.join(
                    self.output_path, template_folder, template_file_name
                )
                shape_parameter_list = [
                    (
                        # f"{experiment_name}_{shaped_parameter_config['shape_parameter_name']}"
                        # if not shaped_parameter_config.get(
                        #     "shape_parameter_shared", False
                        # )
                        # else shaped_parameter_config["shape_parameter_name"]
                        self._get_shape_parameter_name(
                            shape_parameter_config, experiment_name
                        )
                    )
                    for shape_parameter_config in shaped_bkg_config["shape_parameters"]
                ]
                experiment_sources.append(
                    {
                        "name": (
                            shaped_bkg_config["shaped_bkg_name"]
                            if shaped_bkg_config.get("shared_rate", False)
                            else f"{experiment_name}_{shaped_bkg_config['shaped_bkg_name']}"
                        ),
                        "histname": replace_alias(shaped_bkg_config)["shaped_bkg_name"],
                        "parameters": [
                            self._get_rate_name(shaped_bkg_config, experiment_name),
                        ]
                        + shape_parameter_list,
                        "named_parameters": shape_parameter_list,
                        "template_filename": os.path.abspath(template_file_path),
                    }
                )

            # Signal component
            experiment_sources.append(
                {
                    "name": self.config["signal"]["signal_name"],
                    "histname": replace_alias(self.config["signal"])["signal_name"],
                    "parameters": [
                        f"{self.config['signal']['signal_name']}_rate_multiplier",
                        f"{self.config['signal']['parameter_name']}",
                    ],
                    "template_filename": "",  # placeholder
                }
            )

            experiment_likelihood["sources"] = experiment_sources
            likelihood_terms.append(experiment_likelihood)

        alea_config["likelihood_config"]["likelihood_terms"] = likelihood_terms
        self._cached_alea_config = alea_config

        return alea_config

    def update_alea_config_signal(self, signal_parameter_value):
        try:
            alea_config = copy.deepcopy(self._cached_alea_config)
        except AttributeError:
            alea_config = copy.deepcopy(self.generate_alea_config())
        signal_config = self.config["signal"]
        signal_name = signal_config["signal_name"]
        signal_parameter_name = signal_config["parameter_name"]
        alea_config["parameter_definition"][signal_parameter_name][
            "nominal_value"
        ] = signal_parameter_value

        # Signal template file name & signal rate multiplier
        args = copy.deepcopy(signal_config.get("args", {}))
        args[signal_parameter_name] = signal_parameter_value

        signal_rate_multiplier = []
        max_rate_multiplier = 0  # To check if the expected events is too large
        for index, experiment_instance in enumerate(self.experiment_instances):
            args["fiducial_mass"] = experiment_instance.config["fiducial_mass"]
            file_hash = create_hash(experiment_instance.config["roi"], **args)
            template_file_name = (
                f"{experiment_instance.experiment_name}_signal_{signal_name}"
                f"_{file_hash}.ii.h5"
            )
            template_file_path = os.path.join(
                self.output_path, template_folder, template_file_name
            )
            alea_config["likelihood_config"]["likelihood_terms"][index]["sources"][-1][
                "template_filename"
            ] = os.path.abspath(template_file_path)

            # Load bkg, signal templates to estimate signal rate multiplier
            summed_bkg_mh = None
            for bkg_terms in alea_config["likelihood_config"]["likelihood_terms"][
                index
            ]["sources"]:
                if bkg_terms["name"] == signal_name:
                    continue
                bkg_template_file_name = bkg_terms["template_filename"]
                if "named_parameters" in bkg_terms:
                    nominal_named_parameter_dict = {
                        named_parameter: alea_config["parameter_definition"][
                            named_parameter
                        ]["nominal_value"]
                        for named_parameter in bkg_terms["named_parameters"]
                    }
                    bkg_template_file_name = bkg_template_file_name.format(
                        **nominal_named_parameter_dict
                    )
                bkg_mh = template_to_multihist(
                    bkg_template_file_name, hist_name=bkg_terms["histname"]
                )
                if summed_bkg_mh is None:
                    summed_bkg_mh = (
                        bkg_mh
                        * (
                            alea_config["parameter_definition"][
                                bkg_terms["parameters"][0]
                            ]["nominal_value"]
                        )
                        * experiment_instance.config["livetime"]
                    )
                else:
                    summed_bkg_mh += (
                        bkg_mh
                        * (
                            alea_config["parameter_definition"][
                                bkg_terms["parameters"][0]
                            ]["nominal_value"]
                        )
                        * experiment_instance.config["livetime"]
                    )

            signal_mh = (
                template_to_multihist(template_file_path, hist_name=signal_name)
                * experiment_instance.config["livetime"]
            )
            expected_events = np.sum(signal_mh.histogram)
            if expected_events <= 0:
                print(
                    f"Expected events for {experiment_instance.experiment_name} is {expected_events}."
                )
                warnings.warn(
                    f"If {signal_parameter_name} is {signal_parameter_value},"
                    f"the template gives no events for {experiment_instance.experiment_name}."
                    "This parameter will be skipped."
                )
                return None
            # If expected_events is too large, e.g., larger than 1e10,
            # the fit will not be sensitive to the signal rate multiplier.
            elif expected_events >= 1e10:
                print(
                    f"Expected events for {experiment_instance.experiment_name} "
                    f"is {expected_events}."
                )
                warnings.warn(
                    f"If {signal_parameter_name} is {signal_parameter_value}, "
                    f"the template gives too many events for {experiment_instance.experiment_name}."
                    "The results may not be reliable. Consider multiplying the input spectrum "
                    "by a small factor."
                )

            data = experiment_instance.get_data()
            data_names = list(experiment_instance.config["roi"].keys())
            data_mh = mh.Histdd(
                data[data_names[0]], data[data_names[1]], bins=bkg_mh.bin_edges
            )
            mask = (signal_mh.histogram > 0) | (bkg_mh.histogram > 0)
            data_mh.histogram[~mask] = 0
            estimated_signal_multiplier = signal_multiplier_estimator(
                signal_mh.histogram,
                summed_bkg_mh.histogram,
                data_mh.histogram,
            )
            if estimated_signal_multiplier * expected_events > 1.5 * len(
                data
            ) or np.isnan(estimated_signal_multiplier):
                # The estimator failed to converge
                estimated_signal_multiplier = 1 / expected_events
            signal_rate_multiplier.append(estimated_signal_multiplier)
            if max_rate_multiplier == 0:
                max_rate_multiplier = 1.5 * len(data) / expected_events
            else:
                max_rate_multiplier = min(
                    max_rate_multiplier, 1.5 * len(data) / expected_events
                )
        # Sometimes the estimator gives a very large value, which makes the fit unstable.
        # So we cap the nominal value to be the maximum allowed value.
        alea_config["parameter_definition"][f"{signal_name}_rate_multiplier"][
            "nominal_value"
        ] = min(float(np.array(signal_rate_multiplier).mean()), max_rate_multiplier)

        # Also, we can update the upper parameter_interval_bounds to be max_rate_multiplier
        # if upper parameter_interval_bounds is larger.
        upper_parameter_bound = alea_config["parameter_definition"][
            f"{signal_name}_rate_multiplier"
        ]["parameter_interval_bounds"][1]
        if upper_parameter_bound is None or upper_parameter_bound > max_rate_multiplier:
            alea_config["parameter_definition"][f"{signal_name}_rate_multiplier"][
                "parameter_interval_bounds"
            ][1] = max_rate_multiplier
        else:
            warnings.warn(
                f"Upper parameter_interval_bounds {upper_parameter_bound} of signal rate multiplier "
                f"for signal parameter value  {signal_parameter_value} "
                f"is smaller than the estimated maximum {max_rate_multiplier}."
                "This may not cause issues, but if you see infinity in upper limits, consider "
                "increasing upper parameter_interval_bounds."
            )

        return alea_config

    def get_data(self, alea_model, alea_config):
        toy_data = alea_model.generate_data()

        data_dict = {}
        for experiment_instance in self.experiment_instances:
            data_dict[experiment_instance.experiment_name] = (
                experiment_instance.get_data()
            )
        data_dict["ancillary"] = toy_data["ancillary"]
        if len(data_dict["ancillary"]) > 0:  # Avoid empty ancillary data
            for name in data_dict["ancillary"].dtype.names:
                data_dict["ancillary"][0][name] = alea_config["parameter_definition"][
                    name
                ]["nominal_value"]

        data_dict["generate_values"] = toy_data["generate_values"]
        return data_dict

    def get_alea_model(self, signal_parameter_value, save_config=False):
        alea_config = self.update_alea_config_signal(signal_parameter_value)
        if alea_config is None:
            raise ValueError(
                f"Signal parameter value {signal_parameter_value} is not valid."
            )
        if save_config:
            with open(
                os.path.join(
                    self.output_path, f"alea_config_{signal_parameter_value}.yaml"
                ),
                "w",
            ) as alea_config_file:
                yaml.dump(alea_config, alea_config_file)
        with HiddenTqdm():  # Suppress print from alea
            alea_model = DiamxModel(**alea_config)
        alea_model.data = self.get_data(alea_model, alea_config)
        return alea_model

    def run_inference(
        self,
        confidence_level=0.9,
        confidence_interval_kind="central",
        fit_strategy=None,
        exact_asymptotic=True,
        stabilize_fit=False,
        output_file_name=None,
    ):
        ci_and_discovery = []
        if stabilize_fit:
            stabilized_parameter = (
                f"{self.config['signal']['signal_name']}_rate_multiplier"
            )
        else:
            stabilized_parameter = None

        saved_yaml_config = False
        for signal_parameter_value in tqdm(
            generate_bin_array(self.config["signal"]["parameter_range"]).tolist(),
            "Running inference",
        ):
            alea_config = self.update_alea_config_signal(signal_parameter_value)
            if alea_config is None:
                continue
            if not saved_yaml_config:
                with open(
                    os.path.join(
                        self.output_path, f"alea_config_{signal_parameter_value}.yaml"
                    ),
                    "w",
                ) as alea_config_file:
                    yaml.dump(alea_config, alea_config_file)
                saved_yaml_config = True
            alea_model = None
            with HiddenTqdm():  # Suppress print from alea
                alea_model = DiamxModel(**alea_config)

                alea_model.data = self.get_data(alea_model, alea_config)

            best_fit, max_ll = alea_model.fit(
                stabilized_parameter=stabilized_parameter, fit_strategy=fit_strategy
            )
            best_fit_value = best_fit[
                f"{self.config['signal']['signal_name']}_rate_multiplier"
            ]
            ll_zero = None
            if exact_asymptotic:
                assert (
                    confidence_interval_kind == "central"
                ), "Non-central asymptotic confidence interval is not implemented."

                with HiddenTqdm():
                    extra_results = {}
                    lower_limit, upper_limit = (
                        alea_model.confidence_interval_asymptotic(
                            poi_name=f"{self.config['signal']['signal_name']}_rate_multiplier",
                            stabilized_parameter=stabilized_parameter,
                            confidence_level=confidence_level,
                            fit_strategy=fit_strategy,
                            best_fit=best_fit,
                            best_ll=max_ll,
                            extra_results=extra_results,
                        )
                    )
                    ll_zero = extra_results.get("ll_zero")
            else:
                lower_limit, upper_limit = alea_model.confidence_interval(
                    poi_name=f"{self.config['signal']['signal_name']}_rate_multiplier",
                    stabilized_parameter=stabilized_parameter,
                    confidence_level=confidence_level,
                    confidence_interval_kind=confidence_interval_kind,
                    fit_strategy=fit_strategy,
                )
            if ll_zero is None:
                _, ll_zero = alea_model.fit(
                    **{f"{self.config['signal']['signal_name']}_rate_multiplier": 0},
                    stabilized_parameter=stabilized_parameter,
                    fit_strategy=fit_strategy,
                )
            # Cowan et al. 2011, Eq. 52
            # Clipping to avoid nan significance due to numerical issues
            significance = np.sqrt(2 * np.clip(max_ll - ll_zero, 0, None))

            ci_and_discovery.append(
                np.array(
                    [
                        signal_parameter_value,
                        lower_limit,
                        upper_limit,
                        significance,
                        best_fit_value,
                    ]
                )
            )

            # Release the per-mass model before moving on. The Minuit objective
            # closes over alea_model (alea_model -> minuit_object -> cost ->
            # alea_model), a cycle through iminuit's C extension that the cyclic
            # GC does not reliably reclaim; without breaking it each mass point
            # leaks a full set of template histograms. Then drop the references
            # and force a collection so the templates do not accumulate over the
            # scan.
            alea_model.minuit_object = None
            alea_model._likelihood = None
            del alea_model, alea_config, best_fit
            gc.collect()

        ci_and_discovery = np.array(ci_and_discovery)
        if output_file_name is None:
            output_file_name = f"ci_{self.config['signal']['signal_name']}.csv"
        np.savetxt(
            os.path.join(self.output_path, output_file_name),
            ci_and_discovery,
            delimiter=",",
        )

    def print_best_fit(
        self, signal_parameter_value, stabilize_fit=False, disable_rounding=False
    ):
        if stabilize_fit:
            stabilized_parameter = (
                f"{self.config['signal']['signal_name']}_rate_multiplier"
            )
        else:
            stabilized_parameter = None

        alea_config = self.update_alea_config_signal(signal_parameter_value)
        if alea_config is None:
            return
        with HiddenTqdm():  # Suppress print from alea
            alea_model = DiamxModel(**alea_config)

            alea_model.data = self.get_data(alea_model, alea_config)

            best_fit, max_ll = alea_model.fit(stabilized_parameter=stabilized_parameter)

        def get_exp_unc_from_config(bkg_config, livetime):
            nominal = bkg_config["rate_nominal"] * livetime
            if "rate_uncertainty" not in bkg_config:
                unc = None
            elif (
                "rate_relative_uncertainty" in bkg_config
                and bkg_config["rate_relative_uncertainty"]
            ):
                unc = nominal * bkg_config["rate_uncertainty"]
            else:
                unc = bkg_config["rate_uncertainty"] * livetime
            return nominal, unc

        def get_shape_parameter_unc_from_config(bkg_config):
            nominal = bkg_config["shape_parameter_nominal"]
            if "shape_parameter_uncertainty" not in bkg_config:
                unc = None
            elif (
                "shape_parameter_relative_uncertainty" in bkg_config
                and bkg_config["shape_parameter_relative_uncertainty"]
            ):
                unc = nominal * bkg_config["shape_parameter_uncertainty"]
            else:
                unc = bkg_config["shape_parameter_uncertainty"]
            return nominal, unc

        def get_bkg_name(bkg_config):
            if "bkg_name" in bkg_config:
                return bkg_config["bkg_name"]
            elif "shaped_bkg_name" in bkg_config:
                return bkg_config["shaped_bkg_name"]
            else:
                raise ValueError("Bkg name not found in config.")

        total_ton_year = sum(
            [
                experiment_config["livetime"] * experiment_config["fiducial_mass"]
                for experiment_config in self.config["experiments"]
            ]
        )
        for experiment_config in self.config["experiments"]:
            print(experiment_config["experiment_name"])
            livetime = experiment_config["livetime"]
            ton_year = (
                experiment_config["livetime"] * experiment_config["fiducial_mass"]
            )
            result_nominal = {
                get_bkg_name(bkg_config): format_value_uncertainty(
                    *get_exp_unc_from_config(bkg_config, livetime),
                    disable_rounding=disable_rounding,
                )
                for bkg_config in experiment_config["bkgs"]
                + experiment_config["shaped_bkgs"]
            }
            result_bestfit = {
                get_bkg_name(bkg_config): format_value_uncertainty(
                    alea_model.minuit_object.values[
                        self._get_rate_name(
                            bkg_config, experiment_config["experiment_name"]
                        )
                    ]
                    * livetime
                    * bkg_config["rate_nominal"],
                    alea_model.minuit_object.errors[
                        self._get_rate_name(
                            bkg_config, experiment_config["experiment_name"]
                        )
                    ]
                    * livetime
                    * bkg_config["rate_nominal"],
                    disable_rounding=disable_rounding,
                )
                for bkg_config in experiment_config["bkgs"]
                + experiment_config["shaped_bkgs"]
            }
            print(
                pd.DataFrame.from_dict(
                    {"Nominal": result_nominal, "Best fit": result_bestfit}
                )
            )
            if len(experiment_config["shaped_bkgs"]) > 0:
                shape_parameter_nominal_dict = {}
                shape_parameter_bestfit_dict = {}
                for bkg_config in experiment_config["shaped_bkgs"]:
                    for shape_parameter_config in bkg_config["shape_parameters"]:
                        if shape_parameter_config.get("shape_parameter_shared", False):
                            shape_parameter_name = shape_parameter_config[
                                "shape_parameter_name"
                            ]
                        else:
                            shape_parameter_name = f"{experiment_config['experiment_name']}_{shape_parameter_config['shape_parameter_name']}"
                        shape_parameter_nominal_dict[shape_parameter_name] = (
                            format_value_uncertainty(
                                *get_shape_parameter_unc_from_config(
                                    shape_parameter_config
                                ),
                                disable_rounding=disable_rounding,
                            )
                        )
                        shape_parameter_bestfit_dict[shape_parameter_name] = (
                            format_value_uncertainty(
                                best_fit[shape_parameter_name],
                                alea_model.minuit_object.errors[shape_parameter_name],
                                disable_rounding=disable_rounding,
                            )
                        )
                print("Shape parameters:")
                print(
                    pd.DataFrame.from_dict(
                        {
                            "Nominal": shape_parameter_nominal_dict,
                            "Best fit": shape_parameter_bestfit_dict,
                        }
                    )
                )
            signal_name = self.config["signal"]["signal_name"]
            print(
                f"Signal best fit: {alea_model.get_expectation_values(**best_fit)[signal_name] * ton_year / total_ton_year}"
            )
            print()

    def _get_shaped_bkg_template(
        self, shaped_bkg_config, experiment_instance, shape_parameter_values
    ):
        """
        Internal method to get the shaped background template histogram.
        This method is used to retrieve the template for a shaped background
        based on the provided configuration and shape parameter value.

        Parameters
        ----------
        shaped_bkg_config : dict
            The configuration dictionary for the shaped background.
        experiment_instance : diamx.experiment.Experiment
            The instance of the experiment for which the shaped background is defined.
        shape_parameter_values : tuple or list
            The values of the shape parameters for the shaped background.

        Returns
        -------
        histogram: Histdd
            The shaped background template histogram for the specified experiment and shape parameter value.
        """
        args = copy.deepcopy(shaped_bkg_config.get("args", {}))
        file_hash = create_hash(experiment_instance.config["roi"], **args)
        template_suffix_parts = []
        for shape_parameter_config, shape_parameter_value in zip(
            shaped_bkg_config["shape_parameters"], shape_parameter_values
        ):
            # shape_parameter_name = (
            #     f"{experiment_instance.experiment_name}_{shape_parameter_config['shape_parameter_name']}"
            #     if not shape_parameter_config.get("shape_parameter_shared", False)
            #     else shape_parameter_config["shape_parameter_name"]
            # )
            shape_parameter_name = self._get_shape_parameter_name(
                shape_parameter_config, experiment_instance.experiment_name
            )
            template_suffix_parts.append(
                f"{shape_parameter_name}_{shape_parameter_value:{shape_parameter_config['formatter']}}"
            )
        template_suffix = "_".join(template_suffix_parts)

        template_file_name = (
            f"{experiment_instance.experiment_name}_shaped_bkg_{shaped_bkg_config['shaped_bkg_name']}"
            f"_{file_hash}_{template_suffix}.ii.h5"
        )
        template_file_path = os.path.join(
            self.output_path, template_folder, template_file_name
        )
        bkg_mh = template_to_multihist(
            template_file_path,
            hist_name=replace_alias(shaped_bkg_config)["shaped_bkg_name"],
        )
        return bkg_mh

    def get_bkg_template(self, experiment_name, bkg_name, shape_parameter_values=None):
        """
        Get the background template for a given experiment and background name.
        It also supports shaped backgrounds by providing a shape parameter value.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to get the background template.
        bkg_name : str
            The name of the background for which to get the template.
        shape_parameter_values : list or tuple, optional
            The values of the shape parameters for shaped backgrounds. Required if the background is shaped.
            Otherwise it should be None.

        Returns
        -------
        histogram: Histdd
            The background template histogram for the specified experiment and background name.
        """
        experiment_instance = None
        for instance in self.experiment_instances:
            if instance.experiment_name == experiment_name:
                experiment_instance = instance
                break
        if experiment_instance is None:
            raise ValueError(f"Experiment {experiment_name} not found.")

        def _anchor_grid_iterator(anchor_z_grid):
            # Copied from https://github.com/JelleAalbers/blueice/blob/master/blueice/pdf_morphers.py
            """Iterates over the anchor grid, yielding index, z-values"""
            fake_grid = np.zeros(list(anchor_z_grid.shape)[:-1])
            it = np.nditer(fake_grid, flags=["multi_index"])
            while not it.finished:
                anchor_grid_index = list(it.multi_index)
                yield (
                    anchor_grid_index,
                    tuple(anchor_z_grid[tuple(anchor_grid_index + [slice(None)])]),
                )
                it.iternext()

        if shape_parameter_values is not None:
            for shaped_bkg_config in experiment_instance.config["shaped_bkgs"]:
                if shaped_bkg_config["shaped_bkg_name"] == bkg_name:
                    # Check whether shape_parameter_value is already in blueice anchors.
                    # If not, use an interpolated template.
                    anchor_arrays = [
                        np.sort(
                            generate_bin_array(
                                shape_parameter_config["shape_parameter_range"]
                            )
                        )
                        for shape_parameter_config in shaped_bkg_config[
                            "shape_parameters"
                        ]
                    ]
                    anchor_grid = arrays_to_grid(anchor_arrays)
                    extra_dims = None
                    anchor_scores = None
                    for (
                        anchor_grid_index,
                        grid_shape_parameter_values,
                    ) in _anchor_grid_iterator(anchor_grid):
                        # Compute f at this point, and store it in anchor_scores
                        anchor_template = self._get_shaped_bkg_template(
                            shaped_bkg_config,
                            experiment_instance,
                            grid_shape_parameter_values,
                        )
                        if extra_dims is None:
                            extra_dims = anchor_template.histogram.shape
                            anchor_scores = np.empty(
                                anchor_grid.shape[:-1] + extra_dims
                            )
                        assert (
                            extra_dims == anchor_template.histogram.shape
                        ), "All templates must have the same shape."
                        anchor_scores[
                            tuple(anchor_grid_index + [slice(None)] * len(extra_dims))
                        ] = anchor_template.histogram

                    itp = RegularGridInterpolator(anchor_arrays, anchor_scores)
                    interpolated_histogram = itp(np.array(shape_parameter_values))[0]
                    bkg_mh = mh.Histdd.from_histogram(
                        histogram=interpolated_histogram,
                        bin_edges=anchor_template.bin_edges,
                    )
                    return bkg_mh
            raise ValueError(
                f"Shaped bkg {bkg_name} not found for experiment {experiment_name}."
            )
        for bkg_config in experiment_instance.config["bkgs"]:
            if bkg_config["bkg_name"] == bkg_name:
                args = bkg_config.get("args", {})
                file_hash = create_hash(experiment_instance.config["roi"], **args)
                template_file_name = (
                    f"{experiment_instance.experiment_name}_bkg_{bkg_name}"
                    f"_{file_hash}.ii.h5"
                )
                template_file_path = os.path.join(
                    self.output_path, template_folder, template_file_name
                )
                bkg_mh = template_to_multihist(
                    template_file_path, hist_name=replace_alias(bkg_config)["bkg_name"]
                )
                # mh.plot()
                return bkg_mh
        raise ValueError(f"Bkg {bkg_name} not found for experiment {experiment_name}.")

    def get_best_fit_bkg_mh(
        self,
        experiment_name,
        signal_parameter_value,
        bkg_to_include=None,
        stabilize_fit=True,
    ):
        """
        Get the total best-fit background model histogram for a given experiment and signal parameter value.
        This method aggregates the background templates for the specified experiment and signal parameter value,
        optionally filtering by a list of background names to include.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to get the background model.
        signal_parameter_value : float
            The value of the signal parameter to use for the background model when fitting.
        bkg_to_include : list of str, optional
            List of background names to include in the total background model. If None, all backgrounds are included.
            If provided, only the backgrounds with names in this list will be included in the total model.

        Returns
        -------
        histogram: Histdd
            The total background model histogram for the specified experiment and signal parameter value.
        """
        if stabilize_fit:
            stabilized_parameter = (
                f"{self.config['signal']['signal_name']}_rate_multiplier"
            )
        else:
            stabilized_parameter = None

        bkg_mh = None
        exp_id = None
        for idx, experiment in enumerate(self.experiment_instances):
            if experiment.experiment_name == experiment_name:
                exp_id = idx
                break
        if exp_id is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in context.")

        alea_model = self.get_alea_model(signal_parameter_value, save_config=False)
        best_fit, max_ll = alea_model.fit(stabilized_parameter=stabilized_parameter)

        for bkg_config in self.experiment_instances[exp_id].config["bkgs"]:
            if (
                bkg_to_include is not None
                and bkg_config["bkg_name"] not in bkg_to_include
            ):
                continue
            rate_name = self._get_rate_name(bkg_config, experiment_name)
            best_fit_multiplier = alea_model.minuit_object.values[rate_name]
            if bkg_mh is None:
                bkg_mh = (
                    self.get_bkg_template(experiment_name, bkg_config["bkg_name"])
                    * best_fit_multiplier
                )
            else:
                bkg_mh += (
                    self.get_bkg_template(experiment_name, bkg_config["bkg_name"])
                    * best_fit_multiplier
                )

        for shaped_bkg_config in self.experiment_instances[exp_id].config[
            "shaped_bkgs"
        ]:
            if (
                bkg_to_include is not None
                and shaped_bkg_config["shaped_bkg_name"] not in bkg_to_include
            ):
                continue
            rate_name = self._get_rate_name(shaped_bkg_config, experiment_name)
            best_fit_multiplier = alea_model.minuit_object.values[rate_name]
            shape_parameter_values = [
                alea_model.minuit_object.values[
                    (
                        f"{experiment_name}_{shape_parameter_config['shape_parameter_name']}"
                        if not shape_parameter_config.get(
                            "shape_parameter_shared", False
                        )
                        else shape_parameter_config["shape_parameter_name"]
                    )
                ]
                for shape_parameter_config in shaped_bkg_config["shape_parameters"]
            ]
            bkg_component_mh = self.get_bkg_template(
                experiment_name,
                shaped_bkg_config["shaped_bkg_name"],
                shape_parameter_values=shape_parameter_values,
            )
            if bkg_mh is None:
                bkg_mh = bkg_component_mh * best_fit_multiplier
            else:
                bkg_mh += bkg_component_mh * best_fit_multiplier
        return bkg_mh

    def get_signal_template(self, experiment_name, signal_parameter_value):
        """
        Get the signal template for a given experiment and signal parameter value.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to get the signal template.
        signal_parameter_value : float
            The value of the signal parameter for which to get the template.

        Returns
        -------
        histogram: Histdd
            The signal template histogram for the specified experiment and signal parameter value.
        """
        for experiment_instance in self.experiment_instances:
            if experiment_instance.experiment_name == experiment_name:
                break
        args = copy.deepcopy(self.config["signal"].get("args", {}))
        signal_name = self.config["signal"]["signal_name"]
        args[self.config["signal"]["parameter_name"]] = signal_parameter_value
        args["fiducial_mass"] = experiment_instance.config["fiducial_mass"]
        file_hash = create_hash(experiment_instance.config["roi"], **args)
        template_file_name = (
            f"{experiment_instance.experiment_name}_signal_{signal_name}"
            f"_{file_hash}.ii.h5"
        )
        template_file_path = os.path.join(
            self.output_path, template_folder, template_file_name
        )
        mh = template_to_multihist(template_file_path, hist_name=signal_name)
        return mh

    def _plot_template(
        self,
        mh,
        mode=["histogram", "contour", "contourf"],
        histogram_kwargs=None,
        contour_kwargs=None,
        contourf_kwargs=None,
    ):
        """
        Plot the template contours for a given histogram object.

        Parameters
        ----------
        mh : Histdd
            The Histdd object containing the template data.
        mode : list of str
            Modes for plotting. Options are "histogram", "contour", and "contourf".
        histogram_kwargs : dict
            Additional keyword arguments for the histogram plot.
        contour_kwargs : dict
            Additional keyword arguments for the contour plot.
        contourf_kwargs : dict
            Additional keyword arguments for the contourf plot.

        Returns
        -------
        quadmesh : QuadMesh
            The QuadMesh object for the histogram plot.
        contours : ContourSet
            The ContourSet object for the contour plot.
        contourfs : QuadContourSet
            The QuadContourSet object for the contourf plot.
        """
        H = mh.histogram
        xcenters, ycenters = mh.bin_centers()
        xedges, yedges = mh.bin_edges
        X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")
        total = H.sum()
        H_flat = H.flatten()

        # Sort the flattened histogram in descending order (highest density first)
        inds = np.argsort(H_flat)[::-1]
        H_sorted = H_flat[inds]
        H_cumsum = np.cumsum(H_sorted) / total

        # Define a function to determine the contour level corresponding to a given fraction
        def get_contour_level(fraction):
            idx = np.searchsorted(H_cumsum, fraction)
            return H_sorted[idx]

        level68 = get_contour_level(0.68)
        level95 = get_contour_level(0.95)

        # Use pcolormesh to plot the histogram. Note: we transpose H so that the orientation matches the x and y axes.
        if "histogram" in mode:
            quadmesh = plt.pcolormesh(xedges, yedges, H.T, **histogram_kwargs)
            plt.colorbar()
        else:
            quadmesh = None

        # Overlay the contours with custom linestyles:
        # 95% contour (red) is dashed and 68% contour (blue) is solid.
        if "contour" in mode:
            contours = plt.contour(X, Y, H, levels=[level95, level68], **contour_kwargs)
        else:
            contours = None
        if "contourf" in mode:
            contourfs = plt.contourf(
                X, Y, H, levels=[level95, level68], **contourf_kwargs
            )
        else:
            contourfs = None

        return quadmesh, contours, contourfs

    def plot_bkg_template(
        self,
        experiment_name,
        bkg_name,
        shape_parameter_value=None,
        mode=["histogram", "contour"],
        histogram_kwargs={},
        contour_kwargs={},
        contourf_kwargs={},
    ):
        """
        Plot the background template for a given experiment and background name.
        It also supports shaped backgrounds by providing a shape parameter value.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to plot the background template.
        bkg_name : str
            The name of the background for which to plot the template.
        shape_parameter_value : float, optional
            The value of the shape parameter for shaped backgrounds. Required if the background is shaped.
            Otherwise it should be None.
        mode : list of str
            Modes for plotting. Options are "histogram", "contour", and "contourf".
        histogram_kwargs : dict
            Additional keyword arguments for the histogram plot.
        contour_kwargs : dict
            Additional keyword arguments for the contour plot.
        contourf_kwargs : dict
            Additional keyword arguments for the contourf plot.

        Returns
        -------
        quadmesh : QuadMesh
            The QuadMesh object for the histogram plot.
        contours : ContourSet
            The ContourSet object for the contour plot.
        contourfs : QuadContourSet
            The QuadContourSet object for the contourf plot.
        """
        mh = self.get_bkg_template(experiment_name, bkg_name, shape_parameter_value)
        return self._plot_template(
            mh, mode, histogram_kwargs, contour_kwargs, contourf_kwargs
        )

    def plot_best_fit_bkg_mh(
        self,
        experiment_name,
        signal_parameter_value,
        bkg_to_include=None,
        mode=["histogram", "contour"],
        histogram_kwargs={},
        contour_kwargs={},
        contourf_kwargs={},
    ):
        """
        Plot the total best-fit background model histogram for a given experiment and signal parameter value.
        This method aggregates the background templates for the specified experiment and signal parameter value,
        optionally filtering by a list of background names to include.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to plot the background model.
        signal_parameter_value : float
            The value of the signal parameter to use for the background model when fitting.
        bkg_to_include : list of str, optional
            List of background names to include in the total background model. If None, all backgrounds are included.
        mode : list of str
            Modes for plotting. Options are "histogram", "contour", and "contourf".
        histogram_kwargs : dict
            Additional keyword arguments for the histogram plot.
        contour_kwargs : dict
            Additional keyword arguments for the contour plot.
        contourf_kwargs : dict
            Additional keyword arguments for the contourf plot.

        Returns
        -------
        quadmesh : QuadMesh
            The QuadMesh object for the histogram plot.
        contours : ContourSet
            The ContourSet object for the contour plot.
        contourfs : QuadContourSet
            The QuadContourSet object for the contourf plot.
        """
        bkg_mh = self.get_best_fit_bkg_mh(
            experiment_name,
            signal_parameter_value,
            bkg_to_include=bkg_to_include,
            stabilize_fit=True,
        )
        return self._plot_template(
            bkg_mh, mode, histogram_kwargs, contour_kwargs, contourf_kwargs
        )

    def plot_signal_template(
        self,
        experiment_name,
        signal_parameter_value,
        mode=["histogram", "contour"],
        histogram_kwargs={},
        contour_kwargs={},
        contourf_kwargs={},
    ):
        """
        Plot the signal template for a given experiment and signal parameter value.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to plot the signal template.
        signal_parameter_value : float
            The value of the signal parameter to use for the template.
        mode : list of str
            Modes for plotting. Options are "histogram", "contour", and "contourf".
        histogram_kwargs : dict
            Additional keyword arguments for the histogram plot.
        contour_kwargs : dict
            Additional keyword arguments for the contour plot.
        contourf_kwargs : dict
            Additional keyword arguments for the contourf plot.

        Returns
        -------
        quadmesh : QuadMesh
            The QuadMesh object for the histogram plot.
        contours : ContourSet
            The ContourSet object for the contour plot.
        contourfs : QuadContourSet
            The QuadContourSet object for the contourf plot.
        """
        mh = self.get_signal_template(experiment_name, signal_parameter_value)
        return self._plot_template(
            mh, mode, histogram_kwargs, contour_kwargs, contourf_kwargs
        )

    def get_best_fit_local_pdf(
        self, experiment_name, signal_parameter_value, data_points, rtol=1e-10
    ):
        """
        Get the best-fit local probability density function (PDF) for a given experiment and signal parameter value.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment for which to get the local PDF.
        signal_parameter_value : float
            The value of the signal parameter to use for the PDF when fitting.
        data_points : array-like
            The data points at which to evaluate the local PDF. It should be of shape (N, 2) for 2D histograms.
        rtol : float, optional
            The relative tolerance for bin-type determination. Default is 1e-10.

        Returns
        -------
        local_pdf : dict
            A dictionary containing the local PDF for each background and signal component.
        """
        experiment_instance = None
        for instance in self.experiment_instances:
            if instance.experiment_name == experiment_name:
                experiment_instance = instance
                break
        if experiment_instance is None:
            raise ValueError(f"Experiment {experiment_name} not found.")

        alea_model = self.get_alea_model(signal_parameter_value)
        best_fit, _max_ll = alea_model.fit()
        local_pdf = {}
        for bkg_config in experiment_instance.config["bkgs"]:
            bkg_mh = (
                self.get_bkg_template(experiment_name, bkg_config["bkg_name"])
                * best_fit[self._get_rate_name(bkg_config, experiment_name)]
            )
            local_pdf[bkg_config["bkg_name"]] = get_local_pdf_from_template(
                bkg_mh, data_points, rtol=rtol
            )
        for shaped_bkg_config in experiment_instance.config["shaped_bkgs"]:
            shaped_bkg_mh = (
                self.get_bkg_template(
                    experiment_name,
                    shaped_bkg_config["shaped_bkg_name"],
                    shape_parameter_values=[
                        best_fit[
                            self._get_shape_parameter_name(
                                shape_parameter_config, experiment_name
                            )
                        ]
                        for shape_parameter_config in shaped_bkg_config[
                            "shape_parameters"
                        ]
                    ],
                )
                * best_fit[self._get_rate_name(shaped_bkg_config, experiment_name)]
            )
            local_pdf[shaped_bkg_config["shaped_bkg_name"]] = (
                get_local_pdf_from_template(shaped_bkg_mh, data_points, rtol=rtol)
            )
        signal_mh = (
            self.get_signal_template(experiment_name, signal_parameter_value)
            * best_fit[f"{self.config['signal']['signal_name']}_rate_multiplier"]
        )
        local_pdf[self.config["signal"]["signal_name"]] = get_local_pdf_from_template(
            signal_mh, data_points, rtol=rtol
        )

        return local_pdf

    def check_config_sanity(self):
        config_attributes = ["experiments", "signal"]
        for config_attribute in config_attributes:
            if config_attribute not in self.config:
                raise ValueError(f"Missing {config_attribute} in context config.")
        signal_config_attributes = [
            "signal_name",
            "rate_fit_limits",
            "parameter_name",
            "parameter_range",
        ]
        for signal_config_attribute in signal_config_attributes:
            if signal_config_attribute not in self.config["signal"]:
                raise ValueError(f"Missing {signal_config_attribute} in signal config.")

    def register_experiment(self, experiment):
        if (
            not issubclass(experiment, diamx.experiment.Experiment)
            or experiment.experiment_name is None
        ):
            raise ValueError(f"Invalid register of experiment.")
        self.registered_experiments[experiment.experiment_name] = experiment
