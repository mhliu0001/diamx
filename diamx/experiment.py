from typing import Optional, Union
import json
from json.decoder import JSONDecodeError
import warnings
import os
import copy
from tqdm import tqdm
import diamx
from diamx.utils import generate_bin_array, create_hash, template_folder, replace_alias
import inference_interface
import numpy as np
import multiprocessing
from functools import partial


def process_bkg_template(
    bkg_config,
    experiment_config,
    experiment_name,
    output_path,
    template_folder,
    generate_template,
):
    """Process one background template."""
    args = bkg_config.get("args", {})
    file_hash = create_hash(experiment_config["roi"], **args)
    template_file_name = (
        f"{experiment_name}_bkg_{bkg_config['bkg_name']}_{file_hash}.ii.h5"
    )
    bkg_config = replace_alias(bkg_config)
    template_file_path = os.path.join(output_path, template_folder, template_file_name)
    if not os.path.exists(template_file_path):
        generate_template(
            bkg_config["bkg_name"],
            bkg_config["rate_nominal"],
            template_file_path,
            **args,
        )
        assert os.path.exists(
            template_file_path
        ), f"Template generation for {bkg_config['bkg_name']} has failed!"
    return template_file_path


def process_shaped_bkg_template(
    task,
    experiment_config,
    experiment_name,
    output_path,
    template_folder,
    generate_template,
):
    """Process one shaped background template.
    `task` is a tuple: (shaped_bkg_config, shape_parameter_values).
    `shape_parameter_values` is a list of shape parameter values to process.
    """
    shaped_bkg_config, shape_parameter_values = task
    shape_parameter_configs = shaped_bkg_config["shape_parameters"]

    # Use a deep copy so that modifications do not affect the original args.
    args = copy.deepcopy(shaped_bkg_config.get("args", {}))
    file_hash = create_hash(experiment_config["roi"], **args)
    # Warn once per shaped_bkg_config if the parameter is already in args.
    assert isinstance(
        shape_parameter_values, (list, tuple)
    ), "shape_parameter_values must be a list or tuple."
    assert len(shape_parameter_values) == len(
        shape_parameter_configs
    ), "Length of shape_parameter_values must match length of shape_parameter_configs."
    template_suffix_parts = []
    for shape_parameter_config, shape_parameter_value in zip(
        shape_parameter_configs, shape_parameter_values
    ):
        shape_parameter_name = (
            f"{experiment_name}_{shape_parameter_config['shape_parameter_name']}"
            if not shape_parameter_config.get("shape_parameter_shared", False)
            else shape_parameter_config["shape_parameter_name"]
        )
        template_suffix_parts.append(
            f"{shape_parameter_name}_{shape_parameter_value:{shape_parameter_config['formatter']}}"
        )
    template_suffix = "_".join(template_suffix_parts)

    template_file_name = (
        f"{experiment_name}_shaped_bkg_{shaped_bkg_config['shaped_bkg_name']}"
        f"_{file_hash}_{template_suffix}.ii.h5"
    )
    template_file_path = os.path.join(output_path, template_folder, template_file_name)
    if not os.path.exists(template_file_path):
        shaped_bkg_config = replace_alias(shaped_bkg_config)
        for shape_parameter_config, shape_parameter_value in zip(
            shape_parameter_configs, shape_parameter_values
        ):
            shape_parameter_config = replace_alias(shape_parameter_config)
            shape_parameter_name = shape_parameter_config["shape_parameter_name"]
            if shape_parameter_name in args:
                warnings.warn(
                    f"Found {shape_parameter_name} in shaped bkg args that is supposed to be scanned over "
                    "shaped_parameter_range. It will be updated and will not be used."
                )
            args[shape_parameter_name] = shape_parameter_value
        generate_template(
            shaped_bkg_config["shaped_bkg_name"],
            shaped_bkg_config["rate_nominal"],
            template_file_path,
            **args,
        )
        assert os.path.exists(
            template_file_path
        ), f"Template generation for {shaped_bkg_config['shaped_bkg_name']} has failed!"
    return template_file_path


def process_signal_template(
    signal_config,
    parameter,
    experiment_config,
    experiment_name,
    output_path,
    template_folder,
    generate_template,
):
    """Process one signal template given a parameter value."""
    args = copy.deepcopy(signal_config.get("args", {}))
    # Warn once if the parameter is present in args.
    if signal_config["parameter_name"] in args:
        warnings.warn(
            f"Found {signal_config['parameter_name']} in signal args that is supposed to be scanned over "
            "parameter_range. It will be updated and will not be used."
        )
    args[signal_config["parameter_name"]] = parameter
    args["fiducial_mass"] = experiment_config["fiducial_mass"]
    file_hash = create_hash(experiment_config["roi"], **args)
    template_file_name = (
        f"{experiment_name}_signal_{signal_config['signal_name']}_{file_hash}.ii.h5"
    )
    template_file_path = os.path.join(output_path, template_folder, template_file_name)
    if not os.path.exists(template_file_path):
        generate_template(
            signal_config["signal_name"], None, template_file_path, **args
        )
        assert os.path.exists(
            template_file_path
        ), f"Template generation for {signal_config['signal_name']} has failed!"
    return template_file_path


class Experiment:
    """Base class of an experiment for dark matter inference."""

    # Do not initialize this class because it is base
    __is_base = True

    # Name of experiment should be set by user
    experiment_name = None

    def __init__(
        self, config: Union[str, dict], output_path: Optional[str] = "./diamx_output"
    ):
        """Initialization."""
        # Normally config is passed from context, so a dict is expected.
        # File input is for debugging.
        if isinstance(config, str):
            try:
                with open(config, "r") as config_file:
                    self.config = json.load(config_file)
            except OSError:
                raise ValueError(f"Experiment config file {config} cannot be opened.")
            except JSONDecodeError:
                raise ValueError(
                    f"Experiment config file {config} is not a valid JSON file."
                )
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Experiment config must be str or dict.")
        self.config_sanity_check()
        assert self.config["experiment_name"] == self.experiment_name
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        template_folder = "templates"
        if not os.path.exists(os.path.join(self.output_path, template_folder)):
            os.mkdir(os.path.join(self.output_path, template_folder))

    def config_sanity_check(self):
        config_attrs = [
            "experiment_name",
            "livetime",
            "fiducial_mass",
            "data",
            "roi",
            "bkgs",
            "shaped_bkgs",
            "eff",
        ]
        for attr in config_attrs:
            if attr not in self.config:
                raise ValueError(f"Missing attribute {attr} in experiment config.")
        for bkg in self.config["bkgs"]:
            bkg_attrs = [
                "bkg_name",
                "rate_nominal",
            ]
            for attr in bkg_attrs:
                if attr not in bkg:
                    raise ValueError(
                        f"Missing attribute {attr} for background component."
                    )
        for shaped_bkg in self.config["shaped_bkgs"]:
            shaped_bkg_attrs = [
                "shaped_bkg_name",
                "shape_parameters",
                "rate_nominal",
            ]
            for attr in shaped_bkg_attrs:
                if attr not in shaped_bkg:
                    raise ValueError(
                        f"Missing attribute {attr} for shaped background component."
                    )

    def get_bkg_templates(self):
        """Generate background templates in parallel."""
        template_folder = "templates"  # Set your folder name here.
        if "multiprocess_threads" in self.config:
            processes = self.config["multiprocess_threads"]
            ctx = multiprocessing.get_context("spawn")
            func = partial(
                process_bkg_template,
                experiment_config=self.config,
                experiment_name=self.experiment_name,
                output_path=self.output_path,
                template_folder=template_folder,
                generate_template=self.generate_template,
            )
            tasks = [
                bkg_config
                for bkg_config in self.config["bkgs"]
                if "share_template_with" not in bkg_config
            ]
            with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
                it = pool.imap_unordered(func, tasks, chunksize=1)
                for _ in tqdm(
                    it,
                    total=len(tasks),
                    desc=f"Generating background templates for {self.experiment_name}",
                ):
                    pass
        else:
            # Do not use multiprocessing, especially when using JAX like in appletree
            for bkg_config in tqdm(
                [
                    bkg_config
                    for bkg_config in self.config["bkgs"]
                    if "share_template_with" not in bkg_config
                ],
                f"Generating background templates for {self.experiment_name}",
            ):
                process_bkg_template(
                    bkg_config,
                    experiment_config=self.config,
                    experiment_name=self.experiment_name,
                    output_path=self.output_path,
                    template_folder=template_folder,
                    generate_template=self.generate_template,
                )
        # Handle shared templates
        for bkg_config in self.config["bkgs"]:
            if "share_template_with" in bkg_config:
                shared_bkg_name = bkg_config["share_template_with"]
                bkg_config_to_share = None
                for _bkg_config in self.config["bkgs"]:
                    if _bkg_config["bkg_name"] == shared_bkg_name:
                        bkg_config_to_share = _bkg_config
                        break
                if bkg_config_to_share is None:
                    raise ValueError(
                        f"Background {shared_bkg_name} to share template with is not found."
                    )
                original_file_hash = create_hash(
                    self.config["roi"], **bkg_config_to_share.get("args", {})
                )
                original_template_file_name = f"{self.experiment_name}_bkg_{shared_bkg_name}_{original_file_hash}.ii.h5"
                new_file_hash = create_hash(
                    self.config["roi"], **bkg_config.get("args", {})
                )
                new_template_file_name = f"{self.experiment_name}_bkg_{shared_bkg_name}_{new_file_hash}.ii.h5"
                self.get_template_from_file(
                    name=bkg_config["bkg_name"],
                    rate=bkg_config["rate_nominal"],
                    template_file_original=os.path.join(
                        self.output_path,
                        template_folder,
                        original_template_file_name,
                    ),
                    hist_name=shared_bkg_name,
                    template_file_path=os.path.join(
                        self.output_path,
                        template_folder,
                        new_template_file_name,
                    ),
                )

    def get_shaped_bkg_templates(self):
        """Generate shaped background templates in parallel."""
        template_folder = "templates"
        tasks = []
        for shaped_bkg_config in self.config["shaped_bkgs"]:
            shape_parameter_ranges = []
            for shape_parameter_config in shaped_bkg_config["shape_parameters"]:
                # Build a list of (shaped_bkg_config, shape_parameter_value) tasks.
                shape_parameter_ranges.append(
                    generate_bin_array(
                        shape_parameter_config["shape_parameter_range"]
                    ).tolist()
                )
            shape_parameter_values_list = np.stack(
                [
                    m.flatten()
                    for m in np.meshgrid(*shape_parameter_ranges, indexing="ij")
                ],
                axis=-1,
            )
            for shape_parameter_values in shape_parameter_values_list:
                tasks.append((shaped_bkg_config, shape_parameter_values.tolist()))
        if "multiprocess_threads" in self.config:
            processes = self.config["multiprocess_threads"]
            ctx = multiprocessing.get_context("spawn")
            func = partial(
                process_shaped_bkg_template,
                experiment_config=self.config,
                experiment_name=self.experiment_name,
                output_path=self.output_path,
                template_folder=template_folder,
                generate_template=self.generate_template,
            )
            with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
                it = pool.imap_unordered(func, tasks, chunksize=1)
                for _ in tqdm(
                    it,
                    total=len(tasks),
                    desc=f"Generating shaped background templates for {self.experiment_name}",
                ):
                    pass
        else:
            # Do not use multiprocessing, especially when using JAX like in appletree
            for task in tqdm(
                tasks,
                f"Generating shaped background templates for {self.experiment_name}",
            ):
                process_shaped_bkg_template(
                    task,
                    experiment_config=self.config,
                    experiment_name=self.experiment_name,
                    output_path=self.output_path,
                    template_folder=template_folder,
                    generate_template=self.generate_template,
                )

    def get_signal_templates(self, signal_config):
        """Generate signal templates in parallel."""
        template_folder = "templates"
        parameter_range = generate_bin_array(signal_config["parameter_range"]).tolist()
        if "multiprocess_threads" in self.config:
            processes = self.config["multiprocess_threads"]
            ctx = multiprocessing.get_context("spawn")
            func = partial(
                process_signal_template,
                signal_config,
                experiment_config=self.config,
                experiment_name=self.experiment_name,
                output_path=self.output_path,
                template_folder=template_folder,
                generate_template=self.generate_template,
            )
            with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
                it = pool.imap_unordered(func, parameter_range, chunksize=1)
                for _ in tqdm(
                    it,
                    total=len(parameter_range),
                    desc=f"Generating signal templates for {self.experiment_name}",
                ):
                    pass
        else:
            # Do not use multiprocessing, especially when using JAX like in appletree
            for parameter in tqdm(
                parameter_range,
                f"Generating signal templates for {self.experiment_name}",
            ):
                process_signal_template(
                    signal_config,
                    parameter,
                    experiment_config=self.config,
                    experiment_name=self.experiment_name,
                    output_path=self.output_path,
                    template_folder=template_folder,
                    generate_template=self.generate_template,
                )

    def get_template_from_file(
        self, name, rate, template_file_original, hist_name, template_file_path
    ):
        """Helper function to copy existing template file for diamx to use."""
        # If template_path is provided, validate it and copy it to template_file_path
        if not os.path.exists(template_file_original):
            # Look for the file in diamx/data
            template_file_original = os.path.join(
                os.path.dirname(diamx.__file__),
                "data",
                template_file_original,
            )
            if not os.path.exists(template_file_original):
                raise ValueError(
                    f"Template file {template_file_original} does not exist."
                )
        mh = inference_interface.template_to_multihist(
            template_file_original, hist_name
        )

        # Check if the template has the same binning as the roi
        roi = self.config["roi"]
        for idx, axis_name in enumerate(mh.axis_names):
            if axis_name not in roi:
                raise ValueError(
                    f"{axis_name} is in the template provided, but not in the roi."
                )
            bin_edges = generate_bin_array(roi[axis_name])
            if not np.allclose(mh.bin_edges[idx], bin_edges):
                raise ValueError(
                    f"{axis_name} binning from the provided template does not match roi."
                )

        # Renormalize the histograms
        mh.histogram = mh.histogram / np.sum(mh.histogram) * rate

        # Save to template_file_path
        inference_interface.multihist_to_template(
            [mh],
            template_file_path,
            histogram_names=[name],
        )

    def get_eff_uncertainty(self, signal_config):
        """
        Calculates efficiency uncertainty for signal.
        Returns a dict: key is the parameter in signal_config["parameter_range"],
                        value is uncertainty.
        This function should be implemented by user.
        """
        raise NotImplementedError(
            f"Efficiency uncertainty calculation is not implemented!"
        )

    def generate_template(self, name, rate, template_file_path, **kwargs):
        """Generates template for one background/signal model. This should be implemented by user."""
        raise NotImplementedError(f"Template generation for {name} is not implemented!")

    def get_data(self):
        """Experimental data. This should be implemented by user."""
        raise NotImplementedError(f"Experimental data generation is not implemented!")
