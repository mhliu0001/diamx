import os
import json
import copy
import diamx
from matplotlib.colors import LogNorm
import numpy as np

diamx_path = os.path.dirname(diamx.__file__)


def config_preprocess(
    config_file_name, batch_size=10000, parameter_range=[6, 40, 200, 1000]
):
    config_path = os.path.join(diamx_path, "..", "config", config_file_name)
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    new_config = copy.deepcopy(config)
    for experiment_idx, experiment_config in enumerate(config["experiments"]):
        for bkg_idx, bkg_config in enumerate(experiment_config["bkgs"]):
            new_config["experiments"][experiment_idx]["bkgs"][bkg_idx]["args"][
                "batch_size"
            ] = batch_size
        for shaped_bkg_idx, shaped_bkg_config in enumerate(
            experiment_config["shaped_bkgs"]
        ):
            new_config["experiments"][experiment_idx]["shaped_bkgs"][shaped_bkg_idx][
                "args"
            ]["batch_size"] = batch_size
            for shape_parameter_idx, shape_parameter_config in enumerate(
                shaped_bkg_config["shape_parameters"]
            ):
                if "shape_parameter_range" in shape_parameter_config:
                    assert "shape_parameter_fit_limits" in shape_parameter_config
                    shape_parameter_fit_limits = shape_parameter_config[
                        "shape_parameter_fit_limits"
                    ]
                    shape_parameter_nominal = shape_parameter_config[
                        "shape_parameter_nominal"
                    ]
                    new_config["experiments"][experiment_idx]["shaped_bkgs"][
                        shaped_bkg_idx
                    ]["shape_parameters"][shape_parameter_idx][
                        "shape_parameter_range"
                    ] = [
                        shape_parameter_fit_limits[0],
                        shape_parameter_nominal,
                        shape_parameter_fit_limits[1],
                    ]
        new_config["experiments"][experiment_idx]["eff"]["args"][
            "batch_size"
        ] = batch_size
    new_config["signal"]["parameter_range"] = parameter_range
    new_config["signal"]["args"]["batch_size"] = batch_size
    new_config["signal"]["args"]["signal_spectrum_path"] = os.path.abspath(
        os.path.join(diamx_path, "data", "wimp_{mass:d}GeV.csv")
    )
    return new_config


def test_xenonnt_sr0_context(tmp_path):
    st = diamx.Context(config_preprocess("xenonnt_sr0_wimp_config.json"), tmp_path)
    st.register_experiment(diamx.experiments.XENONnTSR0)
    st.generate_templates()
    st.run_inference()
    st.run_inference(stabilize_fit=True)
    st.run_inference(exact_asymptotic=False)
    diamx.run_inference_pool(st, processes=2, stabilize_fit=False)
    st.print_best_fit(200)
    st.print_best_fit(200, disable_rounding=True)
    st.plot_bkg_template(
        "xenonnt_sr0",
        "er",
        histogram_kwargs={"norm": LogNorm()},
        contour_kwargs={"colors": ["blue", "blue"], "linestyles": ["--", "-"]},
    )
    st.plot_signal_template(
        "xenonnt_sr0",
        200,
        mode=["contour"],
        contour_kwargs={"colors": ["orange", "orange"], "linestyles": ["--", "-"]},
    )
    st.plot_best_fit_bkg_mh(
        "xenonnt_sr0",
        200,
        mode=["contourf"],
        bkg_to_include=[
            "er",
            "neutron",
        ],
        histogram_kwargs={"norm": LogNorm()},
        contourf_kwargs={"colors": ["green", "green"], "alpha": [0.1, 0.2]},
    )
    data = st.experiment_instances[0].get_data()
    local_pdf_raw = st.get_best_fit_local_pdf(
        experiment_name="xenonnt_sr0",
        signal_parameter_value=200,
        data_points=np.stack(
            (data["cs1"], data["cs2"]),
            axis=-1,
        ),
    )


def test_xenonnt_sr0_and_1_context(tmp_path):
    st = diamx.Context(
        config_preprocess("xenonnt_sr0_and_1_wimp_config.json"), tmp_path
    )
    st.register_experiment(diamx.experiments.XENONnTSR0)
    st.register_experiment(diamx.experiments.XENONnTSR1a)
    st.register_experiment(diamx.experiments.XENONnTSR1b)
    st.generate_templates()
    st.run_inference(stabilize_fit=False, exact_asymptotic=False)
    st.print_best_fit(200)


def test_xenonnt_sr0_and_1_dec_context(tmp_path):
    st = diamx.Context(
        config_preprocess("xenonnt_sr0_and_1_wimp_config_dec.json"), tmp_path
    )
    st.register_experiment(diamx.experiments.XENONnTSR0)
    st.register_experiment(diamx.experiments.XENONnTSR1a)
    st.register_experiment(diamx.experiments.XENONnTSR1b)
    st.generate_templates()
    st.run_inference(stabilize_fit=False, exact_asymptotic=False)
    st.print_best_fit(200)


def test_lz_ws2022_context(tmp_path):
    st = diamx.Context(config_preprocess("lz_ws2022_wimp_config.json"), tmp_path)
    st.register_experiment(diamx.experiments.LZWS2022)
    st.generate_templates()
    st.run_inference(stabilize_fit=False, exact_asymptotic=False)
    st.print_best_fit(40)
    st.plot_best_fit_bkg_mh(
        "lz_ws2022",
        40,
        histogram_kwargs={"norm": LogNorm()},
        contour_kwargs={"colors": ["blue", "blue"], "linestyles": ["--", "-"]},
    )


def test_lz_ws2024_context(tmp_path):
    st = diamx.Context(config_preprocess("lz_ws2024_wimp_config.json"), tmp_path)
    st.register_experiment(diamx.experiments.LZWS2022)
    st.register_experiment(diamx.experiments.LZWS2024)
    st.generate_templates()
    st.run_inference(stabilize_fit=False, exact_asymptotic=False)
    st.print_best_fit(40)
    st.plot_best_fit_bkg_mh(
        "lz_ws2024",
        40,
        bkg_to_include=[
            "pb214",
            "kr85_ar39_detgamma",
            "solar_neutrino_er",
            "pb212_po218",
            "tritium_c14",
            "xe136",
            "xe127_xe125",
            "xe124",
        ],
        histogram_kwargs={"norm": LogNorm()},
        contour_kwargs={"colors": ["blue", "blue"], "linestyles": ["--", "-"]},
    )


def test_lz_ws2024_shape_par_context(tmp_path):
    st = diamx.Context(
        config_preprocess("lz_ws2024_wimp_config_shape_par.json"), tmp_path
    )
    st.register_experiment(diamx.experiments.LZWS2022)
    st.register_experiment(diamx.experiments.LZWS2024)
    st.generate_templates()
    st.run_inference(stabilize_fit=False, exact_asymptotic=False)
    st.print_best_fit(40)
    st.plot_best_fit_bkg_mh(
        "lz_ws2024",
        40,
        bkg_to_include=[
            "pb214",
            "kr85_ar39_detgamma",
            "solar_neutrino_er",
            "pb212_po218",
            "tritium_c14",
            "xe136",
            "xe127_xe125",
            "xe124",
        ],
        histogram_kwargs={"norm": LogNorm()},
        contour_kwargs={"colors": ["blue", "blue"], "linestyles": ["--", "-"]},
    )
