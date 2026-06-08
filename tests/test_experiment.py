import diamx


def test_xenonnt_sr0(tmp_path):
    test_config = {
        "experiment_name": "xenonnt_sr0",
        "livetime": 0.2605479452054794,
        "fiducial_mass": 4.18,
        "data": "xenonnt_sr0_wimp_data.csv",
        "roi": {
            "cs1": "np.linspace(0, 100, 101)",
            "cs2": "np.logspace(2.6020599913279625, 4.1, 81)",
        },
        "bkgs": [
            {
                "bkg_name": "er",
                "rate_nominal": 514.3007360672976,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "neutron",
                "rate_nominal": 4.221871713985279,
                "rate_uncertainty": 0.5,
                "rate_relative_uncertainty": True,
                "args": {"batch_size": 1000},
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    xenonnt_experiment_sr0 = diamx.experiments.XENONnTSR0(test_config, tmp_path)
    xenonnt_experiment_sr0.get_data()
    xenonnt_experiment_sr0.get_bkg_templates()
    xenonnt_experiment_sr0.get_shaped_bkg_templates()


def test_xenonnt_sr1a(tmp_path):
    test_config = {
        "experiment_name": "xenonnt_sr1a",
        "livetime": 0.1824657534246575,
        "fiducial_mass": 4.00,
        "data": "xenonnt_sr1a_wimp_data.csv",
        "roi": {
            "cs1": "np.linspace(0, 100, 101)",
            "cs2": "np.logspace(2.6020599913279625, 4.1, 81)",
        },
        "bkgs": [
            {
                "bkg_name": "er_flat",
                "rate_nominal": 2356.606606606607,
                "rate_uncertainty": 164.41441441441444,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "tritium",
                "rate_nominal": 339.78978978978984,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "ar37",
                "rate_nominal": 317.8678678678679,
                "rate_uncertainty": 32.88288288288289,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xenonnt_neutron",
                "rate_nominal": 2.575825825825826,
                "rate_uncertainty": 0.4,
                "rate_relative_uncertainty": True,
                "shared_rate": True,
                "args": {"batch_size": 1000},
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    xenonnt_experiment_sr1a = diamx.experiments.XENONnTSR1a(test_config, tmp_path)
    xenonnt_experiment_sr1a.get_data()
    xenonnt_experiment_sr1a.get_bkg_templates()
    xenonnt_experiment_sr1a.get_shaped_bkg_templates()


def test_xenonnt_sr1b(tmp_path):
    test_config = {
        "experiment_name": "xenonnt_sr1b",
        "livetime": 0.3284931506849315,
        "fiducial_mass": 4.00,
        "data": "xenonnt_sr1b_wimp_data.csv",
        "roi": {
            "cs1": "np.linspace(0, 100, 101)",
            "cs2": "np.logspace(2.6020599913279625, 4.1, 81)",
        },
        "bkgs": [
            {
                "bkg_name": "er_flat",
                "rate_nominal": 459.674728940784,
                "rate_uncertainty": 33.48623853211009,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "tritium",
                "rate_nominal": 307.46455379482904,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xenonnt_neutron",
                "rate_nominal": 2.1309424520433695,
                "rate_uncertainty": 0.4,
                "rate_relative_uncertainty": True,
                "shared_rate": True,
                "args": {"batch_size": 1000},
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    xenonnt_experiment_sr1b = diamx.experiments.XENONnTSR1b(test_config, tmp_path)
    xenonnt_experiment_sr1b.get_data()
    xenonnt_experiment_sr1b.get_bkg_templates()
    xenonnt_experiment_sr1b.get_shaped_bkg_templates()


def test_lz_ws2022(tmp_path):
    test_config = {
        "experiment_name": "lz_ws2022",
        "livetime": 0.1643835616438356,
        "fiducial_mass": 5.5,
        "data": "lz_ws2022_wimp_data.csv",
        "roi": {"s1c": "np.linspace(3, 80, 78)", "s2c": "np.logspace(2.75, 5, 81)"},
        "bkgs": [
            {
                "bkg_name": "beta",
                "rate_nominal": 1307.9166666666667,
                "rate_uncertainty": 219.0,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "neutrino",
                "rate_nominal": 167.29166666666669,
                "rate_uncertainty": 9.733333333333334,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xe127",
                "rate_nominal": 55.96666666666667,
                "rate_uncertainty": 4.866666666666667,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xe124",
                "rate_nominal": 30.416666666666668,
                "rate_uncertainty": 8.516666666666666,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xe136",
                "rate_nominal": 91.85833333333333,
                "rate_uncertainty": 14.6,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "ar37",
                "rate_nominal": 304.1666666666667,
                "rate_fit_limits": [0, 1752.0],
                "args": {"batch_size": 1000},
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    lz_experiment_ws2022 = diamx.experiments.LZWS2022(test_config, tmp_path)
    lz_experiment_ws2022.get_data()
    lz_experiment_ws2022.get_bkg_templates()
    lz_experiment_ws2022.get_shaped_bkg_templates()


def test_lz_ws2024(tmp_path):
    test_config = {
        "experiment_name": "lz_ws2024",
        "livetime": 0.7671232876712328,
        "fiducial_mass": 5.5,
        "data": "lz_ws2024_wimp_data.csv",
        "roi": {"s1c": "np.linspace(3, 80, 78)", "s2c": "np.logspace(2.6, 4.5, 81)"},
        "bkgs": [
            {
                "bkg_name": "pb214",
                "rate_nominal": 968.5535714285714,
                "rate_uncertainty": 114.71428571428572,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "kr85_ar39_detgamma",
                "rate_nominal": 211.17857142857144,
                "rate_uncertainty": 28.67857142857143,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "solar_neutrino_er",
                "rate_nominal": 132.96428571428572,
                "rate_uncertainty": 7.821428571428572,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "pb212_po218",
                "rate_nominal": 81.73392857142858,
                "rate_uncertainty": 9.776785714285715,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "tritium_c14",
                "rate_nominal": 75.99821428571428,
                "rate_uncertainty": 4.301785714285714,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xe136",
                "rate_nominal": 72.47857142857143,
                "rate_uncertainty": 10.819642857142858,
                "args": {"batch_size": 1000},
            },
            {
                "bkg_name": "xe127_xe125",
                "rate_nominal": 4.171428571428572,
                "rate_uncertainty": 0.7821428571428571,
                "args": {"batch_size": 1000},
            },
        ],
        "shaped_bkgs": [
            {
                "shaped_bkg_name": "xe124",
                "shape_parameters": [
                    {
                        "shape_parameter_name": "dec_quenching_factor",
                        "shape_parameter_fittable": True,
                        "shape_parameter_range": "np.arange(65, 88, 2)",
                        "shape_parameter_nominal": 71,
                        "formatter": "d",
                        "shape_parameter_fit_limits": [65, 87],
                    }
                ],
                "rate_nominal": 25.289285714285715,
                "rate_uncertainty": 5.083928571428571,
                "args": {"batch_size": 1000},
            }
        ],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {"batch_size": 1000},
        },
    }
    lz_experiment_ws2024 = diamx.experiments.LZWS2024(test_config, tmp_path)
    lz_experiment_ws2024.get_data()
    lz_experiment_ws2024.get_bkg_templates()
    lz_experiment_ws2024.get_shaped_bkg_templates()


# The PandaX-4T analysis space is (cS1, log10(cS2_b/cS1)); the AC component is a
# contour-driven template loaded from disk whose binning must match the roi, so
# these tests use the real (fine) roi from the config.
_PANDAX_ROI = {
    "cs1": "np.linspace(2, 135, 134)",
    "logcs2_s1": "np.linspace(0.5, 3.5, 101)",
}


def test_pandax4t_run0(tmp_path):
    test_config = {
        "experiment_name": "pandax4t_run0",
        "livetime": 0.2268907563,
        "fiducial_mass": 2.38,
        "data": "pandax4t_run0_wimp_data.csv",
        "roi": _PANDAX_ROI,
        "bkgs": [
            {"bkg_name": "other_er", "rate_nominal": 2221.333, "args": {"batch_size": 5000}},
            {"bkg_name": "tritium", "rate_nominal": 2450.519, "args": {"batch_size": 5000}},
            {"bkg_name": "xe124", "rate_nominal": 10.137, "args": {"batch_size": 5000}},
            {"bkg_name": "xe127", "rate_nominal": 33.937, "args": {"batch_size": 5000}},
            {"bkg_name": "neutron", "rate_nominal": 2.6444, "args": {"batch_size": 5000}},
            {"bkg_name": "b8", "rate_nominal": 1.3222, "args": {"batch_size": 5000}},
            {
                "bkg_name": "ac",
                "rate_nominal": 48.4815,
                "args": {
                    "template_path": "pandax4t_run0_ac_template.h5",
                    "hist_name": "cs1-logcs2_s1",
                },
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    pandax_run0 = diamx.experiments.PandaX4TRun0(test_config, tmp_path)
    pandax_run0.get_data()
    pandax_run0.get_bkg_templates()
    pandax_run0.get_shaped_bkg_templates()


def test_pandax4t_run1(tmp_path):
    test_config = {
        "experiment_name": "pandax4t_run1",
        "livetime": 0.4032258065,
        "fiducial_mass": 2.48,
        "data": "pandax4t_run1_wimp_data.csv",
        "roi": _PANDAX_ROI,
        "bkgs": [
            {"bkg_name": "other_er", "rate_nominal": 3040.48, "args": {"batch_size": 5000}},
            {"bkg_name": "tritium", "rate_nominal": 282.72, "args": {"batch_size": 5000}},
            {"bkg_name": "xe124", "rate_nominal": 10.168, "args": {"batch_size": 5000}},
            {"bkg_name": "neutron", "rate_nominal": 2.728, "args": {"batch_size": 5000}},
            {"bkg_name": "b8", "rate_nominal": 1.736, "args": {"batch_size": 5000}},
            {
                "bkg_name": "ac",
                "rate_nominal": 32.24,
                "args": {
                    "template_path": "pandax4t_run1_ac_template.h5",
                    "hist_name": "cs1-logcs2_s1",
                },
            },
        ],
        "shaped_bkgs": [],
        "eff": {
            "fit_limits": [0.5, 1.5],
            "parameter_interval_bounds": [0.5, 1.5],
            "args": {},
        },
    }
    pandax_run1 = diamx.experiments.PandaX4TRun1(test_config, tmp_path)
    pandax_run1.get_data()
    pandax_run1.get_bkg_templates()
    pandax_run1.get_shaped_bkg_templates()
