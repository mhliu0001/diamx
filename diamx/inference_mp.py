import os
import multiprocessing as mp
import copy
import warnings
import numpy as np
from tqdm import tqdm
from diamx.utils import generate_bin_array, HiddenTqdm
from diamx.model import DiamxModel
import traceback


def _pool_task(pool_parameters):
    """Worker task: build model, run fit, return result row."""
    (
        signal_parameter_value,
        alea_config,
        poi_name,
        data_dict,
        stabilized_parameter,
        confidence_level,
        confidence_interval_kind,
        exact_asymptotic,
        fit_strategy,
    ) = pool_parameters
    try:
        with HiddenTqdm():  # silence internal prints
            alea_model = DiamxModel(**alea_config)
            data_dict_new = copy.deepcopy(data_dict)
            toy_data = alea_model.generate_data()

            data_dict_new["ancillary"] = toy_data["ancillary"]
            if len(data_dict_new["ancillary"]) > 0:  # Avoid empty ancillary data
                for name in data_dict_new["ancillary"].dtype.names:
                    data_dict_new["ancillary"][0][name] = alea_config[
                        "parameter_definition"
                    ][name]["nominal_value"]

            data_dict_new["generate_values"] = toy_data["generate_values"]

            alea_model.data = data_dict_new

            # best fit
            best_fit, max_ll = alea_model.fit(stabilized_parameter=stabilized_parameter)

            # CI
            if exact_asymptotic:
                # only central is supported by your asymptotic path
                lower, upper = alea_model.confidence_interval_asymptotic(
                    poi_name=poi_name,
                    stabilized_parameter=stabilized_parameter,
                    confidence_level=confidence_level,
                    fit_strategy=fit_strategy,
                )
            else:
                lower, upper = alea_model.confidence_interval(
                    poi_name=poi_name,
                    stabilized_parameter=stabilized_parameter,
                    confidence_level=confidence_level,
                    confidence_interval_kind=confidence_interval_kind,
                    fit_strategy=fit_strategy,
                )

            # Discovery Z (Cowan+ 2011 Eq. 52)
            _, ll_zero = alea_model.fit(**{poi_name: 0})
            significance = float(np.sqrt(2.0 * (np.clip(max_ll - ll_zero, 0, None))))

        return np.array(
            [signal_parameter_value, lower, upper, significance], dtype=float
        )
    except Exception:
        tb = traceback.format_exc(limit=8)
        # Return a soft error record; the parent will print and continue
        return ("__ERR__", signal_parameter_value, tb)


def run_inference_pool(
    context,
    confidence_level=0.9,
    confidence_interval_kind="central",
    fit_strategy={"minuit_strategy": 2},
    exact_asymptotic=True,
    stabilize_fit=False,
    output_file_name=None,
    processes=None,  # default: mp.cpu_count()
    chunksize=1,  # tune for many tiny tasks; for long fits 1 is fine
    maxtasksperchild=None,  # set e.g. 50 to recycle workers if you suspect leaks
    show_progress=True,
    start_method=None,  # e.g. "spawn" for cross-platform consistency
):
    """
    Multiprocessing (Pool) version of run_inference().
    """
    # Resolve output name
    if output_file_name is None:
        output_file_name = f"ci_{context.config['signal']['signal_name']}.csv"
    out_path = os.path.join(context.output_path, output_file_name)

    if stabilize_fit:
        stabilized_parameter = (
            f"{context.config['signal']['signal_name']}_rate_multiplier"
        )
    else:
        stabilized_parameter = None
    poi_name = f"{context.config['signal']['signal_name']}_rate_multiplier"

    # Build the pool parameters
    signal_grid = generate_bin_array(
        context.config["signal"]["parameter_range"]
    ).tolist()
    pool_parameters = []
    data_dict = {}
    for experiment_instance in context.experiment_instances:
        data_dict[experiment_instance.experiment_name] = experiment_instance.get_data()
    for v in signal_grid:
        alea_config = context.update_alea_config_signal(v)
        if alea_config is None:
            # e.g. invalid spectrum that gives zero events, skip
            continue
        pool_parameters.append(
            (
                v,
                copy.deepcopy(alea_config),
                poi_name,
                data_dict,
                stabilized_parameter,
                confidence_level,
                confidence_interval_kind,
                exact_asymptotic,
                fit_strategy,
            )
        )

    if len(pool_parameters) == 0:
        warnings.warn("Empty signal parameter grid.")
        return

    # Choose a context if requested (spawn is safest cross-platform)
    mp_ctx = mp.get_context(start_method) if start_method else mp

    pool_kwargs = dict(
        processes=processes or mp_ctx.cpu_count(),
    )
    if maxtasksperchild is not None:
        pool_kwargs["maxtasksperchild"] = maxtasksperchild

    results = []
    n_total = len(pool_parameters)

    # Optional progress
    progress_iter = None
    if show_progress:
        try:
            progress_iter = tqdm(total=n_total, desc="Running inference (Pool)")
        except Exception:
            progress_iter = None  # fall back to prints

    error = False
    with mp_ctx.Pool(**pool_kwargs) as pool:
        # Stream results as they complete; imap_unordered yields as tasks finish
        for res in pool.imap_unordered(
            _pool_task, pool_parameters, chunksize=chunksize
        ):
            if isinstance(res, tuple) and len(res) == 3 and res[0] == "__ERR__":
                # Soft error from worker
                _, spv, tb = res
                print(f"\n[worker error] signal={spv}\n{tb}")
                error = True
            elif res is None:
                # Skipped parameter (e.g. invalid template)
                pass
            else:
                results.append(res)

            if progress_iter is not None:
                progress_iter.update(1)

    if progress_iter is not None:
        progress_iter.close()

    if len(results) == 0:
        warnings.warn("No valid results produced in run_inference_pool().")
        if error:
            raise RuntimeError("Errors occurred in worker processes; see output above.")
        return

    # Stack + sort by parameter (unordered stream)
    arr = np.vstack(results)
    arr = arr[np.argsort(arr[:, 0])]

    # Single write from parent
    np.savetxt(out_path, arr, delimiter=",")

    if error:
        raise RuntimeError(
            "Errors occurred in worker processes; see output above. "
            "The output file is still created."
        )
