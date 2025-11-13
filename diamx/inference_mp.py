import os
import multiprocessing as mp
import copy
import warnings
import numpy as np
from tqdm import tqdm
from diamx.utils import generate_bin_array, HiddenTqdm
from diamx.model import DiamxModel
import traceback
import signal
import functools


# ---------- Timeout decorator (Unix SIGALRM; warns & disables elsewhere) ----------
def timeout(seconds=300):
    """
    Decorator to raise TimeoutError if the wrapped function runs longer than `seconds`.
    Uses signal.SIGALRM (Unix only). On unsupported platforms (e.g. Windows),
    emits a RuntimeWarning (once per process) and disables timeout.
    """

    def decorator(func):
        if not hasattr(signal, "SIGALRM"):
            warned = False

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal warned
                if not warned:
                    warnings.warn(
                        f"Timeout not available on this platform; '{func.__name__}' "
                        f"will not be interrupted.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned = True
                return func(*args, **kwargs)

            return wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(
                    f"Function '{func.__name__}' timed out after {seconds} seconds"
                )

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


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
        timeout_seconds,
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

            # Define the work function without decoration first…
            def do_fit(
                alea_model,
                poi_name,
                stabilized_parameter,
                confidence_level,
                confidence_interval_kind,
                fit_strategy,
                exact_asymptotic,
            ):
                best_fit, max_ll = alea_model.fit(
                    stabilized_parameter=stabilized_parameter
                )
                best_fit_value = best_fit[poi_name]

                if exact_asymptotic:
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

                _, ll_zero = alea_model.fit(**{poi_name: 0})
                significance = float(
                    np.sqrt(2.0 * (np.clip(max_ll - ll_zero, 0, None)))
                )
                return np.array([lower, upper, significance, best_fit_value], float)

            # …then wrap it with the *current* timeout value.
            do_fit_with_timeout = timeout(timeout_seconds)(do_fit)

            lower, upper, significance, best_fit_value = do_fit_with_timeout(
                alea_model,
                poi_name,
                stabilized_parameter,
                confidence_level,
                confidence_interval_kind,
                fit_strategy,
                exact_asymptotic,
            )

        return np.array(
            [signal_parameter_value, lower, upper, significance, best_fit_value],
            dtype=float,
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
    processes=None,
    chunksize=1,
    maxtasksperchild=None,
    show_progress=True,
    start_method="spawn",
    timeout_seconds=3600,
):
    """
    Multiprocessing (Pool) version of diamx.Context.run_inference.

    Parameters
    ----------
    context : diamx.Context
        The Diamx Context with experiments registered and templates generated.
    confidence_level : float, optional
        Confidence level for the intervals (default: 0.9).
    confidence_interval_kind : str, optional
        Kind of confidence interval: "central", "upper" or "lower" (default: "central").
    fit_strategy : dict, optional
        Fit strategy options passed to DiamxModel.fit() (default: {"minuit_strategy": 2}).
    exact_asymptotic : bool, optional
        Whether to use exact asymptotic formulae for confidence intervals
        (default: True). If False, uses a naive chi-squared approximation.
    stabilize_fit : bool, optional
        Whether to stabilize the fit by profiling over a rate multiplier
        (default: False).
    output_file_name : str or None, optional
        Name of the output CSV file (default: None, which uses
        "ci_{signal_name}.csv").
    processes : int or None, optional
        Number of worker processes to use (default: None, which uses mp.cpu_count()).
    chunksize : int, optional
        Number of tasks per worker chunk (default: 1).
    maxtasksperchild : int or None, optional
        Maximum tasks per worker process before recycling (default: None).
    show_progress : bool, optional
        Whether to show a progress bar (default: True).
    start_method : str or None, optional
        Multiprocessing start method (default: "spawn"). If None, uses the default for the platform.
    timeout_seconds : int, optional
        Timeout in seconds for each worker fit (default: 3600).

    Returns
    -------
    None
        Writes the confidence interval results to a CSV file.
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
                timeout_seconds,  # <-- pass through
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
