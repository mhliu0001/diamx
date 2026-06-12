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
                    stabilized_parameter=stabilized_parameter,
                    fit_strategy=fit_strategy,
                )
                best_fit_value = best_fit[poi_name]

                ll_zero = None
                if exact_asymptotic:
                    extra_results = {}
                    lower, upper = alea_model.confidence_interval_asymptotic(
                        poi_name=poi_name,
                        stabilized_parameter=stabilized_parameter,
                        confidence_level=confidence_level,
                        fit_strategy=fit_strategy,
                        best_fit=best_fit,
                        best_ll=max_ll,
                        extra_results=extra_results,
                    )
                    ll_zero = extra_results.get("ll_zero")
                else:
                    lower, upper = alea_model.confidence_interval(
                        poi_name=poi_name,
                        stabilized_parameter=stabilized_parameter,
                        confidence_level=confidence_level,
                        confidence_interval_kind=confidence_interval_kind,
                        fit_strategy=fit_strategy,
                    )

                if ll_zero is None:
                    _, ll_zero = alea_model.fit(
                        **{poi_name: 0},
                        stabilized_parameter=stabilized_parameter,
                        fit_strategy=fit_strategy,
                    )
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


# Environment for inference worker processes, applied (for variables not
# already set) while the Pool is alive and removed afterwards. Workers only
# run Minuit/numpy fits; JAX is imported in each worker merely as a side
# effect of importing diamx, yet its CPU backend creates O(n_cores) threads
# per worker. On many-core shared nodes that multiplies to thousands of
# threads and exhausts the per-user thread limit (pthread_create EAGAIN
# crashes). Keep workers off the GPU and cap the XLA/BLAS thread pools.
_DEFAULT_WORKER_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": (
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
    ),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def run_inference_pool(
    context,
    confidence_level=0.9,
    confidence_interval_kind="central",
    fit_strategy=None,
    exact_asymptotic=True,
    stabilize_fit=False,
    output_file_name=None,
    processes=None,
    chunksize=1,
    maxtasksperchild=5,
    show_progress=True,
    start_method="spawn",
    timeout_seconds=3600,
    worker_env=None,
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
        Fit strategy options passed to DiamxModel.fit(). The default (None) uses
        the alea default: Minuit strategy 1 with an automatic strategy-2
        simplex+migrad refit if the optimization does not converge.
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
        Maximum tasks per worker process before recycling (default: 5). Worker
        processes are recycled after this many fits so that any memory retained
        per fit (e.g. template histograms held by Minuit reference cycles) is
        returned to the OS instead of accumulating over the mass scan. Set to
        None to keep workers alive for the whole pool (faster startup, higher
        peak memory).
    show_progress : bool, optional
        Whether to show a progress bar (default: True).
    start_method : str or None, optional
        Multiprocessing start method (default: "spawn"). If None, uses the default for the platform.
    timeout_seconds : int, optional
        Timeout in seconds for each worker fit (default: 3600).
    worker_env : dict or None, optional
        Environment variables for the worker processes (default: None, which
        uses _DEFAULT_WORKER_ENV: workers stay off the GPU and the XLA/BLAS
        thread pools are capped, since each worker otherwise creates
        O(n_cores) idle threads and many-core nodes can hit the per-user
        thread limit). Variables already set in the environment are left
        untouched, so exported values take precedence. The variables are only
        set while the pool is alive and removed afterwards: template
        generation for other signal models in the same process can still use
        the GPU. Pass {} to disable. On Linux the parent's CPU affinity is
        additionally restricted to one core per worker for the lifetime of
        the pool (and restored afterwards), which is what actually bounds the
        XLA pool sizes the workers create.

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

    n_processes = processes or mp_ctx.cpu_count()
    pool_kwargs = dict(processes=n_processes)
    if maxtasksperchild is not None:
        pool_kwargs["maxtasksperchild"] = maxtasksperchild

    results = []
    n_total = len(pool_parameters)

    # Apply the worker environment (only variables not already set, so
    # exported values win) and restrict the CPU affinity to one core per
    # worker while the pool is alive. Spawned workers inherit both. XLA sizes
    # its thread pools by the number of schedulable CPUs and ignores the
    # thread-count flags for some pools, so the affinity restriction is what
    # caps the per-worker thread count at O(processes) instead of O(n_cores);
    # the workers are single-threaded Minuit/numpy fits and lose nothing.
    # Everything is restored afterwards so that later work in this process
    # (e.g. template generation for another signal model) can use the GPU and
    # all cores again.
    if worker_env is None:
        worker_env = _DEFAULT_WORKER_ENV
    saved_env = {}
    for key, value in worker_env.items():
        # Treat set-but-empty the same as unset; only a non-empty exported
        # value takes precedence over the defaults.
        if not os.environ.get(key):
            saved_env[key] = os.environ.get(key)
            os.environ[key] = value
    original_affinity = None
    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        if n_processes < len(available_cpus):
            original_affinity = available_cpus
            os.sched_setaffinity(0, sorted(available_cpus)[:n_processes])

    # Optional progress
    progress_iter = None
    if show_progress:
        try:
            progress_iter = tqdm(total=n_total, desc="Running inference (Pool)")
        except Exception:
            progress_iter = None  # fall back to prints

    error = False
    try:
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
    finally:
        for key, previous in saved_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)

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
