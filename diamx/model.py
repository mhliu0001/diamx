from typing import Optional, Tuple, Callable
from alea import BlueiceExtendedModel
from alea.utils import within_limits
from scipy.optimize import brentq
from copy import deepcopy
import numpy as np
from blueice.likelihood import _needs_data
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from diamx.asimov import get_asimov_sigma


class DiamxModel(BlueiceExtendedModel):
    """
    The Diamx model class extends the BlueiceExtendedModel class, mainly to implement a custom
    fitting interface to stabilize the fit, and the exact asymptotic confidence interval calculation
    from Cowan et al. (2011) (https://arxiv.org/abs/1007.1727).
    """

    @_needs_data
    def fit(
        self,
        stabilized_parameter: Optional[str] = None,
        verbose: Optional[bool] = False,
        fit_strategy: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[dict, float]:
        """Try to stabilize the fit by fixing stabilized_parameter in the fit."""
        if stabilized_parameter is None or stabilized_parameter in kwargs:
            return super().fit(verbose=verbose, fit_strategy=fit_strategy, **kwargs)

        if stabilized_parameter not in self.parameters.names:
            raise ValueError(
                f"Parameter {stabilized_parameter} not in model parameters."
            )

        def ll_to_minimize(x):
            fit_args = kwargs.copy()
            fit_args[stabilized_parameter] = x
            _, ll = super(DiamxModel, self).fit(
                verbose, fit_strategy=fit_strategy, **fit_args
            )
            return -ll

        par = self.parameters[stabilized_parameter]
        re = minimize(ll_to_minimize, x0=par.nominal_value, bounds=[par.fit_limits])
        if not re.success:
            warnings.warn(f"Fit failed to converge. Fixed parameters: {kwargs}")

        fit_args = kwargs.copy()
        fit_args[stabilized_parameter] = re.x[0]
        return super().fit(**fit_args)

    def confidence_interval_asymptotic(
        self,
        poi_name: str,
        stabilized_parameter: Optional[str] = None,
        parameter_interval_bounds: Optional[Tuple[float, float]] = None,
        confidence_level: Optional[float] = 0.9,
        fit_strategy: Optional[dict] = {"minuit_strategy": 2},
        best_fit: Optional[dict] = None,
        best_ll: Optional[float] = None,
        brentq_rtol: float = 1e-3,
        extra_results: Optional[dict] = None,
    ) -> Tuple[float, float]:
        """Compute asymptotic confidence intervals for a certain named parameter.

        Args:
            poi_name (str): name of the parameter of interest
            stabilized_parameter (str, optional (default=None)): name of the parameter to
                stabilize the fit by fixing it in the fit.
            confidence_level (float, optional (default=None)):
                confidence level for confidence intervals.
                If None, the default confidence level of the model is used.
            fit_strategy (dict, optional (default=None)): strategy for the fit,
                see _DEFAULT_FIT_STRATEGY for possible settings.
            best_fit (dict, optional (default=None)): unconditional best-fit parameters.
                Pass together with best_ll to reuse a fit the caller already performed
                instead of refitting here.
            best_ll (float, optional (default=None)): log-likelihood at best_fit.
            brentq_rtol (float, optional (default=1e-3)): relative tolerance of the
                brentq root search for the interval edges. The Asimov sigma entering
                the p-value fluctuates at the per-mille level between Minuit/Hesse
                evaluations, so the root cannot be resolved much more finely anyway;
                the scipy default (machine precision) wastes about half of the
                p-value evaluations after the root has converged.
            extra_results (dict, optional (default=None)): if a dict is passed,
                intermediate results are stored into it. Currently: "ll_zero", the
                conditional log-likelihood with the poi fixed to 0 (reusable for the
                discovery significance without an extra fit).
        """
        if best_fit is None or best_ll is None:
            best_fit, best_ll = self.fit(
                stabilized_parameter=stabilized_parameter, fit_strategy=fit_strategy
            )
        parameter_of_interest = self.parameters[poi_name]
        if not parameter_of_interest.fittable:
            raise ValueError("The parameter of interest must be fittable")

        if parameter_interval_bounds is None:
            parameter_interval_bounds = parameter_of_interest.parameter_interval_bounds

        assert (
            parameter_interval_bounds[0] == 0
        ), "Asymptotic CI only implemented for lower bound at 0."

        # Each p-value evaluation needs the conditional fit twice (test statistic
        # and Asimov dataset) and brentq re-evaluates its bracket endpoints, so
        # memoize conditional fits and p-values by hypothesis value.
        conditional_fit_cache = {}

        def conditional_fit(hypothesis_value):
            if hypothesis_value not in conditional_fit_cache:
                conditional_fit_cache[hypothesis_value] = self.fit(
                    **{poi_name: hypothesis_value},
                    stabilized_parameter=stabilized_parameter,
                    fit_strategy=fit_strategy,
                )
            return conditional_fit_cache[hypothesis_value]

        def t_tilde(hypothesis_value):
            _, ll = conditional_fit(hypothesis_value)
            # Clip the test statistic to be non-negative
            return np.clip(2.0 * (best_ll - ll), 0, None)

        def cumulative_t_tilde(hypothesis_value):
            t_tilde_value = t_tilde(hypothesis_value)
            conditional_best_fit, _ = conditional_fit(hypothesis_value)
            sigma = get_asimov_sigma(
                self,
                poi_name,
                hypothesis_value,
                fit_strategy=fit_strategy,
                conditional_best_fit=conditional_best_fit,
            )
            if (
                hypothesis_value == 0
                or t_tilde_value <= (hypothesis_value / sigma) ** 2
            ):
                # If mu = 0, then hypothesis_value/sigma = 0, so we always use the first case
                return 2 * norm.cdf(np.sqrt(t_tilde_value)) - 1
            else:
                return (
                    norm.cdf(np.sqrt(t_tilde_value))
                    + norm.cdf(
                        (t_tilde_value + (hypothesis_value / sigma) ** 2)
                        / (2 * hypothesis_value / sigma)
                    )
                    - 1
                )

        p_value_cache = {}

        def p_value(hypothesis_value):
            if hypothesis_value not in p_value_cache:
                p_value_cache[hypothesis_value] = 1 - cumulative_t_tilde(
                    hypothesis_value
                )
            return p_value_cache[hypothesis_value]

        best_p_value = p_value(best_fit[poi_name])
        if best_p_value < 1 - confidence_level:
            warnings.warn(
                f"The best-fit {best_fit[poi_name]} has a p-value {best_p_value} "
                f"lower than 1-confidence_level {1-confidence_level}. Cannot compute "
                f"confidence interval."
            )
            return np.nan, np.nan
        lower_p_value = p_value(parameter_interval_bounds[0])
        upper_p_value = p_value(parameter_interval_bounds[1])
        if extra_results is not None:
            extra_results["ll_zero"] = conditional_fit_cache[
                parameter_interval_bounds[0]
            ][1]

        if lower_p_value < 1 - confidence_level:
            dl = brentq(
                lambda x: p_value(x) - (1 - confidence_level),
                parameter_interval_bounds[0],
                best_fit[poi_name],
                rtol=brentq_rtol,
            )
        else:
            dl = -1 * np.inf
        if upper_p_value < 1 - confidence_level:
            ul = brentq(
                lambda x: p_value(x) - (1 - confidence_level),
                best_fit[poi_name],
                parameter_interval_bounds[1],
                rtol=brentq_rtol,
            )
        else:
            ul = np.inf
        return dl, ul

    def confidence_interval(
        self,
        poi_name: str,
        stabilized_parameter: Optional[str] = None,
        parameter_interval_bounds: Optional[Tuple[float, float]] = None,
        confidence_level: Optional[float] = None,
        confidence_interval_kind: Optional[str] = None,
        confidence_interval_threshold: Optional[Callable[[float], float]] = None,
        confidence_interval_args: Optional[dict] = None,
        best_fit_args: Optional[dict] = None,
        asymptotic_dof: Optional[int] = None,
        fit_strategy: Optional[dict] = None,
    ) -> Tuple[float, float]:
        """Uses self.fit to compute confidence intervals for a certain named parameter. If the
        parameter is a rate parameter, and the model has expectation values implemented, the bounds
        will be interpreted as bounds on the expectation value, so that the range in the fit is
        parameter_interval_bounds/mus. Otherwise the bound is taken as-is.

        Args:
            poi_name (str): name of the parameter of interest
            parameter_interval_bounds (Tuple[float, float], optional (default=None)): range
                in which to search for the confidence interval edges. May be specified as:
                    - setting the property "parameter_interval_bounds" for the parameter
                    - passing a list here
                    - passing None here, the property of the parameter is used
            confidence_level (float, optional (default=None)):
                confidence level for confidence intervals.
                If None, the default confidence level of the model is used.
            confidence_interval_kind (str, optional (default=None)):
                kind of confidence interval to compute.
                If None, the default kind of the model is used.
            confidence_interval_args (dict, optional (default=None)): Parameters that will be fixed
                in the profile likelihood computation. If None, all fittable parameters
                will be profiled except the poi.
            best_fit_args (dict, optional (default=None)): If you require the "global" best-fit
                used to normalise the profile likelihood ratio to fix fewer parameters than the
                profile likelihood-- mainly used for 1-D slices of higher-dimensional confidence
                volumes, where the global best-fit may not be along the profile.
                If None, will be set to confidence_interval_args.
            asymptotic_dof (int, optional (default=None)): Degrees of freedom for asymptotic
            fit_strategy (dict, optional (default=None)): strategy for the fit,
                see _DEFAULT_FIT_STRATEGY for possible settings.

        """
        if confidence_interval_args is None:
            confidence_interval_args = {}
        if best_fit_args is None:
            best_fit_args = confidence_interval_args
        ci_objects = self._confidence_interval_checks(
            poi_name,
            parameter_interval_bounds,
            confidence_level,
            confidence_interval_kind,
            confidence_interval_threshold,
            asymptotic_dof,
            **confidence_interval_args,
        )
        (
            confidence_interval_kind,
            confidence_interval_threshold,
            parameter_interval_bounds,
        ) = ci_objects

        # best_fit_args only provides the best-fit likelihood
        _, best_ll = self.fit(
            stabilized_parameter=stabilized_parameter,
            **best_fit_args,
            fit_strategy=fit_strategy,
        )
        # the optimization of profile-likelihood under
        # confidence_interval_args provides the best_parameter
        best_result, _ = self.fit(
            stabilized_parameter=stabilized_parameter,
            **confidence_interval_args,
            fit_strategy=fit_strategy,
        )
        best_parameter = best_result[poi_name]
        mask = within_limits(best_parameter, parameter_interval_bounds)
        if not mask:
            raise ValueError(
                f"The best-fit {best_parameter} is outside your confidence interval "
                f"search limits in parameter_interval_bounds {parameter_interval_bounds}."
            )

        # define intersection between likelihood ratio curve and the critical curve:
        def t(hypothesis_value):
            # define the intersection
            # between the profile-log-likelihood curve and the rejection threshold
            _confidence_interval_args = deepcopy(confidence_interval_args)
            _confidence_interval_args[poi_name] = hypothesis_value
            _, ll = self.fit(
                stabilized_parameter=stabilized_parameter,
                **_confidence_interval_args,
                fit_strategy=fit_strategy,
            )  # ll is + log-likelihood here
            ret = 2.0 * (best_ll - ll)  # likelihood curve "right way up" (smiling)
            # if positive, hypothesis is excluded
            return ret - confidence_interval_threshold(hypothesis_value)

        t_best_parameter = t(best_parameter)

        if t_best_parameter > 0:
            warnings.warn(
                f"CL calculation failed, given fixed parameters {confidence_interval_args}."
            )

        if confidence_interval_kind in {"upper", "central"} and t_best_parameter < 0:
            if t(parameter_interval_bounds[1]) > 0:
                ul = brentq(t, best_parameter, parameter_interval_bounds[1])
            else:
                ul = np.inf
        else:
            ul = np.nan

        if confidence_interval_kind in {"lower", "central"} and t_best_parameter < 0:
            if t(parameter_interval_bounds[0]) > 0:
                dl = brentq(t, parameter_interval_bounds[0], best_parameter)
            else:
                dl = -1 * np.inf
        else:
            dl = np.nan

        return dl, ul
