import numpy as np
from alea.models.blueice_extended_model import CustomAncillaryLikelihood


class FastAncillaryLikelihood(CustomAncillaryLikelihood):
    """CustomAncillaryLikelihood with Gaussian constraint terms in closed form.

    The parent class evaluates each constraint through a frozen
    scipy.stats.norm.logpdf call per parameter per likelihood evaluation; the
    scipy input validation and broadcasting overhead of those scalar calls
    dominates the cost of the whole likelihood (35-55% of the total inference
    CPU time in profiling). This subclass caches the Gaussian centers and
    widths as arrays and evaluates the sum of log-pdfs in one vectorized
    expression, identical to the parent to floating point accuracy.

    Non-Gaussian constraints (string-based uncertainties, which alea freezes
    as generic scipy distributions) keep the generic logpdf path.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self._prepare_constraint_arrays()

    def set_data(self, d):
        """Set the ancillary measurements and refresh the cached arrays."""
        super().set_data(d)
        self._prepare_constraint_arrays()

    def _prepare_constraint_arrays(self):
        gauss_names, mus, sigmas = [], [], []
        other_terms = {}
        for name, func in self.constraint_functions.items():
            if getattr(getattr(func, "dist", None), "name", None) == "norm":
                gauss_names.append(name)
                mus.append(func.mean())
                sigmas.append(func.std())
            else:
                other_terms[name] = func.logpdf
        self._gauss_names = gauss_names
        self._gauss_mu = np.asarray(mus)
        self._gauss_inv_sigma = 1.0 / np.asarray(sigmas)
        self._gauss_lognorm = float(
            np.sum(np.log(self._gauss_inv_sigma))
            - 0.5 * len(mus) * np.log(2 * np.pi)
        )
        self._other_constraint_terms = other_terms

    def ancillary_sum(self, evaluate_at: dict) -> float:
        """Return the sum of all constraint terms.

        Args:
            evaluate_at (dict): Values of the ancillary measurements.

        Returns:
            float: Sum of all constraint terms.
        """
        z = (
            np.array([evaluate_at[n] for n in self._gauss_names]) - self._gauss_mu
        ) * self._gauss_inv_sigma
        result = self._gauss_lognorm - 0.5 * float(z @ z)
        for name, logpdf in self._other_constraint_terms.items():
            result += float(logpdf(evaluate_at[name]))
        return result
