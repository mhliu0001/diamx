import warnings
import numpy as np
from scipy.special import xlogy, gammaln
from blueice.likelihood import LogLikelihoodBase
from alea.template_source import TemplateSource
from blueice.utils import inherit_docstring_from
from collections import OrderedDict
from blueice.likelihood import UnbinnedLogLikelihood, LogLikelihoodSum
from alea.models.blueice_extended_model import CustomAncillaryLikelihood
from copy import deepcopy

class ExtendedBinnedLogLikelihood(LogLikelihoodBase):
    """
    Binned extended Poisson log-likelihood that accepts non-integer bin contents.
    """
    def set_data(self, d):
        """
        Set the data histogram. The difference with the LogLikelihoodBase class
        is that this class accepts histograms, not individual events, and the bin
        contents can be non-integer values.

        Parameters
        ----------
        d : Histdd
            Data histogram (can contain non-integer values).
        """
        for base_source in self.base_model.sources:
            assert isinstance(base_source, TemplateSource), "ExtendedBinnedLogLikelihood only supports TemplateSource."
        LogLikelihoodBase.set_data(self, d)
    
    def prepare(self, *args):
        """
        """
        LogLikelihoodBase.prepare(self, *args)
        for base_source in self.base_model.sources:
            assert isinstance(base_source, TemplateSource), "ExtendedBinnedLogLikelihood only supports TemplateSource."
        if len(self.shape_parameters):
            assert self.source_wise_interpolation, "ExtendedBinnedLogLikelihood only supports source-wise interpolation."
            self.ps_interpolators = OrderedDict()
            analysis_space_dims = None
            for sn, base_source in zip(self.source_name_list, self.base_model.sources):
                assert isinstance(base_source, TemplateSource), "ExtendedBinnedLogLikelihood only supports TemplateSource."
                _, bins = zip(*base_source.config['analysis_space'])
                if analysis_space_dims is None:
                    analysis_space_dims = [len(bin)-1 for bin in bins]
                else:
                    assert analysis_space_dims == [len(bin)-1 for bin in bins], "All sources must have the same analysis space."
                if sn in self.source_morphers:
                    self.ps_interpolators[sn] = self.source_morphers[sn].make_interpolator(
                        f=lambda s: s.get_pmf_grid()[0],
                        extra_dims=analysis_space_dims,
                        anchor_models=self.anchor_sources[sn])
                else:
                    self.ps_interpolators[sn] = base_source.get_pmf_grid()[0]

            def ps_interpolator(*args):
                # take zs, convert to values for each source's interpolator call the respective interpolator
                ps = np.zeros([len(self.source_name_list)] + analysis_space_dims)
                for i, (sn, ps_interpolator) in enumerate(self.ps_interpolators.items()):
                    if sn in self.source_shape_parameters:
                        these_args = np.asarray([args[0][j] for j in self._get_shape_indices(sn)])
                        ps[i] = ps_interpolator(these_args)
                    else:
                        ps[i] = ps_interpolator
                return ps
            self.ps_interpolator = ps_interpolator
        else:
            self.ps = np.stack([s.get_pmf_grid()[0] for s in self.base_model.sources])
    
    @inherit_docstring_from(LogLikelihoodBase)
    def _compute_single_pdf(self, **kwargs):
        model = self._compute_single_model(**kwargs)
        mus = model.expected_events()
        ps, n_model_events = model.pmf_grids()
        return mus, ps, n_model_events

    def _compute_likelihood(self, mus, pmfs):
        """Return binned Poisson log likelihood
        :param mus: numpy array with expected rates for each source
        :param pmfs: array (sources, *analysis_space) of PMFs for each source in each bin
        """
        expected_counts = pmfs.copy()
        for mu, _p_bin_source in zip(mus, expected_counts):
            _p_bin_source *= mu         # Works because of numpy view magic...
        expected_total = np.sum(expected_counts, axis=0)

        observed_counts = self._data

        ret = extended_poisson_logpmf(observed_counts, expected_total)
        return np.sum(ret)

def extended_poisson_logpmf(observed, expected, atol=0.0):
    """
    Vectorized extended-Poisson log pmf allowing non-integer observed.
    log L = observed * log(expected) - expected - gammaln(observed + 1)

    Parameters
    ----------
    observed : array_like (>=0)
        May be non-integer (e.g. Asimov bin contents).
    expected : array_like (>=0)
        Expected mean(s). Values within [-atol, 0) are clipped to 0.
    atol : float
        Non-neg tolerance for tiny negative expected due to numerical jitter.

    Returns
    -------
    logpmf : ndarray
    """
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)

    if np.any(exp < -atol):
        raise ValueError("Expected events must be non-negative (within tolerance).")
    if np.any(obs < -atol):
        raise ValueError("Observed events must be non-negative (within tolerance).")

    # Clip tiny negatives to zero (numerical jitter)
    exp = np.where(exp < 0.0, 0.0, exp)
    obs = np.where(obs < 0.0, 0.0, obs)

    # Base formula using xlogy to avoid 0*log(0) issues
    logL = xlogy(obs, exp) - exp - gammaln(obs + 1.0)

    # Enforce the exact corner case: exp==0 & obs>0 -> -inf
    mask_impossible = (exp == 0.0) & (obs > 0.0)
    if np.any(mask_impossible):
        warnings.warn("Some bins have expected=0 but observed>0; setting log likelihood to be 0.", RuntimeWarning)
        logL[mask_impossible] = 0.0  # logL = -inf, but set to 0 for sum

    # exp==0 & obs==0 -> 0 exactly (already handled by xlogy and gammaln)
    return logL


def get_asimov_sigma(alea_model, poi_name, poi_value):
    """
    Given the signal strength parameter value, calculate the standard deviation sigma
    of the estimator of the signal strength, using the Asimov dataset.
    
    Parameters
    ----------
    alea_model : alea.Model
        The alea model from which to generate the asimov dataset.
    poi_name : str
        Name of the parameter of interest (signal strength).
    poi_value : float
        Value of the parameter of interest (signal strength).
        
    Returns
    -------
    sigma : float
        Estimated standard deviation of the signal strength estimator.
    """
    # Get the best-fit shape / rate parameters for the Asimov dataset
    best_fit_parameters, _ = alea_model.fit(**{poi_name: poi_value})

    # Generate the Asimov model
    asimov_model = deepcopy(alea_model)
    old_lls = asimov_model._likelihood.likelihood_list
    new_lls = []
    for ll_idx, ll in enumerate(old_lls):
        if isinstance(ll, UnbinnedLogLikelihood):
            extended_binned_llh = ExtendedBinnedLogLikelihood(
                pdf_base_config=ll.pdf_base_config,
                likelihood_config=ll.config,
                source_wise_interpolation=ll.source_wise_interpolation
            )
            for rate_parameter_name, log_prior in ll.rate_parameters.items():
                extended_binned_llh.add_rate_parameter(rate_parameter_name, log_prior=log_prior)
            for shape_parameter_name, (anchors, log_prior, base_value) in ll.shape_parameters.items():
                extended_binned_llh.add_shape_parameter(shape_parameter_name, anchors=anchors, log_prior=log_prior, base_value=base_value)
            extended_binned_llh.prepare()
            
            # Translate kwargs to rate_multipliers and shape_parameter_settings
            all_parameters = asimov_model._likelihood.likelihood_parameters[ll_idx]
            rate_multipliers, shape_parameter_settings = extended_binned_llh._kwargs_to_settings(**{
                k: v for k, v in best_fit_parameters.items() if k in all_parameters
            })

            if len(extended_binned_llh.shape_parameters) > 0:
                # Translate shape_parameter_settings to zs for interpolator
                zs = []
                for setting_name, (_, log_prior, _) in extended_binned_llh.shape_parameters.items():
                    z = shape_parameter_settings[setting_name]
                    zs.append(z)
                zs = np.asarray(zs)
                
                # Generate asimov dataset
                ps = extended_binned_llh.ps_interpolator(zs)
                mus = extended_binned_llh.mus_interpolator(zs)
            else:
                # No shape parameters, so just get the rates and PMFs directly
                mus = extended_binned_llh.base_model.expected_events()
                ps = extended_binned_llh.ps

            mus *= rate_multipliers
            asimov_dataset = np.sum(mus.reshape((-1,) + (1,) * (len(ps.shape) - 1)) * ps, axis=0)
            
            extended_binned_llh.set_data(asimov_dataset)
            new_lls.append(extended_binned_llh)
        else:
            assert isinstance(ll, CustomAncillaryLikelihood), f"Unrecognized likelihood {type(ll)} in alea_model.likelihood_list."
            new_lls.append(ll)
    asimov_model._likelihood = LogLikelihoodSum(new_lls, likelihood_weights=asimov_model._likelihood.likelihood_weights)
    
    asimov_fit_parameters, _ = asimov_model.fit()
    asimov_model.minuit_object.hesse()
    sigma = asimov_model.minuit_object.errors[poi_name]
    
    if np.abs(asimov_fit_parameters[poi_name] - poi_value) > sigma:
        warnings.warn("Asimov fit did not recover the input signal strength within 1 sigma.", RuntimeWarning)
    return sigma