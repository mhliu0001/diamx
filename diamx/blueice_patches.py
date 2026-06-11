"""Performance patches for blueice, applied when diamx is imported.

``LogLikelihoodBase.source_shape_parameters`` is an uncached property and
``_get_shape_indices`` rescans it; both are evaluated several times per source
per likelihood call from the interpolator closures (~18% of the total
inference CPU time in profiling, ~3.4M evaluations per confidence interval).
Both only change when a shape parameter is added, so memoize them on the
instance and invalidate in ``add_shape_parameter``.

The equivalent fix is upstreamable to blueice (compute once in ``prepare()``);
remove this module once it lands and the pinned version includes it.
"""

from blueice.likelihood import LogLikelihoodBase

_orig_source_shape_parameters = LogLikelihoodBase.source_shape_parameters.fget
_orig_get_shape_indices = LogLikelihoodBase._get_shape_indices
_orig_add_shape_parameter = LogLikelihoodBase.add_shape_parameter


def _source_shape_parameters(self):
    cached = self.__dict__.get("_diamx_source_shape_parameters")
    if cached is None:
        cached = _orig_source_shape_parameters(self)
        self.__dict__["_diamx_source_shape_parameters"] = cached
    return cached


def _get_shape_indices(self, source_name):
    cache = self.__dict__.setdefault("_diamx_shape_indices", {})
    indices = cache.get(source_name)
    if indices is None:
        indices = cache[source_name] = _orig_get_shape_indices(self, source_name)
    return indices


def _add_shape_parameter(self, *args, **kwargs):
    self.__dict__.pop("_diamx_source_shape_parameters", None)
    self.__dict__.pop("_diamx_shape_indices", None)
    return _orig_add_shape_parameter(self, *args, **kwargs)


if not getattr(LogLikelihoodBase, "_diamx_patched", False):
    LogLikelihoodBase.source_shape_parameters = property(
        _source_shape_parameters, doc=LogLikelihoodBase.source_shape_parameters.__doc__
    )
    LogLikelihoodBase._get_shape_indices = _get_shape_indices
    LogLikelihoodBase.add_shape_parameter = _add_shape_parameter
    LogLikelihoodBase._diamx_patched = True
