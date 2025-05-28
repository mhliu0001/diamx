from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree.plugin import Plugin
from appletree.config import takes_config, Map, SigmaMap, Constant
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@takes_config(
    SigmaMap(
        name="energy_eff",
        method="LERP",
        default=[
            "xenonnt_sr0_eff_median.json",
            "xenonnt_sr0_eff_lower.json",
            "xenonnt_sr0_eff_upper.json",
            "xenonnt_eff_sigma",
        ],
        help="Total efficiency with energy, considering detection, 3-fold selection and cut acceptances",
    ),
)
class EnergyEff(Plugin):
    depends_on = ["energy"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        eff = self.energy_eff.apply(energy, parameters)
        eff = jnp.clip(eff, 0.0, 1.0)
        return key, eff


class CES(Plugin):
    depends_on = ["cs1", "cs2"]
    provides = ["e_ces"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, cs1, cs2):
        e_ces = parameters["w"] * (cs1 / parameters["g1"] + cs2 / parameters["g2"])
        return key, e_ces
