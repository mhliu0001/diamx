from jax import jit
from functools import partial

from appletree.plugin import Plugin
from appletree.config import takes_config, Constant
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@takes_config(
    Constant(name="cut_acceptance", type=float, default=0.8, help="Flat cut acceptance")
)
class TotalEffNaive(Plugin):
    depends_on = ["acc_s1_recon_eff"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, acc_s1_recon_eff):
        return key, acc_s1_recon_eff * self.cut_acceptance.value
