from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree.plugin import Plugin
from appletree.config import takes_config, Map, SigmaMap, Constant
from appletree.utils import exporter
from appletree import randgen

export, __all__ = exporter(export_self=False)


@export
@takes_config(
    SigmaMap(
        name="s1_recon_eff",
        method="NN",
        default=[
            "xenonnt_sr0_s1_recon_eff_median.json",
            "xenonnt_sr0_s1_recon_eff_lower.json",
            "xenonnt_sr0_s1_recon_eff_upper.json",
        ],
        help="3fold S1 reconstruction efficiency on S1 phd",
    ),
)
class S1ReconEffNaive(Plugin):
    depends_on = ["num_s1_phd"]
    provides = ["acc_s1_recon_eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_s1_phd):
        acc_s1_recon_eff = self.s1_recon_eff.apply(num_s1_phd, parameters)
        acc_s1_recon_eff = jnp.clip(acc_s1_recon_eff, 0.0, 1.0)
        return key, acc_s1_recon_eff


@export
@export
@takes_config(
    SigmaMap(
        name="cut_acc",
        method="NN",
        default=[
            "xenonnt_cut_acc_median.json",
            "xenonnt_cut_acc_lower.json",
            "xenonnt_cut_acc_upper.json",
            "cut_acc_sigma",
        ],
        help="Cut acceptance on energy",
    ),
)
class CutAcc(Plugin):
    depends_on = ["energy"]
    provides = ["acc_cut"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        acc_cut = self.cut_acc.apply(energy, parameters)
        acc_cut = jnp.clip(acc_cut, 0.0, 1.0)
        return key, acc_cut


class TotalEff(Plugin):
    depends_on = ["acc_s1_recon_eff", "acc_cut"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, acc_s1_recon_eff, acc_cut):
        return key, acc_s1_recon_eff * acc_cut
