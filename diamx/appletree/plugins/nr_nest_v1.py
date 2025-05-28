from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import randgen
from appletree.config import Constant
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@appletree.takes_config(
    Constant(name="Z", type=int, default=54, help="Atomic number of xenon")
)
class LindhardQuanta(Plugin):
    depends_on = ["num_quanta", "energy"]
    provides = ["epsilon", "num_quanta_obs"]
    parameters = ("kappa",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta, energy):
        epsilon = 11.5 * self.Z.value ** (-7 / 3) * energy
        g_epsilon = 3 * epsilon**0.15 + 0.7 * epsilon**0.6 + epsilon
        lindhard_factor = (
            parameters["kappa"] * g_epsilon / (1 + parameters["kappa"] * g_epsilon)
        )
        key, num_quanta_obs = randgen.binomial(key, lindhard_factor, num_quanta)
        return key, epsilon, num_quanta_obs


@export
class NRExcitonsIons(Plugin):
    depends_on = ["num_quanta_obs"]
    provides = ["num_exciton", "num_ion"]
    parameters = ("alpha", "beta", "zeta", "field")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta_obs):
        nex_ni_ratio = (
            parameters["alpha"]
            * parameters["field"] ** (-parameters["zeta"])
            * (1 - jnp.exp(-parameters["beta"] * num_quanta_obs))
        )
        key, num_ion = randgen.binomial(key, 1 / (1 + nex_ni_ratio), num_quanta_obs)
        num_exciton = num_quanta_obs - num_ion
        return key, num_exciton, num_ion


@export
class NRRecombination(Plugin):
    provides = ["num_photon_re"]
    depends_on = ["num_ion"]
    parameters = ("gamma", "delta", "field")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_ion):
        xi = parameters["gamma"] * parameters["field"] ** (-parameters["delta"])
        p_recomb = 1 - jnp.log(1 + num_ion * xi) / (num_ion * xi)
        key, num_photons_re = randgen.binomial(key, p_recomb, num_ion)
        return key, num_photons_re


@export
class NRDeexicitation(Plugin):
    provides = ["num_photon_de"]
    depends_on = ["epsilon", "num_exciton"]
    parameters = ("eta", "lambda")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, epsilon, num_exciton):
        f_l = 1 / (1 + parameters["eta"] * epsilon ** parameters["lambda"])
        key, num_photon_de = randgen.binomial(key, f_l, num_exciton)
        return key, num_photon_de


@export
class NRQuantaFinal(Plugin):
    provides = ["num_photon", "num_electron"]
    depends_on = ["num_photon_re", "num_photon_de", "num_ion"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_photon_re, num_photon_de, num_ion):
        num_photon = num_photon_re + num_photon_de
        num_electron = num_ion - num_photon_re
        return key, num_photon, num_electron
