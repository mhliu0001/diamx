"""Self-contained P4-NEST nuclear-recoil yield model for PandaX-4T.

This module reproduces the NESTv2 NR scintillation/ionization model (PandaX-4T
signal-response paper, Phys. Rev. D 110, 023029 (2024), Appendix A Eq. A2) and
adds the two PandaX "P4-NEST" recombination modifications (Eq. 17):

  <r>(xi) = <r>_0(xi) + P3(xi/xi_norm; p0,p1,p2,p3) * exp(-xi/xi_norm)  [+ d_nr]
  dr(xi)  = dr_0(xi) * A^NR

with xi the deposited NR energy [keV], xi_norm = 30 keV (NR), P3 a 3rd-order
Legendre polynomial, and d_nr a per-run recombination shift (0 for Run0).

The chain is implemented directly here (rather than subclassing
``appletree.plugins.nestv2``) so the PandaX model is decoupled from appletree's
internal nestv2 refactors -- it depends only on appletree's stable ``randgen``
primitives. The NESTv2 NR formulae below match appletree 0.5.5
``plugins/nestv2.py`` (which itself is ported from NESTCollaboration/nest
v2.3.x, NEST.cpp); the recombination fluctuation is drawn as a plain Gaussian,
i.e. the NESTv2 skew-normal with the skewness alpha2 set to 0 (PRD Sec. II:
"the recombination fraction is sampled from a Gaussian distribution").

The drift field is taken as a scalar ``field`` parameter (uniform field;
position-dependent corrections are disabled in diamx, as for the other
experiments), following the convention of ``nr_nest_v1``.
"""

from functools import partial

from jax import jit
from jax import numpy as jnp

import appletree
from appletree import randgen
from appletree.config import Constant
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


def _legendre_p3(u, p0, p1, p2, p3):
    """3rd-order Legendre expansion evaluated at ``u`` (PRD Eq. 17 basis)."""
    l0 = 1.0
    l1 = u
    l2 = (3.0 * u**2 - 1.0) / 2.0
    l3 = (5.0 * u**3 - 3.0 * u) / 2.0
    return p0 * l0 + p1 * l1 + p2 * l2 + p3 * l3


@export
class TotalQuantaNRP4NEST(Plugin):
    """Mean total quanta for NR: N_q = alpha * E^beta (NESTv2)."""

    depends_on = ["energy"]
    provides = ["_Nq"]
    parameters = ("alpha", "beta")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        _Nq = parameters["alpha"] * energy ** parameters["beta"]
        return key, _Nq


@export
class ThomasImelNRP4NEST(Plugin):
    """Thomas-Imel box parameter (NESTv2 Eq. A2 ``varsigma``).

    TI = gamma * field^delta * (rho / 2.9)^0.3, with ``field`` a scalar drift
    field [V/cm] passed as a parameter (uniform field; corrections disabled).
    """

    depends_on = ["energy"]
    provides = ["ThomasImel"]
    parameters = ("gamma", "delta", "liquid_xe_density", "field")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        ThomasImel = jnp.ones(shape=jnp.shape(energy))
        ThomasImel *= parameters["gamma"] * parameters["field"] ** parameters["delta"]
        ThomasImel *= (parameters["liquid_xe_density"] / 2.9) ** 0.3
        return key, ThomasImel


@export
class ChargeYieldNRP4NEST(Plugin):
    """NR charge yield Qy (NESTv2)."""

    depends_on = ["energy", "ThomasImel"]
    provides = ["charge_yield"]
    parameters = ("epsilon", "zeta", "eta")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, ThomasImel):
        charge_yield = 1 / ThomasImel / jnp.sqrt(energy + parameters["epsilon"])
        charge_yield *= 1 - 1 / (1 + (energy / parameters["zeta"]) ** parameters["eta"])
        charge_yield = jnp.clip(charge_yield, 0, jnp.inf)
        return key, charge_yield


@export
class LightYieldNRP4NEST(Plugin):
    """NR light yield Ly = N_q/E - Qy, with low-energy suppression (NESTv2)."""

    depends_on = ["energy", "_Nq", "charge_yield"]
    provides = ["light_yield"]
    parameters = ("theta", "iota")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, _Nq, charge_yield):
        light_yield = _Nq / energy - charge_yield
        light_yield *= 1 - 1 / (1 + (energy / parameters["theta"]) ** parameters["iota"])
        light_yield = jnp.clip(light_yield, 0, jnp.inf)
        return key, light_yield


@export
class MeanNphNeNRP4NEST(Plugin):
    """Mean photon / electron numbers; zeroed below NEST's validity threshold."""

    depends_on = ["light_yield", "charge_yield", "energy"]
    provides = ["_Nph", "_Ne"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, light_yield, charge_yield, energy):
        _Nph = light_yield * energy
        _Ne = charge_yield * energy
        # NEST YieldResultValidity: both vanish when the total mean yield < 1.
        mask = (_Nph + _Ne) >= 1.0
        return key, jnp.where(mask, _Nph, 0.0), jnp.where(mask, _Ne, 0.0)


@export
@appletree.takes_config(
    Constant(
        name="xi_norm_nr",
        type=float,
        default=30.0,
        help="NR recombination-correction normalization energy [keV] (PRD Eq. 17)",
    ),
)
class MeanExcitonIonNRP4NEST(Plugin):
    """Mean exciton/ion split and mean recombination fraction with the P4-NEST
    correction added to the NESTv2 baseline <r>_0 (PRD Eq. 17).

    Baseline (NESTv2): <r>_0 = 1 - (nex_ni_ratio + 1) * elecFrac = 1 - <Ne>/<Ni>.
    P4-NEST: <r> = clip(<r>_0 + P3(xi/xi_norm)*exp(-xi/xi_norm) + d_nr, 0, 1).
    """

    depends_on = ["ThomasImel", "_Nph", "_Ne", "energy"]
    provides = ["_Nex", "_Ni", "nex_ni_ratio", "alf", "elecFrac", "recombProb"]
    parameters = ("p0_nr", "p1_nr", "p2_nr", "p3_nr", "d_nr")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, ThomasImel, _Nph, _Ne, energy):
        _Nex = (-1.0 / ThomasImel) * (
            4.0 * jnp.exp(_Ne * ThomasImel / 4.0) - (_Ne + _Nph) * ThomasImel - 4.0
        )
        _Ni = (4.0 / ThomasImel) * (jnp.exp(_Ne * ThomasImel / 4.0) - 1.0)
        nex_ni_ratio = jnp.where(_Ni > 0, _Nex / _Ni, 1.0)
        _Nq = _Nph + _Ne
        alf = 1.0 / (1.0 + nex_ni_ratio)
        elecFrac = jnp.where(_Nq > 0, _Ne / _Nq, 0.0)
        recomb_baseline = 1.0 - (nex_ni_ratio + 1.0) * elecFrac

        # P4-NEST mean-recombination correction (PRD Eq. 17).
        u = energy / self.xi_norm_nr.value
        correction = _legendre_p3(
            u,
            parameters["p0_nr"],
            parameters["p1_nr"],
            parameters["p2_nr"],
            parameters["p3_nr"],
        ) * jnp.exp(-energy / self.xi_norm_nr.value)
        recombProb = jnp.clip(
            recomb_baseline + correction + parameters["d_nr"], 0.0, 1.0
        )
        return key, _Nex, _Ni, nex_ni_ratio, alf, elecFrac, recombProb


@export
class TrueExcitonIonNRP4NEST(Plugin):
    """Sample integer ion / exciton counts with NEST Fano-like fluctuations."""

    depends_on = ["_Nph", "_Ne", "nex_ni_ratio", "alf"]
    provides = ["Ni", "Nex", "Nq"]
    parameters = ("fano_ni", "fano_nex")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, _Nph, _Ne, nex_ni_ratio, alf):
        Nq_mean = _Nph + _Ne
        key, Ni = randgen.truncate_normal(
            key, Nq_mean * alf, jnp.sqrt(parameters["fano_ni"] * Nq_mean * alf), vmin=0
        )
        Ni = Ni.round().astype(int)
        key, Nex = randgen.truncate_normal(
            key,
            Nq_mean * nex_ni_ratio * alf,
            jnp.sqrt(parameters["fano_nex"] * Nq_mean * nex_ni_ratio * alf),
            vmin=0,
        )
        Nex = Nex.round().astype(int)
        Nq = Nex + Ni
        return key, Ni, Nex, Nq


@export
class RecombFluctNRP4NEST(Plugin):
    """Recombination-fluctuation variance with the P4-NEST scaling dr = dr_0 * A^NR.

    ``omega`` is the NESTv2 baseline recombination width dr_0; it is multiplied
    by ``a_nr`` (= A^NR, PRD Eq. 17) before forming the variance of N_e:
        Var(Ne) = recombProb*(1-recombProb)*Ni + (a_nr*omega)^2 * Ni^2
    (binomial term + recombination-fluctuation term).
    """

    depends_on = ["elecFrac", "recombProb", "Ni"]
    provides = ["omega", "Variance"]
    parameters = ("A", "xi", "omega", "a_nr")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, elecFrac, recombProb, Ni):
        omega = parameters["A"] * jnp.exp(
            -0.5 * (elecFrac - parameters["xi"]) ** 2.0 / (parameters["omega"] ** 2)
        )
        omega = omega * parameters["a_nr"]
        Variance = recombProb * (1.0 - recombProb) * Ni + omega * omega * Ni * Ni
        return key, omega, Variance


@export
class TruePhotonElectronNRP4NEST(Plugin):
    """Split into photons / electrons with a Gaussian recombination fluctuation.

    P4-NEST samples N_e from a Gaussian with mean (1-<r>)*Ni and the variance
    from ``RecombFluctNRP4NEST`` -- this is the NESTv2 skew-normal draw with the
    skewness alpha2 = 0 (verified: alpha2=0 gives widthCorrection=1,
    muCorrection=0). N_e is clipped to [0, Ni]; N_ph keeps at least the excitons.
    """

    depends_on = ["recombProb", "Variance", "Ni", "Nex", "Nq"]
    provides = ["num_photon", "num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recombProb, Variance, Ni, Nex, Nq):
        key, num_electron = randgen.normal(
            key, (1.0 - recombProb) * Ni, jnp.sqrt(Variance)
        )
        num_electron = jnp.clip(num_electron.round().astype(int), 0, Ni)
        num_photon = jnp.maximum(Nq - num_electron, Nex)
        return key, num_photon, num_electron
