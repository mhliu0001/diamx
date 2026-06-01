"""Self-contained P4-NEST nuclear-recoil yield model for PandaX-4T.

Faithful to the PandaX-4T signal-response paper (Phys. Rev. D 110, 023029
(2024)): the quanta are sampled with the *unified* ER/NR scheme of Eqs. (1)-(2),

    N_q  = B(xi / W, L)                      (Eq. 1; L = Lindhard factor)
    N_i  = B(N_q, 1 / (1 + alpha))           (Eq. 2; alpha = <N_ex>/<N_i>)
    r    = G(<r>, dr)                         (Eq. 2 / Sec. II; Gaussian)
    N_e  = B(N_i, 1 - r)                      (Eq. 2)
    N_ph = N_q - N_e                          (Eq. 2)

and the NR Lindhard factor L, exciton-to-ion ratio alpha, baseline mean
recombination <r>_0 and baseline recombination fluctuation dr_0 are taken from
Appendix A Eq. (A2):

    varsigma = 0.0480 * E^(-0.0533) * (rho_Xe / 2.90)^0.30           (E = drift field)
    <N_e>  = xi * (1 - 1/(1 + (xi/0.3)^2)) / (varsigma * sqrt(xi + 12.6))
    <N_ph> = (11.0 * xi^1.1 - <N_e>) * (1 - 1/(1 + (xi/0.3)^2))
    <N_i>  = 4 * (exp(<N_e> * varsigma / 4) - 1) / varsigma
    L      = (<N_ph> + <N_e>) * W / xi
    alpha  = (<N_ph> + <N_e>) / <N_i> - 1
    <r>_0  = 1 - <N_e> / <N_i>
    dr_0   = 0.1 * exp(-(zeta - 0.5)^2 / 0.0722),  zeta = <N_e>/(<N_e> + <N_ph>)

The baseline mean yields and dr_0 reproduce PRD Fig. 17 (NR): N_ph/xi ~ 5->15
photon/keV, N_e/xi ~ 7->3 e/keV over 3-70 keV, and dr peaking ~0.1 at ~5 keV.
``zeta`` is taken as the quenched electron fraction <N_e>/(<N_e>+<N_ph>) (~0.5),
which reproduces the Fig. 17 dr scale. Eq. A2 literally writes
zeta = <N_e>*W/(1000*xi) (~0.04-0.09 for NR), but that gives dr ~0.01, ~10x
below Fig. 17, so it appears to be a misprint in the paper; we follow the figure.

The P4-NEST modifications (Eq. 17) are applied to the recombination only:

    <r>  = clip(<r>_0 + P3(xi/xi_norm; p0,p1,p2,p3) * exp(-xi/xi_norm) + d_nr, 0, 1)
    dr   = dr_0 * A^NR

with xi_norm = 150 keV (NR), P3 a 3rd-order Legendre polynomial, d_nr a per-run
shift (0 for Run0). The recombination fraction r is then drawn from a plain
Gaussian (truncated to [0, 1]), i.e. the NESTv2 skew-normal with skewness 0.

NOTE (xi_norm^NR): Eq. 17 lists xi_norm^NR = 30 keV and xi_norm^ER = 150 keV, but
with 30 keV the standard Legendre argument xi/30 exceeds 1 above 30 keV and the
degree-3 term overshoots exp(-xi/30), driving <r> -> 0 by ~50 keV -- whereas
PRD Fig. 17 shows the P4-NEST correction is small over the whole 1-90 keV NR
range. Using 150 keV (the value the paper assigns to ER) keeps xi/xi_norm <= 0.6
over the NR range, so the correction stays small everywhere and matches Fig. 17.
We therefore use 150 keV for NR (the two xi_norm values appear to be swapped in
Eq. 17; this is also physically sensible since the larger NR energy range needs
the larger normalization). Tried and ruled out: shifted / domain-mapped Legendre
conventions, which all break the low-energy correction. Standard Legendre with
the raw argument is correct; only xi_norm needed fixing.

The drift field and work function are scalar parameters (uniform field;
position-dependent corrections are disabled in diamx, as for the other
experiments).
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
class NRYieldParamsP4NEST(Plugin):
    """NESTv2 NR mean yields (PRD Eq. A2) -> Lindhard L, ion fraction, baseline
    mean recombination and baseline recombination fluctuation.

    All Eq. A2 numeric constants are fixed (NEST nominal; PandaX tunes only the
    recombination correction on top, Eq. 17), so only the drift field, density
    and work function are exposed as parameters.
    """

    depends_on = ["energy"]
    provides = ["lindhard", "ion_fraction", "recomb_mean0", "recomb_std0"]
    parameters = ("w", "field", "liquid_xe_density")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        varsigma = (
            0.0480
            * parameters["field"] ** (-0.0533)
            * (parameters["liquid_xe_density"] / 2.90) ** 0.30
        )
        suppression = 1.0 - 1.0 / (1.0 + (energy / 0.3) ** 2)
        mean_ne = energy * suppression / (varsigma * jnp.sqrt(energy + 12.6))
        mean_nph = (11.0 * energy**1.1 - mean_ne) * suppression
        mean_nq = mean_nph + mean_ne
        mean_ni = 4.0 * (jnp.exp(mean_ne * varsigma / 4.0) - 1.0) / varsigma

        # Lindhard factor L so that <N_q> = (xi/W) * L = <N_ph> + <N_e>.
        lindhard = jnp.where(energy > 0, mean_nq * parameters["w"] / energy, 0.0)
        lindhard = jnp.clip(lindhard, 0.0, 1.0)
        # Ion fraction 1/(1+alpha) = <N_i>/<N_q>.
        ion_fraction = jnp.where(mean_nq > 0, mean_ni / mean_nq, 0.0)
        ion_fraction = jnp.clip(ion_fraction, 0.0, 1.0)
        # Baseline mean recombination <r>_0 = 1 - <N_e>/<N_i>.
        recomb_mean0 = jnp.where(mean_ni > 0, 1.0 - mean_ne / mean_ni, 0.0)
        # Baseline recombination fluctuation dr_0 (PRD Eq. A2). The Gaussian
        # argument is taken as the quenched electron fraction
        # zeta = <N_e>/(<N_e>+<N_ph>) (~0.5 near the peak), which reproduces the
        # Fig. 17 NR dr peak of ~0.1. Eq. A2 literally writes the argument as
        # <N_e>*W/(1000*xi) (~0.04-0.09 for NR), but that yields dr ~0.01, ~10x
        # below Fig. 17 -- so the literal form appears to be a misprint and we
        # follow the figure instead.
        zeta = jnp.where(mean_nq > 0, mean_ne / mean_nq, 0.0)
        recomb_std0 = 0.1 * jnp.exp(-((zeta - 0.5) ** 2) / 0.0722)
        return key, lindhard, ion_fraction, recomb_mean0, recomb_std0


@export
class NRTotalQuantaP4NEST(Plugin):
    """Total quanta N_q = B(xi / W, L) (PRD Eq. 1)."""

    depends_on = ["energy", "lindhard"]
    provides = ["num_quanta"]
    parameters = ("w",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, lindhard):
        n_trials = jnp.clip(energy / parameters["w"], 0.0, jnp.inf)
        n_trials = n_trials.round().astype(int)
        key, num_quanta = randgen.binomial(key, lindhard, n_trials)
        return key, num_quanta


@export
class NRIonizationP4NEST(Plugin):
    """Exciton/ion split N_i = B(N_q, 1/(1+alpha)) (PRD Eq. 2)."""

    depends_on = ["num_quanta", "ion_fraction"]
    provides = ["num_ion", "num_exciton"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta, ion_fraction):
        key, num_ion = randgen.binomial(key, ion_fraction, num_quanta)
        num_exciton = num_quanta - num_ion
        return key, num_ion, num_exciton


@export
@appletree.takes_config(
    Constant(
        name="xi_norm_nr",
        type=float,
        default=150.0,
        help="NR recombination-correction normalization energy [keV] (PRD Eq. 17 "
        "lists 30, but 150 -- its ER value -- matches Fig. 17; see module docstring)",
    ),
)
class NRRecombParamsP4NEST(Plugin):
    """Corrected mean recombination <r> and fluctuation dr (PRD Eq. 17).

    <r>  = clip(<r>_0 + P3(xi/xi_norm)*exp(-xi/xi_norm) + d_nr, 0, 1)
    dr   = dr_0 * a_nr

    Provides ``recomb_mean`` and ``recomb_std`` as data names so they can be
    inspected / deduced directly (the recombination-fluctuation comparison vs
    PRD Fig. 17 reads ``recomb_std``).
    """

    depends_on = ["recomb_mean0", "recomb_std0", "energy"]
    provides = ["recomb_mean", "recomb_std"]
    parameters = ("p0_nr", "p1_nr", "p2_nr", "p3_nr", "d_nr", "a_nr")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_mean0, recomb_std0, energy):
        u = energy / self.xi_norm_nr.value
        correction = _legendre_p3(
            u,
            parameters["p0_nr"],
            parameters["p1_nr"],
            parameters["p2_nr"],
            parameters["p3_nr"],
        ) * jnp.exp(-energy / self.xi_norm_nr.value)
        recomb_mean = jnp.clip(
            recomb_mean0 + correction + parameters["d_nr"], 0.0, 1.0
        )
        recomb_std = recomb_std0 * parameters["a_nr"]
        return key, recomb_mean, recomb_std


@export
class NRPhotonElectronP4NEST(Plugin):
    """Photon/electron split with a Gaussian recombination fraction (PRD Eq. 2).

    r    ~ Gaussian(<r>, dr) truncated to [0, 1]
    N_e  = B(N_i, 1 - r);  N_ph = N_q - N_e
    """

    depends_on = ["num_quanta", "num_ion", "recomb_mean", "recomb_std"]
    provides = ["num_photon", "num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta, num_ion, recomb_mean, recomb_std):
        key, recomb = randgen.truncate_normal(
            key, recomb_mean, recomb_std, vmin=0.0, vmax=1.0
        )
        key, num_electron = randgen.binomial(key, 1.0 - recomb, num_ion)
        num_photon = num_quanta - num_electron
        return key, num_photon, num_electron
