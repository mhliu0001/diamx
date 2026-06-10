"""Self-contained P4-NEST electronic- and nuclear-recoil yield model for PandaX-4T.

Faithful to the PandaX-4T signal-response paper (Phys. Rev. D 110, 023029
(2024)): ER and NR share the *unified* quanta-sampling scheme of Eqs. (1)-(2),

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

The ER mean yields (with Lindhard L = 1) come instead from the NESTv2.0 source --
see the ER section and ``ERYieldParamsP4NEST`` below.

The P4-NEST modifications (Eq. 17) are applied to the recombination only, for both
recoils:

    <r>  = clip(<r>_0 + P3(xi/xi_norm; p0,p1,p2,p3) * exp(-xi/xi_norm) + d, 0, 1)
    dr   = dr_0 * A

P3 is a 3rd-order Legendre polynomial; ``d`` is a per-run shift (0 for Run0) and
``A`` a fluctuation scale -- the parameters are (p0..p3_nr, d_nr, a_nr) for NR and
(p0..p3_er, d_er, a_er) for ER. The recombination fraction r is then drawn from a
plain Gaussian (truncated to [0, 1]), i.e. the NESTv2 skew-normal with skewness 0.

NOTE (xi_norm): diamx uses xi_norm = 30 keV for ER and 150 keV for NR. xi_norm is
only a reparametrization -- the same <r>(xi) curve is fit at any xi_norm given
suitable coefficients -- so p0..p3 are fit to the digitized Fig. 17 yield curves
at these values (the NR coefficients are additionally anchored to the published
high-cS1 NR median; see notebooks/pandax4t_yields.ipynb). The per-recoil values
are chosen so the fit is well-conditioned: at xi_norm=30 the NR Legendre term
overshoots exp(-xi/30) and drives <r> -> 0, so NR uses 150, while ER is
well-conditioned at 30. (The paper's Eq. 17 lists xi_norm^NR=30 / xi_norm^ER=150,
but since we refit the coefficients the exact assignment is immaterial.)

The drift field is a uniform scalar -- the ``field`` parameter from the model JSON
(diamx does not use an appletree ``efield_map``) -- and the work function is the
scalar ``w``; position-dependent corrections are disabled, as for the other
experiments.
"""

from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.scipy.special import erf

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
class TotalQuantaP4NEST(Plugin):
    """Total quanta N_q = B(xi / W, L) (PRD Eq. 1). Recoil-agnostic (ER or NR);
    the recoil type enters only through the Lindhard factor ``lindhard`` (=1 for
    ER) supplied by the {ER,NR}YieldParamsP4NEST plugin."""

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
class IonizationP4NEST(Plugin):
    """Exciton/ion split N_i = B(N_q, 1/(1+alpha)) (PRD Eq. 2). Recoil-agnostic."""

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
        help="NR recombination-correction normalization energy [keV]. The paper's "
        "Eq. 17 lists 30, but the degree-3 Legendre term overshoots exp(-xi/30) and "
        "drives <r> -> 0, so diamx uses 150; p0..p3_nr are refit to the Fig. 17 LY/QY "
        "and the high-cS1 NR median at this xi_norm. ER uses xi_norm_er=30.",
    ),
    Constant(
        name="energy_clip_nr",
        type=float,
        default=150.0,
        help="Clip energy [keV] for the Eq. 17 NR recombination correction: above "
        "this the correction is frozen at its clip-energy value. The degree-3 "
        "Legendre fit is only calibrated up to ~100 keV; extrapolated to "
        "few-hundred-keV recoils it dives to corrections of -0.7..-1.3, clipping "
        "<r> to 0 so the events come out S1-less and S2-rich and fold back into "
        "the WIMP ROI at high log10(cS2_b/cS1). Hard recoil spectra (heavy EFT "
        "signals) are the ones affected; the clip keeps such events on a "
        "band-like trajectory that exits the ROI at high cS1 instead.",
    ),
)
class NRRecombParamsP4NEST(Plugin):
    """Corrected mean recombination <r> and fluctuation dr (PRD Eq. 17).

    <r>  = clip(<r>_0 + P3(xi_c/xi_norm)*exp(-xi_c/xi_norm) + d_nr, 0, 1)
    with xi_c = min(xi, energy_clip_nr), and dr = dr_0 * a_nr.

    Provides ``recomb_mean`` and ``recomb_std`` as data names so they can be
    inspected / deduced directly (the recombination-fluctuation comparison vs
    PRD Fig. 17 reads ``recomb_std``).
    """

    depends_on = ["recomb_mean0", "recomb_std0", "energy"]
    provides = ["recomb_mean", "recomb_std"]
    parameters = ("p0_nr", "p1_nr", "p2_nr", "p3_nr", "d_nr", "a_nr")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_mean0, recomb_std0, energy):
        energy_clipped = jnp.clip(energy, 0.0, self.energy_clip_nr.value)
        u = energy_clipped / self.xi_norm_nr.value
        correction = _legendre_p3(
            u,
            parameters["p0_nr"],
            parameters["p1_nr"],
            parameters["p2_nr"],
            parameters["p3_nr"],
        ) * jnp.exp(-energy_clipped / self.xi_norm_nr.value)
        recomb_mean = jnp.clip(
            recomb_mean0 + correction + parameters["d_nr"], 0.0, 1.0
        )
        recomb_std = recomb_std0 * parameters["a_nr"]
        return key, recomb_mean, recomb_std


@export
class PhotonElectronP4NEST(Plugin):
    """Photon/electron split with a Gaussian recombination fraction (PRD Eq. 2).
    Recoil-agnostic; consumes the corrected recomb_mean/recomb_std.

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


# ---------------------------------------------------------------------------
# Electronic-recoil (ER) yield model.
#
# IMPORTANT: the ER mean yields implement PRD 110 023029 Appendix A1 -- the PandaX
# transcription of the NESTv2.0 beta/CH3T charge yield -- cross-checked against the
# NESTv2.0 source (NESTCollaboration/nest @ v2.0.0, NEST.cpp). A1's printed Qy
# expression matches NESTv2.0 except for ONE typo: the Doke-Birks numerator, printed
# `1.145935e10`, should be 1.415935e10 (NESTv2.0) -- it shifts ER Qy by ~0.5% near
# 25 keV (<0.1% below 10 keV). Everything else in the Qy is as A1 prints it: the Y1
# exponential is the DIVISION e^(rho/0.3393) and the (Y0-Y1) energy exponent is
# xi^2.1393; the code just uses NESTv2.0's finer rounding of the constants (A1's
# 0.3393, 0.02672, 0.02673, 32.99 -> NEST's 0.33926, 0.026715, 0.02673144, 32.988).
# Two further constants the code takes from NESTv2.0 rather than A1 -- both setting
# only the ion/exciton split, not the mean N_e = xi*Qy: the exciton-ratio coefficient
# 0.039693 (A1 prints 0.093963) and <N_i> = N_q/(1+alpha) (A1 prints 1000 xi/(W alpha)).
# The result reproduces Fig. 17 (N_e/xi ~62->20, N_ph/xi ~11->53 over 1-25 keV).
#
# The recombination FLUCTUATION dr_0 is taken from the PRD A1 functional form
# (a skew-Gaussian in the quenched electron fraction), since NESTv2.0's dr is a
# parabola in recombProb that does NOT match Fig. 17 -- the same situation as
# NR, where the PRD form (not NESTv2.0's) reproduced the figure.
# ---------------------------------------------------------------------------


@export
class ERYieldParamsP4NEST(Plugin):
    """PRD Appendix A1 ER mean yields (= the NESTv2.0 beta model) -> Lindhard
    L(=1), ion fraction, baseline mean recombination <r>_0 and baseline
    recombination fluctuation dr_0.

    Implements the A1 Qy expression, with:
      * a FIXED work function W (= the `w` parameter): A1 uses a constant W
        (1000/W in Y0, zeta=<N_e>W/(1000 xi)) with no density-dependent Wq --
        unlike the NESTv2 source; ER Lindhard = 1 exactly.
      * A1's one Qy typo corrected: the Doke-Birks numerator `1.145935e10` ->
        1.415935e10 (NESTv2.0). The rest of the Qy is as A1 prints it -- the Y1
        exponential is the DIVISION rho/0.33926 and the energy exponent is xi^2.1393
        -- the code just uses NESTv2.0's slightly finer rounding of the constants.
      * <N_i> = N_q/(1+alpha) and the exciton-ratio coefficient 0.039693 follow the
        NESTv2.0 / physically-consistent form (A1 prints <N_i>=1000 xi/(W alpha) and
        0.093963); both set only the ion/exciton split and the refit-absorbed
        recombination baseline, not the mean N_e = xi*Qy.
    dr_0 uses the A1 skew-Gaussian in zeta = <N_e>/<N_q> with amplitude A(F).
    """

    depends_on = ["energy"]
    provides = ["lindhard", "ion_fraction", "recomb_mean0", "recomb_std0"]
    parameters = ("w", "field", "liquid_xe_density")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        F = parameters["field"]                  # drift field [V/cm] (A1 "E")
        rho = parameters["liquid_xe_density"]    # rho_Xe [g/cm^3]
        W_eV = parameters["w"] * 1000.0          # fixed W [eV]; A1 has no Wq(rho)

        # exciton-to-ion ratio alpha = (0.067366 + 0.039693 rho) Phi(0.05 xi)
        alpha = (0.067366 + 0.039693 * rho) * erf(0.05 * energy)

        # charge yield Qy = <N_e>/xi (A1, drift field F; the two misprints fixed)
        eta = 1.0 + 0.4607 / (1.0 + (F / 621.74) ** (-2.2717)) ** 53.502
        Y0 = 1000.0 / W_eV + 6.5 * (1.0 - 1.0 / (1.0 + (F / 47.408) ** 1.9851))
        Y1 = 32.988 * eta * (
            1.0 - 1.0 / (1.0 + (F / (0.026715 * jnp.exp(rho / 0.33926))) ** 0.6705)
        )
        tau = (
            1652.264 + (1.415935e10 - 1652.264) / (1.0 + (F / 0.02673144) ** 1.564691)
        ) * energy ** (-2.0)
        qy = (
            Y1
            + (Y0 - Y1) / (1.0 + 1.304 * energy ** 2.1393) ** 0.35535
            + 28.0 / (1.0 + tau)
        )

        # A1 mean quanta: <N_q> = 1000 xi / W, <N_e> = xi Qy, ion frac = 1/(1+alpha)
        mean_nq = energy * 1000.0 / W_eV         # total quanta (ER, L = 1)
        mean_ne = qy * energy
        elec_frac = jnp.where(mean_nq > 0, mean_ne / mean_nq, 0.0)  # zeta = <N_e>/<N_q>

        # baseline mean recombination <r>_0 = 1 - <N_e>/<N_i> = 1 - (1+alpha) zeta
        recomb_mean0 = jnp.clip(1.0 - (1.0 + alpha) * elec_frac, 0.0, 1.0)

        # baseline recombination fluctuation dr_0 (A1 skew-Gaussian in zeta; A(F))
        a_field = 0.1383 - 0.09583 / (1.0 + (F / 1210.0) ** 1.25)
        recomb_std0 = (
            a_field
            * jnp.exp(-((elec_frac - 0.5) ** 2) / 0.084)
            * (1.0 + erf(-0.6899 * (elec_frac - 0.5)))
        )

        # ER Lindhard = 1 exactly (fixed W ties <N_q> to the sampling w = W/1000)
        lindhard = jnp.clip(mean_nq * parameters["w"] / energy, 0.0, 1.0)
        ion_fraction = jnp.clip(1.0 / (1.0 + alpha), 0.0, 1.0)
        return key, lindhard, ion_fraction, recomb_mean0, recomb_std0


@export
@appletree.takes_config(
    Constant(
        name="xi_norm_er",
        type=float,
        default=30.0,
        help="ER recombination-correction normalization energy [keV]. xi_norm is a "
        "reparametrization -- any value fits the same <r>(xi) curve given suitable "
        "coefficients -- so diamx fits p0..p3 to the digitized Fig. 17 yield CURVE "
        "(not the published Table II values, which are degenerate: Table III quotes "
        "+-50-100% errors). At xi_norm=30 that fit is well-conditioned and gives small "
        "coefficients (~[0.24,-0.42,0.38,-0.10]) that reproduce LY/QY to ~0.2% with no "
        "residual structure. Anchoring to ~Table II at xi_norm=150 (which the "
        "degeneracy there allows) instead leaves a ~1% mid-energy (5-15 keV) wobble, "
        "because PandaX's Table II ER coefficients are themselves ~1% inconsistent with "
        "their own Fig. 17 curve. We match the curve (it feeds the templates). NR uses "
        "xi_norm_nr=150 (its Legendre term overshoots at 30); its p0..p3_nr are likewise "
        "refit to the Fig. 17 curve, additionally anchored to the high-cS1 NR median.",
    ),
)
class ERRecombParamsP4NEST(Plugin):
    """Corrected mean recombination <r> and fluctuation dr for ER (PRD Eq. 17).

    <r> = clip(<r>_0 + P3(xi/xi_norm_er) exp(-xi/xi_norm_er) + d_er, 0, 1)
    dr  = dr_0 * a_er
    """

    depends_on = ["recomb_mean0", "recomb_std0", "energy"]
    provides = ["recomb_mean", "recomb_std"]
    parameters = ("p0_er", "p1_er", "p2_er", "p3_er", "d_er", "a_er")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_mean0, recomb_std0, energy):
        u = energy / self.xi_norm_er.value
        correction = _legendre_p3(
            u,
            parameters["p0_er"],
            parameters["p1_er"],
            parameters["p2_er"],
            parameters["p3_er"],
        ) * jnp.exp(-energy / self.xi_norm_er.value)
        recomb_mean = jnp.clip(
            recomb_mean0 + correction + parameters["d_er"], 0.0, 1.0
        )
        recomb_std = recomb_std0 * parameters["a_er"]
        return key, recomb_mean, recomb_std
