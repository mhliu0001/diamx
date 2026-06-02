"""PandaX-4T reconstruction-chain additions on top of appletree's detector /
reconstruction / efficiency plugins.

Only two pieces are PandaX-specific; everything else in the chain (S1/S2
reconstruction bias and smearing, S1/S2 cut acceptance) is handled by the stock
appletree plugins (``appletree.plugins.reconstruction.S1`` / ``.S2`` and
``appletree.plugins.efficiency.S1CutAccept`` / ``.S2CutAccept``) configured with
PandaX maps.

1. ``PhotonDetectionPandaX4T`` -- overrides ``appletree.plugins.detector``'s
   ``PhotonDetection`` to add the **hit-clustering loss** (PRD 110 023029 Eq. 9):
   after detecting ``N_det = B(num_photon, g1_eff)`` hits, a hit is lost during
   clustering with energy/hit-count dependent probability ``eps_hit(N_det)``
   (Fig. 6, left), so the surviving detected photons are

       N'_det = B(N_det, 1 - eps_hit(N_det)).

   Registered *after* ``appletree.plugins.detector`` so it overrides the stock
   ``PhotonDetection`` (same ``provides = num_s1_phd``); the rest of the S1 chain
   (``S1PE`` -> ``S1`` bias -> ``cS1``) is unchanged.

2. ``EffPandaX4T`` -- the per-event acceptance weight is the product of the S1
   and S2 cut acceptances only, ``eff = cut_acc_s1 * cut_acc_s2`` (PandaX's
   selection efficiency, PRD Fig. 12, enters through appletree's ``S1CutAccept`` /
   ``S2CutAccept``). Overrides ``appletree.plugins.efficiency.Eff`` (whose default
   also multiplies an S2 threshold and a 3-fold S1 recon efficiency that PandaX
   folds into its measured selection curves).
"""

from functools import partial

from jax import jit
from jax import numpy as jnp

import appletree
from appletree import randgen
from appletree.config import takes_config, Map
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@takes_config(
    Map(
        name="hit_eff",
        method="LERP",
        default="pandax4t_run0_hit_eff.json",
        help="Hit-clustering loss probability eps_hit as a function of the number "
        "of detected hits N_det (PRD 110 023029 Eq. 9, Fig. 6 left).",
    ),
)
class PhotonDetectionPandaX4T(Plugin):
    """S1 photon detection + hit-clustering loss (overrides PhotonDetection).

    N_det = B(num_photon, g1 * s1_lce / (1 + p_dpe))         (stock detection)
    N'_det = B(N_det, 1 - eps_hit(N_det))                    (PandaX hit clustering)
    """

    depends_on = ["num_photon", "s1_lce"]
    provides = ["num_s1_phd"]
    parameters = ("g1", "p_dpe")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_photon, s1_lce):
        g1_true_no_dpe = jnp.clip(parameters["g1"] * s1_lce / (1.0 + parameters["p_dpe"]), 0, 1.0)
        key, n_det = randgen.binomial(key, g1_true_no_dpe, num_photon)

        # hit-clustering loss: each detected hit survives with prob 1 - eps_hit(N_det)
        eps_hit = jnp.clip(self.hit_eff.apply(n_det), 0.0, 1.0)
        key, num_s1_phd = randgen.binomial(key, 1.0 - eps_hit, n_det)
        return key, num_s1_phd


@export
class EffPandaX4T(Plugin):
    """Per-event selection acceptance weight: eff = cut_acc_s1 * cut_acc_s2.

    Overrides appletree's ``Eff``; PandaX's measured S1/S2 cut-acceptance curves
    (Fig. 12, via ``S1CutAccept`` / ``S2CutAccept``) already fold in reconstruction
    and threshold efficiency, so no separate S2-threshold / 3-fold-S1 term.
    """

    depends_on = ["cut_acc_s1", "cut_acc_s2"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, cut_acc_s1, cut_acc_s2):
        return key, cut_acc_s1 * cut_acc_s2
