"""PandaX-4T nuclear-recoil appletree components (``ComponentSim``).

Mirrors ``diamx.appletree.components.nr_xenonnt``: all NR-related PandaX
components live here, in two groups.

1. **Yield-only** (:class:`NRYieldPandaX4T`): a uniform energy spectrum straight
   into the P4-NEST NR quanta plugins (``diamx.appletree.plugins.p4_nest``), with
   no detector / reconstruction stage. It predicts the mean light yield, charge
   yield and recombination fluctuation vs energy directly from the simulation, to
   validate the model against Fig. 17 of PRD 110 023029
   (``notebooks/pandax4t_yields.ipynb``)::

       comp = NRYieldPandaX4T("nr")
       comp.set_config({"lower_energy": 1.0, "upper_energy": 90.0, "xi_norm_nr": 150.0})
       comp.deduce(data_names=["energy", "num_photon", "num_electron", "recomb_std"],
                   force_no_eff=True)
       comp.compile()
       key, (energy, num_photon, num_electron, recomb_std) = comp.simulate(
           key, batch_size, parameters)

2. **Full signal response** (:class:`NRPandaX4T`, :class:`WIMPPandaX4T`,
   :class:`NeutronPandaX4T`): the complete chain from an energy spectrum to the
   observables ``(cs1, cs2, eff)`` used to build templates -- P4-NEST microphysics
   -> detector response (g1 + hit clustering Eq. 9, S1PE, drift loss, g2_b S2PE)
   -> reconstruction (S1/S2 bias + smearing, cS1/cS2) -> selection efficiency
   (S1/S2 cut acceptance x quality residual x reconstruction efficiency).

Conventions (mirrors ``nr_xenonnt`` / ``er_xenonnt``): ``norm_type = "on_pdf"``,
``add_eps_to_hist = False``; spatial corrections are disabled via the instruct
(``s1_lce == s1_correction``, ``s2_lce == s2_correction``) so uniform g1 / g2_b
carry the response; ``cs2`` is the bottom-array S2 (detector ``g2`` = PandaX
``g2_b``), and the analysis axis log10(cs2/cs1) is formed downstream. The
detector / reconstruction / selection block is shared with the ER components and
is duplicated in :class:`er_pandax4t._ERChainPandaX4T` -- keep the two in sync.
"""

import appletree as apt
from appletree.component import ComponentSim

from diamx.appletree.plugins import p4_nest, pandax_reconstruction


def _register_nr_yields(component):
    """Register the P4-NEST NR quanta chain (PRD Eq. 1/2 + Eq. 17, Appendix A2):
    energy -> yield params (L, ion fraction, <r>_0, dr_0) -> N_q = B(xi/W, L)
    -> N_i = B(N_q, 1/(1+alpha)) -> corrected <r>, dr -> N_e = B(N_i, 1-r),
    N_ph = N_q - N_e."""
    component.register(p4_nest.NRYieldParamsP4NEST)
    component.register(p4_nest.TotalQuantaP4NEST)
    component.register(p4_nest.IonizationP4NEST)
    component.register(p4_nest.NRRecombParamsP4NEST)
    component.register(p4_nest.PhotonElectronP4NEST)


class NRYieldPandaX4T(ComponentSim):
    """NR quanta MC: uniform energy spectrum -> P4-NEST yields (no detector)."""

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        _register_nr_yields(self)


class _NRChainPandaX4T(ComponentSim):
    """Shared NR full-response base: detector + reconstruction + selection.

    The reconstruction / selection block is identical to
    :class:`er_pandax4t._ERChainPandaX4T` (mirrors the nr/er split in
    ``nr_xenonnt`` / ``er_xenonnt``); keep the two in sync.
    """

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def _register_reconstruction(self):
        # detector response: photon detection, DPE, electron drift, S2 gas gain
        self.register_all(apt.plugins.detector)
        # reconstruction: S1/S2 bias + smearing, spatial corrections, cS1/cS2
        self.register_all(apt.plugins.reconstruction)
        # PandaX hit-clustering loss -- overrides the stock PhotonDetection (Eq. 9)
        self.register(pandax_reconstruction.PhotonDetectionPandaX4T)
        # efficiency (PRD Eq. 16): S1/S2 cut acceptance x residual r(xi) x recon
        # efficiency; single-scatter = 1; ROI applied by the binning.
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(pandax_reconstruction.QualityResidualPandaX4T)
        self.register(pandax_reconstruction.ReconEffPandaX4T)
        self.register(pandax_reconstruction.EffPandaX4T)


class NRPandaX4T(_NRChainPandaX4T):
    """Nuclear-recoil signal response over a uniform energy spectrum.

    Used both for cut-acceptance / efficiency studies and as the base for the
    WIMP component (which swaps in a fixed signal spectrum).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        _register_nr_yields(self)
        self._register_reconstruction()


class WIMPPandaX4T(NRPandaX4T):
    """WIMP NR signal -- a fixed (tabulated) recoil spectrum instead of uniform."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # overrides UniformEnergySpectra (both provide ``energy``)
        self.register(apt.plugins.common.FixedEnergySpectra)


class NeutronPandaX4T(WIMPPandaX4T):
    """Radiogenic-neutron NR background -- same chain, neutron recoil spectrum."""
