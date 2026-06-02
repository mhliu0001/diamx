"""PandaX-4T full signal-response components (appletree ``ComponentSim``).

These assemble the complete chain from an energy spectrum to the observables
``(cs1, cs2, eff)`` used to build templates:

    energy spectrum
      -> P4-NEST microphysics (num_photon, num_electron)        [plugins.p4_nest]
      -> detector response                                       [apt.plugins.detector]
           PhotonDetectionPandaX4T (g1 + hit clustering, Eq. 9)  [plugins.pandax_reconstruction]
           S1PE, DriftLoss, ElectronDrifted, S2PE (g2_b)
      -> reconstruction bias / smearing + spatial corrections    [apt.plugins.reconstruction]
           S1 (s1_bias/s1_smear), S2 (s2_bias/s2_smear), cS1, cS2
      -> selection efficiency                                    [apt.plugins.efficiency + EffPandaX4T]
           S1CutAccept (s1_cut_acc), S2CutAccept (s2_cut_acc), eff = cut_acc_s1 * cut_acc_s2

Conventions (mirrors ``diamx.appletree.components.{nr,er}_xenonnt``):
  * ``norm_type = "on_pdf"``, ``add_eps_to_hist = False``.
  * Spatial corrections are *disabled* by setting ``s1_lce == s1_correction`` and
    ``s2_lce == s2_correction`` in the instruct file, so uniform ``g1`` / ``g2_b``
    carry the response (the position dependence cancels in cS1 / cS2).
  * ``cs2`` is the **bottom** S2 when the detector ``g2`` parameter is set to
    PandaX's ``g2_b``; the analysis-space axis log10(cs2/cs1) is formed downstream.

S1/S2 reconstruction bias come from the stock appletree ``S1`` / ``S2`` plugins
(PandaX bias/smear maps), per design. The only PandaX-specific plugins are the
hit-clustering photon detection and the ``eff = cut_acc_s1 * cut_acc_s2`` combiner.
"""

import appletree as apt
from appletree.component import ComponentSim

from diamx.appletree.plugins import p4_nest, pandax_reconstruction


class _PandaX4TChain(ComponentSim):
    """Shared detector + reconstruction + selection chain (no yields / spectrum).

    Subclasses register an energy-spectrum plugin and the recoil-specific
    P4-NEST yield plugins (which must provide ``num_photon`` and ``num_electron``),
    then call :meth:`_register_reconstruction`.
    """

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def _register_reconstruction(self):
        # detector response: photon detection, DPE, electron drift, S2 gas gain
        self.register_all(apt.plugins.detector)
        # reconstruction: S1/S2 bias + smearing, spatial corrections, cS1/cS2
        self.register_all(apt.plugins.reconstruction)
        # PandaX hit-clustering loss -- overrides the stock PhotonDetection
        self.register(pandax_reconstruction.PhotonDetectionPandaX4T)
        # selection efficiency: S1 and S2 cut acceptance -> eff
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(pandax_reconstruction.EffPandaX4T)

    def _register_nr_yields(self):
        self.register(p4_nest.NRYieldParamsP4NEST)
        self.register(p4_nest.TotalQuantaP4NEST)
        self.register(p4_nest.IonizationP4NEST)
        self.register(p4_nest.NRRecombParamsP4NEST)
        self.register(p4_nest.PhotonElectronP4NEST)

    def _register_er_yields(self):
        self.register(p4_nest.ERYieldParamsP4NEST)
        self.register(p4_nest.TotalQuantaP4NEST)
        self.register(p4_nest.IonizationP4NEST)
        self.register(p4_nest.ERRecombParamsP4NEST)
        self.register(p4_nest.PhotonElectronP4NEST)


class NRPandaX4T(_PandaX4TChain):
    """Nuclear-recoil signal response over a uniform energy spectrum.

    Used both for cut-acceptance / efficiency studies and as the base for the
    WIMP component (which swaps in a fixed signal spectrum).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self._register_nr_yields()
        self._register_reconstruction()


class WIMPPandaX4T(NRPandaX4T):
    """WIMP NR signal -- a fixed (tabulated) recoil spectrum instead of uniform."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # overrides UniformEnergySpectra (both provide ``energy``)
        self.register(apt.plugins.common.FixedEnergySpectra)


class NeutronPandaX4T(WIMPPandaX4T):
    """Radiogenic-neutron NR background -- same chain, neutron recoil spectrum."""


class ERFlatPandaX4T(_PandaX4TChain):
    """Flat low-energy ER background (e.g. merged "Other ER") response."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self._register_er_yields()
        self._register_reconstruction()


class ERMonoPandaX4T(_PandaX4TChain):
    """Mono-energetic ER background (e.g. 124Xe double-EC at 10 keV)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self._register_er_yields()
        self._register_reconstruction()


class ERBkgPandaX4T(_PandaX4TChain):
    """Tabulated-spectrum ER background (e.g. tritium beta)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.FixedEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self._register_er_yields()
        self._register_reconstruction()
