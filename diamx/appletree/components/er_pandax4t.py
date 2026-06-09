"""PandaX-4T electronic-recoil appletree components (``ComponentSim``).

Mirrors ``diamx.appletree.components.er_xenonnt`` (and the NR split in
``nr_pandax4t``): all ER-related PandaX components live here, in two groups.

1. **Yield-only** (:class:`ERYieldPandaX4T`): a uniform energy spectrum straight
   into the P4-NEST ER quanta plugins (``diamx.appletree.plugins.p4_nest``), with
   no detector / reconstruction stage, used to validate the model against Fig. 17
   of PRD 110 023029 (``notebooks/pandax4t_yields.ipynb``)::

       comp = ERYieldPandaX4T("er")
       comp.set_config({"lower_energy": 1.0, "upper_energy": 25.0, "xi_norm_er": 30.0})
       comp.deduce(data_names=["energy", "num_photon", "num_electron", "recomb_std"],
                   force_no_eff=True)
       comp.compile()
       key, (energy, num_photon, num_electron, recomb_std) = comp.simulate(
           key, batch_size, parameters)

   The ER mean yields come from the NESTv2.0 source (``ERYieldParamsP4NEST``), not
   PRD Appendix A1 (which has misprints); the recombination fluctuation and the
   Eq. 17 correction follow the PRD form.

2. **Full signal response** (:class:`ERFlatPandaX4T`, :class:`ERMonoPandaX4T`,
   :class:`ERBkgPandaX4T`): the complete chain from an energy spectrum to the
   observables ``(cs1, cs2, eff)`` used to build templates (flat, mono-energetic,
   or tabulated ER spectra respectively). See ``nr_pandax4t`` for the chain and
   conventions; the detector / reconstruction / selection block is shared with
   the NR components and is duplicated in :class:`nr_pandax4t._NRChainPandaX4T` --
   keep the two in sync.
"""

import appletree as apt
from appletree.component import ComponentSim

from diamx.appletree.plugins import p4_nest, pandax_reconstruction


def _register_er_yields(component):
    """Register the P4-NEST ER quanta chain: NESTv2.0 ER yields -> Eq. 1/2
    sampling + Eq. 17 correction (the total-quanta / ionization / photon-electron
    plugins are recoil-agnostic)."""
    component.register(p4_nest.ERYieldParamsP4NEST)
    component.register(p4_nest.TotalQuantaP4NEST)
    component.register(p4_nest.IonizationP4NEST)
    component.register(p4_nest.ERRecombParamsP4NEST)
    component.register(p4_nest.PhotonElectronP4NEST)


class ERYieldPandaX4T(ComponentSim):
    """ER quanta MC: uniform energy spectrum -> P4-NEST ER yields (no detector)."""

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        _register_er_yields(self)


class _ERChainPandaX4T(ComponentSim):
    """Shared ER full-response base: detector + reconstruction + selection.

    The reconstruction / selection block is identical to
    :class:`nr_pandax4t._NRChainPandaX4T` (mirrors the nr/er split in
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


class ERFlatPandaX4T(_ERChainPandaX4T):
    """Flat low-energy ER background (e.g. merged "Other ER") response."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        _register_er_yields(self)
        self._register_reconstruction()


class ERMonoPandaX4T(_ERChainPandaX4T):
    """Mono-energetic ER background (e.g. 124Xe double-EC at 10 keV)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        _register_er_yields(self)
        self._register_reconstruction()


class ERBkgPandaX4T(_ERChainPandaX4T):
    """Tabulated-spectrum ER background (e.g. tritium beta)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.FixedEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        _register_er_yields(self)
        self._register_reconstruction()
