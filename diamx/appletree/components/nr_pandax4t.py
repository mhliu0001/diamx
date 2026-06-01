"""PandaX-4T nuclear-recoil yield component (appletree ComponentSim).

A minimal Monte-Carlo component that samples NR quanta with the P4-NEST model
(``diamx.appletree.plugins.p4_nest``) from a uniform energy spectrum. It is used
to predict the mean light yield, charge yield and recombination fluctuation as a
function of NR energy (cf. PRD 110 023029 Fig. 17) directly from the simulation,
rather than from analytic mean propagation.

Usage (see notebooks/pandax4t_nr_yields.ipynb):

    comp = NRYieldPandaX4T("nr")
    comp.set_config({"lower_energy": 1.0, "upper_energy": 90.0, "xi_norm_nr": 150.0})
    comp.deduce(
        data_names=["energy", "num_photon", "num_electron", "recomb_std"],
        force_no_eff=True,
    )
    comp.compile()
    key, (energy, num_photon, num_electron, recomb_std) = comp.simulate(
        key, batch_size, parameters
    )
"""

import appletree as apt
from appletree.component import ComponentSim

from diamx.appletree.plugins import p4_nest


class NRYieldPandaX4T(ComponentSim):
    """NR quanta MC: uniform energy spectrum -> P4-NEST yields."""

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Energy bootstrap: uniform spectrum over [lower_energy, upper_energy].
        self.register(apt.plugins.common.UniformEnergySpectra)

        # P4-NEST NR quanta chain (PRD Eq. 1/2 + Eq. 17, Appendix A2):
        #   energy -> yield params (L, ion fraction, <r>_0, dr_0)
        #          -> N_q = B(xi/W, L)
        #          -> N_i = B(N_q, 1/(1+alpha))
        #          -> corrected <r>, dr
        #          -> N_e = B(N_i, 1-r), N_ph = N_q - N_e
        self.register(p4_nest.NRYieldParamsP4NEST)
        self.register(p4_nest.NRTotalQuantaP4NEST)
        self.register(p4_nest.NRIonizationP4NEST)
        self.register(p4_nest.NRRecombParamsP4NEST)
        self.register(p4_nest.NRPhotonElectronP4NEST)
