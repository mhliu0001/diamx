"""PandaX-4T electronic-recoil yield component (appletree ComponentSim).

Mirror of NRYieldPandaX4T (diamx/appletree/components/nr_pandax4t.py) for ER:
uniform energy spectrum -> ER P4-NEST plugins. The ER mean yields come from the
NESTv2.0 source (ERYieldParamsP4NEST), not PRD Appendix A1 (which has misprints);
the recombination fluctuation and the Eq. 17 correction follow the PRD form.

    comp = ERYieldPandaX4T("er")
    comp.set_config({"lower_energy": 1.0, "upper_energy": 25.0, "xi_norm_er": 30.0})
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


class ERYieldPandaX4T(ComponentSim):
    """ER quanta MC: uniform energy spectrum -> P4-NEST ER yields."""

    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)

        # ER quanta chain: NESTv2.0 ER yields -> Eq. 1/2 sampling + Eq. 17 correction.
        self.register(p4_nest.ERYieldParamsP4NEST)
        self.register(p4_nest.TotalQuantaP4NEST)     # recoil-agnostic
        self.register(p4_nest.IonizationP4NEST)      # recoil-agnostic
        self.register(p4_nest.ERRecombParamsP4NEST)
        self.register(p4_nest.PhotonElectronP4NEST)  # recoil-agnostic
