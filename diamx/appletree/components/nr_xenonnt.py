import appletree as apt
from appletree.component import ComponentSim

from diamx.appletree.plugins import nr_nest_v1
from diamx.appletree.plugins import energy_eff, eff_s1_cut, position_workaround


class NRXENONnT(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)

        self.register_all(nr_nest_v1)
        self.register_all(energy_eff)
        # self.register_all(apt.plugins.efficiency)


class WIMPXENONnT(NRXENONnT):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.FixedEnergySpectra)
        self.register_all(eff_s1_cut)


class NeutronXENONnT(WIMPXENONnT):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
