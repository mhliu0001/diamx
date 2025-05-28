import appletree as apt
from appletree.component import ComponentSim
from diamx.appletree.plugins import (
    S1ReconEffNaive,
    nr_nest_v1,
    position_workaround,
)


class S1ReconEffXENONnT(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.register(apt.plugins.common.FixedEnergySpectra)
        # self.register(apt.plugins.common.PositionSpectra)
        self.register_all(position_workaround)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)

        self.register_all(nr_nest_v1)
        self.register(S1ReconEffNaive)
