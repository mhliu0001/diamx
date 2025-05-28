import appletree as apt
from appletree.component import ComponentSim
from diamx.appletree.plugins import eff_flat_cut, S1ReconEffNaive

# try:
#     import aptext.cut_acceptance.efficiency_low_e.S1ReconEffNHits as S1ReconEffNHits
# except ImportError:
#     from apt_open.plugins.threefold_cut_eff as S1ReconEffNHits
# from apt_open.plugins import threefold_cut_eff


class ERMonoXENONnT(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)

        self.register(S1ReconEffNaive)
        self.register_all(eff_flat_cut)


class ERFlatXENONnT(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)

        self.register(S1ReconEffNaive)
        self.register_all(eff_flat_cut)


class ERBkgXENONnT(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.FixedEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)

        self.register(S1ReconEffNaive)
        self.register_all(eff_flat_cut)
