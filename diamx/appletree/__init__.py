__version__ = "0.0.0"

import appletree as apt

from . import components
from .components import *
from . import plugins
from .plugins import *
from . import get_file_path
from .get_file_path import *

apt.add_plugin_extensions(apt.plugins, energy_eff)
apt.add_plugin_extensions(apt.plugins, eff_flat_cut)
apt.add_plugin_extensions(apt.plugins, position_workaround)
apt.add_plugin_extensions(apt.plugins, eff_s1_cut)
apt.add_plugin_extensions(apt.plugins, nr_nest_v1)
apt.add_plugin_extensions(apt.plugins, p4_nest)
apt.add_plugin_extensions(apt.plugins, pandax_reconstruction)
