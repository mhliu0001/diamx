from functools import partial

from jax import jit
from jax import numpy as jnp

from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import takes_config, Constant
from appletree.utils import exporter

export, __all__ = exporter()


@export
@takes_config(
    Constant(
        name="z_min",
        type=float,
        default=-133.97,
        help="Z lower limit simulated in uniformly distribution",
    ),
    Constant(
        name="z_max",
        type=float,
        default=-13.35,
        help="Z upper limit simulated in uniformly distribution",
    ),
    Constant(
        name="r_max",
        type=float,
        default=60.0,
        help="Radius upper limit simulated in uniformly distribution",
    ),
)
class PositionSpectraWorkaround(Plugin):
    depends_on = ["energy"]
    provides = ["x", "y", "z"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        key, z = randgen.uniform(
            key, self.z_min.value, self.z_max.value, shape=energy.shape
        )
        key, r2 = randgen.uniform(key, 0, self.r_max.value**2, shape=energy.shape)
        key, theta = randgen.uniform(key, 0, 2 * jnp.pi, shape=energy.shape)

        r = jnp.sqrt(r2)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return key, x, y, z
