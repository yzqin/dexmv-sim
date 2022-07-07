from .base import REGISTERED_ENVS, MujocoEnv
from .ycb_relocate_env import YCBRelocate
import hand_imitation.env.environments.dapg_env

ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
