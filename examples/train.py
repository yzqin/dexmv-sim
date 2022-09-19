#!/usr/bin/env python3

"""Train an agent from states."""

import argparse
import pickle
import sys

import gym.logger

from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

from mjrl.algos.behavior_cloning import BC
from mjrl.algos.dapg import DAPG
from mjrl.algos.density_onpg import DensityONPG
from mjrl.algos.soil import SOIL
from mjrl.algos.trpo import TRPO
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils.train_agent import train_agent
from tpi.core.config import assert_cfg, cfg


class Spec:
    def __init__(self, env=None, env_name="relocate-mug-1"):
        self.observation_dim = env.reset().shape[0]
        self.action_dim = env.action_spec[0].shape[0]
        self.env_id = env_name


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train agent from states'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See pycls/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def train():
    gym.logger.set_level(gym.logger.ERROR)
    # Construct the env
    print(f"env name: {cfg.ENV_NAME}")
    env_info = cfg.ENV_NAME.split('-')
    task = env_info[0]
    obj_name = env_info[1]
    obj_scale = float(env_info[2])
    friction = (1, 0.5, 0.01)
    if task == 'relocate':
        e = YCBRelocate(has_renderer=False, object_name=obj_name,
                        friction=friction, object_scale=obj_scale)
        spec = Spec(e, cfg.ENV_NAME)
    elif task == 'pour':
        e = WaterPouringEnv(has_renderer=False, tank_size=(0.15, 0.15, 0.12))
        spec = Spec(e, cfg.ENV_NAME)
    elif task == 'place':
        e = MugPlaceObjectEnv(has_renderer=False, object_scale=obj_scale, mug_scale=1.7)
        spec = Spec(e, cfg.ENV_NAME)

    # Construct the policy
    policy = MLP(
        spec, hidden_sizes=cfg.POLICY_WS, seed=cfg.RNG_SEED,
        init_log_std=cfg.POLICY_INIT_LOG_STD,
        min_log_std=cfg.POLICY_MIN_LOG_STD
    )
    # Load policy from checkpoint
    if cfg.CHECKPOINT_POLICY:
        with open(cfg.CHECKPOINT_POLICY, 'rb') as f:
            policy = pickle.load(f)

    # Load the demos
    demo_paths = None
    if cfg.BC_INIT or cfg.USE_DAPG or cfg.USE_DENSITY_ONPG or cfg.SOIL.ENABLED:
        with open(cfg.DEMO_FILE, 'rb') as f:
            demo_paths = list(pickle.load(f).values())
            num_traj = len(demo_paths)
            num_traj = int(min(cfg.DEMO_RATIO, num_traj))
            demo_paths = demo_paths[:num_traj]
            print(f'num traj: {num_traj}')

    # Initialize w/ behavior cloning
    if cfg.BC_INIT:
        print('==== Training with BC ====')
        bc_agent = BC(demo_paths, policy=policy, epochs=5, batch_size=32, lr=1e-3)
        bc_agent.train()

    # Construct the baseline for PG
    baseline = MLPBaseline(
        spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3, use_gpu=True
    )
    # Load baseline from checkpoint
    if cfg.CHECKPOINT_BASELINE:
        with open(cfg.CHECKPOINT_BASELINE, 'rb') as f:
            baseline = pickle.load(f)

    # Construct the agent
    if cfg.BC_INIT or cfg.USE_DAPG:
        agent = DAPG(
            spec, policy, baseline,
            demo_paths=(demo_paths if cfg.USE_DAPG else None),
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            lam_0=cfg.DAPG_LAM0, lam_1=cfg.DAPG_LAM1,
            save_logs=True
        )
    elif cfg.SOIL.ENABLED:
        agent = SOIL(
            spec, policy, baseline,
            demo_paths=demo_paths,
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            lam_0=cfg.SOIL.LAM0, lam_1=cfg.SOIL.LAM1,
            save_logs=True
        )
    elif cfg.USE_DENSITY_ONPG:
        agent = DensityONPG(
            spec, policy, baseline,
            demo_paths=demo_paths,
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            lam_0=cfg.DENSITY_ONPG_LAM0, lam_1=cfg.DENSITY_ONPG_LAM1,
            save_logs=True)
    elif cfg.TRPO.ENABLED:
        print('using TRPO for RL algo')
        agent = TRPO(
            spec, policy, baseline,
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            save_logs=True
        )
    else:
        print('using TRPO for RL algo')
        agent = TRPO(
            spec, policy, baseline,
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            save_logs=True)

    # Train the agent
    print('==== Training with RL ====')

    if cfg.SOIL.ENABLED or cfg.USE_DENSITY_ONPG or cfg.GASIL.ENABLED or cfg.USE_DAPG:
        env_name = cfg.ENV_NAME
        task = env_name.split('-')[0]
        obj = env_name.split('-')[1]
        demo_property = cfg.DEMO_FILE.split('/')[-1]
        demo_property = demo_property.split('.pkl')[0]
        demo_property = demo_property.replace(f'_{obj}', '').replace(f'{task}_', '')

    print(cfg.JOB_NAME)
    if cfg.JOB_NAME is not None:
        job_label = f'_{cfg.JOB_NAME}'
    else:
        job_label = ''

    if cfg.SOIL.ENABLED:
        job_name = f'soil_{cfg.ENV_NAME}_{demo_property}_{cfg.SOIL.LAM0}_{cfg.DEMO_RATIO}{job_label}_seed{cfg.RNG_SEED}'
    elif cfg.USE_DENSITY_ONPG:
        job_name = f'gail_{cfg.ENV_NAME}_{demo_property}_{cfg.DEMO_RATIO}{job_label}_seed{cfg.RNG_SEED}'
    elif cfg.USE_DAPG:
        job_name = f'dapg_{cfg.ENV_NAME}_{demo_property}_{cfg.DAPG_LAM0}_{cfg.DEMO_RATIO}{job_label}_seed{cfg.RNG_SEED}'
    else:
        job_name = f'rl_{cfg.ENV_NAME}{job_label}_seed{cfg.RNG_SEED}'

    print(job_name)

    train_agent(
        job_name=job_name,
        agent=agent,
        seed=cfg.RNG_SEED,
        niter=cfg.NUM_ITER,
        gamma=0.995,
        gae_lambda=0.97,
        num_cpu=cfg.NUM_CPU,
        sample_mode='trajectories',
        num_traj=cfg.NUM_TRAJ,
        save_freq=cfg.SAVE_FREQ,
        evaluation_rollouts=cfg.EVAL_RS
    )


if __name__ == '__main__':
    # Load the config
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()
    # Train the agent
    train()
