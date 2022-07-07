import numpy as np
from hand_imitation.env.environments.base import MujocoEnv
from gym.core import Env
from gym import spaces
from hand_imitation.env.utils.random import np_random


class DAPGWrapper(Env):
    def __init__(self, env: MujocoEnv):
        self.env = env
        self.name = type(env).__name__

        # Gym specific attributes
        # self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        obs_dim = obs.shape
        high = np.inf * np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        # DAPG specific interface
        self.np_random = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def set_seed(self, seed=None):
        self.seed(seed)

    @property
    def action_dim(self):
        return np.prod(self.action_space.shape)

    @property
    def observation_dim(self):
        return np.prod(self.observation_space.shape)

    @property
    def horizon(self):
        return self.env.horizon

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        return self.env.reset()

    def render(self):
        self.env.render()

    @property
    def act_repeat(self):
        return int(self.env.control_timestep / self.env.model_timestep)

    def get_obs(self):
        return self.env._get_observations()

    def get_env_infos(self):
        return {"env_name": self.name}

    # ===========================================
    # Trajectory optimization related
    # Environments should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.get_env_state()
        except AttributeError as e:
            raise NotImplementedError from e

    def set_env_state(self, state_dict):
        try:
            self.env.set_env_state(state_dict)
        except AttributeError as e:
            raise NotImplementedError from e

    # ===========================================
    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            score = 0.0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                self.render()
                t = t + 1
                score = score + r
            print("Episode score = %f" % score)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=(),
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):
        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        ep_returns = np.zeros(num_episodes)

        for ep in range(num_episodes):
            self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (not done or not terminate_at_done):
                self.render() if visual is True else None
                o = self.get_obs()
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]
